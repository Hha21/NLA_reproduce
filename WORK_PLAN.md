# Building a Natural Language Autoencoder from scratch (1B model)

A learning-first implementation plan. The goal is to understand *why every piece exists*
by building it yourself, accepting that the result will be rougher than the reference repo
(`kitft/natural_language_autoencoders`). Use the repo as a reference to compare against once
you've built each piece — not as the starting point.

---

## The mental model (read this first)

An NLA forces a model to explain its own internal activations *in natural language*, and
verifies the explanation by checking whether the explanation alone is enough to reconstruct
the activation. Three components:

- **Target model `T`** — frozen. Produces the activation `a ∈ ℝ^d` we want to explain, taken
  from the residual stream at one layer (~2/3 of the way through).
- **Activation Verbalizer `AV`** — a *copy* of `T`, trainable. Reads `a` and emits a free-text
  description `s`. This is the **encoder**, trained with **RL**.
- **Activation Reconstructor `AR`** — a *copy* of `T`, trainable. Reads the text `s` and predicts
  an activation `â`. This is the **decoder**, trained with **supervised regression**.

The training signal is the reconstruction error between `a` and `â`. Natural language is the
bottleneck: the only way `AV` can get a low error is to write a description that genuinely
captures what's in `a`.

**Why two different training methods?** The text `s` is discrete and *sampled* from `AV`, so you
can't backprop reconstruction error through the sampling step into `AV` — hence RL. But the `AR`
sees fixed text in and a fixed target activation out, so it trains by ordinary gradient descent.
Internalising this asymmetry is half the point of the project.

```
            text snippet x
                 │
                 ▼
         ┌───────────────┐
         │  Target  T     │  (frozen)
         └───────┬───────┘
                 │ activation a  (layer L residual stream, ℝ^d)
         ┌───────┴────────────────────────┐
         ▼                                 │
  ┌─────────────┐   text s   ┌─────────────┐
  │  AV (RL)    │──────────▶ │  AR (SGD)   │── â ─▶ reconstruction
  │ encoder     │            │ decoder     │        loss vs a
  └─────────────┘            └─────────────┘             │
         ▲                                               │
         └──────────── reward = −‖a − â‖ (+KL) ──────────┘
```

---

## Model and hardware choices

- **Model used throughout:** Qwen2.5-0.5B (`d_model = 896`, 24 layers). Probe layer = **16** (2/3 depth).
  Originally planned as a fast-iteration model; used for the full pipeline given GPU constraints.
- **Hardware:** a single 24GB GPU (4090 / L40S / A100) is plenty at 1B. Keep `T`, `AV`, `AR` all
  resident; that's ~3×2GB in bf16 plus gradients/optimiser for `AV` and `AR` ≈ 12–15GB. No
  multi-GPU, no quantisation needed.
- **Software:** PyTorch + HuggingFace `transformers`. Deliberately **avoid TRL at first** — you'll
  write the RL loop yourself. Add TRL only in the stretch phase if you want to compare.

A note on weights: use the full-precision HF checkpoint, **not** a GGUF. GGUF is inference-only;
you need `.backward()`. Your earlier llama.cpp activation-extraction idea is still a nice warm-up
exercise for intuition, but the trainable pipeline lives in PyTorch.

---

## Phase 0 — Environment and a frozen target ✅

**Goal:** load the model, generate text, confirm shapes.

- Load Llama 3.2 1B in bf16, `model.eval()`, `requires_grad_(False)` for the target copy.
- Confirm `config.hidden_size == 2048` and `config.num_hidden_layers == 16`.
- Sanity generate a few tokens so you know the tokeniser/chat template works.

**Milestone:** you can print the hidden-state tensor shape `(batch, seq, 2048)` from a forward pass
with `output_hidden_states=True`.

---

## Phase 1 — Activation extraction harness ✅

**Goal:** given a text snippet, return the residual-stream activation at layer `L`.

Build this with a forward hook rather than `output_hidden_states` so you understand exactly what
you're capturing. The thing to be careful about: **which tensor** you grab. The residual stream is
the *input* to a block (pre-LayerNorm) or equivalently the running sum after the previous block's
residual add. Decide this explicitly and document it — it's the single most common source of
"my numbers look weird" later.

```python
# illustrative, untested
acts = {}
def grab(name):
    def hook(module, inp, out):
        # out[0] is the block's residual-stream output (batch, seq, d)
        acts[name] = out[0].detach()
    return hook

L = 11
handle = target.model.layers[L].register_forward_hook(grab("resid"))

def get_activation(text, token_pos=-1):
    ids = tok(text, return_tensors="pt").to(dev)
    with torch.no_grad():
        target(**ids)
    return acts["resid"][0, token_pos]    # (d,)
```

**Design decisions to make and write down:**
- **Token position.** NLAs explain the activation at a *specific token position*. Start with the
  last token of the snippet. Later you can experiment with a fixed position or mean-pooling.
- **Corpus.** Pull a few thousand short snippets (1–3 sentences) from something like a slice of
  C4, OpenWebText, or even Wikipedia. Variety matters more than size for learning.
- **Normalisation.** Compute the mean and std of activation norms across your corpus now — you'll
  need them for FVE and for sane reward scaling.

**Milestone:** a `Dataset` of `(snippet, activation)` pairs cached to disk. Plot a histogram of
activation norms; eyeball a UMAP/PCA of a few hundred activations to confirm there's visible
structure (clusters by topic). This is also a satisfying standalone interpretability artifact.

---

## Phase 2 — The Reconstructor `AR` and a baseline ✅

**Goal:** map text → predicted activation, and establish what "good" even means.

The `AR` is a copy of `T` plus a small **read-out head**: run the text through the transformer,
take a pooled hidden state (last-token or mean), project `ℝ^d → ℝ^d` with a learned linear layer.

```python
# illustrative
class Reconstructor(nn.Module):
    def __init__(self, base, d):
        super().__init__()
        self.base = base                 # copy of T, trainable
        self.head = nn.Linear(d, d)
    def forward(self, input_ids, attn):
        h = self.base(input_ids, attention_mask=attn,
                      output_hidden_states=True).hidden_states[-1]
        pooled = h[:, -1, :]             # last-token pooling
        return self.head(pooled)         # â  (batch, d)
```

**Define the metric before training anything.** Use **Fraction of Variance Explained (FVE)**,
the paper's headline metric:

```python
# FVE over a batch of (a, â)
def fve(a, a_hat):
    num = ((a - a_hat) ** 2).sum(-1)          # per-example residual energy
    den = ((a - a.mean(0)) ** 2).sum(-1)      # variance around the mean activation
    return 1.0 - (num.mean() / den.mean())
```

FVE = 0 means "no better than predicting the mean activation"; FVE = 1 means perfect
reconstruction. The paper reports FVE growing roughly linearly in `log(training steps)`, so you'll
see early signal.

**Critical baselines — build these first, they save you weeks of confusion:**
1. **Mean baseline.** Always predict the corpus-mean activation. FVE = 0 by construction. Sanity
   check your metric.
2. **Text-oracle ceiling.** Train the `AR` to reconstruct from the *original snippet text* (the
   text that produced the activation). This tells you the best FVE achievable when the description
   is "perfect." Your full NLA can never beat this; it tells you how much headroom the language
   bottleneck leaves.

**Milestone:** an `AR` that, trained on `(snippet_text → activation)` pairs, reaches some clearly
positive FVE (the oracle ceiling). You now have a number to chase.

---

## Phase 3 — The Verbalizer `AV` architecture ✅

**Goal:** make a transformer LM *read a vector* and write about it.

The interesting trick: `AV` is an LM that normally consumes token embeddings of shape `(seq, d)`.
You inject `a` as an extra "soft token" at the front of the sequence, then let it generate a
description conditioned on that.

```python
# illustrative
class Verbalizer(nn.Module):
    def __init__(self, base, d):
        super().__init__()
        self.base = base                 # copy of T, trainable
        self.inject = nn.Linear(d, d)    # maps activation into embedding space
    def build_inputs_embeds(self, a, prompt_ids):
        emb = self.base.get_input_embeddings()
        soft = self.inject(a).unsqueeze(1)        # (B, 1, d)
        tok_emb = emb(prompt_ids)                 # (B, P, d)
        return torch.cat([soft, tok_emb], dim=1)  # prepend the soft token
```

Then `AV.generate(inputs_embeds=...)` with a fixed prompt like
`"Describe what this internal representation encodes:"`.

**Things to get right:**
- The injection projection starts near identity-ish because `a` already lives in the model's own
  hidden space (same model family) — but keep it trainable; the embedding space and residual space
  aren't identical.
- Cap generated description length (e.g. 32–64 tokens). Short descriptions force the bottleneck and
  keep RL cheap.
- Keep a **frozen reference copy** of `AV`'s initial weights — you'll need it for the KL penalty.

**Milestone:** `AV` produces *fluent* (not yet *accurate*) text from an injected activation. At this
stage it'll mostly ignore the activation — that's expected and is exactly what the warm-start fixes.

---

## Phase 4 — Warm-start / supervised bootstrap ✅

**Why:** at initialisation `AV` writes activation-agnostic text, so the RL reward is pure noise and
training won't move. The warm-start teaches `AV` to produce *relevant* text before RL refines it.

**Self-contained bootstrap (no external teacher needed):** the activation `a` came from processing
snippet `x`. Teach `AV` to recover a short summary of `x` from `a`. The cleanest fully-local version:
SFT `AV` on `(a → x)` pairs (or `a → first-sentence-of-x`). This is essentially the paper's
"summarisation proxy," and it works because a faithful description of "what this activation encodes"
overlaps heavily with "what the text was about."

- If you have Anthropic API access, an even better target is an actual short summary of `x`
  generated by a stronger model — but the local version is enough to learn from and keeps the
  project self-contained.
- Train with ordinary cross-entropy (teacher forcing) on the description tokens.

**Milestone:** descriptions are now visibly *about* the snippet. Run them through your `AR`
(also warm-started on `text → a`) and you should already see **FVE clearly above 0** — the paper
notes the warm-start alone gets a meaningful chunk of the way (~0.3–0.4 FVE). Hitting positive FVE
here is the project's first real "it works" moment.

---

## Phase 5 — The RL phase: GRPO from scratch ✅

**Why GRPO over PPO:** GRPO drops the value/critic network. You sample a *group* of `G` completions
per prompt, use the group's mean reward as the baseline, and normalise advantages within the group.
For a from-scratch build that's a huge simplification — there's no second network to train.

**The loop, per training step:**

1. Sample a batch of activations `{a_i}` from the corpus.
2. For each `a_i`, sample `G` descriptions `{s_i^1 … s_i^G}` from `AV` (keep the per-token
   log-probs).
3. Reconstruct each: `â_i^g = AR(s_i^g)`.
4. **Reward** `r_i^g = cosine(a_i, â_i^g)` (cosine is better-behaved than raw MSE; bounded, scale-free).
5. **Group-normalised advantage:** `A_i^g = (r_i^g − mean_g r_i) / (std_g r_i + ε)`.
6. **Policy-gradient loss:** `−Σ A_i^g · logπ_AV(s_i^g)`.
7. **KL penalty** against the frozen reference `AV` to keep text fluent and stop reward-hacking:
   add `β · KL(π_AV ‖ π_ref)`. Start `β ≈ 0.05` and tune.
8. Step `AV`.
9. **Separately**, take the sampled `(s_i^g → a_i)` pairs and do a supervised regression step on
   `AR` (MSE/cosine). `AR` keeps improving as a decoder as the description distribution shifts.

```python
# illustrative skeleton of one GRPO step
def grpo_step(batch_acts, G=8, beta=0.05):
    advantages, logps, recon_pairs = [], [], []
    for a in batch_acts:
        descs, lp = av.sample(a, n=G)              # G descriptions + logprobs
        a_hat = ar(descs)                          # (G, d)
        r = F.cosine_similarity(a.expand_as(a_hat), a_hat)   # (G,)
        adv = (r - r.mean()) / (r.std() + 1e-6)
        advantages.append(adv); logps.append(lp)
        recon_pairs.append((descs, a))
    pg = -(torch.stack(advantages) * torch.stack(logps)).mean()
    kl = beta * kl_to_reference(av, av_ref, descs)
    (pg + kl).backward(); av_opt.step(); av_opt.zero_grad()
    update_reconstructor(ar, recon_pairs)          # supervised MSE on AR
```

**Gotchas that will bite you (in rough order of likelihood):**
- **Reward hacking.** Without the KL term, `AV` discovers degenerate strings that the `AR` happens
  to map well — gibberish that scores high. The KL penalty and a capped vocabulary/length are your
  defence. Watch your sampled descriptions, not just the FVE curve.
- **Stale `AR`.** If `AR` stops improving, `AV` optimises against a fixed (and beatable) decoder and
  the descriptions degrade. Keep updating `AR` every step.
- **Group size too small.** `G < 4` makes the advantage estimate too noisy. `G = 8` is a good start.
- **Entropy collapse.** `AV` converges to one description for everything. Monitor description
  diversity; a small entropy bonus or higher KL helps.
- **Reward scale.** Cosine in `[-1,1]` keeps things stable; if you use MSE, normalise by the
  activation-norm stats from Phase 1.

**Milestone:** FVE climbs above the warm-start level and roughly tracks `log(steps)`. Spot-check
descriptions: for an activation taken from a snippet about, say, cooking, the description should
mention food/cooking concepts — and crucially, the `AR` should reconstruct *better* from that
description than from a random one.

---

## Phase 6 — Evaluation and interpretability payoff 🔄

This is where it stops being a training exercise and becomes interpretability.

**Achieved:**
- FVE tracked throughout training; progression AR 0.47 → AV warmstart 0.44 → GRPO **0.594**
- `eval_fve_compare.py` for stable held-out comparison between checkpoint pairs

**Remaining:**
- **FVE curve** vs `log(steps)` plot, with mean baseline (0) and oracle ceiling
- **Qualitative spot-checks** — planned as an interactive frontend (Phase 7)
- **Intervention sanity check** — perturb activation, confirm description changes sensibly

---

## Phase 7 — Stretch goals (open-ended)

- **Interactive frontend** — a web UI to inspect the trained NLA: paste a text snippet, extract its
  activation via the target model, generate a description with the AV, reconstruct with the AR, and
  display the FVE alongside the description. This turns the pipeline into a usable interpretability
  tool rather than just a training artifact.
- **Better Stage 1 descriptions** — regenerate with Claude Haiku (4–5 features, 150–200 words) to
  raise the description quality ceiling. This is the highest-leverage improvement available without
  changing the model or architecture.
- **Swap in TRL's GRPO** and compare against the hand-rolled loop — great for confirming
  understanding and seeing what the library handles (gradient accumulation, KL controllers, etc.).
- **Layer sensitivity study** — probe different layers (8, 12, 16, 20) and compare FVE and
  description content, mirroring the paper's layer-sensitivity discussion.
- **Compare to logit lens / a tiny SAE** on the same activations — three lenses on the same vector.
- **Scale to Qwen2.5-1.5B or 3B** once the 0.5B pipeline is solid.

---

## Suggested timeline

| Week | Focus | Deliverable | Status |
|------|-------|-------------|--------|
| 1 | Phases 0–2 | Activation harness + `AR` baseline + FVE metric | ✅ AR FVE 0.47 |
| 2 | Phases 3–4 | `AV` with injection + warm-start reaching positive FVE | ✅ AV e2e FVE 0.44 |
| 3 | Phase 5 | Working GRPO loop, FVE rising above warm-start | ✅ GRPO FVE 0.594 |
| 4 | Phases 6–7 | Evaluation, qualitative viewer, stretch goals | 🔄 Frontend next |

## The three things most likely to go wrong (bookmark these)

1. **Wrong activation tensor / token position** (Phase 1) — silently poisons everything downstream.
   Verify against the oracle ceiling early.
2. **RL won't move without a good warm-start** (Phase 4 → 5) — if FVE is flat in RL, your warm-start
   probably isn't strong enough; fix Phase 4 before blaming Phase 5.
3. **Reward hacking / stale AR** (Phase 5) — read your descriptions constantly; FVE going up while
   text turns to gibberish means the metric is being gamed.
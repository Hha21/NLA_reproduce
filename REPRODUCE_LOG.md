# NLA Reproduction Log

Reproducing: [Natural Language Autoencoders](https://transformer-circuits.pub/2026/nla/) (Anthropic, 2026)  
Target model: Qwen2.5-0.5B (paper uses Qwen2.5-7B)  
Reference repo: https://github.com/kitft/natural_language_autoencoders

---

## Stage 0 — Activation Extraction

**Status:** ✅ Complete (100K samples)

**What we do:** Forward-pass Qwen2.5-0.5B on FineWeb sample-10BT, hook the residual stream at layer 16 (`PROBE_LAYER`), sample up to 10 positions per document. Output: `activations/dataset` with columns `text_truncated`, `activation`.

**Issues:**
- Initially used wikitext-103 as corpus. Paper uses FineWeb sample-10BT. Switched to FineWeb.
- `--n-samples 100000` produces 100K rows (rows = extraction points, not documents). Paper reports ~1M vectors. Staying at 100K to validate pipeline first.

---

## Stage 1 — Explanation Generation

**Status:** ✅ Complete (83,071 summaries generated)

**What we do:** For each `text_truncated` ≥ 400 chars, call DeepSeek V4-Flash to generate a structured linguistic-feature explanation wrapped in `<analysis>...</analysis>` tags. Stored in the `summary` column.

**Why explanations not summaries:** The AR learns to reconstruct h_l from text. Simple topic summaries carry no information about what h_l encodes (FVE ≈ 0). The structured `<analysis>` features ("unclosed list requires third item", "formal academic tone") directly target the prediction context at the truncation point.

### Issue #1 — Wrong explanation format (first pass)
Initial implementation generated simple 1–3 sentence summaries. FVE ≈ 0 after full AR training — model had nothing to learn from. Fixed: regenerated from scratch with structured linguistic analysis prompt.

### Issue #2 — Prompt version: switched from paper (4–5 features) to reference (2–3 features)
Paper appendix specifies 4–5 features / 150–200 words. Reference codebase uses 2–3 features / ~80–100 words. Initially used the paper version; switched to the reference version (faster, matches published experiments). Final choice: **2–3 feature prompt from `stage2_api_explain.py`**.

### Issue #3 — `max_tokens` truncating API responses
The newer OpenAI SDK deprecated `max_tokens` in favour of `max_completion_tokens`. DeepSeek silently ignored the old parameter, causing `finish_reason='length'` even at `max_tokens=600`. Fixed: removed `max_tokens` entirely, letting DeepSeek use its default (~4096 tokens). Response now `finish_reason='stop'`.

---

## Stage 2 — AR Warm-Start Training

**Status:** ✅ Complete (val FVE ≈ 0.47)

**Architecture:**
- Base: Qwen2.5-0.5B transformer truncated to layers 0..16, final norm → `nn.Identity()` so `last_hidden_state` = raw h_l
- Head: `nn.Linear(896, 896, bias=False)`, identity-initialised
- Input: `AR_PREFIX + summary + AR_SUFFIX`

**Final training config:** full model SFT, head lr=1e-4, base lr=1e-5, cosine → 0, batch=32, 15 epochs, first 50K rows, mse_scale=True

### Issue #4 — Wrong AR architecture (early sessions)
Initially used the full 24-layer model and tried to map h_24 → h_16 with a linear head. Impossible for a single linear layer; persistent negative FVE. Fix: truncate base to layers[:17], strip final norm.

### Issue #5 — AR head had `bias=True`
Reference repo (`models.py`) uses `bias=False`. Fixed.

### Issue #6 — Training on original text instead of explanations
First warm-start attempts used `text_truncated`. The AR prompt during GRPO will receive AV-generated descriptions, not raw text. Fix: `--text-col summary`.

### Issue #7 — FVE ≈ 0 despite architectural fixes
Root cause: explanations were still plain summaries (Issue #1). Fixed by regenerating explanations.

### Issue #8 — LR too conservative
`lr=2e-5` gave FVE stuck at ~0 for 10 epochs. Fix: raised to `lr=1e-4` (head), `1e-5` (base). FVE reached 0.47.

### Issue #9 — Missing gradient clipping
No `clip_grad_norm_` with 390M unfrozen params caused slow / unstable training. Added `clip_grad_norm_(ar.parameters(), 1.0)`.

### Issue #10 — No data split between AR and AV warm-starts
Initially trained AR on all 100K rows. Reference uses a strict 50/50 disjoint split. Fixed: AR now uses rows 0..49999, AV uses rows 50000..99999.

### Issue #11 — Missing mse_scale normalisation on AR targets
Reference uses `mse_scale=sqrt_d_model` — normalise each target h_l to L2 norm = √896 ≈ 29.93 before computing MSE. This stabilises the regression target scale and aligns with AV's injection_scale. Added `--mse-scale` flag (default: True).

---

## Stage 3 — AV Warm-Start Training

**Status:** ✅ Complete (best e2e FVE ≈ 0.44, saved to `checkpoints/av_warmstart.pt`)

**Final training config:** lr=2e-5, 3 epochs, best-val-loss checkpoint saved, rows 50000..99999

### Issue #12 — Wrong AV SFT target (text_truncated)
First AV warm-start run used `text_truncated` as the generation target. Raw texts are long (~200–500 tokens) and highly varied; model memorised training examples rather than learning to condition on the activation. val_loss climbed while train_loss fell (severe overfitting). Fix: use `summary` (LLM explanation) as target — shorter, stylistically uniform, matches what AR expects.

### Issue #13 — Wrong AV prompt format
First implementation used a simplified placeholder `"Explain: <concept>㊗</concept>\n<explanation>"`. Reference uses the full investigator system prompt. Fixed to match `stage3_build.py _DEFAULT_ACTOR_TEMPLATE`.

### Issue #14 — `apply_chat_template` with `return_tensors="pt"` returns tokenizers.Encoding
`tok.apply_chat_template(..., tokenize=True, return_tensors="pt")` returned a `tokenizers.Encoding` object instead of a plain tensor with this tokenizer version. Fix: use `tokenize=False` to get the string, then call `tok(prompt_str, ...)["input_ids"]` separately.

### Issue #15 — AV overfitting: too many epochs, dataset too small
Reference (`actor_sft.sh`) trains for `NUM_EPOCH=1` on 250k AV samples. We trained for 10 epochs on 50k — 50x more gradient steps per parameter. val_loss peaked at epoch 1 (1.55) then diverged to 2.28 by epoch 9; e2e FVE peaked at epoch 1 (0.44) but no best-checkpoint was saved.

Fix:
- Added `checkpoint_path` parameter to `train_av()` — saves best val_loss model during training
- Reduced epochs 10→3 in `run_av_warmstart.sh` (3 × 50k ≈ reference 1 × 250k in total gradient steps)
- LR reduced 5e-5→2e-5 (matching reference)

---

## Stage 4 — Joint GRPO Training

**Status:** ✅ Complete

**Best checkpoint:** `checkpoints/grpo_av_step1000.pt` + `checkpoints/grpo_ar_step1000.pt`  
**Best e2e FVE:** **0.594** (500-sample fixed eval, `activations/dataset`, seed=0)  
**Reference (7B model):** FVE 0.752 at 4199 steps

**Run 1 (original, 5000 steps):**
- Data: `activations/dataset` (100K activations)
- N=16 prompts × K=8 samples, lr=1.41e-5, KL=0.01, constant LR, rollout_batch=4
- FVE at step 1000: 0.5746 (training log) — peak; essentially flat to step 5000 (0.5725)
- Best checkpoint: step 1000

**Run 2 (continuation from step 1000, 1001 steps):**
- Started from grpo_av/ar_step1000.pt, reference AV = step 1000 checkpoint (KL from current position)
- Data: `activations/rl_dataset` (partial 1M dataset)
- KL divergence grew to 6.6 by step 1000 (vs 2.35 in original run at step 5000) — instability
- FVE degraded: 0.5551 → 0.4965; reward degraded -0.29 → -0.37
- Root cause: AV drifted aggressively from step-1000 reference with no stabilising force

**FVE analysis:**
- GRPO improved FVE from warmstart baseline (~0.44) to **0.594** — a meaningful 35% relative gain
- Plateau after step 1000 suggests the 0.5B model is near its capacity ceiling for this task
- Gap vs 7B paper result (0.75) is primarily model capacity (14× fewer params, smaller residual stream)
- Description quality (2–3 features vs paper's 4–5) is a secondary limiting factor

### Issue #16 — No FVE reported at step 1
FVE eval was only triggered at save_interval boundaries. With save_interval=1000, step 1 showed no FVE, making it impossible to see the pre-GRPO baseline on the same val set. Fix: added `or step == 1` to the checkpoint/eval condition in `train_grpo()`.

### Issue #17 — Script appeared frozen during rollout
No progress output during the first rollout (N=16 × K=8 × 150 tokens generation, ~1–3 min). Fix: added `tqdm` progress bar to `_grpo_rollout()` showing `"step X/N rollout"` with per-batch-of-4 granularity.

### Issue #18 — OOM during RL dataset generation
`generate_data.py` crashed at 13% (134K/1M samples) with `torch.OutOfMemoryError: Tried to allocate 26.16 GiB`. Root cause: a ~92,000-token FineWeb document — no upper bound on extraction position caused the `lm_head` allocation to exceed GPU capacity (26 GB for 92K × vocab_size × 2 bytes). Fix: added `--max-seq-len 4096` cap in `run_generate_rl_data.sh`; 4096 tokens → ~1.25 GB lm_head allocation, safe with 16 GB headroom. Also added `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation.

### Issue #19 — RL data generation lost all progress on crash
No intermediate saves meant 23 minutes of GPU work (134K samples) was lost at crash. Fix: added shard-based checkpointing to `generate_data.py` — saves a HuggingFace dataset shard every 50K samples (`--shard-size 50000`). On restart, completed shards are detected automatically and skipped (rng state replayed without GPU), resuming from the last complete shard. Final dataset is assembled by concatenating all shards.

### Issue #20 — Misleading FVE comparison between GRPO runs
The original GRPO run drew its val set from `activations/dataset` (last 200 of 100K); the continuation run used `activations/rl_dataset` (last 200 of 1M) — different activations, so the FVEs were not comparable. The same weights (step-1000 checkpoint) produced 0.5746 on one val set and 0.5551 on the other, making the runs appear to diverge before GRPO had even started. Fix: added `scripts/eval_fve_compare.py` — evaluates two checkpoint pairs on a single fixed 500-sample set (configurable seed) drawn from a specified dataset, giving an apples-to-apples comparison.

### Issue #21 — Continuation run instability (KL explosion)
KL divergence grew from 0 to 6.6 over 1000 continuation steps (vs 2.35 over 5000 original steps). The AV drifted aggressively from the step-1000 reference, and reward/FVE degraded. Root cause: the step-1000 AV was already near a local optimum; further policy gradient pressure pushed it out of the productive region with no strong restoring force. Higher KL coefficient would constrain drift but would also prevent improvement. Conclusion: 0.594 is close to the 0.5B capacity ceiling; further gains require better description quality (Stage 1) or a larger model.

---

## Summary of Results

| Checkpoint | e2e FVE (500 samples) | Notes |
|---|---|---|
| AR baseline (SFT only) | 0.47 | AR warm-start ceiling |
| AV warmstart | ~0.44 | AV generates descriptions the AR can partially decode |
| grpo_av_step1000 | **0.594** | Best result; GRPO peak before plateau |
| grpo_cont_av_step1000 | 0.525 | Continuation run, degraded due to instability |
| Paper (7B model) | 0.752 | 4199 steps, reference run |

---

## Pending / Next Steps

- [ ] Regenerate Stage 1 explanations with 4–5 feature prompt (Claude Haiku or DeepSeek-R1) — most impactful lever for improving FVE ceiling
- [ ] Complete 1M RL dataset extraction (`run_generate_rl_data.sh`, now crash-resumable)
- [ ] Frontend for qualitative inspection of AV outputs (snippet → activation → description)
- [ ] Scale to larger model (e.g. Qwen2.5-1.5B or 3B) if GPU budget allows

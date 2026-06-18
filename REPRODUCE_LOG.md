# NLA Reproduction Log

Reproducing: [Natural Language Autoencoders](https://transformer-circuits.pub/2026/nla/) (Anthropic, 2026)  
Target model: Qwen2.5-0.5B (paper uses Qwen2.5-7B)  
Reference repo: https://github.com/kitft/natural_language_autoencoders

---

## Stage 0 — Activation Extraction

**Status:** Complete (100K samples)

**What we do:** Forward-pass Qwen2.5-0.5B on FineWeb sample-10BT, hook the residual stream at layer 16 (`PROBE_LAYER`), sample up to 10 positions per document. Output: `activations/dataset` with columns `text_truncated`, `activation`.

**Issues:**
- Initially used wikitext-103 as corpus. Paper uses FineWeb sample-10BT. Switched to FineWeb.
- `--n-samples 100000` produces 100K rows (rows = extraction points, not documents). Paper reports ~1M vectors. We may need to regenerate at 1M scale after validating the pipeline end-to-end.

---

## Stage 1 — Explanation Generation

**Status:** In progress (needs re-run with correct prompt — see Issue #2 below)

**What we do:** For each `text_truncated`, call DeepSeek V4-Flash to generate a structured linguistic-feature explanation wrapped in `<analysis>...</analysis>` tags. This is stored in the `summary` column of the dataset.

**Why explanations not summaries:** The AR learns to reconstruct x_l from a text description. Simple summaries ("This text is about X") don't capture what x_l actually encodes — the model's prediction context at the truncation point. The structured `<analysis>` features ("unclosed list requires third item", "formal academic tone") directly target that information.

**Issues:**

### Issue #1 — Wrong explanation format (first pass)
Initial implementation generated simple 1-3 sentence summaries. These have essentially no correlation with x_l (FVE ≈ 0 after full training). The paper generates structured linguistic features, not summaries. Explanation column was regenerated from scratch.

### Issue #2 — Prompt discrepancy: paper vs. codebase
The reference GitHub repo (`nla/datagen/stage2_api_explain.py`) uses a **2-3 feature** prompt (~80-100 words total, 6 feature types).  
The paper appendix (`#warmstart-data-generation`) specifies a **4-5 feature** prompt (~150-200 words total, 10 feature types).  
We use the paper version as it is the one whose results (FVE 0.3-0.4) are reported. Whether the repo version was simplified post-publication or is a different experiment is unclear.

**API provider:** Paper uses Claude Opus 4.5. We use DeepSeek V4-Flash (~$4 per 100K vs. ~$75 for Claude Opus). Quality difference unknown; DeepSeek should follow the structured prompt well.

**"Short paragraphs with bolded topic headings":** The paper's comment about this style refers to Claude's *raw output* before cleaning, not what gets stored. The reference `stage2_api_explain.py` strips bold markers, bullets, and numbering before writing to disk. Our implementation replicates this cleaning step. The paper notes this bold style "persists through NLA training" because the AV training data presumably retains Claude's raw formatting, causing the AV to learn it.

**`<analysis>` is plain text, not a special token.** Both DeepSeek and Qwen tokenize `<analysis>` as ordinary sub-word tokens. Model-agnostic in principle; quality of features may differ from Claude Opus 4.5 in practice.

---

## Stage 2 — AR Warm-Start Training

**Status:** Attempted, failed (FVE ≈ 0.0007). Will re-run after fixing Stage 1 explanations.

**Architecture:**
- Base: Qwen2.5-0.5B transformer truncated to layers 0..16, final norm replaced with `nn.Identity()` so `last_hidden_state` = raw x_l
- Head: `nn.Linear(896, 896, bias=False)`, identity-initialised (`weight=I`)
- Input prompt: `"Summary of the following text: <text>{explanation}</text> <summary>"`
- Target: x_l extracted at the same position

**Training config:** full model SFT (base unfrozen), lr=2e-5 cosine → 0, base_lr=2e-6, batch=32, 30 epochs

**Issues:**

### Issue #3 — Wrong AR architecture (early sessions)
Initially used the full 24-layer model and tried to map x_24 → x_16 with a linear head. This is impossible for a single linear layer and gave persistent negative FVE. Fix: truncate base to layers[:17], strip final norm so the model's own residual stream at layer 16 is the output.

### Issue #4 — AR head had bias=True
Our implementation used `bias=True` on the linear head. The reference repo (`models.py`) uses `bias=False`. Fixed.

### Issue #5 — Training on original text instead of explanations
First warm-start attempts trained AR on `text_truncated` (the original text). The warm-start should use LLM-generated explanations so that AR and AV share a common "language". Fix: `--text-col summary`.

### Issue #6 — FVE ≈ 0 despite architectural fixes
After fixing issues #3-5, FVE remained at ~0.0007 across 30 epochs. Root cause: the explanations at that point were still plain summaries (Issue #1), not the structured `<analysis>` features. The AR had nothing useful to learn from.

---

## Pending

- [ ] Re-run explanation generation with corrected 4-5 feature prompt
- [ ] Re-run AR warm-start training (target FVE 0.3-0.4)
- [ ] Implement AV architecture and warm-start
- [ ] Implement GRPO training loop
- [ ] Consider scaling dataset to 1M vectors if 100K proves insufficient

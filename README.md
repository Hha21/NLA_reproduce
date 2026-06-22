# NLA Reproduce

A from-scratch reproduction of the [Natural Language Autoencoder](https://transformer-circuits.pub/2026/nla/) (Anthropic, 2026) pipeline.
Built on **Qwen2.5-0.5B** (paper uses 7B) for GPU budget reasons. Reference code: [kitft/natural_language_autoencoders](https://github.com/kitft/natural_language_autoencoders).

See [METHOD_PIPELINE.md](METHOD_PIPELINE.md) for a detailed walkthrough of every stage with implementation notes, design decisions, and divergences from the paper.

---

## Progress

| Stage | Description | Status | Result |
|---|---|---|---|
| 0 | Activation extraction (FineWeb → dataset) | ✅ Complete | 100K (text, h_l) pairs |
| 1 | LLM explanation generation | ✅ Complete | 83K summaries, DeepSeek V4-Flash |
| 2 | AR warm-start training | ✅ Complete | val FVE ≈ 0.47 (50K rows, mse_scale) |
| 3 | AV warm-start training | 🔄 In progress | train_loss ~1.54, e2e FVE tracking added |
| 4 | Joint GRPO training | ⏳ Pending | — |

---

## Quick start

```bash
# 1. Create environment
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# PyTorch with CUDA (replace cu121 with your version: nvcc --version)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 2. Verify GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 3. Run stages in order
python scripts/phase00_load_model.py          # sanity check

./scripts/run_generate_data.sh                # Stage 0: extract activations

export DEEPSEEK_API_KEY=sk-...
./scripts/run_generate_summaries.sh           # Stage 1: generate explanations

./scripts/run_ar_pretraining.sh               # Stage 2: AR warm-start

./scripts/run_av_warmstart.sh                 # Stage 3: AV warm-start
```

Model weights (~1 GB for Qwen2.5-0.5B) are downloaded automatically from HuggingFace on first run.

---

## Directory structure

```
NLA_reproduce/
├── README.md
├── METHOD_PIPELINE.md            # detailed stage-by-stage pipeline documentation
├── REPRODUCE_LOG.md              # issues found and fixed during reproduction
├── requirements.txt
├── src/                          # shared library
│   ├── config.py                 #   MODEL_ID, PROBE_LAYER, DEVICE, DTYPE, AR prompt
│   ├── model.py                  #   load_target(), load_tokenizer()
│   ├── data.py                   #   activation extraction, ActivationDataset
│   ├── ar.py                     #   Reconstructor: text → â ∈ ℝ⁸⁹⁶
│   ├── av.py                     #   Verbalizer: h_l → text (㊗ injection, full 24-layer)
│   └── train.py                  #   train_ar(), train_av(), eval_e2e_fve(), fve()
├── scripts/                      # entry points — run via shell scripts
│   ├── phase00_load_model.py     #   verify GPU and model load
│   ├── generate_data.py          #   Stage 0: build (text, activation) dataset
│   ├── generate_summaries.py     #   Stage 1: LLM explanation generation
│   ├── train_ar_baseline.py      #   Stage 2: AR warm-start
│   ├── train_warmstart.py        #   Stage 3: AV warm-start
│   ├── run_av_warmstart.sh       #   Stage 3 runner
│   └── train_grpo.py             #   Stage 4: joint GRPO training (pending)
├── activations/                  # dataset and checkpoints — gitignored
└── checkpoints/                  # saved model weights — gitignored
```

---

## Key design choices

**Why Qwen2.5-0.5B?** The paper uses 7B, which requires ~84GB for full-model SFT — beyond the 48GB available (2× RTX 4090). The 0.5B model fits comfortably and the pipeline is otherwise identical.

**Why DeepSeek V4-Flash for explanations?** The paper uses Claude Opus 4.5; the reference codebase uses Claude Haiku. DeepSeek V4-Flash is cost-equivalent to Haiku (~$4/100K explanations vs ~$75 for Opus) and follows structured prompts reliably.

**Why truncate AR to layer 16?** The AR needs to output the raw residual stream at the probe layer. Truncating the base transformer to layers 0..16 and replacing the final norm with `nn.Identity()` makes `last_hidden_state` the exact quantity the forward hook captured — no further mapping needed.

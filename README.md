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
| 2 | AR warm-start training | ✅ Complete | val FVE ≈ 0.47 |
| 3 | AV warm-start training | ✅ Complete | e2e FVE ≈ 0.44 (best epoch) |
| 4 | Joint GRPO training | ✅ Complete | e2e FVE **0.594** (step 1000, 500-sample eval) |
| 5 | Interactive web demo | 🔄 In progress | tokenise + per-token AV explanation working |

**Reference (paper, 7B model):** FVE ≈ 0.75 at ~4200 steps.

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

./scripts/run_generate_data.sh                # Stage 0: extract 100K activations

export DEEPSEEK_API_KEY=sk-...
./scripts/run_generate_summaries.sh           # Stage 1: generate explanations

./scripts/run_ar_pretraining.sh               # Stage 2: AR warm-start

./scripts/run_av_warmstart.sh                 # Stage 3: AV warm-start

./scripts/run_grpo.sh                         # Stage 4: GRPO joint training
```

Model weights (~1 GB for Qwen2.5-0.5B) are downloaded automatically from HuggingFace on first run.

---

## Demo (web UI)

A FastAPI server + static HTML/JS frontend laid out as three columns:

1. **Chat** (left) — type a message, the target model generates a response with the chosen
   sampling settings (max tokens, temperature, top-p). "New chat" resets the conversation.
2. **Model context** (middle) — every token the target model actually sees, with the chat
   template applied. Special tokens (`<|im_start|>`, …) are highlighted. Clicking any token
   selects it and faintly shades all tokens that came before it (the activation at token *i*
   depends on tokens 0…*i*).
3. **AV explanation** (right) — the AV's natural-language explanation of the selected token's
   layer-16 activation, plus two reconstruction metrics:
   - **Reconstruction (cosine)** — angle between AR's prediction and the (sqrt-d-normalised)
     activation. Bounded [−1, 1], higher is better.
   - **Per-sample FVE** — 1 − ‖a − â‖² / ‖a − ā‖², where ā is the corpus-mean activation
     (computed at startup from `activations/dataset`). Same definition as the corpus FVE in
     [REPRODUCE_LOG.md](REPRODUCE_LOG.md), evaluated on a single sample.

```bash
# Once: copy your best GRPO checkpoints into models/
cp checkpoints/grpo_av_step1000.pt models/av.pt
cp checkpoints/grpo_ar_step1000.pt models/ar.pt

# Run the server (binds 0.0.0.0:8000 so it's reachable over SSH/LAN)
./.venv/bin/uvicorn server.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/` in a browser. The first request takes a few seconds while the
models load; `GET /api/health` confirms `"status": "ok"`.

**Running on a remote machine over SSH?** Forward the port from your laptop:

```bash
ssh -L 8000:localhost:8000 your-user@<linux-box-host>
```

Then `http://localhost:8000` in your local browser reaches the remote server. See
[server/README.md](server/README.md) for the JSON API.

> **Note.** The target model is **base** Qwen2.5-0.5B (not Instruct), because the AV/AR were
> trained on base-model activations. Chat-template generation will work but produce incoherent
> responses — the demo is for inspecting *activations*, not for chatting.

---

## Directory structure

```
NLA_reproduce/
├── README.md
├── METHOD_PIPELINE.md            # detailed stage-by-stage pipeline documentation
├── REPRODUCE_LOG.md              # issues found and fixed during reproduction
├── requirements.txt
├── src/                          # shared library
│   ├── config.py                 #   MODEL_ID, PROBE_LAYER, DEVICE, DTYPE, AR/AV prompts
│   ├── model.py                  #   load_target(), load_tokenizer()
│   ├── data.py                   #   activation extraction, ActivationDataset
│   ├── ar.py                     #   Reconstructor: text → â ∈ ℝ⁸⁹⁶
│   ├── av.py                     #   Verbalizer: h_l → text (㊗ injection, full 24-layer)
│   └── train.py                  #   train_ar(), train_av(), train_grpo(), eval_e2e_fve(), fve()
├── scripts/                      # entry points — run via shell scripts
│   ├── phase00_load_model.py     #   verify GPU and model load
│   ├── generate_data.py          #   Stage 0: build (text, activation) dataset (shard-safe)
│   ├── generate_summaries.py     #   Stage 1: LLM explanation generation
│   ├── train_ar_baseline.py      #   Stage 2: AR warm-start
│   ├── train_warmstart.py        #   Stage 3: AV warm-start
│   ├── train_grpo.py             #   Stage 4: joint GRPO training
│   ├── eval_fve_compare.py       #   compare e2e FVE between two checkpoint pairs
│   ├── run_generate_data.sh      #   Stage 0 runner (100K)
│   ├── run_generate_rl_data.sh   #   Stage 0 runner (1M, for RL; shard-safe, crash-resumable)
│   ├── run_generate_summaries.sh #   Stage 1 runner
│   ├── run_ar_pretraining.sh     #   Stage 2 runner
│   ├── run_av_warmstart.sh       #   Stage 3 runner
│   └── run_grpo.sh               #   Stage 4 runner
├── activations/                  # datasets — gitignored
│   ├── dataset/                  #   100K (text, activation) pairs + summaries
│   └── rl_dataset/               #   1M activation-only pairs for GRPO (in progress)
├── checkpoints/                  # raw training outputs — gitignored
│   ├── ar_baseline.pt            #   AR after Stage 2 warm-start (FVE 0.47)
│   ├── av_warmstart.pt           #   AV after Stage 3 warm-start (best val_loss epoch)
│   ├── grpo_av_step1000.pt       #   AV after 1000 GRPO steps — best result (FVE 0.594)
│   └── grpo_ar_step1000.pt       #   AR after 1000 GRPO steps — best result
├── models/                       # checkpoints served by the demo — gitignored
│   ├── av.pt                     #   copy of grpo_av_step1000.pt
│   └── ar.pt                     #   copy of grpo_ar_step1000.pt
├── server/                       # FastAPI inference server
│   ├── main.py                   #   routes, lifespan, static mount
│   ├── inference.py              #   NLAInference: tokenize, analyze
│   └── README.md
└── frontend/                     # static client — no build tools
    ├── index.html
    ├── styles.css
    └── app.js
```

---

## Key design choices

**Why Qwen2.5-0.5B?** The paper uses 7B, which requires ~84GB for full-model SFT — beyond the 48GB available (2× RTX 4090). The 0.5B model fits comfortably and the pipeline is otherwise identical.

**Why DeepSeek V4-Flash for explanations?** The paper uses Claude Opus 4.5; the reference codebase uses Claude Haiku. DeepSeek V4-Flash is cost-equivalent to Haiku (~$4/100K explanations vs ~$75 for Opus) and follows structured prompts reliably.

**Why truncate AR to layer 16?** The AR needs to output the raw residual stream at the probe layer. Truncating the base transformer to layers 0..16 and replacing the final norm with `nn.Identity()` makes `last_hidden_state` the exact quantity the forward hook captured — no further mapping needed.

**FVE gap vs paper (0.594 vs 0.75):** The gap is primarily model capacity — a 0.5B residual stream at layer 16 encodes less recoverable information than a 7B stream at layer 20, and the AV/AR have less generative capacity. Description quality (2–3 features vs paper's 4–5) is a secondary factor. See REPRODUCE_LOG.md for the full analysis.

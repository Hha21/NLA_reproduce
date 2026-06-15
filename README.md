# NLA Reproduce

A from-scratch implementation of the Natural Language Autoencoder (NLA) pipeline,
following the [Anthropic NLA paper](https://transformer-circuits.pub/2026/nla/).
Built phase-by-phase on **Qwen2.5-0.5B** for fast iteration, with Llama 3.2 1B as the
target scale. See `WORK_PLAN.md` for the full learning-first approach.

---

## Directory structure

```
NLA_reproduce/
├── .gitignore
├── .venv/                   # local virtual environment (not committed)
├── README.md
├── WORK_PLAN.md
├── requirements.txt
├── scripts/                 # one runnable script per phase
│   ├── phase00_load_model.py        # Phase 0 — load model, confirm shapes
│   ├── phase01_activation_harness.py   (upcoming)
│   ├── phase02_reconstructor.py        (upcoming)
│   ├── phase03_verbalizer.py           (upcoming)
│   ├── phase04_warmstart.py            (upcoming)
│   └── phase05_grpo.py                 (upcoming)
├── data/                    # raw text snippets — gitignored
├── activations/             # cached (snippet, activation) pairs — gitignored
└── checkpoints/             # saved model weights during training — gitignored
```

---

## Environment setup

This project uses a standard Python `venv` (no conda required).

**Create and activate the environment:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Install dependencies:**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

For PyTorch with CUDA (replace `cu121` with your CUDA version — check with `nvcc --version`):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**Verify GPU is visible:**

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## Running the phases

Each script is self-contained and run directly:

```bash
python scripts/phase00_load_model.py
```

Model weights are downloaded automatically from HuggingFace on first run (~1 GB for
Qwen2.5-0.5B) and cached in the default HuggingFace cache (`~/.cache/huggingface/`).

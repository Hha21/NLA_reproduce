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
├── .venv/                        # local virtual environment (not committed)
├── README.md
├── WORK_PLAN.md
├── requirements.txt
├── src/                          # shared library — imported by all scripts
│   ├── config.py                 #   MODEL_ID, PROBE_LAYER, DEVICE, DTYPE
│   ├── model.py                  #   load_target(), load_tokenizer()
│   ├── data.py                   #   activation extraction, (text, activation) dataset
│   ├── ar.py                     #   Reconstructor: text → activation̂
│   ├── av.py                     #   Verbalizer: activation → text description
│   └── train.py                  #   AR supervised loop + AV GRPO loop + FVE metric
├── scripts/                      # thin entry points — run these directly
│   ├── phase00_load_model.py     #   sanity check: load model, confirm shapes
│   ├── generate_data.py          #   build and cache (text, activation) pairs
│   ├── train_ar_baseline.py      #   AR on ground-truth text → oracle FVE ceiling
│   ├── train_warmstart.py        #   SFT warm-start for AV before RL
│   └── train_grpo.py             #   full AV + AR GRPO training
├── data/                         # raw text snippets — gitignored
├── activations/                  # cached (text, activation) pairs — gitignored
└── checkpoints/                  # saved model weights during training — gitignored
```

### Order of execution

```
scripts/phase00_load_model.py     # once — verify GPU and model load
scripts/generate_data.py          # once — build the activation dataset
scripts/train_ar_baseline.py      # establishes the oracle FVE ceiling
scripts/train_warmstart.py        # warm-starts AV with SFT
scripts/train_grpo.py             # full NLA training
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

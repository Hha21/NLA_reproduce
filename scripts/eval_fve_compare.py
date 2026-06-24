"""
Compare e2e FVE between two AV+AR checkpoint pairs on a fixed eval set.

Uses a fixed random seed and a larger sample than training (default 500) so
the comparison is stable and apples-to-apples regardless of which dataset each
training run used for its own val set.

Usage:
    python scripts/eval_fve_compare.py \
      --av-a checkpoints/grpo_av_step1000.pt \
      --ar-a checkpoints/grpo_ar_step1000.pt \
      --av-b checkpoints/grpo_cont_av_step1000.pt \
      --ar-b checkpoints/grpo_cont_ar_step1000.pt \
      --data-dir activations/dataset \
      --n-eval 500
"""
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import torch
from datasets import load_from_disk, concatenate_datasets

from src.config import DEVICE
from src.av import load_av
from src.ar import load_ar
from src.train import eval_e2e_fve

parser = argparse.ArgumentParser()
parser.add_argument("--av-a",      required=True, help="AV checkpoint A")
parser.add_argument("--ar-a",      required=True, help="AR checkpoint A")
parser.add_argument("--av-b",      required=True, help="AV checkpoint B")
parser.add_argument("--ar-b",      required=True, help="AR checkpoint B")
parser.add_argument("--label-a",   default=None,  help="Label for checkpoint A (default: filename)")
parser.add_argument("--label-b",   default=None,  help="Label for checkpoint B (default: filename)")
parser.add_argument("--data-dir",  default="activations/dataset")
parser.add_argument("--n-eval",    type=int, default=500,
                    help="Samples to evaluate on (default: 500)")
parser.add_argument("--seed",      type=int, default=0,
                    help="Seed for sample selection (default: 0)")
parser.add_argument("--gen-batch", type=int, default=8)
args = parser.parse_args()

label_a = args.label_a or Path(args.av_a).stem
label_b = args.label_b or Path(args.av_b).stem

# --- Load dataset ---
def load_ds(data_dir):
    p = Path(data_dir)
    if (p / "dataset_info.json").exists():
        return load_from_disk(str(p))
    shards_dir = p / "shards"
    if shards_dir.exists():
        paths = sorted(s for s in shards_dir.glob("shard_*") if (s / "dataset_info.json").exists())
        if paths:
            return concatenate_datasets([load_from_disk(str(s)) for s in paths])
    raise FileNotFoundError(f"No dataset found at {data_dir!r}")

print(f"Loading dataset from {args.data_dir}...")
ds = load_ds(args.data_dir)
print(f"  {len(ds)} samples total")

rng = np.random.default_rng(args.seed)
n = min(args.n_eval, len(ds))
indices = rng.choice(len(ds), size=n, replace=False)
indices.sort()

val_acts = torch.tensor(
    np.stack(ds.select(indices.tolist())["activation"]),
    dtype=torch.float32,
)
print(f"  Using {n} samples (seed={args.seed}) for eval\n")

# --- Load tokenizer once ---
print("Loading tokenizer...")
_, tok = load_av(DEVICE)

def run_eval(av_path, ar_path, label):
    print(f"Loading {label}...")
    av, _ = load_av(DEVICE)
    av.load_state_dict(torch.load(av_path, map_location=DEVICE))

    ar = load_ar(DEVICE, freeze_base=False)
    ar.load_state_dict(torch.load(ar_path, map_location=DEVICE))

    fve_val = eval_e2e_fve(av, ar, val_acts, tok, DEVICE,
                           n_eval=n, gen_batch=args.gen_batch)
    print(f"  {label}: e2e FVE = {fve_val:.4f}")
    return fve_val

print()
fve_a = run_eval(args.av_a, args.ar_a, label_a)
print()
fve_b = run_eval(args.av_b, args.ar_b, label_b)

print(f"\n{'─'*50}")
print(f"  {label_a:<35} FVE = {fve_a:.4f}")
print(f"  {label_b:<35} FVE = {fve_b:.4f}")
delta = fve_b - fve_a
sign  = "+" if delta >= 0 else ""
print(f"  Δ (B − A)                            {sign}{delta:.4f}")
print(f"{'─'*50}")

os._exit(0)

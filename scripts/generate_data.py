"""
Build and cache the (text_truncated, activation) dataset.

Run once before any training script:
    python scripts/generate_data.py

Output is saved to activations/dataset/ as a HuggingFace Dataset (parquet).
Re-running is a no-op if the directory already exists — pass --overwrite to
regenerate.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np

from src.config import DEVICE
from src.data import build_dataset
from src.model import load_target, load_tokenizer

OUT_DIR = Path("activations/dataset")

parser = argparse.ArgumentParser()
parser.add_argument("--n-samples",  type=int, default=5_000)
parser.add_argument("--overwrite",  action="store_true")
args = parser.parse_args()

if OUT_DIR.exists() and not args.overwrite:
    print(f"Dataset already exists at {OUT_DIR}. Pass --overwrite to regenerate.")
    sys.exit(0)

print("Loading model and tokenizer...")
tok    = load_tokenizer()
target = load_target(DEVICE)

print(f"Building dataset ({args.n_samples} samples)...")
ds = build_dataset(target, tok, n_samples=args.n_samples)

OUT_DIR.mkdir(parents=True, exist_ok=True)
ds.save_to_disk(str(OUT_DIR))

# Print activation norm statistics — useful later for reward scaling in GRPO.
acts  = np.stack(ds["activation"])
norms = np.linalg.norm(acts, axis=-1)
print(f"\nSaved {len(ds)} samples to {OUT_DIR}")
print(f"Activation norms — mean: {norms.mean():.2f}  std: {norms.std():.2f}"
      f"  min: {norms.min():.2f}  max: {norms.max():.2f}")

# Force clean exit before Python's finalizer hits PyTorch's CUDA threads.
sys.exit(0)

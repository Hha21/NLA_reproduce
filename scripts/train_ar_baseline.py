"""
Stage 2 — AR warm-start: supervised regression on (summary, activation) pairs.

Trains the Activation Reconstructor to map text descriptions → activations.
Uses the first half of the dataset (rows 0..n//2) so the second half remains
unseen for AV SFT, matching the reference 50/50 data split.

Usage:
    python scripts/train_ar_baseline.py --n-epochs 2 --batch-size 8   # smoke-test
    python scripts/train_ar_baseline.py                                 # full run
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import os
import numpy as np
import torch
from datasets import load_from_disk

from src.config import DEVICE
from src.ar import load_ar
from src.model import load_tokenizer
from src.train import fve, train_ar

CHECKPOINT = Path("checkpoints/ar_baseline.pt")

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir",      default="activations/dataset")
parser.add_argument("--data-start",    type=int, default=0,
                    help="First dataset row to use (default: 0)")
parser.add_argument("--data-end",      type=int, default=None,
                    help="Last dataset row (exclusive). Default: first half of dataset.")
parser.add_argument("--n-epochs",      type=int,   default=5)
parser.add_argument("--batch-size",    type=int,   default=16)
parser.add_argument("--lr",            type=float, default=None,
                    help="Head LR (default: 1e-4 with --unfreeze-base, 3e-4 head-only)")
parser.add_argument("--base-lr",       type=float, default=None,
                    help="Base LR when --unfreeze-base (default: lr/10).")
parser.add_argument("--unfreeze-base", action="store_true",
                    help="Fine-tune the full transformer, not just the linear head")
parser.add_argument("--text-col",      default="summary",
                    help="Dataset column for AR input text (summary for warm-start, "
                         "text_truncated for oracle ceiling)")
parser.add_argument("--mse-scale",     action="store_true", default=True,
                    help="Normalise activation targets to L2 norm=sqrt(d_model) "
                         "before MSE loss (matches reference mse_scale=sqrt_d_model)")
parser.add_argument("--no-mse-scale",  dest="mse_scale", action="store_false")
args = parser.parse_args()

if args.lr is None:
    args.lr = 1e-4 if args.unfreeze_base else 3e-4
if args.unfreeze_base and args.base_lr is None:
    args.base_lr = args.lr / 10

# --- Dataset ---
print("Loading dataset...")
ds = load_from_disk(args.data_dir)
print(f"  {len(ds)} samples total")

data_end = args.data_end if args.data_end is not None else len(ds) // 2
ds = ds.select(range(args.data_start, data_end))
print(f"  Using rows {args.data_start}..{data_end} ({len(ds)} samples)")

# --- Mean baseline sanity check ---
acts_all  = torch.tensor(np.stack(ds["activation"]), dtype=torch.float32)
mean_pred = acts_all.mean(0, keepdim=True).expand_as(acts_all)
print(f"Mean baseline FVE: {fve(acts_all, mean_pred):.4f}  (expect ~0.0)")

# --- AR ---
freeze = not args.unfreeze_base
print(f"\nLoading AR (freeze_base={freeze})...")
tok = load_tokenizer()
ar  = load_ar(DEVICE, freeze_base=freeze)

n_trainable = sum(p.numel() for p in ar.parameters() if p.requires_grad)
print(f"Trainable parameters: {n_trainable:,}")
print(f"mse_scale: {args.mse_scale}")

# --- Train ---
print()
ar = train_ar(
    ar, ds, tok, DEVICE,
    n_epochs   = args.n_epochs,
    batch_size = args.batch_size,
    lr         = args.lr,
    base_lr    = args.base_lr if not freeze else None,
    text_col   = args.text_col,
    mse_scale  = args.mse_scale,
)

# --- Save ---
CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
if freeze:
    torch.save(ar.head.state_dict(), CHECKPOINT)
    print(f"\nSaved head checkpoint to {CHECKPOINT}")
else:
    torch.save(ar.state_dict(), CHECKPOINT)
    print(f"\nSaved full AR checkpoint to {CHECKPOINT}")

os._exit(0)

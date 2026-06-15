"""
Phase 2 — train the Reconstructor (AR) on ground-truth text (oracle ceiling).

AR sees the original text_truncated — perfect information about what produced
the activation. The FVE achieved here is the upper bound: all NLA results
(where AR sees AV descriptions instead) must stay below this number.

Usage:
    # Quick smoke-test on a small dataset
    python scripts/train_ar_baseline.py --n-epochs 2 --batch-size 8

    # Full run
    python scripts/train_ar_baseline.py
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
parser.add_argument("--n-epochs",      type=int,   default=5)
parser.add_argument("--batch-size",    type=int,   default=16)
parser.add_argument("--lr",            type=float, default=None,
                    help="Learning rate (default: 1e-5 with --unfreeze-base, 3e-4 head-only)")
parser.add_argument("--unfreeze-base", action="store_true",
                    help="Fine-tune the full transformer, not just the linear head")
args = parser.parse_args()

if args.lr is None:
    args.lr = 1e-5 if args.unfreeze_base else 3e-4

# --- Dataset ---
print("Loading dataset...")
ds = load_from_disk(args.data_dir)
print(f"  {len(ds)} samples")

# --- Mean baseline sanity check ---
# Always predicting the corpus mean gives FVE = 0 by construction.
# If this prints something far from 0.0, the FVE implementation is broken.
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

# --- Train ---
print()
ar = train_ar(
    ar, ds, tok, DEVICE,
    n_epochs   = args.n_epochs,
    batch_size = args.batch_size,
    lr         = args.lr,
)

# --- Save ---
# With freeze_base=True only the head changed, so we only need to save
# the head weights. With --unfreeze-base we save the full state dict.
CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
if freeze:
    torch.save(ar.head.state_dict(), CHECKPOINT)
    print(f"\nSaved head checkpoint to {CHECKPOINT}")
else:
    torch.save(ar.state_dict(), CHECKPOINT)
    print(f"\nSaved full AR checkpoint to {CHECKPOINT}")

os._exit(0)

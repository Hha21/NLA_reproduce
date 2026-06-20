"""
Stage 3 — AV warm-start: SFT on (h_l → text_truncated) pairs.

The Activation Verbalizer receives h_l injected as a soft token (㊗) and is
trained to regenerate the original text that produced that activation. This
gives AV a meaningful starting point before GRPO, where it must produce novel
descriptions that allow AR to reconstruct h_l.

Uses the second half of the dataset (rows n//2..n) so AV and AR see disjoint
examples during their respective warm-starts (reference: 50/50 split).

Usage:
    python scripts/train_warmstart.py --n-epochs 1 --batch-size 4   # smoke-test
    python scripts/train_warmstart.py                                 # full run
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import os
import torch
from datasets import load_from_disk

from src.config import DEVICE
from src.av import load_av
from src.train import train_av

CHECKPOINT = Path("checkpoints/av_warmstart.pt")

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir",   default="activations/dataset")
parser.add_argument("--data-start", type=int, default=None,
                    help="First row to use. Default: second half of dataset (n//2).")
parser.add_argument("--data-end",   type=int, default=None,
                    help="Last row (exclusive). Default: end of dataset.")
parser.add_argument("--n-epochs",   type=int,   default=5)
parser.add_argument("--batch-size", type=int,   default=8)
parser.add_argument("--lr",         type=float, default=2e-5)
parser.add_argument("--max-length", type=int,   default=512,
                    help="Max token length for input+response (text_truncated can be long)")
args = parser.parse_args()

# --- Dataset ---
print("Loading dataset...")
ds = load_from_disk(args.data_dir)
print(f"  {len(ds)} samples total")

data_start = args.data_start if args.data_start is not None else len(ds) // 2
data_end   = args.data_end   if args.data_end   is not None else len(ds)
ds = ds.select(range(data_start, data_end))
print(f"  Using rows {data_start}..{data_end} ({len(ds)} samples)")

# --- AV ---
print("\nLoading AV...")
av, tok = load_av(DEVICE)
n_trainable = sum(p.numel() for p in av.parameters() if p.requires_grad)
print(f"Trainable parameters: {n_trainable:,}")

# --- Train ---
print()
av = train_av(
    av, ds, tok, DEVICE,
    n_epochs   = args.n_epochs,
    batch_size = args.batch_size,
    lr         = args.lr,
    max_length = args.max_length,
    text_col   = "text_truncated",
)

# --- Save ---
CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
torch.save(av.state_dict(), CHECKPOINT)
print(f"\nSaved AV checkpoint to {CHECKPOINT}")

os._exit(0)

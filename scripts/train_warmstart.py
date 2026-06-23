"""
Stage 3 — AV warm-start: SFT on (h_l → summary) pairs.

The Activation Verbalizer receives h_l injected as a soft token (㊗) and is
trained to generate the LLM-produced linguistic explanation for that activation.
Using the summary (not text_truncated) ensures the AV and AR share the same
"language": AR was trained to reconstruct from summaries, so AV must learn to
produce summaries of the same style.

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
from src.ar import load_ar
from src.train import train_av

CHECKPOINT    = Path("checkpoints/av_warmstart.pt")
AR_CHECKPOINT = Path("checkpoints/ar_baseline.pt")

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
                    help="Max token length for input+response")
parser.add_argument("--no-fve",     action="store_true",
                    help="Skip end-to-end FVE evaluation (faster, no AR needed in memory)")
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

# --- AR (frozen, for end-to-end FVE only) ---
ar = None
if not args.no_fve:
    if AR_CHECKPOINT.exists():
        print(f"\nLoading frozen AR from {AR_CHECKPOINT} for e2e FVE...")
        import torch
        ar = load_ar(DEVICE, freeze_base=True)
        ar.load_state_dict(torch.load(AR_CHECKPOINT, map_location=DEVICE))
        ar.eval()
        ar.requires_grad_(False)
        print("  AR loaded (frozen, inference only)")
    else:
        print(f"\nAR checkpoint not found at {AR_CHECKPOINT} — skipping e2e FVE")

# --- Train ---
print()
av = train_av(
    av, ds, tok, DEVICE,
    n_epochs        = args.n_epochs,
    batch_size      = args.batch_size,
    lr              = args.lr,
    max_length      = args.max_length,
    text_col        = "summary",
    ar              = ar,
    checkpoint_path = str(CHECKPOINT),
)
print(f"\nBest val_loss checkpoint saved to {CHECKPOINT}")

os._exit(0)

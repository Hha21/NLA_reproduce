"""
Stage 4 — GRPO: joint AV (policy gradient) + AR (supervised MSE).

Loads AV and AR from their SFT warm-start checkpoints, then runs the GRPO
loop: rollout → advantage → policy gradient + KL + AR MSE → joint update.

Hyperparameters match reference rl.sh (scaled down for single-GPU):
  N=16 prompts × K=8 samples = 128 sequences/step  (reference: 128×8 on 8 GPUs)
  lr = 1.41e-5 constant for both AV and AR         (reference rl.sh lines 102, 126)
  KL coef = 0.01                                   (reference rl.sh line 54)
  max_new_tokens = 150                             (reference rl.sh line 108)

Usage:
    python scripts/train_grpo.py --n-steps 10 --no-kl   # smoke-test (no ref AV)
    python scripts/train_grpo.py                          # full run
"""
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import torch
from datasets import load_from_disk

from src.config import DEVICE
from src.av import load_av
from src.ar import load_ar
from src.train import train_grpo

AV_CHECKPOINT = Path("checkpoints/av_warmstart.pt")
AR_CHECKPOINT = Path("checkpoints/ar_baseline.pt")
GRPO_BASE     = Path("checkpoints/grpo")   # → grpo_av_stepN.pt / grpo_ar_stepN.pt

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir",       default="activations/dataset")
parser.add_argument("--n-steps",        type=int,   default=500)
parser.add_argument("--n-prompts",      type=int,   default=16,
                    help="Activations per rollout (N). Reference uses 128 on 8 GPUs.")
parser.add_argument("--n-samples",      type=int,   default=8,
                    help="Descriptions per activation (K, GRPO group size).")
parser.add_argument("--av-lr",          type=float, default=1.41e-5)
parser.add_argument("--ar-lr",          type=float, default=1.41e-5)
parser.add_argument("--kl-coef",        type=float, default=0.01)
parser.add_argument("--max-new-tokens", type=int,   default=150)
parser.add_argument("--rollout-batch",  type=int,   default=4,
                    help="Activations batched per av.generate() call. "
                         "Higher = faster rollout; reduce if OOM during generation.")
parser.add_argument("--save-interval",  type=int,   default=100)
parser.add_argument("--log-interval",   type=int,   default=10)
parser.add_argument("--no-kl",         action="store_true",
                    help="Disable KL penalty — skips loading reference AV. "
                         "Useful for smoke-testing or when memory is tight.")
args = parser.parse_args()

# --- Dataset (use full 100k; RL learns from reward not labels, overlap is OK) ---
print("Loading dataset...")
ds = load_from_disk(args.data_dir)
print(f"  {len(ds)} samples")

# Fixed val set for e2e FVE at checkpoints (200 samples, drawn from the end of the dataset)
N_VAL = 200
val_acts = torch.tensor(
    np.stack(ds.select(range(len(ds) - N_VAL, len(ds)))["activation"]),
    dtype=torch.float32,
)
print(f"  Val set: {N_VAL} samples for e2e FVE at each checkpoint")

# --- AV ---
print("\nLoading AV...")
av, tok = load_av(DEVICE)
if AV_CHECKPOINT.exists():
    av.load_state_dict(torch.load(AV_CHECKPOINT, map_location=DEVICE))
    print(f"  Loaded from {AV_CHECKPOINT}")
else:
    print(f"  WARNING: {AV_CHECKPOINT} not found — starting from base model weights")
av.train()

# --- Reference AV (frozen SFT copy, for KL penalty) ---
av_ref = None
if not args.no_kl:
    if not AV_CHECKPOINT.exists():
        print("\n  WARNING: no AV checkpoint for reference — disabling KL penalty")
        args.no_kl = True
    else:
        print("\nLoading reference AV (frozen, for KL)...")
        av_ref, _ = load_av(DEVICE)
        av_ref.load_state_dict(torch.load(AV_CHECKPOINT, map_location=DEVICE))
        av_ref.eval()
        av_ref.requires_grad_(False)
        print("  Reference AV loaded (frozen)")

# --- AR (trained alongside AV — live reward model) ---
print("\nLoading AR...")
ar = load_ar(DEVICE, freeze_base=False)
if AR_CHECKPOINT.exists():
    ar.load_state_dict(torch.load(AR_CHECKPOINT, map_location=DEVICE))
    print(f"  Loaded from {AR_CHECKPOINT}")
else:
    print(f"  WARNING: {AR_CHECKPOINT} not found — starting from base model weights")
ar.train()

# --- Train ---
print(f"\nStarting GRPO: {args.n_steps} steps, "
      f"N={args.n_prompts} prompts × K={args.n_samples} samples, "
      f"KL={'off' if args.no_kl else args.kl_coef}")
print()

GRPO_BASE.parent.mkdir(parents=True, exist_ok=True)

train_grpo(
    av, av_ref, ar, ds, tok, DEVICE,
    n_steps         = args.n_steps,
    n_prompts       = args.n_prompts,
    n_samples       = args.n_samples,
    av_lr           = args.av_lr,
    ar_lr           = args.ar_lr,
    kl_coef         = 0.0 if args.no_kl else args.kl_coef,
    max_new_tokens  = args.max_new_tokens,
    rollout_batch   = args.rollout_batch,
    checkpoint_path = str(GRPO_BASE),
    save_interval   = args.save_interval,
    log_interval    = args.log_interval,
    val_acts        = val_acts,
)

sys.stdout.flush()
os._exit(0)

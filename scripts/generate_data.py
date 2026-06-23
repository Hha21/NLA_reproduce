"""
Build and cache the (text_truncated, activation) dataset.

Run once before any training script:
    python scripts/generate_data.py

Output is saved to activations/dataset/ as a HuggingFace Dataset (parquet).

For large datasets the script checkpoints a shard every --shard-size samples.
If the run crashes, re-run with the same arguments — completed shards are
detected automatically and skipped so work is not lost.

Re-running on a complete dataset is a no-op; pass --overwrite to regenerate.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import os
import numpy as np

from src.config import DEVICE
from src.data import POSITIONS_PER_DOC, build_dataset
from src.model import load_target, load_tokenizer

DEFAULT_OUT_DIR = Path("activations/dataset")

parser = argparse.ArgumentParser()
parser.add_argument("--n-samples",         type=int, default=5_000)
parser.add_argument("--positions-per-doc", type=int, default=POSITIONS_PER_DOC,
                    help="Extraction points sampled per document (default: %(default)s)")
parser.add_argument("--out-dir",           default=str(DEFAULT_OUT_DIR),
                    help="Output directory for the dataset (default: %(default)s)")
parser.add_argument("--max-seq-len",       type=int, default=4096,
                    help="Cap extraction position to this many tokens. "
                         "Prevents OOM on very long FineWeb documents. 0 = no cap.")
parser.add_argument("--shard-size",        type=int, default=50_000,
                    help="Save an intermediate shard every N samples (default: %(default)s). "
                         "0 = no sharding.")
parser.add_argument("--overwrite",         action="store_true")
args = parser.parse_args()

OUT_DIR    = Path(args.out_dir)
SHARDS_DIR = OUT_DIR / "shards"
shard_size = args.shard_size

# --- Check completion state ---
is_complete = (OUT_DIR / "dataset_info.json").exists()

if is_complete and not args.overwrite:
    print(f"Dataset already exists at {OUT_DIR}. Pass --overwrite to regenerate.")
    sys.exit(0)

if args.overwrite:
    import shutil
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    print("Overwrite: removed existing data.")

# --- Detect completed shards for auto-resume ---
completed_shards = []
if shard_size and SHARDS_DIR.exists():
    completed_shards = sorted(
        p for p in SHARDS_DIR.glob("shard_*")
        if (p / "dataset_info.json").exists()
    )

skip_n = len(completed_shards) * shard_size if shard_size else 0

if skip_n > 0:
    print(f"Resuming: {len(completed_shards)} shards done ({skip_n:,} / {args.n_samples:,} samples). "
          f"Need {args.n_samples - skip_n:,} more.")
else:
    print(f"Fresh start: {args.n_samples:,} samples, max_seq_len={args.max_seq_len}, "
          f"shard_size={shard_size:,}.")

if skip_n < args.n_samples:
    print("Loading model and tokenizer...")
    tok    = load_tokenizer()
    target = load_target(DEVICE)

    if shard_size:
        SHARDS_DIR.mkdir(parents=True, exist_ok=True)

    next_shard_idx = [len(completed_shards)]

    def on_shard(idx, ds):
        path = SHARDS_DIR / f"shard_{idx:04d}"
        ds.save_to_disk(str(path))
        next_shard_idx[0] = idx + 1
        print(f"\n  Shard {idx:04d}: {len(ds):,} samples saved to {path}")

    print(f"Building dataset ({args.n_samples:,} samples, "
          f"{args.positions_per_doc} positions/doc)...")

    ds_tail = build_dataset(
        target, tok,
        n_samples=args.n_samples,
        positions_per_doc=args.positions_per_doc,
        max_position=args.max_seq_len,
        shard_size=shard_size,
        on_shard=on_shard if shard_size else None,
        skip_n=skip_n,
    )

    if not shard_size:
        # No sharding: save the full dataset directly.
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        ds_tail.save_to_disk(str(OUT_DIR))
        acts  = np.stack(ds_tail["activation"])
        norms = np.linalg.norm(acts, axis=-1)
        print(f"\nSaved {len(ds_tail):,} samples to {OUT_DIR}")
        print(f"Activation norms — mean: {norms.mean():.2f}  std: {norms.std():.2f}"
              f"  min: {norms.min():.2f}  max: {norms.max():.2f}")
        os._exit(0)

    # Save the final partial shard (may be empty if n_samples % shard_size == 0).
    if len(ds_tail) > 0:
        on_shard(next_shard_idx[0], ds_tail)

# --- Concatenate all shards into the final dataset ---
print("\nConcatenating shards into final dataset...")
from datasets import concatenate_datasets, load_from_disk

all_shard_paths = sorted(
    p for p in SHARDS_DIR.glob("shard_*")
    if (p / "dataset_info.json").exists()
)
if not all_shard_paths:
    print("ERROR: no completed shards found.")
    sys.exit(1)

print(f"  Loading {len(all_shard_paths)} shards...")
all_ds   = [load_from_disk(str(p)) for p in all_shard_paths]
final_ds = concatenate_datasets(all_ds)

OUT_DIR.mkdir(parents=True, exist_ok=True)
final_ds.save_to_disk(str(OUT_DIR))

acts  = np.stack(final_ds["activation"])
norms = np.linalg.norm(acts, axis=-1)
print(f"\nSaved {len(final_ds):,} samples to {OUT_DIR}")
print(f"Activation norms — mean: {norms.mean():.2f}  std: {norms.std():.2f}"
      f"  min: {norms.min():.2f}  max: {norms.max():.2f}")
print(f"\nNote: shards kept at {SHARDS_DIR} (~{len(all_shard_paths) * 200}MB). "
      f"Delete once you've verified the final dataset.")

os._exit(0)

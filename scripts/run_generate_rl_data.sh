#!/usr/bin/env bash
# Extract 1M activation vectors from FineWeb for GRPO RL training.
# No summaries needed — RL reward comes from AR reconstruction, not labels.
# Saves to activations/rl_dataset (separate from the 100k SFT dataset).
#
# At ~10 positions/doc, 1M samples requires ~100k FineWeb documents.
# Expected time: 2-4 hours on a single GPU.
#
# Crash recovery: the script saves a shard every 50k samples (~5%).
# If it crashes, just re-run — completed shards are detected and skipped.
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

# Reduces fragmentation that can cause OOM after many small allocations.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG="checkpoints/generate_rl_data_$(date +%Y%m%d_%H%M%S).log"
mkdir -p checkpoints

echo "Logging to $LOG"
echo "Started: $(date)" | tee -a "$LOG"

python scripts/generate_data.py \
  --n-samples 1000000 \
  --out-dir activations/rl_dataset \
  --max-seq-len 4096 \
  --shard-size 50000 \
  2>&1 | tee -a "$LOG"

echo "Finished: $(date)" | tee -a "$LOG"

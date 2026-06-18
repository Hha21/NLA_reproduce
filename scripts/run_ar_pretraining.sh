#!/usr/bin/env bash
# Stage 3: AR warm-start — train the affine head on (summary, activation) pairs.
# Requires activations/dataset to have a 'summary' column (run run_generate_summaries.sh first).
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

LOG="checkpoints/ar_pretraining_$(date +%Y%m%d_%H%M%S).log"
mkdir -p checkpoints

echo "Logging to $LOG"
echo "Started: $(date)" | tee -a "$LOG"

python scripts/train_ar_baseline.py \
  --text-col summary \
  --n-epochs 30 \
  --batch-size 32 \
  --lr 1e-4 \
  --unfreeze-base \
  2>&1 | tee -a "$LOG"

echo "Finished: $(date)" | tee -a "$LOG"

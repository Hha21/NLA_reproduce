#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

LOG="checkpoints/ar_unfreeze_$(date +%Y%m%d_%H%M%S).log"
mkdir -p checkpoints

echo "Logging to $LOG"
echo "Started: $(date)" | tee "$LOG"

python scripts/train_ar_baseline.py \
  --unfreeze-base \
  --n-epochs 5 \
  --batch-size 8 \
  2>&1 | tee -a "$LOG"

echo "Finished: $(date)" | tee -a "$LOG"

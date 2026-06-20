#!/usr/bin/env bash
# Stage 3: AV warm-start — SFT on (h_l → text_truncated) using second 50K rows.
# Run after run_ar_pretraining.sh completes.
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

LOG="checkpoints/av_warmstart_$(date +%Y%m%d_%H%M%S).log"
mkdir -p checkpoints

echo "Logging to $LOG"
echo "Started: $(date)" | tee -a "$LOG"

python scripts/train_warmstart.py \
  --n-epochs 5 \
  --batch-size 8 \
  --lr 2e-5 \
  --max-length 512 \
  2>&1 | tee -a "$LOG"

echo "Finished: $(date)" | tee -a "$LOG"

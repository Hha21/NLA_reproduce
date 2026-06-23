#!/usr/bin/env bash
# Stage 3: AV warm-start — SFT on (h_l → summary) using second 50K rows.
# Run after run_ar_pretraining.sh completes.
# Reference uses 1 epoch on 250k samples; with 50k samples, 3 epochs is equivalent
# data exposure. Best checkpoint (by val_loss) is saved automatically.
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

LOG="checkpoints/av_warmstart_$(date +%Y%m%d_%H%M%S).log"
mkdir -p checkpoints

echo "Logging to $LOG"
echo "Started: $(date)" | tee -a "$LOG"

python scripts/train_warmstart.py \
  --n-epochs 3 \
  --batch-size 8 \
  --lr 2e-5 \
  --max-length 512 \
  2>&1 | tee -a "$LOG"

echo "Finished: $(date)" | tee -a "$LOG"

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

LOG="checkpoints/ar_baseline_$(date +%Y%m%d_%H%M%S).log"
mkdir -p checkpoints

echo "Logging to $LOG"
echo "Started: $(date)" | tee -a "$LOG"
echo "Python: $(which python)" | tee -a "$LOG"

# Train the AR affine head on oracle text (freeze_base=True).
#
# AR is truncated at PROBE_LAYER so last_hidden_state = raw x_l.
# The head only needs to learn the consistent effect of the prompt prefix —
# a much easier problem than before. Target FVE: ~0.3-0.4.
echo "" | tee -a "$LOG"
echo "=== AR oracle pretraining (frozen base, head only) ===" | tee -a "$LOG"
echo "Started: $(date)" | tee -a "$LOG"

python scripts/train_ar_baseline.py \
  --n-epochs 30 \
  --batch-size 32 \
  --lr 3e-4 \
  2>&1 | tee -a "$LOG"

echo "Finished: $(date)" | tee -a "$LOG"

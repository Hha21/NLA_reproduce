#!/usr/bin/env bash
# Stage 0+1: extract activations from FineWeb and save dataset to disk.
# Run this once, then run run_generate_summaries.sh and run_ar_pretraining.sh.
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

LOG="activations/generate_data_$(date +%Y%m%d_%H%M%S).log"
mkdir -p activations checkpoints

echo "Logging to $LOG"
echo "Started: $(date)" | tee -a "$LOG"

python scripts/generate_data.py \
  --n-samples 100000 \
  --overwite \
  2>&1 | tee -a "$LOG"

echo "Finished: $(date)" | tee -a "$LOG"

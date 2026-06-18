#!/usr/bin/env bash
# Stage 2: generate DeepSeek summaries for every text_truncated in the dataset.
# Requires OPENROUTER_API_KEY to be set.
# Safe to interrupt — completed summaries are checkpointed and skipped on rerun.
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
    echo "ERROR: DEEPSEEK_API_KEY is not set."
    exit 1
fi

LOG="activations/generate_summaries_$(date +%Y%m%d_%H%M%S).log"
mkdir -p activations

echo "Logging to $LOG"
echo "Started: $(date)" | tee -a "$LOG"

python scripts/generate_summaries.py \
  --model "deepseek-v4-flash" \
  --concurrency 1000 \
  2>&1 | tee -a "$LOG"

echo "Finished: $(date)" | tee -a "$LOG"

#!/usr/bin/env bash
# Stage 4: GRPO joint AV + AR training.
# Run after run_av_warmstart.sh and run_ar_pretraining.sh complete.
#
# Hyperparameters from reference rl.sh scaled to single-GPU:
#   N=16 prompts × K=8 samples = 128 sequences/step (reference: 128×8 on 8 GPUs)
#   lr=1.41e-5 constant, KL coef=0.01, max_new_tokens=150
#
# Smoke-test: --n-steps 5 --no-kl (skips loading reference AV, fast sanity check)
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

LOG="checkpoints/grpo_$(date +%Y%m%d_%H%M%S).log"
mkdir -p checkpoints

echo "Logging to $LOG"
echo "Started: $(date)" | tee -a "$LOG"

python scripts/train_grpo.py \
  --n-steps 500 \
  --n-prompts 16 \
  --n-samples 8 \
  --av-lr 1.41e-5 \
  --ar-lr 1.41e-5 \
  --kl-coef 0.01 \
  --max-new-tokens 150 \
  --save-interval 100 \
  --log-interval 10 \
  2>&1 | tee -a "$LOG"

echo "Finished: $(date)" | tee -a "$LOG"

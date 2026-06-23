#!/usr/bin/env bash
# Stage 4: GRPO joint AV + AR training.
# Run after run_av_warmstart.sh and run_ar_pretraining.sh complete.
#
# Hyperparameters from reference rl.sh scaled to single-GPU:
#   N=16 prompts × K=8 samples = 128 sequences/step  (reference: 128×8 on 8 GPUs)
#   lr=1.41e-5 constant, KL coef=0.01, max_new_tokens=150
#   rollout_batch=4: generates 4 activations × 8 samples = 32 seqs per av.generate()
#                    ~3x faster than rollout_batch=1 (original default)
#
# Data: uses activations/rl_dataset (1M samples) if available,
#       falls back to activations/dataset (100k) for quick iteration.
#
# Smoke-test: --n-steps 5 --no-kl --data-dir activations/dataset
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

# Use the large RL dataset if it exists, else fall back to the original 100k
if [ -d "activations/rl_dataset" ]; then
  DATA_DIR="activations/rl_dataset"
else
  DATA_DIR="activations/dataset"
  echo "NOTE: activations/rl_dataset not found — using 100k dataset. Run run_generate_rl_data.sh first for the full scaled run."
fi

LOG="checkpoints/grpo_$(date +%Y%m%d_%H%M%S).log"
mkdir -p checkpoints

echo "Data: $DATA_DIR"
echo "Logging to $LOG"
echo "Started: $(date)" | tee -a "$LOG"

python scripts/train_grpo.py \
  --data-dir "$DATA_DIR" \
  --n-steps 5000 \
  --n-prompts 16 \
  --n-samples 8 \
  --rollout-batch 4 \
  --av-lr 1.41e-5 \
  --ar-lr 1.41e-5 \
  --kl-coef 0.01 \
  --max-new-tokens 150 \
  --save-interval 1000 \
  --log-interval 50 \
  2>&1 | tee -a "$LOG"

echo "Finished: $(date)" | tee -a "$LOG"

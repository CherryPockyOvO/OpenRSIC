#!/usr/bin/env bash
set -e

# ==============================================================================
# Stage 2: Single RTX 5090 (32GB VRAM) RK3588 INT8 QAT Fine-Tuning
# ==============================================================================

RESUME_ARG=""
if [ -f "checkpoints_qat8/latest.pt" ]; then
    echo "📌 Found checkpoints_qat8/latest.pt, automatically resuming QAT training..."
    RESUME_ARG="--resume checkpoints_qat8/latest.pt"
fi

python train.py \
  --quality-profile rsic_qat8 \
  --decoder-type swin \
  --init-checkpoint checkpoints_fp/best.pt \
  --train-dir datasets/train \
  --val-dir datasets/val \
  --checkpoint-dir checkpoints_qat8 \
  --batch-size 32 \
  --num-workers 8 \
  $RESUME_ARG

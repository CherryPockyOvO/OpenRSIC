#!/usr/bin/env bash
set -e

# ==============================================================================
# Stage 2: CompressAI Official Highest Quality Standard (Quality 8: lambda=0.1800)
# RK3588 INT8 / FP16 Mixed Precision QAT Fine-Tuning
# Target: Zero Loss in PSNR/SSIM when deployed on RK3588 NPU
# ==============================================================================

RESUME_ARG=""
if [ -f "checkpoints_qat8/latest.pt" ]; then
    echo "📌 Found checkpoints_qat8/latest.pt, automatically resuming QAT training..."
    RESUME_ARG="--resume checkpoints_qat8/latest.pt"
fi

python train.py \
  --quality-profile compressai_q8_qat8 \
  --decoder-type swin \
  --init-checkpoint checkpoints_fp/best.pt \
  --train-dir ../datasets/train \
  --val-dir ../datasets/val \
  --checkpoint-dir checkpoints_qat8 \
  --batch-size 32 \
  --epochs 40 \
  --lr 1e-5 \
  --num-workers 8 \
  $RESUME_ARG

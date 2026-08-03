#!/usr/bin/env bash
set -e

# ==============================================================================
# Stage 1: CompressAI Official Highest Quality Standard (Quality 8: lambda=0.1800)
# Full-Precision Pre-training (RTX 5090 / 4090 / 3090)
# Target: PSNR > 40 dB, SSIM > 0.98, High Fidelity Remote Sensing Reconstruction
# ==============================================================================

RESUME_ARG=""
if [ -f "checkpoints_fp/latest.pt" ]; then
    echo "📌 Found checkpoints_fp/latest.pt, automatically resuming training..."
    RESUME_ARG="--resume checkpoints_fp/latest.pt"
fi

python train.py \
  --quality-profile compressai_q8_fp \
  --decoder-type swin \
  --train-dir ../datasets/train \
  --val-dir ../datasets/val \
  --checkpoint-dir checkpoints_fp \
  --batch-size 32 \
  --epochs 200 \
  --lr 1e-4 \
  --num-workers 8 \
  $RESUME_ARG

#!/usr/bin/env bash
set -e

# ==============================================================================
# Stage 1: CompressAI Official Highest Quality Standard (Quality 8: lambda=0.1800)
# Full-Precision Multi-GPU Pre-training (4x GPUs Distributed)
# Target: PSNR > 40 dB, SSIM > 0.98, High Fidelity Remote Sensing Reconstruction
# ==============================================================================

RESUME_ARG=""
if [ -f "checkpoints_fp/latest.pt" ]; then
    echo "📌 Found checkpoints_fp/latest.pt, automatically resuming training..."
    RESUME_ARG="--resume checkpoints_fp/latest.pt"
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py \
  --quality-profile compressai_q8_fp \
  --decoder-type swin \
  --train-dir ../datasets2/train \
  --val-dir ../datasets2/val \
  --checkpoint-dir checkpoints_fp \
  --batch-size 32 \
  --epochs 1000 \
  --lr 1e-4 \
  --num-workers 8 \
  $RESUME_ARG

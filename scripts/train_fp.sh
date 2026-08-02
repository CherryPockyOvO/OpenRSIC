#!/usr/bin/env bash
set -e

# ==============================================================================
# Stage 1: Single RTX 5090 (32GB VRAM) Full-Precision Pre-training
# ==============================================================================

RESUME_ARG=""
if [ -f "checkpoints_fp/latest.pt" ]; then
    echo "📌 Found checkpoints_fp/latest.pt, automatically resuming training..."
    RESUME_ARG="--resume checkpoints_fp/latest.pt"
fi

python train.py \
  --quality-profile rsic_fp \
  --decoder-type swin \
  --train-dir datasets/train \
  --val-dir datasets/val \
  --checkpoint-dir checkpoints_fp \
  --batch-size 32 \
  --num-workers 8 \
  $RESUME_ARG

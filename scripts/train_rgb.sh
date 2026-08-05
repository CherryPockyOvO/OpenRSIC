#!/usr/bin/env bash
set -e

# ==============================================================================
# RGB Training Script: CompressAI Quality 8 High-Fidelity RGB Compression
# ==============================================================================

RESUME_ARG=""
if [ -f "checkpoints_rgb/latest.pt" ]; then
    echo "📌 Found checkpoints_rgb/latest.pt, automatically resuming RGB training..."
    RESUME_ARG="--resume checkpoints_rgb/latest.pt"
elif [ -f "checkpoints_fp/latest.pt" ]; then
    echo "📌 Found checkpoints_fp/latest.pt, automatically resuming training..."
    RESUME_ARG="--resume checkpoints_fp/latest.pt"
fi

CKPT_DIR="checkpoints_rgb"
if [ ! -d "checkpoints_rgb" ] && [ -d "checkpoints_fp" ]; then
    CKPT_DIR="checkpoints_fp"
fi

python train.py \
  --quality-profile compressai_q8_fp \
  --decoder-type swin \
  --train-dir ../datasets/train \
  --val-dir ../datasets/val \
  --checkpoint-dir $CKPT_DIR \
  --batch-size 32 \
  --epochs 200 \
  --lr 1e-4 \
  --num-workers 8 \
  $RESUME_ARG

#!/usr/bin/env bash
set -e

# ==============================================================================
# OpenRSIC RGB Model Training Script (train_rgb.sh)
# Optimized LR (4e-5) with CosineAnnealingLR Scheduler
# ==============================================================================

RESUME_ARG=""
if [ -f "checkpoints_fp/latest.pt" ]; then
    echo "📌 Found checkpoints_fp/latest.pt, automatically resuming training..."
    RESUME_ARG="--resume checkpoints_fp/latest.pt"
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py \
  --quality-profile compressai_q8_fp \
  --decoder-type swin \
  --train-dir ../datasets/train \
  --val-dir ../datasets/val \
  --checkpoint-dir checkpoints_fp \
  --batch-size 32 \
  --epochs 500 \
  --lr 4e-5 \
  --min-lr 1e-6 \
  --scheduler cosine \
  --num-workers 8 \
  $RESUME_ARG

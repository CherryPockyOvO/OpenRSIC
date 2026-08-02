#!/usr/bin/env bash
set -e

# ==============================================================================
# Stage 1: Single RTX 5090 (32GB VRAM) Full-Precision Pre-training
# ==============================================================================

python train.py \
  --quality-profile rsic_fp \
  --decoder-type swin \
  --train-dir data/train \
  --val-dir data/val \
  --checkpoint-dir checkpoints_fp \
  --batch-size 32 \
  --num-workers 8

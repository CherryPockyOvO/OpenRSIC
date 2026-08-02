#!/usr/bin/env bash
set -e

# ==============================================================================
# Stage 2: Single RTX 5090 (32GB VRAM) RK3588 INT8 QAT Fine-Tuning
# ==============================================================================

python train.py \
  --quality-profile hyper_ms_nano_qat8 \
  --decoder-type swin \
  --init-checkpoint checkpoints_fp/best.pt \
  --train-dir data/train \
  --val-dir data/val \
  --checkpoint-dir checkpoints_qat8 \
  --batch-size 32 \
  --num-workers 8

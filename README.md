# OpenRSIC

Symmetric Mean-Scale Hyperprior Image Compression Neural Network engineered for **RK3588 NPU Edge Compression** and **PC / Server High-Quality Reconstruction (Swin Transformer Decoder)**.

---

## 1. Model Architecture Overview

- **Model Class**: `NanoHyperMeanScaleQ` (`rsic/models.py`)
- **Default Variant**: `nano_hyper_ms_q_nano` ($N=160, M=256, Z=128$)
- **Symmetric Design**:
  - **Encoder ($g_a$)**: 16x downsampling NPU-friendly residual encoder (FP16 / INT8 mixed precision, `ReLU6` activations, no BatchNorm, 1x1 skip connections).
  - **Hyper Encoder ($h_a$)**: 4x downsampling hyper-latent encoder ($M \to Z$).
  - **Hyper Decoder ($h_s$)**: Predicts mean ($\boldsymbol{\mu}_y$) and scale ($\boldsymbol{\sigma}_y$) parameter maps from quantized hyper-latent $\hat{z}$.
  - **Decoder ($g_s$)**: 16x upsampling 1:1 balanced PC synthesis decoder (`decoder_channels=160`, `decoder_res_blocks=1`, `refinement_blocks=1`).

---

## 2. Model Training & Quantization (QAT)

### Full-Precision Training (FP)
```bash
torchrun --standalone --nproc_per_node=3 train.py \
  --quality-profile hyper_ms_nano_fp \
  --train-dir data/train \
  --val-dir data/val \
  --checkpoint-dir checkpoints_rsic \
  --checkpoint-interval-steps 100 \
  --eval-interval-steps 100 \
  --num-workers 4
```

### Quantization-Aware Training (QAT 8-bit)
```bash
torchrun --standalone --nproc_per_node=3 train.py \
  --quality-profile hyper_ms_nano_qat8 \
  --init-checkpoint checkpoints_rsic/best.pt \
  --train-dir data/train \
  --val-dir data/val \
  --checkpoint-dir checkpoints_rsic_qat8 \
  --lr 2e-6 \
  --no-amp \
  --checkpoint-interval-steps 100 \
  --eval-interval-steps 100 \
  --num-workers 4
```

---

## 3. Project Structure

```text
.
├── rsic/                 # PyTorch RSIC model definitions, layers, entropy & image utils
│   ├── models.py         # NanoHyperMeanScaleQ model definition
│   ├── entropy.py        # NanoEntropyBottleneck & GaussianConditionalEntropy
│   ├── layers.py         # Conv/Deconv & QuantResidualBlock layers
│   └── utils.py          # Tensor, image & checkpoint I/O utilities
├── train.py              # Multi-GPU PyTorch DDP training & QAT script
├── requirements.txt      # PyTorch dependencies
└── README.md             # Project documentation
```

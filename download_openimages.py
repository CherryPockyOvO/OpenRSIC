#!/usr/bin/env python3
"""Download 20,000 OpenImages v7 RGB images using FiftyOne into datasets2/train."""

import os
from pathlib import Path

import fiftyone as fo
import fiftyone.zoo as foz

export_dir = "/root/autodl-tmp/datasets2/train"
os.makedirs(export_dir, exist_ok=True)

print("🚀 Starting download of 20,000 OpenImages v7 RGB images for CompressAI training...")

dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    label_types=[],  # Skip bounding box / segmentation annotation files
    max_samples=20000,  # Download 20,000 diverse RGB images
    shuffle=True,  # Random sampling for maximum scene diversity
)

print(f"📦 Exporting 20,000 images to: {export_dir}...")
dataset.export(
    export_dir=export_dir,
    dataset_type=fo.types.ImageDirectory,
    overwrite=False,
)

print(f"✅ Download and export complete! Images saved in: {export_dir}")

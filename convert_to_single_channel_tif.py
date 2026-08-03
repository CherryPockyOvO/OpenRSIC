#!/usr/bin/env python3
"""Convert RGB images into Single-Channel (Grayscale) Remote Sensing TIF files.

Supports:
- Single file conversion or batch folder conversion
- Standard ITU-R 601 luminance weighting: Gray = 0.299*R + 0.587*G + 0.114*B
- Output bit-depth options: 8-bit (uint8), 16-bit (uint16), or 32-bit (float32)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def convert_single_image(input_path: Path, output_path: Path, bit_depth: int = 8) -> None:
    """Convert one RGB image into a single-channel TIF file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    path_str = str(input_path)

    # 1. Read Image
    try:
        arr = cv2.imread(path_str, cv2.IMREAD_UNCHANGED)
        if arr is not None and arr.ndim == 3 and arr.shape[2] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    except Exception:
        arr = None

    if arr is None:
        img = Image.open(path_str)
        arr = np.array(img)

    arr_f32 = arr.astype(np.float32)

    # 2. Convert RGB to 1-Channel Grayscale using ITU-R 601 formula
    if arr_f32.ndim == 3 and arr_f32.shape[2] >= 3:
        r, g, b = arr_f32[:, :, 0], arr_f32[:, :, 1], arr_f32[:, :, 2]
        gray_f32 = 0.299 * r + 0.587 * g + 0.114 * b
    elif arr_f32.ndim == 2:
        gray_f32 = arr_f32
    elif arr_f32.ndim == 3 and arr_f32.shape[2] == 1:
        gray_f32 = arr_f32[:, :, 0]
    else:
        raise ValueError(f"Unsupported image shape: {arr_f32.shape}")

    # 3. Scale and cast to target bit-depth
    if bit_depth == 8:
        # Scale to uint8 (0 ~ 255)
        max_val = gray_f32.max()
        min_val = gray_f32.min()
        if max_val > 255.0 or min_val < 0.0:
            gray_norm = (gray_f32 - min_val) / (max_val - min_val + 1e-7) * 255.0
        else:
            gray_norm = gray_f32
        out_arr = np.clip(gray_norm, 0, 255).astype(np.uint8)

    elif bit_depth == 16:
        # Scale to uint16 (0 ~ 65535)
        max_val = gray_f32.max()
        min_val = gray_f32.min()
        if max_val <= 1.0:
            gray_norm = gray_f32 * 65535.0
        elif max_val <= 255.0:
            gray_norm = (gray_f32 / 255.0) * 65535.0
        else:
            gray_norm = (gray_f32 - min_val) / (max_val - min_val + 1e-7) * 65535.0
        out_arr = np.clip(gray_norm, 0, 65535).astype(np.uint16)

    elif bit_depth == 32:
        # Output float32
        out_arr = gray_f32.astype(np.float32)
    else:
        raise ValueError(f"Unsupported bit depth: {bit_depth}. Choose 8, 16, or 32.")

    # 4. Save as 1-Channel TIF file
    cv2.imwrite(str(output_path), out_arr)
    print(f"  ✅ Converted: {input_path.name} -> {output_path} ({out_arr.dtype}, shape: {out_arr.shape})")


def batch_convert(input_dir: Path, output_dir: Path, bit_depth: int = 8) -> None:
    """Batch convert all images in input_dir to single-channel TIF files."""
    files = sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not files:
        print(f"⚠️ No images found in {input_dir}")
        return

    print(f"🚀 Batch converting {len(files)} images in {input_dir} to {bit_depth}-bit single-channel TIF...")
    for file_path in files:
        rel_path = file_path.relative_to(input_dir)
        out_path = output_dir / rel_path.with_suffix(".tif")
        convert_single_image(file_path, out_path, bit_depth=bit_depth)
    print(f"🎉 Completed! Converted {len(files)} files saved to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert RGB images to Single-Channel TIF files.")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input image file or folder path.")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output TIF file or folder path.")
    parser.add_argument(
        "--bit-depth",
        "-b",
        type=int,
        choices=[8, 16, 32],
        default=8,
        help="Target single-channel bit depth: 8 (uint8), 16 (uint16), or 32 (float32). Default: 8.",
    )
    args = parser.parse_args()

    if args.input.is_dir():
        batch_convert(args.input, args.output, bit_depth=args.bit_depth)
    else:
        convert_single_image(args.input, args.output, bit_depth=args.bit_depth)


if __name__ == "__main__":
    main()

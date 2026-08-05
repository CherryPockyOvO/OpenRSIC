#!/usr/bin/env python3
"""OpenRSIC Remote Sensing Image Compression & Decompression Simulation Script.

Simulates the end-to-end workflow:
1. Input Remote Sensing TIF (8-bit / 16-bit / 32-bit, single-channel / RGB)
2. Decouple GeoTIFF spatial metadata
3. Model Compress -> Package into .rsic binary bitstream
4. Model Decompress -> Restore native bit-depth TIF
5. Re-inject GeoTIFF spatial metadata
6. Print detailed Compression Metrics (BPP, Compression Ratio, PSNR, SSIM, File Sizes)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from rsic import (
    RSIC,
    extract_geotiff_metadata,
    get_model,
    inject_geotiff_metadata,
    load_checkpoint,
)
from rsic.utils import (
    calculate_color_and_reconstruction_metrics,
    crop_to_size,
    pad_to_multiple,
    read_remote_sensing_tif,
    tensor_to_remote_sensing_tif,
)

RSIC_FILE_HEADER_MAGIC = b"RSIC1"
RSIC_FILE_HEADER_FORMAT = "<5sBIfId"  # magic, orig_channels, min_val, max_val, header_len


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate Structural Similarity Index (SSIM) between two numpy arrays [H, W, C] or [H, W]."""
    import cv2

    if img1.shape != img2.shape:
        return 0.0
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.T)

    if img1.ndim == 3:
        ssims = []
        for i in range(img1.shape[2]):
            mu1 = cv2.filter2D(img1[:, :, i], -1, window)
            mu2 = cv2.filter2D(img2[:, :, i], -1, window)
            mu1_sq = mu1**2
            mu2_sq = mu2**2
            mu1_mu2 = mu1 * mu2
            sigma1_sq = cv2.filter2D(img1[:, :, i] ** 2, -1, window) - mu1_sq
            sigma2_sq = cv2.filter2D(img2[:, :, i] ** 2, -1, window) - mu2_sq
            sigma12 = cv2.filter2D(img1[:, :, i] * img2[:, :, i], -1, window) - mu1_mu2
            ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
                (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
            )
            ssims.append(ssim_map.mean())
        return float(np.mean(ssims))
    else:
        mu1 = cv2.filter2D(img1, -1, window)
        mu2 = cv2.filter2D(img2, -1, window)
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1**2, -1, window) - mu1_sq
        sigma2_sq = cv2.filter2D(img2**2, -1, window) - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window) - mu1_mu2
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
            (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        )
        return float(ssim_map.mean())


def compress_image(
    model: RSIC,
    input_path: Path,
    output_rsic_path: Path,
    device: torch.device,
) -> tuple[float, float, float]:
    """Compress TIF image to .rsic bitstream."""
    print(f"\n📦 [1/2] Compressing: {input_path.name}")
    t0 = time.perf_counter()

    # 1. Read TIF & metadata
    tensor, orig_dtype, min_val, max_val, orig_channels = read_remote_sensing_tif(input_path)
    geotiff_meta = extract_geotiff_metadata(input_path)

    # 2. Prepare Tensor [1, 3, H, W]
    x = tensor.unsqueeze(0).to(device)
    x_padded, original_size = pad_to_multiple(x, multiple=64)

    # 3. Model Neural Compress Pass
    with torch.no_grad():
        payload = model.compress(x_padded)

    # 4. Create metadata JSON header
    meta_dict = {
        "orig_dtype": str(orig_dtype),
        "min_val": float(min_val),
        "max_val": float(max_val),
        "orig_channels": int(orig_channels),
        "original_size": original_size,
        "geotiff_meta": geotiff_meta,
        "z_shape": payload["z_shape"],
        "y_shape": payload["y_shape"],
        "quant_step": payload["quant_step"],
    }
    meta_bytes = json.dumps(meta_dict).encode("utf-8")

    # 5. Pack binary bitstream: RSIC header + meta JSON + z_bytes + y_bytes
    with open(output_rsic_path, "wb") as f:
        f.write(RSIC_FILE_HEADER_MAGIC)
        f.write(struct.pack("<I", len(meta_bytes)))
        f.write(meta_bytes)
        f.write(struct.pack("<I", len(payload["z_bytes"])))
        f.write(payload["z_bytes"])
        f.write(struct.pack("<I", len(payload["y_bytes"])))
        f.write(payload["y_bytes"])

    t1 = time.perf_counter()
    encode_time_ms = (t1 - t0) * 1000.0

    raw_file_size = os.path.getsize(input_path)
    rsic_file_size = os.path.getsize(output_rsic_path)

    total_pixels = original_size[0] * original_size[1]
    bpp = (rsic_file_size * 8.0) / float(total_pixels)
    compression_ratio = float(raw_file_size) / float(max(1, rsic_file_size))

    print(f"  └─ Encoding Time: {encode_time_ms:.2f} ms")
    print(f"  └─ Raw TIF Size : {raw_file_size / 1024.0:.2f} KB")
    print(f"  └─ Compressed   : {rsic_file_size / 1024.0:.2f} KB (.rsic)")
    print(f"  └─ Actual BPP   : {bpp:.4f} bpp")
    print(f"  └─ Ratio        : {compression_ratio:.2f}x")

    return bpp, compression_ratio, encode_time_ms


def decompress_image(
    model: RSIC,
    input_rsic_path: Path,
    output_tif_path: Path,
    device: torch.device,
) -> float:
    """Decompress .rsic bitstream to native TIF image."""
    print(f"\n📂 [2/2] Decompressing: {input_rsic_path.name}")
    t0 = time.perf_counter()

    # 1. Read binary bitstream
    with open(input_rsic_path, "rb") as f:
        magic = f.read(5)
        if magic != RSIC_FILE_HEADER_MAGIC:
            raise ValueError(f"Invalid RSIC file magic: {magic!r}")
        meta_len = struct.unpack("<I", f.read(4))[0]
        meta_bytes = f.read(meta_len)
        meta_dict = json.loads(meta_bytes.decode("utf-8"))

        z_len = struct.unpack("<I", f.read(4))[0]
        z_bytes = f.read(z_len)
        y_len = struct.unpack("<I", f.read(4))[0]
        y_bytes = f.read(y_len)

    payload = {
        "z_bytes": z_bytes,
        "y_bytes": y_bytes,
        "z_shape": meta_dict["z_shape"],
        "y_shape": meta_dict["y_shape"],
        "quant_step": meta_dict["quant_step"],
    }

    # 2. Model Neural Decompress Pass
    with torch.no_grad():
        x_hat_padded = model.decompress(payload, device=device)
        x_hat = crop_to_size(x_hat_padded, tuple(meta_dict["original_size"]))

    # 3. Save Native Bit-Depth TIF
    tensor_to_remote_sensing_tif(
        tensor=x_hat.squeeze(0),
        orig_dtype=meta_dict["orig_dtype"],
        min_val=meta_dict["min_val"],
        max_val=meta_dict["max_val"],
        save_path=output_tif_path,
        orig_channels=meta_dict["orig_channels"],
    )

    # 4. Re-inject GeoTIFF Metadata if present
    if meta_dict.get("geotiff_meta"):
        inject_geotiff_metadata(output_tif_path, meta_dict["geotiff_meta"])

    t1 = time.perf_counter()
    decode_time_ms = (t1 - t0) * 1000.0

    print(f"  └─ Decoding Time: {decode_time_ms:.2f} ms")
    print(f"  └─ Saved TIF to : {output_tif_path}")

    return decode_time_ms


def compute_color_difference(img1_rgb: np.ndarray, img2_rgb: np.ndarray) -> tuple[float, float]:
    """Compute CIELAB Delta E (CIE76) and RGB Channel MAE.

    Returns:
        delta_e: CIELAB Delta E perceptual color difference (< 1.0 means imperceptible color change).
        rgb_mae: Mean Absolute Error across RGB channels (0 ~ 255 scale).
    """
    try:
        import cv2

        u8_1 = (np.clip(img1_rgb, 0.0, 1.0) * 255.0).astype(np.uint8) if img1_rgb.dtype != np.uint8 else img1_rgb
        u8_2 = (np.clip(img2_rgb, 0.0, 1.0) * 255.0).astype(np.uint8) if img2_rgb.dtype != np.uint8 else img2_rgb

        rgb_mae = float(np.mean(np.abs(u8_1.astype(np.float32) - u8_2.astype(np.float32))))

        lab1 = cv2.cvtColor(u8_1, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab2 = cv2.cvtColor(u8_2, cv2.COLOR_RGB2LAB).astype(np.float32)

        l1, a1, b1 = lab1[:, :, 0] * (100.0 / 255.0), lab1[:, :, 1] - 128.0, lab1[:, :, 2] - 128.0
        l2, a2, b2 = lab2[:, :, 0] * (100.0 / 255.0), lab2[:, :, 1] - 128.0, lab2[:, :, 2] - 128.0

        delta_e = np.sqrt((l1 - l2) ** 2 + (a1 - a2) ** 2 + (b1 - b2) ** 2)
        return float(np.mean(delta_e)), rgb_mae
    except Exception:
        rgb_mae = float(np.mean(np.abs(img1_rgb.astype(np.float32) - img2_rgb.astype(np.float32))))
        return 0.0, rgb_mae


def evaluate_simulation(
    input_tif: Path,
    reconstructed_tif: Path,
    bpp: float,
    compression_ratio: float,
    encode_ms: float,
    decode_ms: float,
) -> None:
    """Calculate and display PSNR, MSE, SSIM, and color difference quality metrics."""
    raw_tensor, orig_dtype, min_val, max_val, orig_channels = read_remote_sensing_tif(input_tif)
    rec_tensor, _, _, _, _ = read_remote_sensing_tif(reconstructed_tif)

    mse_val = float(F.mse_loss(rec_tensor, raw_tensor).item())
    psnr_val = 99.99 if mse_val <= 1e-10 else float(10.0 * math.log10(1.0 / mse_val))

    # Compute native scale MAE error for 32-bit float / 16-bit uint
    native_mae = float(torch.mean(torch.abs(rec_tensor - raw_tensor)).item()) * (max_val - min_val if max_val > min_val else 1.0)

    raw_arr = (raw_tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    rec_arr = (rec_tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    ssim_val = calculate_ssim(raw_arr, rec_arr)
    delta_e, rgb_mae = compute_color_difference(raw_arr, rec_arr)

    dtype_str = str(orig_dtype)

    color_metrics = calculate_color_and_reconstruction_metrics(
        img_orig=raw_tensor,
        img_recon=rec_tensor,
        is_tif=True,
        orig_dtype=str(orig_dtype),
        min_val=min_val,
        max_val=max_val,
        orig_channels=orig_channels,
    )

    print("\n" + "=" * 60)
    print(" 🚀 OpenRSIC Neural Compression Simulation Results")
    print("=" * 60)
    print(f"  📄 Input Image        : {input_tif}")
    print(f"  💾 Bitstream Output    : {reconstructed_tif.with_suffix('.rsic')}")
    print(f"  🖼️  Reconstructed TIF  : {reconstructed_tif} ({orig_channels} 通道)")
    print(f"  📊 Bitrate (BPP)      : {bpp:.4f} bpp")
    print(f"  📦 Compression Ratio  : {compression_ratio:.2f}x")
    print(f"  📈 PSNR               : {psnr_val:.2f} dB")
    print(f"  ✨ SSIM               : {ssim_val:.4f}")
    if orig_channels == 1:
        print("  --------------------------------------------------------")
        print(f"  📏 MAE (Native Scale)  : {color_metrics.get('native_mae', 0.0):.6f}")
        print(f"  📐 RMSE (Native Scale) : {color_metrics.get('native_rmse', 0.0):.6f}")
        print(f"  💥 MaxAE (Peak Error)  : {color_metrics.get('max_abs_error', 0.0):.6f}")
        print(f"  📊 Relative Error (MAPE): {color_metrics.get('relative_error_pct', 0.0):.4f} %")
        print(f"  📡 SNR (Signal-Noise)  : {color_metrics.get('snr_db', 0.0):.2f} dB")
    else:
        print("  --------------------------------------------------------")
        print(f"  🎨 CIEDE2000 色差 ΔE00 : {color_metrics.get('delta_e00_mean', 0.0):.4f} (Max: {color_metrics.get('delta_e00_max', 0.0):.4f})")
        print(f"  🎨 CIE76 色差 ΔE76     : {color_metrics.get('delta_e76_mean', 0.0):.4f}")
        print(f"  🔴 Red 通道 MAE (0-255): {color_metrics.get('r_mae', 0.0):.4f}")
        print(f"  🟢 Green 通道 MAE      : {color_metrics.get('g_mae', 0.0):.4f}")
        print(f"  🔵 Blue 通道 MAE       : {color_metrics.get('b_mae', 0.0):.4f}")
    print(f"  ⏱️  Encode Latency     : {encode_ms:.2f} ms")
    print(f"  ⏱️  Decode Latency     : {decode_ms:.2f} ms")
    print("=" * 60 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate OpenRSIC Neural Compression & Decompression Pipeline.")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input remote sensing TIF image path.")
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=Path,
        default=Path("checkpoints_fp/best.pt"),
        help="Path to trained model checkpoint (.pt).",
    )
    parser.add_argument(
        "--compressed-output",
        "-o",
        type=Path,
        default=Path("simulated_output.rsic"),
        help="Target compressed bitstream path (.rsic).",
    )
    parser.add_argument(
        "--decompressed-output",
        "-r",
        type=Path,
        default=Path("reconstructed_output.tif"),
        help="Target reconstructed TIF path.",
    )
    parser.add_argument("--decoder-type", choices=["standard", "cheng2020_attention", "swin"], default="swin")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📌 Using Device: {device}")

    # Load Model
    model = get_model(decoder_type=args.decoder_type).to(device)
    if args.checkpoint.exists():
        load_checkpoint(model, args.checkpoint)
        print(f"✅ Loaded checkpoint: {args.checkpoint}")
    else:
        print(f"⚠️ Checkpoint {args.checkpoint} not found. Running with initial weights for demonstration.")

    model.eval()

    # Step 1: Compress
    bpp, ratio, encode_ms = compress_image(model, args.input, args.compressed_output, device)

    # Step 2: Decompress
    decode_ms = decompress_image(model, args.compressed_output, args.decompressed_output, device)

    # Step 3: Evaluate
    evaluate_simulation(
        input_tif=args.input,
        reconstructed_tif=args.decompressed_output,
        bpp=bpp,
        compression_ratio=ratio,
        encode_ms=encode_ms,
        decode_ms=decode_ms,
    )


if __name__ == "__main__":
    main()

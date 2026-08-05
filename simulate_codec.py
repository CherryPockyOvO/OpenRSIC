#!/usr/bin/env python3
"""
===============================================================================
OpenRSIC 遙感圖片壓縮與解壓縮模擬腳本 (Neural Image Compression Simulation)
===============================================================================
基於 OpenRSIC 權重 (best.pt) 的圖片壓縮與解壓縮模擬工具。

主要功能：
  1. [--mode simulate] (預設): 完整模擬「原圖 -> 壓縮寫入 .rsic_bin -> 解壓縮重建圖 -> 計算指標報告」
  2. [--mode compress]: 進行純壓縮，將圖片編碼並儲存為極小體積的二進位碼流 (.rsic_bin)
  3. [--mode decompress]: 進行純解壓縮，將二進位碼流 (.rsic_bin) 解碼重建為高品質圖片
  4. 支持單檔與資料夾批次處理 (PNG, JPG, BMP, WEBP, TIF/TIFF)
  5. 完全支援 TIF / GeoTIFF 遙感圖像: 壓縮時自動記錄數據類型 (uint8, uint16, float32) 與數值範圍，解壓時預設自動還原為 TIF 格式！
  6. 提供 PSNR, SSIM, MSE, BPP (Bits Per Pixel), 壓縮率 (CR), 壓縮/解壓耗時等多維度評估報告

環境要求: lemon Conda 環境
===============================================================================
"""

from __future__ import annotations

import argparse
import io
import math
import os
import struct
import sys
import time
import zlib
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional

# 設定控制台 stdout / stderr 編碼，防止 Windows GBK 亂碼
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 自動搜尋並加入 OpenRSIC 套件路徑
SCRIPT_DIR = Path(__file__).resolve().parent
POSSIBLE_PATHS = [
    SCRIPT_DIR,
    SCRIPT_DIR / "OpenRSIC",
    SCRIPT_DIR.parent / "OpenRSIC",
]
for p in POSSIBLE_PATHS:
    if (p / "rsic").exists():
        sys.path.insert(0, str(p))
        break

try:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from rsic import (
        get_model,
        load_checkpoint,
        pad_to_multiple,
        crop_to_size,
    )
    from rsic.utils import (
        calculate_color_and_reconstruction_metrics,
        load_remote_sensing_image,
        read_remote_sensing_tif,
        tensor_to_remote_sensing_tif,
    )
    from rsic.cnz import pack_symbols, unpack_symbols
except ImportError as e:
    print(f"❌ 匯入模組失敗: {e}")
    print("請確保在 lemon Conda 環境中執行本腳本，或將 OpenRSIC 資料夾置於同目錄下。")
    sys.exit(1)


# -----------------------------------------------------------------------------
# 結構定義與二進位標頭 (Container Specs for .rsic_bin)
# -----------------------------------------------------------------------------
MAGIC = b"RSIC"
VERSION = 2
# Header: Magic(4s), Version(H), orig_h(I), orig_w(I), pad_h(I), pad_w(I), quant_step(f),
#         is_tif(B), dtype_code(B), min_val(f), max_val(f), dz_code(H), dy_code(H), len_cz(I), len_cy(I)
HEADER_FORMAT = "<4sHIIIIfBBffHHII"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

DTYPE_MAP = {
    "uint8": 0,
    "uint16": 1,
    "float32": 2,
}
INV_DTYPE_MAP = {v: k for k, v in DTYPE_MAP.items()}


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """計算 PSNR (Peak Signal-to-Noise Ratio)"""
    mse = float(F.mse_loss(img1, img2).item())
    if mse < 1e-10:
        return 100.0
    return 10.0 * math.log10(1.0 / mse)


def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> float:
    """純 PyTorch 實現的高速 SSIM (Structural Similarity Index)"""
    C1 = (0.01 * 1.0) ** 2
    C2 = (0.03 * 1.0) ** 2
    sigma = 1.5

    gauss = torch.exp(-torch.arange(window_size).sub(window_size // 2).pow(2) / (2 * sigma**2))
    gauss = gauss / gauss.sum()
    kernel = gauss.unsqueeze(1) @ gauss.unsqueeze(0)
    kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(img1.size(1), 1, 1, 1).to(device=img1.device, dtype=img1.dtype)

    mu1 = F.conv2d(img1, kernel, padding=window_size // 2, groups=img1.size(1))
    mu2 = F.conv2d(img2, kernel, padding=window_size // 2, groups=img1.size(1))

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, kernel, padding=window_size // 2, groups=img1.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, kernel, padding=window_size // 2, groups=img1.size(1)) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel, padding=window_size // 2, groups=img1.size(1)) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean().item())


# -----------------------------------------------------------------------------
# Codec 核心類別: CodecSimulator
# -----------------------------------------------------------------------------
class CodecSimulator:
    def __init__(self, checkpoint_path: str | Path, device_name: str = "auto"):
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"找不到模型權重檔: {self.checkpoint_path}")

        # 決定執行設備 (GPU CUDA 或 CPU)
        if device_name == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_name)

        print(f"⚡ [初始化] 使用設備: {self.device.type.upper()} ({torch.cuda.get_device_name(0) if self.device.type == 'cuda' else 'System CPU'})")
        print(f"📦 [載入權重] 正在載入 {self.checkpoint_path.name} ...")

        # 讀取權重檔資訊
        ckpt = torch.load(str(self.checkpoint_path), map_location="cpu")
        model_variant = ckpt.get("model_variant", "rsic") if isinstance(ckpt, dict) else "rsic"
        decoder_type = ckpt.get("decoder_type", "swin") if isinstance(ckpt, dict) else "swin"

        # 實例化 OpenRSIC 模型並載入參數
        self.model = get_model(model_variant=model_variant, decoder_type=decoder_type)
        load_checkpoint(self.model, self.checkpoint_path)
        self.model.to(self.device)
        self.model.eval()

        print(f"✅ [模型準備就緒] Variant: {model_variant} | Decoder: {decoder_type}\n")

    @torch.no_grad()
    def compress_tensor(
        self,
        x: torch.Tensor,
        orig_shape: Tuple[int, int],
        is_tif: bool = False,
        orig_dtype_str: str = "uint8",
        min_val: float = 0.0,
        max_val: float = 255.0,
    ) -> Tuple[bytes, Dict[str, Any]]:
        """將圖片張量轉為壓縮二進位檔 (.rsic_bin) 內容"""
        start_time = time.perf_counter()

        # 1. 填充張量使長寬符合神經網路下採樣倍率 (64倍)
        x_padded, (orig_h, orig_w) = pad_to_multiple(x.to(self.device), 64)
        _, _, pad_h, pad_w = x_padded.shape

        # 2. Encoder 提特徵 ($g_a$) 與 Hyper-Encoder ($h_a$)
        y = self.model.encoder(x_padded)
        z = self.model.hyper_encoder(y)

        # 3. 超潛在變數 Z 量化 ($Q_z$) 與 Hyper-Decoder ($h_s$) 預測 $\mu_y, \sigma_y$
        q_z = self.model.entropy_bottleneck_z.quantize(z)
        z_hat = self.model.entropy_bottleneck_z.dequantize(q_z)
        scales_y, means_y = self.model.hyper_decoder(z_hat)

        # 4. 主潛在變數 Y 條件量化 ($Q_y$)
        q_y = self.model.conditional_entropy_y.quantize(y, means_y)

        # 5. 打包符號與 zlib 無損壓縮
        packed_z = pack_symbols(q_z)
        packed_y = pack_symbols(q_y)

        cz = zlib.compress(packed_z.raw_bytes, level=6)
        cy = zlib.compress(packed_y.raw_bytes, level=6)

        quant_step = self.model.get_quant_step()
        dtype_code = DTYPE_MAP.get(str(orig_dtype_str).lower(), 0)

        header = struct.pack(
            HEADER_FORMAT,
            MAGIC,
            VERSION,
            orig_h,
            orig_w,
            pad_h,
            pad_w,
            quant_step,
            1 if is_tif else 0,
            dtype_code,
            float(min_val),
            float(max_val),
            packed_z.dtype_code,
            packed_y.dtype_code,
            len(cz),
            len(cy),
        )

        binary_payload = header + cz + cy
        enc_time = (time.perf_counter() - start_time) * 1000.0  # ms

        # 計算編碼體積指標
        compressed_bytes = len(binary_payload)
        
        # 動態計算原始位元組大小 (支援 32-bit 與單通道)
        c = x.shape[1]
        bytes_per_pixel = 1
        if "16" in str(orig_dtype_str):
            bytes_per_pixel = 2
        elif "32" in str(orig_dtype_str):
            bytes_per_pixel = 4
        elif "64" in str(orig_dtype_str):
            bytes_per_pixel = 8
            
        uncompressed_raw_bytes = orig_h * orig_w * c * bytes_per_pixel
        
        bpp = (compressed_bytes * 8.0) / (orig_h * orig_w)
        cr = uncompressed_raw_bytes / max(1, compressed_bytes)

        stats = {
            "orig_h": orig_h,
            "orig_w": orig_w,
            "pad_h": pad_h,
            "pad_w": pad_w,
            "compressed_bytes": compressed_bytes,
            "raw_bytes": uncompressed_raw_bytes,
            "bpp": bpp,
            "compression_ratio": cr,
            "enc_time_ms": enc_time,
            "is_tif": is_tif,
            "orig_dtype": orig_dtype_str,
        }
        return binary_payload, stats

    @torch.no_grad()
    def decompress_bytes(self, binary_payload: bytes) -> Tuple[torch.Tensor, Tuple[int, int], Dict[str, Any]]:
        """將二進位檔 (.rsic_bin) 內容解碼還原為圖像張量與原格式元資料"""
        start_time = time.perf_counter()

        if len(binary_payload) < HEADER_SIZE:
            raise ValueError("二進位檔案標頭過短，非有效的 .rsic_bin 壓縮檔")

        # 1. 解析二進位標頭
        (
            magic,
            ver,
            orig_h,
            orig_w,
            pad_h,
            pad_w,
            quant_step,
            is_tif_flag,
            dtype_code,
            min_val,
            max_val,
            dz_code,
            dy_code,
            len_cz,
            len_cy,
        ) = struct.unpack(HEADER_FORMAT, binary_payload[:HEADER_SIZE])

        if magic != MAGIC:
            raise ValueError(f"無效的魔數 (Magic Number): {magic}，預期為 {MAGIC}")

        offset = HEADER_SIZE
        raw_cz = binary_payload[offset : offset + len_cz]
        raw_cy = binary_payload[offset + len_cz : offset + len_cz + len_cy]

        # 2. zlib 解壓縮還原二進位陣列
        bytes_z = zlib.decompress(raw_cz)
        bytes_y = zlib.decompress(raw_cy)

        # 3. 解封符號為 PyTorch int32 張量
        shape_z = (1, self.model.Z, pad_h // 64, pad_w // 64)
        shape_y = (1, self.model.M, pad_h // 16, pad_w // 16)

        q_z = unpack_symbols(bytes_z, dz_code, shape_z).to(self.device)
        q_y = unpack_symbols(bytes_y, dy_code, shape_y).to(self.device)

        # 4. 超解碼器還原特徵與解碼器 ($g_s$) 合成影像
        z_hat = self.model.entropy_bottleneck_z.dequantize(q_z)
        scales_y, means_y = self.model.hyper_decoder(z_hat)
        y_hat = self.model.conditional_entropy_y.dequantize(q_y, means_y, quant_step=quant_step)

        x_hat = self.model.decoder(y_hat)

        # 5. 裁剪回原始圖像尺寸
        x_hat_cropped = crop_to_size(x_hat, (orig_h, orig_w))
        dec_time = (time.perf_counter() - start_time) * 1000.0  # ms

        stats = {
            "orig_h": orig_h,
            "orig_w": orig_w,
            "dec_time_ms": dec_time,
            "is_tif": bool(is_tif_flag),
            "orig_dtype": INV_DTYPE_MAP.get(dtype_code, "uint8"),
            "min_val": min_val,
            "max_val": max_val,
        }
        return x_hat_cropped, (orig_h, orig_w), stats


# -----------------------------------------------------------------------------
# 輔助工具函式
# -----------------------------------------------------------------------------
def create_demo_image(save_path: Path) -> Path:
    """生成一張測試用高頻細節圖像 (512x512)"""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((512, 512, 3), dtype=np.uint8)
    for i in range(512):
        for j in range(512):
            arr[i, j, 0] = (i + j) % 256
            arr[i, j, 1] = (i * 2) % 256
            arr[i, j, 2] = (j * 3) % 256
    arr[100:400:10, :, :] = 255
    arr[:, 100:400:10, :] = 255
    img = Image.fromarray(arr)
    img.save(save_path)
    print(f"🎨 [自動生成測試圖]: {save_path.resolve()}")
    return save_path


def print_simulation_report(
    img_name: str,
    orig_size_kb: float,
    comp_size_kb: float,
    bpp: float,
    cr: float,
    psnr: float,
    ssim: float,
    mse: float,
    enc_ms: float,
    dec_ms: float,
    metrics: dict[str, float] | None = None,
    is_tif: bool = False,
    dtype_str: str = "uint8",
):
    """印出控制台評估報告表單"""
    orig_channels = int(metrics.get("orig_channels", 3)) if metrics else (1 if is_tif else 3)
    print("\n" + "=" * 70)
    print(f"📊 【OpenRSIC 圖片壓縮 / 解壓縮模擬測試報告】: {img_name}")
    print("=" * 70)
    print(f"  • 圖像類型 / 原位深度:             {'GeoTIFF / ' + dtype_str if is_tif else 'Standard Image (RGB)'}")
    print(f"  • 圖像通道數 (Channels):            {orig_channels} 通道 ({'灰階/單波段' if orig_channels == 1 else '彩色 RGB'})")
    print(f"  • 原始影像大小 (Uncompressed):       {orig_size_kb:>10.2f} KB")
    print(f"  • 壓縮檔大小 (.rsic_bin):            {comp_size_kb:>10.2f} KB")
    print(f"  • 壓縮倍率 (Compression Ratio):     {cr:>10.2f} x  (節省 {(1.0 - 1.0/cr)*100:.1f}%)")
    print(f"  • 碼率 (Bits Per Pixel):            {bpp:>10.4f} bpp")
    print("-" * 70)
    print(f"  • 重建圖像畫質 PSNR:                 {psnr:>10.2f} dB")
    print(f"  • 重建圖像畫質 SSIM:                 {ssim:>10.4f}")
    print(f"  • 均方誤差 MSE:                     {mse:>10.6f}")

    if metrics:
        if orig_channels == 1:
            print("-" * 70)
            print("  🎨 【單通道 / 原位物理數值還原誤差指標】:")
            print(f"  • 平均絕對誤差 MAE (Native Scale):  {metrics.get('native_mae', 0.0):>10.6f}")
            print(f"  • 均方根誤差 RMSE (Native Scale):  {metrics.get('native_rmse', 0.0):>10.6f}")
            print(f"  • 最大單點峰值誤差 MaxAE:           {metrics.get('max_abs_error', 0.0):>10.6f}")
            print(f"  • 平均相對誤差 MAPE:               {metrics.get('relative_error_pct', 0.0):>10.4f} %")
            print(f"  • 信噪比 (Signal-to-Noise Ratio):   {metrics.get('snr_db', 0.0):>10.2f} dB")
        else:
            print("-" * 70)
            print("  🎨 【RGB 彩色圖像色差與通道誤差指標】:")
            print(f"  • CIEDE2000 色差 ΔE00 (Mean):       {metrics.get('delta_e00_mean', 0.0):>10.4f}  (視覺不可察覺 < 1.0)")
            print(f"  • CIEDE2000 色差 ΔE00 (Max):        {metrics.get('delta_e00_max', 0.0):>10.4f}")
            print(f"  • CIE76 色差 ΔE76 (Mean):           {metrics.get('delta_e76_mean', 0.0):>10.4f}")
            print(f"  • Red (紅) 通道 MAE (0-255):         {metrics.get('r_mae', 0.0):>10.4f}")
            print(f"  • Green (綠) 通道 MAE (0-255):       {metrics.get('g_mae', 0.0):>10.4f}")
            print(f"  • Blue (藍) 通道 MAE (0-255):        {metrics.get('b_mae', 0.0):>10.4f}")

    print("-" * 70)
    print(f"  • 壓縮 (Encode) 耗時:               {enc_ms:>10.2f} ms")
    print(f"  • 解壓 (Decode) 耗時:               {dec_ms:>10.2f} ms")
    print(f"  • 總端到端耗時 (Total Time):          {enc_ms + dec_ms:>10.2f} ms")
    print("=" * 70 + "\n")


# -----------------------------------------------------------------------------
# 主執行進入點 (CLI Logic)
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="OpenRSIC 圖片壓縮與解壓縮模擬工具 (使用 best.pt)")
    parser.add_argument("--mode", choices=["simulate", "compress", "decompress"], default="simulate", help="執行模式: simulate (端到端測試報告), compress (純壓縮), decompress (純解壓)")
    parser.add_argument("-i", "--input", type=str, default=None, help="輸入圖片檔名 / 二進位壓縮檔 / 或資料夾路徑")
    parser.add_argument("-o", "--output", type=str, default=None, help="輸出路徑 (圖片檔或 .rsic_bin)")
    parser.add_argument("-c", "--checkpoint", type=str, default=None, help="模型權重檔路徑 (預設自動尋找 best.pt)")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="運算設備選擇 (預設 CUDA 自動優先)")

    args = parser.parse_args()

    # 自動定位 best.pt 權重檔
    checkpoint_path = None
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        candidates = [
            SCRIPT_DIR / "best.pt",
            SCRIPT_DIR.parent / "best.pt",
            SCRIPT_DIR / "OpenRSIC" / "best.pt",
            SCRIPT_DIR / "checkpoints_rsic" / "best.pt",
        ]
        for cand in candidates:
            if cand.exists():
                checkpoint_path = cand
                break

    if not checkpoint_path or not checkpoint_path.exists():
        print("❌ 找不到 best.pt 權重檔！請使用 -c 指定權重檔位置。")
        sys.exit(1)

    print(f"🚀 [啟動工具] Mode: {args.mode.upper()} | Checkpoint: {checkpoint_path.resolve()}")
    simulator = CodecSimulator(checkpoint_path, device_name=args.device)

    # 1. 處理模式：端到端模擬測試 (SIMULATE)
    if args.mode == "simulate":
        input_path = Path(args.input) if args.input else None
        if not input_path or not input_path.exists():
            input_path = create_demo_image(SCRIPT_DIR / "demo_input.png")

        files_to_process = [input_path] if input_path.is_file() else list(input_path.glob("*.*"))
        valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
        files_to_process = [f for f in files_to_process if f.suffix.lower() in valid_exts]

        if not files_to_process:
            print(f"❌ 找不到有效的圖片檔案於: {input_path}")
            return

        out_dir = Path(args.output) if args.output else SCRIPT_DIR / "output_simulation"
        out_dir.mkdir(parents=True, exist_ok=True)

        for img_file in files_to_process:
            print(f"\n📸 [處理圖片]: {img_file.name}")
            is_tif = img_file.suffix.lower() in {".tif", ".tiff"}
            min_val, max_val = 0.0, 255.0
            orig_dtype_str = "uint8"
            orig_channels = 3

            if is_tif:
                x_tensor, orig_dtype, min_val, max_val, orig_channels = read_remote_sensing_tif(img_file)
                x_tensor = x_tensor.unsqueeze(0)
                orig_dtype_str = str(orig_dtype)
            else:
                pil_img = load_remote_sensing_image(img_file)
                arr = np.asarray(pil_img).astype(np.float32) / 255.0
                x_tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

            orig_h, orig_w = x_tensor.shape[2], x_tensor.shape[3]

            # A. 執行壓縮 (Compress)
            bin_data, comp_stats = simulator.compress_tensor(
                x_tensor,
                (orig_h, orig_w),
                is_tif=is_tif,
                orig_dtype_str=orig_dtype_str,
                min_val=min_val,
                max_val=max_val,
            )
            bin_save_path = out_dir / f"{img_file.stem}.rsic_bin"
            bin_save_path.write_bytes(bin_data)

            # B. 執行解壓 (Decompress)
            x_rec, _, dec_stats = simulator.decompress_bytes(bin_data)

            # C. 儲存重建圖 (如果原圖是 TIF，解壓就自動輸出為 TIF！)
            if is_tif:
                rec_save_path = out_dir / f"{img_file.stem}_recon.tif"
                tensor_to_remote_sensing_tif(x_rec.squeeze(0), orig_dtype, min_val, max_val, rec_save_path, orig_channels=orig_channels)
            else:
                rec_save_path = out_dir / f"{img_file.stem}_recon.png"
                arr_rec = (x_rec.squeeze(0).detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
                Image.fromarray(arr_rec).save(rec_save_path)

            # D. 計算指標 (PSNR, SSIM, MSE, 色差 metrics)
            x_orig_cuda = x_tensor.to(simulator.device)
            psnr_val = calculate_psnr(x_orig_cuda, x_rec)
            ssim_val = calculate_ssim(x_orig_cuda, x_rec)
            mse_val = float(F.mse_loss(x_orig_cuda, x_rec).item())

            color_metrics = calculate_color_and_reconstruction_metrics(
                img_orig=x_tensor,
                img_recon=x_rec,
                is_tif=is_tif,
                orig_dtype=orig_dtype_str,
                min_val=min_val,
                max_val=max_val,
                orig_channels=orig_channels,
            )

            orig_size_kb = comp_stats["raw_bytes"] / 1024.0
            comp_size_kb = len(bin_data) / 1024.0

            print_simulation_report(
                img_name=img_file.name,
                orig_size_kb=orig_size_kb,
                comp_size_kb=comp_size_kb,
                bpp=comp_stats["bpp"],
                cr=comp_stats["compression_ratio"],
                psnr=psnr_val,
                ssim=ssim_val,
                mse=mse_val,
                enc_ms=comp_stats["enc_time_ms"],
                dec_ms=dec_stats["dec_time_ms"],
                metrics=color_metrics,
                is_tif=is_tif,
                dtype_str=orig_dtype_str,
            )
            print(f"💾 [壓縮碼流已存檔]: {bin_save_path.resolve()}")
            print(f"🖼️ [重建影像已存檔]: {rec_save_path.resolve()}")

    # 2. 處理模式：單純壓縮 (COMPRESS)
    elif args.mode == "compress":
        if not args.input:
            print("❌ [--mode compress] 需要指定輸入圖片 (-i path/to/image.tif)")
            return
        input_path = Path(args.input)
        out_path = Path(args.output) if args.output else input_path.with_suffix(".rsic_bin")

        is_tif = input_path.suffix.lower() in {".tif", ".tiff"}
        min_val, max_val = 0.0, 255.0
        orig_dtype_str = "uint8"

        if is_tif:
            x_tensor, orig_dtype, min_val, max_val, orig_channels = read_remote_sensing_tif(input_path)
            x_tensor = x_tensor.unsqueeze(0)
            orig_dtype_str = str(orig_dtype)
        else:
            pil_img = load_remote_sensing_image(input_path)
            arr = np.asarray(pil_img).astype(np.float32) / 255.0
            x_tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

        bin_data, comp_stats = simulator.compress_tensor(
            x_tensor,
            (x_tensor.shape[2], x_tensor.shape[3]),
            is_tif=is_tif,
            orig_dtype_str=orig_dtype_str,
            min_val=min_val,
            max_val=max_val,
        )
        out_path.write_bytes(bin_data)

        print(f"✅ [壓縮完成] 輸入圖片: {input_path.name}")
        print(f"  • 格式類型: {'GeoTIFF (' + orig_dtype_str + ')' if is_tif else 'Standard Image'}")
        print(f"  • 壓縮檔路徑: {out_path.resolve()}")
        print(f"  • 壓縮後體積: {len(bin_data)/1024:.2f} KB | BPP: {comp_stats['bpp']:.4f} | 壓縮倍率: {comp_stats['compression_ratio']:.2f}x")
        print(f"  • 耗時: {comp_stats['enc_time_ms']:.2f} ms")

    # 3. 處理模式：單純解壓 (DECOMPRESS)
    elif args.mode == "decompress":
        if not args.input:
            print("❌ [--mode decompress] 需要指定輸入二進位檔 (-i path/to/file.rsic_bin)")
            return
        input_path = Path(args.input)

        bin_data = input_path.read_bytes()
        x_rec, (h, w), dec_stats = simulator.decompress_bytes(bin_data)

        is_tif = dec_stats["is_tif"]
        orig_dtype = dec_stats["orig_dtype"]
        min_val = dec_stats["min_val"]
        max_val = dec_stats["max_val"]

        # 自動決定輸出副檔名 (若是 TIF 壓縮來的，預設就還原為 .tif)
        if args.output:
            out_path = Path(args.output)
        else:
            default_ext = ".tif" if is_tif else ".png"
            out_path = input_path.with_name(f"{input_path.stem}_decompressed{default_ext}")

        if is_tif or out_path.suffix.lower() in {".tif", ".tiff"}:
            tensor_to_remote_sensing_tif(x_rec.squeeze(0), orig_dtype, min_val, max_val, out_path)
        else:
            arr_rec = (x_rec.squeeze(0).detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
            Image.fromarray(arr_rec).save(out_path)

        print(f"✅ [解壓完成] 輸入碼流: {input_path.name}")
        print(f"  • 還原影像路徑: {out_path.resolve()}")
        print(f"  • 還原格式: {'GeoTIFF (' + str(orig_dtype) + ')' if is_tif else 'Standard Image (PNG)'}")
        print(f"  • 影像尺寸: {w} x {h}")
        print(f"  • 耗時: {dec_stats['dec_time_ms']:.2f} ms")


if __name__ == "__main__":
    main()

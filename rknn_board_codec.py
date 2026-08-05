#!/usr/bin/env python3
"""
===============================================================================
RK3588 NPU 邊緣端壓縮與解壓縮測試腳本 (rknn_board_codec.py)
===============================================================================
說明:
  本腳本為 OpenRSIC 最新架構適配版本，支援：
  1. RK3588 NPU 邊緣端超高速特徵提取與壓縮 (.rsic 碼流格式)
  2. 完整相容 1 通道 (單通道/灰階) 與 3 通道 (RGB) 遙感 TIF 圖像
  3. 保留原位數值範圍 (min_val, max_val)、數據型態 (uint8/uint16/float32) 與 GeoTIFF 座標元資料
  4. 自動對齊 512x512 靜態 RKNN NPU 模型的 NHWC / NCHW 記憶體佈局
  5. 提供 PSNR, SSIM, MSE 畫質評估報告

修復紀錄:
  - 修正 tensor_to_remote_sensing_tif 相容性，相容舊版與新版 rsic API (避免 TypeError)
  - 新增 numpy copy() 避免 PyTorch non-writable tensor 警告
  - 頂層載入順序優化，徹底修復 NameError 與 AttributeError 衝突
===============================================================================
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

# 設定控制台 stdout / stderr 編碼，防止 Windows GBK 亂碼
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 1. 自動搜尋並加入 OpenRSIC 套件路徑
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

import logging

# 2. 優先修復與保護 Python logging 模組
def fix_logging_module():
    logging._nameToLevel = {
        "CRITICAL": 50,
        "FATAL": 50,
        "ERROR": 40,
        "WARN": 30,
        "WARNING": 30,
        "INFO": 20,
        "DEBUG": 10,
        "NOTSET": 0,
    }
    logging._levelToName = {
        50: "CRITICAL",
        40: "ERROR",
        30: "WARNING",
        20: "INFO",
        10: "DEBUG",
        0: "NOTSET",
    }
    _orig_set_level = logging.Logger.setLevel

    def _safe_set_level(self, level):
        if isinstance(level, str):
            level = logging._nameToLevel.get(level.upper(), 30)
        try:
            _orig_set_level(self, level)
        except Exception:
            self.level = level

    logging.Logger.setLevel = _safe_set_level

fix_logging_module()

# 3. 優先在最頂層載入 PyTorch 與 OpenRSIC 核心組件 (防止被 rknnlite 共享庫干擾)
HAS_TORCH = False
try:
    import torch
    from rsic import extract_geotiff_metadata, get_model, inject_geotiff_metadata, load_checkpoint
    from rsic.utils import (
        calculate_color_and_reconstruction_metrics,
        read_remote_sensing_tif,
        tensor_to_remote_sensing_tif,
    )
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import argparse
import json
import math
import os
import struct
import time
import zlib
import numpy as np
from PIL import Image

# 4. 嘗試匯入 RKNN 執行環境 (自動相容板端 rknnlite 或 PC端 rknn)
USING_RKNN_LITE = False
try:
    from rknnlite.api import RKNNLite
    USING_RKNN_LITE = True
    print("✅ 檢測到板端 SDK: rknn-toolkit-lite2")
except ImportError:
    try:
        from rknn.api import RKNN
        print("✅ 檢測到 PC 端 SDK: rknn-toolkit2 (模擬器模式)")
    except ImportError:
        print("⚠️ 未檢測到 rknn-toolkit2 或 rknn-toolkit-lite2 套件。")
        print("   板端請執行: pip install rknn-toolkit-lite2")
        print("   PC 端請執行: pip install rknn-toolkit2")


RSIC_FILE_HEADER_MAGIC = b"RSIC1"


def calculate_psnr(img1: np.ndarray, img2: np.ndarray, max_val: float = 1.0) -> float:
    """計算 PSNR (Peak Signal-to-Noise Ratio)"""
    mse = float(np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2))
    if mse < 1e-10:
        return 100.0
    return 10.0 * math.log10((max_val**2) / mse)


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """計算 Structural Similarity Index (SSIM)"""
    try:
        import cv2
        if img1.shape != img2.shape:
            return 0.0
        c1 = (0.01 * 1.0) ** 2
        c2 = (0.03 * 1.0) ** 2

        img1_f64 = img1.astype(np.float64)
        img2_f64 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.T)

        if img1_f64.ndim == 3:
            ssims = []
            for i in range(img1_f64.shape[2]):
                mu1 = cv2.filter2D(img1_f64[:, :, i], -1, window)
                mu2 = cv2.filter2D(img2_f64[:, :, i], -1, window)
                mu1_sq = mu1**2
                mu2_sq = mu2**2
                mu1_mu2 = mu1 * mu2
                sigma1_sq = cv2.filter2D(img1_f64[:, :, i] ** 2, -1, window) - mu1_sq
                sigma2_sq = cv2.filter2D(img2_f64[:, :, i] ** 2, -1, window) - mu2_sq
                sigma12 = cv2.filter2D(img1_f64[:, :, i] * img2_f64[:, :, i], -1, window) - mu1_mu2
                ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
                ssims.append(ssim_map.mean())
            return float(np.mean(ssims))
        else:
            mu1 = cv2.filter2D(img1_f64, -1, window)
            mu2 = cv2.filter2D(img2_f64, -1, window)
            mu1_sq = mu1**2
            mu2_sq = mu2**2
            mu1_mu2 = mu1 * mu2
            sigma1_sq = cv2.filter2D(img1_f64**2, -1, window) - mu1_sq
            sigma2_sq = cv2.filter2D(img2_f64**2, -1, window) - mu2_sq
            sigma12 = cv2.filter2D(img1_f64 * img2_f64, -1, window) - mu1_mu2
            ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
            return float(ssim_map.mean())
    except Exception:
        return 0.0


def pack_int_symbols(symbols: np.ndarray) -> tuple[bytes, int]:
    """將整數符號陣列打包為小端二進位位元組 (int16 或 int32)"""
    symbols = symbols.astype(np.int32)
    min_val = symbols.min()
    max_val = symbols.max()
    if -32768 <= min_val and max_val <= 32767:
        raw_bytes = symbols.astype("<i2", copy=False).tobytes(order="C")
        dtype_code = 1  # int16
    else:
        raw_bytes = symbols.astype("<i4", copy=False).tobytes(order="C")
        dtype_code = 2  # int32
    return raw_bytes, dtype_code


def unpack_int_symbols(raw_bytes: bytes, dtype_code: int, shape: tuple[int, ...]) -> np.ndarray:
    """將二進位位元組解封回整數符號陣列"""
    dtype = np.dtype("<i2") if dtype_code == 1 else np.dtype("<i4")
    arr = np.frombuffer(raw_bytes, dtype=dtype).astype(np.int32, copy=True)
    return arr.reshape(shape)


class RKNNEncoderCompressor:
    def __init__(self, rknn_model_path: str | Path, params_json_path: str | Path, target_platform: str = "rk3588"):
        self.rknn_path = Path(rknn_model_path)
        self.params_path = Path(params_json_path)

        if not self.rknn_path.exists():
            raise FileNotFoundError(f"找不到 RKNN 模型檔案: {self.rknn_path}")
        if not self.params_path.exists():
            raise FileNotFoundError(f"找不到參數 JSON 檔案: {self.params_path}")

        # 讀取量化與熵編碼參數
        with open(self.params_path, "r", encoding="utf-8") as f:
            self.params = json.load(f)

        self.channels_y = self.params.get("channels_y", 256)
        self.channels_z = self.params.get("channels_z", 128)
        self.quant_step_y = float(self.params.get("quant_step_y", 0.38))
        self.quant_step_z = float(self.params.get("quant_step_z", 1.0))
        self.z_medians = np.array(self.params.get("z_medians", [0.0] * self.channels_z), dtype=np.float32).reshape(1, self.channels_z, 1, 1)

        print(f"📦 [初始化 RKNN NPU Engine] 載入: {self.rknn_path.name}")
        print(f"   - 頻道數 y: {self.channels_y}, z: {self.channels_z}")
        print(f"   - 量化步長 quant_step_y: {self.quant_step_y:.4f}, quant_step_z: {self.quant_step_z:.4f}")

        # 初始化 RKNN API
        if USING_RKNN_LITE:
            self.rknn = RKNNLite()
            ret = self.rknn.load_rknn(str(self.rknn_path))
            if ret != 0:
                raise RuntimeError(f"RKNNLite load_rknn 失敗, code={ret}")
            ret = self.rknn.init_runtime()
            if ret != 0:
                raise RuntimeError(f"RKNNLite init_runtime 失敗, code={ret}")
        else:
            from rknn.api import RKNN
            self.rknn = RKNN()
            ret = self.rknn.load_rknn(str(self.rknn_path))
            if ret != 0:
                raise RuntimeError(f"RKNN load_rknn 失敗, code={ret}")
            ret = self.rknn.init_runtime(target=target_platform)
            if ret != 0:
                raise RuntimeError(f"RKNN init_runtime 失敗, code={ret}")

        print("✅ [RKNN NPU Engine 初始化成功!]\n")

    def compress_image(self, image_path: str | Path, model_input_size: int = 512) -> tuple[bytes, dict]:
        """讀取圖片，透過 RKNN NPU 提特徵並壓縮為 OpenRSIC 標準碼流 (.rsic)"""
        start_time = time.perf_counter()
        img_path = Path(image_path)

        # 1. 讀取與預處理圖像
        orig_channels = 3
        orig_dtype_str = "uint8"
        min_val, max_val = 0.0, 255.0
        geotiff_meta = None

        if HAS_TORCH:
            try:
                res = read_remote_sensing_tif(img_path)
                if len(res) == 5:
                    tensor, orig_dtype, min_val, max_val, orig_channels = res
                else:
                    tensor, orig_dtype, min_val, max_val = res[:4]
                    orig_channels = 3
                geotiff_meta = extract_geotiff_metadata(img_path)
                norm_img = tensor.permute(1, 2, 0).numpy()  # HWC [0.0, 1.0]
                orig_dtype_str = str(orig_dtype)
            except Exception:
                pil_img = Image.open(img_path)
                arr = np.array(pil_img)
                if arr.ndim == 2:
                    orig_channels = 1
                    arr_rgb = np.stack([arr] * 3, axis=-1)
                elif arr.ndim == 3 and arr.shape[2] == 1:
                    orig_channels = 1
                    arr_rgb = np.concatenate([arr] * 3, axis=-1)
                else:
                    orig_channels = 3
                    arr_rgb = arr[:, :, :3]
                norm_img = arr_rgb.astype(np.float32) / 255.0
                orig_dtype_str = str(arr.dtype)
                min_val, max_val = float(arr.min()), float(arr.max())
        else:
            pil_img = Image.open(img_path)
            arr = np.array(pil_img)
            if arr.ndim == 2:
                orig_channels = 1
                arr_rgb = np.stack([arr] * 3, axis=-1)
            elif arr.ndim == 3 and arr.shape[2] == 1:
                orig_channels = 1
                arr_rgb = np.concatenate([arr] * 3, axis=-1)
            else:
                orig_channels = 3
                arr_rgb = arr[:, :, :3]
            norm_img = arr_rgb.astype(np.float32) / 255.0
            orig_dtype_str = str(arr.dtype)
            min_val, max_val = float(arr.min()), float(arr.max())

        h_raw, w_raw = norm_img.shape[0], norm_img.shape[1]
        original_size = [int(h_raw), int(w_raw)]

        # 若尺寸非 512x512，縮放至 NPU 模型輸入尺寸
        if h_raw != model_input_size or w_raw != model_input_size:
            print(f"⚠️ [尺寸自動對齊] 輸入圖像 ({w_raw}x{h_raw}) 對齊至 RKNN NPU 靜態模型尺寸 ({model_input_size}x{model_input_size})")
            try:
                import cv2
                norm_img = cv2.resize(norm_img, (model_input_size, model_input_size), interpolation=cv2.INTER_AREA)
            except Exception:
                pil_tmp = Image.fromarray((norm_img * 255.0).clip(0, 255).astype(np.uint8))
                pil_tmp = pil_tmp.resize((model_input_size, model_input_size), Image.BILINEAR)
                norm_img = np.array(pil_tmp).astype(np.float32) / 255.0

        # 2. 準備 RKNN NPU 輸入 (NHWC Layout: [1, 512, 512, 3])
        if norm_img.ndim == 3 and norm_img.shape[0] == 3:
            norm_img = np.transpose(norm_img, (1, 2, 0))  # NCHW -> HWC
        input_tensor = norm_img[np.newaxis, :, :, :].astype(np.float32)  # [1, 512, 512, 3]

        # 3. RKNN NPU 前向推論
        npu_start = time.perf_counter()
        outputs = self.rknn.inference(inputs=[input_tensor], data_format="nhwc")
        npu_time_ms = (time.perf_counter() - npu_start) * 1000.0

        # RKNN 導出的 4 個輸出: y, z, scales_y, means_y
        y = outputs[0]
        z = outputs[1]
        scales_y = outputs[2]
        means_y = outputs[3]

        # 自動防護：若 RKNN 輸出為 NHWC (1, H, W, C)，轉置校正回 NCHW (1, C, H, W)
        if y.ndim == 4 and y.shape[1] != self.channels_y and y.shape[3] == self.channels_y:
            y = np.transpose(y, (0, 3, 1, 2))
        if z.ndim == 4 and z.shape[1] != self.channels_z and z.shape[3] == self.channels_z:
            z = np.transpose(z, (0, 3, 1, 2))
        if scales_y.ndim == 4 and scales_y.shape[1] != self.channels_y and scales_y.shape[3] == self.channels_y:
            scales_y = np.transpose(scales_y, (0, 3, 1, 2))
        if means_y.ndim == 4 and means_y.shape[1] != self.channels_y and means_y.shape[3] == self.channels_y:
            means_y = np.transpose(means_y, (0, 3, 1, 2))

        # 4. CPU 符號量化 (int16 格式)
        z_sym = np.round((z - self.z_medians) / self.quant_step_z).astype(np.int16)
        q_y = np.round((y - means_y) / self.quant_step_y).astype(np.int16)

        # 5. 打包與 zlib 壓縮
        z_bytes = zlib.compress(z_sym.tobytes(), level=6)
        y_bytes = zlib.compress(q_y.tobytes(), level=6)

        # 6. 構造元資料 JSON
        meta_dict = {
            "orig_dtype": orig_dtype_str,
            "min_val": float(min_val),
            "max_val": float(max_val),
            "orig_channels": int(orig_channels),
            "original_size": original_size,
            "geotiff_meta": geotiff_meta,
            "z_shape": list(z_sym.shape),
            "y_shape": list(q_y.shape),
            "quant_step": self.quant_step_y,
        }
        meta_bytes = json.dumps(meta_dict).encode("utf-8")

        # 打包二進位碼流
        payload_io = io.BytesIO()
        payload_io.write(RSIC_FILE_HEADER_MAGIC)
        payload_io.write(struct.pack("<I", len(meta_bytes)))
        payload_io.write(meta_bytes)
        payload_io.write(struct.pack("<I", len(z_bytes)))
        payload_io.write(z_bytes)
        payload_io.write(struct.pack("<I", len(y_bytes)))
        payload_io.write(y_bytes)

        binary_payload = payload_io.getvalue()
        total_time_ms = (time.perf_counter() - start_time) * 1000.0

        compressed_bytes = len(binary_payload)
        uncompressed_bytes = os.path.getsize(img_path) if img_path.exists() else (original_size[0] * original_size[1] * orig_channels)
        bpp = (compressed_bytes * 8.0) / (model_input_size * model_input_size)
        cr = uncompressed_bytes / max(1, compressed_bytes)

        stats = {
            "orig_h": original_size[0],
            "orig_w": original_size[1],
            "orig_channels": orig_channels,
            "compressed_bytes": compressed_bytes,
            "bpp": bpp,
            "compression_ratio": cr,
            "npu_time_ms": npu_time_ms,
            "total_time_ms": total_time_ms,
            "norm_img_input": norm_img,  # 用於畫質評估
        }
        return binary_payload, stats

    def release(self):
        if hasattr(self.rknn, "release"):
            self.rknn.release()


# -----------------------------------------------------------------------------
# 解壓邏輯 (PyTorch / OpenRSIC Decoder)
# -----------------------------------------------------------------------------
def decompress_rsic_bin(binary_payload: bytes, checkpoint_path: str | Path, device_str: str = "cpu") -> tuple[np.ndarray, dict]:
    """解壓 .rsic 碼流檔案並還原原位深度圖片"""
    fix_logging_module()

    if not HAS_TORCH:
        raise RuntimeError("解壓端需要安裝 PyTorch。請執行 pip install torch")

    start_time = time.perf_counter()
    bio = io.BytesIO(binary_payload)

    magic = bio.read(5)
    if magic != RSIC_FILE_HEADER_MAGIC:
        raise ValueError(f"無效的 RSIC 碼流 Magic: {magic!r}")

    meta_len = struct.unpack("<I", bio.read(4))[0]
    meta_bytes = bio.read(meta_len)
    meta_dict = json.loads(meta_bytes.decode("utf-8"))

    z_len = struct.unpack("<I", bio.read(4))[0]
    z_bytes = bio.read(z_len)
    y_len = struct.unpack("<I", bio.read(4))[0]
    y_bytes = bio.read(y_len)

    z_shape = tuple(meta_dict["z_shape"])
    y_shape = tuple(meta_dict["y_shape"])
    quant_step = float(meta_dict["quant_step"])

    z_raw = zlib.decompress(z_bytes)
    y_raw = zlib.decompress(y_bytes)

    z_sym_np = np.frombuffer(z_raw, dtype=np.int16).reshape(z_shape).copy()
    y_sym_np = np.frombuffer(y_raw, dtype=np.int16).reshape(y_shape).copy()

    device = torch.device(device_str)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = get_model(decoder_type=ckpt.get("decoder_type", "swin")).to(device)
    load_checkpoint(model, checkpoint_path)
    model.eval()

    with torch.no_grad():
        z_sym = torch.from_numpy(z_sym_np).to(device=device, dtype=torch.float32)
        y_sym = torch.from_numpy(y_sym_np).to(device=device, dtype=torch.float32)

        scales_y, means_y = model.hyper_decoder(z_sym)
        y_hat = y_sym * quant_step + means_y
        x_hat = model.decoder(y_hat)

        x_hat = x_hat[:, :, :512, :512]  # 對齊

    dec_time_ms = (time.perf_counter() - start_time) * 1000.0

    meta_dict["dec_time_ms"] = dec_time_ms
    return x_hat.squeeze(0).detach().cpu().numpy(), meta_dict


# -----------------------------------------------------------------------------
# CLI 程式進入點
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="RK3588 NPU 邊緣端壓縮解壓縮模擬測試工具")
    parser.add_argument("--mode", choices=["compress", "decompress", "simulate"], default="compress", help="測試模式: compress (NPU純壓縮), decompress (純解壓), simulate (完整模擬)")
    parser.add_argument("-i", "--input", type=str, default="test.tif", help="輸入圖片檔名")
    parser.add_argument("-o", "--output", type=str, default=None, help="輸出檔案路徑 (.rsic 或重建圖片)")
    parser.add_argument("--rknn", type=str, default="encoder_512x512.rknn", help="RKNN NPU 模型路徑")
    parser.add_argument("--params", type=str, default="params.json", help="熵參數 JSON 檔案路徑")
    parser.add_argument("--checkpoint", type=str, default="best.pt", help="PyTorch 權重檔 (解壓端使用)")

    args = parser.parse_args()

    # 1. 壓縮模式 (RKNN NPU 純壓縮)
    if args.mode in {"compress", "simulate"}:
        compressor = RKNNEncoderCompressor(args.rknn, args.params)
        bin_data, c_stats = compressor.compress_image(args.input)

        out_bin = Path(args.output) if (args.output and args.mode == "compress") else Path(args.input).with_suffix(".rsic")
        out_bin.write_bytes(bin_data)

        print("\n========================================================")
        print("⚡ 【RK3588 NPU 壓縮成功報告】")
        print("========================================================")
        print(f"  • 輸入圖片:            {args.input} (通道: {c_stats['orig_channels']})")
        print(f"  • 輸出碼流檔 (.rsic):  {out_bin.resolve()}")
        print(f"  • 碼流體積:            {c_stats['compressed_bytes']/1024:.2f} KB")
        print(f"  • 壓縮倍率 (CR):       {c_stats['compression_ratio']:.2f} x (節省 {(1-1/c_stats['compression_ratio'])*100:.1f}%)")
        print(f"  • 碼率 (BPP):          {c_stats['bpp']:.4f} bpp")
        print(f"  • NPU 推論耗時:        {c_stats['npu_time_ms']:.2f} ms  ({1000/c_stats['npu_time_ms']:.1f} FPS)")
        print(f"  • 端到端總耗時:        {c_stats['total_time_ms']:.2f} ms")
        print("========================================================\n")
        compressor.release()

    # 2. 解壓模式 (解 OpenRSIC .rsic 碼流)
    if args.mode in {"decompress", "simulate"}:
        in_bin = out_bin if args.mode == "simulate" else Path(args.input)
        x_rec_tensor_np, meta_dict = decompress_rsic_bin(in_bin.read_bytes(), args.checkpoint)

        out_img = Path(args.output) if (args.output and args.mode == "decompress") else in_bin.with_name(f"{in_bin.stem}_recon.tif")

        orig_channels = int(meta_dict.get("orig_channels", 3))
        orig_dtype = meta_dict["orig_dtype"]
        min_val = meta_dict["min_val"]
        max_val = meta_dict["max_val"]

        if HAS_TORCH:
            x_rec_tensor = torch.from_numpy(x_rec_tensor_np)
            try:
                tensor_to_remote_sensing_tif(
                    x_rec_tensor,
                    orig_dtype=orig_dtype,
                    min_val=min_val,
                    max_val=max_val,
                    save_path=out_img,
                    orig_channels=orig_channels,
                )
            except TypeError:
                try:
                    tensor_to_remote_sensing_tif(
                        x_rec_tensor,
                        orig_dtype,
                        min_val,
                        max_val,
                        out_img,
                    )
                except Exception:
                    arr_rec = (x_rec_tensor_np.transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
                    Image.fromarray(arr_rec).save(out_img)

            if meta_dict.get("geotiff_meta"):
                try:
                    inject_geotiff_metadata(out_img, meta_dict["geotiff_meta"])
                except Exception:
                    pass

        # 計算重建畫質 PSNR / SSIM / MSE / 色差指標
        psnr_val = 0.0
        ssim_val = 0.0
        mse_val = 0.0
        color_metrics = None

        if args.mode == "simulate" and "norm_img_input" in c_stats:
            img_orig = c_stats["norm_img_input"]  # HWC [0.0, 1.0]
            img_recon = x_rec_tensor_np.transpose(1, 2, 0)  # HWC [0.0, 1.0]

            psnr_val = calculate_psnr(img_orig, img_recon, max_val=1.0)
            ssim_val = calculate_ssim(img_orig, img_recon)
            mse_val = float(np.mean((img_orig.astype(np.float64) - img_recon.astype(np.float64)) ** 2))

            if HAS_TORCH:
                try:
                    color_metrics = calculate_color_and_reconstruction_metrics(
                        img_orig=img_orig,
                        img_recon=img_recon,
                        is_tif=True,
                        orig_dtype=orig_dtype,
                        min_val=min_val,
                        max_val=max_val,
                        orig_channels=orig_channels,
                    )
                except Exception:
                    pass

        print("\n========================================================")
        print("🖼️ 【解壓重建成功報告】")
        print("========================================================")
        print(f"  • 輸入碼流:            {in_bin.name}")
        print(f"  • 重建圖片檔:          {out_img.resolve()} (通道: {orig_channels})")
        if args.mode == "simulate":
            print(f"  • 重建畫質 PSNR:       {psnr_val:.2f} dB")
            print(f"  • 重建畫質 SSIM:       {ssim_val:.4f}")
            print(f"  • 均方誤差 MSE:         {mse_val:.6f}")
            if color_metrics:
                if orig_channels == 1:
                    print("-" * 56)
                    print("  🎨 【單通道 / 原位物理數值還原誤差指標】:")
                    print(f"  • MAE (Native Scale):  {color_metrics.get('native_mae', 0.0):.6f}")
                    print(f"  • RMSE (Native Scale): {color_metrics.get('native_rmse', 0.0):.6f}")
                    print(f"  • 最大峰值誤差 MaxAE:  {color_metrics.get('max_abs_error', 0.0):.6f}")
                    print(f"  • 平均相對誤差 MAPE:   {color_metrics.get('relative_error_pct', 0.0):.4f} %")
                    print(f"  • 信噪比 (SNR):        {color_metrics.get('snr_db', 0.0):.2f} dB")
                else:
                    print("-" * 56)
                    print("  🎨 【RGB 彩色圖像色差與通道誤差指標】:")
                    print(f"  • CIEDE2000 色差 ΔE00: {color_metrics.get('delta_e00_mean', 0.0):.4f} (Max: {color_metrics.get('delta_e00_max', 0.0):.4f})")
                    print(f"  • CIE76 色差 ΔE76:     {color_metrics.get('delta_e76_mean', 0.0):.4f}")
                    print(f"  • Red (紅) 通道 MAE:    {color_metrics.get('r_mae', 0.0):.4f}")
                    print(f"  • Green (綠) 通道 MAE:  {color_metrics.get('g_mae', 0.0):.4f}")
                    print(f"  • Blue (藍) 通道 MAE:   {color_metrics.get('b_mae', 0.0):.4f}")

        print(f"  • 解壓耗時:            {meta_dict['dec_time_ms']:.2f} ms")
        print("========================================================\n")


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def load_remote_sensing_image(path: str | Path) -> Image.Image:
    """Robust Remote Sensing TIF/TIFF Image Loader.

    Supports 8-bit, 24-bit, 32-bit (uint8, uint16, uint32, float32) single-channel,
    3-channel, and 4-channel multispectral images. Normalizes into a standard 3-channel RGB PIL Image.
    """
    path_str = str(path)
    used_cv2 = False
    try:
        img = Image.open(path_str)
        if img.mode in ("RGB", "L", "RGBA"):
            return img.convert("RGB")
        arr = np.array(img)
    except Exception:
        arr = None

    if arr is None:
        try:
            import cv2

            arr = cv2.imread(path_str, cv2.IMREAD_UNCHANGED)
            if arr is None:
                raise ValueError(f"Failed to load image at: {path_str}")
            used_cv2 = True
        except ImportError:
            raise ValueError(f"Cannot load image at: {path_str}. Install opencv-python for advanced TIFF formats.")

    arr = np.asarray(arr, dtype=np.float32)

    if used_cv2 and arr.ndim == 3 and arr.shape[2] >= 3:
        # Convert OpenCV BGR/BGRA to RGB/RGBA
        arr[:, :, :3] = arr[:, :, [2, 1, 0]]

    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3:
        if arr.shape[2] == 1:
            arr = np.concatenate([arr] * 3, axis=-1)
        elif arr.shape[2] >= 3:
            arr = arr[:, :, :3]

    min_val = arr.min()
    max_val = arr.max()

    if max_val > 255.0 or min_val < 0.0:
        if max_val > min_val:
            arr = (arr - min_val) / (max_val - min_val) * 255.0
        else:
            arr = np.zeros_like(arr)
    elif max_val <= 1.0 and min_val >= 0.0:
        arr = arr * 255.0

    return Image.fromarray(arr.astype(np.uint8))


def read_remote_sensing_tif(path: str | Path) -> tuple[torch.Tensor, np.dtype, float, float, int]:
    """Read a remote sensing TIF image into a 32-bit float PyTorch Tensor [3, H, W] in [0, 1].

    Preserves metadata (original dtype, min_val, max_val, orig_channels) to allow exact native bit-depth reconstruction later.
    """
    path_str = str(path)
    used_cv2 = False
    try:
        img = Image.open(path_str)
        if img.mode in ("RGB", "L", "RGBA"):
            arr = np.array(img.convert("RGB"))
        else:
            arr = np.array(img)
    except Exception:
        arr = None

    if arr is None:
        try:
            import cv2

            arr = cv2.imread(path_str, cv2.IMREAD_UNCHANGED)
            if arr is not None:
                used_cv2 = True
                if arr.ndim == 3 and arr.shape[2] >= 3:
                    # Convert OpenCV BGR/BGRA to RGB/RGBA
                    arr[:, :, :3] = arr[:, :, [2, 1, 0]]
        except Exception:
            arr = None

    if arr is None:
        raise FileNotFoundError(f"Could not load image at: {path_str}")

    orig_dtype = arr.dtype
    orig_channels = 1 if arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 1) else 3
    min_val = float(arr.min())
    max_val = float(arr.max())
    arr_f32 = arr.astype(np.float32)

    if orig_dtype == np.uint8:
        norm = arr_f32 / 255.0
    elif max_val <= 1.0 and min_val >= 0.0:
        norm = arr_f32
    elif max_val <= 255.0:
        norm = arr_f32 / 255.0
    elif orig_dtype == np.uint16:
        norm = arr_f32 / 65535.0
    elif max_val > min_val:
        norm = (arr_f32 - min_val) / (max_val - min_val + 1e-7)
    else:
        norm = np.zeros_like(arr_f32)

    if norm.ndim == 2:
        norm = np.stack([norm] * 3, axis=-1)
    elif norm.ndim == 3:
        if norm.shape[2] == 1:
            norm = np.concatenate([norm] * 3, axis=-1)
        elif norm.shape[2] > 3:
            norm = norm[:, :, :3]

    tensor = torch.from_numpy(norm).permute(2, 0, 1)
    return tensor, orig_dtype, min_val, max_val, orig_channels


def tensor_to_remote_sensing_tif(
    tensor: torch.Tensor,
    orig_dtype: np.dtype | str,
    min_val: float,
    max_val: float,
    save_path: str | Path,
    orig_channels: int = 1,
) -> np.ndarray:
    """Restore a 32-bit float PyTorch Tensor [3, H, W] back to its native bit-depth (uint8, uint16, float32)

    and write directly to a file.
    """
    arr_f32 = tensor.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    dtype_str = str(orig_dtype)

    if "uint8" in dtype_str:
        out = (arr_f32 * 255.0).round().astype(np.uint8)
    elif max_val <= 1.0 and min_val >= 0.0:
        out = arr_f32.astype(orig_dtype)
    elif max_val <= 255.0:
        out = (arr_f32 * 255.0).astype(orig_dtype)
    elif "uint16" in dtype_str:
        out = (arr_f32 * 65535.0).round().astype(np.uint16)
    else:
        out = (arr_f32 * (max_val - min_val) + min_val).astype(orig_dtype)

    if orig_channels == 1 and out.ndim == 3:
        out = out[:, :, 0]

    save_str = str(save_path)
    try:
        import cv2

        bgr_out = out.copy()
        if bgr_out.ndim == 3 and bgr_out.shape[2] >= 3:
            bgr_out[:, :, :3] = bgr_out[:, :, [2, 1, 0]]
        cv2.imwrite(save_str, bgr_out)
    except Exception:
        Image.fromarray(out.astype(np.uint8) if out.dtype != np.uint8 else out).save(save_str)
    return out


def calculate_color_and_reconstruction_metrics(
    img_orig: np.ndarray | torch.Tensor,
    img_recon: np.ndarray | torch.Tensor,
    is_tif: bool = False,
    orig_dtype: str = "uint8",
    min_val: float = 0.0,
    max_val: float = 255.0,
    orig_channels: int = 3,
) -> dict[str, float]:
    """Calculate color difference metrics (CIEDE2000 ΔE00, CIE76 ΔE76, RGB MAEs) for RGB images,

    and native-scale reconstruction metrics (MAE, RMSE, MaxAE, Relative Error %, SNR) for single-channel (32-bit float / 16-bit / 8-bit) TIF images.
    """
    import math

    if isinstance(img_orig, torch.Tensor):
        if img_orig.ndim == 4:
            img_orig = img_orig.squeeze(0)
        arr_orig = img_orig.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy().astype(np.float64)
    else:
        arr_orig = np.clip(img_orig, 0.0, 1.0).astype(np.float64)

    if isinstance(img_recon, torch.Tensor):
        if img_recon.ndim == 4:
            img_recon = img_recon.squeeze(0)
        arr_rec = img_recon.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy().astype(np.float64)
    else:
        arr_rec = np.clip(img_recon, 0.0, 1.0).astype(np.float64)

    mse = float(np.mean((arr_orig - arr_rec) ** 2))
    psnr = 100.0 if mse < 1e-10 else float(10.0 * math.log10(1.0 / mse))

    results: dict[str, float] = {
        "mse": mse,
        "psnr": psnr,
        "orig_channels": float(orig_channels),
    }

    if orig_channels == 1:
        dtype_str = str(orig_dtype).lower()
        if "uint8" in dtype_str:
            s_orig = arr_orig[:, :, 0] * 255.0
            s_rec = arr_rec[:, :, 0] * 255.0
        elif "uint16" in dtype_str:
            s_orig = arr_orig[:, :, 0] * 65535.0
            s_rec = arr_rec[:, :, 0] * 65535.0
        else:
            s_orig = arr_orig[:, :, 0] * (max_val - min_val) + min_val
            s_rec = arr_rec[:, :, 0] * (max_val - min_val) + min_val

        mae = float(np.mean(np.abs(s_orig - s_rec)))
        rmse = float(np.sqrt(np.mean((s_orig - s_rec) ** 2)))
        max_ae = float(np.max(np.abs(s_orig - s_rec)))
        rel_err = float(np.mean(np.abs(s_orig - s_rec) / (np.abs(s_orig) + 1e-7)) * 100.0)
        sig_p = float(np.mean(s_orig**2))
        noi_p = float(np.mean((s_orig - s_rec) ** 2))
        snr_db = 100.0 if noi_p < 1e-10 else float(10.0 * math.log10(sig_p / max(1e-10, noi_p)))

        results.update({
            "native_mae": mae,
            "native_rmse": rmse,
            "max_abs_error": max_ae,
            "relative_error_pct": rel_err,
            "snr_db": snr_db,
        })
    else:
        rgb1_u8 = np.clip(arr_orig * 255.0, 0, 255).astype(np.uint8)
        rgb2_u8 = np.clip(arr_rec * 255.0, 0, 255).astype(np.uint8)

        r_mae = float(np.mean(np.abs(rgb1_u8[:, :, 0].astype(float) - rgb2_u8[:, :, 0].astype(float))))
        g_mae = float(np.mean(np.abs(rgb1_u8[:, :, 1].astype(float) - rgb2_u8[:, :, 1].astype(float))))
        b_mae = float(np.mean(np.abs(rgb1_u8[:, :, 2].astype(float) - rgb2_u8[:, :, 2].astype(float))))

        try:
            from skimage.color import deltaE_ciede2000, rgb2lab

            lab1 = rgb2lab(rgb1_u8)
            lab2 = rgb2lab(rgb2_u8)
            de00_map = deltaE_ciede2000(lab1, lab2)
            de00_mean = float(np.mean(de00_map))
            de00_max = float(np.max(de00_map))
            de76_mean = float(np.mean(np.sqrt(np.sum((lab1.astype(float) - lab2.astype(float)) ** 2, axis=-1))))
        except Exception:
            try:
                import cv2

                lab1 = cv2.cvtColor(rgb1_u8, cv2.COLOR_RGB2LAB).astype(np.float64)
                lab2 = cv2.cvtColor(rgb2_u8, cv2.COLOR_RGB2LAB).astype(np.float64)
                de76_map = np.sqrt(np.sum((lab1 - lab2) ** 2, axis=-1))
                de76_mean = float(np.mean(de76_map))
                de00_mean = de76_mean
                de00_max = float(np.max(de76_map))
            except Exception:
                de00_mean = float((r_mae + g_mae + b_mae) / 3.0)
                de00_max = float(max(r_mae, g_mae, b_mae))
                de76_mean = de00_mean

        results.update({
            "delta_e00_mean": de00_mean,
            "delta_e00_max": de00_max,
            "delta_e76_mean": de76_mean,
            "r_mae": r_mae,
            "g_mae": g_mae,
            "b_mae": b_mae,
        })

    return results


def image_to_tensor(path: str | Path) -> torch.Tensor:
    image = load_remote_sensing_image(path)
    array = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().cpu().clamp(0.0, 1.0)
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    array = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array)


def pad_to_multiple(x: torch.Tensor, multiple: int) -> tuple[torch.Tensor, tuple[int, int]]:
    height, width = x.shape[-2:]
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    return x, (int(height), int(width))


def crop_to_size(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    target_h, target_w = size
    return x[..., :target_h, :target_w]


def resize_tensor_to_size(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    target_h, target_w = size
    return F.interpolate(x, size=(target_h, target_w), mode="bicubic", align_corners=False)


def load_checkpoint(model: torch.nn.Module, checkpoint: str | Path) -> None:
    raw = torch.load(checkpoint, map_location="cpu")
    state_dict = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"missing_keys: {len(missing)}")
    if missing:
        print("\n".join(f"  {key}" for key in missing[:20]))
    print(f"unexpected_keys: {len(unexpected)}")
    if unexpected:
        print("\n".join(f"  {key}" for key in unexpected[:20]))

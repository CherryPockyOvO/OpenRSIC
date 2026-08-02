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
    try:
        img = Image.open(path_str)
        if img.mode in ("RGB", "L", "RGBA"):
            return img.convert("RGB")
        arr = np.array(img)
    except Exception:
        try:
            import cv2

            arr = cv2.imread(path_str, cv2.IMREAD_UNCHANGED)
            if arr is None:
                raise ValueError(f"Failed to load image at: {path_str}")
        except ImportError:
            raise ValueError(f"Cannot load image at: {path_str}. Install opencv-python for advanced TIFF formats.")

    arr = np.asarray(arr, dtype=np.float32)

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

def read_remote_sensing_tif(path: str | Path) -> tuple[torch.Tensor, np.dtype, float, float]:
    """Read a remote sensing TIF image into a 32-bit float PyTorch Tensor [3, H, W] in [0, 1].

    Preserves metadata (original dtype, min_val, max_val) to allow exact native bit-depth reconstruction later.
    """
    path_str = str(path)
    try:
        import cv2

        arr = cv2.imread(path_str, cv2.IMREAD_UNCHANGED)
        if arr is not None and arr.ndim == 3 and arr.shape[2] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    except Exception:
        arr = None

    if arr is None:
        img = Image.open(path_str)
        arr = np.array(img)

    orig_dtype = arr.dtype
    min_val = float(arr.min())
    max_val = float(arr.max())
    arr_f32 = arr.astype(np.float32)

    if orig_dtype == np.uint8:
        norm = arr_f32 / 255.0
    elif orig_dtype == np.uint16:
        norm = arr_f32 / 65535.0
    else:
        norm = (arr_f32 - min_val) / (max_val - min_val + 1e-7) if max_val > min_val else np.zeros_like(arr_f32)

    if norm.ndim == 2:
        norm = np.stack([norm] * 3, axis=-1)
    elif norm.ndim == 3:
        if norm.shape[2] == 1:
            norm = np.concatenate([norm] * 3, axis=-1)
        elif norm.shape[2] > 3:
            norm = norm[:, :, :3]

    tensor = torch.from_numpy(norm).permute(2, 0, 1)
    return tensor, orig_dtype, min_val, max_val


def tensor_to_remote_sensing_tif(
    tensor: torch.Tensor,
    orig_dtype: np.dtype | str,
    min_val: float,
    max_val: float,
    save_path: str | Path,
) -> np.ndarray:
    """Restore a 32-bit float PyTorch Tensor [3, H, W] back to its native bit-depth (uint8, uint16, float32)

    and write directly to a .tif / .tiff file.
    """
    arr_f32 = tensor.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    dtype_str = str(orig_dtype)

    if "uint8" in dtype_str:
        out = (arr_f32 * 255.0).round().astype(np.uint8)
    elif "uint16" in dtype_str:
        out = (arr_f32 * 65535.0).round().astype(np.uint16)
    else:
        out = (arr_f32 * (max_val - min_val) + min_val).astype(orig_dtype)

    save_str = str(save_path)
    try:
        import cv2

        bgr_out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR) if out.ndim == 3 and out.shape[2] == 3 else out
        cv2.imwrite(save_str, bgr_out)
    except Exception:
        Image.fromarray(out.astype(np.uint8) if out.dtype != np.uint8 else out).save(save_str)
    return out


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

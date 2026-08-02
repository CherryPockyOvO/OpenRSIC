from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
Image.MAX_IMAGE_PIXELS = None
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from rsic import (
    MODEL_CONFIGS,
    MODEL_VARIANT_RSIC,
    QATSettings,
    get_model,
    infer_model_variant_from_checkpoint,
    model_config_to_dict,
    normalize_model_variant,
)
from rsic.utils import load_remote_sensing_image

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}

TRAIN_PROFILES = {
    "rsic_fp": {
        "model_variant": MODEL_VARIANT_RSIC,
        "lmbda": 0.0483,
        "max_bpp": 1.50,
        "l1_weight": 0.0,
        "epochs": 100,
        "batch_size": 32,
        "crop_size": 512,
        "lr": 1.5e-4,
        "enable_latent_fake_quant": False,
        "enable_z_fake_quant": False,
        "enable_scale_fake_quant": False,
        "latent_range_weight": 0.0,
        "z_range_weight": 0.0,
    },
    "rsic_qat8": {
        "model_variant": MODEL_VARIANT_RSIC,
        "lmbda": 0.0483,
        "max_bpp": 1.50,
        "l1_weight": 0.0,
        "epochs": 30,
        "batch_size": 32,
        "crop_size": 512,
        "lr": 1e-5,
        "enable_latent_fake_quant": True,
        "latent_fake_quant_bits": 8,
        "latent_fake_quant_clip": 6.0,
        "enable_z_fake_quant": True,
        "z_fake_quant_bits": 8,
        "z_fake_quant_clip": 6.0,
        "enable_scale_fake_quant": True,
        "scale_fake_quant_bits": 8,
        "scale_fake_quant_clip": 8.0,
        "latent_range_weight": 0.01,
        "z_range_weight": 0.01,
    },
}


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


@dataclass(frozen=True)
class CheckpointState:
    epoch: int
    global_step: int
    model_variant: str


def init_distributed() -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if world_size <= 1:
        return DistributedContext(enabled=False)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return DistributedContext(enabled=True, rank=rank, local_rank=local_rank, world_size=world_size)


def cleanup_distributed(distributed: DistributedContext) -> None:
    if distributed.enabled and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model: nn.Module) -> nn.Module:
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


class ImageFolderDataset(Dataset):
    def __init__(self, root: Path, transform: Callable[[Image.Image], torch.Tensor]) -> None:
        self.root = Path(root)
        self.transform = transform
        self.paths = sorted(
            p for p in self.root.rglob("*")
            if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise FileNotFoundError(f"No images found in {self.root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = load_remote_sensing_image(self.paths[index])
        if image is None:
            image = Image.new("RGB", (512, 512), (0, 0, 0))
        return self.transform(image)


class Compose:
    def __init__(self, transforms: list[Callable[[Any], Any]]) -> None:
        self.transforms = transforms

    def __call__(self, image: Image.Image) -> torch.Tensor:
        for t in self.transforms:
            image = t(image)
        return image


class RandomCrop:
    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        if w <= self.size or h <= self.size:
            return image.resize((self.size, self.size), Image.Resampling.BICUBIC)
        left = random.randint(0, w - self.size)
        top = random.randint(0, h - self.size)
        return image.crop((left, top, left + self.size, top + self.size))


class RandomHorizontalFlip:
    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() < 0.5:
            return ImageOps.mirror(image)
        return image


class ToTensor:
    def __call__(self, image: Image.Image) -> torch.Tensor:
        arr = np.asarray(image).astype(np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)


def make_train_transform(crop_size: int) -> Compose:
    return Compose([RandomCrop(crop_size), RandomHorizontalFlip(), ToTensor()])


def make_eval_transform(crop_size: int) -> Compose:
    return Compose([RandomCrop(crop_size), ToTensor()])


def compute_bpp(likelihoods: dict[str, torch.Tensor], num_pixels: int) -> torch.Tensor:
    bits = torch.zeros((), device=next(iter(likelihoods.values())).device, dtype=torch.float32)
    for likelihood in likelihoods.values():
        lh_f32 = likelihood.float().clamp(1e-9, 1.0)
        bits = bits + torch.sum(-torch.log2(lh_f32))
    return bits / float(num_pixels)


def psnr_from_mse(mse: float) -> float:
    if mse <= 1e-10:
        return 99.99
    return float(10.0 * math.log10(1.0 / mse))


class RateDistortionLoss(nn.Module):
    """OpenRSIC Rate-Distortion Loss Function with Max BPP Ceiling Penalty."""

    def __init__(
        self,
        lmbda: float = 0.0483,
        max_bpp: float | None = 1.50,
        l1_weight: float = 0.0,
        latent_range_weight: float = 0.01,
        z_range_weight: float = 0.01,
    ) -> None:
        super().__init__()
        self.lmbda = float(lmbda)
        self.max_bpp = float(max_bpp) if max_bpp is not None else None
        self.l1_weight = float(l1_weight)
        self.latent_range_weight = float(latent_range_weight)
        self.z_range_weight = float(z_range_weight)

    def forward(self, output: dict[str, Any], target: torch.Tensor) -> dict[str, torch.Tensor]:
        x_hat = output["x_hat"]
        n, _, h, w = target.shape
        num_pixels = n * h * w

        bpp = compute_bpp(output["likelihoods"], num_pixels)
        mse = F.mse_loss(x_hat, target)
        l1 = F.l1_loss(x_hat, target)

        distortion = 255.0**2 * (mse + self.l1_weight * l1)
        loss = self.lmbda * distortion + bpp

        # Strict Max BPP Ceiling Penalty (ONLY penalizes if BPP exceeds 1.50 ceiling)
        if self.max_bpp is not None and bpp > self.max_bpp:
            loss = loss + 10.0 * (bpp - self.max_bpp) ** 2

        # QAT Soft Range Penalties
        if self.latent_range_weight > 0 and "y" in output:
            loss = loss + self.latent_range_weight * torch.relu(output["y"].abs() - 5.8).mean()
        if self.z_range_weight > 0 and "z" in output:
            loss = loss + self.z_range_weight * torch.relu(output["z"].abs() - 5.8).mean()

        return {
            "loss": loss,
            "mse": mse,
            "bpp": bpp,
            "distortion": distortion,
        }


def save_train_config_json(args: argparse.Namespace, checkpoint_dir: Path, model: nn.Module) -> None:
    model = unwrap_model(model)
    config_dict = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_variant": getattr(model, "model_variant", MODEL_VARIANT_RSIC),
        "model_config": model.model_config_dict() if hasattr(model, "model_config_dict") else {},
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    scaler: torch.amp.GradScaler | None,
    epoch: int,
    global_step: int,
    args: argparse.Namespace,
    metrics: dict[str, float],
) -> None:
    model = unwrap_model(model)
    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "model_variant": getattr(model, "model_variant", MODEL_VARIANT_RSIC),
        "model_config": model.model_config_dict() if hasattr(model, "model_config_dict") else {},
        "decoder_type": getattr(args, "decoder_type", "swin"),
        "quality_profile": args.quality_profile,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics": metrics,
    }
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> CheckpointState:
    raw = torch.load(path, map_location="cpu")
    state_dict = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    model_variant = infer_model_variant_from_checkpoint(raw)
    model.load_state_dict(state_dict, strict=False)
    if optimizer is not None and isinstance(raw, dict) and "optimizer" in raw:
        optimizer.load_state_dict(raw["optimizer"])
    if scheduler is not None and isinstance(raw, dict) and "scheduler" in raw:
        scheduler.load_state_dict(raw["scheduler"])
    if scaler is not None and isinstance(raw, dict) and "scaler" in raw:
        scaler.load_state_dict(raw["scaler"])
    if isinstance(raw, dict):
        return CheckpointState(
            epoch=int(raw.get("epoch", 0)),
            global_step=int(raw.get("global_step", 0)),
            model_variant=model_variant,
        )
    return CheckpointState(epoch=0, global_step=0, model_variant=model_variant)


def train_one_epoch(
    model: nn.Module,
    criterion: RateDistortionLoss,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    grad_clip: float,
    amp_enabled: bool,
    epoch: int,
    global_step: int,
    checkpoint_interval_steps: int,
    save_step_cb: Callable[[int, int, dict[str, float]], bool],
    distributed: DistributedContext,
) -> tuple[dict[str, float], int, bool]:
    model.train()
    total_loss, total_mse, total_bpp, samples = 0.0, 0.0, 0.0, 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d}", disable=not distributed.is_main)
    for x in pbar:
        x = x.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            out = model(x)
            loss_dict = criterion(out, x)
            loss = loss_dict["loss"]

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        global_step += 1
        bs = x.size(0)
        samples += bs
        total_loss += float(loss.item()) * bs
        total_mse += float(loss_dict["mse"].item()) * bs
        total_bpp += float(loss_dict["bpp"].item()) * bs

        cur_psnr = psnr_from_mse(loss_dict["mse"].item())
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "bpp": f"{loss_dict['bpp'].item():.3f}", "psnr": f"{cur_psnr:.2f}dB"})

        if global_step % checkpoint_interval_steps == 0:
            avg_metrics = {
                "loss": total_loss / max(1, samples),
                "mse": total_mse / max(1, samples),
                "psnr": psnr_from_mse(total_mse / max(1, samples)),
                "bpp": total_bpp / max(1, samples),
            }
            if not save_step_cb(epoch, global_step, avg_metrics):
                return avg_metrics, global_step, True

    avg_metrics = {
        "loss": total_loss / max(1, samples),
        "mse": total_mse / max(1, samples),
        "psnr": psnr_from_mse(total_mse / max(1, samples)),
        "bpp": total_bpp / max(1, samples),
    }
    return avg_metrics, global_step, False


@torch.no_grad()
def evaluate(
    model: nn.Module,
    criterion: RateDistortionLoss,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss, total_mse, total_bpp, samples = 0.0, 0.0, 0.0, 0
    for x in loader:
        x = x.to(device, non_blocking=True)
        out = model(x)
        loss_dict = criterion(out, x)
        bs = x.size(0)
        samples += bs
        total_loss += float(loss_dict["loss"].item()) * bs
        total_mse += float(loss_dict["mse"].item()) * bs
        total_bpp += float(loss_dict["bpp"].item()) * bs

    avg_mse = total_mse / max(1, samples)
    return {
        "val_loss": total_loss / max(1, samples),
        "val_mse": avg_mse,
        "val_psnr": psnr_from_mse(avg_mse),
        "val_bpp": total_bpp / max(1, samples),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenRSIC PyTorch Training Pipeline.")
    parser.add_argument("--quality-profile", choices=list(TRAIN_PROFILES.keys()), default="rsic_fp")
    parser.add_argument("--decoder-type", choices=["standard", "cheng2020_attention", "swin"], default="swin")
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint-interval-steps", type=int, default=500)
    parser.add_argument("--eval-interval-steps", type=int, default=500)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    # Load profile defaults
    profile = TRAIN_PROFILES[args.quality_profile]
    for key, val in profile.items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, val)
    return args


def main() -> None:
    distributed = init_distributed()
    args = parse_args()
    device = torch.device(f"cuda:{distributed.local_rank}" if torch.cuda.is_available() else "cpu")
    amp_enabled = not args.no_amp and torch.cuda.is_available()

    # Datasets
    train_dataset = ImageFolderDataset(args.train_dir, transform=make_train_transform(args.crop_size))
    train_sampler = DistributedSampler(train_dataset) if distributed.enabled else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = None
    if args.val_dir and args.val_dir.exists():
        val_dataset = ImageFolderDataset(args.val_dir, transform=make_eval_transform(args.crop_size))
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # QAT Settings & Model
    qat = QATSettings(
        enable_latent_fake_quant=getattr(args, "enable_latent_fake_quant", False),
        latent_fake_quant_bits=getattr(args, "latent_fake_quant_bits", 8),
        latent_fake_quant_clip=getattr(args, "latent_fake_quant_clip", 6.0),
        enable_z_fake_quant=getattr(args, "enable_z_fake_quant", False),
        z_fake_quant_bits=getattr(args, "z_fake_quant_bits", 8),
        z_fake_quant_clip=getattr(args, "z_fake_quant_clip", 6.0),
        enable_scale_fake_quant=getattr(args, "enable_scale_fake_quant", False),
        scale_fake_quant_bits=getattr(args, "scale_fake_quant_bits", 8),
        scale_fake_quant_clip=getattr(args, "scale_fake_quant_clip", 8.0),
    )
    base_model = get_model(decoder_type=args.decoder_type, qat=qat).to(device)
    criterion = RateDistortionLoss(
        lmbda=getattr(args, "lmbda", 0.0483),
        max_bpp=getattr(args, "max_bpp", 1.50),
        latent_range_weight=getattr(args, "latent_range_weight", 0.0),
        z_range_weight=getattr(args, "z_range_weight", 0.0),
    ).to(device)

    model: nn.Module = base_model
    if distributed.enabled:
        model = DistributedDataParallel(base_model, device_ids=[distributed.local_rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    global_step, start_epoch = 0, 0
    if args.init_checkpoint:
        state = load_checkpoint(args.init_checkpoint, base_model)
        if distributed.is_main:
            print(f"Loaded weights from {args.init_checkpoint}")
    elif args.resume:
        state = load_checkpoint(args.resume, base_model, optimizer, scheduler, scaler)
        start_epoch, global_step = state.epoch, state.global_step
        if distributed.is_main:
            print(f"Resumed training from {args.resume} (epoch {start_epoch}, step {global_step})")

    if distributed.is_main:
        save_train_config_json(args, args.checkpoint_dir, base_model)
        print(f"🚀 Training OpenRSIC | Model: RSIC | Decoder: {args.decoder_type} | Profile: {args.quality_profile}")

    best_val_loss = math.inf

    def save_step_checkpoint(epoch: int, step: int, train_metrics: dict[str, float]) -> bool:
        nonlocal best_val_loss
        metrics = dict(train_metrics)
        if val_loader is not None:
            val_metrics = evaluate(model, criterion, val_loader, device)
            metrics.update(val_metrics)
            scheduler.step(val_metrics["val_loss"])

        improved = "val_loss" in metrics and metrics["val_loss"] < best_val_loss
        if improved:
            best_val_loss = metrics["val_loss"]

        if distributed.is_main:
            ckpt_name = f"iter_{step:07d}.pt"
            print(f"📌 Step {step:07d} (Epoch {epoch:03d}): Loss={metrics['loss']:.4f}, PSNR={metrics.get('val_psnr', metrics['psnr']):.2f}dB, BPP={metrics.get('val_bpp', metrics['bpp']):.3f} {'[BEST]' if improved else ''}")
            save_checkpoint(args.checkpoint_dir / ckpt_name, base_model, optimizer, scheduler, scaler, epoch, step, args, metrics)
            save_checkpoint(args.checkpoint_dir / "latest.pt", base_model, optimizer, scheduler, scaler, epoch, step, args, metrics)
            if improved:
                save_checkpoint(args.checkpoint_dir / "best.pt", base_model, optimizer, scheduler, scaler, epoch, step, args, metrics)
        return True

    try:
        for epoch in range(start_epoch + 1, args.epochs + 1):
            if train_sampler:
                train_sampler.set_epoch(epoch)
            train_metrics, global_step, stop = train_one_epoch(
                model, criterion, train_loader, optimizer, scaler, device,
                grad_clip=1.0, amp_enabled=amp_enabled, epoch=epoch,
                global_step=global_step, checkpoint_interval_steps=args.checkpoint_interval_steps,
                save_step_cb=save_step_checkpoint, distributed=distributed,
            )
            if stop:
                break
    finally:
        cleanup_distributed(distributed)


if __name__ == "__main__":
    main()

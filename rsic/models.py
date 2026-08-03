from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .entropy import EntropyPayload, NanoEntropyBottleneck
from .layers import conv, deconv, init_module, make_activation


MODEL_VARIANT_RSIC = "rsic"
MODEL_VARIANT_HYPER_MS_Q_NANO = MODEL_VARIANT_RSIC
MODEL_VARIANT_HYPER_MS_Q = MODEL_VARIANT_RSIC


@dataclass(frozen=True)
class ModelConfig:
    N: int
    M: int
    quant_step: float
    decoder_channels: int
    decoder_res_blocks: int
    refinement_blocks: int
    name: str
    model_variant: str = MODEL_VARIANT_RSIC
    model_type: str = "mean_scale_hyperprior"
    encoder_type: str = "residual_quant_friendly_signed"
    activation: str = "relu6"
    encoder_norm: str = "none"
    Z: int | None = None
    latent_clip: float | None = 6.0
    signed_latent: bool = True
    z_clip: float | None = 6.0
    scale_min: float = 1e-3
    scale_max: float = 20.0


MODEL_CONFIGS: dict[str, ModelConfig] = {
    MODEL_VARIANT_RSIC: ModelConfig(
        N=160,
        M=256,
        Z=128,
        quant_step=0.38,
        decoder_channels=160,
        decoder_res_blocks=1,
        refinement_blocks=1,
        name=MODEL_VARIANT_RSIC,
        model_variant=MODEL_VARIANT_RSIC,
        model_type="mean_scale_hyperprior",
        encoder_type="residual_quant_friendly_signed",
        activation="relu6",
        encoder_norm="none",
        latent_clip=6.0,
        signed_latent=True,
        z_clip=6.0,
        scale_min=1e-3,
        scale_max=20.0,
    ),
}

MODEL_CONFIG = MODEL_CONFIGS[MODEL_VARIANT_RSIC]


def normalize_model_variant(model_variant: str | None = None) -> str:
    if model_variant is None:
        return MODEL_VARIANT_HYPER_MS_Q_NANO
    normalized = str(model_variant).strip()
    if not normalized:
        return MODEL_VARIANT_HYPER_MS_Q_NANO
    if normalized not in MODEL_CONFIGS:
        choices = ", ".join(sorted(MODEL_CONFIGS))
        raise ValueError(f"unknown model_variant={normalized!r}; choices: {choices}")
    return normalized


def get_model_config(model_variant: str | None = None) -> ModelConfig:
    return MODEL_CONFIGS[normalize_model_variant(model_variant)]


def model_config_to_dict(config: ModelConfig) -> dict[str, Any]:
    return asdict(config)


def infer_model_variant_from_checkpoint(raw: object) -> str:
    if not isinstance(raw, dict):
        return MODEL_VARIANT_HYPER_MS_Q_NANO

    variant = raw.get("model_variant")
    if isinstance(variant, str) and variant:
        return normalize_model_variant(variant)

    config = raw.get("model_config")
    if isinstance(config, dict):
        variant = config.get("model_variant") or config.get("name")
        if isinstance(variant, str) and variant in MODEL_CONFIGS:
            return normalize_model_variant(variant)

    return MODEL_VARIANT_HYPER_MS_Q_NANO


def clip_latent(value: Tensor, clip: float | None) -> Tensor:
    if clip is None or clip <= 0:
        return value
    clip_tensor = value.new_tensor(float(clip))
    return clip_tensor * torch.tanh(value / clip_tensor)


def fake_quant_symmetric_ste(x: Tensor, bits: int = 8, clip: float = 6.0) -> Tensor:
    if bits <= 0:
        raise ValueError(f"fake quant bits must be positive, got {bits}")
    if clip <= 0:
        return x
    qmax = 2 ** (bits - 1) - 1
    scale = float(clip) / float(qmax)
    x_clip = x.clamp(-float(clip), float(clip))
    q = torch.round(x_clip / scale).clamp(-qmax, qmax)
    x_q = q * scale
    return x + (x_q - x).detach()


def fake_quant_positive_ste(x: Tensor, bits: int = 8, clip: float = 8.0) -> Tensor:
    if bits <= 0:
        raise ValueError(f"fake quant bits must be positive, got {bits}")
    if clip <= 0:
        return x
    qmax = 2**bits - 1
    scale = float(clip) / float(qmax)
    x_clip = x.clamp(0.0, float(clip))
    q = torch.round(x_clip / scale).clamp(0, qmax)
    x_q = q * scale
    return x + (x_q - x).detach()


@dataclass
class QATSettings:
    enable_latent_fake_quant: bool = False
    latent_fake_quant_bits: int = 8
    latent_fake_quant_clip: float = 6.0
    enable_z_fake_quant: bool = False
    z_fake_quant_bits: int = 8
    z_fake_quant_clip: float = 6.0
    enable_scale_fake_quant: bool = False
    scale_fake_quant_bits: int = 8
    scale_fake_quant_clip: float = 8.0


class ConvAct(nn.Sequential):
    """Conv2d + ReLU/ReLU6 without normalization for RKNN quantization experiments."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str = "relu6",
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        bias: bool = True,
    ) -> None:
        if kernel_size not in {1, 3, 5}:
            raise ValueError("ConvAct kernel_size must be 1, 3, or 5")
        if padding is None:
            padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            make_activation(activation),
        )


class QuantResidualBlock(nn.Module):
    """Quantization-friendly residual block: Conv3x3 -> ReLU6 -> Conv3x3 -> Add -> ReLU6."""

    def __init__(self, channels: int, activation: str = "relu6") -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act1 = make_activation(activation)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act2 = make_activation(activation)

    def forward(self, x: Tensor) -> Tensor:
        residual = self.conv2(self.act1(self.conv1(x)))
        return self.act2(x + residual)


class DownsampleResidualBlock(nn.Module):
    """Stride-2 residual downsampler with an explicit 1x1 skip branch."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str = "relu6",
        output_activation: bool = True,
    ) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            make_activation(activation),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
        self.out_act = make_activation(activation) if output_activation else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.out_act(self.main(x) + self.skip(x))


class QuantFriendlyResidualEncoder(nn.Module):
    """Residual analysis transform prepared for RKNN INT8/FP16 mixed precision."""

    def __init__(
        self,
        N: int = 128,
        M: int = 160,
        activation: str = "relu6",
        latent_clip: float | None = 6.0,
        signed_latent: bool = True,
    ) -> None:
        super().__init__()
        self.latent_clip = latent_clip
        self.signed_latent = bool(signed_latent)
        self.down1 = DownsampleResidualBlock(3, N, activation=activation)
        self.res1 = QuantResidualBlock(N, activation=activation)
        self.down2 = DownsampleResidualBlock(N, N, activation=activation)
        self.res2 = QuantResidualBlock(N, activation=activation)
        self.down3 = DownsampleResidualBlock(N, N, activation=activation)
        self.res3 = QuantResidualBlock(N, activation=activation)
        self.down4 = DownsampleResidualBlock(
            N,
            M,
            activation=activation,
            output_activation=not self.signed_latent,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.res1(self.down1(x))
        x = self.res2(self.down2(x))
        x = self.res3(self.down3(x))
        y = self.down4(x)
        return clip_latent(y, self.latent_clip)


class HyperEncoder(nn.Module):
    """Lightweight scale-only hyper encoder h_a: y -> z."""

    def __init__(
        self,
        M: int,
        N: int,
        Z: int,
        activation: str = "relu6",
        z_clip: float | None = 6.0,
    ) -> None:
        super().__init__()
        self.z_clip = z_clip
        self.net = nn.Sequential(
            ConvAct(M, N, activation=activation, kernel_size=3, stride=1),
            ConvAct(N, N, activation=activation, kernel_size=3, stride=2),
            nn.Conv2d(N, Z, kernel_size=3, stride=2, padding=1),
        )

    def forward(self, y: Tensor) -> Tensor:
        return clip_latent(self.net(y), self.z_clip)


class HyperDecoder(nn.Module):
    """Lightweight scale-only hyper decoder h_s: z_hat -> scales_y."""

    def __init__(
        self,
        Z: int,
        N: int,
        M: int,
        activation: str = "relu6",
        scale_min: float = 1e-3,
        scale_max: float = 20.0,
    ) -> None:
        super().__init__()
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.net = nn.Sequential(
            deconv(Z, N, kernel_size=3, stride=2),
            make_activation(activation),
            deconv(N, N, kernel_size=3, stride=2),
            make_activation(activation),
            nn.Conv2d(N, M, kernel_size=3, padding=1),
        )

    def make_positive_scale(self, raw: Tensor) -> Tensor:
        scale = F.softplus(raw) + self.scale_min
        return scale.clamp(self.scale_min, self.scale_max)

    def forward(self, z_hat: Tensor) -> Tensor:
        return self.make_positive_scale(self.net(z_hat))


class HyperMeanScaleDecoder(nn.Module):
    """Lightweight h_s head: z_hat -> (scales_y, means_y)."""

    def __init__(
        self,
        Z: int,
        N: int,
        M: int,
        activation: str = "relu6",
        scale_min: float = 1e-3,
        scale_max: float = 20.0,
    ) -> None:
        super().__init__()
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.net = nn.Sequential(
            deconv(Z, N, kernel_size=3, stride=2),
            make_activation(activation),
            deconv(N, N, kernel_size=3, stride=2),
            make_activation(activation),
            nn.Conv2d(N, 2 * M, kernel_size=3, padding=1),
        )

    def make_positive_scale(self, raw: Tensor) -> Tensor:
        scale = F.softplus(raw) + self.scale_min
        return scale.clamp(self.scale_min, self.scale_max)

    def forward(self, z_hat: Tensor) -> tuple[Tensor, Tensor]:
        raw = self.net(z_hat)
        raw_scales, means = raw.chunk(2, dim=1)
        return self.make_positive_scale(raw_scales), means


class GaussianConditionalEntropy(nn.Module):
    """Gaussian conditional entropy model for y with optional mean prediction."""

    def __init__(
        self,
        quant_step: float = 1.0,
        scale_min: float = 1e-3,
        scale_max: float = 20.0,
        likelihood_bound: float = 1e-9,
    ) -> None:
        super().__init__()
        if quant_step <= 0:
            raise ValueError("quant_step must be positive")
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.likelihood_bound = float(likelihood_bound)
        self.register_buffer("quant_step", torch.tensor(float(quant_step)))

    def forward(
        self,
        y: Tensor,
        scales_y: Tensor,
        means_y: Tensor | None = None,
        training: bool | None = None,
    ) -> tuple[Tensor, Tensor]:
        if training is None:
            training = self.training
        step = self._step_like(y)
        means = self._means_like(y, means_y)
        if training:
            y_hat = y + (torch.rand_like(y) - 0.5) * step
        else:
            y_hat = self.quantize(y, means_y).to(dtype=y.dtype) * step + means
        likelihoods = self._likelihood(y_hat, scales_y, means_y)
        return y_hat, likelihoods

    def quantize(self, y: Tensor, means_y: Tensor | None = None) -> Tensor:
        step = self._step_like(y)
        means = self._means_like(y, means_y)
        return torch.round((y - means) / step).to(torch.int32)

    def dequantize(
        self,
        symbols: Tensor,
        means_y: Tensor | None = None,
        quant_step: float | Tensor | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> Tensor:
        if device is None:
            device = symbols.device
        symbols = symbols.to(device=device, dtype=dtype)
        if quant_step is None:
            step = self.quant_step.to(device=device, dtype=dtype)
        else:
            step = torch.as_tensor(quant_step, dtype=dtype, device=device)
        means = 0.0
        if means_y is not None:
            means = means_y.to(device=device, dtype=dtype)
        return symbols * step + means

    def _likelihood(
        self,
        y_hat: Tensor,
        scales_y: Tensor,
        means_y: Tensor | None = None,
    ) -> Tensor:
        step = self._step_like(y_hat).float()
        means = self._means_like(y_hat, means_y).float()
        scales = scales_y.to(device=y_hat.device, dtype=torch.float32).clamp(
            self.scale_min,
            self.scale_max,
        )
        centered = y_hat.float() - means
        upper = self._standardized_cumulative((centered + 0.5 * step) / scales)
        lower = self._standardized_cumulative((centered - 0.5 * step) / scales)
        return (upper - lower).clamp(self.likelihood_bound, 1.0)

    @staticmethod
    def _standardized_cumulative(inputs: Tensor) -> Tensor:
        inputs_f32 = inputs.float()
        return 0.5 * torch.erfc(-inputs_f32 / 1.4142135623730951)

    def _step_like(self, tensor: Tensor) -> Tensor:
        return self.quant_step.to(device=tensor.device, dtype=tensor.dtype)

    @staticmethod
    def _means_like(tensor: Tensor, means_y: Tensor | None = None) -> Tensor:
        if means_y is None:
            return tensor.new_zeros(())
        return means_y.to(device=tensor.device, dtype=tensor.dtype)


class ResidualBlock(nn.Module):
    """PC-side residual block used only by the decoder."""

    def __init__(self, channels: int, activation: str = "leaky_relu") -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            make_activation(activation),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + 0.1 * self.body(x)


class UpsampleResidualBlock(nn.Module):
    """2x upsampling followed by several residual refinement blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        activation: str = "leaky_relu",
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = [
            deconv(in_channels, out_channels),
            make_activation(activation),
        ]
        blocks.extend(ResidualBlock(out_channels, activation) for _ in range(num_blocks))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Decoder(nn.Module):
    """PC-quality synthesis transform g_s: quantized latent y_hat -> x_hat."""

    def __init__(
        self,
        N: int = 128,
        M: int = 128,
        decoder_channels: int = 256,
        decoder_res_blocks: int = 3,
        refinement_blocks: int = 5,
        activation: str = "leaky_relu",
        clamp_output: bool = True,
    ) -> None:
        super().__init__()
        del N
        c0 = int(decoder_channels)
        c1 = max(128, c0)
        c2 = max(96, c0 // 2)
        c3 = max(64, c0 // 4)

        self.stem = nn.Sequential(
            nn.Conv2d(M, c0, kernel_size=3, padding=1),
            make_activation(activation),
            *[ResidualBlock(c0, activation) for _ in range(decoder_res_blocks)],
        )
        self.up1 = UpsampleResidualBlock(c0, c1, decoder_res_blocks, activation)
        self.up2 = UpsampleResidualBlock(c1, c2, decoder_res_blocks, activation)
        self.up3 = UpsampleResidualBlock(c2, c3, decoder_res_blocks, activation)
        self.up4 = UpsampleResidualBlock(c3, 64, refinement_blocks, activation)
        self.to_rgb = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            make_activation(activation),
            *[ResidualBlock(64, activation) for _ in range(refinement_blocks)],
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
        )
        self.clamp_output = bool(clamp_output)

    def forward(self, y_hat: Tensor) -> Tensor:
        x = self.stem(y_hat)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x_hat = self.to_rgb(x)
        if self.clamp_output:
            x_hat = torch.sigmoid(x_hat)
        return x_hat


class ResidualBlockWithAttention(nn.Module):
    """CompressAI Cheng2020 Residual Block with Spatial Attention Gate."""

    def __init__(self, channels: int, activation: str = "leaky_relu") -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act1 = make_activation(activation)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.attn = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=1),
            make_activation(activation),
            nn.Conv2d(channels // 2, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        residual = self.conv2(self.act1(self.conv1(x)))
        attention = self.attn(residual)
        return x + residual * attention


class Cheng2020AttentionDecoder(nn.Module):
    """CompressAI Cheng2020 Attention Synthesis Transform g_s."""

    def __init__(
        self,
        N: int = 160,
        M: int = 256,
        activation: str = "leaky_relu",
        clamp_output: bool = True,
    ) -> None:
        super().__init__()
        c0 = N
        c1 = N
        c2 = max(96, N // 2)
        c3 = 64

        self.stem = nn.Conv2d(M, c0, kernel_size=3, padding=1)
        self.stage1 = nn.Sequential(
            ResidualBlockWithAttention(c0, activation),
            deconv(c0, c1),
            make_activation(activation),
        )
        self.stage2 = nn.Sequential(
            ResidualBlockWithAttention(c1, activation),
            deconv(c1, c2),
            make_activation(activation),
        )
        self.stage3 = nn.Sequential(
            ResidualBlockWithAttention(c2, activation),
            deconv(c2, c3),
            make_activation(activation),
        )
        self.stage4 = nn.Sequential(
            ResidualBlockWithAttention(c3, activation),
            deconv(c3, 3),
        )
        self.clamp_output = bool(clamp_output)

    def forward(self, y_hat: Tensor) -> Tensor:
        x = self.stem(y_hat)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x_hat = self.stage4(x)
        if self.clamp_output:
            x_hat = torch.sigmoid(x_hat)
        return x_hat


def window_partition(x: Tensor, window_size: int) -> tuple[Tensor, int, int]:
    B, C, H, W = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    _, _, Hp, Wp = x.shape
    x = x.view(B, C, Hp // window_size, window_size, Wp // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, window_size * window_size, C)
    return windows, Hp, Wp


def window_reverse(windows: Tensor, window_size: int, Hp: int, Wp: int, H: int, W: int) -> Tensor:
    B = windows.shape[0] // ((Hp // window_size) * (Wp // window_size))
    C = windows.shape[-1]
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, C)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, C, Hp, Wp)
    return x[:, :, :H, :W]


class WindowAttention(nn.Module):
    """Windowed Multi-Head Self-Attention (W-MSA) with Relative Position Bias."""

    def __init__(self, dim: int, window_size: int = 8, num_heads: int = 4) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(x)


class SwinBlock(nn.Module):
    """Swin Transformer Block for 2D Feature Maps."""

    def __init__(self, dim: int, num_heads: int = 4, window_size: int = 8) -> None:
        super().__init__()
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=window_size, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        shortcut = x

        windows, Hp, Wp = window_partition(x, self.window_size)
        windows_norm = self.norm1(windows)
        attn_windows = self.attn(windows_norm)
        x_attn = window_reverse(attn_windows, self.window_size, Hp, Wp, H, W)

        x = shortcut + x_attn
        B, C, H, W = x.shape
        x_perm = x.permute(0, 2, 3, 1).contiguous()
        x_mlp = self.mlp(self.norm2(x_perm)).permute(0, 3, 1, 2).contiguous()
        return x + x_mlp


class PixelShuffleUpsample(nn.Module):
    """Sub-pixel Convolution (PixelShuffle) 2x Upsampler."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1)
        self.up = nn.PixelShuffle(2)

    def forward(self, x: Tensor) -> Tensor:
        return self.up(self.conv(x))


class SwinTransformerDecoder(nn.Module):
    """State-of-the-Art Swin Transformer Synthesis Transform g_s for RSIC."""

    def __init__(
        self,
        N: int = 160,
        M: int = 256,
        window_size: int = 8,
        activation: str = "leaky_relu",
        clamp_output: bool = True,
    ) -> None:
        super().__init__()
        c0 = N
        c1 = N
        c2 = max(96, N // 2)
        c3 = 64

        self.stem = nn.Conv2d(M, c0, kernel_size=3, padding=1)

        self.stage1 = nn.Sequential(
            SwinBlock(c0, num_heads=4, window_size=window_size),
            PixelShuffleUpsample(c0, c1),
            make_activation(activation),
        )
        self.stage2 = nn.Sequential(
            SwinBlock(c1, num_heads=4, window_size=window_size),
            PixelShuffleUpsample(c1, c2),
            make_activation(activation),
        )
        self.stage3 = nn.Sequential(
            SwinBlock(c2, num_heads=3, window_size=window_size),
            PixelShuffleUpsample(c2, c3),
            make_activation(activation),
        )
        self.stage4 = nn.Sequential(
            SwinBlock(c3, num_heads=2, window_size=window_size),
            PixelShuffleUpsample(c3, 32),
            make_activation(activation),
        )
        self.to_rgb = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        self.clamp_output = bool(clamp_output)

    def forward(self, y_hat: Tensor) -> Tensor:
        x = self.stem(y_hat)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x_hat = self.to_rgb(x)
        if self.clamp_output:
            x_hat = torch.sigmoid(x_hat)
        return x_hat


def build_decoder(
    decoder_type: str = "standard",
    N: int = 160,
    M: int = 256,
    decoder_channels: int = 160,
    decoder_res_blocks: int = 1,
    refinement_blocks: int = 1,
    activation: str = "leaky_relu",
    clamp_output: bool = True,
) -> nn.Module:
    if decoder_type == "standard":
        return Decoder(
            N=N,
            M=M,
            decoder_channels=decoder_channels,
            decoder_res_blocks=decoder_res_blocks,
            refinement_blocks=refinement_blocks,
            activation=activation,
            clamp_output=clamp_output,
        )
    if decoder_type == "cheng2020_attention":
        return Cheng2020AttentionDecoder(
            N=N,
            M=M,
            activation=activation,
            clamp_output=clamp_output,
        )
    if decoder_type in {"swin", "transformer"}:
        return SwinTransformerDecoder(
            N=N,
            M=M,
            window_size=8,
            activation=activation,
            clamp_output=clamp_output,
        )
    raise ValueError(f"unknown decoder_type={decoder_type!r}")


class RSIC(nn.Module):
    """OpenRSIC Mean-Scale Hyperprior Neural Network Codec."""

    supports_cnz_v4 = False

    def __init__(
        self,
        model_variant: str = MODEL_VARIANT_RSIC,
        activation: str | None = None,
        decoder_activation: str = "leaky_relu",
        clamp_decoder_output: bool = True,
        decoder_type: str = "swin",
        qat: QATSettings | None = None,
    ) -> None:
        super().__init__()
        variant = normalize_model_variant(model_variant)
        config = get_model_config(variant)
        if config.model_type != "mean_scale_hyperprior":
            raise ValueError(f"{variant} is not a mean-scale hyperprior config")
        if activation is None:
            activation = config.activation
        if config.Z is None:
            raise ValueError(f"{variant} requires config.Z")

        self.model_variant = variant
        self.model_name = config.name
        self.config = config
        self.N = config.N
        self.M = config.M
        self.Z = int(config.Z)
        self.decoder_channels = config.decoder_channels
        self.downsampling_factor = 2**4
        self.scale_min = float(config.scale_min)
        self.scale_max = float(config.scale_max)
        self.latent_clip = config.latent_clip
        self.z_clip = config.z_clip
        self.qat = qat if qat is not None else QATSettings()

        self.encoder = QuantFriendlyResidualEncoder(
            N=self.N,
            M=self.M,
            activation=activation,
            latent_clip=config.latent_clip,
            signed_latent=config.signed_latent,
        )
        self.hyper_encoder = HyperEncoder(
            M=self.M,
            N=self.N,
            Z=self.Z,
            activation=activation,
            z_clip=config.z_clip,
        )
        self.entropy_bottleneck_z = NanoEntropyBottleneck(
            channels=self.Z,
            quant_step=1.0,
        )
        self.hyper_decoder = HyperMeanScaleDecoder(
            Z=self.Z,
            N=self.N,
            M=self.M,
            activation=activation,
            scale_min=config.scale_min,
            scale_max=config.scale_max,
        )
        self.conditional_entropy_y = GaussianConditionalEntropy(
            quant_step=config.quant_step,
            scale_min=config.scale_min,
            scale_max=config.scale_max,
        )
        self.decoder = build_decoder(
            decoder_type=decoder_type,
            N=self.N,
            M=self.M,
            decoder_channels=config.decoder_channels,
            decoder_res_blocks=config.decoder_res_blocks,
            refinement_blocks=config.refinement_blocks,
            activation=decoder_activation,
            clamp_output=clamp_decoder_output,
        )

        init_module(self.encoder)
        init_module(self.hyper_encoder)
        init_module(self.hyper_decoder)
        init_module(self.decoder)



    @property
    def g_a(self) -> QuantFriendlyResidualEncoder:
        return self.encoder

    @property
    def h_a(self) -> HyperEncoder:
        return self.hyper_encoder

    @property
    def h_s(self) -> HyperMeanScaleDecoder:
        return self.hyper_decoder

    @property
    def g_s(self) -> Decoder:
        return self.decoder

    def model_config_dict(self) -> dict[str, Any]:
        return model_config_to_dict(self.config)

    def set_qat_settings(self, qat: QATSettings) -> None:
        self.qat = qat

    def set_quant_step(self, quant_step: float) -> None:
        self.conditional_entropy_y.quant_step.fill_(float(quant_step))

    def get_quant_step(self) -> float:
        return float(self.conditional_entropy_y.quant_step.detach().cpu())

    def analysis_transform(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        y = self.encoder(x)
        z = self.hyper_encoder(y)
        z_hat, _ = self.entropy_bottleneck_z(z, training=False)
        scales_y, means_y = self.hyper_decoder(z_hat)
        return y, z, scales_y, means_y

    def _maybe_fake_quant_latent(self, y: Tensor) -> tuple[Tensor, Tensor]:
        if not self.qat.enable_latent_fake_quant:
            return y, y.new_zeros(())
        y_q = fake_quant_symmetric_ste(
            y,
            bits=self.qat.latent_fake_quant_bits,
            clip=self.qat.latent_fake_quant_clip,
        )
        return y_q, torch.mean(torch.abs(y_q.detach() - y.detach()))

    def _maybe_fake_quant_z(self, z: Tensor) -> tuple[Tensor, Tensor]:
        if not self.qat.enable_z_fake_quant:
            return z, z.new_zeros(())
        z_q = fake_quant_symmetric_ste(
            z,
            bits=self.qat.z_fake_quant_bits,
            clip=self.qat.z_fake_quant_clip,
        )
        return z_q, torch.mean(torch.abs(z_q.detach() - z.detach()))

    def _maybe_fake_quant_scale(self, scales: Tensor) -> tuple[Tensor, Tensor]:
        if not self.qat.enable_scale_fake_quant:
            return scales, scales.new_zeros(())
        scales_q = fake_quant_positive_ste(
            scales,
            bits=self.qat.scale_fake_quant_bits,
            clip=self.qat.scale_fake_quant_clip,
        ).clamp(self.scale_min, self.scale_max)
        return scales_q, torch.mean(torch.abs(scales_q.detach() - scales.detach()))

    def forward(self, x: Tensor) -> dict[str, Any]:
        y = self.encoder(x)
        y_for_hyper, fq_y_error = self._maybe_fake_quant_latent(y)
        z = self.hyper_encoder(y_for_hyper)
        z_for_entropy, fq_z_error = self._maybe_fake_quant_z(z)
        z_hat, z_likelihoods = self.entropy_bottleneck_z(z_for_entropy)
        scales_y, means_y = self.hyper_decoder(z_hat)
        scales_y, fq_scale_error = self._maybe_fake_quant_scale(scales_y)
        y_hat, y_likelihoods = self.conditional_entropy_y(y_for_hyper, scales_y, means_y)
        x_hat = self.decoder(y_hat)
        return {
            "x_hat": x_hat,
            "y": y,
            "y_for_hyper": y_for_hyper,
            "y_hat": y_hat,
            "z": z,
            "z_for_entropy": z_for_entropy,
            "z_hat": z_hat,
            "scales_y": scales_y,
            "means_y": means_y,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "symbols": {
                "y": self.conditional_entropy_y.quantize(y_for_hyper, means_y).detach(),
                "z": self.entropy_bottleneck_z.quantize(z_for_entropy).detach(),
            },
            "fake_quant_errors": {
                "y": fq_y_error,
                "z": fq_z_error,
                "scale": fq_scale_error,
            },
            "quant_step": self.conditional_entropy_y.quant_step,
            "latent_clip": self.latent_clip,
            "z_clip": self.z_clip,
            "scale_min_value": self.scale_min,
            "scale_max_value": self.scale_max,
            "model_variant": self.model_variant,
        }

    @torch.no_grad()
    def compress(self, x: Tensor) -> dict[str, Any]:
        """Compress input image tensor [1, 3, H, W] into integer symbols & zlib byte streams."""
        import zlib

        y = self.encoder(x)
        z = self.hyper_encoder(y)
        z_sym = torch.round(z).to(torch.int16)
        z_hat = z_sym.to(dtype=y.dtype)

        scales_y, means_y = self.hyper_decoder(z_hat)
        step = self.conditional_entropy_y.quant_step
        y_sym = torch.round((y - means_y) / step).to(torch.int16)

        z_bytes = zlib.compress(z_sym.cpu().numpy().tobytes(), level=6)
        y_bytes = zlib.compress(y_sym.cpu().numpy().tobytes(), level=6)

        return {
            "z_bytes": z_bytes,
            "y_bytes": y_bytes,
            "z_shape": list(z.shape),
            "y_shape": list(y.shape),
            "quant_step": float(step.item()),
        }

    @torch.no_grad()
    def decompress(self, payload: dict[str, Any], device: torch.device | str | None = None) -> Tensor:
        """Decompress zlib byte streams back to reconstructed image tensor [1, 3, H, W]."""
        import zlib
        import numpy as np

        if device is None:
            device = next(self.parameters()).device

        z_shape = tuple(payload["z_shape"])
        y_shape = tuple(payload["y_shape"])
        quant_step = float(payload["quant_step"])

        z_raw = zlib.decompress(payload["z_bytes"])
        z_sym = torch.from_numpy(np.frombuffer(z_raw, dtype=np.int16).reshape(z_shape)).to(device=device, dtype=torch.float32)

        scales_y, means_y = self.hyper_decoder(z_sym)

        y_raw = zlib.decompress(payload["y_bytes"])
        y_sym = torch.from_numpy(np.frombuffer(y_raw, dtype=np.int16).reshape(y_shape)).to(device=device, dtype=torch.float32)

        y_hat = y_sym * quant_step + means_y
        x_hat = self.decoder(y_hat)
        if hasattr(self.decoder, "clamp_output") and self.decoder.clamp_output:
            x_hat = torch.clamp(x_hat, 0.0, 1.0)
        return x_hat


NanoHyperMeanScaleQ = RSIC


def get_model(
    model_variant: str | None = None,
    activation: str | None = None,
    decoder_activation: str = "leaky_relu",
    clamp_decoder_output: bool = True,
    decoder_type: str = "swin",
    qat: QATSettings | None = None,
) -> nn.Module:
    variant = normalize_model_variant(model_variant)
    return RSIC(
        model_variant=variant,
        activation=activation,
        decoder_activation=decoder_activation,
        clamp_decoder_output=clamp_decoder_output,
        decoder_type=decoder_type,
        qat=qat,
    )

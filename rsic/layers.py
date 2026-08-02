from __future__ import annotations

import torch.nn as nn


def conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 5,
    stride: int = 2,
    bias: bool = True,
) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
        bias=bias,
    )


def deconv(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 5,
    stride: int = 2,
    bias: bool = True,
) -> nn.ConvTranspose2d:
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
        output_padding=stride - 1,
        bias=bias,
    )


def make_activation(name: str = "relu", inplace: bool = True) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=inplace)
    if name == "relu6":
        return nn.ReLU6(inplace=inplace)
    if name in {"leaky_relu", "leaky-relu", "lrelu"}:
        return nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
    raise ValueError(f"Unsupported activation: {name}")


def init_module(module: nn.Module) -> None:
    for layer in module.modules():
        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

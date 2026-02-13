# Copyright (C) 2026 Li Auto Inc. All Rights Reserved.
"""Decoder blocks for MetricAnything DepthMap."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


class MultiresConvDecoder(nn.Module):
    """Fuse multi-resolution encoder features."""

    def __init__(self, dims_encoder: Iterable[int], dim_decoder: int) -> None:
        super().__init__()
        self.dims_encoder = list(dims_encoder)
        self.dim_decoder = dim_decoder

        num_encoders = len(self.dims_encoder)

        in_dims = (
            [self.dims_encoder[-3]]
            + [self.dims_encoder[-4]] * 4
            + [self.dims_encoder[-2], self.dims_encoder[-1]]
        )
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(dim_in, dim_decoder, kernel_size=3, stride=1, padding=1, bias=False)
                for dim_in in in_dims
            ]
        )

        deconv_flags = [False, True, False, False, True, True, True]
        self.fusions = nn.ModuleList(
            [
                FeatureFusionBlock2d(
                    num_features=dim_decoder,
                    deconv=deconv_flags[i],
                    batch_norm=False,
                    disable_resnet1=(i == num_encoders - 1),
                )
                for i in range(num_encoders)
            ]
        )

    def forward(self, encodings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode the multi-resolution encodings."""
        num_levels = len(encodings)
        num_encoders = len(self.dims_encoder)
        if num_levels != num_encoders:
            raise ValueError(
                f"Got encoder output levels={num_levels}, expected levels={num_encoders}."
            )

        encodings_forward_ids = [4, 3, 2, 1, 0, 5, 6]

        features = self.convs[-1](encodings[-1])
        lowres_features = features

        features = self.fusions[-1](features, None)

        for i in range(num_levels - 2, -1, -1):
            features_i = self.convs[i](encodings[encodings_forward_ids[i]])
            features = self.fusions[i](features, features_i)

        return features, lowres_features


class ResidualBlock(nn.Module):
    """Generic residual block (He et al., 2016)."""

    def __init__(self, residual: nn.Module, shortcut: nn.Module | None = None) -> None:
        super().__init__()
        self.residual = residual
        self.shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta_x = self.residual(x)
        if self.shortcut is not None:
            x = self.shortcut(x)
        return x + delta_x


class FeatureFusionBlock2d(nn.Module):
    """Feature fusion with residual refinement and optional upsampling."""

    def __init__(
        self,
        num_features: int,
        deconv: bool = False,
        batch_norm: bool = False,
        disable_resnet1: bool = False,
    ) -> None:
        super().__init__()

        self.resnet1 = nn.Identity() if disable_resnet1 else self._residual_block(num_features, batch_norm)
        self.resnet2 = self._residual_block(num_features, batch_norm)

        self.use_deconv = deconv
        if deconv:
            self.deconv = nn.ConvTranspose2d(
                in_channels=num_features,
                out_channels=num_features,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=False,
            )

        self.out_conv = nn.Conv2d(
            num_features,
            num_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x0: torch.Tensor, x1: torch.Tensor | None = None) -> torch.Tensor:
        x = x0
        if x1 is not None:
            x1_res = self.resnet1(x1)
            x = self.skip_add.add(x, x1_res)

        x = self.resnet2(x)
        if self.use_deconv:
            x = self.deconv(x)
        return self.out_conv(x)

    @staticmethod
    def _residual_block(num_features: int, batch_norm: bool) -> ResidualBlock:
        def _create_block(dim: int, batch_norm: bool) -> list[nn.Module]:
            layers: list[nn.Module] = [
                nn.ReLU(False),
                nn.Conv2d(
                    dim,
                    dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=not batch_norm,
                ),
            ]
            if batch_norm:
                layers.append(nn.BatchNorm2d(dim))
            return layers

        residual = nn.Sequential(
            *_create_block(dim=num_features, batch_norm=batch_norm),
            *_create_block(dim=num_features, batch_norm=batch_norm),
        )
        return ResidualBlock(residual)


__all__ = ["MultiresConvDecoder", "FeatureFusionBlock2d", "ResidualBlock"]

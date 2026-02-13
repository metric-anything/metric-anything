# Copyright (C) 2026 Li Auto Inc. All Rights Reserved.
"""MetricAnything DepthMap encoder."""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class MetricAnythingEncoder(nn.Module):
    """Multi-resolution encoder using a ViT patch backbone."""

    def __init__(
        self,
        dims_encoder: Iterable[int],
        patch_encoder: nn.Module,
        hook_block_ids: Iterable[int],
    ) -> None:
        super().__init__()

        self.dims_encoder = list(dims_encoder)
        self.patch_encoder = patch_encoder
        self.hook_block_ids = list(hook_block_ids)

        embed_dim = patch_encoder.embed_dim
        patch_size = patch_encoder.patch_embed.patch_size[0]
        self.out_size = int(patch_encoder.patch_embed.img_size[0] // patch_size)

        self.upsample_latent0 = self._project_upsample(embed_dim, self.dims_encoder[0], upsample_layers=1)
        self.upsample_latent1 = self._project_upsample(embed_dim, self.dims_encoder[0], upsample_layers=2)
        self.upsample_latent2 = self._project_upsample(embed_dim, self.dims_encoder[0], upsample_layers=2)
        self.upsample_latent3 = self._project_upsample(embed_dim, self.dims_encoder[0], upsample_layers=2)

        self.upsample0 = self._project_upsample(embed_dim, self.dims_encoder[1], upsample_layers=3)
        self.upsample1 = self._project_upsample(embed_dim, self.dims_encoder[2], upsample_layers=1)
        self.upsample2 = self._project_upsample(embed_dim, self.dims_encoder[3], upsample_layers=1)

        self.upsample_lowres = nn.ConvTranspose2d(
            in_channels=embed_dim,
            out_channels=self.dims_encoder[3],
            kernel_size=2,
            stride=2,
            padding=0,
            bias=True,
        )
        self.fuse_lowres = nn.Conv2d(
            in_channels=self.dims_encoder[3] * 2,
            out_channels=self.dims_encoder[3],
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.patch_encoder.blocks[self.hook_block_ids[0]].register_forward_hook(self._hook0)
        self.patch_encoder.blocks[self.hook_block_ids[1]].register_forward_hook(self._hook1)
        self.patch_encoder.blocks[self.hook_block_ids[2]].register_forward_hook(self._hook2)
        self.patch_encoder.blocks[self.hook_block_ids[3]].register_forward_hook(self._hook3)

    @staticmethod
    def _project_upsample(
        dim_in: int,
        dim_out: int,
        upsample_layers: int,
        dim_int: int | None = None,
    ) -> nn.Sequential:
        if dim_int is None:
            dim_int = dim_out

        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels=dim_in,
                out_channels=dim_int,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
        ]
        layers += [
            nn.ConvTranspose2d(
                in_channels=dim_int if i == 0 else dim_out,
                out_channels=dim_out,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=False,
            )
            for i in range(upsample_layers)
        ]
        return nn.Sequential(*layers)

    def _hook0(self, _module, _input, output) -> None:
        self.backbone_highres_hook0 = output

    def _hook1(self, _module, _input, output) -> None:
        self.backbone_highres_hook1 = output

    def _hook2(self, _module, _input, output) -> None:
        self.backbone_highres_hook2 = output

    def _hook3(self, _module, _input, output) -> None:
        self.backbone_highres_hook3 = output

    @property
    def img_size(self) -> int:
        """Network input resolution (typically 1536)."""
        return self.patch_encoder.patch_embed.img_size[0] * 4

    def _create_pyramid(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create a 3-level image pyramid."""
        x0 = x
        x1 = F.interpolate(x, size=None, scale_factor=0.5, mode="bilinear", align_corners=False)
        x2 = F.interpolate(x, size=None, scale_factor=0.25, mode="bilinear", align_corners=False)
        return x0, x1, x2

    def split(self, x: torch.Tensor, overlap_ratio: float = 0.25) -> torch.Tensor:
        """Split the input into overlapped 384x384 patches."""
        patch_size = 384
        patch_stride = int(patch_size * (1 - overlap_ratio))

        image_size = x.shape[-1]
        steps = int(math.ceil((image_size - patch_size) / patch_stride)) + 1

        patches = []
        for j in range(steps):
            j0 = j * patch_stride
            j1 = j0 + patch_size
            for i in range(steps):
                i0 = i * patch_stride
                i1 = i0 + patch_size
                patches.append(x[..., j0:j1, i0:i1])

        return torch.cat(patches, dim=0)

    def merge(self, x: torch.Tensor, batch_size: int, padding: int = 3) -> torch.Tensor:
        """Merge overlapped patches back to a full feature map."""
        steps = int(math.sqrt(x.shape[0] // batch_size))

        idx = 0
        rows = []
        for j in range(steps):
            cols = []
            for i in range(steps):
                patch = x[batch_size * idx : batch_size * (idx + 1)]

                if j != 0:
                    patch = patch[..., padding:, :]
                if i != 0:
                    patch = patch[..., :, padding:]
                if j != steps - 1:
                    patch = patch[..., :-padding, :]
                if i != steps - 1:
                    patch = patch[..., :, :-padding]

                cols.append(patch)
                idx += 1

            rows.append(torch.cat(cols, dim=-1))

        return torch.cat(rows, dim=-2)

    @staticmethod
    def reshape_feature(embeddings: torch.Tensor, width: int, height: int, cls_token_offset: int = 1) -> torch.Tensor:
        """Discard class token and reshape 1D tokens to a 2D feature map."""
        batch, tokens, channels = embeddings.shape
        if cls_token_offset > 0:
            embeddings = embeddings[:, cls_token_offset:, :]

        return embeddings.reshape(batch, height, width, channels).permute(0, 3, 1, 2)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Encode input at multiple resolutions."""
        batch_size = x.shape[0]

        x0, x1, x2 = self._create_pyramid(x)

        x0_patches = self.split(x0, overlap_ratio=0.25)
        x1_patches = self.split(x1, overlap_ratio=0.5)
        x2_patches = x2

        x_pyramid_patches = torch.cat((x0_patches, x1_patches, x2_patches), dim=0)
        x_pyramid_encodings = self.patch_encoder(x_pyramid_patches)
        x_pyramid_encodings = self.reshape_feature(
            x_pyramid_encodings, self.out_size, self.out_size, cls_token_offset=1
        )

        if isinstance(self.backbone_highres_hook0, list):
            self.backbone_highres_hook0 = self.backbone_highres_hook0[0]
            self.backbone_highres_hook1 = self.backbone_highres_hook1[0]
            self.backbone_highres_hook2 = self.backbone_highres_hook2[0]
            self.backbone_highres_hook3 = self.backbone_highres_hook3[0]

        high_patch_count = x0_patches.shape[0]

        x_latent0_features = self.merge(
            self.reshape_feature(
                self.backbone_highres_hook0,
                self.out_size,
                self.out_size,
                cls_token_offset=5,
            )[:high_patch_count],
            batch_size=batch_size,
            padding=3,
        )
        x_latent1_features = self.merge(
            self.reshape_feature(
                self.backbone_highres_hook1,
                self.out_size,
                self.out_size,
                cls_token_offset=5,
            )[:high_patch_count],
            batch_size=batch_size,
            padding=3,
        )
        x_latent2_features = self.merge(
            self.reshape_feature(
                self.backbone_highres_hook2,
                self.out_size,
                self.out_size,
                cls_token_offset=5,
            )[:high_patch_count],
            batch_size=batch_size,
            padding=3,
        )
        x_latent3_features = self.merge(
            self.reshape_feature(
                self.backbone_highres_hook3,
                self.out_size,
                self.out_size,
                cls_token_offset=5,
            )[:high_patch_count],
            batch_size=batch_size,
            padding=3,
        )

        x0_encodings, x1_encodings, x2_encodings = torch.split(
            x_pyramid_encodings,
            [len(x0_patches), len(x1_patches), len(x2_patches)],
            dim=0,
        )

        x0_features = self.merge(x0_encodings, batch_size=batch_size, padding=3)
        x1_features = self.merge(x1_encodings, batch_size=batch_size, padding=6)
        x2_features = x2_encodings

        x_global_features = x2_features.clone()

        x_latent0_features = self.upsample_latent0(x_latent0_features)
        x_latent1_features = self.upsample_latent1(x_latent1_features)
        x_latent2_features = self.upsample_latent2(x_latent2_features)
        x_latent3_features = self.upsample_latent3(x_latent3_features)

        x0_features = self.upsample0(x0_features)
        x1_features = self.upsample1(x1_features)
        x2_features = self.upsample2(x2_features)

        x_global_features = self.upsample_lowres(x_global_features)
        x_global_features = self.fuse_lowres(torch.cat((x2_features, x_global_features), dim=1))

        return [
            x_latent0_features,
            x_latent1_features,
            x_latent2_features,
            x_latent3_features,
            x0_features,
            x1_features,
            x_global_features,
        ]


__all__ = ["MetricAnythingEncoder"]

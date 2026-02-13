# Copyright (C) 2026 Li Auto Inc. All Rights Reserved.
"""Vision Transformer factory for MetricAnything DepthMap."""

from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
from typing import Dict, List, Literal

import torch
import torch.nn as nn


ViTPreset = Literal[
    "dinov3_vith16plus_224",
]


@dataclass(frozen=True)
class ViTConfig:
    """Minimal ViT config needed by the depth encoder."""

    in_chans: int
    embed_dim: int
    img_size: int = 384
    encoder_feature_layer_ids: List[int] | None = None
    encoder_feature_dims: List[int] | None = None


VIT_CONFIG_DICT: Dict[ViTPreset, ViTConfig] = {
    "dinov3_vith16plus_224": ViTConfig(
        in_chans=3,
        embed_dim=1280,
        img_size=384,
        encoder_feature_layer_ids=[7, 13, 19, 25],
        encoder_feature_dims=[320, 256, 320, 640],
    )
}


def create_vit(preset: ViTPreset) -> nn.Module:
    """Create and configure a ViT backbone."""
    config = VIT_CONFIG_DICT[preset]
    img_size = (config.img_size, config.img_size)

    if preset.startswith("dinov3"):
        model = torch.hub.load(
            "network",
            "dinov3_vith16plus",
            source="local",
            pretrained=False,
            weights=None,
            map_location=torch.device("cpu"),
        )

        model.patch_embed.img_size = img_size

        original_forward_features = model.forward_features

        def forward_features_with_cls(self, x):
            feats = original_forward_features(x)
            cls_token = feats["x_norm_clstoken"].unsqueeze(1)
            patch_tokens = feats["x_norm_patchtokens"]
            return torch.cat([cls_token, patch_tokens], dim=1)

        model.forward_features = MethodType(forward_features_with_cls, model)
    else:
        raise NotImplementedError(f"Unknown model preset: '{preset}'")

    # Match the minimal attributes expected by the encoder.
    model.start_index = 1
    model.patch_size = model.patch_embed.patch_size
    model.is_vit = True
    model.forward = model.forward_features

    return model


__all__ = ["ViTPreset", "ViTConfig", "VIT_CONFIG_DICT", "create_vit"]

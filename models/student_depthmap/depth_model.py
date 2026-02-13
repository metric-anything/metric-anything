# Copyright (C) 2026 Li Auto Inc. All Rights Reserved.
"""MetricAnything DepthMap model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Dict, Mapping, Optional, Union

import torch
from torch import nn

from network.decoder import MultiresConvDecoder
from network.encoder import MetricAnythingEncoder
from network.vit_factory import VIT_CONFIG_DICT, ViTConfig, ViTPreset, create_vit


@dataclass(frozen=True)
class MetricAnythingConfig:
    """Configuration for MetricAnything DepthMap."""

    patch_encoder_preset: ViTPreset
    decoder_features: int


DEFAULT_CONFIG = MetricAnythingConfig(
    patch_encoder_preset="dinov3_vith16plus_224",
    decoder_features=256,
)


def _create_backbone(preset: ViTPreset) -> tuple[nn.Module, ViTConfig]:
    """Load a ViT backbone and its preset config."""
    if preset not in VIT_CONFIG_DICT:
        raise KeyError(f"Unknown ViT preset: {preset}")
    return create_vit(preset), VIT_CONFIG_DICT[preset]


def create_model(
    config: MetricAnythingConfig = DEFAULT_CONFIG,
    device: torch.device | str | int = "cpu",
) -> "MetricAnythingDepthMap":
    """Build the MetricAnything DepthMap model."""
    patch_encoder, patch_cfg = _create_backbone(config.patch_encoder_preset)

    encoder = MetricAnythingEncoder(
        dims_encoder=patch_cfg.encoder_feature_dims,
        patch_encoder=patch_encoder,
        hook_block_ids=patch_cfg.encoder_feature_layer_ids,
    )

    decoder_dims = (
        [config.decoder_features]
        + [encoder.dims_encoder[0]] * 2
        + list(encoder.dims_encoder)
    )
    decoder = MultiresConvDecoder(dims_encoder=decoder_dims, dim_decoder=config.decoder_features)

    return MetricAnythingDepthMap(
        encoder=encoder,
        decoder=decoder,
        last_dims=(32, 1),
    ).to(device)


class MetricAnythingDepthMap(nn.Module):
    """MetricAnything DepthMap network."""

    def __init__(
        self,
        encoder: MetricAnythingEncoder,
        decoder: MultiresConvDecoder,
        last_dims: tuple[int, int],
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.head = self._build_head(decoder.dim_decoder, last_dims)

        # Initialize the final conv bias for stable depth scaling.
        self.head[-2].bias.data.fill_(0)

    @staticmethod
    def _build_head(dim_decoder: int, last_dims: tuple[int, int]) -> nn.Sequential:
        layers: list[nn.Module] = [
            nn.Conv2d(dim_decoder, dim_decoder // 2, kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose2d(
                in_channels=dim_decoder // 2,
                out_channels=dim_decoder // 2,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=True,
            ),
            nn.Conv2d(dim_decoder // 2, last_dims[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        ]

        # Extra refinement layers at the final resolution.
        for _ in range(4):
            layers += [
                nn.Conv2d(last_dims[0], last_dims[0], kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
            ]

        layers += [
            nn.Conv2d(last_dims[0], last_dims[1], kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        ]
        return nn.Sequential(*layers)

    @property
    def img_size(self) -> int:
        """Network input resolution."""
        return self.encoder.img_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict canonical inverse depth at the network resolution."""
        _, _, height, width = x.shape
        assert height == self.img_size and width == self.img_size

        encodings = self.encoder(x)
        features, _ = self.decoder(encodings)
        return self.head(features)

    @torch.no_grad()
    def infer(
        self,
        x: torch.Tensor,
        f_px: float | torch.Tensor | None = None,
        interpolation_mode: str = "bilinear",
    ) -> Mapping[str, torch.Tensor]:
        """Infer metric depth for an input image tensor."""
        if x.ndim == 3:
            x = x.unsqueeze(0)

        _, _, height, width = x.shape
        resize = height != self.img_size or width != self.img_size

        if resize:
            x = nn.functional.interpolate(
                x,
                size=(self.img_size, self.img_size),
                mode=interpolation_mode,
                align_corners=False,
            )

        canonical_inverse_depth = self.forward(x)
        if f_px is None:
            f_px = 1000

        inverse_depth = canonical_inverse_depth * (width / f_px)

        if resize:
            inverse_depth = nn.functional.interpolate(
                inverse_depth,
                size=(height, width),
                mode=interpolation_mode,
                align_corners=False,
            )

        depth = 1.0 / torch.clamp(inverse_depth, min=1e-3, max=1e3)
        return {"depth": depth.squeeze()}

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path, IO[bytes]],
        model_kwargs: Optional[Dict[str, Any]] = None,
        **hf_kwargs: Any,
    ) -> "MetricAnythingDepthMap":
        """Load weights from a local path or a Hugging Face Hub repo."""
        model_kwargs = dict(model_kwargs or {})

        config = model_kwargs.pop("config", DEFAULT_CONFIG)
        device = model_kwargs.pop("device", "cpu")
        strict = model_kwargs.pop("strict", True)
        weights_only = model_kwargs.pop("weights_only", True)

        def _resolve_map_location(value: Any) -> torch.device | str:
            if isinstance(value, int):
                return torch.device(f"cuda:{value}") if torch.cuda.is_available() else torch.device("cpu")
            return value

        map_location = _resolve_map_location(model_kwargs.pop("map_location", device))

        if isinstance(pretrained_model_name_or_path, (str, Path)):
            path = Path(pretrained_model_name_or_path)
            if path.exists():
                checkpoint_path = path
            else:
                try:
                    from huggingface_hub import hf_hub_download
                except ImportError as exc:
                    raise ImportError(
                        "huggingface_hub is required for loading from the Hub. "
                        "Install it with `pip install huggingface_hub`."
                    ) from exc

                filename = hf_kwargs.pop("filename", "model.pt")
                checkpoint_path = hf_hub_download(
                    repo_id=str(pretrained_model_name_or_path),
                    repo_type="model",
                    filename=filename,
                    **hf_kwargs,
                )

            checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=weights_only)
        else:
            checkpoint = torch.load(pretrained_model_name_or_path, map_location=map_location, weights_only=weights_only)

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        model = create_model(config=config, device=device)
        model.load_state_dict(checkpoint, strict=strict)
        return model


__all__ = [
    "MetricAnythingConfig",
    "DEFAULT_CONFIG",
    "MetricAnythingDepthMap",
    "create_model",
]

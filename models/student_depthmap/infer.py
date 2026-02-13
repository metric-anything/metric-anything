#!/usr/bin/env python3
"""Inference for MetricAnything DepthMap using from_pretrained."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import json
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import v2

from depth_model import MetricAnythingDepthMap, DEFAULT_CONFIG

LOGGER = logging.getLogger(__name__)


def colorize_depth(
    depth: np.ndarray,
    mask: np.ndarray | None = None,
    normalize: bool = True,
    cmap: str = "turbo_r",
    max_depth: float = 150.0,
) -> np.ndarray:
    """Colorize depth using inverse-depth and a 0–max_depth meter range; invalid/far pixels are white."""
    valid = np.isfinite(depth) & (depth > 0) & (depth <= max_depth)
    if mask is not None:
        valid &= mask
    if not np.any(valid):
        return np.full((*depth.shape, 3), 255, dtype=np.uint8)

    depth = np.where(valid, depth, np.nan)
    disp = 1.0 / depth
    if normalize:
        min_disp, max_disp = np.nanquantile(disp, 0.001), np.nanquantile(disp, 0.99)
        disp = (disp - min_disp) / (max_disp - min_disp) if max_disp > min_disp else disp * 0.0
    colored = np.nan_to_num(matplotlib.colormaps[cmap](1.0 - disp)[..., :3], 0.0)
    colored = (colored.clip(0.0, 1.0) * 255).astype(np.uint8)
    colored[~valid] = 255
    return np.ascontiguousarray(colored)


def make_transform() -> v2.Compose:
    """Build the image preprocessing pipeline."""
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def load_rgb(rgb_path: Path) -> Image.Image:
    """Load an RGB image from disk with basic sanity checks."""
    if not os.path.exists(rgb_path):
        raise FileNotFoundError(f"RGB image does not exist: {rgb_path}")
    try:
        image = Image.open(rgb_path)
        if image.size[0] == 0 or image.size[1] == 0:
            raise ValueError(f"{rgb_path}: invalid image size: {image.size}")
        if image.mode != "RGB":
            image = image.convert("RGB")
        if image.getbbox() is None:
            raise ValueError(f"{rgb_path}: image data is empty or corrupted")
        return image
    except Exception as exc:
        raise RuntimeError(f"Failed to load image {rgb_path}: {exc}") from exc


def save_plt(save_path: Path, depth: np.ndarray, valid_mask: np.ndarray, save_name: str) -> None:
    """Save a matplotlib depth plot with a colorbar and min/max statistics."""
    import matplotlib.pyplot as plt

    valid_depth = depth[valid_mask]
    plt.figure(figsize=(6, 5))
    im = plt.imshow(depth, cmap="turbo")
    plt.colorbar(im)
    plt.title(f"D: min={np.min(valid_depth):.2f}, max={np.max(valid_depth):.2f}")
    plt.axis("off")
    plt.savefig(save_path / save_name, bbox_inches="tight", dpi=300)
    plt.close()


def _parse_device(value: str | None) -> str | int:
    if value is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return int(value) if value.isdigit() else value


def run(args: argparse.Namespace) -> None:
    """Run inference on a single image or directory."""
    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    device = _parse_device(args.device)
    model = MetricAnythingDepthMap.from_pretrained(
        args.pretrained,
        model_kwargs={
            "device": "cuda",
            "config": DEFAULT_CONFIG
        },
        filename='student_depthmap.pt',
    )
    model.eval()

    transform = make_transform()

    image_paths: Iterable[Path] = [args.image_path]
    relative_path = args.image_path.parent

    if args.image_path.is_dir():
        image_paths = list(args.image_path.glob("**/*"))
        include_suffices = ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']
        image_paths = [p for p in image_paths if p.is_file() and p.suffix in include_suffices]

    for image_path in tqdm(image_paths):
        try:
            LOGGER.info("Loading image %s ...", image_path)
            image = load_rgb(image_path)
        except Exception as exc:
            LOGGER.error("%s", exc)
            continue

        input_tensor = transform(image).unsqueeze(0).to(device)

        # Determine focal length in pixels (f_px) with the following priority:
        # 1) Explicit --f_px argument.
        # 2) JSON file with the same stem as the image, containing "cam_in" = [fx, fy, cx, cy].
        # 3) Fallback to image width (input_tensor.shape[-1]).
        if args.f_px is not None:
            f_px = args.f_px
        else:
            json_path = image_path.with_suffix(".json")
            if json_path.exists():
                try:
                    with open(json_path, "r") as f:
                        cam_intrinsic = json.load(f)
                    cam_in = cam_intrinsic.get("cam_in", None)
                    if isinstance(cam_in, (list, tuple)) and len(cam_in) >= 1:
                        f_px = float(cam_in[0])
                    else:
                        LOGGER.warning("Invalid cam_in format in %s, using image width as f_px.", json_path)
                        f_px = float(input_tensor.shape[-1])
                except Exception as exc:
                    LOGGER.warning("Failed to read intrinsics from %s: %s. Using image width as f_px.", json_path, exc)
                    f_px = float(input_tensor.shape[-1])
            else:
                f_px = float(input_tensor.shape[-1])

        prediction = model.infer(input_tensor, f_px=f_px)
        depth = prediction["depth"].detach().cpu().numpy().squeeze()

        if args.output_path is not None:
            output_file = args.output_path / image_path.relative_to(relative_path).parent / image_path.stem
            LOGGER.info("Saving depth map to: %s", output_file)
            output_file.mkdir(parents=True, exist_ok=True)

            LOGGER.info(f"{depth.max()=}, {depth.min()=}")
            np.save(str(output_file / 'depth.npy'), depth)

            image.save(str(output_file / "image.jpg"), quality=95)
            color_depth = colorize_depth(depth, max_depth=150.0)
            cv2.imwrite(str(output_file / "depth_vis.png"), cv2.cvtColor(color_depth, cv2.COLOR_RGB2BGR))

            mask = np.isfinite(depth) & (depth < 150.0)
            depth[~mask] = np.inf
            save_plt(output_file, depth, mask, "depth_pred_bar_vis.png")


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="MetricAnything DepthMap inference (PyTorch).")
    parser.add_argument("-i", "--image_path", type=Path, default=Path("./data/example.jpg"), help="Path to input image or directory.")
    parser.add_argument("-o", "--output_path", type=Path, help="Path to store output files.")
    parser.add_argument("--f_px", type=float, default=None, help="Focal length in pixels used for metric depth conversion.")
    parser.add_argument("--pretrained", type=str, default="yjh001/metricanything_student_depthmap", help="Local checkpoint path or Hub repo id.")
    parser.add_argument("--device", type=str, default=None, help="Device string or CUDA index.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging.")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
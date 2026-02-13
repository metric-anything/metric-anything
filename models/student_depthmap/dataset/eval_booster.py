"""Booster dataset loader for evaluation."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from glob import glob
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2


def build_transform(target_size: int) -> v2.Compose:
    """Default preprocessing for evaluation."""
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((target_size, target_size), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def _parse_opencv_matrix(node: ET.Element, rows: int, cols: int) -> np.ndarray:
    data_node = node.find("data")
    if data_node is None or data_node.text is None:
        raise ValueError("Missing <data> for OpenCV matrix")
    values = [float(x) for x in data_node.text.strip().split()]
    return np.array(values, dtype=np.float64).reshape(rows, cols)


def read_scene_calib(calib_path: str) -> Tuple[float, float, float, float, float]:
    """Read fx, fy, cx, cy, baseline from a Booster calibration file."""
    root = ET.parse(calib_path).getroot()

    mtxL = root.find("mtxL")
    if mtxL is None:
        raise ValueError(f"Missing mtxL in {calib_path}")

    K = _parse_opencv_matrix(mtxL, 3, 3)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    baseline_elem = root.find("baselineLR")
    if baseline_elem is not None and baseline_elem.text is not None:
        baseline = float(baseline_elem.text.strip())
    else:
        projR = root.find("proj_matR")
        if projR is None:
            raise ValueError(f"Missing baselineLR and proj_matR in {calib_path}")
        P = _parse_opencv_matrix(projR, 3, 4)
        Tx = float(P[0, 3])
        if fx == 0:
            raise ValueError("fx is zero; cannot derive baseline")
        baseline = -Tx / fx

    return fx, fy, cx, cy, baseline


class BoosterDataset(Dataset):
    """Booster dataset loader for metric depth evaluation."""

    def __init__(
        self,
        root: str | Path,
        target_size: int = 1536,
        transform: v2.Compose | None = None,
        convert_to_depth: bool = True,
        min_depth: float = 1e-3,
        max_depth: float = 10.0,
    ) -> None:
        self.root = Path(root)
        self.target_size = target_size
        self.transform = transform or build_transform(target_size)
        self.convert_to_depth = convert_to_depth
        self.min_depth = min_depth
        self.max_depth = max_depth

        self.samples: list[Dict[str, Any]] = []
        pattern = str(self.root / "balanced" / "*" / "camera_00" / "*.png")
        for image_path in sorted(glob(pattern)):
            image_path = Path(image_path)
            scene_root = image_path.parent.parent
            disp_path = scene_root / "disp_00.npy"
            calib_path = scene_root / "calib_00-02.xml"
            mask_path = scene_root / "mask_00.png"

            if not disp_path.is_file() or not calib_path.is_file():
                continue

            fx, fy, cx, cy, baseline = read_scene_calib(str(calib_path))
            self.samples.append(
                {
                    "image": image_path,
                    "disp": disp_path,
                    "mask": mask_path if mask_path.is_file() else None,
                    "fx": float(fx),
                    "fy": float(fy),
                    "cx": float(cx),
                    "cy": float(cy),
                    "baseline": float(baseline),
                }
            )

        if not self.samples:
            raise RuntimeError(f"No Booster samples found under {pattern}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.samples[idx]

        image = Image.open(record["image"]).convert("RGB")
        disp = np.load(record["disp"]).astype(np.float32)
        valid_mask = disp > 0

        if self.convert_to_depth:
            depth = (record["fx"] * record["baseline"]) / (disp + 1e-6) / 1000.0
        else:
            depth = disp

        if record["mask"] is not None:
            occ_mask = np.array(Image.open(record["mask"])).astype(np.uint8) != 0
        else:
            occ_mask = np.ones_like(valid_mask, dtype=bool)

        depth_tensor = torch.from_numpy(depth).float()
        valid_tensor = (
            (depth_tensor >= self.min_depth)
            & (depth_tensor <= self.max_depth)
            & torch.from_numpy(valid_mask)
            & torch.from_numpy(occ_mask)
        )

        orig_h, orig_w = depth_tensor.shape[-2], depth_tensor.shape[-1]
        canonical_image = self.transform(image)

        return {
            "image": canonical_image,
            "depth": depth_tensor,
            "valid_mask": valid_tensor,
            "f_px": torch.tensor(record["fx"], dtype=torch.float32),
            "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.int32),
            "rgb_path": str(record["image"]),
        }

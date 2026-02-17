"""
Depth Anything V2 – Metric Indoor Estimator
============================================
Thin wrapper around the HuggingFace transformers pipeline for the
Depth-Anything-V2-Metric-Indoor family of models.
"""

import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


# HuggingFace model IDs keyed by size name
_HF_MODEL_IDS = {
    "small": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    "base":  "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
}


class DAv2Estimator:
    """Depth Anything V2 – metric indoor depth estimator.

    Parameters
    ----------
    model_size : str
        One of ``"small"``, ``"base"``, ``"large"``.
    weights_dir : str | Path | None
        If given, load weights from this local directory instead of
        downloading from HuggingFace Hub.
    device : str
        PyTorch device string (``"cpu"``, ``"cuda"``, ``"mps"``).
    fp16 : bool
        Load model weights in float16 to halve memory.
    """

    def __init__(
        self,
        model_size: str = "small",
        weights_dir: str | Path | None = None,
        device: str = "cpu",
        fp16: bool = False,
    ):
        if model_size not in _HF_MODEL_IDS:
            raise ValueError(f"Unknown model_size={model_size!r}. Choose from {list(_HF_MODEL_IDS)}")

        self.model_size = model_size
        self.device = torch.device(device)
        self.dtype = torch.float16 if fp16 else torch.float32

        # Resolve model source – local directory or HF Hub
        if weights_dir is not None:
            model_path = Path(weights_dir) / f"dav2-{model_size}"
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Local weights not found at {model_path}. "
                    f"Run scripts/download_models.sh first."
                )
            source = str(model_path)
        else:
            source = _HF_MODEL_IDS[model_size]

        self.processor = AutoImageProcessor.from_pretrained(source)
        self.model = (
            AutoModelForDepthEstimation
            .from_pretrained(source, torch_dtype=self.dtype)
            .to(self.device)
        )
        self.model.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Return a full metric depth map (H, W) in metres.

        Parameters
        ----------
        frame_rgb : np.ndarray
            Input image as (H, W, 3) uint8 RGB array.

        Returns
        -------
        np.ndarray
            Depth map of shape (H, W) with values in metres.
        """
        pil_image = Image.fromarray(frame_rgb)
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        predicted_depth = outputs.predicted_depth  # (1, h_model, w_model)

        h, w = frame_rgb.shape[:2]
        depth = (
            torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )
        return depth

    def estimate_at_points(
        self,
        frame_rgb: np.ndarray,
        points: List[Tuple[int, int]],
    ) -> List[float]:
        """Return metric depth (metres) at specific pixel coordinates.

        Parameters
        ----------
        frame_rgb : np.ndarray
            Input image as (H, W, 3) uint8 RGB.
        points : list of (x, y) tuples
            Pixel coordinates to probe.

        Returns
        -------
        list of float
            Depth value at each requested point.
        """
        depth_map = self.estimate(frame_rgb)
        h, w = depth_map.shape
        results = []
        for x, y in points:
            # Clamp coordinates to [0, dim-1]
            x_safe = max(0, min(int(x), w - 1))
            y_safe = max(0, min(int(y), h - 1))
            results.append(float(depth_map[y_safe, x_safe]))
        return results

    def estimate_timed(self, frame_rgb: np.ndarray):
        """Like ``estimate`` but also returns elapsed seconds.

        Returns
        -------
        (depth_map, elapsed_s)
        """
        t0 = time.perf_counter()
        depth = self.estimate(frame_rgb)
        elapsed = time.perf_counter() - t0
        return depth, elapsed

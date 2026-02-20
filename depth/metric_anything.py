"""
MetricAnything – Metric Depth Estimator
=======================================
Thin wrapper around the MoGe student_pointmap model for metric depth.
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

# Enable MPS→CPU fallback for ops not yet supported on Apple Silicon
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Ensure the moge package is importable
_DEPTH_PKG_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _DEPTH_PKG_DIR.parent
_MOGE_PARENT = _REPO_ROOT / "models" / "student_pointmap"
if str(_MOGE_PARENT) not in sys.path:
    sys.path.insert(0, str(_MOGE_PARENT))

from moge.model.v2 import MoGeModel  # noqa: E402


class MetricAnythingEstimator:
    """MetricAnything (MoGe student_pointmap) depth estimator.

    Parameters
    ----------
    resolution_level : int
        0-9: higher = more accurate but slower.
    weights_dir : str | Path | None
        If given, load weights from this local directory.
    device : str
        PyTorch device string.
    fp16 : bool
        Load model in float16.  Only works on CUDA; CPU will
        automatically fall back to fp32.
    """

    def __init__(
        self,
        resolution_level: int = 3,
        weights_dir: str | Path | None = None,
        device: str = "cpu",
        fp16: bool = False,
    ):
        self.resolution_level = resolution_level
        self.device = torch.device(device)

        # fp16 not supported on CPU for this model (interpolation ops)
        use_fp16 = fp16 and self.device.type not in ("cpu", "mps")
        self.dtype = torch.float16 if use_fp16 else torch.float32

        if weights_dir is not None:
            model_path = Path(weights_dir) / "metric-anything" / "student_pointmap.pt"
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Local weights not found at {model_path}. "
                    f"Run scripts/download_models.sh first."
                )
            source = str(model_path)
        else:
            source = "yjh001/metricanything_student_pointmap"

        self.model = MoGeModel.from_pretrained(source).to(self.dtype).to(self.device)
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
            Depth map with values in metres.
        """
        use_fp16 = self.device.type not in ("mps", "cpu")
        img_tensor = (
            torch.tensor(frame_rgb / 255.0, dtype=torch.float32, device=self.device)
            .permute(2, 0, 1)
        )
        with torch.no_grad():
            output = self.model.infer(
                img_tensor,
                resolution_level=self.resolution_level,
                use_fp16=use_fp16,
            )
        return output["depth"].cpu().numpy()

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
            # Handle potential size mismatch between depth map and input
            if depth_map.shape != frame_rgb.shape[:2]:
                sx = depth_map.shape[1] / frame_rgb.shape[1]
                sy = depth_map.shape[0] / frame_rgb.shape[0]
                dx = min(int(x * sx), w - 1)
                dy = min(int(y * sy), h - 1)
            else:
                dx, dy = x, y
            
            # Ensure coordinates are safe (handle negative or > max)
            dx = max(0, min(int(dx), w - 1))
            dy = max(0, min(int(dy), h - 1))
            
            results.append(float(depth_map[dy, dx]))
        return results

    def estimate_timed(self, frame_rgb: np.ndarray):
        """Like ``estimate`` but also returns elapsed seconds."""
        t0 = time.perf_counter()
        depth = self.estimate(frame_rgb)
        elapsed = time.perf_counter() - t0
        return depth, elapsed

"""
MetricAnything – Webcam Depth Demo
===================================
Shows a live webcam feed. When the user clicks a pixel, the current
frame is sent to the model and the metric depth at that pixel is
printed along with the inference time.

Usage
-----
    python demo_depth_click.py              # default
    python demo_depth_click.py --fp16       # half precision (halves memory)
    python demo_depth_click.py --res 5      # resolution level 0-9

Controls
--------
    Left-click  : run inference & print depth at pixel
    Q / ESC     : quit
"""

import sys, os

# Enable MPS→CPU fallback for ops not yet implemented on Apple Silicon
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# ── make sure the moge package (shipped inside models/student_pointmap) is importable
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)  # one level up from demos/
_MOGE_PARENT = os.path.join(_REPO_ROOT, "models", "student_pointmap")
if _MOGE_PARENT not in sys.path:
    sys.path.insert(0, _MOGE_PARENT)

import argparse
import time
import cv2
import numpy as np
import torch
from moge.model.v2 import MoGeModel


# ─────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────
PRETRAINED       = "yjh001/metricanything_student_pointmap"
WINDOW_NAME      = "MetricAnything – Depth Demo (click to probe)"
RESOLUTION_LEVEL = 3          # 0-9: higher = more accurate but slower
# On M1 Mac, CPU is often faster than MPS due to many fallback ops.
FORCE_DEVICE     = "cpu"


# ─────────────────────────────────────────────────────────
# Globals
# ─────────────────────────────────────────────────────────
pending_click: tuple | None = None        # (x, y) queued by mouse callback
latest_frame_rgb: np.ndarray | None = None


def on_mouse(event, x, y, flags, param):
    """OpenCV mouse callback – queues a click for the main loop."""
    global pending_click
    if event == cv2.EVENT_LBUTTONDOWN:
        pending_click = (x, y)


def colorize_depth_map(depth: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Normalise a depth map to 0-255 and apply a colour map."""
    d = depth.copy()
    valid = d[mask] if mask is not None else d.ravel()
    if valid.size == 0:
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    lo, hi = np.percentile(valid, [2, 98])
    d = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    colored = cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    if mask is not None:
        colored[~mask] = 0
    return colored


def pick_device() -> torch.device:
    if FORCE_DEVICE and FORCE_DEVICE != "auto":
        return torch.device(FORCE_DEVICE)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_inference(model, frame_rgb, device, resolution_level):
    """Run depth inference on a single frame. Returns (depth, mask, elapsed_s)."""
    use_fp16 = device.type not in ("mps", "cpu")
    img_tensor = (
        torch.tensor(frame_rgb / 255.0, dtype=torch.float32, device=device)
        .permute(2, 0, 1)
    )
    t0 = time.perf_counter()
    with torch.no_grad():
        output = model.infer(img_tensor, resolution_level=resolution_level, use_fp16=use_fp16)
    elapsed = time.perf_counter() - t0

    depth = output["depth"].cpu().numpy()
    mask  = output["mask"].cpu().numpy()
    return depth, mask, elapsed


def main():
    global pending_click, latest_frame_rgb

    parser = argparse.ArgumentParser(description="MetricAnything – Webcam Depth Demo")
    parser.add_argument("--fp16", action="store_true",
                        help="Load model in float16 (halves memory, may speed up)")
    parser.add_argument("--res", type=int, default=RESOLUTION_LEVEL,
                        help=f"Resolution level 0-9 (default: {RESOLUTION_LEVEL})")
    args = parser.parse_args()

    device = pick_device()

    use_fp16_model = args.fp16
    if use_fp16_model and device.type == "cpu":
        print("[warn] fp16 not supported for MetricAnything on CPU (interpolation ops). Falling back to fp32.")
        use_fp16_model = False

    dtype = torch.float16 if use_fp16_model else torch.float32
    prec_label = "fp16" if use_fp16_model else "fp32"

    print(f"[info] Using device: {device}  |  precision: {prec_label}  |  resolution_level: {args.res}")

    print(f"[info] Loading model {PRETRAINED} …")
    model = MoGeModel.from_pretrained(PRETRAINED).to(dtype).to(device)
    model.eval()
    print("[info] Model loaded ✓")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[error] Cannot open webcam.")
        sys.exit(1)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    print("[info] Webcam opened. Click on the image to probe depth.")
    print("       Press 'Q' / ESC to quit.\n")

    last_status    = "Click anywhere to measure depth"
    click_point    = None       # (x, y) of the last click
    last_depth_val = None       # depth value at the last click

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                continue

            latest_frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # ── handle pending click ────────────────────
            if pending_click is not None:
                cx, cy = pending_click
                pending_click = None
                click_point = (cx, cy)

                print(f"[click] Running inference for pixel ({cx}, {cy}) …")
                depth, mask, elapsed = run_inference(model, latest_frame_rgb, device, args.res)

                # map click coords to depth map coords if sizes differ
                h, w = frame_bgr.shape[:2]
                if depth.shape != (h, w):
                    scale_x = depth.shape[1] / w
                    scale_y = depth.shape[0] / h
                    dx = min(int(cx * scale_x), depth.shape[1] - 1)
                    dy = min(int(cy * scale_y), depth.shape[0] - 1)
                else:
                    dx, dy = cx, cy

                d_val = depth[dy, dx]
                valid = mask[dy, dx] if mask is not None else True
                tag = "" if valid else " [outside valid mask]"

                last_depth_val = d_val
                last_status = f"({cx},{cy}): {d_val:.3f}m  |  {elapsed:.2f}s{tag}"
                print(f"[result] Pixel ({cx}, {cy})  →  depth = {d_val:.4f} m  |  inference: {elapsed:.2f}s{tag}")

            # ── compose display (always live feed) ──────
            display = frame_bgr.copy()

            # draw crosshair + depth label at last click
            if click_point is not None and last_depth_val is not None:
                px, py = click_point
                cv2.drawMarker(display, (px, py), (0, 255, 0),
                               cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)
                label = f"{last_depth_val:.3f}m"
                (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                lx = min(px + 15, display.shape[1] - tw - 5)
                ly = max(py - 10, th + 5)
                cv2.rectangle(display, (lx - 2, ly - th - 2), (lx + tw + 2, ly + baseline + 2),
                              (0, 0, 0), cv2.FILLED)
                cv2.putText(display, label, (lx, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.putText(display, last_status,
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[info] Done.")


if __name__ == "__main__":
    main()

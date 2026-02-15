"""
Depth Anything V2 – Webcam Metric Depth Demo (Indoor)
======================================================
Same interface as demo_depth_click.py but uses the much lighter
Depth-Anything-V2-Metric-Indoor models via HuggingFace transformers.

Usage
-----
    python demo_dav2.py                  # default: small (24.8M params)
    python demo_dav2.py --model base     # 97.5M params
    python demo_dav2.py --model large    # 335M params
    python demo_dav2.py --fp16           # half precision (halves memory)

Controls
--------
    Left-click  : run inference & print depth at pixel
    Q / ESC     : quit
"""

import argparse
import sys
import time
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


# ─────────────────────────────────────────────────────────
# Model variants
# ─────────────────────────────────────────────────────────
MODELS = {
    "small": ("depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf", "24.8M params"),
    "base":  ("depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",  "97.5M params"),
    "large": ("depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf", "335M params"),
}


# ─────────────────────────────────────────────────────────
# Globals
# ─────────────────────────────────────────────────────────
pending_click = None


def on_mouse(event, x, y, flags, param):
    global pending_click
    if event == cv2.EVENT_LBUTTONDOWN:
        pending_click = (x, y)


def run_inference(model, processor, frame_rgb, device):
    """Run depth inference on a single RGB frame (H, W, 3 uint8).
    Returns (depth_map, elapsed_seconds).
    depth_map shape matches the original frame (H, W) in metres.
    """
    pil_image = Image.fromarray(frame_rgb)
    inputs = processor(images=pil_image, return_tensors="pt").to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(**inputs)
    elapsed = time.perf_counter() - t0

    predicted_depth = outputs.predicted_depth  # (1, h_model, w_model)

    # resize to original frame size
    h, w = frame_rgb.shape[:2]
    depth = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=(h, w),
        mode="bicubic",
        align_corners=False,
    ).squeeze().cpu().numpy()

    return depth, elapsed


def main():
    global pending_click

    parser = argparse.ArgumentParser(description="Depth Anything V2 – Metric Indoor Demo")
    parser.add_argument("--model", choices=["small", "base", "large"], default="small",
                        help="Model size: small (24.8M), base (97.5M), large (335M)")
    parser.add_argument("--fp16", action="store_true",
                        help="Load model in float16 (halves memory, may speed up)")
    args = parser.parse_args()

    model_id, param_info = MODELS[args.model]
    dtype = torch.float16 if args.fp16 else torch.float32
    prec_label = "fp16" if args.fp16 else "fp32"
    window_name = f"DAv2 {args.model} ({param_info}, {prec_label}) – click to probe"

    device = torch.device("cpu")  # safest on M1; change to "mps" if you like
    print(f"[info] Using device: {device}  |  precision: {prec_label}")

    print(f"[info] Loading model {model_id} ({param_info}) …")
    processor = AutoImageProcessor.from_pretrained(model_id)
    model_net = AutoModelForDepthEstimation.from_pretrained(model_id, torch_dtype=dtype).to(device)
    model_net.eval()
    print("[info] Model loaded ✓")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[error] Cannot open webcam.")
        sys.exit(1)

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, on_mouse)
    print("[info] Webcam opened. Click on the image to probe depth.")
    print("       Press 'Q' / ESC to quit.\n")

    last_status    = "Click anywhere to measure depth"
    click_point    = None
    last_depth_val = None

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # ── handle pending click ────────────────────
            if pending_click is not None:
                cx, cy = pending_click
                pending_click = None
                click_point = (cx, cy)

                print(f"[click] Running inference for pixel ({cx}, {cy}) …")
                depth, elapsed = run_inference(model_net, processor, frame_rgb, device)

                d_val = depth[cy, cx]
                last_depth_val = d_val
                last_status = f"({cx},{cy}): {d_val:.3f}m  |  {elapsed:.2f}s"
                print(f"[result] Pixel ({cx}, {cy})  →  depth = {d_val:.4f} m  |  inference: {elapsed:.2f}s")

            # ── compose display (always live feed) ──────
            display = frame_bgr.copy()

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

            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[info] Done.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Gradio demo for MetricAnything DepthMap."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Tuple

import gradio as gr
import matplotlib
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2

from depth_model import MetricAnythingDepthMap


EXAMPLES_DIR = Path(__file__).parent / "example_images"
MODEL_ID = "yjh001/metricanything_student_depthmap"
MODEL_FILENAME = "student_depthmap.pt"
MAX_DEPTH = 200.0


def list_examples() -> list[Path]:
    exts = {".png", ".jpg", ".jpeg"}
    if not EXAMPLES_DIR.exists():
        return []
    return sorted([p for p in EXAMPLES_DIR.iterdir() if p.suffix.lower() in exts])


def read_intrinsics(json_path: Path) -> float | None:
    if not json_path.exists():
        return None
    data = json.loads(json_path.read_text())
    cam_in = data.get("cam_in")
    if cam_in is None:
        return None
    if isinstance(cam_in, (list, tuple)) and len(cam_in) > 0:
        return float(cam_in[0])
    if isinstance(cam_in, dict):
        for key in ("fx", "f_x", "focal_length", "focal_length_px"):
            if key in cam_in:
                return float(cam_in[key])
    return None


def make_transform() -> v2.Compose:
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def colorize_depth(depth: np.ndarray, max_depth: float = MAX_DEPTH, cmap: str = "turbo_r") -> np.ndarray:
    """Inverse-depth visualization in a 0–max_depth meter range; invalid/far pixels are white."""
    valid = np.isfinite(depth) & (depth > 0) & (depth <= max_depth)
    if not np.any(valid):
        return np.full((*depth.shape, 3), 255, dtype=np.uint8)

    disp = np.where(valid, 1.0 / depth, np.nan)
    min_disp, max_disp = np.nanquantile(disp, 0.001), np.nanquantile(disp, 0.99)
    disp = (disp - min_disp) / (max_disp - min_disp) if max_disp > min_disp else disp * 0.0

    colored = np.nan_to_num(matplotlib.colormaps[cmap](1.0 - disp)[..., :3], 0.0)
    colored = (colored.clip(0.0, 1.0) * 255).astype(np.uint8)
    colored[~valid] = 255
    return np.ascontiguousarray(colored)


def prepare_focal(image: Image.Image, image_path: Path | None) -> Tuple[float, str, gr.Slider]:
    width = image.width
    fx = None
    info = ""
    if image_path is not None:
        fx = read_intrinsics(image_path.with_suffix(".json"))
        if fx is not None:
            info = f"Intrinsics found. Using focal length (pixels): **{fx:.2f}**."
        else:
            info = f"No intrinsics found. Using image width (W={width}) as focal length (pixels)."
    else:
        info = f"No intrinsics found. Using image width (W={width}) as focal length (pixels)."

    if fx is None:
        fx = float(width)

    # slider = gr.Slider.update(value=fx, minimum=1, maximum=max(2000, int(width * 2)), step=1)
    slider = gr.update(value=fx, minimum=1, maximum=max(2000, int(width * 2)), step=1)
    return fx, info, slider


def select_example(example_paths: list[str], evt: gr.SelectData):
    path = Path(example_paths[evt.index])
    image = Image.open(path).convert("RGB")
    _, info, slider = prepare_focal(image, path)
    return image, slider, info, "example"


def on_input_change(image: Image.Image | None, source: str):
    if image is None:
        # return gr.Slider.update(), gr.update(), ""
        return gr.update(), gr.update(), ""
    if source == "example":
        # return gr.Slider.update(), gr.update(), ""
        return gr.update(), gr.update(), ""
    _, info, slider = prepare_focal(image, None)
    return slider, info, ""


def load_model() -> MetricAnythingDepthMap:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MetricAnythingDepthMap.from_pretrained(
        MODEL_ID,
        model_kwargs={"device": device},
        filename=MODEL_FILENAME,
    )
    model.eval()
    return model


TRANSFORM = make_transform()
MODEL = load_model()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def run_inference(image: Image.Image | None, focal_px: float):
    if image is None:
        return None, "Please provide an input image."

    tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)
    prediction = MODEL.infer(tensor, f_px=float(focal_px))
    depth = prediction["depth"].detach().cpu().numpy().squeeze()

    vis = colorize_depth(depth, max_depth=MAX_DEPTH)
    valid = np.isfinite(depth) & (depth > 0) & (depth <= MAX_DEPTH)
    if np.any(valid):
        min_d = float(depth[valid].min())
        max_d = float(depth[valid].max())
        stats = f"Depth range (0–{MAX_DEPTH:.0f} m): min={min_d:.2f} m, max={max_d:.2f} m"
    else:
        stats = f"No valid depth in 0–{MAX_DEPTH:.0f} m range."

    return vis, stats


def build_demo() -> gr.Blocks:
    example_paths = list_examples()
    gallery_items = [(str(p), p.name) for p in example_paths]

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# MetricAnything DepthMap")
        gr.Markdown("Select an example or upload your own image to estimate metric depth.")

        with gr.Row():
            with gr.Column(scale=3):
                gallery = gr.Gallery(
                    value=gallery_items,
                    label="Examples",
                    columns=4,
                    rows=2,
                    height=220,
                )
                input_image = gr.Image(type="pil", label="Input", height=320)
                focal_slider = gr.Slider(label="Focal length (pixels)", minimum=1, maximum=4000, step=1, value=1000)
                info = gr.Markdown("Select an example or upload an image.")
                run_btn = gr.Button("Run")

            with gr.Column(scale=4):
                output_image = gr.Image(type="numpy", label="Depth (visualized)", height=420)
                output_stats = gr.Markdown("")

        example_state = gr.State([str(p) for p in example_paths])
        source_state = gr.State("")

        if example_paths:
            gallery.select(
                select_example,
                inputs=[example_state],
                outputs=[input_image, focal_slider, info, source_state],
            )

        input_image.change(
            on_input_change,
            inputs=[input_image, source_state],
            outputs=[focal_slider, info, source_state],
        )

        run_btn.click(
            run_inference,
            inputs=[input_image, focal_slider],
            outputs=[output_image, output_stats],
        )

    return demo


demo = build_demo()

if __name__ == "__main__":
    demo.launch()

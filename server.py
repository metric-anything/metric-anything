"""
Depth Estimation API Server
============================
FastAPI service exposing metric depth estimation via DAv2 and MetricAnything.

Usage
-----
    uvicorn server:app --host 0.0.0.0 --port 8081

Environment variables
---------------------
    DEPTH_WEIGHTS_DIR   Local weights directory (default: ./weights)
    DEPTH_DEFAULT_MODEL Default model (default: dav2-small)
    DEPTH_DEVICE        PyTorch device (default: cpu)
"""

import base64
import io
import logging
import os
import time
from typing import List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

from depth import DAv2Estimator, MetricAnythingEstimator

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
WEIGHTS_DIR = os.environ.get("DEPTH_WEIGHTS_DIR", "./weights")
DEFAULT_MODEL = os.environ.get("DEPTH_DEFAULT_MODEL", "dav2-small")
DEVICE = os.environ.get("DEPTH_DEVICE", "cpu")

logger = logging.getLogger("depth-server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")

# ─────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────

class DepthPointsRequest(BaseModel):
    image_base64: str
    points: List[Tuple[int, int]]
    model: Optional[str] = None  # override per-request


class DepthPointsResponse(BaseModel):
    depths: List[float]
    model: str
    inference_ms: float


class DepthMapRequest(BaseModel):
    image_base64: str
    model: Optional[str] = None


class DepthMapResponse(BaseModel):
    depth_map_base64: str  # base64-encoded 32-bit float raw bytes
    width: int
    height: int
    model: str
    inference_ms: float


class ModelSwitchRequest(BaseModel):
    model: str


class HealthResponse(BaseModel):
    status: str
    active_model: str
    available_models: List[str]


# ─────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────
AVAILABLE_MODELS = [
    "dav2-small",
    "dav2-base",
    "dav2-large",
    "metric-anything",
]

_loaded_models: dict = {}
_active_model: str = DEFAULT_MODEL


def _resolve_weights_dir():
    """Return weights_dir if it exists, else None (fall back to HF Hub)."""
    if os.path.isdir(WEIGHTS_DIR):
        return WEIGHTS_DIR
    return None


def _load_model(model_name: str):
    """Load (or return cached) estimator for the given model name."""
    if model_name in _loaded_models:
        return _loaded_models[model_name]

    weights = _resolve_weights_dir()
    logger.info("Loading model %s (device=%s, weights_dir=%s) …", model_name, DEVICE, weights)

    if model_name.startswith("dav2-"):
        size = model_name.split("-", 1)[1]
        estimator = DAv2Estimator(model_size=size, weights_dir=weights, device=DEVICE)
    elif model_name == "metric-anything":
        estimator = MetricAnythingEstimator(weights_dir=weights, device=DEVICE)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    _loaded_models[model_name] = estimator
    logger.info("Model %s loaded ✓", model_name)
    return estimator


def _get_estimator(override: Optional[str] = None):
    """Get the active estimator, optionally overridden per request."""
    name = override or _active_model
    if name not in AVAILABLE_MODELS:
        raise HTTPException(400, f"Unknown model: {name}. Available: {AVAILABLE_MODELS}")
    return _load_model(name), name


def _decode_image(image_base64: str) -> np.ndarray:
    """Decode a base64 JPEG/PNG string into an RGB numpy array."""
    try:
        image_bytes = base64.b64decode(image_base64)
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return np.array(pil_image)
    except Exception as exc:
        raise HTTPException(400, f"Failed to decode image: {exc}")


# ─────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────
app = FastAPI(
    title="Depth Estimation API",
    description="Metric depth estimation via DAv2 and MetricAnything",
    version="0.1.0",
)


@app.on_event("startup")
async def startup():
    """Pre-load the default model."""
    logger.info("Starting depth estimation server …")
    _load_model(_active_model)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        active_model=_active_model,
        available_models=AVAILABLE_MODELS,
    )


@app.post("/depth/points", response_model=DepthPointsResponse)
async def depth_points(req: DepthPointsRequest):
    """Return metric depth values at specific pixel coordinates."""
    estimator, model_name = _get_estimator(req.model)
    frame_rgb = _decode_image(req.image_base64)

    t0 = time.perf_counter()
    depths = estimator.estimate_at_points(frame_rgb, req.points)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return DepthPointsResponse(
        depths=depths,
        model=model_name,
        inference_ms=round(elapsed_ms, 1),
    )


@app.post("/depth/map", response_model=DepthMapResponse)
async def depth_map(req: DepthMapRequest):
    """Return the full depth map as base64-encoded float32 bytes."""
    estimator, model_name = _get_estimator(req.model)
    frame_rgb = _decode_image(req.image_base64)

    t0 = time.perf_counter()
    depth = estimator.estimate(frame_rgb)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    depth_bytes = depth.astype(np.float32).tobytes()
    depth_b64 = base64.b64encode(depth_bytes).decode("ascii")

    return DepthMapResponse(
        depth_map_base64=depth_b64,
        width=depth.shape[1],
        height=depth.shape[0],
        model=model_name,
        inference_ms=round(elapsed_ms, 1),
    )


@app.post("/depth/model")
async def switch_model(req: ModelSwitchRequest):
    """Switch the active model for subsequent requests."""
    global _active_model
    if req.model not in AVAILABLE_MODELS:
        raise HTTPException(400, f"Unknown model: {req.model}. Available: {AVAILABLE_MODELS}")
    _load_model(req.model)  # pre-load to validate
    _active_model = req.model
    logger.info("Switched active model to %s", _active_model)
    return {"status": "ok", "active_model": _active_model}

# Depth Estimation Service

A **FastAPI-based metric depth estimation** backend that wraps [MetricAnything](https://github.com/metric-anything/metric-anything) and [Depth Anything V2](https://huggingface.co/depth-anything) into a production-ready API. Designed to run as a sidecar alongside a Tauri/MediaPipe frontend for real-time exercise coaching.

## Models

| Model | ID | Params | Inference (CPU) | Accuracy |
| --- | --- | --- | --- | --- |
| DAv2 Small | `dav2-small` | 24.8M | ~0.5s | Good |
| DAv2 Base | `dav2-base` | 97.5M | ~1.0s | Better |
| DAv2 Large | `dav2-large` | 335M | ~3.0s | Best (DAv2) |
| MetricAnything | `metric-anything` | 326M | ~3.0s | Most accurate |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download models (all, or pick one)
bash scripts/download_models.sh              # all models
bash scripts/download_models.sh dav2-small   # just one

# Start the server
uvicorn server:app --port 8081
```

## API Endpoints

### `GET /health`

```bash
curl http://localhost:8081/health
```

```json
{
  "status": "ok",
  "active_model": "dav2-small",
  "available_models": ["dav2-small", "dav2-base", "dav2-large", "metric-anything"]
}
```

### `POST /depth/points`

Get metric depth (metres) at specific pixel coordinates.

```bash
curl -X POST http://localhost:8081/depth/points \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "<base64-encoded JPEG>",
    "points": [[640, 360], [500, 200]],
    "model": "dav2-small"
  }'
```

```json
{
  "depths": [0.6523, 3.1245],
  "model": "dav2-small",
  "inference_ms": 450.2
}
```

### `POST /depth/map`

Get the full depth map as base64-encoded float32 bytes.

```bash
curl -X POST http://localhost:8081/depth/map \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "<base64-encoded JPEG>",
    "model": "dav2-small"
  }'
```

```json
{
  "depth_map_base64": "<base64-encoded float32 array>",
  "width": 640,
  "height": 480,
  "model": "dav2-small",
  "inference_ms": 832.8
}
```

### `POST /depth/model`

Switch the active model at runtime.

```bash
curl -X POST http://localhost:8081/depth/model \
  -H "Content-Type: application/json" \
  -d '{"model": "metric-anything"}'
```

```json
{
  "status": "ok",
  "active_model": "metric-anything"
}
```

## Docker

### Build (cross-platform from Mac → AMD64)

```bash
docker buildx build --platform linux/amd64 -t kinekernel/depth-service:latest --push .
```

### Run (on the target machine)

```bash
# Pull the image
docker pull kinekernel/depth-service:latest

# Download models to a host directory
bash scripts/download_models.sh dav2-small

# Run with weights mounted as a volume
docker run --rm -p 8081:8081 \
  -v /data/weights:/app/weights \
  kinekernel/depth-service:latest
```

### Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `DEPTH_WEIGHTS_DIR` | `./weights` | Path to downloaded model weights |
| `DEPTH_DEFAULT_MODEL` | `dav2-small` | Model loaded on startup |
| `DEPTH_DEVICE` | `cpu` | PyTorch device (`cpu`, `cuda`) |

## Project Structure

```bash
├── server.py                 # FastAPI app (entry point)
├── depth/                    # Depth estimation module
│   ├── __init__.py
│   ├── dav2.py               # DAv2 wrapper (small/base/large)
│   └── metric_anything.py    # MetricAnything wrapper
├── models/student_pointmap/  # Original MoGe model code
├── scripts/
│   └── download_models.sh    # One-command model download
├── demos/                    # Interactive webcam demos
│   ├── demo_depth_click.py   # MetricAnything click-to-probe
│   └── demo_dav2.py          # DAv2 click-to-probe
├── weights/                  # Downloaded weights (gitignored)
├── Dockerfile
├── requirements.txt
└── WALKTHROUGH.md
```

## Credits

- [MetricAnything](https://github.com/metric-anything/metric-anything) — Ma et al., 2026
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) — Yang et al., 2024

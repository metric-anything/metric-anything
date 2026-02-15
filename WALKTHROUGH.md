# Repo Reorganization Walkthrough

Restructured MetricAnything from flat demo scripts into a **FastAPI depth estimation service** for Tauri/MediaPipe integration.

## New Structure

```bash
MetricAnything/
├── server.py                 # FastAPI app (entry point)
├── depth/                    # Depth estimation module
│   ├── __init__.py           # Exports DAv2Estimator, MetricAnythingEstimator
│   ├── dav2.py               # DAv2 wrapper (small/base/large)
│   └── metric_anything.py    # MetricAnything wrapper
├── models/student_pointmap/  # Original MoGe model code
├── scripts/download_models.sh# One-command model setup
├── demos/                    # Moved demo scripts
├── Dockerfile                # Production container
├── requirements.txt          # Updated with FastAPI deps
└── .gitignore                # weights/ excluded
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

Returns metric depth (metres) at specific pixel coordinates.

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

Returns the full depth map as base64-encoded float32 bytes.

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

Switch the active model for subsequent requests.

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

## Quick Start

```bash
pip install -r requirements.txt
uvicorn server:app --port 8081
```

## Production (GTR7) Setup

```bash
# 1. Download models once
bash scripts/download_models.sh dav2-small

# 2. Run with Docker
docker build -t depth-service .
docker run --rm -p 8081:8081 -v /data/weights:/app/weights depth-service
```

## Verification Results

- ✅ Server starts and loads `dav2-small` in ~2s
- ✅ `GET /health` → `{"status":"ok","active_model":"dav2-small",...}`
- ✅ `POST /depth/points` → `{"depths":[0.93,0.88],"model":"dav2-small","inference_ms":832.8}`
- ✅ Runtime model switching via `/depth/model`
- ✅ Demo scripts preserved in `demos/`

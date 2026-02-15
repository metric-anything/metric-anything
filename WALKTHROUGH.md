# Repo Reorganization Walkthrough

Restructured MetricAnything from flat demo scripts into a **FastAPI depth estimation service** for Tauri/MediaPipe integration.

## New Structure

```
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

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Status + available models |
| `/depth/points` | POST | Depth at pixel coords (base64 JPEG + points) |
| `/depth/map` | POST | Full depth map |
| `/depth/model` | POST | Switch active model at runtime |

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

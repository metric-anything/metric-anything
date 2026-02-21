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
bash deploy/download_models.sh              # all models
bash deploy/download_models.sh dav2-small   # just one

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

### `WebSocket /depth/stream`

Real-time streaming endpoint for high-performance depth estimation. Bypasses Base64 and JPEG overhead.

**Client -> Server (Binary Buffer):**

- `[2 bytes: N]` (number of coordinates)
- `[N * 4 bytes]` (X, Y 16-bit integers)
- `[RGB Pixels]` (Raw RGB image bytes)

**Server -> Client (JSON):**

```json
{
  "depths": [0.652, 0.640],
  "model": "dav2-small",
  "inference_ms": 110.5
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

### Build (Multi-Platform: AMD64 & ARM64)

The project leverages Docker's `buildx` plugin to assemble multi-arch containers. The `Dockerfile` natively adapts to the target architecture to fetch the right CPU dependencies (e.g., bypassing enormous CUDA wheels on `amd64` while grabbing native CPU packages on `arm64`).

```bash
# Build and keep locally (single platform only for direct --load)
docker buildx build --platform linux/amd64 --load -t kinekernel/depth-service:latest .

# Build multi-arch and push directly to Docker Hub (recommended for cross-platform)
docker buildx build --platform linux/amd64,linux/arm64 --push -t kinekernel/depth-service:latest .
```

> **Note:** The `--load` flag is required to keep the image in your local Docker so it can be exported. Without it, buildx only stores it in the build cache.

### Direct Transfer (LAN)

Transfer the image directly to the mini PC without pushing to Docker Hub:

```bash
# One-liner: stream directly over SSH (fastest, no temp file)
docker save kinekernel/depth-service:latest | gzip | ssh kk@192.168.0.236 'podman load'
```

Or in two steps:

```bash
# 1. Export to tarball
docker save kinekernel/depth-service:latest | gzip > depth-service.tar.gz

# 2. Copy and load
scp depth-service.tar.gz kk@192.168.0.236:~/deploy/
ssh kk@192.168.0.236 'podman load -i ~/deploy/depth-service.tar.gz'
```

### Deploy (on the target machine)

Copy the `deploy/` folder to the target machine, then:

```bash
cd deploy

# Download models
bash download_models.sh dav2-small

# Pull and start (with the Linux GPU LLM service)
docker compose --profile linux-gpu up -d
```

> **Note for Mac/Windows Development:** You can run a standard `docker compose up -d` (without the profile flag) on your local machine. This will start only the Depth Service and Frontend UI, bypassing the Linux-specific Vulkan GPU drivers. The UI will then connect to your local natively-running Ollama instance!

See [deploy/docker-compose.yml](deploy/docker-compose.yml) for the full configuration.

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
├── deploy/                   # Everything needed on the target machine
│   ├── docker-compose.yml
│   ├── download_models.sh
│   └── weights/              # Downloaded model weights (gitignored)
├── demos/                    # Interactive webcam demos
│   ├── demo_depth_click.py   # MetricAnything click-to-probe
│   └── demo_dav2.py          # DAv2 click-to-probe
├── Dockerfile
├── requirements.txt
└── WALKTHROUGH.md
```

## Credits

- [MetricAnything](https://github.com/metric-anything/metric-anything) — Ma et al., 2026
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) — Yang et al., 2024

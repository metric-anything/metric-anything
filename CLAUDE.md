# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **full-stack real-time exercise coaching system** that combines:

- **Backend**: FastAPI-based metric depth estimation service (Python)
- **Frontend**: Real-time web application with pose detection and AI coaching (JavaScript/HTML/CSS)
- **Infrastructure**: Docker-based deployment with multiple services

The system provides **real-time reach gesture analysis** with metric depth measurement and AI coaching feedback. It uses computer vision (MediaPipe Pose) for body tracking, depth estimation models for measuring distances, and LLMs for providing personalized feedback.

## Key Technologies

### Backend (Python)

- **FastAPI** - High-performance web framework for building APIs
- **PyTorch** - Deep learning framework for model inference
- **Transformers** (HuggingFace) - For Depth Anything V2 models
- **PIL/Pillow** - Image processing
- **NumPy** - Numerical computations

### Frontend (JavaScript)

- **Vite** - Modern build tool and development server
- **MediaPipe Tasks Vision** - Client-side pose detection
- **Canvas API** - Real-time video overlay rendering
- **ES Modules** - Modern JavaScript module system

### Models Supported

1. **Depth Anything V2 (DAv2)** - Three sizes: small (24.8M), base (97.5M), large (335M)
2. **MetricAnything** - Most accurate model (326M parameters)
3. **LLM Models** - Qwen3 0.6B, Qwen2.5 1.5B/3B for coaching feedback

### Infrastructure

- **Docker** - Containerization
- **Docker Compose** - Multi-service orchestration
- **Nginx** - Frontend web server (in production)
- **llama.cpp** - LLM inference server

## Development Commands

### Backend Development

```bash
# Install dependencies
pip install -r requirements.txt

# Download models (all, or pick one)
bash deploy/download_models.sh              # all models
bash deploy/download_models.sh dav2-small   # just one

# Start the server
uvicorn server:app --port 8081
```

### Frontend Development

```bash
cd MyJointDemo
npm install
npm run dev  # Starts Vite dev server at http://localhost:5173
```

### Production Build

```bash
# Backend
docker build -t kinekernel/depth-service:latest .

# Frontend
cd MyJointDemo
docker build -t kinekernel/joint-demo:latest .
```

### Full Deployment

```bash
# Using docker-compose
cd deploy
docker compose up -d
```

## System Architecture

### Components

1. **Depth Estimation Service** (`depth-service:8081`)
   - FastAPI server with endpoints:
     - `GET /health` - Service health check
     - `POST /depth/points` - Get depth at specific coordinates
     - `POST /depth/map` - Get full depth map
     - `POST /depth/model` - Switch active model

2. **LLM Service** (`llm-service:11434`)
   - llama.cpp server for Qwen models
   - Provides coaching feedback based on exercise data

3. **Frontend Web App** (`joint-demo:3000`)
   - Real-time webcam capture and pose detection
   - Reach gesture tracking with state machine
   - Integration with depth and LLM services
   - Real-time visual feedback and coaching

### Data Flow

1. Webcam → MediaPipe Pose → Body landmarks
2. Landmarks → Reach gesture state machine → Depth query points
3. Video frame + points → Depth service → Metric depth values
4. Exercise data (after set) → LLM service → Personalized feedback

### Key Features

- **Real-time processing** (~30 FPS pose detection)
- **Bandwidth optimization** (images downscaled to 640px for depth queries)
- **Model switching** at runtime
- **Memory management** (unloads previous models when switching)
- **Multi-platform support** (CPU, CUDA, MPS for Apple Silicon)
- **Production-ready deployment** with Docker

## Directory Structure

```bash
MetricAnything/
├── server.py                 # FastAPI app (entry point)
├── depth/                    # Depth estimation module
│   ├── __init__.py           # Exports DAv2Estimator, MetricAnythingEstimator
│   ├── dav2.py               # DAv2 wrapper (small/base/large)
│   └── metric_anything.py    # MetricAnything wrapper
├── models/student_pointmap/  # Original MoGe model code
├── MyJointDemo/              # Frontend web application
│   ├── src/
│   │   ├── main.js           # Main application entry point
│   │   ├── pose.js           # MediaPipe Pose wrapper
│   │   ├── reach-joint.js    # Reach gesture state machine
│   │   ├── depth-client.js   # Depth API client
│   │   ├── llm-client.js     # LLM API client
│   │   ├── ui.js             # Canvas rendering and UI updates
│   │   └── style.css         # Dark theme styling
│   ├── index.html            # Main HTML page
│   ├── chat.html             # Separate chat interface
│   ├── vite.config.js        # Build configuration
│   └── package.json          # Frontend dependencies
├── deploy/                   # Deployment configuration
│   ├── docker-compose.yml    # Multi-service orchestration
│   ├── download_models.sh    # Script to download model weights
│   └── weights/              # Downloaded model weights (gitignored)
├── demos/                    # Interactive webcam demos
│   ├── demo_depth_click.py   # MetricAnything click-to-probe
│   └── demo_dav2.py          # DAv2 click-to-probe
├── Dockerfile                # Backend container definition
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── WALKTHROUGH.md            # Deployment instructions
```

## Environment Variables

### Backend

- `DEPTH_WEIGHTS_DIR` - Path to model weights (default: `./weights`)
- `DEPTH_DEFAULT_MODEL` - Default model (default: `dav2-small`)
- `DEPTH_DEVICE` - PyTorch device (`cpu`, `cuda`, `mps`)

### Frontend

- `VITE_DEPTH_URL` - Depth service URL (frontend)
- `VITE_LLM_URL` - LLM service URL (frontend)
- `VITE_REPS_PER_SET` - Reps before LLM feedback (default: 5)

## API Endpoints

### Depth Service (`http://localhost:8081`)

- `GET /health` - Service health check
- `POST /depth/points` - Get metric depth at specific pixel coordinates
- `POST /depth/map` - Get full depth map as base64-encoded float32 bytes
- `POST /depth/model` - Switch active model at runtime

### LLM Service (`http://localhost:11434`)

- Standard llama.cpp server API for Qwen models

## Performance Optimizations

1. **Image downscaling**: Frontend downscales images to 640px width before sending to depth service (~80% bandwidth saving)
2. **Model caching**: Backend caches loaded models but unloads previous models when switching to free memory
3. **Concurrent requests**: Frontend allows concurrent depth requests for different phases to avoid missing data
4. **Throttled processing**: Pose detection limited to ~30 FPS to maintain performance

## Development Notes

- The frontend uses Vite with ES modules and modern JavaScript
- Backend uses FastAPI with Pydantic models for type-safe API
- Docker builds are cross-platform (Mac → AMD64) using buildx
- Model weights are stored in `deploy/weights/` and gitignored
- Health checks run every 10 seconds to monitor service status
- Real-time feedback is displayed via canvas overlay on video stream

## Testing

- Backend API can be tested with curl commands (examples in README.md)
- Frontend development server runs at `http://localhost:5173`
- Production deployment uses Docker Compose with three services
- Model switching can be tested via `/depth/model` endpoint

## Troubleshooting

- **Model loading issues**: Check `DEPTH_WEIGHTS_DIR` environment variable and ensure weights are downloaded
- **Performance problems**: Consider using smaller models (dav2-small) or GPU acceleration
- **CORS errors**: Frontend needs to connect to backend services on different ports
- **Memory issues**: Backend unloads previous models when switching, but large models may still consume significant memory

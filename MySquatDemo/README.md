# MySquat Demo

Real-time squat depth measurement with AI coaching feedback. Built as a lightweight frontend demo for testing the MetricAnything **depth service** and **LLM service** backend responsiveness.

## How It Works

1. **Webcam** captures live video in the browser
2. **MediaPipe Pose** (runs client-side) detects knee & hip landmarks
3. A **squat state machine** detects rep phases (standing → descending → bottom → ascending)
4. At key moments, the **depth service** is queried for metric knee depth (metres)
5. After 5 reps, movement data is sent to the **LLM** (Qwen3 0.6B) for coaching feedback

## Quick Start (development)

```bash
npm install
npm run dev
# Open http://localhost:5173
```

The backends must be running (see `../deploy/docker-compose.yml`).

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VITE_DEPTH_URL` | `http://localhost:8081` | Depth service URL |
| `VITE_LLM_URL` | `http://localhost:11434` | LLM service URL |
| `VITE_REPS_PER_SET` | `5` | Reps before LLM feedback |
| `VITE_DEPTH_MODEL` | `dav2-small` | Depth model to use |

## Deploy to Mini PC

### Build & transfer the Docker image

```bash
# Build for AMD64
docker buildx build --platform linux/amd64 --load -t kinekernel/squat-demo:latest .

# Transfer to mini PC
docker save kinekernel/squat-demo:latest | gzip | ssh kk@192.168.0.236 'podman load'
```

### Start all services on the mini PC

```bash
cd deploy
docker compose up -d
# Frontend: http://<mini-pc-ip>:3000
# Depth:    http://<mini-pc-ip>:8081
# LLM:     http://<mini-pc-ip>:11434
```

## Project Structure

```
MySquatDemo/
├── src/
│   ├── main.js             # App entry point
│   ├── pose.js             # MediaPipe Pose wrapper
│   ├── squat.js            # Squat state machine
│   ├── depth-client.js     # Depth service API client
│   ├── llm-client.js       # LLM chat API client
│   ├── ui.js               # DOM/Canvas rendering
│   └── style.css           # Dark theme styles
├── index.html              # HTML shell
├── Dockerfile              # Multi-stage build → nginx
├── nginx.conf              # Production server config
├── vite.config.js          # Build config
└── package.json
```

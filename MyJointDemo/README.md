# My Joint Demo

Real-time **Torso Lean Gesture** analysis with metric depth measurement and AI coaching feedback. Built as a lightweight frontend demo for testing the MetricAnything **depth service** and **LLM service** backend responsiveness.

## How It Works

1. **Webcam** captures live video in the browser.
2. **MediaPipe Pose** (runs client-side) detects upper body landmarks (specifically the Shoulders).
3. **Lean State Machine** tracks the gesture using relative Z-coordinates (distance to camera):
    - **Start**: "Ready" Pose (Sitting back in the chair).
    - **Action**: Lean your trunk forward towards the camera.
    - **End**: Return to the sitting position.
4. At key moments (Start/Peak Lean), an optimized **binary WebSocket** streams RGB crop data to the **depth service** to measure the exact depth of the user's **Shoulders** (body).
5. After 5 reps, movement data is sent to the **LLM** (e.g. Qwen) for coaching feedback.

## Quick Start (development)

```bash
cd MyJointDemo
npm install
npm run dev
# Open http://localhost:5173
```

The backends must be running (see `../deploy/docker-compose.yml`).

### Environment Variables

| Variable | Default | Description |
| ---------- | --------- | ------------- |
| `VITE_BACKEND_IP` | `127.0.0.1` | Local or remote prod IP for Vite proxy to route `/api` calls |
| `VITE_REPS_PER_SET` | `5` | Reps before LLM feedback |
| `VITE_DEPTH_MODEL` | `dav2-small` | Depth model to use |

## Deploy to Mini PC

### Build & transfer the Docker image

```bash
# Build for AMD64
docker buildx build --platform linux/amd64 --load -t kinekernel/joint-demo:latest .

# Transfer to mini PC
docker save kinekernel/joint-demo:latest | gzip | ssh kk@172.20.10.10 'podman load'
```

### Start all services on the mini PC

```bash
cd deploy
docker compose up -d
# Frontend: http://<mini-pc-ip>:3000
# Depth:    http://<mini-pc-ip>:8081
# LLM:      http://<mini-pc-ip>:11434
```

## Project Structure

```bash
MyJointDemo/
├── src/
│   ├── main.js             # App entry point
│   ├── pose.js             # MediaPipe Pose wrapper
│   ├── reach-joint.js      # Reach gesture state machine
│   ├── depth-client.js     # Depth service API client
│   ├── llm-client.js       # LLM chat API client (Reach analysis)
│   ├── ui.js               # DOM/Canvas rendering
│   └── style.css           # Dark theme styles
├── index.html              # HTML shell
├── Dockerfile              # Multi-stage build → nginx
├── nginx.conf              # Production server config
├── vite.config.js          # Build config
└── package.json
```

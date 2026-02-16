# Deployment Walkthrough

This guide explains how to deploy the **Depth Service** and **Qwen3 LLM Service** on the AlmaLinux Mini PC (AMD GPU).

## Prerequisites

- **Docker** & **Docker Compose** installed
- **AMD GPU Drivers** installed on host (kernel modules loaded)
- **8GB RAM** (4GB System + 4GB VRAM) is perfect for this setup.

## Folder Structure

The `deploy/` folder is self-contained. You can copy it via USB.

```bash
deploy/
├── docker-compose.yml       # Orchestrates both services
├── download_models.sh       # Fetches models (DAv2 + Qwen3)
└── weights/                 # Model files (gitignored)
    ├── dav2-small/
    └── qwen3-0.6b/Qwen3-0.6B-Q4_K_M.gguf
```

## Setup Steps

1. **Copy the `deploy/` folder** to the mini PC.
2. **Download Models** (if not already done):

    ```bash
    cd deploy
    bash download_models.sh        # Downloads everything
    # OR
    bash download_models.sh qwen3-0.6b  # Just the LLM
    ```

3. **Start Services**:

    ```bash
    docker compose up -d
    ```

## Services

### 1. Depth Service

- **Port**: `8081`
- **Image**: `kinekernel/depth-service:latest`
- **GPU**: Uses `/dev/dri` + `/dev/kfd` (Vulkan/Compute)

### 2. LLM Service (Qwen3 0.6B)

- **Port**: `11434` (OpenAI-compatible)
- **Image**: `ghcr.io/ggml-org/llama.cpp:server-rocm` (~5.6GB)
- **GPU**: Fully offloaded to VRAM (`-ngl 99`)
- **Endpoint**: `http://localhost:11434/v1/chat/completions`

## Troubleshooting

- **"dlopen failed"**: If you try to run `llamafile` directly on host, it fails because ROCm libs are missing. Use the Docker setup (it works).
- **Permission Denied**: On AlmaLinux (SELinux enabled), Docker volumes need the `:z` suffix (e.g. `- ./path:/mount:z`). I've added this to `docker-compose.yml` for you.
- **Restart Loop / Crash**: On Radeon 780M (RDNA3), the container may crash due to ROCm version mismatch. Fix: add `HSA_OVERRIDE_GFX_VERSION=11.0.0` environment variable (added in config).
- **VRAM Errors**: If you see memory errors, try reducing context size (`-c 2048` instead of `4096`).
- **Disk Space**: The ROCm Docker image is large (~5.6GB). Ensure you have ~10GB free space.

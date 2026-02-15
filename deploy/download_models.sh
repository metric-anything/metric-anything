#!/usr/bin/env bash
# ──────────────────────────────────────────────────────
# Download model weights to the local weights/ directory
# ──────────────────────────────────────────────────────
# Usage:
#   bash deploy/download_models.sh              # download all
#   bash deploy/download_models.sh dav2-small   # download one
#
# Requirements: pip install huggingface_hub

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WEIGHTS_DIR="${SCRIPT_DIR}/weights"

get_hf_id() {
    case "$1" in
        dav2-small)      echo "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf" ;;
        dav2-base)       echo "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf" ;;
        dav2-large)      echo "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf" ;;
        metric-anything) echo "yjh001/metricanything_student_pointmap" ;;
        qwen3-0.6b)      echo "__gguf__" ;;  # handled separately
        *) return 1 ;;
    esac
}

ALL_MODELS="dav2-small dav2-base dav2-large metric-anything qwen3-0.6b"

QWEN3_GGUF_URL="https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf"

download_model() {
    local name="$1"
    local hf_id
    hf_id="$(get_hf_id "$name")" || { echo "✗ Unknown model: ${name}"; echo "  Available: ${ALL_MODELS}"; exit 1; }
    local dest="${WEIGHTS_DIR}/${name}"

    if [ -d "$dest" ] && [ "$(ls -A "$dest" 2>/dev/null)" ]; then
        echo "✓ ${name} already exists at ${dest}, skipping."
        return
    fi

    mkdir -p "$dest"

    # GGUF models: direct file download
    if [ "$hf_id" = "__gguf__" ]; then
        local url=""
        case "$name" in
            qwen3-0.6b) url="$QWEN3_GGUF_URL" ;;
        esac
        local filename
        filename="$(basename "$url")"
        echo "⬇ Downloading ${name} (GGUF) → ${dest}/${filename} …"
        if command -v wget &>/dev/null; then
            wget -q --show-progress -O "${dest}/${filename}" "$url"
        else
            curl -L --progress-bar -o "${dest}/${filename}" "$url"
        fi
    else
        echo "⬇ Downloading ${name} (${hf_id}) → ${dest} …"
        huggingface-cli download "$hf_id" --local-dir "$dest"
    fi

    echo "✓ ${name} downloaded."
}

# ── main ────────────────────────────────────────────
if [ $# -gt 0 ]; then
    for model in "$@"; do
        download_model "$model"
    done
else
    echo "Downloading all models to ${WEIGHTS_DIR} …"
    echo ""
    for model in $ALL_MODELS; do
        download_model "$model"
    done
    echo ""
    echo "All models downloaded ✓"
fi

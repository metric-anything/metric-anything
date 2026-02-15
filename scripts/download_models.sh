#!/usr/bin/env bash
# ──────────────────────────────────────────────────────
# Download model weights to the local weights/ directory
# ──────────────────────────────────────────────────────
# Usage:
#   bash scripts/download_models.sh              # download all
#   bash scripts/download_models.sh dav2-small   # download one
#
# Requirements: pip install huggingface_hub

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
WEIGHTS_DIR="${REPO_ROOT}/weights"

declare -A MODEL_MAP=(
    ["dav2-small"]="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
    ["dav2-base"]="depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf"
    ["dav2-large"]="depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
    ["metric-anything"]="yjh001/metricanything_student_pointmap"
)

download_model() {
    local name="$1"
    local hf_id="${MODEL_MAP[$name]}"
    local dest="${WEIGHTS_DIR}/${name}"

    if [ -d "$dest" ] && [ "$(ls -A "$dest" 2>/dev/null)" ]; then
        echo "✓ ${name} already exists at ${dest}, skipping."
        return
    fi

    echo "⬇ Downloading ${name} (${hf_id}) → ${dest} …"
    mkdir -p "$dest"
    huggingface-cli download "$hf_id" --local-dir "$dest"
    echo "✓ ${name} downloaded."
}

# ── main ────────────────────────────────────────────
if [ $# -gt 0 ]; then
    # Download specific model(s)
    for model in "$@"; do
        if [[ -v "MODEL_MAP[$model]" ]]; then
            download_model "$model"
        else
            echo "✗ Unknown model: ${model}"
            echo "  Available: ${!MODEL_MAP[*]}"
            exit 1
        fi
    done
else
    # Download all
    echo "Downloading all models to ${WEIGHTS_DIR} …"
    echo ""
    for model in "${!MODEL_MAP[@]}"; do
        download_model "$model"
    done
    echo ""
    echo "All models downloaded ✓"
fi

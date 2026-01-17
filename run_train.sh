#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG_FILE:-$HOME/.cache/huggingface/accelerate/default_config.yaml}"

if [ ! -f "$ACCELERATE_CONFIG_FILE" ]; then
  echo "Creating Accelerate config at $ACCELERATE_CONFIG_FILE"
  accelerate config default --config_file "$ACCELERATE_CONFIG_FILE"
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$ROOT_DIR"
accelerate launch \
  --multi_gpu \
  --num_processes 2 \
  --config_file "$ACCELERATE_CONFIG_FILE" \
  train.py "$@"

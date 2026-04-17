#!/bin/bash
# Setup environment for pi0.5 Libero fine-tuning
set -euo pipefail

cd "$(dirname "$0")/../.."
PROJECT_ROOT=$(pwd)

# Use scratch for HuggingFace caches (avoid filling $HOME quota).
# Override by exporting HF_HOME / HF_LEROBOT_HOME before running this script.
export HF_HOME=${HF_HOME:-/scratch/$USER/huggingface}
export HF_LEROBOT_HOME=${HF_LEROBOT_HOME:-$HF_HOME/lerobot}
mkdir -p "$HF_HOME" "$HF_LEROBOT_HOME"

# Create venv with Python 3.11
uv venv --python 3.11 .venv-pi05
UV_PROJECT_ENVIRONMENT=.venv-pi05 uv sync

# Patch transformers for PyTorch model support
cp -r ./src/openpi/models_pytorch/transformers_replace/* \
    .venv-pi05/lib/python3.11/site-packages/transformers/

# NOTE: On HPC we rsync the pre-converted PyTorch checkpoint and norm stats
# from the local machine (storage.googleapis.com is blocked here), so the
# JAX->PyTorch conversion and compute_norm_stats steps are skipped.
# Expected artifacts (rsync these in before training):
#   checkpoints/pi05_base_pytorch/
#   assets/pi05_libero/physical-intelligence/libero/norm_stats.json
#   assets/pi05_libero_discrete/physical-intelligence/libero/norm_stats.json

echo "Setup complete."
echo "  HF_HOME=$HF_HOME"
echo "  HF_LEROBOT_HOME=$HF_LEROBOT_HOME"
echo "  Make sure the following exist (rsync from local if missing):"
echo "    checkpoints/pi05_base_pytorch/"
echo "    assets/pi05_libero/ and assets/pi05_libero_discrete/"

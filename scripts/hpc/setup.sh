#!/bin/bash
# Setup environment for pi0.5 Libero fine-tuning
set -euo pipefail

cd "$(dirname "$0")/../.."
PROJECT_ROOT=$(pwd)

# Create venv with Python 3.11
uv venv --python 3.11 .venv-pi05
UV_PROJECT_ENVIRONMENT=.venv-pi05 uv sync

# Patch transformers for PyTorch model support
cp -r ./src/openpi/models_pytorch/transformers_replace/* \
    .venv-pi05/lib/python3.11/site-packages/transformers/

# Convert JAX base checkpoint to PyTorch
CUDA_VISIBLE_DEVICES=0 UV_PROJECT_ENVIRONMENT=.venv-pi05 uv run \
    examples/convert_jax_model_to_pytorch.py \
    --checkpoint-dir gs://openpi-assets/checkpoints/pi05_base \
    --config-name pi05_libero \
    --output-path checkpoints/pi05_base_pytorch

# Compute normalization stats (needed for both configs)
UV_PROJECT_ENVIRONMENT=.venv-pi05 uv run \
    scripts/compute_norm_stats.py --config-name pi05_libero
UV_PROJECT_ENVIRONMENT=.venv-pi05 uv run \
    scripts/compute_norm_stats.py --config-name pi05_libero_discrete

echo "Setup complete."
echo "  Checkpoint: checkpoints/pi05_base_pytorch/"
echo "  Norm stats: assets/pi05_libero/ and assets/pi05_libero_discrete/"

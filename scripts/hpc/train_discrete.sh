#!/bin/bash
# Train pi0.5 on Libero (MSE + discrete action token CE loss)
set -euo pipefail

cd "$(dirname "$0")/../.."

export HF_HOME=${HF_HOME:-/scratch/$USER/huggingface}
export HF_LEROBOT_HOME=${HF_LEROBOT_HOME:-$HF_HOME/lerobot}
export WANDB_ENTITY=${WANDB_ENTITY:-ai4ce}
export WANDB_PROJECT=${WANDB_PROJECT:-openpi}

NUM_GPUS=${NUM_GPUS:-2}
MASTER_PORT=${MASTER_PORT:-29501}
EXP_NAME=${EXP_NAME:-libero_discrete}

UV_PROJECT_ENVIRONMENT=.venv-pi05 .venv-pi05/bin/torchrun \
    --standalone \
    --nproc_per_node="$NUM_GPUS" \
    --master-port="$MASTER_PORT" \
    scripts/train_pytorch.py pi05_libero_discrete \
    --exp-name "$EXP_NAME"

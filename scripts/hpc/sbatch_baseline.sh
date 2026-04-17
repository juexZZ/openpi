#!/bin/bash
#SBATCH --job-name=pi05_libero_baseline
#SBATCH --partition=h200_tandon,h100_tandon
#SBATCH --constraint="h100|h200"
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j_pi05_libero_baseline.out
#SBATCH --account=torch_pr_50_tandon_advanced
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jz4725@nyu.edu

# ---------------------------------------------------------------------------
# pi0.5 Libero baseline (MSE only) on NYU HPC, inside Singularity.
#
# Prereqs (one-time, on a login node):
#   1) wandb login            # writes ~/.netrc so jobs auto-auth
#   2) bash scripts/hpc/setup.sh
#   3) rsync from local:
#        checkpoints/pi05_base_pytorch/
#        assets/pi05_libero/  assets/pi05_libero_discrete/
#
# Usage:
#   sbatch scripts/hpc/sbatch_baseline.sh
#   NUM_GPUS=4 EXP_NAME=baseline_run1 sbatch scripts/hpc/sbatch_baseline.sh
# ---------------------------------------------------------------------------

set -euo pipefail

OPENPI_DIR=${OPENPI_DIR:-/scratch/$USER/openpi}
SIF_IMAGE=${SIF_IMAGE:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}

# HuggingFace caches on scratch (avoid $HOME quota).
export HF_HOME=${HF_HOME:-/scratch/$USER/huggingface}
export HF_LEROBOT_HOME=${HF_LEROBOT_HOME:-$HF_HOME/lerobot}
mkdir -p "$HF_HOME" "$HF_LEROBOT_HOME"

# Forward run-time knobs to the inner train script.
export NUM_GPUS=${NUM_GPUS:-2}
export MASTER_PORT=${MASTER_PORT:-29500}
export EXP_NAME=${EXP_NAME:-libero_baseline}

mkdir -p "$OPENPI_DIR/logs"
cd "$OPENPI_DIR"

# Bind ~/.netrc so wandb auth (set up via `wandb login`) is visible inside the container.
singularity exec --nv \
    --bind "$HOME/.netrc:/root/.netrc:ro" \
    --bind "$HF_HOME:$HF_HOME" \
    --bind "$OPENPI_DIR:$OPENPI_DIR" \
    "$SIF_IMAGE" \
    bash -c "
        set -euo pipefail
        cd '$OPENPI_DIR'
        source .venv-pi05/bin/activate
        export HF_HOME='$HF_HOME'
        export HF_LEROBOT_HOME='$HF_LEROBOT_HOME'
        export NUM_GPUS='$NUM_GPUS'
        export MASTER_PORT='$MASTER_PORT'
        export EXP_NAME='$EXP_NAME'
        bash scripts/hpc/train_baseline.sh
    "

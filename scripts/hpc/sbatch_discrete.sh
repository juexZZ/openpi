#!/bin/bash
#SBATCH --job-name=pi05_libero_discrete
#SBATCH --partition=h200_tandon,h100_tandon
#SBATCH --constraint="h100|h200"
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j_pi05_libero_discrete.out
#SBATCH --account=torch_pr_50_tandon_advanced
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jz4725@nyu.edu

# ---------------------------------------------------------------------------
# pi0.5 Libero with discrete action token CE loss, on NYU HPC, in Singularity.
# See scripts/hpc/sbatch_baseline.sh for prereqs.
#
# Usage:
#   sbatch scripts/hpc/sbatch_discrete.sh
#   NUM_GPUS=4 EXP_NAME=discrete_run1 sbatch scripts/hpc/sbatch_discrete.sh
# ---------------------------------------------------------------------------

set -euo pipefail

OPENPI_DIR=${OPENPI_DIR:-/scratch/$USER/openpi}
SIF_IMAGE=${SIF_IMAGE:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}

export HF_HOME=${HF_HOME:-/scratch/$USER/huggingface}
export HF_LEROBOT_HOME=${HF_LEROBOT_HOME:-$HF_HOME/lerobot}
mkdir -p "$HF_HOME" "$HF_LEROBOT_HOME"

export NUM_GPUS=${NUM_GPUS:-2}
export MASTER_PORT=${MASTER_PORT:-29501}
export EXP_NAME=${EXP_NAME:-libero_discrete}

mkdir -p "$OPENPI_DIR/logs"
cd "$OPENPI_DIR"

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
        bash scripts/hpc/train_discrete.sh
    "

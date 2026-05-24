#!/bin/bash
#SBATCH --job-name=lr_unet_autoencoder_train
#SBATCH --account=hai_1164
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=05:00:00
#SBATCH --output=%x_%j.out

set -euo pipefail

# Resolve repo root relative to this script so the job works regardless of
# where it is submitted from.
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$REPO_DIR/logs"
CONFIG_PATH="$REPO_DIR/configs/train.yaml"
VENV_PATH="$REPO_DIR/.venv"

mkdir -p "$LOG_DIR"
module load Python/3.13.5
module load uv/0.8.17

source "$VENV_PATH/bin/activate"

export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
export UV_NO_MANAGED_PYTHON=1

cd "$REPO_DIR"

uv run ae-train --config "$CONFIG_PATH"
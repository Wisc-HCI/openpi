#!/bin/bash
set -euo pipefail

echo "Host: $(hostname)"
echo "Start: $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

cd "$_CONDOR_SCRATCH_DIR"
tar -xzf openpi.tar.gz
cd openpi

export HOME="$PWD/home"
export UV_CACHE_DIR="$PWD/.uv_cache"
export HF_HOME="$PWD/.hf_cache"
#########
export HF_LEROBOT_HOME="$HF_HOME/lerobot"
########
export OPENPI_DATA_HOME="$PWD/.openpi_cache"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.97
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export WANDB_MODE=offline

mkdir -p "$HOME" "$UV_CACHE_DIR" "$HF_HOME" "$OPENPI_DATA_HOME"

# mkdir -p "$HF_HOME/lerobot/Wisc-HCI"
# tar -xzf "$_CONDOR_SCRATCH_DIR/smoothie2.tar.gz" -C "$HF_HOME/lerobot/Wisc-HCI/"

########### update
DATASET_ROOT="$HF_LEROBOT_HOME/Wisc-HCI/libero_legibility_v1"
mkdir -p "$HF_LEROBOT_HOME/Wisc-HCI"

tar -xzf "$_CONDOR_SCRATCH_DIR/libero_legibility_v1.tar.gz" \
  -C "$HF_LEROBOT_HOME/Wisc-HCI"

if [[ ! -f "$DATASET_ROOT/meta/info.json" ]]; then
  echo "ERROR: Dataset metadata was not extracted to:"
  echo "$DATASET_ROOT/meta/info.json"
  tar -tzf "$_CONDOR_SCRATCH_DIR/libero_legibility_v1.tar.gz" | head -5
  exit 2
fi

echo "Dataset extracted successfully:"
ls -lah "$DATASET_ROOT/meta"
#########

export PATH="$PWD/bin:$HOME/.local/bin:$PATH"

echo "uv path: $(command -v uv || true)"
if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv was not found. Expected it at $PWD/bin/uv"
  ls -lah "$PWD/bin" || true
  exit 127
fi

uv --version

uv python install 3.11
uv venv --python 3.11

GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

mkdir -p "$PWD/checkpoints"

# each chtc account only have default 40GB storage, so only keep the latest checkpoint for efficient usage
uv run scripts/train.py pi05_droid_finetune \
  --exp-name=libero_legibility_v1 \
  --checkpoint-base-dir "$PWD/checkpoints" \
  --batch-size=2 \
  --num-train-steps=20000 \
  --save-interval=5000 \
  --keep-period=None \
  --ema-decay=None \
  --no-wandb-enabled

echo "Packaging checkpoints..."
tar -czf "$_CONDOR_SCRATCH_DIR/libero_legibility_v1_ckpt.tar.gz" -C "$_CONDOR_SCRATCH_DIR/openpi" checkpoints/
echo "End: $(date)"

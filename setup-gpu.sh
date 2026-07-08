#!/bin/bash
set -euo pipefail

# create environment:
UV_VENV_CLEAR=1 uv venv --python 3.12

# activate environment:
source .venv/bin/activate

# install standard gpu dependencies:
uv pip install -r requirements-gpu.txt --index-strategy unsafe-best-match

# Optional accelerations (Hopper / H200). These are NOT required — the trainer
# falls back to SDPA attention and bf16/fp16 when they are absent — so failures
# here are non-fatal.

# flash attention (best on Hopper; code auto-detects and uses SDPA otherwise):
python -m ensurepip --default-pip
python -m pip install flash-attn-3 --extra-index-url https://download.pytorch.org/whl/cu130 || \
  echo "flash-attn unavailable; continuing with SDPA attention."

# set up environment variables for cudnn:
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib
export C_INCLUDE_PATH=$C_INCLUDE_PATH:$(pwd)/.venv/lib/python3.12/site-packages/nvidia/cudnn/include
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$(pwd)/.venv/lib/python3.12/site-packages/nvidia/cudnn/include

# TransformerEngine enables FP8 on Hopper; skipped gracefully if the build fails:
uv pip install "transformer-engine[pytorch]" --no-build-isolation || \
  echo "transformer-engine unavailable; FP8 disabled, continuing with bf16."
uv pip install nvidia-cublas --extra-index-url=https://download.pytorch.org/whl/cu130 --force-reinstall || true

# to use flash attention within TransformerEngine, prefer its flash backend:
export NVTE_FUSED_ATTN=0
export NVTE_FLASH_ATTN=1

# set huggingface cache location:
export HF_HOME="./hf_cache"

# start app:
.venv/bin/python train.py

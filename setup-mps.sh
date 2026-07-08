#!/bin/bash
set -euo pipefail

# create environment:
UV_VENV_CLEAR=1 uv venv --python 3.12

# activate environment:
source .venv/bin/activate

# install mps-compatible dependencies:
uv pip install -r requirements-mps.txt

# Apple Silicon (MPS) has no Flash-Attention kernels; the model falls back to
# PyTorch SDPA automatically — nothing extra to install.

# set huggingface cache location:
export HF_HOME="./hf_cache"

# start app:
.venv/bin/python train.py

#!/bin/bash
set -euo pipefail

# create environment:
UV_VENV_CLEAR=1 uv venv --python 3.12

# activate environment:
source .venv/bin/activate

# install mps-compatible standard dependencies:
uv pip install -r requirements-mps.txt

# install mps flash attention (Apple Silicon specific):
python -m ensurepip --default-pip
python -m pip install mps-flash-attn

# set huggingface cache location:
export HF_HOME="./hf_cache"

# start app:
.venv/bin/python train.py
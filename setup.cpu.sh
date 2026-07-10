#!/bin/bash
set -euo pipefail

# create environment:
UV_VENV_CLEAR=1 uv venv --python 3.14

# activate environment:
source .venv/bin/activate

# install cpu dependencies:
uv pip install -r requirements-cpu.txt --index-strategy unsafe-best-match

# set huggingface cache location:
export HF_HOME="./hf_cache"

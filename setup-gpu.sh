#!/bin/bash

# create environment:
UV_VENV_CLEAR=1 uv venv --python 3.12

# activate environment:
source .venv/bin/activate

# install standard gpu dependencies:
uv pip install -r requirements-gpu.txt

# install flash-attn using standard pip as requested:
python -m ensurepip --default-pip
python -m pip install flash-attn-3 --extra-index-url https://download.pytorch.org/whl/cu130

# set up environment variables for cudnn:
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ML/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib
export C_INCLUDE_PATH=$C_INCLUDE_PATH:/home/ML/.venv/lib/python3.12/site-packages/nvidia/cudnn/include
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:/home/ML/.venv/lib/python3.12/site-packages/nvidia/cudnn/include

# install specific packages requiring special flags:
uv pip install "transformer-engine[pytorch]" --no-build-isolation
uv pip install nvidia-cublas --extra-index-url=https://download.pytorch.org/whl/cu130 --force-reinstall

# to use flash attention, set NVTE_FUSED_ATTN=0 and NVTE_FLASH_ATTN=1:
export NVTE_FUSED_ATTN=0
export NVTE_FLASH_ATTN=1

# set huggingface cache location:
export HF_HOME="./hf_cache"

# start app:
.venv/bin/python train.py
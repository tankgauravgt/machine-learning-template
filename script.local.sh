#!/bin/bash

# create environment:
UV_VENV_CLEAR=1 uv venv --python 3.12

# activate environment:
source .venv/bin/activate

# install dependencies:
uv pip install jarvislabs
uv pip install tensorboard
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
uv pip install datasets tokenizers transformers pyarrow accelerate
python -m ensurepip --default-pip
python -m pip install flash-attn-3 --extra-index-url https://download.pytorch.org/whl/cu130
uv pip install wheel
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ML/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib
export C_INCLUDE_PATH=$C_INCLUDE_PATH:/home/ML/.venv/lib/python3.12/site-packages/nvidia/cudnn/include
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:/home/ML/.venv/lib/python3.12/site-packages/nvidia/cudnn/include
uv pip install "transformer-engine[pytorch]" --no-build-isolation
uv pip install nvidia-cublas --extra-index-url=https://download.pytorch.org/whl/cu130 --force-reinstall

# to use flash attention, set `NVTE_FUSED_ATTN=0` and `NVTE_FLASH_ATTN=1`:
export NVTE_FUSED_ATTN=0
export NVTE_FLASH_ATTN=1

# start app:
.venv/bin/python train.py

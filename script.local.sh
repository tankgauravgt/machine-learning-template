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
uv pip install flash-attn-3 --no-deps --index-url https://download.pytorch.org/whl/cu130
wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn_jit/linux-x86_64/cudnn_jit-linux-x86_64-9.23.1.3_cuda13-archive.tar.xz -O cudnn.tar.xz
tar -xf cudnn.tar.xz
cp cudnn_jit-linux-x86_64-9.23.1.3_cuda13-archive/include/* /usr/local/cuda/include/
cp cudnn_jit-linux-x86_64-9.23.1.3_cuda13-archive/lib/* /usr/local/cuda/lib64/
uv pip install "transformer-engine[pytorch]"

# start app:
.venv/bin/python train.py

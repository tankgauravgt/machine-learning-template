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

# start app:
.venv/bin/python train.py

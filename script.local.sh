#!/bin/bash

# create environment:
UV_VENV_CLEAR=1 uv venv

# activate environment:
source .venv/bin/activate

# install dependencies:
uv pip install -r requirements.txt
uv pip install pip wheel setuptools ninja packaging
uv pip install flash-attn-3 --no-deps --index-url https://download.pytorch.org/whl/cu128

# start app:
python train.py

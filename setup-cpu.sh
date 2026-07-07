#!/bin/bash

# create environment:
UV_VENV_CLEAR=1 uv venv --python 3.12

# activate environment:
source .venv/bin/activate

# install cpu dependencies:
uv pip install -r requirements-cpu.txt

# set huggingface cache location:
export HF_HOME="./hf_cache"

# start app:
.venv/bin/python train.py
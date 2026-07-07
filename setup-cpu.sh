#!/bin/bash

# create environment:
UV_VENV_CLEAR=1 uv venv --python 3.12

# activate environment:
source .venv/bin/activate

# install cpu dependencies:
uv pip install -r requirements-cpu.txt

# start app:
.venv/bin/python train.py
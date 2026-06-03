#!/bin/bash

# create environment:
UV_VENV_CLEAR=1 uv venv

# activate environment:
source .venv/bin/activate

# install dependencies:
uv pip install -r requirements.txt

# start app:
python train.py
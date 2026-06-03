#!/bin/bash

UV_VENV_CLEAR=1 uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
python main.py
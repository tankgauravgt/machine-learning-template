#!/bin/bash
set -euo pipefail

# create environment:
UV_VENV_CLEAR=1 uv venv --python 3.14

# activate environment:
source .venv/bin/activate

# set num processors for installation:
export MAX_JOBS=$(nproc)

# install gpu dependencies:
UV_TORCH_BACKEND=auto uv pip install -r requirements-gpu.txt --index-strategy unsafe-best-match

# install pip:
uv run python -m ensurepip --default-pip

# source paths for compilation:
source .env


# =========================================================
# INSTALL FLASH ATTENTION:
# =========================================================

COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)

echo "Detected Compute Capability: $COMPUTE_CAP"

# 2. Strip the decimal for bash integer comparison (e.g., "8.0" -> "80", "9.0" -> "90")
CAP_INT=$(echo "$COMPUTE_CAP" | tr -d '.')

export FLASH_ATTN_CUDA_ARCHS="$CAP_INT"

# 3. Conditionally install based on the hardware capability
if [ "$CAP_INT" -ge 90 ]; then
    echo "Hopper architecture detected. Installing FlashAttention-3..."
    UV_TORCH_BACKEND=auto uv pip install flash_attn_3 --no-build-isolation

elif [ "$CAP_INT" -ge 80 ]; then
    echo "Ampere/Ada architecture detected. Installing FlashAttention-2..."
    FA_BASE="https://000010000000100100000000.blob.core.windows.net/semantictank-wheels"
    FA2_WHEEL="flash_attn-2.8.3+cu130torch2.12-cp314-cp314-linux_x86_64.whl"
    FA2_URL="$FA_BASE/$FA2_WHEEL"
    wget -q --show-progress $FA2_URL
    UV_TORCH_BACKEND=auto uv pip install $FA2_WHEEL

else
    echo "Warning: Compute capability $COMPUTE_CAP is below 8.0."
    echo "Flash Attention requires an Ampere card or newer."
fi


# =========================================================
# INSTALL TRANSFORMER ENGINE:
# =========================================================

# install transformer-engine:
UV_TORCH_BACKEND=auto uv pip install "transformer-engine[pytorch]" --no-build-isolation

# set huggingface cache location:
export HF_HOME="./hf_cache"

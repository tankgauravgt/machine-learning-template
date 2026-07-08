# BERT Base MLM

A BERT base Masked Language Model (MLM) trained from scratch on [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb), using Hugging Face Transformers. The code auto-detects the available hardware and runs unchanged on CPU, Apple Silicon (MPS), any CUDA GPU, and NVIDIA H200 (Hopper) — enabling the fastest kernels each platform supports.

## Architecture

| Parameter | Value |
|---|---|
| Model | BERT base |
| Hidden size | 768 |
| Attention heads | 12 |
| Transformer layers | 12 |
| FFN intermediate size | 3072 |
| Max sequence length | 512 |
| MLM mask probability | 15% |
| Tokeniser | BPE (trained on corpus) |

## Hardware Optimisations

Optimisations are enabled automatically based on the detected device (`src/hardware.py`). Each `use_*` flag in `src/config.py` defaults to `"auto"`; set it to `True`/`False` to force, and an unsupported forced flag is clamped with a warning rather than crashing.

| Optimisation | Enabled on |
|---|---|
| **FP8 compute** (TransformerEngine + Accelerate) | Hopper (H200) with TransformerEngine installed |
| **BF16 weights** | CUDA GPUs with bf16 support, and MPS |
| **Flash Attention** (falls back to SDPA) | CUDA with `flash-attn` installed; SDPA everywhere else |
| **TF32** matmuls / cuDNN | Ampere+ CUDA GPUs |
| **`torch.compile`** (inductor) | CUDA and CPU (disabled on MPS) |
| **Fused AdamW** | CUDA (portable `adamw_torch` elsewhere) |
| **Pinned memory + prefetch** DataLoader | CUDA (single-process loader on MPS/CPU) |

## Prerequisites

- Python 3.10+
- One of: CPU, Apple Silicon (MPS), or an NVIDIA GPU (CUDA 12.x). FP8 additionally requires Hopper (H200).
- `uv` package manager (`pip install uv`)
- A Hugging Face account with `HF_TOKEN` set (FineWeb requires acceptance of terms)

Pick the setup script for your platform: `setup-cpu.sh`, `setup-mps.sh`, or `setup-gpu.sh`.

## Training

Each setup script creates a `uv` virtual environment, installs the platform's dependencies, and launches `train.py`.

```bash
export HF_TOKEN=<your_token>

bash setup-cpu.sh   # plain CPU (any OS)
bash setup-mps.sh   # Apple Silicon (MPS)
bash setup-gpu.sh   # NVIDIA CUDA GPU (FP8 fast path on H200)
```

### Cloud (H200 via JarvisLabs)

`launch.cloud.sh` provisions an H200 spot instance and runs `setup-gpu.sh` on it. Use it to provision, SSH in and monitor progress (TensorBoard on port 6006), or stream logs.

### Testing mode

To run a quick sanity-check on 2,000 samples before a full production run, set `is_testing=True` in `src/config.py`:

```python
config = MLMConfig(is_testing=True)
```

Or edit the default directly:

```python
# src/config.py
is_testing: bool = True
```

### Key config knobs (`src/config.py`)

| Field | Default | Description |
|---|---|---|
| `production_steps` | 400,000 | Total optimiser steps for a full run |
| `batch_size` | 256 | Per-device micro-batch size |
| `gradient_accumulation_steps` | 4 | Effective batch = 256 × 4 = 1,024 |
| `learning_rate` | 1e-4 | Peak LR (cosine schedule) |
| `warmup_steps` | 10,000 | Linear warmup steps |
| `max_length` | 512 | Tokeniser truncation / padding length |
| `tokenizer_samples` | 100,000 | Documents used to train the BPE tokeniser |

Checkpoints are saved every 5,000 steps to `checkpoints/`. The final model is also saved there at the end of training.

## Inference

After training completes, run:

```bash
python inference.py
```

This loads the tokeniser from `tokenizer_output/` and the model from `checkpoints/`, then fills `[MASK]` tokens in the test sentences defined in `inference.py`.

To predict on your own text, edit the `test_sentences` list at the bottom of `inference.py`:

```python
test_sentences = [
    "The capital of France is [MASK].",
    "She [MASK] the book on the table.",
]
```

The predictor picks the highest-probability token for each `[MASK]` position and prints it alongside the original input.

## Project Structure

```
.
├── src/
│   ├── config.py          # All hyperparameters and flags
│   ├── hardware.py        # Runtime device detection + acceleration toggles
│   ├── data_pipeline.py   # Dataset streaming, BPE tokeniser training, collation
│   ├── model.py           # BERT base model factory
│   └── trainer.py         # HF Trainer wrapper with hardware-adaptive TrainingArguments
├── train.py               # Entry point — wires pipeline → model → trainer
├── inference.py           # Loads checkpoint and predicts masked tokens
├── requirements-cpu.txt   # CPU dependencies
├── requirements-mps.txt   # Apple Silicon (MPS) dependencies
├── requirements-gpu.txt   # CUDA GPU dependencies
├── setup-cpu.sh           # CPU run script (uv venv + train)
├── setup-mps.sh           # Apple Silicon run script
├── setup-gpu.sh           # CUDA GPU run script (optional FP8 / flash-attn)
└── launch.cloud.sh        # JarvisLabs REPL — provision / SSH / logs / destroy
```

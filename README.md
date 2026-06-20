# BERT Base MLM

A BERT base Masked Language Model (MLM) trained from scratch on [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb), using Hugging Face Transformers and optimised for NVIDIA H200 (Hopper) GPUs.

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

- **FP8 compute** via NVIDIA TransformerEngine + Accelerate (`ACCELERATE_MIXED_PRECISION=fp8`)
- **BF16 master weights** to prevent gradient underflow
- **Flash Attention 2** for O(n) memory attention
- **TF32** enabled globally for matmuls and cuDNN
- **`torch.compile`** for kernel fusion
- **Fused AdamW** (`adamw_torch_fused`) kernel
- **Pinned memory + prefetch** DataLoader (`num_workers=4`, `prefetch_factor=2`)
- **`group_by_length`** to minimise padding waste per batch

## Prerequisites

- Python 3.10+
- NVIDIA H200 GPU (Hopper architecture required for FP8)
- CUDA 12.1+
- `uv` package manager (`pip install uv`)
- A Hugging Face account with `HF_TOKEN` set (FineWeb requires acceptance of terms)

## Training

### Cloud (H200 via JarvisLabs)

```bash
export HF_TOKEN=<your_token>
bash script.cloud.sh
```

Select **option 1** to provision an H200 spot instance and start training. The script handles environment setup, dependency installation, and launches `train.py` automatically.

Use **option 2** to SSH in and monitor progress (TensorBoard forwarded on port 6006), and **option 3** to stream logs directly.

### Local

```bash
export HF_TOKEN=<your_token>
bash script.local.sh
```

This creates a virtual environment with `uv`, installs dependencies, and runs `train.py`.

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
│   ├── data_pipeline.py   # Dataset streaming, BPE tokeniser training, collation
│   ├── model.py           # BERT base model factory
│   └── trainer.py         # HF Trainer wrapper with H200-optimised TrainingArguments
├── train.py               # Entry point — wires pipeline → model → trainer
├── inference.py           # Loads checkpoint and predicts masked tokens
├── requirements.txt       # Python dependencies
├── script.local.sh        # Local run script (uv venv + train)
└── script.cloud.sh        # JarvisLabs REPL — provision / SSH / logs / destroy
```

"""Configuration parameters for model architecture and training."""
from dataclasses import dataclass

@dataclass
class MLMConfig:
    # Execution Mode
    is_testing: bool = False
    test_samples: int = 2000
    tokenizer_samples: int = 100000

    # Define an explicit step ceiling for production runs to satisfy the learning rate scheduler
    production_steps: int = 400_000

    # Data Parameters
    dataset_path: str = "HuggingFaceFW/fineweb"
    dataset_name: str = "sample-10BT"
    max_length: int = 512           # BERT base standard sequence length
    mlm_probability: float = 0.15

    # Model Architecture (BERT base)
    d_model: int = 768
    nhead: int = 12
    num_layers: int = 12
    dropout: float = 0.1

    # Training Hyperparameters
    # Per-device batch of 256 × 4 gradient accumulation steps = 1024 effective batch size
    batch_size: int = 256
    gradient_accumulation_steps: int = 4
    epochs: int = 20              # Note: Epochs are only utilised during testing mode
    learning_rate: float = 1e-4
    warmup_steps: int = 10_000
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"

    # DataLoader
    num_workers: int = 16
    shuffle_buffer_size: int = 10_000

    # Hardware Optimisation Flags (Hopper / H200)
    use_bf16: bool = True
    use_tf32: bool = True
    use_torch_compile: bool = True
    use_flash_attention: bool = True

    # Directories
    checkpoint_dir: str = "checkpoints"
    tokenizer_dir: str = "tokenizer_output"
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
    max_length: int = 128
    mlm_probability: float = 0.15

    # Model Architecture
    d_model: int = 256
    nhead: int = 4
    num_layers: int = 4
    dropout: float = 0.1

    # Training Hyperparameters
    batch_size: int = 32 * 32
    epochs: int = 20              # Note: Epochs are only utilised during testing mode
    learning_rate: float = 1e-4

    # Hardware Optimisation Flags (e.g., for Hopper Architectures)
    use_bf16: bool = True
    use_tf32: bool = True
    use_torch_compile: bool = True

    # Directories
    checkpoint_dir: str = "checkpoints"
    tokenizer_dir: str = "tokenizer_output"
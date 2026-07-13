"""
Configuration file.
"""

from hardware import HardwareManager
from pydantic import BaseModel


class MLMConfig(BaseModel):

    # =====================================================
    # Data Parameters
    # =====================================================

    # dataset config:
    # ---------------
    dataset_path: str = "HuggingFaceFW/fineweb"
    dataset_name: str = "sample-10BT"
    max_length: int = 512
    mlm_probability: float = 0.15
    
    # dataloader config:
    # ------------------
    tokenizer_samples: int = 100000
    batch_size: int = 256

    # =====================================================
    # Model Architecture
    # =====================================================

    d_model: int = 768
    nhead: int = 12
    num_layers: int = 12
    dropout: float = 0.1

    # =====================================================
    # Training Hyperparameters
    # =====================================================
    
    is_testing: bool = False
    
    # testing = True
    # -------------- >>
    testrun_epochs: int = 20
    testrun_samples: int = 2000
    # -------------- <<


    # testing = False
    # --------------- >>
    training_steps: int = 400_000
    # --------------- <<

    gradient_accumulation_steps: int = 4
    
    warmup_steps: int = 10_000
    lr_scheduler_type: str = "cosine"
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    
    # =====================================================
    # Directories
    # =====================================================

    tokenizer_dir: str = "tokenizer_output"
    cache_dir: str = "cache"
    checkpoint_dir: str = "checkpoints"

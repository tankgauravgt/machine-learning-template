import math
import torch
import torch._dynamo
from transformers import Trainer, TrainingArguments

from src.config import MLMConfig
from src.hardware import HardwareInfo

class MLMTrainer:
    """Manages the training lifecycle using the Hugging Face Trainer API."""

    def __init__(self, config: MLMConfig, model, dataset, collator, hw: HardwareInfo):
        self.config = config
        self.model = model
        self.dataset = dataset
        self.collator = collator
        self.hw = hw

    def execute(self):
        """Compiles training arguments and initiates the training process."""

        if self.config.is_testing:
            print(f"--- EXECUTING IN TESTING MODE ({self.config.test_samples} Samples) ---")
            steps_per_epoch = math.ceil(
                self.config.test_samples
                / (self.config.batch_size * self.config.gradient_accumulation_steps)
            )
            calculated_max_steps = steps_per_epoch * self.config.epochs
            calculated_save_steps = steps_per_epoch
        else:
            print(f"--- EXECUTING IN PRODUCTION MODE ({self.config.production_steps:,} Steps) ---")
            # Supply an explicit integer to satisfy the learning rate scheduler on streaming datasets
            calculated_max_steps = self.config.production_steps
            calculated_save_steps = 5000

        if self.hw.torch_compile:
            # Allow .item() calls to be captured rather than breaking the graph
            torch._dynamo.config.capture_scalar_outputs = True
            # Treat layer_idx as dynamic so compile doesn't respecialize per layer
            torch._dynamo.config.allow_unspec_int_on_nn_module = True

        # Multi-process DataLoaders are only worthwhile with pinned CUDA memory;
        # on MPS/CPU workers add fork/pickling overhead for little gain.
        workers = self.config.num_workers if self.hw.device == "cuda" else 0

        dataloader_kwargs = {}
        if workers > 0:
            dataloader_kwargs["dataloader_prefetch_factor"] = self.config.prefetch_factor
            dataloader_kwargs["dataloader_persistent_workers"] = True

        training_args = TrainingArguments(
            output_dir=self.config.checkpoint_dir,
            max_steps=calculated_max_steps,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            lr_scheduler_type=self.config.lr_scheduler_type,
            logging_steps=10,
            save_steps=calculated_save_steps,
            remove_unused_columns=False,
            report_to="none",
            # DataLoader
            dataloader_pin_memory=self.hw.pin_memory,
            dataloader_num_workers=workers,
            dataloader_drop_last=True,
            # Hardware Acceleration (auto-detected per machine)
            # FP8 is activated via ACCELERATE_MIXED_PRECISION=fp8 (set in configure_environment).
            # bf16 is kept True alongside FP8 as the high-precision fallback for ops that don't support FP8.
            bf16=self.hw.bf16,
            fp16=False,
            tf32=self.hw.tf32,
            torch_compile=self.hw.torch_compile,
            torch_compile_backend="inductor" if self.hw.torch_compile else None,
            # no-cudagraphs: keeps kernel autotuning but avoids CUDA graph tensor aliasing
            # errors that occur when the MLM head output is overwritten between forward passes
            torch_compile_mode="max-autotune-no-cudagraphs" if self.hw.torch_compile else None,
            # Fused AdamW kernel on CUDA; portable AdamW elsewhere.
            optim="adamw_torch_fused" if self.hw.fused_optim else "adamw_torch",
            **dataloader_kwargs,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset,
            data_collator=self.collator,
        )

        trainer.train()

        # Persist the final model state
        trainer.save_model(self.config.checkpoint_dir)

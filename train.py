import os

# Avoid tokeniser fork warnings; safe to set unconditionally before torch import.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pyarrow as pa
# PyExtensionType was renamed to ExtensionType in pyarrow 15; restore the old name
# so that older datasets versions that still reference it can import cleanly.
if not hasattr(pa, "PyExtensionType"):
    pa.PyExtensionType = pa.ExtensionType

from src.config import MLMConfig
from src.hardware import detect_hardware, configure_environment
from src.data_pipeline import DataPipeline
from src.model import ModelFactory
from src.trainer import MLMTrainer

def main():
    config = MLMConfig()

    # Detect the machine and enable only supported accelerations. Must run before
    # the Trainer/Accelerator is built so FP8 mixed precision is picked up.
    hw = detect_hardware(config)
    configure_environment(hw)
    print(hw.summary())

    print("Initialising Data Pipeline...")
    pipeline = DataPipeline(config)

    print("Downloading dataset...")
    dataset = pipeline.load_dataset()

    print("Building tokeniser...")
    tokenizer = pipeline.build_tokenizer(dataset)

    print("Preparing tokenised dataset and collator...")
    tokenized_dataset, collator = pipeline.prepare_dataset(dataset)

    print("Constructing Hugging Face model architecture...")
    model = ModelFactory.create_masked_lm(config, tokenizer, hw)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters : {total_params:,}")

    print("Commencing training execution...")
    trainer = MLMTrainer(config, model, tokenized_dataset, collator, hw)
    trainer.execute()

    print("Training complete. Model and tokeniser successfully persisted.")

if __name__ == "__main__":
    main()

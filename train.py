import os

# Must be set before any CUDA / torch import to take effect
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")          # avoid tokeniser fork warnings
os.environ.setdefault("ACCELERATE_MIXED_PRECISION", "bf16")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # reduce fragmentation
os.environ.setdefault("NCCL_P2P_DISABLE", "0")

import pyarrow as pa
# PyExtensionType was renamed to ExtensionType in pyarrow 15; restore the old name
# so that older datasets versions that still reference it can import cleanly.
if not hasattr(pa, "PyExtensionType"):
    pa.PyExtensionType = pa.ExtensionType

import torch

# Enable TF32 globally before model/data init
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from src.config import MLMConfig
from src.data_pipeline import DataPipeline
from src.model import ModelFactory
from src.trainer import MLMTrainer

def main():
    config = MLMConfig()
    
    print("Initialising Data Pipeline...")
    pipeline = DataPipeline(config)
    dataset_stream = pipeline.load_stream()
    
    print("Building tokeniser...")
    tokenizer = pipeline.build_tokenizer(dataset_stream)
    
    print("Preparing tokenised dataset and collator...")
    # Re-initialise the stream for the mapping phase to ensure the iterator resets
    dataset_stream = pipeline.load_stream() 
    tokenized_dataset, collator = pipeline.prepare_dataset(dataset_stream)
    
    print("Constructing Hugging Face model architecture...")
    model = ModelFactory.create_masked_lm(config, tokenizer)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters : {total_params:,}")
    
    print("Commencing training execution...")
    trainer = MLMTrainer(config, model, tokenized_dataset, collator)
    trainer.execute()
    
    print("Training complete. Model and tokeniser successfully persisted.")

if __name__ == "__main__":
    main()
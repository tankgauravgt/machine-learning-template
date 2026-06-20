import os
import itertools
from typing import Tuple
from datasets import load_dataset, IterableDataset

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from transformers import PreTrainedTokenizerFast, DataCollatorForLanguageModeling

from src.config import MLMConfig

class DataPipeline:
    """Encapsulates data ingestion, tokenisation, and collation logic."""
    
    SPECIAL_TOKENS = ["[_UNK_]", "[_PAD_]", "[_INIT_]", "[_STOP_]"]

    def __init__(self, config: MLMConfig):
        self.config = config
        self.tokenizer: PreTrainedTokenizerFast = None

    def load_stream(self) -> IterableDataset:
        """Retrieves the streaming corpus without artificial boundaries."""
        return load_dataset(
            path=self.config.dataset_path,
            name=self.config.dataset_name,
            split="train",
            streaming=True,
        )

    def build_tokenizer(self, dataset: IterableDataset) -> PreTrainedTokenizerFast:
        """Trains a BPE tokeniser and persists it to disk."""
        tokenizer = Tokenizer(BPE(unk_token="[_UNK_]"))
        trainer = BpeTrainer(special_tokens=self.SPECIAL_TOKENS)
        
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()
        
        # Safely constrain the tokeniser training to prevent out-of-memory (OOM) errors
        limited_dataset = itertools.islice(dataset, self.config.tokenizer_samples)
        text_iterator = (example["text"] for example in limited_dataset)
        
        tokenizer.train_from_iterator(text_iterator, trainer, length=self.config.tokenizer_samples)
        
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            unk_token="[_UNK_]",
            pad_token="[_PAD_]",
            bos_token="[_INIT_]",
            eos_token="[_STOP_]",
        )
        self.tokenizer.add_special_tokens({"mask_token": "[MASK]"})
        
        os.makedirs(self.config.tokenizer_dir, exist_ok=True)
        self.tokenizer.save_pretrained(self.config.tokenizer_dir)
        
        return self.tokenizer

    def prepare_dataset(self, dataset: IterableDataset) -> Tuple[IterableDataset, DataCollatorForLanguageModeling]:
        """Maps the tokeniser to the dataset and prepares the Hugging Face collator."""
        if not self.tokenizer:
            raise ValueError("Tokeniser must be built before preparing the dataset.")

        def tokenize_fn(batch):
            return self.tokenizer(
                batch["text"],
                truncation=True,
                max_length=self.config.max_length,
                padding="max_length",
            )

        tokenized_dataset = dataset.map(tokenize_fn, batched=True, batch_size=1000)
        
        # Isolate numerical tensors and discard string metadata
        tokenized_dataset = tokenized_dataset.select_columns(["input_ids", "attention_mask"])
        tokenized_dataset = tokenized_dataset.with_format("torch")

        collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, 
            mlm=True, 
            mlm_probability=self.config.mlm_probability
        )
        
        return tokenized_dataset, collator
import os
import itertools
from typing import Tuple
from datasets import load_dataset, Dataset

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

    def load_dataset(self) -> Dataset:
        """Downloads the full dataset to the HuggingFace cache and returns a Dataset."""
        return load_dataset(
            path=self.config.dataset_path,
            name=self.config.dataset_name,
            split="train",
            streaming=False,
        )

    def build_tokenizer(self, dataset: Dataset) -> PreTrainedTokenizerFast:
        """Trains a BPE tokeniser on the first tokenizer_samples examples and persists it."""
        tokenizer = Tokenizer(BPE(unk_token="[_UNK_]"))
        trainer = BpeTrainer(special_tokens=self.SPECIAL_TOKENS)

        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()

        limited = itertools.islice(dataset, self.config.tokenizer_samples)
        text_iterator = (example["text"] for example in limited)

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

    def prepare_dataset(self, dataset: Dataset) -> Tuple[Dataset, DataCollatorForLanguageModeling]:
        """Tokenises the dataset in parallel, caches the result to disk, and returns it."""
        if not self.tokenizer:
            raise ValueError("Tokeniser must be built before preparing the dataset.")

        def tokenize_fn(batch):
            return self.tokenizer(
                batch["text"],
                truncation=True,
                max_length=self.config.max_length,
                padding="max_length",
            )

        tokenized_dataset = dataset.map(
            tokenize_fn,
            batched=True,
            batch_size=1000,
            num_proc=self.config.tokenize_num_proc,
            remove_columns=dataset.column_names,
            desc="Tokenising",
        )

        tokenized_dataset = tokenized_dataset.shuffle(seed=42)
        tokenized_dataset.set_format("torch")

        collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=self.config.mlm_probability,
        )

        return tokenized_dataset, collator

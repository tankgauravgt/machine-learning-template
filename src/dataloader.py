"""
Data loading and preprocessing utilities.
"""

import itertools
import hashlib
import json
import os

from typing import Tuple

# import datasets:
from datasets import load_from_disk
from datasets import load_dataset
from datasets import Dataset

# import tokenizers:
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from transformers import DataCollatorForLanguageModeling
from transformers import PreTrainedTokenizerFast

# import config:
from src.hardware import HardwareManager
from src.config import MLMConfig


# =========================================================
# DATA PIPELINE CLASS:
# =========================================================

class DataPipeline:
    """
    Encapsulates data ingestion, tokenisation, and collation logic.
    """
    
    SPECIAL_TOKENS = [
        "[_UNK_]", 
        "[_PAD_]", 
        "[_INIT_]", 
        "[_STOP_]"
    ]

    def __init__(self, config: MLMConfig, hwm: HardwareManager):
        self.tokenizer: PreTrainedTokenizerFast = None
        self.hwm: HardwareManager = None
        self.cfg: MLMConfig = None


    # =====================================================
    # CALCULATE FINGERPRINT OF SAVED FILES:
    # =====================================================

    @staticmethod
    def _fingerprint(payload: dict) -> str:
        """Short stable hash of the values a cache depends on."""
        blob = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def _tokenizer_fingerprint(self) -> str:
        return self._fingerprint({
            "dataset_path": self.cfg.dataset_path,
            "dataset_name": self.cfg.dataset_name,
            "tokenizer_samples": self.cfg.tokenizer_samples,
            "special_tokens": self.SPECIAL_TOKENS,
        })
    
    def _dataset_fingerprint(self) -> str:
        return self._fingerprint({
            "dataset_path": self.cfg.dataset_path,
            "dataset_name": self.cfg.dataset_name,
            "max_length": self.cfg.max_length,
            "tokenizer": self._tokenizer_fingerprint(),
        })

    # =====================================================
    # DOWNLOAD DATASET:
    # =====================================================

    def load_dataset(self) -> Dataset:
        """
        Download dataset.
        """
        return load_dataset(
            path=self.cfg.dataset_path,
            name=self.cfg.dataset_name,
            split="train",
            streaming=False,
            num_proc=self.hwm.n_cpus,
        )

    # =====================================================
    # LOAD THE DATASET FROM SAVED FILE:
    # =====================================================

    def _load_cached_tokenizer(self) -> bool:
        """Load a previously trained tokeniser if its fingerprint still matches."""
        fp_path = os.path.join(self.config.tokenizer_dir, ".fingerprint")
        if not os.path.exists(fp_path):
            return False
        with open(fp_path) as f:
            if f.read().strip() != self._tokenizer_fingerprint():
                return False
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(self.config.tokenizer_dir)
        return True

    # =====================================================
    # TRAIN TOKENIZER:
    # =====================================================

    def build_tokenizer(self, dataset: Dataset) -> PreTrainedTokenizerFast:
        """
        Trains a BPE tokeniser (or loads it from cache) and persists it.
        """

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


    # =====================================================
    # BUILD COLLATOR:
    # =====================================================

    def _build_collator(self) -> DataCollatorForLanguageModeling:
        # Masking is applied on-the-fly per batch, so it is intentionally NOT cached.
        return DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=self.config.mlm_probability,
            pad_to_multiple_of=8,  # Optimizes Tensor Core execution on H200
        )

    # =====================================================
    # PREPARE DATASET:
    # =====================================================

    def prepare_dataset(self, dataset: Dataset) -> Tuple[Dataset, DataCollatorForLanguageModeling]:
        """Tokenises the dataset (or loads it from cache) and returns it with a collator."""
        if not self.tokenizer:
            raise ValueError("Tokeniser must be built before preparing the dataset.")

        cache_path = os.path.join(self.config.cache_dir, self._dataset_fingerprint())
        if os.path.isdir(cache_path):
            print(f"Tokenised-dataset cache hit ({cache_path}); skipping tokenisation.")
            tokenized_dataset = load_from_disk(cache_path)
            tokenized_dataset.set_format("torch")
            return tokenized_dataset, self._build_collator()

        if not getattr(self.tokenizer, "is_fast", False):
            print("Warning: You are using a slow tokenizer. Multi-threading will be limited.")

        def tokenize_fn(batch):
            return self.tokenizer(
                batch["text"],
                truncation=True,
                max_length=self.config.max_length,
                padding=False,  # Do NOT pad here — the DataCollator pads dynamically per batch.
            )
            
        columns_to_remove = [c for c in dataset.column_names if c not in ["input_ids", "attention_mask"]]

        tokenized_dataset = dataset.map(
            tokenize_fn,
            batched=True,
            batch_size=1024,
            writer_batch_size=65536,
            num_proc=2 * self.config.tokenize_num_proc,
            remove_columns=columns_to_remove, 
            desc="Tokenising",
        )

        tokenized_dataset = tokenized_dataset.shuffle(seed=42)

        # Persist the fully processed dataset so re-runs load instantly.
        os.makedirs(self.config.cache_dir, exist_ok=True)
        tokenized_dataset.save_to_disk(cache_path)

        tokenized_dataset.set_format("torch")
        return tokenized_dataset, self._build_collator()
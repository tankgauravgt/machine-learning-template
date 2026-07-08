import torch
from transformers import BertConfig, BertForMaskedLM, PreTrainedTokenizerFast

from src.config import MLMConfig
from src.hardware import HardwareInfo

class ModelFactory:
    """Constructs Hugging Face native models based on project configuration."""

    @staticmethod
    def create_masked_lm(
        config: MLMConfig,
        tokenizer: PreTrainedTokenizerFast,
        hw: HardwareInfo,
    ) -> BertForMaskedLM:
        """Initialises a BERT base model from scratch for the detected hardware."""
        # len(tokenizer) already includes added special tokens (e.g. [MASK]).
        vocab_size = len(tokenizer)

        # Initialize natively in bfloat16 if the hardware supports it.
        # Now that TrainingArguments(bf16=True) is active, the Trainer will run the 
        # forward pass in a bf16 autocast context, satisfying Transformer Engine 
        # and Flash Attention 3 without requiring fp32 master weights.
        dtype = torch.bfloat16 if hw.bf16 else torch.float32

        # Attention implementation selection.
        if hw.is_hopper and hw.flash_attn and hw.fp8:
            attn_impl = "flash_attention_3"
        elif hw.is_hopper and hw.flash_attn and hw.bf16:
            attn_impl = "flash_attention_3"
        elif hw.flash_attn:
            attn_impl = "flash_attention_2"
        else:
            attn_impl = "sdpa"

        bert_config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=config.d_model,
            num_hidden_layers=config.num_layers,
            num_attention_heads=config.nhead,
            intermediate_size=config.d_model * 4,
            max_position_embeddings=config.max_length,
            hidden_dropout_prob=config.dropout,
            attention_probs_dropout_prob=config.dropout,
            pad_token_id=tokenizer.pad_token_id,
            is_decoder=False,
            torch_dtype=dtype,
            attn_implementation=attn_impl,
        )

        model = BertForMaskedLM(bert_config).to(dtype)

        return model
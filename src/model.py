import torch
from transformers import BertConfig, BertForMaskedLM, PreTrainedTokenizerFast
from src.config import MLMConfig

class ModelFactory:
    """Constructs Hugging Face native models based on project configuration."""

    @staticmethod
    def create_masked_lm(config: MLMConfig, tokenizer: PreTrainedTokenizerFast) -> BertForMaskedLM:
        """Initialises a BERT base model from scratch with H200-optimal settings."""
        vocab_size = tokenizer.vocab_size + len(tokenizer.added_tokens_encoder)

        dtype = torch.bfloat16 if config.use_bf16 else torch.float32

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
            attn_implementation="flash_attention_3" if config.use_flash_attention else "sdpa",
        )

        model = BertForMaskedLM(bert_config).to(dtype)

        return model
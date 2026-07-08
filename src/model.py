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

        # When TransformerEngine's FP8 autocast is active, accelerate replaces
        # the embedding LayerNorm with a TE op that expects fp32 master
        # weights — bf16-initialised parameters trip a "invalid argument"
        # inside the TE CUDA kernel. Force fp32 init on the FP8 path and let
        # TE cast on entry; keep bf16 as the master dtype otherwise.
        if hw.fp8:
            dtype = torch.float32
        else:
            dtype = torch.bfloat16 if hw.bf16 else torch.float32

        # flash_attention_2 when kernels are present, else the portable SDPA path
        # (works on CUDA, MPS, and CPU).
        attn_impl = "flash_attention_3" if hw.is_hopper and hw.flash_attn else ("flash_attention_2" if hw.flash_attn else "sdpa")

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

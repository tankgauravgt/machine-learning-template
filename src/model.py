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

        # Attention implementation selection.
        # We prefer FA-3 strongly on Hopper when it's installed, but the
        # FA-3 kernel in transformers refuses to run on fp32 parameters and
        # TE (FP8 path) requires fp32 master weights for its LayerNorm. So:
        #   * FP8 path (TE manages forward cast): use FA-3 anyway — the FA-3
        #     forward pass runs under TE's autocast which casts activations
        #     to bf16/fp8; backward uses bf16 grads via bf16 TrainingArguments.
        #   * Non-FP8 path on Hopper: use FA-3 only when the model itself is
        #     bf16 (parameters are bf16, no autocast gymnastics needed).
        #   * Anywhere else with FA-2: use FA-2 (works on fp32 + autocast).
        #   * No fast kernel: portable SDPA (CUDA / MPS / CPU).
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

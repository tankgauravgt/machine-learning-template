from transformers import BertConfig, BertForMaskedLM, PreTrainedTokenizerFast
from src.config import MLMConfig

class ModelFactory:
    """Constructs Hugging Face native models based on project configuration."""
    
    @staticmethod
    def create_masked_lm(config: MLMConfig, tokenizer: PreTrainedTokenizerFast) -> BertForMaskedLM:
        """Initialises a structurally accurate BERT model from scratch."""
        vocab_size = tokenizer.vocab_size + len(tokenizer.added_tokens_encoder)
        
        bert_config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=config.d_model,
            num_hidden_layers=config.num_layers,
            num_attention_heads=config.nhead,
            max_position_embeddings=config.max_length,
            hidden_dropout_prob=config.dropout,
            attention_probs_dropout_prob=config.dropout,
            pad_token_id=tokenizer.pad_token_id,
            is_decoder=False
        )
        
        return BertForMaskedLM(bert_config)
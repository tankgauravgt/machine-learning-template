import torch
from transformers import AutoModelForMaskedLM, PreTrainedTokenizerFast

from src.config import MLMConfig
from src.hardware import detect_hardware

class MaskedLanguagePredictor:
    """Handles inference tasks for a trained Masked Language Model."""

    def __init__(self, config: MLMConfig):
        hw = detect_hardware(config)
        self.device = torch.device(hw.device)
        dtype = torch.bfloat16 if hw.bf16 else torch.float32

        print(f"Loading tokeniser and model weights on {hw.device}...")
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(config.tokenizer_dir)
        self.model = AutoModelForMaskedLM.from_pretrained(
            config.checkpoint_dir, torch_dtype=dtype
        ).to(self.device)
        self.model.eval()

    def predict(self, sentence: str) -> None:
        """Processes a string containing a [MASK] token and outputs the prediction."""
        mask_id = self.tokenizer.mask_token_id
        
        with torch.no_grad():
            enc = self.tokenizer(sentence, return_tensors="pt").to(self.device)
            logits = self.model(enc["input_ids"], enc["attention_mask"]).logits
            
            mask_positions = (enc["input_ids"] == mask_id).nonzero(as_tuple=True)[1]
            
            print(f"Input     : {sentence}")
            for pos in mask_positions:
                predicted_id = logits[0, pos].argmax(-1).item()
                predicted_token = self.tokenizer.convert_ids_to_tokens([predicted_id])[0]
                print(f"Predicted : {predicted_token}")
            print("-" * 40)

def main():
    config = MLMConfig()
    predictor = MaskedLanguagePredictor(config)
    
    test_sentences = [
        "I [MASK] playing cricket.",
        "You can't [MASK] my mobile number.",
    ]

    print("\n--- Commencing Inference ---\n")
    for sentence in test_sentences:
        predictor.predict(sentence)

if __name__ == "__main__":
    main()
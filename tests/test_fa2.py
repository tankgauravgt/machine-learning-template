from transformers import AutoModelForCausalLM
import torch

try:
    # Use any small model architecture that supports FA2 (e.g., Llama-based)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto"
    )
    print("✅ Model successfully initialized with FlashAttention-2!")
except Exception as e:
    print(f"❌ Failed to initialize model with FA2: {e}")
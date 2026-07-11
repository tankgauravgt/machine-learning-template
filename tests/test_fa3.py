import torch
import flash_attn_interface

print(f"CUDA Available: {torch.cuda.is_available()}")

# Define shapes: Batch size 2, Seq length 1024, 8 heads, head dimension 128
batch_size, seqlen, nheads, headdim = 2, 1024, 8, 128

# FA3 requires fp16 or bf16 data types
q = torch.randn(batch_size, seqlen, nheads, headdim, dtype=torch.float16, device="cuda")
k = torch.randn(batch_size, seqlen, nheads, headdim, dtype=torch.float16, device="cuda")
v = torch.randn(batch_size, seqlen, nheads, headdim, dtype=torch.float16, device="cuda")

# Execute the Flash-Attention 3 kernel
output = flash_attn_interface.flash_attn_func(q, k, v)

print(f"Success! Output shape: {output.shape}")
print("Flash-Attention 3 successfully installed and initialized!")

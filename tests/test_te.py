import torch
import transformer_engine.pytorch as te

print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"PyTorch CUDA Version: {torch.version.cuda}")

# Initialize a TE Linear layer and push it to the GPU
# This will fail instantly with an ImportError or C++ backend error if the installation is broken
te_layer = te.Linear(1024, 4096).cuda()

print("Transformer Engine successfully installed and initialized!")
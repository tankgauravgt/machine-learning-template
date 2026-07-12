"""
Hardware capabilities and device management.
"""

import importlib.util
import torch
from functools import cached_property

class HardwareManager:
    """
    Detects hardware capabilities and selects the best 
    available configuration.
    """
    
    # =====================================================
    # PACKAGE DETECTOR
    # =====================================================

    def __has_package(self, name: str) -> bool:
        """
        Checks if provided package is available or not. 
        """
        return importlib.util.find_spec(name) is not None

    # =====================================================
    # REPRESENTATION OF THIS OBJECT:
    # =====================================================

    def __repr__(self):
        """
        Object representation string.
        """
        rstr = ["DEVICE INFO:"]
        rstr.append(f"- Device: {self.device}")
        
        if self.device == "cuda" and not self.is_amd_rocm:
            rstr.append(f"- Compute Capability: {self.compute_capability}")
        elif self.is_amd_rocm:
            rstr.append(f"- Architecture: AMD ROCm (gfx{self.compute_capability[0]}{self.compute_capability[1]}x)")
            
        rstr.append("\nSUPPORTED DATA TYPES:")
        for t in ["fp32", "fp16", "bf16", "fp8"]:
            status = "✅" if getattr(self, f"{t}_support") else "❌"
            rstr.append(f"- {t.upper():<4}: {status}")
            
        rstr.append("\nFEATURES:")
        rstr.append(f"- Attention Implementation: {self.attn_impl}")
        te_status = "✅" if self.transformer_engine_support else "❌"
        rstr.append(f"- Transformer Engine: {te_status}")
        
        return "\n".join(rstr)

    # =====================================================
    # DEVICE DETECTION
    # =====================================================

    @cached_property
    def device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
        return "cpu"

    @cached_property
    def is_amd_rocm(self) -> bool:
        """
        PyTorch lumps ROCm under 'cuda'. This explicitly isolates AMD GPUs.
        """
        return self.device == "cuda" and getattr(torch.version, "hip", None) is not None

    # =====================================================
    # CUDA CAPABILITY DETECTION
    # =====================================================

    @cached_property
    def compute_capability(self) -> tuple:
        # Now safely returns capability for both NVIDIA and AMD
        # Note: On AMD, this returns the ROCm architecture (e.g., (9, 4) for MI300)
        if self.device == "cuda":
            return torch.cuda.get_device_capability()
        return (0, 0)

    # =====================================================
    # ATTENTION IMPLEMENTATION SELECTION
    # =====================================================

    @cached_property
    def attn_impl(self) -> str:
        # Note: PyTorch native SDPA automatically routes to FlashAttention-2 
        # on Ampere+ hardware. External packages are only needed for strict overrides.
        if self.device != "cuda" or self.is_amd_rocm:
            return "sdpa"
            
        if self.compute_capability >= (9, 0) and self.__has_package("flash_attn_3"):
            return "flash_attention_3"
        elif self.compute_capability >= (8, 0) and self.__has_package("flash_attn"):
            return "flash_attention_2"
            
        return "sdpa"

    # =====================================================
    # TRANSFORMER ENGINE DETECTION
    # =====================================================

    @cached_property
    def transformer_engine_support(self) -> bool:
        if self.device != "cuda" or self.is_amd_rocm:
            return False
        return self.compute_capability >= (8, 0) and self.__has_package("transformer_engine")

    # =====================================================
    # DATA TYPES SUPPORT
    # =====================================================

    @cached_property
    def fp32_support(self) -> bool:
        return True
        
    @cached_property
    def fp16_support(self) -> bool:
        # All modern devices (including CPUs and MPS) support fp16 allocation and compute natively or via emulation.
        return True

    @cached_property
    def bf16_support(self) -> bool:
        if self.device == "cuda":
            return torch.cuda.is_bf16_supported()
        elif self.device == "xpu":
            return hasattr(torch.xpu, "is_bf16_supported") and torch.xpu.is_bf16_supported()
        elif self.device == "mps":
            return True
        elif self.device == "cpu":
            return True 
        return False

    @cached_property
    def fp8_support(self) -> bool:
        if self.device == "cuda":
            if self.is_amd_rocm:
                # AMD MI300 (gfx940/gfx942) architectures and newer
                return self.compute_capability >= (9, 4)
            else:
                # NVIDIA Ada Lovelace (8, 9), Hopper (9, 0), and newer
                return self.compute_capability >= (8, 9)
        return False

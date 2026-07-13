"""
Hardware capabilities and device management.
"""

import importlib.util
import functools
import torch
import os


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
            major = self.compute_capability[0]
            minor = self.compute_capability[1]
            rstr.append(f"- Architecture: AMD ROCm (gfx{major}{minor}x)")
            
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

    @functools.cached_property
    def device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
        return "cpu"

    @functools.cached_property
    def is_amd_rocm(self) -> bool:
        return self.device == "cuda" and getattr(torch.version, "hip", None) is not None


    # =====================================================
    # NO OF CPUs AVAILABLE:
    # =====================================================

    @functools.cached_property
    def n_cpus(self):
        return os.process_cpu_count()
    
    # =====================================================
    # CUDA CAPABILITY DETECTION
    # =====================================================

    @functools.cached_property
    def compute_capability(self) -> tuple:
        if self.device == "cuda":
            return torch.cuda.get_device_capability()
        return (0, 0)

    # =====================================================
    # ATTENTION IMPLEMENTATION SELECTION
    # =====================================================

    @functools.cached_property
    def attn_impl(self) -> str:
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

    @functools.cached_property
    def transformer_engine_support(self) -> bool:
        if self.device != "cuda" or self.is_amd_rocm:
            return False
        return self.compute_capability >= (8, 0) and self.__has_package("transformer_engine")

    # =====================================================
    # DATA TYPES SUPPORT
    # =====================================================

    @functools.cached_property
    def fp32_support(self) -> bool:
        return True
        
    @functools.cached_property
    def fp16_support(self) -> bool:
        return True

    @functools.cached_property
    def bf16_support(self) -> bool:
        if self.device == "cuda":
            return hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
        elif self.device == "xpu":
            return hasattr(torch.xpu, "is_bf16_supported") and torch.xpu.is_bf16_supported()
        elif self.device == "mps":
            return True
        elif self.device == "cpu":
            return True 
        return False

    @functools.cached_property
    def fp8_support(self) -> bool:
        if self.device == "cuda":
            if self.is_amd_rocm:
                return self.compute_capability >= (9, 4)
            else:
                return self.compute_capability >= (8, 9)
        return False

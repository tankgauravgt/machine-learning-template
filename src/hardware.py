"""
Hardware capabilities and device management.
"""

import torch
import importlib.util

class HardwareManager:
    """
    Detects hardware capabilities and selects the best 
    available configuration.
    """
    
    # =====================================================
    # PACKAGE DETECTOR
    # =====================================================

    def __has_package(self, name: str) -> bool:
        return importlib.util.find_spec(name) is not None


    # =====================================================
    # DEVICE DETECTION
    # =====================================================

    @property
    def device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"


    # =====================================================
    # CUDA CAPABILITY DETECTION
    # =====================================================

    @property
    def cuda_capabilities(self) -> tuple:
        if self.device == "cuda":
            return torch.cuda.get_device_capability()
        else:
            return (0, 0)


    # =====================================================
    # ATTENTION IMPLEMENTATION SELECTION
    # =====================================================

    @property
    def attn_impl(self) -> str:
        if self.device != "cuda":
            return "sdpa"
        elif self.cuda_capabilities >= (9, 0) and self.__has_package("flash_attn_3"):
            return "flash_attention_3"
        elif self.cuda_capabilities >= (8, 0) and self.__has_package("flash_attn"):
            return "flash_attention_2"


    # =====================================================
    # TRANSFORMER ENGINE DETECTION
    # =====================================================

    @property
    def transformer_engine_support(self) -> bool:
        if self.device != "cuda":
            return False
        return self.cuda_capabilities >= (8, 0) and self.__has_package("transformer_engine")

    # =====================================================
    # DATA TYPES SUPPORT
    # =====================================================

    @property
    def fp8_support(self) -> bool:
        return False if self.device != "cuda" else self.cuda_capabilities >= (8, 9)
    
    @property
    def bf16_support(self) -> bool:
        return False if self.device != "cuda" else self.cuda_capabilities >= (8, 0)
    
    @property
    def fp16_support(self) -> bool:
        return False if self.device != "cuda" else self.cuda_capabilities >= (5, 0)
    
    @property
    def fp32_support(self) -> bool:
        return True

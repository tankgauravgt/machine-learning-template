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
        if self.device == "cuda":
            sm_major = self.cuda_capabilities[0]
            # -------------------------
            # FA3: Hopper+ (sm_90+)
            # -------------------------            
            if sm_major >= 9 and self.__has_package("flash_attn_3"):
                return "flash_attention_3"
            # -------------------------
            # FA2: Ampere+ (sm_80+)
            # -------------------------
            if sm_major >= 8 and self.__has_package("flash_attn"):
                return "flash_attention_2"
            # -------------------------
        return "sdpa"


    # =====================================================
    # TRANSFORMER ENGINE DETECTION
    # =====================================================

    @property
    def transformer_engine_support(self) -> bool:
        if self.device == "cuda":
            sm_major = self.cuda_capabilities[0]
            # -------------------------
            # TE: Ampere+ (sm_80+)
            # -------------------------
            return sm_major >= 8 and self.__has_package("transformer_engine")
            # -------------------------
        return False


    # =====================================================
    # DATA TYPES SUPPORT
    # =====================================================

    @property
    def fp8_support(self) -> bool:
        if self.device == "cuda":
            sm_major = self.cuda_capabilities[0]
            # -------------------------
            # FP8: Hopper+ (sm_90+)
            # -------------------------
            return sm_major >= 9
        return False
    
    @property
    def bf16_support(self) -> bool:
        if self.device == "cuda":
            sm_major = self.cuda_capabilities[0]
            # -------------------------
            # BF16: Ampere+ (sm_80+)
            # -------------------------
            return sm_major >= 8
        return False
    
    @property
    def fp16_support(self) -> bool:
        if self.device == "cuda":
            sm_major = self.cuda_capabilities[0]
            # -------------------------
            # FP16: Maxwell+ (sm_50+)
            # -------------------------
            return sm_major >= 5
        return False
    
    @property
    def fp32_support(self) -> bool:
        # -------------------------
        # FP32: All devices
        # -------------------------
        return True
    
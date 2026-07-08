"""Runtime hardware detection and environment configuration.

Single source of truth for which accelerations are available on the current
machine. Detects the compute device (CUDA / MPS / CPU) and probes for
Hopper-class GPUs and optional acceleration packages, so the same code path
runs on a plain CPU, an Apple Silicon Mac, a commodity CUDA GPU, or an H200.
"""
import os
import importlib.util
from dataclasses import dataclass

import torch

from src.config import MLMConfig


def _has_package(name: str) -> bool:
    """True if an optional package is importable without importing it."""
    return importlib.util.find_spec(name) is not None


@dataclass
class HardwareInfo:
    device: str            # "cuda" | "mps" | "cpu"
    is_hopper: bool        # CUDA compute capability >= (9, 0)
    bf16: bool             # bfloat16 supported for compute
    fp8: bool              # FP8 via TransformerEngine (Hopper+)
    flash_attn: bool       # Flash-Attention kernels available (CUDA)
    tf32: bool             # TF32 matmuls (Ampere+)
    torch_compile: bool    # torch.compile / inductor usable
    pin_memory: bool       # pinned-memory DataLoader (CUDA)
    fused_optim: bool      # fused AdamW kernel (CUDA)

    def summary(self) -> str:
        parts = [
            f"device={self.device}",
            f"hopper={self.is_hopper}",
            f"bf16={self.bf16}",
            f"fp8={self.fp8}",
            f"flash_attn={self.flash_attn}",
            f"tf32={self.tf32}",
            f"torch_compile={self.torch_compile}",
        ]
        return "Hardware: " + ", ".join(parts)


def _resolve(flag, capability: bool, name: str) -> bool:
    """Combine a tri-state config flag with a detected hardware capability.

    - "auto"  -> use whatever the hardware supports
    - True    -> honour only if supported, else warn and disable
    - False   -> always off
    """
    if isinstance(flag, str) and flag.lower() == "auto":
        return capability
    if flag and not capability:
        print(f"Warning: '{name}' was requested but is unsupported here; disabling.")
        return False
    return bool(flag) and capability


def detect_hardware(config: MLMConfig) -> HardwareInfo:
    """Probe the current machine and reconcile it with the config flags."""
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    capability = torch.cuda.get_device_capability() if device == "cuda" else (0, 0)
    is_hopper = device == "cuda" and capability[0] >= 9
    ampere_plus = device == "cuda" and capability[0] >= 8

    # Raw hardware/software capabilities before applying user overrides.
    cap_bf16 = (
        torch.cuda.is_bf16_supported() if device == "cuda"
        else device == "mps"  # MPS supports bf16 on recent macOS/PyTorch
    )
    cap_fp8 = is_hopper and _has_package("transformer_engine")
    cap_flash = device == "cuda" and (
        _has_package("flash_attn") or _has_package("flash_attn_3")
    )
    cap_tf32 = ampere_plus
    # torch.compile does not compose with TransformerEngine: Dynamo cannot
    # trace TE's pybind LayerNorm/linear ops and falls into a broken fused path
    # that errors inside the TE CUDA kernel. On CUDA, only compile when FP8 is off.
    cap_compile = device == "cpu" or (device == "cuda" and not cap_fp8)
    cap_pin = device == "cuda"
    cap_fused = device == "cuda"

    hw = HardwareInfo(
        device=device,
        is_hopper=is_hopper,
        bf16=_resolve(config.use_bf16, cap_bf16, "bf16"),
        fp8=_resolve(config.use_fp8, cap_fp8, "fp8"),
        flash_attn=_resolve(config.use_flash_attention, cap_flash, "flash_attention"),
        tf32=_resolve(config.use_tf32, cap_tf32, "tf32"),
        torch_compile=_resolve(config.use_torch_compile, cap_compile, "torch_compile"),
        pin_memory=cap_pin,
        fused_optim=cap_fused,
    )

    if hw.fp8 and hw.torch_compile:
        # Defence-in-depth: cap_compile already forbids this, but never combine.
        hw.torch_compile = False
    elif hw.fp8 and (
        config.use_torch_compile is True or config.use_torch_compile == "auto"
    ):
        print("Warning: torch.compile disabled — it does not compose with FP8/TransformerEngine "
              "(Dynamo cannot trace TE's pybind kernels).")
    return hw


def configure_environment(hw: HardwareInfo) -> None:
    """Set process env vars and torch backend flags for the detected hardware.

    Must run before the HF Trainer / Accelerator is constructed so that FP8
    mixed precision is picked up.
    """
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if hw.device == "cuda":
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        os.environ.setdefault("NCCL_P2P_DISABLE", "0")
        torch.backends.cuda.matmul.allow_tf32 = hw.tf32
        torch.backends.cudnn.allow_tf32 = hw.tf32

    if hw.fp8:
        # Accelerate reads this at init to route compute through TransformerEngine.
        os.environ["ACCELERATE_MIXED_PRECISION"] = "fp8"

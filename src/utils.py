import torch
import platform
from typing import Optional


def get_device(device: Optional[str] = None) -> str:
    if device:
        if device == "cuda" and not torch.cuda.is_available():
            device = None
        elif device == "mps" and not torch.backends.mps.is_available():
            device = None
        else:
            return device
    
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def get_dtype(device: str) -> torch.dtype:
    if device == "mps":
        return torch.float32
    elif device == "cuda":
        return torch.float16
    else:
        return torch.float32


def optimize_for_metal(model: torch.nn.Module, device: str):
    if device == "mps":
        try:
            if hasattr(torch.backends.mps, 'enable_mps_fallback'):
                torch.backends.mps.enable_mps_fallback()
        except:
            pass
        model = model.to(device)
    return model


def get_optimal_batch_size(device: str, base_batch: int) -> int:
    if device == "mps":
        return min(base_batch, 16)
    elif device == "cpu":
        return min(base_batch, 8)
    else:
        return base_batch


def is_apple_silicon() -> bool:
    return platform.processor() == "arm" and platform.system() == "Darwin"


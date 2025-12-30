"""
Device selection for PyTorch.

Supports:
- MPS (Apple Silicon)
- CUDA (if available)
- CPU (fallback)
"""

from __future__ import annotations

import torch


def get_device(preference: str = None) -> torch.device:
    """
    Get the best available device.

    Args:
        preference: Optional device preference ("mps", "cuda", "cpu")
                   If None, auto-selects best available.

    Returns:
        torch.device
    """
    if preference is not None:
        if preference == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        elif preference == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif preference == "cpu":
            return torch.device("cpu")
        else:
            # Preference not available, fall through to auto-detect
            pass

    # Auto-detect best device
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def get_device_info() -> dict:
    """Get information about available devices."""
    info = {
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_name"] = torch.cuda.get_device_name(0)

    return info


def move_to_device(data, device: torch.device):
    """
    Move data to device.

    Handles tensors, lists, tuples, and dicts recursively.
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, (list, tuple)):
        return type(data)(move_to_device(x, device) for x in data)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    else:
        return data

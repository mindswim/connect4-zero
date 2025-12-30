"""
Random seed management for reproducibility.
"""

from __future__ import annotations

import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    Sets seeds for:
    - Python random
    - NumPy
    - PyTorch (CPU and GPU)

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # For MPS, manual_seed covers it

    # Enable deterministic algorithms where possible
    # Note: This may impact performance
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

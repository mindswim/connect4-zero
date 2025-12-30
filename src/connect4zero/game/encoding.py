"""
Tensor encoding for neural network input/output.

Input encoding:
- Shape: (C, 6, 7) where C=2
- Channel 0: current player stones (board == +1)
- Channel 1: opponent stones (board == -1)

Output:
- Policy: 7 logits (one per column)
- Value: scalar in [-1, 1]
"""

from __future__ import annotations

import numpy as np
import torch

from .connect4 import GameState, legal_moves, ROWS, COLS


NUM_CHANNELS = 2


def encode_state(state: GameState) -> np.ndarray:
    """
    Encode game state as tensor for neural network input.

    Args:
        state: Game state in canonical form

    Returns:
        numpy array of shape (2, 6, 7) with float32 dtype
    """
    encoded = np.zeros((NUM_CHANNELS, ROWS, COLS), dtype=np.float32)

    # Channel 0: current player stones (+1)
    encoded[0] = (state.board == 1).astype(np.float32)

    # Channel 1: opponent stones (-1)
    encoded[1] = (state.board == -1).astype(np.float32)

    return encoded


def encode_state_batch(states: list[GameState]) -> np.ndarray:
    """
    Encode multiple states as a batch.

    Args:
        states: List of game states

    Returns:
        numpy array of shape (batch, 2, 6, 7)
    """
    return np.stack([encode_state(s) for s in states])


def encode_state_torch(state: GameState, device: torch.device = None) -> torch.Tensor:
    """
    Encode game state as PyTorch tensor.

    Args:
        state: Game state
        device: Target device (CPU/MPS)

    Returns:
        torch.Tensor of shape (1, 2, 6, 7) ready for network
    """
    encoded = encode_state(state)
    tensor = torch.from_numpy(encoded).unsqueeze(0)  # Add batch dim
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def get_action_mask(state: GameState) -> np.ndarray:
    """
    Get mask of legal actions.

    Args:
        state: Game state

    Returns:
        Boolean array of shape (7,) where True = legal move
    """
    return legal_moves(state)


def get_action_mask_torch(state: GameState, device: torch.device = None) -> torch.Tensor:
    """
    Get legal action mask as PyTorch tensor.

    Args:
        state: Game state
        device: Target device

    Returns:
        torch.Tensor of shape (7,) with True for legal moves
    """
    mask = get_action_mask(state)
    tensor = torch.from_numpy(mask)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def mask_illegal_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Apply legal move mask to policy logits.

    Sets illegal move logits to -inf so they have 0 probability after softmax.

    Args:
        logits: Policy logits of shape (..., 7)
        mask: Boolean mask of shape (..., 7) where True = legal

    Returns:
        Masked logits
    """
    masked = logits.clone()
    masked[~mask] = float("-inf")
    return masked


def decode_policy(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Convert logits to probability distribution over legal moves.

    Args:
        logits: Raw policy logits of shape (..., 7)
        mask: Boolean mask of shape (..., 7) where True = legal

    Returns:
        Probability distribution (sums to 1 over legal moves)
    """
    masked_logits = mask_illegal_logits(logits, mask)
    return torch.softmax(masked_logits, dim=-1)

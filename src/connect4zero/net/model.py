"""
Neural network model for Connect 4 AlphaZero.

Architecture:
- Trunk: 5 convolutional blocks (3x3, 64 channels)
- Policy head: 1x1 conv -> flatten -> 7 logits
- Value head: 1x1 conv -> flatten -> hidden -> tanh

Input: (batch, 2, 6, 7)
Output: policy logits (batch, 7), value (batch, 1)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..game import NUM_CHANNELS, ROWS, COLS


class ConvBlock(nn.Module):
    """Single convolutional block: Conv -> ReLU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.conv(x))


class ResBlock(nn.Module):
    """Residual block: Conv -> ReLU -> Conv + skip connection."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out = F.relu(out + residual)
        return out


class PolicyHead(nn.Module):
    """Policy head: outputs logits for each column."""

    def __init__(self, in_channels: int, hidden_channels: int = 32):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=True)
        self.fc = nn.Linear(hidden_channels * ROWS * COLS, 7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        out = F.relu(self.conv(x))
        out = out.view(batch, -1)
        return self.fc(out)


class ValueHead(nn.Module):
    """Value head: outputs scalar value in [-1, 1]."""

    def __init__(self, in_channels: int, hidden_channels: int = 32, hidden_size: int = 64):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=True)
        self.fc1 = nn.Linear(hidden_channels * ROWS * COLS, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        out = F.relu(self.conv(x))
        out = out.view(batch, -1)
        out = F.relu(self.fc1(out))
        return torch.tanh(self.fc2(out))


class Connect4Net(nn.Module):
    """
    Full Connect 4 AlphaZero network.

    Args:
        num_channels: Number of channels in trunk (default 64)
        num_blocks: Number of residual blocks (default 5)
    """

    def __init__(self, num_channels: int = 64, num_blocks: int = 5):
        super().__init__()

        self.num_channels = num_channels
        self.num_blocks = num_blocks

        # Input projection
        self.input_conv = ConvBlock(NUM_CHANNELS, num_channels)

        # Residual trunk
        self.res_blocks = nn.ModuleList([
            ResBlock(num_channels) for _ in range(num_blocks)
        ])

        # Output heads
        self.policy_head = PolicyHead(num_channels)
        self.value_head = ValueHead(num_channels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, 2, 6, 7)

        Returns:
            policy_logits: Shape (batch, 7)
            value: Shape (batch, 1)
        """
        # Trunk
        out = self.input_conv(x)
        for block in self.res_blocks:
            out = block(out)

        # Heads
        policy_logits = self.policy_head(out)
        value = self.value_head(out)

        return policy_logits, value

    def predict(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with optional action masking.

        Args:
            x: Input tensor of shape (batch, 2, 6, 7)
            mask: Optional boolean mask of shape (batch, 7)

        Returns:
            policy: Probability distribution (batch, 7)
            value: Value prediction (batch, 1)
        """
        policy_logits, value = self(x)

        if mask is not None:
            # Mask illegal moves
            policy_logits = policy_logits.clone()
            policy_logits[~mask] = float("-inf")

        policy = F.softmax(policy_logits, dim=-1)
        return policy, value


def create_model(
    num_channels: int = 64,
    num_blocks: int = 5,
    device: torch.device | None = None,
) -> Connect4Net:
    """
    Create a new Connect4Net model.

    Args:
        num_channels: Channels in trunk
        num_blocks: Number of residual blocks
        device: Target device

    Returns:
        Initialized model
    """
    model = Connect4Net(num_channels=num_channels, num_blocks=num_blocks)
    if device is not None:
        model = model.to(device)
    return model


def save_checkpoint(
    model: Connect4Net,
    optimizer: torch.optim.Optimizer | None,
    iteration: int,
    path: str,
) -> None:
    """Save model checkpoint."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "num_channels": model.num_channels,
        "num_blocks": model.num_blocks,
        "iteration": iteration,
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    device: torch.device | None = None,
) -> tuple[Connect4Net, dict]:
    """
    Load model from checkpoint.

    Returns:
        (model, checkpoint_dict)
    """
    checkpoint = torch.load(path, map_location=device, weights_only=True)

    model = Connect4Net(
        num_channels=checkpoint.get("num_channels", 64),
        num_blocks=checkpoint.get("num_blocks", 5),
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    if device is not None:
        model = model.to(device)

    return model, checkpoint

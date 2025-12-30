"""
Generic AlphaZero neural network.

This network automatically adapts to any game based on GameSpec.
The architecture follows AlphaZero:
- Input projection layer
- Residual tower (configurable depth)
- Policy head (outputs action probabilities)
- Value head (outputs win probability)

The network size can be scaled based on game complexity:
- Simple games (Tic-Tac-Toe): 2-3 blocks, 32-64 channels
- Medium games (Connect 4): 5-10 blocks, 64-128 channels
- Complex games (Chess): 15-20 blocks, 256 channels
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..games.base import GameSpec


class ConvBlock(nn.Module):
    """Convolutional block: Conv -> BatchNorm -> ReLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    """Residual block: Conv -> BN -> ReLU -> Conv -> BN + skip -> ReLU."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual)
        return out


class PolicyHead(nn.Module):
    """
    Policy head: outputs logits for each action.

    Architecture: Conv 1x1 -> BN -> ReLU -> Flatten -> FC
    """

    def __init__(
        self,
        in_channels: int,
        board_shape: tuple[int, ...],
        num_actions: int,
        hidden_channels: int = 32,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(hidden_channels)

        # Calculate flattened size
        flat_size = hidden_channels
        for dim in board_shape:
            flat_size *= dim

        self.fc = nn.Linear(flat_size, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        out = F.relu(self.bn(self.conv(x)))
        out = out.view(batch, -1)
        return self.fc(out)


class ValueHead(nn.Module):
    """
    Value head: outputs scalar value in [-1, 1].

    Architecture: Conv 1x1 -> BN -> ReLU -> Flatten -> FC -> ReLU -> FC -> Tanh
    """

    def __init__(
        self,
        in_channels: int,
        board_shape: tuple[int, ...],
        hidden_channels: int = 32,
        hidden_size: int = 64,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(hidden_channels)

        # Calculate flattened size
        flat_size = hidden_channels
        for dim in board_shape:
            flat_size *= dim

        self.fc1 = nn.Linear(flat_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        out = F.relu(self.bn(self.conv(x)))
        out = out.view(batch, -1)
        out = F.relu(self.fc1(out))
        return torch.tanh(self.fc2(out))


class AlphaZeroNet(nn.Module):
    """
    Generic AlphaZero network that adapts to any game.

    Args:
        game_spec: GameSpec describing the game's structure
        num_channels: Number of channels in residual tower
        num_blocks: Number of residual blocks
    """

    def __init__(
        self,
        game_spec: GameSpec,
        num_channels: int = 128,
        num_blocks: int = 10,
    ):
        super().__init__()

        self.game_spec = game_spec
        self.num_channels = num_channels
        self.num_blocks = num_blocks

        # Input projection
        self.input_conv = ConvBlock(game_spec.num_input_channels, num_channels)

        # Residual tower
        self.res_blocks = nn.ModuleList([
            ResBlock(num_channels) for _ in range(num_blocks)
        ])

        # Output heads
        self.policy_head = PolicyHead(
            in_channels=num_channels,
            board_shape=game_spec.board_shape,
            num_actions=game_spec.num_actions,
        )
        self.value_head = ValueHead(
            in_channels=num_channels,
            board_shape=game_spec.board_shape,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, *board_shape)

        Returns:
            policy_logits: Shape (batch, num_actions)
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
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with optional action masking.

        Args:
            x: Input tensor
            mask: Optional boolean mask of shape (batch, num_actions)

        Returns:
            policy: Probability distribution (batch, num_actions)
            value: Value prediction (batch, 1)
        """
        policy_logits, value = self(x)

        if mask is not None:
            # Mask illegal moves with -inf
            policy_logits = policy_logits.clone()
            policy_logits[~mask] = float("-inf")

        policy = F.softmax(policy_logits, dim=-1)
        return policy, value

    @torch.no_grad()
    def predict_single(
        self,
        state_encoded: np.ndarray,
        action_mask: np.ndarray,
        device: torch.device,
    ) -> tuple[np.ndarray, float]:
        """
        Convenience method to predict for a single state.

        Args:
            state_encoded: Encoded state as numpy array
            action_mask: Boolean mask of legal actions
            device: Torch device

        Returns:
            (policy, value) as numpy array and float
        """
        self.eval()

        x = torch.from_numpy(state_encoded).unsqueeze(0).to(device)
        mask = torch.from_numpy(action_mask).unsqueeze(0).to(device)

        policy, value = self.predict(x, mask)

        return policy.squeeze(0).cpu().numpy(), value.squeeze().item()


def create_model(
    game_spec: GameSpec,
    num_channels: int = 128,
    num_blocks: int = 10,
    device: Optional[torch.device] = None,
) -> AlphaZeroNet:
    """
    Create a new AlphaZeroNet model.

    Args:
        game_spec: Game specification
        num_channels: Channels in residual tower
        num_blocks: Number of residual blocks
        device: Target device

    Returns:
        Initialized model
    """
    model = AlphaZeroNet(
        game_spec=game_spec,
        num_channels=num_channels,
        num_blocks=num_blocks,
    )
    if device is not None:
        model = model.to(device)
    return model


def save_checkpoint(
    model: AlphaZeroNet,
    optimizer: Optional[torch.optim.Optimizer],
    iteration: int,
    path: str,
    extra: Optional[dict] = None,
) -> None:
    """
    Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optional optimizer state
        iteration: Training iteration
        path: Save path
        extra: Extra data to include
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "game_spec": {
            "name": model.game_spec.name,
            "board_shape": model.game_spec.board_shape,
            "num_actions": model.game_spec.num_actions,
            "num_input_channels": model.game_spec.num_input_channels,
        },
        "num_channels": model.num_channels,
        "num_blocks": model.num_blocks,
        "iteration": iteration,
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if extra is not None:
        checkpoint.update(extra)

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    device: Optional[torch.device] = None,
) -> tuple[AlphaZeroNet, dict]:
    """
    Load model from checkpoint.

    Args:
        path: Checkpoint path
        device: Target device

    Returns:
        (model, checkpoint_dict)
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # Reconstruct GameSpec
    spec_data = checkpoint["game_spec"]
    game_spec = GameSpec(
        name=spec_data["name"],
        board_shape=tuple(spec_data["board_shape"]),
        num_actions=spec_data["num_actions"],
        num_input_channels=spec_data["num_input_channels"],
    )

    model = AlphaZeroNet(
        game_spec=game_spec,
        num_channels=checkpoint.get("num_channels", 128),
        num_blocks=checkpoint.get("num_blocks", 10),
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    if device is not None:
        model = model.to(device)

    return model, checkpoint


# Recommended configurations for different game complexities
MODEL_CONFIGS = {
    "tiny": {"num_channels": 32, "num_blocks": 2},    # Tic-tac-toe
    "small": {"num_channels": 64, "num_blocks": 5},   # Connect 4
    "medium": {"num_channels": 128, "num_blocks": 10}, # Othello
    "large": {"num_channels": 256, "num_blocks": 20},  # Chess/Go
}


def get_model_config(game_name: str) -> dict:
    """Get recommended model configuration for a game."""
    game_configs = {
        "tictactoe": "tiny",
        "connect4": "small",
        "othello": "medium",
        "chess": "large",
        "go": "large",
    }
    config_name = game_configs.get(game_name, "medium")
    return MODEL_CONFIGS[config_name]

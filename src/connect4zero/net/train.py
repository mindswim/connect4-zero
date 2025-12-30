"""
Training module for Connect 4 AlphaZero.

Loss = policy_loss + value_loss
- Policy loss: cross-entropy between MCTS policy and network output
- Value loss: MSE between game outcome and network prediction
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler

from .model import Connect4Net
from ..selfplay import ReplayBuffer


@dataclass
class TrainMetrics:
    """Training metrics for one batch/epoch."""

    policy_loss: float
    value_loss: float
    total_loss: float
    policy_entropy: float = 0.0


class Trainer:
    """
    Neural network trainer.

    Args:
        model: Connect4Net model
        device: Torch device
        lr: Learning rate
        weight_decay: L2 regularization
        value_loss_weight: Weight for value loss (default 1.0)
    """

    def __init__(
        self,
        model: Connect4Net,
        device: torch.device,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        value_loss_weight: float = 1.0,
    ):
        self.model = model
        self.device = device
        self.value_loss_weight = value_loss_weight

        self.optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        self.scheduler: Optional[LRScheduler] = None

    def set_scheduler(self, total_steps: int) -> None:
        """Set cosine annealing scheduler."""
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=1e-5,
        )

    def train_batch(
        self,
        states: np.ndarray,
        policies: np.ndarray,
        values: np.ndarray,
    ) -> TrainMetrics:
        """
        Train on a single batch.

        Args:
            states: shape (batch, 2, 6, 7)
            policies: shape (batch, 7) - target policies
            values: shape (batch,) - target values

        Returns:
            TrainMetrics for this batch
        """
        self.model.train()

        # Convert to tensors
        states_t = torch.from_numpy(states).to(self.device)
        policies_t = torch.from_numpy(policies).to(self.device)
        values_t = torch.from_numpy(values).to(self.device).unsqueeze(1)

        # Forward pass
        policy_logits, value_pred = self.model(states_t)

        # Policy loss: cross-entropy
        # Using log_softmax + NLL is more stable
        log_probs = F.log_softmax(policy_logits, dim=-1)
        policy_loss = -torch.sum(policies_t * log_probs, dim=-1).mean()

        # Value loss: MSE
        value_loss = F.mse_loss(value_pred, values_t)

        # Total loss
        total_loss = policy_loss + self.value_loss_weight * value_loss

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        # Compute policy entropy for logging
        with torch.no_grad():
            probs = F.softmax(policy_logits, dim=-1)
            entropy = -torch.sum(probs * log_probs, dim=-1).mean()

        return TrainMetrics(
            policy_loss=policy_loss.item(),
            value_loss=value_loss.item(),
            total_loss=total_loss.item(),
            policy_entropy=entropy.item(),
        )

    def train_epoch(
        self,
        buffer: ReplayBuffer,
        batch_size: int = 256,
    ) -> TrainMetrics:
        """
        Train one epoch over the entire buffer.

        Returns:
            Averaged metrics over all batches
        """
        total_policy = 0.0
        total_value = 0.0
        total_loss = 0.0
        total_entropy = 0.0
        num_batches = 0

        for states, policies, values in buffer.iterate_batches(batch_size):
            metrics = self.train_batch(states, policies, values)
            total_policy += metrics.policy_loss
            total_value += metrics.value_loss
            total_loss += metrics.total_loss
            total_entropy += metrics.policy_entropy
            num_batches += 1

        return TrainMetrics(
            policy_loss=total_policy / num_batches,
            value_loss=total_value / num_batches,
            total_loss=total_loss / num_batches,
            policy_entropy=total_entropy / num_batches,
        )

    def train_steps(
        self,
        buffer: ReplayBuffer,
        num_steps: int,
        batch_size: int = 256,
    ) -> TrainMetrics:
        """
        Train for a fixed number of gradient steps.

        Args:
            buffer: Replay buffer to sample from
            num_steps: Number of gradient steps
            batch_size: Batch size

        Returns:
            Averaged metrics over all steps
        """
        total_policy = 0.0
        total_value = 0.0
        total_loss = 0.0
        total_entropy = 0.0

        for _ in range(num_steps):
            states, policies, values = buffer.sample_batch(batch_size)
            metrics = self.train_batch(states, policies, values)
            total_policy += metrics.policy_loss
            total_value += metrics.value_loss
            total_loss += metrics.total_loss
            total_entropy += metrics.policy_entropy

        return TrainMetrics(
            policy_loss=total_policy / num_steps,
            value_loss=total_value / num_steps,
            total_loss=total_loss / num_steps,
            policy_entropy=total_entropy / num_steps,
        )

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]["lr"]


def compute_loss(
    model: Connect4Net,
    states: torch.Tensor,
    policies: torch.Tensor,
    values: torch.Tensor,
    value_weight: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute losses without gradient updates.

    Returns:
        (total_loss, policy_loss, value_loss)
    """
    policy_logits, value_pred = model(states)

    log_probs = F.log_softmax(policy_logits, dim=-1)
    policy_loss = -torch.sum(policies * log_probs, dim=-1).mean()

    value_loss = F.mse_loss(value_pred, values.unsqueeze(1))

    total_loss = policy_loss + value_weight * value_loss

    return total_loss, policy_loss, value_loss

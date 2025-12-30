"""
Replay buffer for storing self-play training data.

Each sample is a tuple of (state_tensor, policy, value).
"""

from __future__ import annotations

import os
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Iterator
from pathlib import Path


@dataclass
class Sample:
    """Single training sample."""

    state: np.ndarray  # shape (2, 6, 7)
    policy: np.ndarray  # shape (7,)
    value: float


class ReplayBuffer:
    """
    Ring buffer for self-play samples.

    Stores samples in memory with a maximum capacity.
    Old samples are overwritten when capacity is reached.
    """

    def __init__(self, capacity: int = 200_000):
        self.capacity = capacity
        self.states: list[np.ndarray] = []
        self.policies: list[np.ndarray] = []
        self.values: list[float] = []
        self.position = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(self, state: np.ndarray, policy: np.ndarray, value: float) -> None:
        """Add a single sample to the buffer."""
        if self._size < self.capacity:
            self.states.append(state)
            self.policies.append(policy)
            self.values.append(value)
            self._size += 1
        else:
            self.states[self.position] = state
            self.policies[self.position] = policy
            self.values[self.position] = value

        self.position = (self.position + 1) % self.capacity

    def add_game(
        self,
        states: list[np.ndarray],
        policies: list[np.ndarray],
        outcome: float,
    ) -> None:
        """
        Add all samples from a game.

        Args:
            states: List of encoded states (from perspective of player to move)
            policies: List of MCTS policies
            outcome: Final outcome (+1 if player 0 wins, -1 if player 1 wins, 0 draw)
        """
        # Assign values from each player's perspective
        # State i was from player (i % 2)'s perspective
        for i, (state, policy) in enumerate(zip(states, policies)):
            # Player 0 moves at i=0,2,4,... Player 1 at i=1,3,5,...
            # outcome is from player 0's perspective
            # If i is even (player 0's turn), value = outcome
            # If i is odd (player 1's turn), value = -outcome
            if i % 2 == 0:
                value = outcome
            else:
                value = -outcome
            self.add(state, policy, value)

    def sample_batch(
        self,
        batch_size: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a random batch.

        Returns:
            states: shape (batch, 2, 6, 7)
            policies: shape (batch, 7)
            values: shape (batch,)
        """
        indices = np.random.choice(self._size, size=min(batch_size, self._size), replace=False)

        states = np.stack([self.states[i] for i in indices])
        policies = np.stack([self.policies[i] for i in indices])
        values = np.array([self.values[i] for i in indices], dtype=np.float32)

        return states, policies, values

    def iterate_batches(
        self,
        batch_size: int,
        shuffle: bool = True,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Iterate over all samples in batches.

        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle samples

        Yields:
            (states, policies, values) batches
        """
        indices = np.arange(self._size)
        if shuffle:
            np.random.shuffle(indices)

        for start in range(0, self._size, batch_size):
            end = min(start + batch_size, self._size)
            batch_indices = indices[start:end]

            states = np.stack([self.states[i] for i in batch_indices])
            policies = np.stack([self.policies[i] for i in batch_indices])
            values = np.array([self.values[i] for i in batch_indices], dtype=np.float32)

            yield states, policies, values

    def save(self, path: str) -> None:
        """Save buffer to disk."""
        np.savez_compressed(
            path,
            states=np.stack(self.states[:self._size]),
            policies=np.stack(self.policies[:self._size]),
            values=np.array(self.values[:self._size], dtype=np.float32),
            position=self.position,
        )

    def load(self, path: str) -> None:
        """Load buffer from disk."""
        data = np.load(path)
        states = data["states"]
        policies = data["policies"]
        values = data["values"]

        self.states = [states[i] for i in range(len(states))]
        self.policies = [policies[i] for i in range(len(policies))]
        self.values = [float(values[i]) for i in range(len(values))]
        self._size = len(self.states)
        self.position = int(data.get("position", self._size % self.capacity))

    def clear(self) -> None:
        """Clear the buffer."""
        self.states.clear()
        self.policies.clear()
        self.values.clear()
        self.position = 0
        self._size = 0

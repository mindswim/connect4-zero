"""
Batched MCTS for GPU efficiency.

Instead of evaluating one leaf at a time, we collect multiple leaves
and batch-evaluate them in a single GPU call. This dramatically
improves throughput on GPUs.

Key technique: Virtual Loss
When traversing to find leaves, we apply "virtual loss" to nodes
we visit. This assumes the path will result in a loss (-1), which
discourages other traversals from taking the same path. After
real evaluation, we revert the virtual loss and apply the true value.

This allows multiple simulations to run in parallel without
all colliding on the same path.
"""

from __future__ import annotations

from typing import Optional, Any
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F

from .node import Node
from ..games.base import Game
from ..net.model import AlphaZeroNet


@dataclass
class PendingEvaluation:
    """A leaf node waiting for neural network evaluation."""
    node: Node
    path: list[tuple[Node, int]]  # (parent, action) pairs


class BatchedMCTS:
    """
    MCTS with batched neural network evaluation for GPU efficiency.

    Collects multiple leaf nodes and evaluates them in a single
    GPU forward pass, providing significant speedup.

    Args:
        game: Game instance
        model: AlphaZeroNet model
        device: Torch device
        c_puct: Exploration constant
        dirichlet_alpha: Alpha for Dirichlet noise
        dirichlet_epsilon: Mixing weight for noise
        batch_size: Maximum leaves to batch together
        add_noise: Whether to add Dirichlet noise at root
    """

    def __init__(
        self,
        game: Game,
        model: AlphaZeroNet,
        device: torch.device,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        batch_size: int = 8,
        add_noise: bool = True,
    ):
        self.game = game
        self.model = model
        self.device = device
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.batch_size = batch_size
        self.add_noise = add_noise

    def search(
        self,
        state: Any,
        num_simulations: int,
        root: Optional[Node] = None,
    ) -> Node:
        """
        Run batched MCTS from the given state.

        Args:
            state: Starting game state
            num_simulations: Number of simulations to run
            root: Optional existing root node

        Returns:
            Root node with updated statistics
        """
        num_actions = self.game.spec.num_actions

        # Create or reuse root
        if root is None:
            root = Node(state=state, num_actions=num_actions)

        # Expand root if needed
        if not root.is_expanded:
            policy, value = self._evaluate_single(state)
            legal_mask = self.game.get_action_mask(state)
            root.expand(policy, legal_mask)

        # Add Dirichlet noise to root
        if self.add_noise:
            self._add_dirichlet_noise(root)

        # Run simulations in batches
        sim = 0
        while sim < num_simulations:
            pending: list[PendingEvaluation] = []

            # Collect batch of leaves
            batch_target = min(self.batch_size, num_simulations - sim)

            for _ in range(batch_target):
                result = self._traverse_to_leaf(root)

                if result is None:
                    # Terminal node was found and backed up
                    sim += 1
                else:
                    pending.append(result)

            # Batch evaluate non-terminal leaves
            if pending:
                self._batch_evaluate_and_backup(pending)
                sim += len(pending)

        return root

    def _traverse_to_leaf(self, root: Node) -> Optional[PendingEvaluation]:
        """
        Traverse from root to a leaf, applying virtual loss.

        Returns:
            PendingEvaluation if leaf found, None if terminal was hit
        """
        node = root
        path: list[tuple[Node, int]] = []

        while node.is_expanded:
            # Check if terminal
            done, value = self.game.is_terminal(node.state)
            if done:
                # Backup terminal value immediately
                self._backup(path, value)
                # Revert virtual losses
                for parent, action in path:
                    parent.revert_virtual_loss(action)
                return None

            # Select action using PUCT (with virtual loss in Q)
            action = self._select_action(node)

            # Apply virtual loss before descending
            node.apply_virtual_loss(action)
            path.append((node, action))

            # Get or create child
            child = node.get_child(action)
            if child is None:
                child_state = self.game.apply_action(node.state, action)
                child = node.add_child(
                    action,
                    child_state,
                    self.game.spec.num_actions,
                )

            node = child

        # Reached unexpanded leaf - check if terminal
        done, value = self.game.is_terminal(node.state)
        if done:
            self._backup(path, value)
            for parent, action in path:
                parent.revert_virtual_loss(action)
            return None

        return PendingEvaluation(node=node, path=path)

    def _select_action(self, node: Node) -> int:
        """Select action using PUCT with virtual loss."""
        legal_mask = self.game.get_action_mask(node.state)

        # Q already includes virtual loss effect
        sqrt_total = np.sqrt(node.total_visits + 1)

        puct = (
            node.Q
            + self.c_puct * node.P * sqrt_total / (1 + node.N + node.virtual_loss)
        )

        puct[~legal_mask] = float("-inf")
        return int(np.argmax(puct))

    def _batch_evaluate_and_backup(self, pending: list[PendingEvaluation]) -> None:
        """Batch evaluate leaves and backup values."""
        if not pending:
            return

        # Encode all states
        states = [p.node.state for p in pending]
        encoded = np.stack([self.game.encode_state(s) for s in states])
        masks = np.stack([self.game.get_action_mask(s) for s in states])

        # Batch inference
        policies, values = self._evaluate_batch(encoded, masks)

        # Expand nodes and backup
        for i, p in enumerate(pending):
            policy = policies[i]
            value = values[i]

            # Expand leaf
            legal_mask = masks[i]
            p.node.expand(policy, legal_mask)

            # Backup value
            self._backup(p.path, value)

            # Revert virtual losses
            for parent, action in p.path:
                parent.revert_virtual_loss(action)

    @torch.no_grad()
    def _evaluate_batch(
        self,
        encoded: np.ndarray,
        masks: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batch evaluate states with the neural network."""
        self.model.eval()

        x = torch.from_numpy(encoded).to(self.device)
        mask = torch.from_numpy(masks).to(self.device)

        policy_logits, values = self.model(x)

        # Mask and softmax
        policy_logits[~mask] = float("-inf")
        policies = F.softmax(policy_logits, dim=-1)

        return (
            policies.cpu().numpy(),
            values.squeeze(-1).cpu().numpy(),
        )

    @torch.no_grad()
    def _evaluate_single(self, state: Any) -> tuple[np.ndarray, float]:
        """Evaluate a single state."""
        self.model.eval()

        encoded = self.game.encode_state(state)
        mask = self.game.get_action_mask(state)

        x = torch.from_numpy(encoded).unsqueeze(0).to(self.device)
        m = torch.from_numpy(mask).unsqueeze(0).to(self.device)

        policy_logits, value = self.model(x)
        policy_logits[~m] = float("-inf")
        policy = F.softmax(policy_logits, dim=-1)

        return (
            policy.squeeze(0).cpu().numpy(),
            value.squeeze().item(),
        )

    def _backup(self, path: list[tuple[Node, int]], value: float) -> None:
        """Backup value through the path, flipping sign at each level."""
        for parent, action in reversed(path):
            parent.N[action] += 1
            parent.W[action] += value
            value = -value

    def _add_dirichlet_noise(self, node: Node) -> None:
        """Add Dirichlet noise to root priors."""
        num_actions = self.game.spec.num_actions
        noise = np.random.dirichlet([self.dirichlet_alpha] * num_actions)
        legal_mask = self.game.get_action_mask(node.state)

        noise = noise * legal_mask.astype(np.float32)
        if noise.sum() > 0:
            noise = noise / noise.sum()

        node.P = (
            (1 - self.dirichlet_epsilon) * node.P
            + self.dirichlet_epsilon * noise
        )


class BatchedMCTSPlayer:
    """
    Convenient wrapper for using BatchedMCTS for playing games.

    Handles model loading, device selection, and difficulty settings.
    """

    def __init__(
        self,
        game: Game,
        model: AlphaZeroNet,
        device: torch.device,
        simulations: int = 100,
        temperature: float = 0.0,
        c_puct: float = 1.5,
        batch_size: int = 8,
    ):
        self.game = game
        self.mcts = BatchedMCTS(
            game=game,
            model=model,
            device=device,
            c_puct=c_puct,
            batch_size=batch_size,
            add_noise=False,  # No noise when playing
        )
        self.simulations = simulations
        self.temperature = temperature

    def get_action(self, state: Any) -> int:
        """Get best action for the given state."""
        root = self.mcts.search(state, self.simulations)
        return root.select_action(self.temperature)

    def get_action_with_policy(self, state: Any) -> tuple[int, np.ndarray]:
        """Get action and full policy distribution."""
        root = self.mcts.search(state, self.simulations)
        policy = root.get_policy(self.temperature)
        action = root.select_action(self.temperature)
        return action, policy

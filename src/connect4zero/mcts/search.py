"""
MCTS search implementation with PUCT.

PUCT selection formula:
U(a) = Q(a) + c_puct * P(a) * sqrt(sum_b N(b)) / (1 + N(a))

The search:
1. Select: traverse tree using PUCT until reaching a leaf
2. Expand: evaluate leaf with network, store priors
3. Backup: propagate value up the tree (flipping sign each level)
"""

from __future__ import annotations

from typing import Callable, Tuple, Optional
import numpy as np
import torch

from .node import Node
from ..game import (
    GameState,
    initial_state,
    legal_moves,
    apply_move,
    is_terminal,
    encode_state_torch,
    get_action_mask,
    COLS,
)


class MCTS:
    """
    Monte Carlo Tree Search with PUCT.

    Args:
        evaluate_fn: Function that takes a state and returns (policy, value)
        c_puct: Exploration constant (default 1.5)
        dirichlet_alpha: Alpha for Dirichlet noise at root (default 0.3)
        dirichlet_epsilon: Mixing weight for noise (default 0.25)
        device: Torch device for neural network
    """

    def __init__(
        self,
        evaluate_fn: Callable[[GameState], Tuple[np.ndarray, float]],
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        add_noise: bool = True,
    ):
        self.evaluate_fn = evaluate_fn
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.add_noise = add_noise

    def search(
        self,
        state: GameState,
        num_simulations: int,
        root: Optional[Node] = None,
    ) -> Node:
        """
        Run MCTS from the given state.

        Args:
            state: Starting game state
            num_simulations: Number of simulations to run
            root: Optional existing root node (for tree reuse)

        Returns:
            Root node with updated statistics
        """
        # Create or reuse root
        if root is None:
            root = Node(state=state)

        # Expand root if needed
        if not root.is_expanded:
            policy, value = self.evaluate_fn(state)
            legal_mask = get_action_mask(state)
            root.expand(policy, legal_mask)

        # Add Dirichlet noise to root priors (AlphaZero exploration trick)
        if self.add_noise:
            self._add_dirichlet_noise(root)

        # Run simulations
        for _ in range(num_simulations):
            self._simulate(root)

        return root

    def _add_dirichlet_noise(self, node: Node) -> None:
        """Add Dirichlet noise to root priors for exploration."""
        noise = np.random.dirichlet([self.dirichlet_alpha] * COLS)
        legal_mask = get_action_mask(node.state)

        # Only add noise to legal moves
        noise = noise * legal_mask
        if noise.sum() > 0:
            noise = noise / noise.sum()

        node.P = (
            (1 - self.dirichlet_epsilon) * node.P
            + self.dirichlet_epsilon * noise
        )

    def _simulate(self, root: Node) -> None:
        """Run one simulation: select -> expand -> backup."""
        node = root
        search_path = [node]

        # Selection: traverse until we hit a leaf or terminal
        while node.is_expanded:
            done, value = is_terminal(node.state)
            if done:
                # Terminal node - backup the terminal value
                self._backup(search_path, value)
                return

            action = self._select_action(node)
            child = node.get_child(action)

            if child is None:
                # Create child node
                child_state = apply_move(node.state, action)
                child = node.add_child(action, child_state)

            node = child
            search_path.append(node)

        # Check if leaf is terminal
        done, value = is_terminal(node.state)
        if done:
            self._backup(search_path, value)
            return

        # Expansion: evaluate leaf with network
        policy, value = self.evaluate_fn(node.state)
        legal_mask = get_action_mask(node.state)
        node.expand(policy, legal_mask)

        # Backup
        self._backup(search_path, value)

    def _select_action(self, node: Node) -> int:
        """Select action using PUCT formula."""
        legal_mask = get_action_mask(node.state)
        sqrt_total = np.sqrt(node.total_visits + 1)

        # PUCT scores
        puct = (
            node.Q
            + self.c_puct * node.P * sqrt_total / (1 + node.N)
        )

        # Mask illegal moves
        puct[~legal_mask] = float("-inf")

        return int(np.argmax(puct))

    def _backup(self, search_path: list[Node], value: float) -> None:
        """
        Backup value through the search path.

        Value is from perspective of player at the leaf.
        We flip sign at each level going up.
        """
        for node in reversed(search_path):
            if node.parent is not None:
                action = node.parent_action
                node.parent.N[action] += 1
                node.parent.W[action] += value
            value = -value


def create_evaluator(
    model: torch.nn.Module,
    device: torch.device,
) -> Callable[[GameState], Tuple[np.ndarray, float]]:
    """
    Create an evaluation function from a neural network.

    Args:
        model: Connect4Net model
        device: Torch device

    Returns:
        Function that evaluates a state and returns (policy, value)
    """
    model.eval()

    def evaluate(state: GameState) -> Tuple[np.ndarray, float]:
        with torch.no_grad():
            x = encode_state_torch(state, device)
            mask = torch.from_numpy(get_action_mask(state)).unsqueeze(0).to(device)

            policy, value = model.predict(x, mask)

            policy_np = policy.squeeze(0).cpu().numpy()
            value_np = value.squeeze().item()

        return policy_np, value_np

    return evaluate


def create_random_evaluator() -> Callable[[GameState], Tuple[np.ndarray, float]]:
    """Create a random evaluation function for testing."""

    def evaluate(state: GameState) -> Tuple[np.ndarray, float]:
        legal_mask = get_action_mask(state)
        policy = legal_mask.astype(np.float32)
        policy /= policy.sum()
        value = 0.0
        return policy, value

    return evaluate

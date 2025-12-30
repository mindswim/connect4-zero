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

from typing import Callable, Tuple, Optional, Any
import numpy as np
import torch

from .node import Node
from ..games.base import Game


class MCTS:
    """
    Monte Carlo Tree Search with PUCT.

    This is the basic (non-batched) version. For GPU efficiency,
    use BatchedMCTS instead.

    Args:
        game: Game instance
        evaluate_fn: Function that takes a state and returns (policy, value)
        c_puct: Exploration constant (default 1.5)
        dirichlet_alpha: Alpha for Dirichlet noise at root (default 0.3)
        dirichlet_epsilon: Mixing weight for noise (default 0.25)
        add_noise: Whether to add Dirichlet noise at root
    """

    def __init__(
        self,
        game: Game,
        evaluate_fn: Callable[[Any], Tuple[np.ndarray, float]],
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        add_noise: bool = True,
    ):
        self.game = game
        self.evaluate_fn = evaluate_fn
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.add_noise = add_noise

    def search(
        self,
        state: Any,
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
        num_actions = self.game.spec.num_actions

        # Create or reuse root
        if root is None:
            root = Node(state=state, num_actions=num_actions)

        # Expand root if needed
        if not root.is_expanded:
            policy, value = self.evaluate_fn(state)
            legal_mask = self.game.get_action_mask(state)
            root.expand(policy, legal_mask)

        # Add Dirichlet noise to root priors
        if self.add_noise:
            self._add_dirichlet_noise(root)

        # Run simulations
        for _ in range(num_simulations):
            self._simulate(root)

        return root

    def _add_dirichlet_noise(self, node: Node) -> None:
        """Add Dirichlet noise to root priors for exploration."""
        num_actions = self.game.spec.num_actions
        noise = np.random.dirichlet([self.dirichlet_alpha] * num_actions)
        legal_mask = self.game.get_action_mask(node.state)

        # Only add noise to legal moves
        noise = noise * legal_mask.astype(np.float32)
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
        actions_taken = []

        # Selection: traverse until we hit a leaf or terminal
        while node.is_expanded:
            done, value = self.game.is_terminal(node.state)
            if done:
                # Terminal node - backup the terminal value
                self._backup(search_path, actions_taken, value)
                return

            action = self._select_action(node)
            actions_taken.append(action)
            child = node.get_child(action)

            if child is None:
                # Create child node
                child_state = self.game.apply_action(node.state, action)
                child = node.add_child(
                    action,
                    child_state,
                    self.game.spec.num_actions,
                )

            node = child
            search_path.append(node)

        # Check if leaf is terminal
        done, value = self.game.is_terminal(node.state)
        if done:
            self._backup(search_path, actions_taken, value)
            return

        # Expansion: evaluate leaf with network
        policy, value = self.evaluate_fn(node.state)
        legal_mask = self.game.get_action_mask(node.state)
        node.expand(policy, legal_mask)

        # Backup
        self._backup(search_path, actions_taken, value)

    def _select_action(self, node: Node) -> int:
        """Select action using PUCT formula."""
        legal_mask = self.game.get_action_mask(node.state)
        sqrt_total = np.sqrt(node.total_visits + 1)

        # PUCT scores
        puct = (
            node.Q
            + self.c_puct * node.P * sqrt_total / (1 + node.N)
        )

        # Mask illegal moves
        puct[~legal_mask] = float("-inf")

        return int(np.argmax(puct))

    def _backup(
        self,
        search_path: list[Node],
        actions_taken: list[int],
        value: float
    ) -> None:
        """
        Backup value through the search path.

        Value is from perspective of player at the leaf.
        We flip sign at each level going up.
        """
        # Start from the leaf (end of path)
        for i in range(len(search_path) - 1, 0, -1):
            node = search_path[i]
            parent = search_path[i - 1]
            action = actions_taken[i - 1]

            parent.N[action] += 1
            parent.W[action] += value
            value = -value


def create_evaluator(
    game: Game,
    model: torch.nn.Module,
    device: torch.device,
) -> Callable[[Any], Tuple[np.ndarray, float]]:
    """
    Create an evaluation function from a neural network.

    Args:
        game: Game instance
        model: AlphaZeroNet model
        device: Torch device

    Returns:
        Function that evaluates a state and returns (policy, value)
    """
    model.eval()

    def evaluate(state: Any) -> Tuple[np.ndarray, float]:
        with torch.no_grad():
            encoded = game.encode_state(state)
            x = torch.from_numpy(encoded).unsqueeze(0).to(device)
            mask = torch.from_numpy(
                game.get_action_mask(state)
            ).unsqueeze(0).to(device)

            policy, value = model.predict(x, mask)

            policy_np = policy.squeeze(0).cpu().numpy()
            value_np = value.squeeze().item()

        return policy_np, value_np

    return evaluate


def create_random_evaluator(game: Game) -> Callable[[Any], Tuple[np.ndarray, float]]:
    """Create a random evaluation function for testing."""

    def evaluate(state: Any) -> Tuple[np.ndarray, float]:
        legal_mask = game.get_action_mask(state)
        policy = legal_mask.astype(np.float32)
        if policy.sum() > 0:
            policy /= policy.sum()
        value = 0.0
        return policy, value

    return evaluate

"""
MCTS Node data structure.

Each node represents a game state and stores:
- N[a]: visit counts per action
- W[a]: total value per action
- Q[a]: mean value per action (W[a] / N[a])
- P[a]: prior probabilities from policy network
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional

from ..game import GameState, COLS


@dataclass
class Node:
    """
    MCTS tree node.

    Stores statistics for each action (column 0-6).
    Children are created lazily during expansion.
    """

    state: GameState
    parent: Optional[Node] = None
    parent_action: int = -1  # Action that led to this node

    # Per-action statistics (initialized in __post_init__)
    N: np.ndarray = field(default=None)  # Visit counts
    W: np.ndarray = field(default=None)  # Total value
    P: np.ndarray = field(default=None)  # Prior probabilities

    # Child nodes (lazily created)
    children: Dict[int, Node] = field(default_factory=dict)

    # Whether this node has been expanded (network evaluated)
    is_expanded: bool = False

    # Cached terminal status
    _terminal: Optional[tuple[bool, float]] = None

    def __post_init__(self):
        if self.N is None:
            self.N = np.zeros(COLS, dtype=np.float32)
        if self.W is None:
            self.W = np.zeros(COLS, dtype=np.float32)
        if self.P is None:
            self.P = np.zeros(COLS, dtype=np.float32)

    @property
    def Q(self) -> np.ndarray:
        """Mean action value Q(a) = W(a) / N(a)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            q = self.W / self.N
            q = np.nan_to_num(q, nan=0.0)
        return q

    @property
    def total_visits(self) -> int:
        """Total visits to this node (sum of child visits)."""
        return int(np.sum(self.N))

    def is_leaf(self) -> bool:
        """Check if this is a leaf node (not yet expanded)."""
        return not self.is_expanded

    def get_child(self, action: int) -> Optional[Node]:
        """Get child node for action, if it exists."""
        return self.children.get(action)

    def add_child(self, action: int, child_state: GameState) -> Node:
        """Create and add a child node for the given action."""
        child = Node(
            state=child_state,
            parent=self,
            parent_action=action,
        )
        self.children[action] = child
        return child

    def expand(self, priors: np.ndarray, legal_mask: np.ndarray) -> None:
        """
        Expand this node with network priors.

        Args:
            priors: Policy probabilities from network (length 7)
            legal_mask: Boolean mask of legal actions
        """
        # Zero out illegal moves and renormalize
        masked_priors = priors * legal_mask
        total = masked_priors.sum()
        if total > 0:
            masked_priors /= total
        else:
            # Fallback: uniform over legal moves
            masked_priors = legal_mask.astype(np.float32)
            masked_priors /= masked_priors.sum()

        self.P = masked_priors
        self.is_expanded = True

    def backup(self, value: float) -> None:
        """
        Backup value through the tree.

        Value is from the perspective of the player at the expanded leaf.
        As we go up the tree, we flip the sign at each level.
        """
        node = self
        v = value

        while node.parent is not None:
            # Update parent's statistics for the action that led here
            action = node.parent_action
            node.parent.N[action] += 1
            node.parent.W[action] += v

            # Move to parent and flip value perspective
            node = node.parent
            v = -v

    def get_policy(self, temperature: float = 1.0) -> np.ndarray:
        """
        Get policy from visit counts.

        Args:
            temperature: Temperature for exploration.
                - tau=1: proportional to visits
                - tau->0: argmax
                - tau>1: more uniform

        Returns:
            Probability distribution over actions (length 7)
        """
        if temperature == 0:
            # Argmax (greedy)
            best = np.argmax(self.N)
            policy = np.zeros(COLS, dtype=np.float32)
            policy[best] = 1.0
            return policy

        # Proportional to N^(1/tau)
        counts = self.N ** (1.0 / temperature)
        total = counts.sum()
        if total > 0:
            return counts / total
        else:
            # No visits, return uniform over legal moves
            return self.P.copy()

    def select_action(self, temperature: float = 1.0) -> int:
        """
        Select action based on visit counts and temperature.

        Args:
            temperature: Controls exploration vs exploitation

        Returns:
            Selected action (column index)
        """
        policy = self.get_policy(temperature)
        if temperature == 0:
            return int(np.argmax(policy))
        else:
            return int(np.random.choice(COLS, p=policy))

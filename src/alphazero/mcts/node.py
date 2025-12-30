"""
MCTS Node data structure.

Each node represents a game state and stores:
- N[a]: visit counts per action
- W[a]: total value per action
- Q[a]: mean value per action (W[a] / N[a])
- P[a]: prior probabilities from policy network
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..games.base import Game


@dataclass
class Node:
    """
    MCTS tree node.

    Stores statistics for each action.
    Children are created lazily during expansion.
    """

    state: Any  # Game state
    num_actions: int  # Number of possible actions

    parent: Optional[Node] = None
    parent_action: int = -1  # Action that led to this node

    # Per-action statistics
    N: np.ndarray = field(default=None)  # Visit counts
    W: np.ndarray = field(default=None)  # Total value
    P: np.ndarray = field(default=None)  # Prior probabilities

    # Virtual loss tracking (for batched MCTS)
    virtual_loss: np.ndarray = field(default=None)

    # Child nodes (lazily created)
    children: Dict[int, Node] = field(default_factory=dict)

    # Whether this node has been expanded (network evaluated)
    is_expanded: bool = False

    def __post_init__(self):
        if self.N is None:
            self.N = np.zeros(self.num_actions, dtype=np.float32)
        if self.W is None:
            self.W = np.zeros(self.num_actions, dtype=np.float32)
        if self.P is None:
            self.P = np.zeros(self.num_actions, dtype=np.float32)
        if self.virtual_loss is None:
            self.virtual_loss = np.zeros(self.num_actions, dtype=np.float32)

    @property
    def Q(self) -> np.ndarray:
        """Mean action value Q(a) = W(a) / N(a)."""
        # Include virtual loss in calculation
        total_n = self.N + self.virtual_loss
        total_w = self.W - self.virtual_loss  # Virtual loss assumes -1

        with np.errstate(divide="ignore", invalid="ignore"):
            q = total_w / total_n
            q = np.nan_to_num(q, nan=0.0)
        return q

    @property
    def total_visits(self) -> int:
        """Total visits to this node (sum of child visits)."""
        return int(np.sum(self.N))

    def get_child(self, action: int) -> Optional[Node]:
        """Get child node for action, if it exists."""
        return self.children.get(action)

    def add_child(
        self,
        action: int,
        child_state: Any,
        num_actions: int,
    ) -> Node:
        """Create and add a child node for the given action."""
        child = Node(
            state=child_state,
            num_actions=num_actions,
            parent=self,
            parent_action=action,
        )
        self.children[action] = child
        return child

    def expand(self, priors: np.ndarray, legal_mask: np.ndarray) -> None:
        """
        Expand this node with network priors.

        Args:
            priors: Policy probabilities from network
            legal_mask: Boolean mask of legal actions
        """
        # Zero out illegal moves and renormalize
        masked_priors = priors * legal_mask.astype(np.float32)
        total = masked_priors.sum()
        if total > 0:
            masked_priors /= total
        else:
            # Fallback: uniform over legal moves
            masked_priors = legal_mask.astype(np.float32)
            if masked_priors.sum() > 0:
                masked_priors /= masked_priors.sum()

        self.P = masked_priors
        self.is_expanded = True

    def apply_virtual_loss(self, action: int) -> None:
        """Apply virtual loss to an action (for parallel MCTS)."""
        self.virtual_loss[action] += 1

    def revert_virtual_loss(self, action: int) -> None:
        """Revert virtual loss after real evaluation."""
        self.virtual_loss[action] = max(0, self.virtual_loss[action] - 1)

    def get_policy(self, temperature: float = 1.0) -> np.ndarray:
        """
        Get policy from visit counts.

        Args:
            temperature: Temperature for exploration.
                - tau=0: argmax (greedy)
                - tau=1: proportional to visits
                - tau>1: more uniform

        Returns:
            Probability distribution over actions
        """
        if temperature == 0:
            # Argmax (greedy)
            best = np.argmax(self.N)
            policy = np.zeros(self.num_actions, dtype=np.float32)
            policy[best] = 1.0
            return policy

        # Proportional to N^(1/tau)
        counts = self.N ** (1.0 / temperature)
        total = counts.sum()
        if total > 0:
            return counts / total
        else:
            # No visits, return priors
            return self.P.copy()

    def select_action(self, temperature: float = 1.0) -> int:
        """
        Select action based on visit counts and temperature.

        Args:
            temperature: Controls exploration vs exploitation

        Returns:
            Selected action index
        """
        policy = self.get_policy(temperature)
        if temperature == 0:
            return int(np.argmax(policy))
        else:
            return int(np.random.choice(self.num_actions, p=policy))

    def __repr__(self) -> str:
        return f"Node(visits={self.total_visits}, expanded={self.is_expanded})"

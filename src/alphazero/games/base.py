"""
Abstract base classes for AlphaZero games.

Any game that implements the Game interface can be trained with AlphaZero.
The algorithm doesn't need to know anything about the game rules - it just
needs these methods to:
1. Know what moves are legal
2. Apply moves and get new states
3. Know when the game is over
4. Encode states for the neural network
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar, Generic, Any
import numpy as np


@dataclass(frozen=True)
class GameSpec:
    """
    Describes a game's structure for the neural network and MCTS.

    This tells the network what size inputs/outputs to expect.
    """
    name: str
    board_shape: tuple[int, ...]  # e.g., (6, 7) for Connect4, (8, 8) for Chess
    num_actions: int              # e.g., 7 for Connect4, 4672 for Chess
    num_input_channels: int       # e.g., 2 for Connect4, 119 for Chess

    @property
    def board_size(self) -> int:
        """Total number of board cells."""
        result = 1
        for dim in self.board_shape:
            result *= dim
        return result


# Type variables for game state and action
State = TypeVar('State')
Action = TypeVar('Action')


class Game(ABC, Generic[State, Action]):
    """
    Abstract base class for any two-player zero-sum perfect-information game.

    Implement this interface for any game you want to train with AlphaZero.

    Key concepts:
    - State: The full game state (board position, whose turn, etc.)
    - Action: A legal move in the game
    - Canonical form: State from current player's perspective (opponent = -1)

    The canonical form is important: after each move, the board is "flipped"
    so the current player always sees themselves as +1 and opponent as -1.
    This lets a single neural network learn to play both sides.
    """

    @property
    @abstractmethod
    def spec(self) -> GameSpec:
        """Return the game specification."""
        pass

    @abstractmethod
    def initial_state(self) -> State:
        """
        Return the starting state of the game.

        This should be in canonical form for player 1.
        """
        pass

    @abstractmethod
    def legal_actions(self, state: State) -> list[Action]:
        """
        Return list of legal actions from this state.

        Args:
            state: Current game state

        Returns:
            List of legal actions the current player can take
        """
        pass

    @abstractmethod
    def apply_action(self, state: State, action: Action) -> State:
        """
        Apply action and return new state.

        The returned state should be in canonical form for the NEXT player.
        This means flipping the perspective (multiply board by -1).

        Args:
            state: Current game state
            action: Action to apply

        Returns:
            New state in canonical form for the next player
        """
        pass

    @abstractmethod
    def is_terminal(self, state: State) -> tuple[bool, float]:
        """
        Check if the game is over.

        Args:
            state: Current game state

        Returns:
            (done, value) where:
            - done: True if game is over
            - value: Outcome from CURRENT player's perspective
                - +1.0 if current player wins
                - -1.0 if current player loses
                - 0.0 if draw
        """
        pass

    @abstractmethod
    def encode_state(self, state: State) -> np.ndarray:
        """
        Encode state as tensor for neural network input.

        Args:
            state: Game state to encode

        Returns:
            numpy array of shape (num_input_channels, *board_shape)
        """
        pass

    @abstractmethod
    def action_to_index(self, action: Action) -> int:
        """
        Convert action to policy index (0 to num_actions-1).

        Args:
            action: Game-specific action

        Returns:
            Index into the policy vector
        """
        pass

    @abstractmethod
    def index_to_action(self, index: int) -> Action:
        """
        Convert policy index to action.

        Args:
            index: Index from policy vector

        Returns:
            Game-specific action
        """
        pass

    def get_action_mask(self, state: State) -> np.ndarray:
        """
        Return boolean mask of legal actions.

        Default implementation uses legal_actions(), but games can override
        for efficiency.

        Args:
            state: Current game state

        Returns:
            Boolean array of shape (num_actions,) where True = legal
        """
        mask = np.zeros(self.spec.num_actions, dtype=bool)
        for action in self.legal_actions(state):
            mask[self.action_to_index(action)] = True
        return mask

    def get_symmetries(
        self,
        state: State,
        policy: np.ndarray
    ) -> list[tuple[State, np.ndarray]]:
        """
        Return symmetric positions for data augmentation.

        Many games have symmetries (rotation, reflection) that can be
        exploited to generate more training data.

        Default implementation returns just the original (no symmetries).
        Override for games with symmetries.

        Args:
            state: Game state
            policy: Policy vector of shape (num_actions,)

        Returns:
            List of (state, policy) pairs including original and symmetries
        """
        return [(state, policy)]

    def render(self, state: State) -> str:
        """
        Render state as string for display.

        Optional - default returns empty string.

        Args:
            state: Game state to render

        Returns:
            Human-readable string representation
        """
        return ""

    def copy_state(self, state: State) -> State:
        """
        Create a copy of the state.

        Override if your state needs special copying logic.
        """
        if hasattr(state, 'copy'):
            return state.copy()
        return state


# Registry of available games
_GAME_REGISTRY: dict[str, type[Game]] = {}


def register_game(name: str):
    """Decorator to register a game class."""
    def decorator(cls: type[Game]):
        _GAME_REGISTRY[name] = cls
        return cls
    return decorator


def get_game(name: str) -> Game:
    """Get a game instance by name."""
    if name not in _GAME_REGISTRY:
        available = ", ".join(_GAME_REGISTRY.keys())
        raise ValueError(f"Unknown game '{name}'. Available: {available}")
    return _GAME_REGISTRY[name]()


def list_games() -> list[str]:
    """List all registered games."""
    return list(_GAME_REGISTRY.keys())

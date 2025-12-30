"""
Connect 4 game implementation.

Rules:
- 6 rows x 7 columns board
- Players drop pieces into columns
- First to get 4 in a row (horizontal, vertical, or diagonal) wins
- If board fills up with no winner, it's a draw

Board representation uses canonical form:
- +1 = current player's pieces
- -1 = opponent's pieces
- 0 = empty
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np

from .base import Game, GameSpec, register_game


# Board dimensions
ROWS = 6
COLS = 7
WIN_LENGTH = 4


@dataclass
class Connect4State:
    """Connect 4 game state in canonical form."""
    board: np.ndarray  # shape (6, 7), dtype int8

    def __post_init__(self):
        if self.board.shape != (ROWS, COLS):
            raise ValueError(f"Board must be {ROWS}x{COLS}")
        if self.board.dtype != np.int8:
            self.board = self.board.astype(np.int8)

    def copy(self) -> Connect4State:
        return Connect4State(board=self.board.copy())


@register_game("connect4")
class Connect4Game(Game[Connect4State, int]):
    """
    Connect 4 implementation.

    Actions are column indices (0-6).
    """

    _spec = GameSpec(
        name="connect4",
        board_shape=(ROWS, COLS),
        num_actions=COLS,
        num_input_channels=2,  # Current player pieces, opponent pieces
    )

    @property
    def spec(self) -> GameSpec:
        return self._spec

    def initial_state(self) -> Connect4State:
        """Return empty board."""
        return Connect4State(board=np.zeros((ROWS, COLS), dtype=np.int8))

    def legal_actions(self, state: Connect4State) -> list[int]:
        """Return columns that aren't full."""
        return [c for c in range(COLS) if state.board[0, c] == 0]

    def apply_action(self, state: Connect4State, action: int) -> Connect4State:
        """
        Drop piece in column and return new state.

        The new state is canonicalized for the next player (board * -1).
        """
        if action < 0 or action >= COLS:
            raise ValueError(f"Invalid action {action}, must be 0-{COLS-1}")

        if state.board[0, action] != 0:
            raise ValueError(f"Column {action} is full")

        new_board = state.board.copy()

        # Find lowest empty row in column
        row = ROWS - 1
        while row >= 0 and new_board[row, action] != 0:
            row -= 1

        # Place piece (+1 for current player)
        new_board[row, action] = 1

        # Canonicalize: flip perspective for next player
        new_board = -new_board

        return Connect4State(board=new_board)

    def is_terminal(self, state: Connect4State) -> Tuple[bool, float]:
        """
        Check if game is over.

        Because we use canonical form and flip after each move:
        - Opponent's pieces are -1
        - If opponent just made 4-in-a-row, current player lost
        """
        board = state.board

        # Check if opponent (now -1) just won
        if self._has_winner(board, -1):
            return True, -1.0  # Current player lost

        # Check if current player won (shouldn't happen normally)
        if self._has_winner(board, 1):
            return True, 1.0

        # Check for draw (board full)
        if not np.any(board == 0):
            return True, 0.0

        return False, 0.0

    def encode_state(self, state: Connect4State) -> np.ndarray:
        """
        Encode as 2 binary planes:
        - Channel 0: Current player's pieces (1 where board == +1)
        - Channel 1: Opponent's pieces (1 where board == -1)
        """
        board = state.board
        encoded = np.zeros((2, ROWS, COLS), dtype=np.float32)
        encoded[0] = (board == 1).astype(np.float32)
        encoded[1] = (board == -1).astype(np.float32)
        return encoded

    def action_to_index(self, action: int) -> int:
        """Action is already an index (0-6)."""
        return action

    def index_to_action(self, index: int) -> int:
        """Index is already an action (0-6)."""
        return index

    def get_action_mask(self, state: Connect4State) -> np.ndarray:
        """Return boolean mask of legal columns."""
        return state.board[0, :] == 0

    def get_symmetries(
        self,
        state: Connect4State,
        policy: np.ndarray
    ) -> list[tuple[Connect4State, np.ndarray]]:
        """
        Connect 4 has horizontal reflection symmetry.

        Returns original and horizontally flipped version.
        """
        original = (state, policy)

        # Horizontal flip
        flipped_board = np.fliplr(state.board).copy()
        flipped_policy = np.flip(policy).copy()
        flipped = (Connect4State(board=flipped_board), flipped_policy)

        return [original, flipped]

    def render(self, state: Connect4State) -> str:
        """Render board as ASCII art."""
        symbols = {0: ".", 1: "X", -1: "O"}

        lines = []
        lines.append(" " + " ".join(str(i) for i in range(COLS)))
        lines.append("-" * (COLS * 2 + 1))

        for r in range(ROWS):
            row_str = "|" + "|".join(
                symbols[state.board[r, c]] for c in range(COLS)
            ) + "|"
            lines.append(row_str)

        lines.append("-" * (COLS * 2 + 1))
        return "\n".join(lines)

    # --- Helper methods ---

    def _has_winner(self, board: np.ndarray, player: int) -> bool:
        """Check if the given player has 4 in a row."""
        for r in range(ROWS):
            for c in range(COLS):
                # Horizontal
                if c <= COLS - WIN_LENGTH:
                    if self._check_line(board, r, c, 0, 1, player):
                        return True
                # Vertical
                if r <= ROWS - WIN_LENGTH:
                    if self._check_line(board, r, c, 1, 0, player):
                        return True
                # Diagonal down-right
                if r <= ROWS - WIN_LENGTH and c <= COLS - WIN_LENGTH:
                    if self._check_line(board, r, c, 1, 1, player):
                        return True
                # Diagonal down-left
                if r <= ROWS - WIN_LENGTH and c >= WIN_LENGTH - 1:
                    if self._check_line(board, r, c, 1, -1, player):
                        return True
        return False

    def _check_line(
        self,
        board: np.ndarray,
        r: int,
        c: int,
        dr: int,
        dc: int,
        player: int
    ) -> bool:
        """Check if there are 4 in a row starting from (r,c) in direction (dr,dc)."""
        for i in range(WIN_LENGTH):
            if board[r + i * dr, c + i * dc] != player:
                return False
        return True


# Convenience functions for direct use
def create_game() -> Connect4Game:
    """Create a Connect 4 game instance."""
    return Connect4Game()

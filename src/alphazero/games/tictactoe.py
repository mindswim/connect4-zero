"""
Tic-Tac-Toe game implementation.

Simple 3x3 game - perfect for testing the AlphaZero pipeline.
Should reach perfect play (always draw with optimal play) within minutes.

Rules:
- 3x3 board
- Players alternate placing their mark
- First to get 3 in a row (horizontal, vertical, diagonal) wins
- If board fills with no winner, it's a draw
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np

from .base import Game, GameSpec, register_game


BOARD_SIZE = 3


@dataclass
class TicTacToeState:
    """Tic-Tac-Toe game state in canonical form."""
    board: np.ndarray  # shape (3, 3), dtype int8

    def __post_init__(self):
        if self.board.shape != (BOARD_SIZE, BOARD_SIZE):
            raise ValueError(f"Board must be {BOARD_SIZE}x{BOARD_SIZE}")
        if self.board.dtype != np.int8:
            self.board = self.board.astype(np.int8)

    def copy(self) -> TicTacToeState:
        return TicTacToeState(board=self.board.copy())


@register_game("tictactoe")
class TicTacToeGame(Game[TicTacToeState, int]):
    """
    Tic-Tac-Toe implementation.

    Actions are cell indices (0-8), mapping to positions:
    0 | 1 | 2
    ---------
    3 | 4 | 5
    ---------
    6 | 7 | 8
    """

    _spec = GameSpec(
        name="tictactoe",
        board_shape=(BOARD_SIZE, BOARD_SIZE),
        num_actions=BOARD_SIZE * BOARD_SIZE,  # 9
        num_input_channels=2,  # Current player, opponent
    )

    # Winning lines (indices into flattened board)
    WINNING_LINES = [
        # Rows
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        # Columns
        [0, 3, 6],
        [1, 4, 7],
        [2, 5, 8],
        # Diagonals
        [0, 4, 8],
        [2, 4, 6],
    ]

    @property
    def spec(self) -> GameSpec:
        return self._spec

    def initial_state(self) -> TicTacToeState:
        """Return empty board."""
        return TicTacToeState(
            board=np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        )

    def legal_actions(self, state: TicTacToeState) -> list[int]:
        """Return empty cells as actions."""
        flat = state.board.flatten()
        return [i for i in range(9) if flat[i] == 0]

    def apply_action(self, state: TicTacToeState, action: int) -> TicTacToeState:
        """
        Place piece and return new state.

        The new state is canonicalized for the next player (board * -1).
        """
        if action < 0 or action >= 9:
            raise ValueError(f"Invalid action {action}, must be 0-8")

        row, col = divmod(action, BOARD_SIZE)
        if state.board[row, col] != 0:
            raise ValueError(f"Cell {action} is already occupied")

        new_board = state.board.copy()
        new_board[row, col] = 1  # Current player

        # Canonicalize: flip perspective
        new_board = -new_board

        return TicTacToeState(board=new_board)

    def is_terminal(self, state: TicTacToeState) -> Tuple[bool, float]:
        """
        Check if game is over.

        Opponent's pieces are -1 in canonical form.
        """
        flat = state.board.flatten()

        # Check if opponent (now -1) just won
        for line in self.WINNING_LINES:
            if all(flat[i] == -1 for i in line):
                return True, -1.0  # Current player lost

        # Check if current player won (shouldn't happen normally)
        for line in self.WINNING_LINES:
            if all(flat[i] == 1 for i in line):
                return True, 1.0

        # Check for draw (no empty cells)
        if not np.any(flat == 0):
            return True, 0.0

        return False, 0.0

    def encode_state(self, state: TicTacToeState) -> np.ndarray:
        """
        Encode as 2 binary planes:
        - Channel 0: Current player's pieces
        - Channel 1: Opponent's pieces
        """
        board = state.board
        encoded = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        encoded[0] = (board == 1).astype(np.float32)
        encoded[1] = (board == -1).astype(np.float32)
        return encoded

    def action_to_index(self, action: int) -> int:
        """Action is already an index (0-8)."""
        return action

    def index_to_action(self, index: int) -> int:
        """Index is already an action (0-8)."""
        return index

    def get_action_mask(self, state: TicTacToeState) -> np.ndarray:
        """Return boolean mask of legal cells."""
        return state.board.flatten() == 0

    def get_symmetries(
        self,
        state: TicTacToeState,
        policy: np.ndarray
    ) -> list[tuple[TicTacToeState, np.ndarray]]:
        """
        Tic-Tac-Toe has 8-fold symmetry (rotations + reflections).

        Returns all 8 symmetric versions for data augmentation.
        """
        symmetries = []
        board = state.board
        pi = policy.reshape(BOARD_SIZE, BOARD_SIZE)

        # 4 rotations
        for k in range(4):
            rotated_board = np.rot90(board, k)
            rotated_pi = np.rot90(pi, k)

            symmetries.append((
                TicTacToeState(board=rotated_board.copy()),
                rotated_pi.flatten().copy()
            ))

            # Reflection for each rotation
            flipped_board = np.fliplr(rotated_board)
            flipped_pi = np.fliplr(rotated_pi)

            symmetries.append((
                TicTacToeState(board=flipped_board.copy()),
                flipped_pi.flatten().copy()
            ))

        return symmetries

    def render(self, state: TicTacToeState) -> str:
        """Render board as ASCII art."""
        symbols = {0: ".", 1: "X", -1: "O"}

        lines = []
        for r in range(BOARD_SIZE):
            row_str = " | ".join(
                symbols[state.board[r, c]] for c in range(BOARD_SIZE)
            )
            lines.append(f" {row_str} ")
            if r < BOARD_SIZE - 1:
                lines.append("-----------")

        return "\n".join(lines)


def create_game() -> TicTacToeGame:
    """Create a Tic-Tac-Toe game instance."""
    return TicTacToeGame()

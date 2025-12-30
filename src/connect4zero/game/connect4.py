"""
Connect 4 game logic.

Board representation:
- 6 rows x 7 columns
- 0 = empty
- +1 = current player's stones
- -1 = opponent's stones

State is always stored in canonical form (from perspective of player to move).
When switching turns, the board is multiplied by -1.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Tuple

ROWS = 6
COLS = 7
WIN_LENGTH = 4


@dataclass
class GameState:
    """Immutable game state in canonical form."""
    board: np.ndarray  # shape (6, 7), dtype int8

    def __post_init__(self):
        if self.board.shape != (ROWS, COLS):
            raise ValueError(f"Board must be {ROWS}x{COLS}")
        if self.board.dtype != np.int8:
            self.board = self.board.astype(np.int8)

    def copy(self) -> GameState:
        return GameState(board=self.board.copy())


def initial_state() -> GameState:
    """Create the initial empty board state."""
    return GameState(board=np.zeros((ROWS, COLS), dtype=np.int8))


def legal_moves(state: GameState) -> np.ndarray:
    """
    Return a boolean mask of length 7 indicating legal moves.
    A move is legal if the top row of that column is empty.
    """
    return state.board[0, :] == 0


def legal_moves_list(state: GameState) -> list[int]:
    """Return list of legal column indices."""
    mask = legal_moves(state)
    return [i for i in range(COLS) if mask[i]]


def apply_move(state: GameState, action: int) -> GameState:
    """
    Apply a move (drop piece in column) and return new state.

    The new state is automatically canonicalized (board flipped for next player).

    Args:
        state: Current game state
        action: Column index (0-6) to drop piece

    Returns:
        New GameState from the perspective of the next player
    """
    if action < 0 or action >= COLS:
        raise ValueError(f"Invalid action {action}, must be 0-{COLS-1}")

    if state.board[0, action] != 0:
        raise ValueError(f"Column {action} is full")

    new_board = state.board.copy()

    # Find the lowest empty row in the column
    row = ROWS - 1
    while row >= 0 and new_board[row, action] != 0:
        row -= 1

    # Place the piece (+1 for current player)
    new_board[row, action] = 1

    # Canonicalize: flip perspective for next player
    new_board = -new_board

    return GameState(board=new_board)


def _check_line(board: np.ndarray, r: int, c: int, dr: int, dc: int, player: int) -> bool:
    """Check if there are 4 in a row starting from (r,c) in direction (dr,dc)."""
    for i in range(WIN_LENGTH):
        nr, nc = r + i * dr, c + i * dc
        if nr < 0 or nr >= ROWS or nc < 0 or nc >= COLS:
            return False
        if board[nr, nc] != player:
            return False
    return True


def _has_winner(board: np.ndarray, player: int) -> bool:
    """Check if the given player has won."""
    # Check all possible 4-in-a-row starting positions
    for r in range(ROWS):
        for c in range(COLS):
            # Horizontal
            if c <= COLS - WIN_LENGTH:
                if _check_line(board, r, c, 0, 1, player):
                    return True
            # Vertical
            if r <= ROWS - WIN_LENGTH:
                if _check_line(board, r, c, 1, 0, player):
                    return True
            # Diagonal down-right
            if r <= ROWS - WIN_LENGTH and c <= COLS - WIN_LENGTH:
                if _check_line(board, r, c, 1, 1, player):
                    return True
            # Diagonal down-left
            if r <= ROWS - WIN_LENGTH and c >= WIN_LENGTH - 1:
                if _check_line(board, r, c, 1, -1, player):
                    return True
    return False


def is_terminal(state: GameState) -> Tuple[bool, float]:
    """
    Check if the game is over.

    Returns:
        (done, value) where:
        - done: True if game is over
        - value: outcome from current player's perspective
            - +1 if current player wins
            - -1 if current player loses
            - 0 if draw

    Note: Because we use canonical form and flip after each move,
    if the opponent (now -1 in the board) just won, they have 4 in a row
    of -1 values. This means current player lost.
    """
    # Check if opponent just won (they are -1 in canonical form)
    # This happens when the previous player completed a line
    if _has_winner(state.board, -1):
        return True, -1.0  # Current player lost

    # Check if current player has won (shouldn't happen in normal play
    # but included for completeness)
    if _has_winner(state.board, 1):
        return True, 1.0  # Current player won

    # Check for draw (board full)
    if not np.any(state.board == 0):
        return True, 0.0

    return False, 0.0


def render(state: GameState, last_move: int = -1) -> str:
    """
    Render the board as a string for display.

    Shows from current player's perspective:
    - 'X' = current player (+1)
    - 'O' = opponent (-1)
    - '.' = empty
    """
    lines = []
    lines.append(" " + " ".join(str(i) for i in range(COLS)))
    lines.append("-" * (COLS * 2 + 1))

    symbols = {0: ".", 1: "X", -1: "O"}

    for r in range(ROWS):
        row_str = "|" + "|".join(symbols[state.board[r, c]] for c in range(COLS)) + "|"
        lines.append(row_str)

    lines.append("-" * (COLS * 2 + 1))

    if last_move >= 0:
        pointer = " " * (last_move * 2 + 1) + "^"
        lines.append(pointer)

    return "\n".join(lines)


def get_symmetries(state: GameState, pi: np.ndarray) -> list[Tuple[GameState, np.ndarray]]:
    """
    Get symmetric positions for data augmentation.

    Connect 4 has horizontal reflection symmetry.

    Args:
        state: Game state
        pi: Policy vector of length 7

    Returns:
        List of (state, pi) pairs including original and reflected
    """
    original = (state, pi)

    # Horizontal flip
    flipped_board = np.fliplr(state.board).copy()
    flipped_pi = np.flip(pi).copy()
    flipped = (GameState(board=flipped_board), flipped_pi)

    return [original, flipped]

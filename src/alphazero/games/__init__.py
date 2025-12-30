"""
Game implementations for AlphaZero.

Each game implements the Game interface from base.py.
"""

from .base import (
    Game,
    GameSpec,
    State,
    Action,
    register_game,
    get_game,
    list_games,
)

# Import games to register them
from . import connect4
from . import tictactoe

__all__ = [
    "Game",
    "GameSpec",
    "State",
    "Action",
    "register_game",
    "get_game",
    "list_games",
]

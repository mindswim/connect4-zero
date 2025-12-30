"""
AlphaZero - Game-agnostic deep reinforcement learning.

Train AI to play any two-player perfect-information game using
the AlphaZero algorithm (self-play + MCTS + neural network).

Supported games:
- Tic-Tac-Toe (trivial - for testing)
- Connect 4
- More coming soon (Othello, Chess, etc.)

Usage:
    from alphazero.games import get_game, list_games
    from alphazero.net import create_model
    from alphazero.mcts import BatchedMCTS
    from alphazero.play import Difficulty, get_difficulty_config

    # List available games
    print(list_games())  # ['connect4', 'tictactoe']

    # Get a game
    game = get_game('connect4')

    # Create model for the game
    model = create_model(game.spec)

    # Play with difficulty settings
    config = get_difficulty_config(Difficulty.HARD)
    mcts = BatchedMCTS(game, model, device, ...)
    root = mcts.search(state, config.simulations)
    action = root.select_action(config.temperature)
"""

__version__ = "0.2.0"

from . import games
from . import net
from . import mcts
from . import play

__all__ = [
    "games",
    "net",
    "mcts",
    "play",
    "__version__",
]

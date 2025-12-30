"""
Self-play worker for generating training games.

Plays games using MCTS with the current model, collecting
(state, policy, value) training samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple, List
import numpy as np
import torch

from ..game import (
    GameState,
    initial_state,
    apply_move,
    is_terminal,
    encode_state,
    get_symmetries,
    legal_moves_list,
)
from ..mcts import MCTS, Node


@dataclass
class GameRecord:
    """Record of a complete self-play game."""

    states: List[np.ndarray]  # Encoded states
    policies: List[np.ndarray]  # MCTS policies
    outcome: float  # +1 if player 0 wins, -1 if player 1 wins, 0 draw
    moves: List[int]  # Actions taken
    num_moves: int


class SelfPlayWorker:
    """
    Self-play game generator.

    Uses MCTS to play games against itself, collecting training data.

    Args:
        evaluate_fn: Neural network evaluation function
        num_simulations: MCTS simulations per move
        c_puct: PUCT exploration constant
        dirichlet_alpha: Dirichlet noise alpha
        dirichlet_epsilon: Dirichlet noise weight
        temp_threshold: Move number after which to use greedy selection
    """

    def __init__(
        self,
        evaluate_fn: Callable[[GameState], Tuple[np.ndarray, float]],
        num_simulations: int = 100,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        temp_threshold: int = 10,
    ):
        self.evaluate_fn = evaluate_fn
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.temp_threshold = temp_threshold

    def play_game(self, augment: bool = True) -> GameRecord:
        """
        Play a complete self-play game.

        Args:
            augment: Whether to include symmetric positions

        Returns:
            GameRecord with training data
        """
        state = initial_state()
        states: List[np.ndarray] = []
        policies: List[np.ndarray] = []
        moves: List[int] = []

        mcts = MCTS(
            evaluate_fn=self.evaluate_fn,
            c_puct=self.c_puct,
            dirichlet_alpha=self.dirichlet_alpha,
            dirichlet_epsilon=self.dirichlet_epsilon,
            add_noise=True,
        )

        move_num = 0
        while True:
            # Run MCTS search
            root = mcts.search(state, self.num_simulations)

            # Get policy from visit counts
            temperature = 1.0 if move_num < self.temp_threshold else 0.0
            policy = root.get_policy(temperature)

            # Store state and policy
            encoded = encode_state(state)

            if augment:
                # Add both original and symmetric positions
                for sym_state, sym_pi in get_symmetries(state, policy):
                    states.append(encode_state(sym_state))
                    policies.append(sym_pi)
            else:
                states.append(encoded)
                policies.append(policy)

            # Select action
            action = root.select_action(temperature)
            moves.append(action)

            # Apply move
            state = apply_move(state, action)
            move_num += 1

            # Check for game end
            done, value = is_terminal(state)
            if done:
                # value is from perspective of player TO MOVE (opponent of who just played)
                # After apply_move, state is canonicalized to next player's view
                # So value=-1 means current player lost, i.e., previous player won
                # Convert to player 0's perspective for training
                # If move_num is odd, player 0 just moved
                # If move_num is even, player 1 just moved
                if move_num % 2 == 1:
                    # Player 0 just moved, value is from player 1's perspective
                    outcome = -value  # Flip to player 0's perspective
                else:
                    # Player 1 just moved, value is from player 0's perspective
                    outcome = value

                return GameRecord(
                    states=states,
                    policies=policies,
                    outcome=outcome,
                    moves=moves,
                    num_moves=move_num,
                )

    def generate_games(
        self,
        num_games: int,
        augment: bool = True,
        progress_callback: Callable[[int], None] = None,
    ) -> List[GameRecord]:
        """
        Generate multiple self-play games.

        Args:
            num_games: Number of games to generate
            augment: Whether to use data augmentation
            progress_callback: Optional callback(games_completed)

        Returns:
            List of GameRecords
        """
        games = []
        for i in range(num_games):
            game = self.play_game(augment=augment)
            games.append(game)
            if progress_callback:
                progress_callback(i + 1)
        return games


def play_evaluation_game(
    player1_fn: Callable[[GameState], Tuple[np.ndarray, float]],
    player2_fn: Callable[[GameState], Tuple[np.ndarray, float]],
    num_simulations: int = 100,
    c_puct: float = 1.5,
) -> Tuple[float, int]:
    """
    Play an evaluation game between two players.

    Args:
        player1_fn: Evaluation function for player 1
        player2_fn: Evaluation function for player 2
        num_simulations: MCTS simulations per move
        c_puct: PUCT constant

    Returns:
        (outcome, num_moves) where outcome is +1 if player 1 wins, -1 if loses, 0 draw
    """
    state = initial_state()

    mcts1 = MCTS(
        evaluate_fn=player1_fn,
        c_puct=c_puct,
        add_noise=False,  # No noise for evaluation
    )
    mcts2 = MCTS(
        evaluate_fn=player2_fn,
        c_puct=c_puct,
        add_noise=False,
    )

    move_num = 0
    while True:
        # Alternate players
        if move_num % 2 == 0:
            mcts = mcts1
        else:
            mcts = mcts2

        root = mcts.search(state, num_simulations)
        action = root.select_action(temperature=0.0)  # Greedy
        state = apply_move(state, action)
        move_num += 1

        done, value = is_terminal(state)
        if done:
            # value is from perspective of player to move (loser if someone won)
            # Convert to player 1's perspective
            if move_num % 2 == 1:
                # Player 1 just moved, value is from player 2's perspective
                outcome = -value
            else:
                # Player 2 just moved, value is from player 1's perspective
                outcome = value
            return outcome, move_num


def play_random_game() -> GameRecord:
    """Play a game with random moves (for testing)."""
    state = initial_state()
    states = []
    policies = []
    moves = []

    move_num = 0
    while True:
        legal = legal_moves_list(state)
        policy = np.zeros(7, dtype=np.float32)
        for a in legal:
            policy[a] = 1.0 / len(legal)

        states.append(encode_state(state))
        policies.append(policy)

        action = np.random.choice(legal)
        moves.append(action)

        state = apply_move(state, action)
        move_num += 1

        done, value = is_terminal(state)
        if done:
            # value is from perspective of player to move
            if move_num % 2 == 1:
                # Player 0 just moved, flip to player 0's perspective
                outcome = -value
            else:
                # Player 1 just moved, value is already from player 0's perspective
                outcome = value

            return GameRecord(
                states=states,
                policies=policies,
                outcome=outcome,
                moves=moves,
                num_moves=move_num,
            )

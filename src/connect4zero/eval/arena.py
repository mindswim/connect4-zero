"""
Arena for evaluating models through head-to-head matches.

Used to determine if a new candidate model should replace the current best.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple, Optional
import torch

from ..game import GameState
from ..mcts import MCTS, create_evaluator
from ..selfplay import play_evaluation_game
from ..net import Connect4Net


@dataclass
class ArenaResult:
    """Results from arena evaluation."""

    wins: int
    losses: int
    draws: int
    total_games: int
    win_rate: float

    @property
    def score(self) -> float:
        """Win rate counting draws as half."""
        return (self.wins + 0.5 * self.draws) / self.total_games if self.total_games > 0 else 0.0


class Arena:
    """
    Arena for model evaluation matches.

    Args:
        num_simulations: MCTS simulations per move
        c_puct: PUCT exploration constant
        device: Torch device
    """

    def __init__(
        self,
        num_simulations: int = 100,
        c_puct: float = 1.5,
        device: torch.device = None,
    ):
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.device = device

    def evaluate(
        self,
        candidate: Connect4Net,
        best: Connect4Net,
        num_games: int = 100,
        progress_callback: Callable[[int, str], None] = None,
    ) -> ArenaResult:
        """
        Evaluate candidate model against best model.

        Plays num_games matches, alternating who goes first.

        Args:
            candidate: New candidate model
            best: Current best model
            num_games: Number of games to play
            progress_callback: Optional callback(games_completed, result)

        Returns:
            ArenaResult from candidate's perspective
        """
        candidate.eval()
        best.eval()

        candidate_fn = create_evaluator(candidate, self.device)
        best_fn = create_evaluator(best, self.device)

        wins = 0
        losses = 0
        draws = 0

        for i in range(num_games):
            # Alternate who plays first
            if i % 2 == 0:
                # Candidate plays first (player 1)
                outcome, _ = play_evaluation_game(
                    player1_fn=candidate_fn,
                    player2_fn=best_fn,
                    num_simulations=self.num_simulations,
                    c_puct=self.c_puct,
                )
            else:
                # Best plays first, candidate is player 2
                outcome, _ = play_evaluation_game(
                    player1_fn=best_fn,
                    player2_fn=candidate_fn,
                    num_simulations=self.num_simulations,
                    c_puct=self.c_puct,
                )
                outcome = -outcome  # Flip to candidate's perspective

            if outcome > 0:
                wins += 1
                result = "W"
            elif outcome < 0:
                losses += 1
                result = "L"
            else:
                draws += 1
                result = "D"

            if progress_callback:
                progress_callback(i + 1, result)

        total = wins + losses + draws
        win_rate = wins / total if total > 0 else 0.0

        return ArenaResult(
            wins=wins,
            losses=losses,
            draws=draws,
            total_games=total,
            win_rate=win_rate,
        )

    def evaluate_vs_random(
        self,
        model: Connect4Net,
        num_games: int = 50,
        progress_callback: Callable[[int, str], None] = None,
    ) -> ArenaResult:
        """
        Evaluate model against random player.

        Args:
            model: Model to evaluate
            num_games: Number of games
            progress_callback: Optional callback

        Returns:
            ArenaResult from model's perspective
        """
        from ..mcts import create_random_evaluator

        model.eval()
        model_fn = create_evaluator(model, self.device)
        random_fn = create_random_evaluator()

        wins = 0
        losses = 0
        draws = 0

        for i in range(num_games):
            if i % 2 == 0:
                outcome, _ = play_evaluation_game(
                    player1_fn=model_fn,
                    player2_fn=random_fn,
                    num_simulations=self.num_simulations,
                    c_puct=self.c_puct,
                )
            else:
                outcome, _ = play_evaluation_game(
                    player1_fn=random_fn,
                    player2_fn=model_fn,
                    num_simulations=self.num_simulations,
                    c_puct=self.c_puct,
                )
                outcome = -outcome

            if outcome > 0:
                wins += 1
                result = "W"
            elif outcome < 0:
                losses += 1
                result = "L"
            else:
                draws += 1
                result = "D"

            if progress_callback:
                progress_callback(i + 1, result)

        total = wins + losses + draws
        win_rate = wins / total if total > 0 else 0.0

        return ArenaResult(
            wins=wins,
            losses=losses,
            draws=draws,
            total_games=total,
            win_rate=win_rate,
        )


def should_accept(result: ArenaResult, threshold: float = 0.55) -> bool:
    """
    Determine if candidate should replace best.

    Args:
        result: Arena evaluation result
        threshold: Minimum win rate to accept (default 55%)

    Returns:
        True if candidate should be accepted
    """
    return result.score >= threshold

"""
Difficulty system for AI players.

Difficulty is controlled by two main parameters:
1. Simulations: How many MCTS simulations (thinking depth)
2. Temperature: Randomness in move selection (0 = always best, 1 = proportional to visits)

Higher simulations = stronger play (more lookahead)
Lower temperature = more consistent (less random mistakes)

The system supports:
- Preset difficulties (Easy, Medium, Hard, Impossible)
- Continuous slider (0-100 mapped to simulation count)
- Adaptive difficulty (adjusts based on player win rate)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import math


class Difficulty(Enum):
    """Preset difficulty levels."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    IMPOSSIBLE = "impossible"


@dataclass
class DifficultyConfig:
    """
    Configuration for AI difficulty.

    Attributes:
        simulations: Number of MCTS simulations per move
        temperature: Move selection temperature (0 = greedy, higher = more random)
        name: Human-readable name
        description: Description for UI
    """
    simulations: int
    temperature: float
    name: str = ""
    description: str = ""

    def __post_init__(self):
        if self.simulations < 1:
            raise ValueError("Simulations must be at least 1")
        if self.temperature < 0:
            raise ValueError("Temperature must be non-negative")


# Default presets - work well for most games
DIFFICULTY_PRESETS: dict[Difficulty, DifficultyConfig] = {
    Difficulty.EASY: DifficultyConfig(
        simulations=25,
        temperature=0.8,
        name="Easy",
        description="Beginner friendly - makes mistakes often",
    ),
    Difficulty.MEDIUM: DifficultyConfig(
        simulations=75,
        temperature=0.4,
        name="Medium",
        description="Moderate challenge - occasional mistakes",
    ),
    Difficulty.HARD: DifficultyConfig(
        simulations=200,
        temperature=0.1,
        name="Hard",
        description="Strong play - rare mistakes",
    ),
    Difficulty.IMPOSSIBLE: DifficultyConfig(
        simulations=800,
        temperature=0.0,
        name="Impossible",
        description="Maximum strength - always plays optimally",
    ),
}


# Game-specific presets (some games need different scaling)
GAME_DIFFICULTY_OVERRIDES: dict[str, dict[Difficulty, DifficultyConfig]] = {
    "tictactoe": {
        # Tic-tac-toe is simpler, needs fewer sims
        Difficulty.EASY: DifficultyConfig(
            simulations=10,
            temperature=1.0,
            name="Easy",
            description="Makes random moves sometimes",
        ),
        Difficulty.MEDIUM: DifficultyConfig(
            simulations=25,
            temperature=0.5,
            name="Medium",
            description="Decent but beatable",
        ),
        Difficulty.HARD: DifficultyConfig(
            simulations=50,
            temperature=0.1,
            name="Hard",
            description="Strong play",
        ),
        Difficulty.IMPOSSIBLE: DifficultyConfig(
            simulations=100,
            temperature=0.0,
            name="Impossible",
            description="Perfect play - will always draw or win",
        ),
    },
    "chess": {
        # Chess needs more thinking for good play
        Difficulty.EASY: DifficultyConfig(
            simulations=50,
            temperature=1.0,
            name="Easy",
            description="Beginner level",
        ),
        Difficulty.MEDIUM: DifficultyConfig(
            simulations=200,
            temperature=0.5,
            name="Medium",
            description="Club player level",
        ),
        Difficulty.HARD: DifficultyConfig(
            simulations=800,
            temperature=0.1,
            name="Hard",
            description="Strong amateur",
        ),
        Difficulty.IMPOSSIBLE: DifficultyConfig(
            simulations=1600,
            temperature=0.0,
            name="Impossible",
            description="Maximum strength",
        ),
    },
}


def get_difficulty_config(
    difficulty: Difficulty,
    game_name: Optional[str] = None,
) -> DifficultyConfig:
    """
    Get difficulty configuration.

    Args:
        difficulty: Preset difficulty level
        game_name: Optional game name for game-specific tuning

    Returns:
        DifficultyConfig for the specified difficulty
    """
    if game_name and game_name in GAME_DIFFICULTY_OVERRIDES:
        return GAME_DIFFICULTY_OVERRIDES[game_name][difficulty]
    return DIFFICULTY_PRESETS[difficulty]


def difficulty_from_slider(
    value: float,
    min_sims: int = 10,
    max_sims: int = 1000,
) -> DifficultyConfig:
    """
    Create difficulty config from a continuous slider value.

    Maps a 0-100 slider to simulation count and temperature.
    Uses exponential scaling so low values feel more different
    (difference between 10 and 20 sims is more noticeable than 900 vs 910).

    Args:
        value: Slider value from 0 to 100
        min_sims: Minimum simulations (at value=0)
        max_sims: Maximum simulations (at value=100)

    Returns:
        DifficultyConfig for the slider position
    """
    # Clamp value to valid range
    value = max(0.0, min(100.0, value))

    # Normalize to 0-1
    t = value / 100.0

    # Exponential scaling for simulations
    # This makes the slider feel more linear perceptually
    log_min = math.log(min_sims)
    log_max = math.log(max_sims)
    simulations = int(math.exp(log_min + t * (log_max - log_min)))

    # Temperature decreases as difficulty increases
    # High difficulty (100) = temp 0, Low difficulty (0) = temp 1
    temperature = max(0.0, 1.0 - t)

    # Generate name based on value
    if value < 25:
        name = "Beginner"
    elif value < 50:
        name = "Intermediate"
    elif value < 75:
        name = "Advanced"
    elif value < 95:
        name = "Expert"
    else:
        name = "Maximum"

    return DifficultyConfig(
        simulations=simulations,
        temperature=temperature,
        name=name,
        description=f"{simulations} simulations, temp={temperature:.2f}",
    )


class AdaptiveDifficulty:
    """
    Adaptive difficulty that adjusts based on player performance.

    Tracks win/loss record and adjusts AI strength to target
    a specific win rate for the player (default 50%).

    This creates a more engaging experience - the AI gets
    harder when you're winning and easier when you're losing.
    """

    def __init__(
        self,
        target_win_rate: float = 0.5,
        min_sims: int = 10,
        max_sims: int = 1000,
        initial_sims: int = 100,
        adjustment_rate: float = 0.1,
        window_size: int = 20,
    ):
        """
        Initialize adaptive difficulty.

        Args:
            target_win_rate: Target player win rate (0.5 = 50%)
            min_sims: Minimum simulations
            max_sims: Maximum simulations
            initial_sims: Starting simulation count
            adjustment_rate: How fast to adjust (0-1)
            window_size: Number of recent games to consider
        """
        self.target_win_rate = target_win_rate
        self.min_sims = min_sims
        self.max_sims = max_sims
        self.current_sims = initial_sims
        self.adjustment_rate = adjustment_rate
        self.window_size = window_size

        # Track recent results (1 = player win, 0 = player loss, 0.5 = draw)
        self.results: list[float] = []

    def record_result(self, player_won: bool, draw: bool = False) -> None:
        """
        Record game result.

        Args:
            player_won: True if human player won
            draw: True if game was a draw
        """
        if draw:
            result = 0.5
        elif player_won:
            result = 1.0
        else:
            result = 0.0

        self.results.append(result)

        # Keep only recent games
        if len(self.results) > self.window_size:
            self.results = self.results[-self.window_size:]

        # Adjust difficulty
        self._adjust()

    def _adjust(self) -> None:
        """Adjust simulation count based on recent performance."""
        if len(self.results) < 3:
            return  # Need a few games before adjusting

        # Calculate recent win rate
        recent_win_rate = sum(self.results) / len(self.results)

        # If player winning too much, increase difficulty
        # If player losing too much, decrease difficulty
        error = recent_win_rate - self.target_win_rate

        # Adjust simulations (log scale)
        log_sims = math.log(self.current_sims)
        log_min = math.log(self.min_sims)
        log_max = math.log(self.max_sims)

        # Positive error = player winning too much = increase sims
        adjustment = error * self.adjustment_rate * (log_max - log_min)
        new_log_sims = max(log_min, min(log_max, log_sims + adjustment))

        self.current_sims = int(math.exp(new_log_sims))

    def get_config(self) -> DifficultyConfig:
        """Get current difficulty configuration."""
        # Temperature also scales with difficulty
        t = (math.log(self.current_sims) - math.log(self.min_sims)) / \
            (math.log(self.max_sims) - math.log(self.min_sims))
        temperature = max(0.0, 0.5 * (1.0 - t))

        return DifficultyConfig(
            simulations=self.current_sims,
            temperature=temperature,
            name="Adaptive",
            description=f"Adapting to your skill ({self.current_sims} sims)",
        )

    @property
    def current_win_rate(self) -> Optional[float]:
        """Get player's recent win rate."""
        if not self.results:
            return None
        return sum(self.results) / len(self.results)

    def reset(self) -> None:
        """Reset tracking (start fresh)."""
        self.results = []
        self.current_sims = (self.min_sims + self.max_sims) // 4


# Convenience function
def adaptive_difficulty(**kwargs) -> AdaptiveDifficulty:
    """Create an adaptive difficulty tracker."""
    return AdaptiveDifficulty(**kwargs)

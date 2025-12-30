"""
Configuration management for Connect 4 AlphaZero.

Uses dataclasses for clean configuration with sensible defaults.
Supports loading from YAML files.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class MCTSConfig:
    """MCTS configuration."""

    num_simulations: int = 100
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temp_threshold: int = 10  # Moves before using greedy selection


@dataclass
class NetworkConfig:
    """Neural network configuration."""

    num_channels: int = 64
    num_blocks: int = 5


@dataclass
class TrainConfig:
    """Training configuration."""

    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    train_steps_per_iter: int = 300
    value_loss_weight: float = 1.0


@dataclass
class SelfPlayConfig:
    """Self-play configuration."""

    games_per_iter: int = 25
    max_moves: int = 42
    augment_data: bool = True


@dataclass
class EvalConfig:
    """Evaluation configuration."""

    num_matches: int = 100
    accept_threshold: float = 0.55
    eval_simulations: int = 100


@dataclass
class BufferConfig:
    """Replay buffer configuration."""

    capacity: int = 200_000
    min_samples: int = 1000  # Minimum samples before training


@dataclass
class Config:
    """Full training configuration."""

    # Component configs
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    buffer: BufferConfig = field(default_factory=BufferConfig)

    # Global settings
    num_iterations: int = 100
    checkpoint_dir: str = "checkpoints"
    data_dir: str = "data"
    log_dir: str = "runs"

    # Device (auto-detected if not specified)
    device: Optional[str] = None

    # Random seed
    seed: int = 42

    def save(self, path: str) -> None:
        """Save config to YAML file."""
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

    @classmethod
    def load(cls, path: str) -> Config:
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        # Parse nested configs
        return cls(
            mcts=MCTSConfig(**data.get("mcts", {})),
            network=NetworkConfig(**data.get("network", {})),
            train=TrainConfig(**data.get("train", {})),
            selfplay=SelfPlayConfig(**data.get("selfplay", {})),
            eval=EvalConfig(**data.get("eval", {})),
            buffer=BufferConfig(**data.get("buffer", {})),
            num_iterations=data.get("num_iterations", 100),
            checkpoint_dir=data.get("checkpoint_dir", "checkpoints"),
            data_dir=data.get("data_dir", "data"),
            log_dir=data.get("log_dir", "runs"),
            device=data.get("device"),
            seed=data.get("seed", 42),
        )

    def ensure_dirs(self) -> None:
        """Create output directories if they don't exist."""
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)


def get_default_config() -> Config:
    """Get default configuration optimized for M3 Mac."""
    return Config()

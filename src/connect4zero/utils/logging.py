"""
Logging utilities with rich formatting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

from rich.console import Console
from rich.table import Table
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from rich.panel import Panel


console = Console()


@dataclass
class IterationMetrics:
    """Metrics for one training iteration."""

    iteration: int
    games_played: int
    samples_collected: int
    buffer_size: int
    avg_game_length: float
    policy_loss: float
    value_loss: float
    total_loss: float
    policy_entropy: float
    learning_rate: float
    arena_win_rate: Optional[float] = None
    model_accepted: Optional[bool] = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class Logger:
    """
    Training logger with rich output and JSON logging.

    Args:
        log_dir: Directory for log files
        verbose: Whether to print to console
    """

    def __init__(self, log_dir: str = "runs", verbose: bool = True):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

        # Create log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"train_{timestamp}.jsonl"

        self.metrics_history: list[IterationMetrics] = []

    def log_iteration(self, metrics: IterationMetrics) -> None:
        """Log metrics for one iteration."""
        self.metrics_history.append(metrics)

        # Write to JSON log
        with open(self.log_file, "a") as f:
            f.write(json.dumps(asdict(metrics)) + "\n")

        # Print to console
        if self.verbose:
            self._print_iteration(metrics)

    def _print_iteration(self, m: IterationMetrics) -> None:
        """Print iteration summary to console."""
        table = Table(title=f"Iteration {m.iteration}", show_header=False, box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Games", str(m.games_played))
        table.add_row("Samples", str(m.samples_collected))
        table.add_row("Buffer", str(m.buffer_size))
        table.add_row("Avg Game Len", f"{m.avg_game_length:.1f}")
        table.add_row("Policy Loss", f"{m.policy_loss:.4f}")
        table.add_row("Value Loss", f"{m.value_loss:.4f}")
        table.add_row("Entropy", f"{m.policy_entropy:.3f}")
        table.add_row("LR", f"{m.learning_rate:.2e}")

        if m.arena_win_rate is not None:
            table.add_row("Arena Win%", f"{m.arena_win_rate*100:.1f}%")
            status = "[green]Accepted[/]" if m.model_accepted else "[red]Rejected[/]"
            table.add_row("Status", status)

        console.print(table)
        console.print()

    def log_message(self, message: str, style: str = "white") -> None:
        """Log a message."""
        if self.verbose:
            console.print(f"[{style}]{message}[/]")

    def log_info(self, message: str) -> None:
        """Log info message."""
        self.log_message(message, "blue")

    def log_success(self, message: str) -> None:
        """Log success message."""
        self.log_message(message, "green")

    def log_warning(self, message: str) -> None:
        """Log warning message."""
        self.log_message(message, "yellow")

    def log_error(self, message: str) -> None:
        """Log error message."""
        self.log_message(message, "red")


def create_progress() -> Progress:
    """Create a rich progress bar with elapsed/remaining time."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("eta"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )


def print_config(config: Any) -> None:
    """Print configuration in a nice format."""
    from dataclasses import asdict

    table = Table(title="Configuration", show_header=True)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="white")

    def add_dict(d: dict, prefix: str = "") -> None:
        for k, v in d.items():
            key = f"{prefix}{k}" if prefix else k
            if isinstance(v, dict):
                add_dict(v, f"{key}.")
            else:
                table.add_row(key, str(v))

    add_dict(asdict(config))
    console.print(table)


def print_board(board_str: str, title: str = "Board") -> None:
    """Print a game board in a panel."""
    console.print(Panel(board_str, title=title, border_style="blue"))

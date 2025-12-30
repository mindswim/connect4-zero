"""
Command-line interface for AlphaZero.

Commands:
- train: Train a model for any supported game
- play: Play against a trained model
- list-games: Show available games
- benchmark: Test MCTS performance
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

app = typer.Typer(
    name="alphazero",
    help="AlphaZero - Train AI for any game",
    no_args_is_help=True,
)

console = Console()


def create_progress() -> Progress:
    """Create a rich progress bar."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    )


@app.command("list-games")
def list_games_cmd() -> None:
    """List all available games."""
    from .games import list_games, get_game

    table = Table(title="Available Games")
    table.add_column("Name", style="cyan")
    table.add_column("Board", style="green")
    table.add_column("Actions", style="yellow")

    for name in list_games():
        game = get_game(name)
        spec = game.spec
        board_str = "x".join(str(d) for d in spec.board_shape)
        table.add_row(name, board_str, str(spec.num_actions))

    console.print(table)


@app.command()
def train(
    game_name: str = typer.Argument(..., help="Game to train (e.g., 'connect4', 'tictactoe')"),
    iterations: int = typer.Option(100, "--iterations", "-n", help="Training iterations"),
    resume: Optional[Path] = typer.Option(None, "--resume", "-r", help="Resume from checkpoint"),
    checkpoint_dir: Path = typer.Option(Path("checkpoints"), "--checkpoint-dir", "-c"),
    games_per_iter: int = typer.Option(25, "--games", "-g", help="Self-play games per iteration"),
    simulations: int = typer.Option(100, "--sims", "-s", help="MCTS simulations per move"),
    batch_size: int = typer.Option(8, "--batch-size", "-b", help="MCTS batch size"),
    lr: float = typer.Option(1e-3, "--lr", help="Learning rate"),
    device: str = typer.Option("auto", "--device", "-d", help="Device (auto, cpu, cuda, mps)"),
) -> None:
    """Train AlphaZero for a game."""
    import torch
    import numpy as np
    from .games import get_game
    from .net import create_model, save_checkpoint, load_checkpoint, get_model_config
    from .mcts import BatchedMCTS

    # Get game
    try:
        game = get_game(game_name)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/]")
        raise typer.Exit(1)

    console.print(f"[bold blue]Training AlphaZero for {game_name}[/]")

    # Select device
    if device == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
        elif torch.backends.mps.is_available():
            dev = torch.device("mps")
        else:
            dev = torch.device("cpu")
    else:
        dev = torch.device(device)

    console.print(f"[blue]Using device: {dev}[/]")

    # Create or load model
    model_config = get_model_config(game_name)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if resume and resume.exists():
        model, checkpoint = load_checkpoint(str(resume), dev)
        start_iter = checkpoint.get("iteration", 0) + 1
        console.print(f"[green]Resumed from iteration {start_iter - 1}[/]")
    else:
        model = create_model(
            game.spec,
            num_channels=model_config["num_channels"],
            num_blocks=model_config["num_blocks"],
            device=dev,
        )
        start_iter = 1

    # Training components
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Simple replay buffer
    buffer_states = []
    buffer_policies = []
    buffer_values = []
    max_buffer_size = 50000

    # Training loop
    for iteration in range(start_iter, iterations + 1):
        console.print(f"\n[bold blue]{'='*50}[/]")
        console.print(f"[bold blue]Iteration {iteration}/{iterations}[/]")
        console.print(f"[bold blue]{'='*50}[/]")

        # Self-play
        console.print("[cyan]Generating self-play games...[/]")
        model.eval()

        mcts = BatchedMCTS(
            game=game,
            model=model,
            device=dev,
            c_puct=1.5,
            batch_size=batch_size,
            add_noise=True,
        )

        games_data = []
        total_moves = 0

        with create_progress() as progress:
            task = progress.add_task("Self-play", total=games_per_iter)

            for _ in range(games_per_iter):
                game_states = []
                game_policies = []
                state = game.initial_state()
                move_num = 0

                while True:
                    # MCTS search
                    root = mcts.search(state, simulations)

                    # Get policy from visit counts
                    temp = 1.0 if move_num < 10 else 0.0
                    policy = root.get_policy(temp)

                    # Store for training (with symmetries)
                    for sym_state, sym_pi in game.get_symmetries(state, policy):
                        game_states.append(game.encode_state(sym_state))
                        game_policies.append(sym_pi)

                    # Select and apply action
                    action = root.select_action(temp)
                    state = game.apply_action(state, action)
                    move_num += 1

                    # Check terminal
                    done, value = game.is_terminal(state)
                    if done:
                        # Assign values (alternating perspective)
                        game_values = []
                        for i in range(len(game_states)):
                            # Value from perspective of player who made move i
                            moves_from_end = len(game_states) - i
                            if moves_from_end % 2 == 1:
                                game_values.append(-value)
                            else:
                                game_values.append(value)

                        games_data.append((game_states, game_policies, game_values))
                        total_moves += move_num
                        break

                progress.update(task, advance=1)

        # Add to buffer
        for states, policies, values in games_data:
            buffer_states.extend(states)
            buffer_policies.extend(policies)
            buffer_values.extend(values)

        # Trim buffer if too large
        if len(buffer_states) > max_buffer_size:
            buffer_states = buffer_states[-max_buffer_size:]
            buffer_policies = buffer_policies[-max_buffer_size:]
            buffer_values = buffer_values[-max_buffer_size:]

        samples = sum(len(g[0]) for g in games_data)
        avg_length = total_moves / games_per_iter
        console.print(f"[green]Generated {games_per_iter} games, {samples} samples[/]")
        console.print(f"[green]Average game length: {avg_length:.1f}, Buffer size: {len(buffer_states)}[/]")

        # Training
        if len(buffer_states) >= 256:
            console.print("[cyan]Training...[/]")
            model.train()

            train_steps = 100
            batch = 256
            total_loss = 0.0

            for _ in range(train_steps):
                # Sample batch
                indices = np.random.choice(len(buffer_states), batch, replace=False)
                states_batch = np.stack([buffer_states[i] for i in indices])
                policies_batch = np.stack([buffer_policies[i] for i in indices])
                values_batch = np.array([buffer_values[i] for i in indices])

                # Convert to tensors
                states_t = torch.from_numpy(states_batch).to(dev)
                policies_t = torch.from_numpy(policies_batch).to(dev)
                values_t = torch.from_numpy(values_batch).float().to(dev)

                # Forward
                policy_logits, value_pred = model(states_t)

                # Losses
                policy_loss = -torch.sum(
                    policies_t * torch.log_softmax(policy_logits, dim=-1),
                    dim=-1
                ).mean()
                value_loss = torch.nn.functional.mse_loss(
                    value_pred.squeeze(-1),
                    values_t
                )
                loss = policy_loss + value_loss

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / train_steps
            console.print(f"[green]Average loss: {avg_loss:.4f}[/]")

        # Save checkpoint
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            iteration=iteration,
            path=str(checkpoint_dir / f"{game_name}_iter_{iteration:04d}.pt"),
        )
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            iteration=iteration,
            path=str(checkpoint_dir / f"{game_name}_best.pt"),
        )

    console.print("\n[bold green]Training complete![/]")


@app.command()
def play(
    game_name: str = typer.Argument(..., help="Game to play"),
    model_path: Optional[Path] = typer.Option(None, "--model", "-m", help="Model checkpoint"),
    difficulty: str = typer.Option("medium", "--difficulty", "-d", help="easy/medium/hard/impossible"),
    human_first: bool = typer.Option(True, "--first/--second", help="Human plays first"),
    device: str = typer.Option("auto", "--device", help="Device"),
) -> None:
    """Play against a trained model."""
    import torch
    from .games import get_game
    from .net import load_checkpoint, create_model, get_model_config
    from .mcts import BatchedMCTS
    from .play import Difficulty, get_difficulty_config

    # Get game
    try:
        game = get_game(game_name)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/]")
        raise typer.Exit(1)

    # Select device
    if device == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
        elif torch.backends.mps.is_available():
            dev = torch.device("mps")
        else:
            dev = torch.device("cpu")
    else:
        dev = torch.device(device)

    # Load or create model
    if model_path and model_path.exists():
        model, _ = load_checkpoint(str(model_path), dev)
        console.print(f"[green]Loaded model from {model_path}[/]")
    else:
        model_config = get_model_config(game_name)
        model = create_model(
            game.spec,
            num_channels=model_config["num_channels"],
            num_blocks=model_config["num_blocks"],
            device=dev,
        )
        console.print("[yellow]Using untrained model (random play)[/]")

    model.eval()

    # Get difficulty settings
    try:
        diff = Difficulty(difficulty.lower())
    except ValueError:
        console.print(f"[red]Invalid difficulty. Choose: easy, medium, hard, impossible[/]")
        raise typer.Exit(1)

    diff_config = get_difficulty_config(diff, game_name)
    console.print(f"[blue]Difficulty: {diff_config.name} ({diff_config.simulations} sims)[/]")

    mcts = BatchedMCTS(
        game=game,
        model=model,
        device=dev,
        add_noise=False,
    )

    # Play game
    state = game.initial_state()
    human_turn = human_first

    console.print(f"\n[bold]Playing {game_name}[/]")
    console.print("You are X, AI is O\n")

    while True:
        console.print(game.render(state))
        console.print()

        done, value = game.is_terminal(state)
        if done:
            if value > 0:
                winner = "You" if not human_turn else "AI"
            elif value < 0:
                winner = "AI" if not human_turn else "You"
            else:
                winner = None

            if winner:
                color = "green" if winner == "You" else "red"
                console.print(f"[{color}]{winner} win![/]")
            else:
                console.print("[yellow]Draw![/]")
            break

        if human_turn:
            legal = game.legal_actions(state)
            while True:
                try:
                    action_str = typer.prompt(f"Your move {legal}")
                    action = int(action_str)
                    if action in legal:
                        break
                    console.print("[red]Invalid move[/]")
                except ValueError:
                    console.print("[red]Enter a number[/]")

            state = game.apply_action(state, action)
        else:
            console.print("[cyan]AI thinking...[/]")
            root = mcts.search(state, diff_config.simulations)
            action = root.select_action(diff_config.temperature)
            state = game.apply_action(state, action)
            console.print(f"AI played: {action}\n")

        human_turn = not human_turn


@app.command()
def benchmark(
    game_name: str = typer.Argument("connect4", help="Game to benchmark"),
    simulations: int = typer.Option(100, "--sims", "-s"),
    games: int = typer.Option(5, "--games", "-n"),
    batch_size: int = typer.Option(8, "--batch-size", "-b"),
    device: str = typer.Option("auto", "--device", "-d"),
) -> None:
    """Benchmark MCTS performance."""
    import time
    import torch
    from .games import get_game
    from .net import create_model, get_model_config
    from .mcts import BatchedMCTS

    game = get_game(game_name)

    if device == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
        elif torch.backends.mps.is_available():
            dev = torch.device("mps")
        else:
            dev = torch.device("cpu")
    else:
        dev = torch.device(device)

    console.print(f"[blue]Benchmarking {game_name} on {dev}[/]")
    console.print(f"[blue]Simulations: {simulations}, Batch size: {batch_size}[/]")

    model_config = get_model_config(game_name)
    model = create_model(
        game.spec,
        num_channels=model_config["num_channels"],
        num_blocks=model_config["num_blocks"],
        device=dev,
    )
    model.eval()

    mcts = BatchedMCTS(
        game=game,
        model=model,
        device=dev,
        batch_size=batch_size,
        add_noise=True,
    )

    console.print(f"[cyan]Playing {games} games...[/]")

    start = time.time()
    total_moves = 0

    for i in range(games):
        state = game.initial_state()
        moves = 0

        while True:
            root = mcts.search(state, simulations)
            action = root.select_action(0.0)
            state = game.apply_action(state, action)
            moves += 1

            done, _ = game.is_terminal(state)
            if done:
                break

        total_moves += moves
        console.print(f"  Game {i+1}: {moves} moves")

    elapsed = time.time() - start

    console.print(f"\n[green]Results:[/]")
    console.print(f"  Total time: {elapsed:.2f}s")
    console.print(f"  Games/sec: {games/elapsed:.2f}")
    console.print(f"  Moves/sec: {total_moves/elapsed:.2f}")
    console.print(f"  Sims/sec: {total_moves*simulations/elapsed:.0f}")


if __name__ == "__main__":
    app()

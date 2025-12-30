"""
Command-line interface for Connect 4 AlphaZero.

Commands:
- train: Run the full training loop
- selfplay: Generate self-play games
- eval: Evaluate two models
- play: Play against the model
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import typer
from rich.console import Console

app = typer.Typer(
    name="c4z",
    help="Connect 4 AlphaZero - Train and play",
    no_args_is_help=True,
)

console = Console()


@app.command()
def train(
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to config YAML file"
    ),
    iterations: int = typer.Option(
        100, "--iterations", "-n", help="Number of training iterations"
    ),
    resume: Optional[Path] = typer.Option(
        None, "--resume", "-r", help="Resume from checkpoint"
    ),
) -> None:
    """Run the full AlphaZero training loop."""
    import torch
    from .utils import Config, get_device, set_seed, Logger, IterationMetrics, create_progress, print_config
    from .net import create_model, save_checkpoint, load_checkpoint
    from .net.train import Trainer
    from .mcts import create_evaluator
    from .selfplay import ReplayBuffer, SelfPlayWorker
    from .eval import Arena, should_accept

    # Load config
    if config_path and config_path.exists():
        config = Config.load(str(config_path))
    else:
        config = Config()

    config.num_iterations = iterations
    config.ensure_dirs()

    # Setup
    set_seed(config.seed)
    device = get_device(config.device)
    console.print(f"[blue]Using device: {device}[/]")

    print_config(config)

    # Initialize model
    if resume and resume.exists():
        model, checkpoint = load_checkpoint(str(resume), device)
        start_iter = checkpoint.get("iteration", 0) + 1
        console.print(f"[green]Resumed from iteration {start_iter - 1}[/]")
    else:
        model = create_model(
            num_channels=config.network.num_channels,
            num_blocks=config.network.num_blocks,
            device=device,
        )
        start_iter = 1

    best_model = create_model(
        num_channels=config.network.num_channels,
        num_blocks=config.network.num_blocks,
        device=device,
    )
    best_model.load_state_dict(model.state_dict())

    # Initialize components
    trainer = Trainer(
        model=model,
        device=device,
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
        value_loss_weight=config.train.value_loss_weight,
    )

    buffer = ReplayBuffer(capacity=config.buffer.capacity)
    logger = Logger(log_dir=config.log_dir)
    arena = Arena(
        num_simulations=config.eval.eval_simulations,
        c_puct=config.mcts.c_puct,
        device=device,
    )

    # Training loop
    for iteration in range(start_iter, config.num_iterations + 1):
        console.print(f"\n[bold blue]{'='*50}[/]")
        console.print(f"[bold blue]Iteration {iteration}/{config.num_iterations}[/]")
        console.print(f"[bold blue]{'='*50}[/]\n")

        # Self-play
        console.print("[cyan]Generating self-play games...[/]")
        model.eval()
        evaluate_fn = create_evaluator(model, device)

        worker = SelfPlayWorker(
            evaluate_fn=evaluate_fn,
            num_simulations=config.mcts.num_simulations,
            c_puct=config.mcts.c_puct,
            dirichlet_alpha=config.mcts.dirichlet_alpha,
            dirichlet_epsilon=config.mcts.dirichlet_epsilon,
            temp_threshold=config.mcts.temp_threshold,
        )

        games = []
        total_moves = 0
        with create_progress() as progress:
            task = progress.add_task("Self-play", total=config.selfplay.games_per_iter)
            for i in range(config.selfplay.games_per_iter):
                game = worker.play_game(augment=config.selfplay.augment_data)
                games.append(game)
                total_moves += game.num_moves
                progress.update(task, advance=1)

        # Add games to buffer
        samples_added = 0
        for game in games:
            buffer.add_game(game.states, game.policies, game.outcome)
            samples_added += len(game.states)

        avg_game_length = total_moves / len(games)
        console.print(f"[green]Generated {len(games)} games, {samples_added} samples[/]")
        console.print(f"[green]Average game length: {avg_game_length:.1f} moves[/]")

        # Training
        if len(buffer) >= config.buffer.min_samples:
            console.print("[cyan]Training...[/]")
            metrics = trainer.train_steps(
                buffer=buffer,
                num_steps=config.train.train_steps_per_iter,
                batch_size=config.train.batch_size,
            )
            console.print(
                f"[green]Loss: {metrics.total_loss:.4f} "
                f"(policy: {metrics.policy_loss:.4f}, value: {metrics.value_loss:.4f})[/]"
            )
        else:
            console.print(f"[yellow]Waiting for {config.buffer.min_samples} samples...[/]")
            metrics = None
            continue

        # Evaluation
        console.print("[cyan]Evaluating candidate vs best...[/]")
        eval_wins, eval_losses, eval_draws = 0, 0, 0
        with create_progress() as progress:
            task = progress.add_task("Arena [W:0 L:0 D:0]", total=config.eval.num_matches)

            def eval_callback(n, result_str):
                nonlocal eval_wins, eval_losses, eval_draws
                if result_str == "W":
                    eval_wins += 1
                elif result_str == "L":
                    eval_losses += 1
                else:
                    eval_draws += 1
                progress.update(
                    task,
                    advance=1,
                    description=f"Arena [W:{eval_wins} L:{eval_losses} D:{eval_draws}]"
                )

            result = arena.evaluate(
                candidate=model,
                best=best_model,
                num_games=config.eval.num_matches,
                progress_callback=eval_callback,
            )
        console.print(
            f"[green]Win: {result.wins}, Loss: {result.losses}, Draw: {result.draws} "
            f"(Score: {result.score*100:.1f}%)[/]"
        )

        accepted = should_accept(result, config.eval.accept_threshold)
        if accepted:
            console.print("[bold green]Model accepted! Updating best model.[/]")
            best_model.load_state_dict(model.state_dict())
            save_checkpoint(
                model=model,
                optimizer=trainer.optimizer,
                iteration=iteration,
                path=str(Path(config.checkpoint_dir) / "best.pt"),
            )
        else:
            console.print("[yellow]Model rejected. Keeping current best.[/]")

        # Save iteration checkpoint
        save_checkpoint(
            model=model,
            optimizer=trainer.optimizer,
            iteration=iteration,
            path=str(Path(config.checkpoint_dir) / f"iter_{iteration:04d}.pt"),
        )

        # Log metrics
        logger.log_iteration(IterationMetrics(
            iteration=iteration,
            games_played=len(games),
            samples_collected=samples_added,
            buffer_size=len(buffer),
            avg_game_length=avg_game_length,
            policy_loss=metrics.policy_loss if metrics else 0.0,
            value_loss=metrics.value_loss if metrics else 0.0,
            total_loss=metrics.total_loss if metrics else 0.0,
            policy_entropy=metrics.policy_entropy if metrics else 0.0,
            learning_rate=trainer.get_lr(),
            arena_win_rate=result.score,
            model_accepted=accepted,
        ))

    console.print("\n[bold green]Training complete![/]")


@app.command()
def selfplay(
    model_path: Path = typer.Option(
        ..., "--model", "-m", help="Path to model checkpoint"
    ),
    games: int = typer.Option(50, "--games", "-n", help="Number of games to generate"),
    output: Path = typer.Option(
        Path("data/selfplay.npz"), "--output", "-o", help="Output file"
    ),
    simulations: int = typer.Option(100, "--sims", "-s", help="MCTS simulations per move"),
) -> None:
    """Generate self-play games with a trained model."""
    import torch
    from .utils import get_device, create_progress
    from .net import load_checkpoint
    from .mcts import create_evaluator
    from .selfplay import ReplayBuffer, SelfPlayWorker

    device = get_device()
    console.print(f"[blue]Using device: {device}[/]")

    # Load model
    model, _ = load_checkpoint(str(model_path), device)
    model.eval()
    console.print(f"[green]Loaded model from {model_path}[/]")

    evaluate_fn = create_evaluator(model, device)
    worker = SelfPlayWorker(
        evaluate_fn=evaluate_fn,
        num_simulations=simulations,
    )

    # Generate games
    buffer = ReplayBuffer()
    total_moves = 0

    with create_progress() as progress:
        task = progress.add_task("Generating games", total=games)
        for i in range(games):
            game = worker.play_game()
            buffer.add_game(game.states, game.policies, game.outcome)
            total_moves += game.num_moves
            progress.update(task, advance=1)

    # Save
    output.parent.mkdir(parents=True, exist_ok=True)
    buffer.save(str(output))
    console.print(f"[green]Saved {len(buffer)} samples to {output}[/]")
    console.print(f"[green]Average game length: {total_moves / games:.1f}[/]")


@app.command("eval")
def evaluate(
    model_a: Path = typer.Option(..., "--a", help="First model checkpoint"),
    model_b: Path = typer.Option(..., "--b", help="Second model checkpoint"),
    games: int = typer.Option(100, "--games", "-n", help="Number of games"),
    simulations: int = typer.Option(100, "--sims", "-s", help="MCTS simulations"),
) -> None:
    """Evaluate two models against each other."""
    import torch
    from .utils import get_device, create_progress
    from .net import load_checkpoint
    from .eval import Arena

    device = get_device()

    model1, _ = load_checkpoint(str(model_a), device)
    model2, _ = load_checkpoint(str(model_b), device)
    console.print(f"[green]Loaded models[/]")

    arena = Arena(num_simulations=simulations, device=device)

    console.print("[cyan]Running evaluation...[/]")
    with create_progress() as progress:
        task = progress.add_task("Games", total=games)

        def callback(n, result):
            progress.update(task, advance=1)

        result = arena.evaluate(model1, model2, games, progress_callback=callback)

    console.print(f"\n[bold]Results (Model A perspective):[/]")
    console.print(f"  Wins:   {result.wins}")
    console.print(f"  Losses: {result.losses}")
    console.print(f"  Draws:  {result.draws}")
    console.print(f"  Score:  {result.score*100:.1f}%")


@app.command()
def play(
    model_path: Optional[Path] = typer.Option(
        None, "--model", "-m", help="Model checkpoint (uses random if not specified)"
    ),
    simulations: int = typer.Option(100, "--sims", "-s", help="MCTS simulations"),
    human_first: bool = typer.Option(True, "--first/--second", help="Human plays first"),
) -> None:
    """Play against the model in the terminal."""
    import torch
    from .utils import get_device
    from .game import initial_state, apply_move, is_terminal, legal_moves, render, COLS
    from .mcts import MCTS, create_evaluator, create_random_evaluator
    from .net import load_checkpoint

    device = get_device()

    # Load model or use random
    if model_path and model_path.exists():
        model, _ = load_checkpoint(str(model_path), device)
        model.eval()
        evaluate_fn = create_evaluator(model, device)
        console.print(f"[green]Playing against trained model[/]")
    else:
        evaluate_fn = create_random_evaluator()
        console.print(f"[yellow]Playing against random player[/]")

    mcts = MCTS(
        evaluate_fn=evaluate_fn,
        add_noise=False,  # No exploration noise for playing
    )

    state = initial_state()
    human_turn = human_first
    move_num = 0

    console.print("\n[bold]Connect 4[/]")
    console.print("You are X, AI is O")
    console.print("Enter column number (0-6) to play\n")

    while True:
        console.print(render(state))

        done, value = is_terminal(state)
        if done:
            if value > 0:
                if human_turn:
                    console.print("[red]AI wins![/]")
                else:
                    console.print("[green]You win![/]")
            elif value < 0:
                if human_turn:
                    console.print("[green]You win![/]")
                else:
                    console.print("[red]AI wins![/]")
            else:
                console.print("[yellow]Draw![/]")
            break

        if human_turn:
            # Human move
            legal = legal_moves(state)
            while True:
                try:
                    col = int(typer.prompt("Your move (0-6)"))
                    if 0 <= col < COLS and legal[col]:
                        break
                    console.print("[red]Invalid move, try again[/]")
                except ValueError:
                    console.print("[red]Enter a number 0-6[/]")

            state = apply_move(state, col)
            console.print(f"You played column {col}\n")
        else:
            # AI move
            console.print("[cyan]AI thinking...[/]")
            root = mcts.search(state, simulations)
            action = root.select_action(temperature=0.0)
            state = apply_move(state, action)
            console.print(f"AI played column {action}\n")

        human_turn = not human_turn
        move_num += 1


@app.command()
def benchmark(
    simulations: int = typer.Option(100, "--sims", "-s", help="MCTS simulations"),
    games: int = typer.Option(10, "--games", "-n", help="Games to play"),
) -> None:
    """Benchmark MCTS performance."""
    import time
    import torch
    from .utils import get_device
    from .net import create_model
    from .mcts import create_evaluator
    from .selfplay import SelfPlayWorker

    device = get_device()
    console.print(f"[blue]Using device: {device}[/]")

    model = create_model(device=device)
    model.eval()
    evaluate_fn = create_evaluator(model, device)

    worker = SelfPlayWorker(
        evaluate_fn=evaluate_fn,
        num_simulations=simulations,
    )

    console.print(f"[cyan]Running {games} games with {simulations} simulations...[/]")

    start = time.time()
    total_moves = 0
    for i in range(games):
        game = worker.play_game(augment=False)
        total_moves += game.num_moves
        console.print(f"Game {i+1}: {game.num_moves} moves")

    elapsed = time.time() - start
    console.print(f"\n[green]Total time: {elapsed:.2f}s[/]")
    console.print(f"[green]Games/sec: {games/elapsed:.2f}[/]")
    console.print(f"[green]Moves/sec: {total_moves/elapsed:.2f}[/]")
    console.print(f"[green]Sims/sec: {total_moves*simulations/elapsed:.0f}[/]")


if __name__ == "__main__":
    app()

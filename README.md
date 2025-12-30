# AlphaZero Arcade

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mindswim/connect4-zero/blob/main/notebooks/train_colab.ipynb)

Train superhuman AI for board games using AlphaZero. Supports multiple games with GPU-optimized training.

## What is AlphaZero?

AlphaZero is a reinforcement learning algorithm that masters games through self-play alone. It combines:

1. **Neural Network**: Predicts move probabilities (policy) and win likelihood (value)
2. **Monte Carlo Tree Search (MCTS)**: Uses the network to guide game tree exploration
3. **Self-Play**: Generates training data by playing against itself
4. **Model Evaluation**: Only accepts improved models through head-to-head matches

```
┌─────────────────────────────────────────────────────────────────┐
│                      Training Loop                               │
│                                                                  │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌────────┐ │
│  │ Self-Play│ ──> │  Buffer  │ ──> │ Training │ ──> │  Eval  │ │
│  │   MCTS   │     │ (states, │     │  Update  │     │ Arena  │ │
│  │ + Network│     │  policy, │     │  Network │     │        │ │
│  └──────────┘     │  value)  │     └──────────┘     └────────┘ │
│       ^           └──────────┘                           │      │
│       │                                                  │      │
│       └──────────── if improved ────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

## Requirements

- Python 3.11+
- PyTorch with MPS support (Apple Silicon)
- macOS 13+ recommended for M-series chips

## Installation

```bash
# Clone the repo
cd connect4-zero

# Install with pip
pip install -e .

# Or with development dependencies
pip install -e ".[dev]"
```

## Quick Start

### Train a model

```bash
# Run training (uses defaults optimized for M3)
c4z train --iterations 100

# Resume from checkpoint
c4z train --resume checkpoints/best.pt --iterations 200

# Use custom config
c4z train --config configs/v1.yaml
```

### Play against your model

```bash
# Play against trained model
c4z play --model checkpoints/best.pt

# Play against random (for testing)
c4z play
```

### Other commands

```bash
# Generate self-play games
c4z selfplay --model checkpoints/best.pt --games 50 --output data/games.npz

# Evaluate two models
c4z eval --a checkpoints/iter_0010.pt --b checkpoints/iter_0005.pt --games 100

# Benchmark performance
c4z benchmark --sims 100 --games 10
```

## Project Structure

```
connect4-zero/
  src/connect4zero/
    game/           # Connect 4 rules and encoding
    mcts/           # Monte Carlo Tree Search with PUCT
    net/            # Neural network and training
    selfplay/       # Self-play game generation
    eval/           # Model evaluation arena
    utils/          # Config, logging, device selection
    cli.py          # Command-line interface
  tests/            # Unit tests
  docs/             # Architecture documentation
```

## Configuration

Default config is optimized for Apple M3. Key parameters:

```yaml
mcts:
  num_simulations: 100    # MCTS simulations per move
  c_puct: 1.5             # Exploration constant
  dirichlet_alpha: 0.3    # Root noise for exploration
  dirichlet_epsilon: 0.25

network:
  num_channels: 64        # CNN channel width
  num_blocks: 5           # Residual blocks

train:
  batch_size: 256
  lr: 0.001
  train_steps_per_iter: 300

selfplay:
  games_per_iter: 25

eval:
  num_matches: 100
  accept_threshold: 0.55  # Win rate to accept new model
```

## How It Works

### Training Loop (Each Iteration)

```
┌─────────────────────────────────────────────────────────────┐
│  1. SELF-PLAY                                               │
│     Model plays N games against itself using MCTS           │
│     Stores: (board position, MCTS policy, game outcome)     │
│                                                             │
│  2. TRAIN                                                   │
│     Neural net learns to:                                   │
│     - Predict MCTS move probabilities (policy loss)         │
│     - Predict game winner from position (value loss)        │
│                                                             │
│  3. ARENA                                                   │
│     New model vs current best, head-to-head                 │
│     If win rate > 55%: new model becomes best               │
│     Otherwise: keep training, try again next iteration      │
└─────────────────────────────────────────────────────────────┘
```

**Why this works:**
- **Self-play**: MCTS explores moves even when the neural net is bad. Visit counts become training targets.
- **Training**: Net learns to mimic MCTS (skip slow search) and predict outcomes (better MCTS evaluation).
- **Arena**: Gatekeeping prevents regression - only keep models that actually improve.

### 1. State Representation

The board is stored in **canonical form** from the current player's perspective:
- `+1` = current player's pieces
- `-1` = opponent's pieces
- Board is flipped after each move

This lets the network learn a single policy for "me vs them".

### 2. Neural Network

Small CNN with residual blocks:
- Input: 2 channels (my pieces, their pieces)
- Trunk: 5 residual blocks, 64 channels
- Policy head: outputs 7 logits (one per column)
- Value head: outputs scalar in [-1, 1]

### 3. MCTS with PUCT

Selection uses the PUCT formula:

```
U(a) = Q(a) + c_puct * P(a) * sqrt(N_total) / (1 + N(a))
```

Where:
- `Q(a)` = average value of action a
- `P(a)` = prior probability from network
- `N(a)` = visit count for action a

### 4. Training Target

From each self-play game, we collect:
- `state`: board position
- `policy`: MCTS visit distribution
- `value`: final game outcome (from that player's perspective)

## Training Tips

1. **Start small**: 25 games per iteration is enough initially
2. **Watch loss curves**: Policy loss should decrease; value loss may fluctuate
3. **Check arena results**: Model should beat random easily after ~100 games
4. **Use temperature**: Early moves use temp=1.0 for exploration, later use temp=0

## Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_game_rules.py

# With coverage
pytest --cov=connect4zero
```

## Web Deployment

Export your trained model to play in the browser:

```bash
# Install export dependencies
pip install -e ".[export]"

# Export model to ONNX
c4z export --model checkpoints/best.pt --output web/public

# Start the web app
cd web
npm install
npm run dev
```

The web app uses ONNX Runtime Web to run the neural network directly in the browser with MCTS.

## Supported Games

| Game | Status | Training Time (A100) |
|------|--------|---------------------|
| Tic-Tac-Toe | Ready | ~2 min |
| Connect 4 | Ready | ~30 min |
| Othello | Coming soon | ~2 hrs |
| Chess | Planned | Days |

## Roadmap

- [x] Batched MCTS for GPU efficiency
- [x] Generic game interface (any game)
- [x] Difficulty system (Easy to Impossible)
- [x] ONNX export for deployment
- [x] Web UI with difficulty slider
- [ ] Othello game
- [ ] Parallel self-play

## License

MIT

## References

- [Mastering Chess and Shogi by Self-Play](https://arxiv.org/abs/1712.01815) - Original AlphaZero paper
- [A Simple Alpha(Go) Zero Tutorial](https://web.stanford.edu/~surag/posts/alphazero.html) - Clear explanation

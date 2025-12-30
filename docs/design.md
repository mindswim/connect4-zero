# Connect4-Zero: Architecture & Design

A deep dive into the AlphaZero algorithm as implemented for Connect 4.

## Table of Contents

1. [Canonical Form](#canonical-form)
2. [MCTS with PUCT](#mcts-with-puct)
3. [Neural Network Architecture](#neural-network-architecture)
4. [Training Data Pipeline](#training-data-pipeline)
5. [Model Evaluation](#model-evaluation)

---

## Canonical Form

### The Problem

In Connect 4, two players alternate turns. A naive approach stores the board with player 1 as `1` and player 2 as `2`. This creates a problem: the network must learn separate strategies for both perspectives.

### The Solution

**Canonical form**: Always store the board from the current player's perspective.

- `+1` = my pieces (current player to move)
- `-1` = opponent's pieces
- `0` = empty

After each move, multiply the board by `-1` to flip perspectives.

```python
def apply_move(state, action):
    new_board = state.board.copy()
    # Place piece (+1 for current player)
    new_board[row, action] = 1
    # Flip for next player
    new_board = -new_board
    return GameState(board=new_board)
```

### Benefits

1. Network learns one unified policy for "me"
2. Half the effective training data needed
3. Simpler architecture (no player embedding)
4. Symmetric augmentation works directly

---

## MCTS with PUCT

### Why MCTS?

Pure network inference is fast but shallow. MCTS adds depth by simulating games and backing up results. The network guides which moves to explore.

### The PUCT Formula

At each node, select action `a` maximizing:

```
U(a) = Q(a) + c_puct * P(a) * sqrt(sum(N)) / (1 + N(a))
```

Where:
- `Q(a)` = average backed-up value for action a
- `P(a)` = prior probability from policy network
- `N(a)` = visit count for action a
- `c_puct` = exploration constant (1.5 recommended)

### Intuition

The formula balances:
- **Exploitation**: High Q(a) means action looks good
- **Exploration**: Low N(a) means under-explored
- **Prior**: High P(a) means network likes this move

Early in search, priors dominate. As visits accumulate, Q-values take over.

### Search Procedure

```
1. SELECT: From root, pick actions via PUCT until reaching leaf
2. EXPAND: Evaluate leaf with network, store priors
3. BACKUP: Propagate value up, flipping sign each level
4. REPEAT: Run many simulations
5. SELECT MOVE: Pick action proportional to visit counts
```

### Temperature

Visit counts become the training target policy. Temperature controls exploration:

- `tau=1.0`: Policy proportional to visits (more exploration)
- `tau=0.0`: Argmax selection (exploitation)

Typical schedule: `tau=1.0` for first 10 moves, then `tau=0.0`.

### Dirichlet Noise

At the root only, mix noise into priors:

```
P' = (1 - epsilon) * P + epsilon * Dir(alpha)
```

- `epsilon=0.25`: Noise weight
- `alpha=0.3`: Dirichlet concentration (lower = more extreme)

This ensures exploration of unlikely moves at the root, preventing premature convergence.

---

## Neural Network Architecture

### Input Encoding

Shape: `(2, 6, 7)` - two channels for 6x7 board

- Channel 0: `(board == +1)` current player's pieces
- Channel 1: `(board == -1)` opponent's pieces

### Architecture

```
Input (2, 6, 7)
    |
ConvBlock(2 -> 64)
    |
ResBlock(64) x 5
    |
    +---> PolicyHead -> 7 logits
    |
    +---> ValueHead -> scalar [-1, 1]
```

**ResBlock:**
```
input --> Conv3x3 --> ReLU --> Conv3x3 --> Add --> ReLU
  |                                          ^
  +------------------------------------------+
```

**PolicyHead:**
```
Conv1x1(64->32) -> ReLU -> Flatten -> Linear(32*6*7 -> 7)
```

**ValueHead:**
```
Conv1x1(64->32) -> ReLU -> Flatten -> Linear(32*6*7 -> 64) -> ReLU -> Linear(64 -> 1) -> tanh
```

### Why This Architecture?

- **Small CNN**: Connect 4 is simpler than chess; 5 blocks suffice
- **No BatchNorm**: Keeps training stable, simpler code
- **Residual connections**: Enable deeper networks without degradation
- **Separate heads**: Policy and value have different objectives

---

## Training Data Pipeline

### Self-Play Game Recording

For each move in a game, store:
1. `state`: encoded board position
2. `pi`: MCTS policy (visit distribution)
3. (later) `z`: game outcome from this position

### Assigning Values

After game ends with outcome `z_final` (from player 0's perspective):

```python
for i, (state, policy) in enumerate(game_history):
    if i % 2 == 0:  # Player 0's turn
        value = z_final
    else:           # Player 1's turn
        value = -z_final
    add_sample(state, policy, value)
```

### Data Augmentation

Connect 4 has horizontal symmetry. For each position:
- Original: `(state, pi)`
- Flipped: `(flip_lr(state), flip(pi))`

Doubles training data for free.

### Replay Buffer

Ring buffer of recent samples (200k default):
- Uniform random sampling
- Old samples discarded as new ones arrive
- Prevents overfitting to recent games

### Training Loss

```
L = L_policy + L_value + L_regularization

L_policy = -sum(pi * log(p))     # Cross-entropy
L_value = (z - v)^2               # MSE
L_reg = weight_decay * ||theta||^2
```

---

## Model Evaluation

### The Gating Problem

Training continuously updates the network. But not all updates improve play. We need a way to:
1. Test if new model is actually better
2. Only keep improvements

### Arena Matches

Pit candidate model against current best:
- Play N games (100-200)
- Alternate who goes first
- Use greedy play (no temperature, no noise)
- Use same MCTS simulations as training

### Acceptance Threshold

Accept candidate if win rate >= 55%

Why 55% instead of 50%?
- Accounts for variance in match outcomes
- Prevents accepting noise as improvement
- Conservative update strategy

### After Acceptance

- Copy candidate weights to best model
- Save checkpoint
- Continue training from candidate

### After Rejection

- Keep current best model
- Continue training candidate
- It may improve next iteration

---

## Implementation Notes

### MPS (Apple Silicon)

PyTorch MPS backend enables GPU acceleration on M-series Macs:

```python
device = torch.device("mps")
model = model.to(device)
```

Tips:
- Batch size 256 works well
- Float32 is more stable than float16
- MPS is 5-10x faster than CPU for inference

### Memory Management

- MCTS creates many nodes; use lazy child creation
- Clear search tree between games
- Replay buffer caps memory usage

### Numerical Stability

- Use log-softmax for policy loss
- Clamp values to prevent log(0)
- Mask illegal moves with -inf before softmax

---

## Common Pitfalls

1. **Forgetting to flip board after moves**: Breaks canonical form
2. **Wrong backup direction**: Value flips each ply
3. **Not masking illegal moves**: Network will learn illegal moves
4. **Temperature too high late game**: Weakens play
5. **Too few MCTS simulations**: Network never gets corrected
6. **Accepting at 50% threshold**: Noise causes random walk

---

## References

1. Silver, D., et al. "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm." arXiv:1712.01815 (2017).

2. Silver, D., et al. "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play." Science 362.6419 (2018): 1140-1144.

3. Coulom, R. "Efficient selectivity and backup operators in Monte-Carlo tree search." CG 2006.

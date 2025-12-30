# MCTS Mathematics

The mathematical foundations of Monte Carlo Tree Search with PUCT.

## Core Algorithm

MCTS builds a search tree incrementally through repeated simulations.

### Tree Statistics

Each node stores per-action statistics:

| Symbol | Meaning |
|--------|---------|
| N(a) | Visit count for action a |
| W(a) | Total backed-up value for action a |
| Q(a) | Mean action value = W(a) / N(a) |
| P(a) | Prior probability from neural network |

## PUCT Selection

### Formula

Select action `a*` at node `s` by maximizing:

```
a* = argmax_a [ Q(s,a) + U(s,a) ]
```

Where the exploration bonus is:

```
U(s,a) = c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
```

And `N(s) = sum_b N(s,b)` is total visits to node s.

### Parameters

| Parameter | Recommended | Effect |
|-----------|-------------|--------|
| c_puct | 1.5 | Higher = more exploration |

### Derivation Intuition

PUCT comes from the UCB1 formula in bandit theory:

```
UCB1: Q(a) + c * sqrt(log(N) / N(a))
```

AlphaZero modifies this to:
1. Replace `sqrt(log(N))` with `sqrt(N)` for simplicity
2. Weight by prior P(a) to incorporate network knowledge
3. Add 1 to denominator for numerical stability when N(a)=0

## Value Backup

### Sign Flipping

When backing up value v from a leaf:

```
At depth d from root:
  parent receives: v * (-1)^d
```

Because:
- Positive value = good for current player at leaf
- Parent is opponent's turn, so value is negated
- Alternates up the tree

### Update Rule

After simulation returns value v to node s via action a:

```
N(s,a) <- N(s,a) + 1
W(s,a) <- W(s,a) + v
Q(s,a) <- W(s,a) / N(s,a)
```

## Policy Extraction

### Temperature Scaling

After search, extract policy from visit counts:

```
pi(a) = N(a)^(1/tau) / sum_b N(b)^(1/tau)
```

| tau | Effect |
|-----|--------|
| 1.0 | Proportional to visits |
| 0.1 | Sharper, favors most visited |
| 0 | Argmax (deterministic) |

### Practical Schedule

```
if move_number < 10:
    tau = 1.0  # Exploration
else:
    tau = 0.0  # Exploitation
```

## Dirichlet Noise

### Purpose

Ensure exploration at the root, even when priors are confident.

### Formula

```
P'(a) = (1 - epsilon) * P(a) + epsilon * Dir(alpha)_a
```

Where `Dir(alpha)` is a sample from the Dirichlet distribution.

### Parameters for Connect 4

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| epsilon | 0.25 | 25% noise, 75% prior |
| alpha | 0.3 | Low = more extreme samples |

### Dirichlet Distribution

The Dirichlet with parameters `(alpha, alpha, ..., alpha)` produces random probability vectors:

- `alpha > 1`: Vectors cluster near uniform
- `alpha = 1`: Uniform over simplex
- `alpha < 1`: Vectors cluster near corners (sparse)

With `alpha = 0.3`, noise often emphasizes one or two actions strongly.

## Simulation Complexity

### Time per Simulation

```
T_sim = T_select + T_expand + T_backup
      ~ O(depth) + T_neural + O(depth)
      ~ O(depth) + O(1)  [with batched inference]
```

### Total Search Time

```
T_search = num_simulations * T_sim
```

For Connect 4 with 100 sims:
- Depth ~ 10-20
- T_neural dominates
- ~10-50ms per move on M3

## Convergence Properties

### Asymptotic Optimality

As simulations approach infinity:
- Q-values converge to minimax values
- Visit distribution concentrates on optimal action

### Practical Consideration

With finite simulations:
- Network priors provide strong initialization
- MCTS corrects network errors through search
- More sims = stronger play but slower

## Virtual Loss (Not Implemented in v1)

For parallel MCTS, temporarily reduce Q during selection:

```
Q_virtual(a) = (W(a) - n_inflight) / (N(a) + n_inflight)
```

Prevents threads from all exploring the same path.

## References

1. Rosin, C.D. "Multi-armed bandits with episode context." Annals of Mathematics and AI (2011).

2. Kocsis, L., Szepesvari, C. "Bandit based Monte-Carlo Planning." ECML (2006).

3. Silver, D., et al. "Mastering the game of Go without human knowledge." Nature (2017).

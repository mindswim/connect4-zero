"""Tests for MCTS invariants."""

import numpy as np
import pytest

from connect4zero.game import initial_state, apply_move, legal_moves, COLS
from connect4zero.mcts import MCTS, Node, create_random_evaluator


class TestNode:
    def test_initial_state(self):
        state = initial_state()
        node = Node(state=state)

        assert node.N.shape == (COLS,)
        assert node.W.shape == (COLS,)
        assert node.P.shape == (COLS,)
        assert np.all(node.N == 0)
        assert np.all(node.W == 0)
        assert not node.is_expanded

    def test_q_values(self):
        state = initial_state()
        node = Node(state=state)

        # Manually set some values
        node.N[0] = 10
        node.W[0] = 5

        q = node.Q
        assert q[0] == 0.5  # 5/10

    def test_total_visits(self):
        state = initial_state()
        node = Node(state=state)

        node.N[0] = 10
        node.N[3] = 20
        node.N[6] = 5

        assert node.total_visits == 35

    def test_get_policy_temperature_zero(self):
        state = initial_state()
        node = Node(state=state)

        node.N[3] = 100
        node.N[0] = 10

        policy = node.get_policy(temperature=0.0)

        # Should be argmax
        assert policy[3] == 1.0
        assert np.sum(policy) == 1.0

    def test_get_policy_temperature_one(self):
        state = initial_state()
        node = Node(state=state)

        node.N[0] = 10
        node.N[1] = 20
        node.N[2] = 30

        policy = node.get_policy(temperature=1.0)

        # Should be proportional to visits
        assert policy[2] > policy[1] > policy[0]
        assert np.isclose(np.sum(policy), 1.0)


class TestMCTS:
    def test_search_basic(self):
        state = initial_state()
        evaluate_fn = create_random_evaluator()

        mcts = MCTS(
            evaluate_fn=evaluate_fn,
            add_noise=False,
        )

        root = mcts.search(state, num_simulations=50)

        # Root should be expanded
        assert root.is_expanded

        # Total visits should be close to num_simulations
        # (may be slightly less due to terminal states)
        assert root.total_visits >= 40

    def test_illegal_moves_not_visited(self):
        state = initial_state()
        # Fill column 0
        for _ in range(6):
            state = apply_move(state, 0)

        evaluate_fn = create_random_evaluator()
        mcts = MCTS(evaluate_fn=evaluate_fn, add_noise=False)

        root = mcts.search(state, num_simulations=50)

        # Column 0 is full, should have 0 visits
        assert root.N[0] == 0

    def test_policy_sums_to_one(self):
        state = initial_state()
        evaluate_fn = create_random_evaluator()

        mcts = MCTS(evaluate_fn=evaluate_fn, add_noise=False)
        root = mcts.search(state, num_simulations=50)

        policy = root.get_policy(temperature=1.0)
        assert np.isclose(np.sum(policy), 1.0)

    def test_policy_respects_legality(self):
        state = initial_state()
        # Fill column 3
        for _ in range(6):
            state = apply_move(state, 3)

        evaluate_fn = create_random_evaluator()
        mcts = MCTS(evaluate_fn=evaluate_fn, add_noise=False)

        root = mcts.search(state, num_simulations=50)
        policy = root.get_policy(temperature=1.0)

        # Illegal move should have 0 probability
        assert policy[3] == 0.0

    def test_dirichlet_noise(self):
        state = initial_state()
        evaluate_fn = create_random_evaluator()

        mcts = MCTS(
            evaluate_fn=evaluate_fn,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
            add_noise=True,
        )

        root = mcts.search(state, num_simulations=10)

        # Just verify it doesn't crash and priors sum to 1
        assert np.isclose(np.sum(root.P), 1.0)

    def test_deterministic_without_noise(self):
        """Same position should give consistent results without noise."""
        state = initial_state()
        evaluate_fn = create_random_evaluator()

        np.random.seed(42)
        mcts = MCTS(evaluate_fn=evaluate_fn, add_noise=False)
        root1 = mcts.search(state, num_simulations=100)
        action1 = root1.select_action(temperature=0.0)

        np.random.seed(42)
        mcts = MCTS(evaluate_fn=evaluate_fn, add_noise=False)
        root2 = mcts.search(state, num_simulations=100)
        action2 = root2.select_action(temperature=0.0)

        # With same seed, should get same result
        assert action1 == action2


class TestMCTSEdgeCases:
    def test_near_terminal(self):
        """Test MCTS near a terminal state."""
        state = initial_state()
        # Play a few moves
        state = apply_move(state, 0)
        state = apply_move(state, 1)
        state = apply_move(state, 0)
        state = apply_move(state, 1)
        state = apply_move(state, 0)

        # Now if current player plays 0, they might win
        evaluate_fn = create_random_evaluator()
        mcts = MCTS(evaluate_fn=evaluate_fn, add_noise=False)

        root = mcts.search(state, num_simulations=50)
        policy = root.get_policy(temperature=1.0)

        # Should still produce valid policy
        assert np.isclose(np.sum(policy), 1.0)

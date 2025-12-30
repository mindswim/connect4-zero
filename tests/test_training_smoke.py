"""Smoke tests for training pipeline."""

import numpy as np
import torch
import pytest
import tempfile
from pathlib import Path

from connect4zero.game import ROWS, COLS, NUM_CHANNELS
from connect4zero.net import create_model, save_checkpoint, load_checkpoint
from connect4zero.net.train import Trainer
from connect4zero.selfplay import ReplayBuffer, play_random_game
from connect4zero.mcts import create_evaluator


class TestModel:
    def test_forward_pass(self):
        model = create_model(num_channels=32, num_blocks=2)
        x = torch.randn(4, NUM_CHANNELS, ROWS, COLS)

        policy, value = model(x)

        assert policy.shape == (4, 7)
        assert value.shape == (4, 1)

    def test_predict_with_mask(self):
        model = create_model(num_channels=32, num_blocks=2)
        x = torch.randn(2, NUM_CHANNELS, ROWS, COLS)
        mask = torch.tensor([[True, True, True, True, True, True, True],
                            [True, False, True, False, True, False, True]])

        policy, value = model.predict(x, mask)

        # Policies should sum to 1
        assert torch.allclose(policy.sum(dim=1), torch.ones(2))

        # Masked actions should have 0 probability
        assert policy[1, 1] == 0.0
        assert policy[1, 3] == 0.0
        assert policy[1, 5] == 0.0


class TestCheckpoint:
    def test_save_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"

            model = create_model(num_channels=32, num_blocks=2)
            optimizer = torch.optim.Adam(model.parameters())

            save_checkpoint(model, optimizer, iteration=5, path=str(path))

            loaded_model, checkpoint = load_checkpoint(str(path))

            assert checkpoint["iteration"] == 5
            assert checkpoint["num_channels"] == 32
            assert checkpoint["num_blocks"] == 2

            # Weights should match
            for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
                assert torch.allclose(p1, p2)


class TestTrainer:
    def test_train_batch(self):
        model = create_model(num_channels=32, num_blocks=2)
        trainer = Trainer(model, device=torch.device("cpu"))

        # Create dummy batch
        states = np.random.randn(8, NUM_CHANNELS, ROWS, COLS).astype(np.float32)
        policies = np.random.dirichlet([1] * 7, size=8).astype(np.float32)
        values = np.random.uniform(-1, 1, size=8).astype(np.float32)

        metrics = trainer.train_batch(states, policies, values)

        assert metrics.policy_loss > 0
        assert metrics.value_loss >= 0
        assert metrics.total_loss > 0

    def test_loss_decreases(self):
        model = create_model(num_channels=32, num_blocks=2)
        trainer = Trainer(model, device=torch.device("cpu"), lr=1e-2)

        # Create fixed batch
        np.random.seed(42)
        states = np.random.randn(16, NUM_CHANNELS, ROWS, COLS).astype(np.float32)
        policies = np.random.dirichlet([1] * 7, size=16).astype(np.float32)
        values = np.random.uniform(-1, 1, size=16).astype(np.float32)

        # Train multiple times on same batch
        losses = []
        for _ in range(10):
            metrics = trainer.train_batch(states, policies, values)
            losses.append(metrics.total_loss)

        # Loss should generally decrease
        assert losses[-1] < losses[0]


class TestReplayBuffer:
    def test_add_and_sample(self):
        buffer = ReplayBuffer(capacity=1000)

        # Add some samples
        for _ in range(100):
            state = np.random.randn(NUM_CHANNELS, ROWS, COLS).astype(np.float32)
            policy = np.random.dirichlet([1] * 7).astype(np.float32)
            value = np.random.uniform(-1, 1)
            buffer.add(state, policy, value)

        assert len(buffer) == 100

        # Sample batch
        states, policies, values = buffer.sample_batch(32)

        assert states.shape == (32, NUM_CHANNELS, ROWS, COLS)
        assert policies.shape == (32, 7)
        assert values.shape == (32,)

    def test_capacity_limit(self):
        buffer = ReplayBuffer(capacity=50)

        for _ in range(100):
            state = np.random.randn(NUM_CHANNELS, ROWS, COLS).astype(np.float32)
            policy = np.random.dirichlet([1] * 7).astype(np.float32)
            buffer.add(state, policy, 0.0)

        assert len(buffer) == 50  # Capped at capacity

    def test_save_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "buffer.npz"

            buffer = ReplayBuffer()
            for _ in range(50):
                state = np.random.randn(NUM_CHANNELS, ROWS, COLS).astype(np.float32)
                policy = np.random.dirichlet([1] * 7).astype(np.float32)
                buffer.add(state, policy, 0.5)

            buffer.save(str(path))

            buffer2 = ReplayBuffer()
            buffer2.load(str(path))

            assert len(buffer2) == 50


class TestSelfPlay:
    def test_random_game(self):
        game = play_random_game()

        assert len(game.states) > 0
        assert len(game.policies) == len(game.states)
        assert len(game.moves) == game.num_moves
        assert game.outcome in [-1.0, 0.0, 1.0]

    def test_game_terminates(self):
        """Games should always terminate (no infinite loops)."""
        for _ in range(10):
            game = play_random_game()
            assert game.num_moves <= 42  # Max moves in Connect 4


class TestIntegration:
    def test_training_iteration(self):
        """Test one complete training iteration."""
        model = create_model(num_channels=32, num_blocks=2)
        trainer = Trainer(model, device=torch.device("cpu"))
        buffer = ReplayBuffer()

        # Generate some games
        for _ in range(5):
            game = play_random_game()
            buffer.add_game(game.states, game.policies, game.outcome)

        assert len(buffer) > 0

        # Train
        initial_loss = None
        for _ in range(5):
            states, policies, values = buffer.sample_batch(16)
            metrics = trainer.train_batch(states, policies, values)
            if initial_loss is None:
                initial_loss = metrics.total_loss

        # Should complete without errors
        assert metrics.total_loss > 0

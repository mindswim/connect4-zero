"""Tests for state encoding."""

import numpy as np
import torch
import pytest

from connect4zero.game import (
    ROWS,
    COLS,
    NUM_CHANNELS,
    GameState,
    initial_state,
    apply_move,
    encode_state,
    encode_state_torch,
    get_action_mask,
    mask_illegal_logits,
    decode_policy,
)


class TestEncodeState:
    def test_shape(self):
        state = initial_state()
        encoded = encode_state(state)
        assert encoded.shape == (NUM_CHANNELS, ROWS, COLS)
        assert encoded.dtype == np.float32

    def test_empty_board(self):
        state = initial_state()
        encoded = encode_state(state)
        # All zeros for empty board
        assert np.all(encoded == 0)

    def test_player_channels(self):
        state = initial_state()
        state = apply_move(state, 3)  # Player 1 moves

        encoded = encode_state(state)
        # After canonicalization, the piece is -1 (opponent)
        # So channel 1 (opponent) should have the piece
        assert encoded[1, ROWS - 1, 3] == 1.0
        assert encoded[0, ROWS - 1, 3] == 0.0

    def test_channels_disjoint(self):
        """Current player and opponent channels should never overlap."""
        state = initial_state()
        for _ in range(10):
            legal = get_action_mask(state)
            legal_cols = np.where(legal)[0]
            if len(legal_cols) == 0:
                break
            action = np.random.choice(legal_cols)
            state = apply_move(state, action)

        encoded = encode_state(state)
        # No position should have both channels active
        overlap = encoded[0] * encoded[1]
        assert np.all(overlap == 0)


class TestEncodeTorch:
    def test_shape(self):
        state = initial_state()
        tensor = encode_state_torch(state)
        assert tensor.shape == (1, NUM_CHANNELS, ROWS, COLS)
        assert tensor.dtype == torch.float32

    def test_device(self):
        state = initial_state()
        tensor = encode_state_torch(state, device=torch.device("cpu"))
        assert tensor.device.type == "cpu"


class TestActionMask:
    def test_all_legal_initially(self):
        state = initial_state()
        mask = get_action_mask(state)
        assert mask.shape == (COLS,)
        assert np.all(mask)

    def test_full_column_illegal(self):
        state = initial_state()
        # Fill column 0
        for _ in range(ROWS):
            state = apply_move(state, 0)

        mask = get_action_mask(state)
        assert not mask[0]  # Column 0 is full
        assert mask[1]  # Other columns still legal


class TestMaskIllegalLogits:
    def test_masks_correctly(self):
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])
        mask = torch.tensor([[True, False, True, False, True, False, True]])

        masked = mask_illegal_logits(logits, mask)

        assert masked[0, 0] == 1.0  # Legal, unchanged
        assert masked[0, 1] == float("-inf")  # Illegal
        assert masked[0, 2] == 3.0
        assert masked[0, 3] == float("-inf")

    def test_decode_policy(self):
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])
        mask = torch.tensor([[True, False, True, False, True, False, True]])

        policy = decode_policy(logits, mask)

        # Should sum to 1
        assert torch.isclose(policy.sum(), torch.tensor(1.0))

        # Illegal moves should have 0 probability
        assert policy[0, 1] == 0.0
        assert policy[0, 3] == 0.0
        assert policy[0, 5] == 0.0

        # Legal moves should have positive probability
        assert policy[0, 0] > 0
        assert policy[0, 2] > 0
        assert policy[0, 6] > 0

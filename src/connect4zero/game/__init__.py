"""Game module - Connect 4 rules and encoding."""

from .connect4 import (
    ROWS,
    COLS,
    WIN_LENGTH,
    GameState,
    initial_state,
    legal_moves,
    legal_moves_list,
    apply_move,
    is_terminal,
    render,
    get_symmetries,
)

from .encoding import (
    NUM_CHANNELS,
    encode_state,
    encode_state_batch,
    encode_state_torch,
    get_action_mask,
    get_action_mask_torch,
    mask_illegal_logits,
    decode_policy,
)

__all__ = [
    "ROWS",
    "COLS",
    "WIN_LENGTH",
    "GameState",
    "initial_state",
    "legal_moves",
    "legal_moves_list",
    "apply_move",
    "is_terminal",
    "render",
    "get_symmetries",
    "NUM_CHANNELS",
    "encode_state",
    "encode_state_batch",
    "encode_state_torch",
    "get_action_mask",
    "get_action_mask_torch",
    "mask_illegal_logits",
    "decode_policy",
]

"""Pure rules engine (no RL dependencies)."""

from catan.engine.actions import NUM_ACTIONS, ActionType, decode, describe, encode
from catan.engine.game import Game
from catan.engine.phases import Phase
from catan.engine.rules import IllegalActionError, apply, legal_actions, legal_mask
from catan.engine.state import GameState, RulesConfig

__all__ = [
    "NUM_ACTIONS",
    "ActionType",
    "Game",
    "GameState",
    "IllegalActionError",
    "Phase",
    "RulesConfig",
    "apply",
    "decode",
    "describe",
    "encode",
    "legal_actions",
    "legal_mask",
]

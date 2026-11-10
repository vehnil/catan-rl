"""Flat discrete action space.

Every move in the game maps to one integer in ``[0, NUM_ACTIONS)``. Each
``ActionType`` owns a contiguous block; ``encode``/``decode`` convert between
``(ActionType, arg)`` and the flat index. Player-relative arguments (steal
target, trade partner) are seat *offsets* 1..3 from the acting player so that
policies generalize across seats.
"""

from __future__ import annotations

from enum import IntEnum
from itertools import combinations_with_replacement
from typing import NamedTuple

import numpy as np

from catan.engine.board import NUM_EDGES, NUM_HEXES, NUM_VERTICES
from catan.engine.constants import NUM_RESOURCES


class ActionType(IntEnum):
    ROLL = 0
    END_TURN = 1
    BUILD_SETTLEMENT = 2
    BUILD_CITY = 3
    BUILD_ROAD = 4
    BUY_DEV = 5
    PLAY_KNIGHT = 6
    PLAY_ROAD_BUILDING = 7
    PLAY_YEAR_OF_PLENTY = 8
    PLAY_MONOPOLY = 9
    MOVE_ROBBER = 10
    STEAL = 11
    DISCARD = 12
    MARITIME_TRADE = 13
    PROPOSE_TRADE = 14
    ACCEPT_TRADE = 15
    REJECT_TRADE = 16
    CONFIRM_TRADE = 17
    CANCEL_TRADE = 18


def _multisets(max_size: int, allowed: list[int]) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = []
    for size in range(1, max_size + 1):
        out.extend(combinations_with_replacement(allowed, size))
    return out


def _vec(ms: tuple[int, ...]) -> np.ndarray:
    v = np.zeros(NUM_RESOURCES, dtype=np.int64)
    for r in ms:
        v[r] += 1
    return v


def _build_offers() -> list[tuple[np.ndarray, np.ndarray]]:
    """Domestic offers: give 1-2 cards, receive 1-2 cards, no resource on both sides."""
    offers = []
    all_res = list(range(NUM_RESOURCES))
    for give in _multisets(2, all_res):
        rest = [r for r in all_res if r not in give]
        for get in _multisets(2, rest):
            offers.append((_vec(give), _vec(get)))
    return offers


TRADE_OFFERS: list[tuple[np.ndarray, np.ndarray]] = _build_offers()
OFFER_GIVE = np.stack([g for g, _ in TRADE_OFFERS])  # (n_offers, 5)
OFFER_GET = np.stack([g for _, g in TRADE_OFFERS])

YOP_PAIRS: list[tuple[int, int]] = list(combinations_with_replacement(range(NUM_RESOURCES), 2))
MARITIME_PAIRS: list[tuple[int, int]] = [
    (g, r) for g in range(NUM_RESOURCES) for r in range(NUM_RESOURCES) if g != r
]

ACTION_SIZES: dict[ActionType, int] = {
    ActionType.ROLL: 1,
    ActionType.END_TURN: 1,
    ActionType.BUILD_SETTLEMENT: NUM_VERTICES,
    ActionType.BUILD_CITY: NUM_VERTICES,
    ActionType.BUILD_ROAD: NUM_EDGES,
    ActionType.BUY_DEV: 1,
    ActionType.PLAY_KNIGHT: 1,
    ActionType.PLAY_ROAD_BUILDING: 1,
    ActionType.PLAY_YEAR_OF_PLENTY: len(YOP_PAIRS),
    ActionType.PLAY_MONOPOLY: NUM_RESOURCES,
    ActionType.MOVE_ROBBER: NUM_HEXES,
    ActionType.STEAL: 3,
    ActionType.DISCARD: NUM_RESOURCES,
    ActionType.MARITIME_TRADE: len(MARITIME_PAIRS),
    ActionType.PROPOSE_TRADE: len(TRADE_OFFERS),
    ActionType.ACCEPT_TRADE: 1,
    ActionType.REJECT_TRADE: 1,
    ActionType.CONFIRM_TRADE: 3,
    ActionType.CANCEL_TRADE: 1,
}

ACTION_OFFSETS: dict[ActionType, int] = {}
_off = 0
for _t in ActionType:
    ACTION_OFFSETS[_t] = _off
    _off += ACTION_SIZES[_t]
NUM_ACTIONS = _off

_TYPE_OF = np.empty(NUM_ACTIONS, dtype=np.int64)
_ARG_OF = np.empty(NUM_ACTIONS, dtype=np.int64)
for _t in ActionType:
    _o = ACTION_OFFSETS[_t]
    _TYPE_OF[_o : _o + ACTION_SIZES[_t]] = int(_t)
    _ARG_OF[_o : _o + ACTION_SIZES[_t]] = np.arange(ACTION_SIZES[_t])


class Action(NamedTuple):
    type: ActionType
    arg: int = 0


def encode(t: ActionType, arg: int = 0) -> int:
    if not 0 <= arg < ACTION_SIZES[t]:
        raise ValueError(f"arg {arg} out of range for {t.name}")
    return ACTION_OFFSETS[t] + arg


def decode(index: int) -> Action:
    return Action(ActionType(int(_TYPE_OF[index])), int(_ARG_OF[index]))


def block(t: ActionType) -> slice:
    """Slice of the flat action vector owned by ``t``."""
    o = ACTION_OFFSETS[t]
    return slice(o, o + ACTION_SIZES[t])


def describe(index: int) -> str:
    """Human-readable action label (for logs, replays and debugging)."""
    from catan.engine.constants import RESOURCE_NAMES as RN

    a = decode(index)
    match a.type:
        case ActionType.PLAY_YEAR_OF_PLENTY:
            r1, r2 = YOP_PAIRS[a.arg]
            return f"PLAY_YEAR_OF_PLENTY({RN[r1]}, {RN[r2]})"
        case ActionType.PLAY_MONOPOLY | ActionType.DISCARD:
            return f"{a.type.name}({RN[a.arg]})"
        case ActionType.MARITIME_TRADE:
            g, r = MARITIME_PAIRS[a.arg]
            return f"MARITIME_TRADE({RN[g]} -> {RN[r]})"
        case ActionType.PROPOSE_TRADE:
            give, get = TRADE_OFFERS[a.arg]
            return f"PROPOSE_TRADE(give {_fmt(give)} for {_fmt(get)})"
        case ActionType.STEAL | ActionType.CONFIRM_TRADE:
            return f"{a.type.name}(seat+{a.arg + 1})"
        case _ if ACTION_SIZES[a.type] == 1:
            return a.type.name
        case _:
            return f"{a.type.name}({a.arg})"


def _fmt(v: np.ndarray) -> str:
    from catan.engine.constants import RESOURCE_NAMES as RN

    return "+".join(f"{int(n)} {RN[r]}" for r, n in enumerate(v) if n) or "nothing"

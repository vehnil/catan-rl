"""Observation (+ beliefs) -> flat float32 feature vector for neural policies.

All seat-indexed quantities are rotated so index 0 is "me", 1 is the next
seat, etc. That lets one network play any seat.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from catan.engine.actions import OFFER_GET, OFFER_GIVE
from catan.engine.board import NUM_EDGES, NUM_HEXES, NUM_VERTICES, Board
from catan.engine.constants import (
    DESERT,
    DICE_WEIGHTS,
    GENERIC_PORT,
    MAX_CITIES,
    MAX_ROADS,
    MAX_SETTLEMENTS,
    NUM_PLAYERS,
    NUM_RESOURCES,
)
from catan.engine.phases import NUM_PHASES
from catan.player.belief import BeliefTracker
from catan.player.observation import Observation

_HEX_DIM = 6 + 1 + 1  # terrain one-hot (5 + desert), pips, robber
_VERTEX_DIM = 2 * NUM_PLAYERS + 7  # (settlement, city) x relative owner, port one-hot
_EDGE_DIM = NUM_PLAYERS
_PLAYER_DIM = 12 + NUM_RESOURCES
_GLOBAL_DIM = (
    3 * NUM_RESOURCES  # own hand, playable devs, new devs
    + NUM_RESOURCES
    + 1  # bank, deck size
    + NUM_PHASES
    + 1
    + NUM_PLAYERS  # phase, is-my-move, current (relative)
    + 4  # dev played, proposals, actions, free roads
    + 11  # last roll one-hot
    + 2 * NUM_RESOURCES
    + 2 * NUM_PLAYERS  # trade offer, proposer, acceptors
)

FEATURE_DIM = (
    NUM_HEXES * _HEX_DIM
    + NUM_VERTICES * _VERTEX_DIM
    + NUM_EDGES * _EDGE_DIM
    + NUM_PLAYERS * _PLAYER_DIM
    + _GLOBAL_DIM
)


def _rel(owner: np.ndarray, seat: int) -> np.ndarray:
    """Absolute owner ids -> relative (0 = me), -1 stays -1."""
    return np.where(owner >= 0, (owner - seat) % NUM_PLAYERS, -1)


# Board-only blocks (terrain, pips, ports) never change during a game: built once per
# board and copied. Keyed by id with the board kept alive, so an id can't be reused.
_STATIC: OrderedDict[int, tuple[Board, np.ndarray, np.ndarray]] = OrderedDict()
_STATIC_MAX = 64


def _static_blocks(board: Board) -> tuple[np.ndarray, np.ndarray]:
    hit = _STATIC.get(id(board))
    if hit is not None and hit[0] is board:
        return hit[1], hit[2]
    hexf = np.zeros((NUM_HEXES, _HEX_DIM), np.float32)
    terrain = board.terrain
    hexf[np.arange(NUM_HEXES), np.where(terrain == DESERT, 5, terrain)] = 1
    hexf[:, 6] = [DICE_WEIGHTS.get(int(n), 0) / 6 for n in board.numbers]
    vf = np.zeros((NUM_VERTICES, _VERTEX_DIM), np.float32)
    port = board.vertex_port[:NUM_VERTICES]
    port_slot = np.where(port < 0, 0, np.where(port == GENERIC_PORT, 6, port + 1))
    vf[np.arange(NUM_VERTICES), 2 * NUM_PLAYERS + port_slot] = 1
    _STATIC[id(board)] = (board, hexf, vf)
    if len(_STATIC) > _STATIC_MAX:
        _STATIC.popitem(last=False)
    return hexf, vf


# Offsets of each block in the feature vector.
_V0 = NUM_HEXES * _HEX_DIM
_E0 = _V0 + NUM_VERTICES * _VERTEX_DIM
_P0 = _E0 + NUM_EDGES * _EDGE_DIM
_G0 = _P0 + NUM_PLAYERS * _PLAYER_DIM


def encode(obs: Observation, belief: BeliefTracker | None = None) -> np.ndarray:
    """Fills one preallocated vector block by block. Values (including float64-then-float32
    rounding) must stay identical to the original encoder; tests/player/test_features.py
    checks this against a frozen copy of it."""
    seat = obs.seat
    out = np.zeros(FEATURE_DIM, np.float32)
    order = np.array([(seat + i) % NUM_PLAYERS for i in range(NUM_PLAYERS)])
    static_hex, static_vertex = _static_blocks(obs.board)

    # Hexes.
    out[:_V0] = static_hex.ravel()
    out[obs.robber * _HEX_DIM + 7] = 1

    # Vertices.
    out[_V0:_E0] = static_vertex.ravel()
    vf = out[_V0:_E0].reshape(NUM_VERTICES, _VERTEX_DIM)
    owner = _rel(obs.vertex_owner[:NUM_VERTICES], seat)
    level = obs.vertex_level[:NUM_VERTICES]
    idx = np.flatnonzero(owner >= 0)
    vf[idx, 2 * owner[idx] + (level[idx] - 1)] = 1

    # Edges.
    ef = out[_E0:_P0].reshape(NUM_EDGES, _EDGE_DIM)
    eo = _rel(obs.edge_owner[:NUM_EDGES], seat)
    idx = np.flatnonzero(eo >= 0)
    ef[idx, eo[idx]] = 1

    # Players (relative order). Scalars are divided in float64 then stored as float32, and
    # hand estimates divided in float32, exactly as the original per-player lists were.
    pf = out[_P0:_G0].reshape(NUM_PLAYERS, _PLAYER_DIM)
    pf[:, 0] = obs.hand_sizes[order] / 10
    pf[:, 1] = obs.dev_counts[order] / 5
    pf[:, 2] = obs.knights[order] / 5
    pf[:, 3] = obs.public_vp[order] / 10
    pf[:, 4] = obs.road_length[order] / 10
    pf[:, 5] = obs.settlements_left[order] / MAX_SETTLEMENTS
    pf[:, 6] = obs.cities_left[order] / MAX_CITIES
    pf[:, 7] = obs.roads_left[order] / MAX_ROADS
    pf[:, 8] = order == obs.longest_road_holder
    pf[:, 9] = order == obs.largest_army_holder
    pf[:, 10] = obs.discard_pending[order] / 5
    hidden_vp = belief.expected_hidden_vps() if belief is not None else None
    for i, q in enumerate(order.tolist()):
        if belief is not None:
            pf[i, 11] = hidden_vp[q]  # type: ignore[index]
            expected = belief.hand_marginals(q)
        else:
            expected = obs.hand if q == seat else np.full(NUM_RESOURCES, obs.hand_sizes[q] / 5)
        pf[i, 12:] = np.asarray(expected, np.float32) / 5

    # Globals.
    gl = out[_G0:]
    r = NUM_RESOURCES
    gl[0:r] = obs.hand / 5
    gl[r : 2 * r] = obs.dev_hand / 3
    gl[2 * r : 3 * r] = obs.dev_new / 3
    gl[3 * r : 4 * r] = obs.bank / 19
    gl[4 * r] = obs.dev_deck_size / 25
    o = 4 * r + 1
    gl[o + int(obs.phase)] = 1
    o += NUM_PHASES
    gl[o] = float(obs.is_my_turn)
    o += 1
    gl[o + obs.relative(obs.current)] = 1
    o += NUM_PLAYERS
    gl[o] = float(obs.dev_played_this_turn)
    gl[o + 1] = obs.trade_proposals_this_turn / max(obs.config.max_trade_proposals_per_turn, 1)
    gl[o + 2] = obs.actions_this_turn / obs.config.max_actions_per_turn
    gl[o + 3] = obs.free_roads / 2
    o += 4
    if obs.last_roll:
        gl[o + obs.last_roll - 2] = 1
    o += 11
    if obs.trade_offer >= 0:
        gl[o : o + r] = OFFER_GIVE[obs.trade_offer] / 2
        gl[o + r : o + 2 * r] = OFFER_GET[obs.trade_offer] / 2
    o += 2 * r
    if obs.trade_proposer >= 0:
        gl[o + obs.relative(obs.trade_proposer)] = 1
    o += NUM_PLAYERS
    for a in obs.trade_acceptors:
        gl[o + obs.relative(a)] = 1
    assert o + NUM_PLAYERS == _GLOBAL_DIM
    return out

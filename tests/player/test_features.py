"""The feature encoder must stay exactly what trained policies saw.

``_reference_encode`` is a frozen copy of the original (pre-optimisation)
encoder. ``encode`` may be rewritten for speed, but must match it bit for bit
on every observation, with and without beliefs.
"""

import numpy as np
import pytest

from catan.engine import Game, RulesConfig
from catan.engine.actions import OFFER_GET, OFFER_GIVE
from catan.engine.board import NUM_EDGES, NUM_HEXES, NUM_VERTICES
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
from catan.player.features import _EDGE_DIM, _HEX_DIM, _VERTEX_DIM, FEATURE_DIM, _rel, encode
from catan.player.observation import Observation, observe


def _reference_encode(obs: Observation, belief: BeliefTracker | None = None) -> np.ndarray:
    seat = obs.seat
    parts: list[np.ndarray] = []
    order = [(seat + i) % NUM_PLAYERS for i in range(NUM_PLAYERS)]

    # Hexes.
    hexf = np.zeros((NUM_HEXES, _HEX_DIM), np.float32)
    terrain = obs.board.terrain
    hexf[np.arange(NUM_HEXES), np.where(terrain == DESERT, 5, terrain)] = 1
    hexf[:, 6] = [DICE_WEIGHTS.get(int(n), 0) / 6 for n in obs.board.numbers]
    hexf[obs.robber, 7] = 1
    parts.append(hexf.ravel())

    # Vertices.
    vf = np.zeros((NUM_VERTICES, _VERTEX_DIM), np.float32)
    owner = _rel(obs.vertex_owner[:NUM_VERTICES], seat)
    level = obs.vertex_level[:NUM_VERTICES]
    idx = np.flatnonzero(owner >= 0)
    vf[idx, 2 * owner[idx] + (level[idx] - 1)] = 1
    port = obs.board.vertex_port[:NUM_VERTICES]
    port_slot = np.where(port < 0, 0, np.where(port == GENERIC_PORT, 6, port + 1))
    vf[np.arange(NUM_VERTICES), 2 * NUM_PLAYERS + port_slot] = 1
    parts.append(vf.ravel())

    # Edges.
    ef = np.zeros((NUM_EDGES, _EDGE_DIM), np.float32)
    eo = _rel(obs.edge_owner[:NUM_EDGES], seat)
    idx = np.flatnonzero(eo >= 0)
    ef[idx, eo[idx]] = 1
    parts.append(ef.ravel())

    # Players (relative order).
    for p in order:
        expected = (
            belief.hand_marginals(p)
            if belief is not None
            else (obs.hand if p == seat else np.full(NUM_RESOURCES, obs.hand_sizes[p] / 5))
        )
        parts.append(
            np.array(
                [
                    obs.hand_sizes[p] / 10,
                    obs.dev_counts[p] / 5,
                    obs.knights[p] / 5,
                    obs.public_vp[p] / 10,
                    obs.road_length[p] / 10,
                    obs.settlements_left[p] / MAX_SETTLEMENTS,
                    obs.cities_left[p] / MAX_CITIES,
                    obs.roads_left[p] / MAX_ROADS,
                    float(obs.longest_road_holder == p),
                    float(obs.largest_army_holder == p),
                    obs.discard_pending[p] / 5,
                    belief.expected_hidden_vp(p) if belief is not None else 0.0,
                    *(np.asarray(expected, np.float32) / 5),
                ],
                np.float32,
            )
        )

    # Globals.
    g = [obs.hand / 5, obs.dev_hand / 3, obs.dev_new / 3, obs.bank / 19, [obs.dev_deck_size / 25]]
    phase = np.zeros(NUM_PHASES, np.float32)
    phase[int(obs.phase)] = 1
    cur = np.zeros(NUM_PLAYERS, np.float32)
    cur[obs.relative(obs.current)] = 1
    roll = np.zeros(11, np.float32)
    if obs.last_roll:
        roll[obs.last_roll - 2] = 1
    give = OFFER_GIVE[obs.trade_offer] if obs.trade_offer >= 0 else np.zeros(NUM_RESOURCES)
    get = OFFER_GET[obs.trade_offer] if obs.trade_offer >= 0 else np.zeros(NUM_RESOURCES)
    proposer = np.zeros(NUM_PLAYERS, np.float32)
    if obs.trade_proposer >= 0:
        proposer[obs.relative(obs.trade_proposer)] = 1
    acceptors = np.zeros(NUM_PLAYERS, np.float32)
    for a in obs.trade_acceptors:
        acceptors[obs.relative(a)] = 1
    g += [
        phase,
        [float(obs.is_my_turn)],
        cur,
        [
            float(obs.dev_played_this_turn),
            obs.trade_proposals_this_turn / max(obs.config.max_trade_proposals_per_turn, 1),
            obs.actions_this_turn / obs.config.max_actions_per_turn,
            obs.free_roads / 2,
        ],
        roll,
        give / 2,
        get / 2,
        proposer,
        acceptors,
    ]
    parts.extend(np.asarray(x, np.float32) for x in g)
    out = np.concatenate(parts)
    assert out.shape == (FEATURE_DIM,), out.shape
    return out


@pytest.mark.parametrize("board", ["random", "standard"])
def test_encode_matches_reference_bit_for_bit(board):
    checked = 0
    for seed in range(6):
        g = Game(seed, RulesConfig(board=board, max_turns_per_game=200))
        rng = np.random.default_rng(seed)
        beliefs = [BeliefTracker(s) for s in range(NUM_PLAYERS)]
        while not g.done:
            for s in range(NUM_PLAYERS):
                obs = observe(g.state, s)
                for bel in (beliefs[s], None):
                    new, ref = encode(obs, bel), _reference_encode(obs, bel)
                    assert new.dtype == ref.dtype and np.array_equal(new, ref)
                    checked += 1
            obs = observe(g.state, g.actor)
            for e in g.step(int(rng.choice(np.flatnonzero(obs.legal_mask)))):
                for s, bel in enumerate(beliefs):
                    bel.update(e.view(s))
    assert checked > 5000

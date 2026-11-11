"""Invariants checked at every step of random games."""

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from catan.engine import Game, IllegalActionError, RulesConfig, legal_mask
from catan.engine.board import NUM_VERTICES
from catan.engine.constants import BANK_PER_RESOURCE, DEV_DECK_COUNTS, DevCard
from catan.engine.rules import apply
from catan.engine.scoring import buildings_vp, longest_road
from catan.engine.state import GameState

CONFIGS = [RulesConfig(board="standard"), RulesConfig(board="random", max_turns_per_game=300)]


def check_invariants(s: GameState) -> None:
    assert (s.hands >= 0).all() and (s.bank >= 0).all()
    assert (s.hands.sum(0) + s.bank == BANK_PER_RESOURCE).all()
    held = s.dev_hand.sum(0) + s.dev_new.sum(0)
    played_knights = s.knights.sum()
    assert len(s.dev_deck) + held.sum() <= DEV_DECK_COUNTS.sum()
    assert held[DevCard.KNIGHT] + played_knights <= DEV_DECK_COUNTS[DevCard.KNIGHT]
    owner = s.vertex_owner[:NUM_VERTICES]
    for p in range(4):
        n_sett = ((owner == p) & (s.vertex_level[:NUM_VERTICES] == 1)).sum()
        n_city = ((owner == p) & (s.vertex_level[:NUM_VERTICES] == 2)).sum()
        assert n_sett + s.settlements_left[p] == 5
        assert n_city + s.cities_left[p] == 4
        assert (s.edge_owner == p).sum() + s.roads_left[p] == 15
        assert s.road_length[p] == longest_road(s, p)
    assert s.vertex_owner[-1] == -1 and s.edge_owner[-1] == -1
    vp = buildings_vp(s)
    for p in range(4):
        expected = vp[p] + 2 * (s.longest_road_holder == p) + 2 * (s.largest_army_holder == p)
        assert s.public_vp(p) == expected


@settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 2**31 - 1), cfg=st.sampled_from(CONFIGS))
def test_random_games_preserve_invariants_and_replay(seed, cfg):
    rng = np.random.default_rng(seed)
    g = Game(seed, cfg)
    steps = 0
    while not g.done:
        mask = legal_mask(g.state)
        legal = np.flatnonzero(mask)
        assert len(legal) > 0, f"no legal actions in {g.state.phase.name}"
        if steps % 97 == 0:  # spot-check that a masked-out action is rejected
            illegal = np.flatnonzero(~mask)
            try:
                apply(g.state.clone(), int(illegal[rng.integers(len(illegal))]), rng)
                raise AssertionError("illegal action accepted")
            except IllegalActionError:
                pass
        g.step(int(rng.choice(legal)))
        check_invariants(g.state)
        steps += 1
    s = g.state
    assert s.truncated or s.total_vp(s.winner) >= cfg.vp_to_win

    r = Game.replay(seed, cfg, g.actions)
    assert (r.state.hands == s.hands).all()
    assert (r.state.vertex_owner == s.vertex_owner).all()
    assert r.state.winner == s.winner and r.state.turn == s.turn


def test_clone_is_independent():
    g = Game(0)
    rng = np.random.default_rng(0)
    for _ in range(200):
        g.step(int(rng.choice(np.flatnonzero(legal_mask(g.state)))))
    c = g.state.clone()
    c.hands[0, 0] += 5
    before = g.state.vertex_owner.copy()
    c.vertex_owner[:] = 2
    c.dev_deck.pop()
    assert g.state.hands[0, 0] != c.hands[0, 0]
    assert (g.state.vertex_owner == before).all()
    assert len(g.state.dev_deck) == len(c.dev_deck) + 1

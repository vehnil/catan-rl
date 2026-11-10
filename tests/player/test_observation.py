from dataclasses import fields

import numpy as np

from catan.engine import Game
from catan.engine.constants import DevCard
from catan.player.belief import BeliefTracker
from catan.player.features import FEATURE_DIM, encode
from catan.player.observation import observe
from tests.conftest import play_random


def _advance(seed: int, steps: int) -> Game:
    g = Game(seed)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        if g.done:
            break
        g.step(int(rng.choice(g.legal_actions())))
    return g


def _perturb_hidden(state, seat):
    """Change only information ``seat`` should not see."""
    s = state.clone()
    for q in range(4):
        if q == seat:
            continue
        h = s.hands[q]
        nz = np.flatnonzero(h)
        if len(nz):  # move one card to another type, keeping hand size
            r = int(nz[0])
            h[r] -= 1
            h[(r + 1) % 5] += 1
        if s.dev_hand[q].sum():
            c = int(np.flatnonzero(s.dev_hand[q])[0])
            s.dev_hand[q, c] -= 1
            s.dev_hand[q, (c + 1) % 5] += 1
    s.dev_deck = list(reversed(s.dev_deck))
    return s


def test_observation_does_not_leak_hidden_information():
    g = _advance(3, 600)
    for seat in range(4):
        a = observe(g.state, seat)
        b = observe(_perturb_hidden(g.state, seat), seat)
        for f in fields(a):
            if f.name in ("redacted", "board", "config"):
                continue
            va, vb = getattr(a, f.name), getattr(b, f.name)
            if isinstance(va, np.ndarray):
                assert (va == vb).all(), f.name
            else:
                assert va == vb, f.name
        assert (a.redacted.hands == b.redacted.hands).all()
        assert a.redacted.dev_deck == b.redacted.dev_deck == []
        assert (encode(a) == encode(b)).all()


def test_mask_only_for_actor():
    g = Game(0)
    for seat in range(4):
        obs = observe(g.state, seat)
        assert obs.legal_mask.any() == (seat == g.state.actor)


def test_features_shape_and_finite():
    g = _advance(1, 400)
    tr = BeliefTracker(0)
    for e in g.events:
        tr.update(e.view(0))
    x = encode(observe(g.state, 0), tr)
    assert x.shape == (FEATURE_DIM,) and x.dtype == np.float32
    assert np.isfinite(x).all()


def test_my_vp_includes_hidden_vp_cards():
    g = Game(0)
    g.state.dev_hand[2, DevCard.VICTORY_POINT] = 1
    assert observe(g.state, 2).my_vp == 1
    assert observe(g.state, 0).public_vp[2] == 0


def test_play_random_helper_terminates():
    g = play_random(Game(5), np.random.default_rng(5))
    assert g.done

import numpy as np
import pytest

from catan.engine import Game
from catan.engine.constants import BANK_PER_RESOURCE
from catan.player.belief import BeliefTracker, HandDistribution
from catan.player.observation import observe


def test_unseen_steal_branches_and_viterbi_picks_most_likely():
    fwd = HandDistribution(hands=np.array([[2, 1, 0, 0, 0]]), weights=np.ones(1))
    vit = HandDistribution(hands=np.array([[2, 1, 0, 0, 0]]), weights=np.ones(1), mode="max")
    p = fwd.lose_random_card()
    vit.lose_random_card()
    assert p[:2] == pytest.approx([2 / 3, 1 / 3])
    probs = {tuple(h): w for h, w in zip(fwd.hands.tolist(), fwd.probs(), strict=True)}
    assert probs[(1, 1, 0, 0, 0)] == pytest.approx(2 / 3)
    assert probs[(2, 0, 0, 0, 0)] == pytest.approx(1 / 3)
    assert vit.best().tolist() == [1, 1, 0, 0, 0]

    thief = HandDistribution()
    thief.gain_random_card(p)
    assert thief.expected()[:2] == pytest.approx([2 / 3, 1 / 3])
    # Evidence collapses the posterior: the thief later pays brick.
    thief.require(np.array([0, 1, 0, 0, 0]))
    assert thief.best().tolist() == [0, 1, 0, 0, 0] and len(thief.weights) == 1


def test_viterbi_differs_from_marginal_mode():
    # [1,1] is reachable from two medium-probability histories; [2,0] from one strong one.
    hands = np.array([[3, 0, 0, 0, 0], [2, 1, 0, 0, 0], [1, 2, 0, 0, 0]])
    w = np.array([0.3, 0.35, 0.35])
    fwd = HandDistribution(hands=hands.copy(), weights=w.copy())
    vit = HandDistribution(hands=hands.copy(), weights=w.copy(), mode="max")
    fwd.lose_random_card()
    vit.lose_random_card()
    assert fwd.best().tolist() == [1, 1, 0, 0, 0]  # posterior mode
    assert vit.best().tolist() == [2, 0, 0, 0, 0]  # most likely single history


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_tracker_matches_ground_truth_in_random_games(seed):
    rng = np.random.default_rng(seed)
    g = Game(seed)
    trackers = [BeliefTracker(s) for s in range(4)]
    hidden_steals = 0
    while not g.done:
        for e in g.step(int(rng.choice(g.legal_actions()))):
            hidden_steals += e.kind == "steal"
            for s, tr in enumerate(trackers):
                tr.update(e.view(s))
        st = g.state
        for s, tr in enumerate(trackers):
            assert (tr.my_hand == st.hands[s]).all()
            assert (tr.dev_held == st.dev_hand.sum(1) + st.dev_new.sum(1)).all()
            for q in tr.opponents:
                assert tr.forward[q].contains(st.hands[q])
                assert tr.viterbi[q].contains(st.hands[q])
                assert tr.forward[q].hands.sum(1).tolist() == [st.hands[q].sum()] * tr.uncertainty(
                    q
                )
    assert hidden_steals > 0


def test_hands_known_exactly_after_setup():
    g = Game(0)
    rng = np.random.default_rng(0)
    tr = BeliefTracker(0)
    while g.state.is_setup:
        for e in g.step(int(rng.choice(g.legal_actions()))):
            tr.update(e.view(0))
    for q in tr.opponents:
        assert tr.uncertainty(q) == 1
        assert (tr.map_hand(q) == g.state.hands[q]).all()


def test_determinize_is_consistent_with_public_info():
    rng = np.random.default_rng(7)
    g = Game(7)
    tr = BeliefTracker(1)
    for _ in range(1500):
        if g.done:
            break
        for e in g.step(int(rng.choice(g.legal_actions()))):
            tr.update(e.view(1))
    obs = observe(g.state, 1)
    for _ in range(10):
        d = tr.determinize(obs.redacted, rng)
        assert (d.hands.sum(0) + d.bank == BANK_PER_RESOURCE).all()
        assert (d.hands.sum(1) == g.state.hands.sum(1)).all()
        assert (d.hands[1] == g.state.hands[1]).all()
        held = d.dev_hand.sum(1) + d.dev_new.sum(1)
        assert (held == g.state.dev_hand.sum(1) + g.state.dev_new.sum(1)).all()
        assert len(d.dev_deck) == len(g.state.dev_deck)
        assert (d.vertex_owner == g.state.vertex_owner).all()

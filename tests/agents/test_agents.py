import numpy as np
import pytest

from catan.agents.bandits import UCB1, BetaThompson
from catan.agents.registry import available, make_agent, parse_spec
from catan.engine import RulesConfig
from catan.engine.actions import encode
from catan.sim.runner import play_game

FAST_SPECS = [
    "random",
    "greedy",
    "greedy:preset=expansion",
    "opening_bandit:budget=6,top_k=3,rollout_turns=2",
    "meta_bandit",
    "maxn:depth=1,beam=3",
    "paranoid:depth=2,beam=3",
    "ismcts:iterations=8,max_depth=3",
    "ppo:hidden=[64]",
]


def test_parse_spec():
    assert parse_spec("greedy") == ("greedy", {})
    assert parse_spec("ismcts:iterations=50,c=0.5,flag=true,name=x") == (
        "ismcts",
        {"iterations": 50, "c": 0.5, "flag": True, "name": "x"},
    )
    assert parse_spec("policy:abc") == ("policy", {"ref": "abc"})
    assert "ismcts" in available()


def test_unknown_agent():
    with pytest.raises(KeyError):
        make_agent("nope")


@pytest.mark.parametrize("spec", [s for s in FAST_SPECS if not s.startswith("ppo")])
def test_agent_plays_full_legal_game(spec):
    """Every agent finishes a game; illegal actions would raise inside the engine."""
    agents = [make_agent(spec), make_agent("random"), make_agent("random"), make_agent("greedy")]
    rec = play_game(agents, seed=3, config=RulesConfig(max_turns_per_game=200))
    assert rec.turns > 0 and (rec.winner >= 0 or rec.truncated)


def test_ppo_agent_plays():
    from catan.agents.learned.ppo.agent import PPOAgent

    agents = [
        PPOAgent(hidden=[32]),
        make_agent("random"),
        make_agent("random"),
        make_agent("random"),
    ]
    rec = play_game(agents, seed=0, config=RulesConfig(max_turns_per_game=60))
    assert rec.steps > 0


@pytest.mark.slow
def test_greedy_dominates_random():
    wins = 0
    for seed in range(12):
        seat = seed % 4
        lineup = [make_agent("random") for _ in range(3)]
        lineup.insert(seat, make_agent("greedy"))
        wins += play_game(lineup, seed).winner == seat
    assert wins >= 10


def test_ucb1_finds_best_arm():
    rng = np.random.default_rng(0)
    b = UCB1(3, c=0.5)
    p = [0.2, 0.8, 0.5]
    for _ in range(400):
        a = b.select(rng)
        b.update(a, float(rng.random() < p[a]))
    assert b.best() == 1
    assert UCB1.from_dict(b.to_dict()).counts.tolist() == b.counts.tolist()


def test_thompson_finds_best_arm():
    rng = np.random.default_rng(0)
    t = BetaThompson(3)
    p = [0.3, 0.4, 0.9]
    for _ in range(300):
        a = t.select(rng)
        t.update(a, float(rng.random() < p[a]))
    assert t.best() == 2


def test_meta_bandit_persists_stats(tmp_path):
    path = tmp_path / "meta.json"
    for seed in range(3):
        agents = [make_agent(f"meta_bandit:stats_path={path}")] + [
            make_agent("random") for _ in range(3)
        ]
        play_game(agents, seed, RulesConfig(max_turns_per_game=200))
    from catan.agents.bandits.meta_bandit import MetaBanditAgent

    m = MetaBanditAgent(stats_path=str(path))
    assert m.bandit.counts.sum() == pytest.approx(3)


def test_ismcts_root_statistics():
    from catan.agents.search.ismcts import ISMCTSAgent
    from catan.engine import Game
    from catan.player.observation import observe

    g = Game(0)
    agent = ISMCTSAgent(iterations=30, max_depth=2)
    agent.reset(0, np.random.default_rng(0))
    a = agent.act(observe(g.state, 0))
    stats = agent.root_stats()
    assert sum(v["visits"] for v in stats.values()) == 30
    assert g.legal_mask()[a]


def test_greedy_prefers_city_over_ending_turn():
    from catan.agents.baselines import GreedyAgent
    from catan.engine.actions import ActionType
    from catan.player.observation import observe
    from tests.conftest import give, main_phase_game, place

    g = main_phase_game()
    place(g.state, 0, 20)
    give(g.state, 0, [0, 0, 0, 2, 3])
    agent = GreedyAgent()
    agent.reset(0, np.random.default_rng(0))
    assert agent.act(observe(g.state, 0)) == encode(ActionType.BUILD_CITY, 20)


def _greedy_response(proposer_hand, responder_hand, offer_give, offer_get, **greedy_kwargs):
    """P0 (settlement at 20) proposes to P1 (settlement at 40); returns P1's greedy decision."""
    from catan.agents.baselines import GreedyAgent
    from catan.agents.heuristics import evaluate
    from catan.engine.actions import ActionType
    from catan.engine.rules import legal_mask
    from tests.conftest import give, main_phase_game, place
    from tests.engine.test_trading import offer_index

    g = main_phase_game()
    place(g.state, 0, 20)
    place(g.state, 1, 40)
    give(g.state, 0, proposer_hand)
    give(g.state, 1, responder_hand)
    offer = offer_index(offer_give, offer_get)
    g.step(encode(ActionType.PROPOSE_TRADE, offer))
    assert g.state.actor == 1

    def gain(p, delta):
        s = g.state.clone()
        s.hands[p] += delta
        return evaluate(s, p) - evaluate(g.state, p)

    give_v, get_v = np.array(offer_give), np.array(offer_get)
    gains = {"proposer": gain(0, get_v - give_v), "responder": gain(1, give_v - get_v)}
    agent = GreedyAgent(**greedy_kwargs)
    agent.reset(1, np.random.default_rng(0))
    action = agent.act_state(g.state, legal_mask(g.state), 1)
    return action == encode(ActionType.ACCEPT_TRADE), gains


def test_greedy_rejects_offers_asking_for_more_cards():
    # P0 offers 1 ore for 2 wood; P1 is one ore short of a city and doesn't need wood.
    args = ([0, 0, 0, 0, 1], [2, 0, 0, 2, 2], [0, 0, 0, 0, 1], [2, 0, 0, 0, 0])
    accepted, gains = _greedy_response(*args)
    assert gains["responder"] > 2.0  # the old self-only rule would accept
    assert not accepted
    accepted_old, _ = _greedy_response(*args, max_partner_edge=float("inf"))
    assert accepted_old


def test_greedy_rejects_offers_that_help_the_proposer_more():
    # P0 (9 cards) gives 2 wood for 1 ore: completes a city and gets under the discard limit.
    # P1 completes a road's worth of cards, a smaller gain.
    args = ([4, 1, 0, 2, 2], [0, 1, 0, 0, 1], [2, 0, 0, 0, 0], [0, 0, 0, 0, 1])
    accepted, gains = _greedy_response(*args)
    assert 2.0 < gains["responder"] < gains["proposer"]
    assert not accepted
    accepted_old, _ = _greedy_response(*args, max_partner_edge=float("inf"))
    assert accepted_old
    accepted_lenient, _ = _greedy_response(*args, max_partner_edge=100.0)
    assert accepted_lenient


def test_evaluate_hands_matches_evaluate():
    from catan.agents.heuristics import PRESETS, evaluate, evaluate_hands
    from catan.engine import Game, legal_actions

    rng = np.random.default_rng(7)
    checked = 0
    for seed in range(6):
        g = Game(seed, RulesConfig())
        while not g.done:
            g.step(int(rng.choice(legal_actions(g.state))))
            if rng.random() > 0.05:
                continue
            for p in range(4):
                hands = g.state.hands[p] + rng.integers(-2, 4, size=(12, 5))
                hands = np.maximum(hands, 0)
                for w in PRESETS.values():
                    expected = []
                    for h in hands:
                        s = g.state.clone()
                        s.hands[p] = h
                        expected.append(evaluate(s, p, w))
                    assert evaluate_hands(g.state, p, hands, w).tolist() == expected
                    checked += 1
    assert checked > 100

import json

import numpy as np

from catan.agents.registry import make_agent
from catan.engine import RulesConfig
from catan.engine.board import NUM_EDGES, NUM_HEXES, NUM_VERTICES
from catan.engine.constants import NUM_PLAYERS
from catan.sim.replay import rebuild
from catan.sim.runner import play_game
from catan.sim.timeline import build_timeline

QUICK = RulesConfig(max_turns_per_game=150)


def _record(seed=0, specs=("greedy", "greedy", "random", "random")):
    return play_game([make_agent(s) for s in specs], seed, QUICK, specs=specs)


def test_timeline_shape_and_final_state():
    rec = _record()
    tl = build_timeline(rec)
    json.dumps(tl)  # plain JSON all the way down
    assert tl["meta"]["verified"]
    assert len(tl["board"]["hexes"]) == NUM_HEXES
    assert len(tl["board"]["vertices"]) == NUM_VERTICES
    assert len(tl["board"]["edges"]) == NUM_EDGES
    steps = tl["steps"]
    assert len(steps) == len(rec.actions) + 1
    assert steps[0]["action"] is None
    assert [s["action"]["index"] for s in steps[1:]] == rec.actions

    final, last = rebuild(rec).state, steps[-1]["state"]
    assert last["vertex_owner"] == final.vertex_owner[:NUM_VERTICES].tolist()
    assert last["edge_owner"] == final.edge_owner[:NUM_EDGES].tolist()
    assert last["winner"] == final.winner
    for p, pl in enumerate(last["players"]):
        assert pl["hand"] == final.hands[p].tolist()
        assert pl["total_vp"] == final.total_vp(p)


def test_seat_views_hide_secrets():
    tl = build_timeline(_record(seed=3))
    steals = buys = 0
    for s in tl["steps"]:
        for e in s["events"]:
            if e["kind"] == "steal":
                steals += 1
                outsiders = {q for q in range(NUM_PLAYERS) if q not in (e["player"], e["victim"])}
                assert set(map(int, e["views"])) == outsiders
                assert all("resource" not in v for v in e["views"].values())
            if e["kind"] == "buy_dev":
                buys += 1
                assert "card" in e
                assert e["player"] not in map(int, e["views"])
                assert all("card" not in v for v in e["views"].values())
            if e["kind"] in ("roll", "build_road"):
                assert "views" not in e
    assert steals + buys > 0


def test_exact_beliefs_match_truth():
    tl = build_timeline(_record(seed=1))
    checked = 0
    for s in tl["steps"]:
        hands = [pl["hand"] for pl in s["state"]["players"]]
        for seat, b in enumerate(s["beliefs"]):
            assert b["hands"][str(seat)]["map"] == hands[seat]
            for q, h in b["hands"].items():
                # summary() rounds expectations to 2 decimals
                assert np.isclose(sum(h["expected"]), sum(hands[int(q)]), atol=0.03)
                if h["hypotheses"] == 1:
                    assert h["map"] == hands[int(q)]
                    checked += 1
    assert checked > 0

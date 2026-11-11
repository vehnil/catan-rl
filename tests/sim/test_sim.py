import json

import pytest

from catan.agents.registry import make_agent
from catan.config import TournamentConfig, TrainConfig, load_config
from catan.engine import RulesConfig
from catan.sim.artifacts import export_policy, load_policy, read_manifest, save_policy
from catan.sim.ratings import update_ratings, wilson_interval
from catan.sim.replay import load_replay, rebuild, save_replay, step_through, verify
from catan.sim.runner import Matchup, play_game, run_matchups
from catan.sim.store import Store
from catan.sim.tournament import (
    run_tournament,
    schedule_fixed,
    schedule_gauntlet,
    schedule_round_robin,
)

QUICK = RulesConfig(max_turns_per_game=150)


def _record(seed=0, specs=("greedy", "random", "random", "random")):
    return play_game([make_agent(s) for s in specs], seed, QUICK, specs=specs)


def test_record_ranks_and_replay_roundtrip(tmp_path):
    rec = _record()
    assert sorted(rec.ranks)[0] == 1
    if rec.winner >= 0:
        assert rec.ranks[rec.winner] == 1
    path = save_replay(rec, tmp_path)
    loaded = load_replay(path)
    assert loaded == rec
    assert verify(loaded)
    assert rebuild(loaded).state.turn == rec.turns
    steps = list(step_through(loaded))
    assert len(steps) == len(rec.actions)


def test_parallel_matches_sequential():
    ms = [Matchup(("greedy", "random", "random", "random"), s) for s in range(4)]
    seq = [r.winner for r in run_matchups(ms, QUICK, workers=1)]
    par = [r.winner for r in run_matchups(ms, QUICK, workers=2)]
    assert seq == par


def test_schedules_rotate_seats():
    fixed = schedule_fixed(["a", "b", "c", "d"], 8, 0)
    assert {m.specs[0] for m in fixed} == {"a", "b", "c", "d"}
    rr = schedule_round_robin(["a", "b"], 16, 0)
    assert all(set(m.specs) == {"a", "b"} for m in rr)
    rr5 = schedule_round_robin(list("abcde"), 20, 0)
    assert all(len(set(m.specs)) == 4 for m in rr5)
    g = schedule_gauntlet("x", ["a", "b"], 8, 0)
    assert [m.specs.index("x") for m in g] == [0, 1, 2, 3, 0, 1, 2, 3]


def test_store_roundtrip(tmp_path):
    with Store(tmp_path / "t.db") as store:
        run = store.start_run("sim", {"k": 1})
        rec = _record()
        mid = store.add_match(rec, run, None)
        assert mid > 0
        stats = {s["agent"]: s for s in store.agent_stats(run)}
        assert stats["random"]["games"] == 3
        store.finish_run(run, {"ok": True})
        assert store.runs()[0]["status"] == "done"


def test_ratings_reward_winner_and_share_ties():
    from catan.sim.runner import GameRecord

    rec = GameRecord(
        seed=0,
        rules={},
        agents=["a", "b", "c", "d"],
        winner=0,
        turns=50,
        truncated=False,
        vp=[10, 7, 7, 3],
        ranks=[1, 2, 2, 4],
        steps=500,
        duration_s=0.1,
    )
    r = update_ratings({}, [rec, rec])
    assert r["a"]["mu"] > r["b"]["mu"] > r["d"]["mu"]
    assert r["b"]["mu"] == pytest.approx(r["c"]["mu"])
    assert r["a"]["games"] == 2
    lo, hi = wilson_interval(8, 10)
    assert lo < 0.8 < hi


def test_tournament_end_to_end(tmp_path):
    cfg = TournamentConfig(
        format="fixed",
        agents=["greedy", "random", "random", "random"],
        games=4,
        data_dir=str(tmp_path),
        rules={"max_turns_per_game": 150},
    )
    summary = run_tournament(cfg)
    assert summary["games"] == 4
    assert set(summary["agents"]) == {"greedy", "random"}
    with Store(tmp_path / "catan.db") as store:
        assert len(store.matches(summary["run_id"])) == 4
        assert {r["agent"] for r in store.leaderboard()} == {"greedy", "random"}
    assert len(list((tmp_path / "replays" / summary["run_id"]).glob("*.json.gz"))) == 4


def test_policy_save_load_export(tmp_path):
    agent = make_agent("greedy:preset=cities")
    with Store(tmp_path / "catan.db") as store:
        manifest, path = save_policy(agent, tmp_path, name="cities-v1", store=store)
        assert store.policy("cities-v1")["id"] == manifest["id"]
    assert read_manifest(path)["kind"] == "greedy"
    loaded = load_policy(path)
    assert loaded.w == agent.w
    via_registry = make_agent(f"policy:{manifest['id']}", tmp_path)
    assert via_registry.w == agent.w
    out = export_policy(path, "json")
    assert json.loads(out.read_text())["manifest"]["id"] == manifest["id"]


def test_ppo_policy_roundtrip(tmp_path):
    import torch

    from catan.agents.learned.ppo.agent import PPOAgent

    agent = PPOAgent(hidden=[16])
    _, path = save_policy(agent, tmp_path)
    loaded = load_policy(path)
    for (k, v), (_, w) in zip(
        agent.model.state_dict().items(), loaded.model.state_dict().items(), strict=True
    ):
        assert torch.equal(v, w), k
    exported = json.loads(export_policy(path, "json").read_text())
    assert "trunk.0.weight" in exported["weights"]


def test_configs_load_and_reject_typos(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text("agents: [greedy, random]\ngames: 3\nrules: {board: random}\n")
    cfg = load_config(p, TournamentConfig)
    assert cfg.rules.to_rules().board == "random"
    p.write_text("agents: [greedy]\ngamez: 3\n")
    with pytest.raises(ValueError):
        load_config(p, TournamentConfig)
    with pytest.raises(ValueError):
        TournamentConfig(format="fixed", agents=["a", "b"])


@pytest.mark.slow
def test_ppo_train_smoke(tmp_path):
    from catan.agents.learned.ppo.trainer import train

    cfg = TrainConfig.model_validate(
        {
            "data_dir": str(tmp_path),
            "rules": {"max_turns_per_game": 120},
            "ppo": {
                "total_steps": 256,
                "rollout_steps": 128,
                "minibatch": 64,
                "epochs": 1,
                "hidden": [32],
            },
            "league": {"fixed_opponents": ["random"], "snapshot_every": 128},
            "eval_every": 10_000,
            "eval_games": 2,
        }
    )
    result = train(cfg, log=lambda *_: None)
    assert (tmp_path / "policies" / result["policy_id"] / "weights.safetensors").exists()
    with Store(tmp_path / "catan.db") as store:
        assert store.policy(result["policy_id"])["kind"] == "ppo"
        assert store.runs()[0]["status"] == "done"


@pytest.mark.slow
def test_ppo_train_parallel_envs(tmp_path):
    from catan.agents.learned.ppo.trainer import train

    cfg = TrainConfig.model_validate(
        {
            "data_dir": str(tmp_path),
            "rules": {"max_turns_per_game": 120},
            "ppo": {
                "total_steps": 256,
                "rollout_steps": 128,
                "minibatch": 64,
                "epochs": 1,
                "hidden": [32],
                "num_envs": 2,
            },
            "league": {"fixed_opponents": ["random"], "snapshot_every": 100},
            "eval_every": 200,
            "eval_games": 3,
        }
    )
    result = train(cfg, log=lambda *_: None)
    with Store(tmp_path / "catan.db") as store:
        assert store.runs()[0]["status"] == "done"
    rows = [
        json.loads(line) for line in (tmp_path / "runs" / result["run_id"] / "metrics.jsonl").open()
    ]
    assert rows[-1]["steps"] == 256 and rows[-1]["snapshots"] == 2
    assert "eval_win_rate" in rows[-1] and "eval_win_rate" in result


@pytest.mark.slow
def test_parallel_eval_matches_sequential():
    from catan.agents.learned.ppo.model import ActorCritic
    from catan.agents.learned.ppo.workers import RolloutWorkers, eval_games
    from catan.config.models import LeagueConfig

    model = ActorCritic([32])
    opponents = ["greedy", "random", "random"]
    expected = eval_games(model, [32], opponents, list(range(5)), QUICK, 11)
    league = LeagueConfig(fixed_opponents=["random"])
    workers = RolloutWorkers(2, QUICK, 0.0, 0.995, league, [32], 0)
    try:
        assert workers.evaluate(model, [32], opponents, 5, QUICK, 11) == expected
        segments = workers.collect(model, 200)
        assert sum(len(seg["actions"]) for _, seg in segments) in range(200, 200 + 64)
        assert {w for w, _ in segments} == {0, 1}
    finally:
        workers.close()


def test_buffer_gae_per_env_matches_single_env_buffers():
    """Interleaved rows from several envs get the same advantages as separate buffers."""
    import numpy as np

    from catan.agents.learned.ppo.buffer import RolloutBuffer

    rng = np.random.default_rng(0)
    n, size = 3, 30
    order = rng.integers(n, size=size)  # which env each row comes from (async arrival)
    values = rng.normal(size=size).astype(np.float32)
    rewards = rng.normal(size=size).astype(np.float32)
    dones = (rng.random(size) < 0.2).astype(np.float32)
    last = {e: float(rng.normal()) for e in range(n)}

    def fill(buf, rows, env_of):
        for t in rows:
            buf.add(np.zeros(4), np.ones(2), 0, 0.0, values[t], rewards[t], dones[t], env=env_of(t))

    mixed = RolloutBuffer(size, 4, 2)
    fill(mixed, range(size), lambda t: int(order[t]))
    mixed.compute_returns(last, 0.99, 0.95)
    for e in range(n):
        rows = np.flatnonzero(order == e)
        single = RolloutBuffer(len(rows), 4, 2)
        fill(single, rows, lambda t: 0)
        single.compute_returns({0: last[e]}, 0.99, 0.95)
        assert np.array_equal(mixed.advantages[rows], single.advantages)

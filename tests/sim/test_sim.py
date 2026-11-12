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
    workers = RolloutWorkers(2, QUICK, {}, 0.995, league, [32], 0)
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


def test_league_lineup_is_exact_and_shuffled():
    import numpy as np

    from catan.agents.learned.ppo.league import League
    from catan.config.models import LeagueConfig

    cfg = LeagueConfig(
        fixed_opponents=["random", "random", "greedy"], lineup=True, self_play_prob=0
    )
    league = League(cfg, [32], np.random.default_rng(0))
    seats = [[a.name for a in league.sample_opponents()] for _ in range(60)]
    assert all(sorted(s) == ["greedy", "random", "random"] for s in seats)
    assert len({s.index("greedy") for s in seats}) == 3  # greedy moves between seats
    with pytest.raises(ValueError):
        LeagueConfig(fixed_opponents=["random", "greedy"], lineup=True)


def test_policy_lookup_by_training_config_name(tmp_path):
    with Store(tmp_path / "catan.db") as store:
        for i, name in enumerate(["stage1-aaa", "stage1-bbb", "stage10-ccc"]):
            manifest = {"id": f"id{i}", "name": name, "kind": "ppo", "created_at": float(i)}
            store.add_policy(manifest, tmp_path / name)
        assert store.policy("stage1")["id"] == "id1"  # newest of stage1-*, not stage10-*
        assert store.policy("stage1-aaa")["id"] == "id0"  # exact name still wins
        assert store.policy("stage") is None


def test_fast_win_bonus_only_rewards_wins():
    import numpy as np

    from catan.agents.baselines import GreedyAgent
    from catan.env.gym_single import CatanSingleAgentEnv

    def episode(seed, bonus):  # greedy learner on even seeds (wins), random on odd (doesn't)
        env = CatanSingleAgentEnv(
            opponents=lambda: [make_agent("random") for _ in range(3)],
            config=QUICK,
            learner_seat=0,
            fast_win_bonus=bonus,
            fast_win_turns=100,
        )
        _, info = env.reset(seed=seed)
        rng, rewards, greedy = np.random.default_rng(seed), [], GreedyAgent()
        while True:
            if seed % 2 == 0:
                a = greedy.act_state(env.game.state, info["action_mask"], 0)
            else:
                a = int(rng.choice(np.flatnonzero(info["action_mask"])))
            _, r, term, trunc, info = env.step(a)
            rewards.append(r)
            if term or trunc:
                return rewards, info

    seen = set()
    for seed in range(40):
        plain, info = episode(seed, 0.0)
        bonus, _ = episode(seed, 0.5)
        extra = 0.5 * max(0.0, 1 - info["turns"] / 100) if info.get("winner") == 0 else 0.0
        assert plain[:-1] == bonus[:-1]
        assert bonus[-1] - plain[-1] == pytest.approx(extra)
        seen.add(info.get("winner") == 0)
        if seen == {True, False}:
            break
    assert seen == {True, False}


def test_league_loads_stored_policy_opponents_once(tmp_path):
    import numpy as np

    from catan.agents.learned.ppo.agent import PPOAgent
    from catan.agents.learned.ppo.league import League
    from catan.config.models import LeagueConfig

    with Store(tmp_path / "catan.db") as store:
        manifest, _ = save_policy(PPOAgent([32]), tmp_path, name="v1-run", store=store)
    cfg = LeagueConfig(fixed_opponents=["policy:v1", "random", "random"], lineup=True)
    league = League(cfg, [64], np.random.default_rng(0), tmp_path)
    frozen = [a for _ in range(4) for a in league.sample_opponents() if isinstance(a, PPOAgent)]
    assert len(frozen) == 4 and len({id(a) for a in frozen}) == 4  # a fresh agent per game
    assert len({id(a.model) for a in frozen}) == 1  # sharing one frozen model
    assert frozen[0].hidden == [32] and frozen[0].spec == f"policy:{manifest['id']}"
    assert not any(p.requires_grad for p in frozen[0].model.parameters())


@pytest.mark.slow
def test_ppo_stage_chain_against_frozen_previous_stage(tmp_path):
    from catan.agents.learned.ppo.trainer import train

    def stage(name, opponents, init=None):
        cfg = {
            "name": name,
            "data_dir": str(tmp_path),
            "init_policy": init,
            "rules": {"max_turns_per_game": 12},
            "ppo": {
                "total_steps": 128,
                "rollout_steps": 128,
                "minibatch": 64,
                "epochs": 1,
                "hidden": [32],
                "num_envs": 2,
                "fast_win_bonus": 0.5,
            },
            "league": {"fixed_opponents": opponents, "lineup": True, "self_play_prob": 0},
            "eval_every": 10_000,
            "eval_games": 2,
            "eval_opponents": opponents,
        }
        return train(TrainConfig.model_validate(cfg), log=lambda *_: None)

    first = stage("s1", ["random", "random", "random"])
    second = stage("s2", ["policy:s1", "policy:s1", "random"], init="s1")
    with Store(tmp_path / "catan.db") as store:
        assert store.policy(second["policy_id"])["parent_id"] == first["policy_id"]
        rows = [
            json.loads(line)
            for line in (tmp_path / "runs" / second["run_id"] / "metrics.jsonl").open()
        ]
    assert rows[-1]["turns_100"] is not None and "eval_mean_turns" in second


@pytest.mark.slow
def test_checkpoints_and_curve(tmp_path):
    from catan.agents.learned.ppo.trainer import train
    from catan.sim.curve import evaluate_curve, run_checkpoints

    def run(name, init=None, checkpoint_every=None):
        cfg = {
            "name": name,
            "data_dir": str(tmp_path),
            "init_policy": init,
            "checkpoint_every": checkpoint_every,
            "rules": {"max_turns_per_game": 12},
            "ppo": {
                "total_steps": 192,
                "rollout_steps": 64,
                "minibatch": 32,
                "epochs": 1,
                "hidden": [32],
            },
            "league": {"fixed_opponents": ["random"], "self_play_prob": 0},
            "eval_every": 10_000,
            "eval_games": 2,
        }
        return train(TrainConfig.model_validate(cfg), log=lambda *_: None)

    first = run("a")
    second = run("b", init="a", checkpoint_every=64)
    with Store(tmp_path / "catan.db") as store:
        run_id, ckpts = run_checkpoints(store, "b")
        assert store.policy("b")["id"] == second["policy_id"]  # checkpoints don't shadow the final
    assert run_id == second["run_id"]
    assert [c.label for c in ckpts] == ["pre", "64", "128", "final"]  # 192 == final, skipped
    assert ckpts[0].policy_id == first["policy_id"] and ckpts[-1].policy_id == second["policy_id"]
    lineups = {"rand": ["random"] * 3, "mixed": ["random", "random", "greedy"]}
    rows = evaluate_curve(ckpts, lineups, games=4, workers=1, rules=QUICK, data_dir=tmp_path)
    assert len(rows) == len(ckpts) * 2 and all(r["games"] == 4 for r in rows)
    for r in rows:
        assert 1 <= r["mean_rank"] <= 4 and r["win_ci95"][0] <= r["win_rate"] <= r["win_ci95"][1]
    argmax = evaluate_curve(
        ckpts[:1],
        {"rand": ["random"] * 3},
        games=2,
        workers=1,
        rules=QUICK,
        data_dir=tmp_path,
        deterministic=True,
    )
    assert argmax[0]["games"] == 2


def test_explained_variance_of_rollout():
    import numpy as np

    from catan.agents.learned.ppo.buffer import RolloutBuffer
    from catan.agents.learned.ppo.trainer import _rollout_stats

    buf = RolloutBuffer(4, 2, 2)
    for v, r in [(0.0, 1.0), (0.0, -1.0), (0.0, 1.0), (0.0, -1.0)]:
        buf.add(np.zeros(2), np.ones(2), 0, 0.0, v, r, 1.0)
    buf.compute_returns({0: 0.0}, 0.99, 0.95)
    assert _rollout_stats(buf)["explained_var"] == pytest.approx(0.0)  # predicts nothing
    buf.values[:4] = buf.returns
    assert _rollout_stats(buf)["explained_var"] == pytest.approx(1.0)  # predicts perfectly


@pytest.mark.slow
def test_target_kl_stops_epochs_early_and_lr_decays(tmp_path):
    from catan.agents.learned.ppo.trainer import train

    def run(**ppo):
        cfg = {
            "data_dir": str(tmp_path),
            "rules": {"max_turns_per_game": 12},
            "ppo": {
                "total_steps": 192,
                "rollout_steps": 64,
                "minibatch": 16,
                "epochs": 4,
                "hidden": [32],
                "lr": 1e-2,
                **ppo,
            },
            "league": {"fixed_opponents": ["random"], "self_play_prob": 0},
            "eval_every": 10_000,
            "eval_games": 1,
        }
        result = train(TrainConfig.model_validate(cfg), log=lambda *_: None)
        path = tmp_path / "runs" / result["run_id"] / "metrics.jsonl"
        return [json.loads(line) for line in path.open()]

    full = run()
    assert all(r["epochs_run"] == 4 and r["lr"] == 1e-2 for r in full)
    early = run(target_kl=1e-9, lr_schedule="linear")
    assert all(r["epochs_run"] == 1 for r in early)
    assert [r["lr"] for r in early] == pytest.approx([1e-2, 1e-2 * 2 / 3, 1e-2 / 3])


def test_health_summary_flags_warning_signs():
    from catan.sim.health import summarize

    rows = [
        {"steps": 100 * i, "explained_var": 0.5 if i < 5 else 0.02, "approx_kl": 0.01}
        for i in range(1, 11)
    ]
    windows = summarize(rows, windows=2)
    assert [w["steps"] for w in windows] == [500, 1000]
    assert windows[0]["warnings"] == []
    assert any("explained variance" in m for m in windows[1]["warnings"])
    assert windows[0]["clip_frac"] is None  # older runs lack some columns


def _resume_cfg(tmp_path, total_steps, **ppo):
    return TrainConfig.model_validate(
        {
            "name": "r",
            "data_dir": str(tmp_path),
            "checkpoint_every": 64,
            "rules": {"max_turns_per_game": 30},
            "ppo": {
                "total_steps": total_steps,
                "rollout_steps": 64,
                "minibatch": 32,
                "epochs": 1,
                "hidden": [32],
                **ppo,
            },
            "league": {"fixed_opponents": ["random"], "snapshot_every": 64, "max_snapshots": 2},
            "eval_every": 100,
            "eval_games": 1,
        }
    )


def _rows(tmp_path, run_id):
    return [json.loads(line) for line in (tmp_path / "runs" / run_id / "metrics.jsonl").open()]


@pytest.mark.slow
def test_resume_single_env_matches_uninterrupted_run(tmp_path):
    import torch

    from catan.agents.learned.ppo.trainer import train

    quiet = lambda *_: None  # noqa: E731
    straight = train(_resume_cfg(tmp_path / "a", 256), log=quiet)
    first = train(_resume_cfg(tmp_path / "b", 128), log=quiet)  # "stopped" at 128
    resumed = train(_resume_cfg(tmp_path / "b", 256), log=quiet, resume=first["run_id"])
    assert resumed["run_id"] == first["run_id"]
    a = load_policy(straight["policy_path"]).model.state_dict()
    b = load_policy(resumed["policy_path"]).model.state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)
    drop = {"sps"}
    rows_a = [
        {k: v for k, v in r.items() if k not in drop}
        for r in _rows(tmp_path / "a", straight["run_id"])
    ]
    rows_b = [
        {k: v for k, v in r.items() if k not in drop}
        for r in _rows(tmp_path / "b", first["run_id"])
    ]
    assert rows_a == rows_b  # one row per update, none duplicated across the resume
    with Store(tmp_path / "b" / "catan.db") as store:
        assert store.find_run(first["run_id"])["status"] == "done"


@pytest.mark.slow
def test_resume_parallel_keeps_snapshot_pool(tmp_path):
    from catan.agents.learned.ppo.trainer import train

    quiet = lambda *_: None  # noqa: E731
    first = train(_resume_cfg(tmp_path, 128, num_envs=2), log=quiet)
    resumed = train(_resume_cfg(tmp_path, 256, num_envs=2), log=quiet, resume="r")
    rows = _rows(tmp_path, first["run_id"])
    assert resumed["run_id"] == first["run_id"]
    assert rows[-1]["steps"] >= 256 and rows[-1]["snapshots"] == 2
    assert [r["steps"] for r in rows] == sorted({r["steps"] for r in rows})
    with pytest.raises(FileNotFoundError):
        cfg = _resume_cfg(tmp_path / "none", 64).model_copy(update={"checkpoint_every": None})
        run = train(cfg, log=quiet)["run_id"]
        train(cfg, log=quiet, resume=run)


def test_policy_argmax_spec(tmp_path):
    from catan.agents.learned.ppo.agent import PPOAgent

    with Store(tmp_path / "catan.db") as store:
        manifest, _ = save_policy(PPOAgent([32]), tmp_path, name="p", store=store)
    assert not make_agent(f"policy:{manifest['id']}", tmp_path).deterministic
    assert make_agent(f"policy:{manifest['id']}#argmax", tmp_path).deterministic
    with pytest.raises(ValueError):
        make_agent(f"policy:{manifest['id']}#greedy", tmp_path)


def test_rate_against_fixed_moves_only_the_candidate():
    from catan.sim.ratings import DEFAULT, rate_against_fixed
    from catan.sim.runner import GameRecord

    fixed = {"a": {"mu": 30.0, "sigma": 2.0}, "b": {"mu": 20.0, "sigma": 2.0}}
    snapshot = json.dumps(fixed)

    def game(ranks):
        return GameRecord(
            0, {}, ["cand", "a", "b", "b"], 0, 50, False, [10, 5, 4, 3], ranks, 1, 0.0
        )

    strong = rate_against_fixed(fixed, "cand", [game([1, 2, 3, 4])] * 30)
    weak = rate_against_fixed(fixed, "cand", [game([4, 1, 2, 3])] * 30)
    assert strong["mu"] > 30 > 20 > weak["mu"] and strong["games"] == 30
    assert strong["sigma"] < DEFAULT["sigma"]
    assert json.dumps(fixed) == snapshot  # references untouched


@pytest.mark.slow
def test_reference_calibrate_and_rate(tmp_path):
    from catan.config import ReferenceConfig
    from catan.sim import reference

    cfg = ReferenceConfig(
        agents=["random", "greedy"],
        calibration_games=8,
        gauntlet_games=8,
        rules={"max_turns_per_game": 60},
    )
    with pytest.raises(FileNotFoundError):
        reference.rate("greedy:epsilon=0.5", cfg, workers=1, data_dir=tmp_path)
    cal = reference.calibrate(cfg, workers=1, data_dir=tmp_path)
    assert set(cal["ratings"]) == {"random", "greedy"}
    out = reference.rate("greedy:epsilon=0.5", cfg, workers=1, data_dir=tmp_path)
    assert out["games"] == 8 and 1 <= out["mean_rank"] <= 4
    assert out["conservative"] == pytest.approx(out["mu"] - 3 * out["sigma"])
    assert reference.load_reference(cfg, tmp_path)["ratings"] == cal["ratings"]  # unchanged
    with pytest.raises(ValueError):
        reference.rate("greedy", cfg, workers=1, data_dir=tmp_path)  # a reference itself


def test_compare_groups_mean_and_interval():
    from catan.sim.compare import group_summary, mean_ci

    mean, lo, hi = mean_ci([1.0, 2.0, 3.0])
    assert mean == 2.0 and lo == pytest.approx(2 - 4.303 / 3**0.5) and hi > 3
    assert mean_ci([5.0])[0] == 5.0 and mean_ci([5.0])[1] != mean_ci([5.0])[1]  # no interval
    rows = [
        {
            "policy_id": pid,
            "lineup": "g",
            "vp_margin": m,
            "mean_rank": 2.0,
            "win_rate": 0.1,
            "mean_vp": 5.0,
        }
        for pid, m in [("a", -1.0), ("b", -2.0), ("c", 0.5)]
    ]
    out = group_summary(rows, {"x": ["a", "b"], "y": ["c"]})
    assert [(r["group"], r["n"]) for r in out] == [("x", 2), ("y", 1)]
    assert out[0]["vp_margin"][0] == -1.5 and out[0]["vp_margin_each"] == [-1.0, -2.0]


def test_league_latest_policy_seats():
    import numpy as np

    from catan.agents.learned.ppo.agent import PPOAgent
    from catan.agents.learned.ppo.league import League
    from catan.agents.learned.ppo.model import ActorCritic
    from catan.config.models import LeagueConfig

    current = ActorCritic([32])
    always = League(
        LeagueConfig(fixed_opponents=["random"], latest_prob=1.0), [32], np.random.default_rng(0)
    )
    assert all(a.name == "random" for a in always.sample_opponents())  # not set yet: fixed
    always.latest = current
    seats = always.sample_opponents()
    assert all(isinstance(a, PPOAgent) and a.model is current for a in seats)

    mixed = League(
        LeagueConfig(fixed_opponents=["random"], latest_prob=0.5, self_play_prob=0),
        [32],
        np.random.default_rng(0),
    )
    mixed.latest = current
    kinds = [a.name for _ in range(200) for a in mixed.sample_opponents()]
    assert 0.4 < kinds.count("ppo") / len(kinds) < 0.6


def _actors(games, seed=3, **league):
    from catan.agents.learned.ppo.model import ActorCritic
    from catan.agents.learned.ppo.workers import _Actor, _BatchActor
    from catan.config.models import LeagueConfig

    cfg = LeagueConfig(
        **{
            "fixed_opponents": ["random", "greedy:epsilon=0.5"],
            "latest_prob": 0.4,
            "self_play_prob": 0.5,
            **league,
        }
    )
    args = (QUICK, {"vp_reward_scale": 0.05}, 0.995, cfg, [32], seed, "data")
    actor = _Actor(*args) if games == 0 else _BatchActor(games, *args)
    actor.league.add_snapshot(ActorCritic([32]))
    return actor


@pytest.mark.slow
def test_batched_actor_with_one_game_reproduces_unbatched_actor():
    """Same seeds, same RNG streams, same order of play: pausing for network opponents and
    resuming must not change a single transition. (Run one after the other: both use torch's
    global RNG, which each worker process has to itself.)"""
    import numpy as np

    def run(games):
        actor = _actors(games)
        return [actor.collect(600) if games == 0 else actor.collect(600)[0][1] for _ in range(4)]

    plain, batched = run(0), run(1)
    for a, b in zip(plain, batched, strict=True):
        for f in ("obs", "masks", "actions", "logprobs", "values", "rewards", "dones"):
            assert np.array_equal(a[f], b[f]), f
        assert a["episodes"] == b["episodes"] and a["last_value"] == b["last_value"]
    assert sum(len(seg["episodes"]) for seg in plain) >= 2  # covers game end + reset mid-tick


@pytest.mark.slow
def test_batched_actor_invariants_with_several_games():
    import numpy as np

    actor = _actors(4, latest_prob=0.8)
    episodes = 0
    for _ in range(10):
        segs = actor.collect(300)
        assert [k for k, _ in segs] == [0, 1, 2, 3]
        assert 300 <= sum(len(s["actions"]) for _, s in segs) < 300 + 4  # one round of overshoot
        for _, s in segs:
            assert (
                sum(s["actions"]) >= 0
                and s["masks"][np.arange(len(s["actions"])), s["actions"]].all()
            )
            assert int(s["dones"].sum()) == len(s["episodes"])
            for won, ret, turns in s["episodes"]:
                assert turns > 0 and (won == (ret > 0.5) or not won)
            episodes += len(s["episodes"])
    assert episodes > 0


@pytest.mark.slow
def test_ppo_train_with_games_per_worker(tmp_path):
    from catan.agents.learned.ppo.trainer import train

    cfg = TrainConfig.model_validate(
        {
            "data_dir": str(tmp_path),
            "rules": {"max_turns_per_game": 12},
            "ppo": {
                "total_steps": 512,
                "rollout_steps": 256,
                "minibatch": 64,
                "epochs": 1,
                "hidden": [32],
                "num_envs": 2,
                "games_per_worker": 3,
            },
            "league": {"fixed_opponents": ["random"], "latest_prob": 0.5, "snapshot_every": 200},
            "eval_every": 10_000,
            "eval_games": 2,
        }
    )
    result = train(cfg, log=lambda *_: None)
    rows = _rows(tmp_path, result["run_id"])
    assert rows[-1]["steps"] >= 512 and rows[-1]["episodes"] > 0
    with pytest.raises(ValueError):
        TrainConfig.model_validate({"ppo": {"games_per_worker": 2}})  # needs num_envs > 1

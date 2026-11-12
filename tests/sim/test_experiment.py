"""The experiment runner (``catan experiment``) and its dashboard."""

import json
import time

import pytest
import yaml

from catan.config import ExperimentConfig, load_config
from catan.sim import dashboard, experiment
from catan.sim.curve import Checkpoint


def _train_cfg(name, seed, total_steps=256, checkpoint_every=128):
    return {
        "name": name,
        "seed": seed,
        "data_dir": "data",
        "checkpoint_every": checkpoint_every,
        "rules": {"max_turns_per_game": 12},
        "ppo": {
            "total_steps": total_steps,
            "rollout_steps": 64,
            "minibatch": 32,
            "epochs": 1,
            "hidden": [32],
        },
        "league": {"fixed_opponents": ["random"], "self_play_prob": 0},
        "eval_every": 10_000,
        "eval_games": 1,
    }


def _setup(tmp_path, analysis=None, **run_kw):
    """An experiment folder in tmp_path (used as the working directory) with two runs."""
    exp_dir = tmp_path / "experiments" / "vtest"
    exp_dir.mkdir(parents=True)
    for i in (1, 2):
        cfg = _train_cfg(f"vtest_s{i}", 70 + i, **run_kw)
        (exp_dir / f"s{i}.yaml").write_text(yaml.safe_dump(cfg))
    manifest = {
        "name": "vtest",
        "description": "two tiny runs",
        "comparison": "nothing; a test",
        "runs": ["s1.yaml", "s2.yaml"],
        "analysis": analysis
        or {
            "curve": {"every": 128, "games": 2, "lineups": {"rand": ["random"] * 3}},
            "rate": None,
            "compare": {
                "groups": {"self": "self", "first": ["vtest_s1"]},
                "lineups": {"rand": ["random"] * 3},
                "games": 2,
            },
            "workers": 1,
        },
    }
    path = exp_dir / "experiment.yaml"
    path.write_text(yaml.safe_dump(manifest))
    return path


def _wait(exp_dir, until, timeout=240):
    t = time.time()
    while time.time() - t < timeout:
        state = experiment.read_state(exp_dir)
        if until(state):
            return state
        time.sleep(0.5)
    log = (exp_dir / "driver.log").read_text() if (exp_dir / "driver.log").exists() else ""
    raise AssertionError(f"timed out; state={experiment.read_state(exp_dir)}\n{log}")


def test_manifest_validation(tmp_path):
    path = _setup(tmp_path)
    cfg = load_config(path, ExperimentConfig)
    assert cfg.runs == ["s1.yaml", "s2.yaml"] and cfg.analysis.health
    exp = experiment.Experiment(path, tmp_path)
    assert exp.names == ["vtest_s1", "vtest_s2"]
    assert exp.dir == tmp_path / "data" / "experiments" / "vtest"
    with pytest.raises(ValueError):
        ExperimentConfig(name="x", runs=[])
    with pytest.raises(ValueError):  # unknown key
        ExperimentConfig.model_validate({"name": "x", "runs": ["a"], "typo": 1})


def test_pick_checkpoints():
    ckpts = [Checkpoint("pre", "p", 0)]
    ckpts += [Checkpoint(str(s), f"c{s}", s) for s in range(1_000_000, 10_000_001, 1_000_000)]
    ckpts += [Checkpoint("final", "f", 10_000_500)]
    picked = [c.label for c in experiment.pick_checkpoints(ckpts, 2_000_000)]
    assert picked[0] == "pre" and picked[-1] == "final"
    assert picked[1:-1] == ["1000000", "2000000", "4000000", "6000000", "8000000", "10000000"]


def test_dashboard_snapshot_and_render(tmp_path):
    from rich.console import Console

    path = _setup(tmp_path)
    exp = experiment.Experiment(path, tmp_path)
    exp.dir.mkdir(parents=True)
    run_dir = exp.data_dir / "runs" / "abc123def456"
    run_dir.mkdir(parents=True)
    rows = [
        {
            "steps": 64 * i,
            "sps": 500,
            "sps_recent": 400 + i,
            "explained_var": 0.05,
            "approx_kl": 0.01,
            "clip_frac": 0.1,
            "eval_vp_margin": -6.5,
            "eval_mean_rank": 3.7,
            "eval_win_rate": 0.01,
        }
        for i in range(1, 4)
    ]
    (run_dir / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    state = {
        "name": "vtest",
        "manifest": str(path),
        "cwd": str(tmp_path),
        "created_at": time.time(),
        "phase": "training",
        "runs": {
            "vtest_s1": {"status": "running", "attempts": 2, "run_id": "abc123def456"},
            "vtest_s2": {"status": "pending", "attempts": 0, "run_id": None},
        },
    }
    (exp.dir / "state.json").write_text(json.dumps(state))
    snap = dashboard.snapshot(exp.dir)
    s1, s2 = snap["runs"]
    assert s1["steps"] == 192 and s1["resumes"] == 1 and s1["sps"] == pytest.approx(402)
    assert s1["eta"] > time.time() and s2["steps"] == 0
    console = Console(record=True, width=160)
    console.print(dashboard.render(snap))
    text = console.export_text()
    assert "vtest_s1" in text and "-6.50" in text and "0.05" in text
    assert "vtest_s1 192 " in dashboard.log_line(s1)


@pytest.mark.slow
def test_experiment_runs_to_report(tmp_path):
    path = _setup(tmp_path)
    exp = experiment.Experiment(path, tmp_path)
    state = experiment.launch(path, allow_dirty=True, cwd=tmp_path)
    assert (exp.dir / "code" / "catan" / "cli.py").exists()  # frozen code snapshot
    with pytest.raises(RuntimeError):  # one driver at a time
        experiment.launch(path, allow_dirty=True, cwd=tmp_path)
    state = _wait(exp.dir, lambda s: s.get("phase") in ("done", "failed", "analysis failed"))
    assert state["phase"] == "done", (exp.dir / "driver.log").read_text()
    assert all(e["status"] == "done" and e["run_id"] for e in state["runs"].values())
    report = (exp.dir / "report.md").read_text()
    for section in ("## Speed", "## Health", "### vtest_s1", "## Curve", "## Comparison"):
        assert section in report
    assert "nothing; a test" in report
    with pytest.raises(RuntimeError):  # finished: not relaunched
        experiment.launch(path, allow_dirty=True, cwd=tmp_path)


@pytest.mark.slow
def test_stopped_experiment_resumes_from_checkpoint(tmp_path):
    path = _setup(
        tmp_path,
        analysis={"curve": None, "rate": None, "workers": 1},
        total_steps=4096,
        checkpoint_every=256,
    )
    exp = experiment.Experiment(path, tmp_path)
    experiment.launch(path, allow_dirty=True, cwd=tmp_path)

    def has_checkpoint(state):
        ids = [e.get("run_id") for e in state.get("runs", {}).values()]
        return all(ids) and all(
            (exp.data_dir / "runs" / i / "train_state.pt").exists() for i in ids
        )

    state = _wait(exp.dir, has_checkpoint)
    first_ids = {n: e["run_id"] for n, e in state["runs"].items()}
    assert experiment.stop(exp.dir)
    _wait(exp.dir, lambda s: not experiment.alive(s.get("driver_pid")), timeout=30)
    experiment.launch(path, allow_dirty=True, cwd=tmp_path)
    state = _wait(exp.dir, lambda s: s.get("phase") in ("done", "failed", "analysis failed"))
    assert state["phase"] == "done", (exp.dir / "driver.log").read_text()
    for name, entry in state["runs"].items():
        assert entry["status"] == "done" and entry["attempts"] == 2
        assert entry["run_id"] == first_ids[name]  # resumed the same run, not a new one
        assert "resume" in (exp.dir / f"{name}.log").read_text()

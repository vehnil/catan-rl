"""The experiment runner (``catan experiment``) and its dashboard."""

import json
import time

import pytest
import yaml
from rich.console import Console

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
                "steps": [128],
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
    with pytest.raises(ValueError):  # a group string must be self or experiment:<name>
        ExperimentConfig.model_validate(
            {"name": "x", "runs": ["a"], "analysis": {"compare": {"groups": {"g": "v7"}}}}
        )


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


def test_dashboard_shows_analysis_progress(tmp_path, monkeypatch):
    plan = [
        {"stage": "health", "key": "health", "games": 0},
        {"stage": "curve", "key": "curve r1", "games": 400},
        {"stage": "rate", "key": "rated r1 pre", "games": 100},
        {"stage": "rate", "key": "rated r1 final", "games": 100},
        {"stage": "compare", "key": "compare", "games": 400},
    ]
    monkeypatch.setattr(dashboard, "analysis_plan", lambda exp, state: plan)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() - 60))
    (tmp_path / "driver.log").write_text(
        f"{stamp} phase: analysis\n{stamp} analysis: health done\n"
        f"{stamp} analysis: curve r1 done\n{stamp} analysis: rated r1 pre\n"
    )
    progress = {"task": "rated r1 final", "done": 50, "total": 100, "time": time.time()}
    (tmp_path / experiment.PROGRESS).write_text(json.dumps(progress))
    state = {"phase": "analysis", "runs": {"r1": {"run_id": "x", "status": "done"}}}
    an = dashboard._analysis(None, state, tmp_path)  # type: ignore[arg-type]
    by = {s["stage"]: s for s in an["stages"]}
    assert an["games"] == 1000 and an["played"] == 550
    assert by["ratings"]["done"] == 1 and by["ratings"]["current"] == "rated r1 final"
    assert by["comparison"]["played"] == 0 and an["eta"] > time.time()
    assert "curve r1 done (400 games)" in dashboard.analysis_line("curve r1", time.time(), an)
    console = Console(record=True, width=120)
    console.print(dashboard._render_analysis(an))
    text = console.export_text()
    assert "ratings" in text and "1/2" in text


def test_dashboard_rate_uses_only_the_current_attempt():
    def row(t, steps):
        return {"time": t, "steps": steps}

    before = [row(10 * i, 1000 * i) for i in range(100)]  # 100 steps/s, then stopped
    t0 = before[-1]["time"] + 3600
    burst = [row(t0 + 10 * i, 99_000 + 5000 * i) for i in range(6)]  # fast after resume
    steady = [row(burst[-1]["time"] + 10 * i, burst[-1]["steps"] + 400 * i) for i in range(1, 90)]
    assert dashboard._rate(before + burst) is None  # still warming up: no ETA yet
    assert dashboard._rate(before + burst + steady) == pytest.approx(40)
    assert dashboard._rate(before) == pytest.approx(100)


@pytest.mark.slow
def test_experiment_runs_to_report(tmp_path):
    path = _setup(tmp_path)
    manifest = yaml.safe_load(path.read_text())
    path.write_text(yaml.safe_dump({**manifest, "max_workers": 1}))  # one run at a time
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
    assert "### Final policies" in report and "### At 128 steps" in report
    log = (exp.dir / "driver.log").read_text()
    assert log.index("vtest_s1: done") < log.index("vtest_s2: start")
    points = json.loads((exp.dir / "compare.json").read_text())["points"]
    assert points["final"]["first"] == points["final"]["self"][:1]
    assert points["128"]["self"] != points["final"]["self"]
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
    stopped = experiment.read_state(exp.dir)
    assert stopped["phase"] == "stopped"
    assert all(e["status"] in ("stopped", "done") for e in stopped["runs"].values())
    experiment.launch(path, allow_dirty=True, cwd=tmp_path)
    state = _wait(exp.dir, lambda s: s.get("phase") in ("done", "failed", "analysis failed"))
    assert state["phase"] == "done", (exp.dir / "driver.log").read_text()
    for name, entry in state["runs"].items():
        assert entry["status"] == "done" and entry["attempts"] == 2
        assert entry.get("crashes", 0) == 0  # a stop isn't a crash
        assert entry["run_id"] == first_ids[name]  # resumed the same run, not a new one
        assert "resume" in (exp.dir / f"{name}.log").read_text()


@pytest.mark.slow
def test_analysis_can_be_rerun(tmp_path):
    path = _setup(tmp_path, analysis={"curve": None, "rate": None, "workers": 1})
    exp = experiment.Experiment(path, tmp_path)
    with pytest.raises(RuntimeError):  # never launched
        experiment.reanalyse(path, tmp_path, echo=lambda _: None)
    experiment.launch(path, allow_dirty=True, cwd=tmp_path)
    _wait(exp.dir, lambda s: s.get("phase") == "done" and not experiment.alive(s["driver_pid"]))
    (exp.dir / "report.md").unlink()
    lines: list[str] = []
    report = experiment.reanalyse(path, tmp_path, echo=lines.append)
    assert report.exists() and "## Health" in report.read_text()
    state = experiment.read_state(exp.dir)
    assert state["phase"] == "done" and len(state["analyses"]) == 1
    assert any("phase: analysis" in line for line in lines)
    assert (exp.dir / "driver.log").read_text().count("phase: analysis") == 2


def test_worker_budget_starts_runs_in_order():
    import threading
    from types import SimpleNamespace

    names = ["a", "b", "c", "d"]
    exp = SimpleNamespace(
        names=names,
        cfg=SimpleNamespace(max_workers=4),
        state=lambda: {"runs": {n: {"status": "pending"} for n in names}},
    )
    driver = experiment._Driver(exp)  # type: ignore[arg-type]
    driver.waiting = set(names)
    started, lock, peak = [], threading.Lock(), [0]

    def job(name, n):
        driver._acquire(name, n)
        with lock:
            started.append(name)
            peak[0] = max(peak[0], driver.used)
        time.sleep(0.05)
        driver._release(name, n)

    sizes = {"a": 2, "b": 2, "c": 3, "d": 1}
    threads = [threading.Thread(target=job, args=(n, sizes[n])) for n in reversed(names)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(5)
    assert started == names and peak[0] <= 4 and driver.used == 0


def test_seed_sweep_expands_into_runs(tmp_path):
    exp_dir = tmp_path / "experiments" / "vsweep"
    exp_dir.mkdir(parents=True)
    (exp_dir / "arm.yaml").write_text(yaml.safe_dump(_train_cfg("vsweep_arm", 1)))
    (exp_dir / "other.yaml").write_text(yaml.safe_dump(_train_cfg("vsweep_other", 9)))
    manifest = {
        "name": "vsweep",
        "runs": [{"config": "arm.yaml", "seeds": [50, 51, 52]}, "other.yaml"],
        "analysis": {"curve": None, "rate": None},
    }
    path = exp_dir / "experiment.yaml"
    path.write_text(yaml.safe_dump(manifest))
    exp = experiment.Experiment(path, tmp_path)
    assert exp.names == ["vsweep_arm_s1", "vsweep_arm_s2", "vsweep_arm_s3", "vsweep_other"]
    assert [c.seed for c in exp.train_cfgs] == [50, 51, 52, 9]
    exp.write_generated()
    written = load_config(exp.run_paths[1], experiment.TrainConfig)
    assert written.name == "vsweep_arm_s2" and written.seed == 51
    assert written.ppo == exp.train_cfgs[1].ppo
    # Once written, the files are what trains: editing the base config doesn't change them.
    (exp_dir / "arm.yaml").write_text(yaml.safe_dump(_train_cfg("vsweep_arm", 1, total_steps=999)))
    again = experiment.Experiment(path, tmp_path)
    assert again.train_cfgs[0].ppo.total_steps == 256 and again.run_paths == exp.run_paths
    with pytest.raises(ValueError):  # a sweep needs seeds
        ExperimentConfig.model_validate({"name": "x", "runs": [{"config": "a", "seeds": []}]})

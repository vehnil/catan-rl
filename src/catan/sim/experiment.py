"""Run an experiment (several training runs, then their analysis) unattended.

``launch`` checks the working tree, snapshots the installed ``catan`` package into
the experiment folder and starts a detached driver (its own process session, kept
awake with ``caffeinate`` on macOS), so closing the terminal or editing the code
doesn't touch a running experiment. The driver trains every run at once, each
under a supervisor that resumes from the last checkpoint after a crash, then runs
the analysis and writes ``report.md``. Launching again continues where it stopped:
finished runs are skipped, live ones are waited on, stopped ones resume.

Everything lives in ``<data_dir>/experiments/<name>/``: ``state.json`` (phase and
per-run status, read by the dashboard), ``driver.log``, one ``<run>.log`` per run,
the ``code/`` snapshot and the analysis outputs.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from catan.config.models import ExperimentConfig, ReferenceConfig, TrainConfig, load_config
from catan.sim.provenance import git_state, provenance

STATE = "state.json"


# ------------------------------------------------------------------ layout


class Experiment:
    """An experiment file plus the folder its state and outputs live in."""

    def __init__(self, manifest: str | Path, cwd: str | Path | None = None) -> None:
        self.manifest = Path(manifest).resolve()
        self.cwd = Path(cwd or Path.cwd()).resolve()
        self.cfg = load_config(self.manifest, ExperimentConfig)
        self.run_paths = [(self.manifest.parent / r).resolve() for r in self.cfg.runs]
        self.train_cfgs = [load_config(p, TrainConfig) for p in self.run_paths]
        dirs = {c.data_dir for c in self.train_cfgs}
        if len(dirs) != 1:
            raise ValueError(f"all runs must share one data_dir, got {sorted(dirs)}")
        names = [c.name for c in self.train_cfgs]
        if len(set(names)) != len(names):
            raise ValueError(f"run names must be distinct, got {names}")
        self.data_dir = (self.cwd / dirs.pop()).resolve()
        self.dir = self.data_dir / "experiments" / self.cfg.name

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.train_cfgs]

    def state(self) -> dict[str, Any]:
        return read_state(self.dir)


def read_state(exp_dir: str | Path) -> dict[str, Any]:
    path = Path(exp_dir) / STATE
    return json.loads(path.read_text()) if path.exists() else {}


def _write_state(exp_dir: Path, state: dict[str, Any]) -> None:
    tmp = exp_dir / (STATE + ".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(exp_dir / STATE)


def alive(pid: int | None) -> bool:
    if not pid:
        return False
    # If it is our own child and has exited, reap it (a zombie still answers kill(pid, 0)).
    with contextlib.suppress(ChildProcessError):
        if os.waitpid(pid, os.WNOHANG)[0] == pid:
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# ------------------------------------------------------------------ launch / stop


def launch(
    manifest: str | Path, allow_dirty: bool = False, cwd: str | Path | None = None
) -> dict[str, Any]:
    """Start (or continue) the experiment in the background; returns its state."""
    exp = Experiment(manifest, cwd)
    exp.dir.mkdir(parents=True, exist_ok=True)
    state = exp.state()
    if state and alive(state.get("driver_pid")):
        raise RuntimeError(f"{exp.cfg.name} is already running (driver pid {state['driver_pid']})")
    if state.get("phase") == "done":
        raise RuntimeError(
            f"{exp.cfg.name} already finished: {exp.dir / 'report.md'}"
            " (`catan experiment analyse` reruns the analysis)"
        )
    code = exp.dir / "code"
    if not state:  # first launch: check the tree, freeze the code
        git = git_state(exp.cwd)
        if git["dirty"] and not allow_dirty:
            changes = "\n  ".join(git["changes"][:20])
            raise RuntimeError(
                f"uncommitted changes (commit them, or pass --allow-dirty):\n  {changes}"
            )
        import catan

        shutil.copytree(Path(catan.__file__).parent, code / "catan", dirs_exist_ok=True)
        state = {
            "name": exp.cfg.name,
            "manifest": str(exp.manifest),
            "cwd": str(exp.cwd),
            "created_at": time.time(),
            "provenance": provenance(exp.cfg.model_dump(), exp.cwd),
            "phase": "starting",
            "runs": {n: {"status": "pending", "attempts": 0, "run_id": None} for n in exp.names},
        }
    state["phase"] = "starting"
    state["launches"] = [*state.get("launches", []), time.time()]
    _write_state(exp.dir, state)
    env = {**os.environ, "PYTHONPATH": str(code)}
    cmd = [sys.executable, "-m", "catan.sim.experiment", "drive", str(exp.manifest), str(exp.cwd)]
    if shutil.which("caffeinate"):
        cmd = ["caffeinate", "-i", *cmd]
    with open(exp.dir / "driver.log", "a") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=exp.cwd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,  # survives the terminal / Claude Code session
        )
    state["driver_pid"] = proc.pid
    _write_state(exp.dir, state)
    return state


def stop(exp_dir: str | Path) -> bool:
    """Stop the driver and every run it started (they share its process group)."""
    state = read_state(exp_dir)
    pid = state.get("driver_pid")
    if not pid or not alive(pid):
        return False
    with contextlib.suppress(ProcessLookupError):
        os.killpg(int(pid), signal.SIGTERM)
    deadline = time.time() + 15
    while alive(pid) and time.time() < deadline:
        time.sleep(0.2)
    state = read_state(exp_dir)  # the driver may have written since
    state["phase"] = "stopped"
    for entry in state["runs"].values():
        if entry["status"] in ("running", "queued"):
            entry["status"], entry["pid"] = "stopped", None
    _write_state(Path(exp_dir), state)
    return True


# ------------------------------------------------------------------ driver


_RUN_ID = re.compile(r"^run ([0-9a-f]{12}): metrics")


def _last_run_id(log: Path) -> str | None:
    found = None
    if log.exists():
        for line in log.read_text().splitlines():
            m = _RUN_ID.match(line)
            if m:
                found = m.group(1)
    return found


class _Driver:
    def __init__(self, exp: Experiment) -> None:
        self.exp = exp
        self.lock = threading.Lock()
        self.state = exp.state()
        # Worker budget (``max_workers``): runs start in file order while their workers fit.
        self.slots = threading.Condition()
        self.used = 0
        self.order = {n: i for i, n in enumerate(exp.names)}
        self.waiting: set[str] = set()

    def _acquire(self, name: str, n: int) -> None:
        cap = self.exp.cfg.max_workers
        if cap is None:
            return
        with self.slots:
            self.waiting.add(name)
            while not (
                min(self.waiting, key=self.order.__getitem__) == name
                and (self.used == 0 or self.used + n <= cap)
            ):
                self.slots.wait()
            self.waiting.discard(name)
            self.used += n
            self.slots.notify_all()

    def _release(self, name: str, n: int | None) -> None:
        if self.exp.cfg.max_workers is None:
            return
        with self.slots:
            self.waiting.discard(name)
            self.used -= n or 0
            self.slots.notify_all()

    def save(self) -> None:
        with self.lock:
            _write_state(self.exp.dir, self.state)

    def log(self, msg: str) -> None:
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)

    def set_phase(self, phase: str) -> None:
        self.state["phase"] = phase
        self.save()
        self.log(f"phase: {phase}")

    def supervise(self, name: str, cfg_path: Path, workers: int) -> None:
        try:
            self._supervise(name, cfg_path, workers)
        finally:
            self._release(name, None)  # never leave a dead run at the front of the queue

    def _supervise(self, name: str, cfg_path: Path, workers: int) -> None:
        entry = self.state["runs"][name]
        log_path = self.exp.dir / f"{name}.log"
        if entry["status"] == "done":
            return
        if alive(entry.get("pid")):  # left running by a previous driver: wait for it
            self.log(f"{name}: waiting for running pid {entry['pid']}")
            with self.slots:
                self.waiting.discard(name)
                self.used += workers
            while alive(entry["pid"]):
                time.sleep(10)
            self._release(name, workers)
            entry["status"] = "done" if self._finished(entry.get("run_id")) else "crashed"
            self.save()
            if entry["status"] == "done":
                return
        # Only crashes seen here count against max_resumes: a stop, or the machine going
        # down, kills the driver too, and relaunching continues without using one up.
        while entry.get("crashes", 0) <= self.exp.cfg.max_resumes:
            if self.exp.cfg.max_workers is not None:
                entry["status"] = "queued"
                self.save()
                self._acquire(name, workers)
            run_id = entry.get("run_id")
            has_state = (
                bool(run_id)
                and (self.exp.data_dir / "runs" / str(run_id) / "train_state.pt").exists()
            )
            args = [sys.executable, "-m", "catan.cli", "train", str(cfg_path)]
            if has_state:
                args += ["--resume", str(run_id)]
            entry["attempts"] += 1
            entry["status"] = "running"
            what = f"resume {run_id}" if has_state else "start"
            self.log(f"{name}: {what} (attempt {entry['attempts']})")
            with open(log_path, "a") as f:
                f.write(f"== {time.ctime()} {what} (attempt {entry['attempts']})\n")
                f.flush()
                proc = subprocess.Popen(args, cwd=self.exp.cwd, stdout=f, stderr=subprocess.STDOUT)
                entry["pid"] = proc.pid
                self.save()
                # The run id appears in the log's first line; record it as soon as it does.
                while proc.poll() is None:
                    time.sleep(2)
                    rid = _last_run_id(log_path)
                    if rid and rid != entry.get("run_id"):
                        entry["run_id"] = rid
                        self.save()
            entry["run_id"] = _last_run_id(log_path) or entry.get("run_id")
            entry["pid"] = None
            if proc.returncode == 0:
                entry["status"] = "done"
                self.save()
                self.log(f"{name}: done")
                self._release(name, workers)
                return
            entry["status"] = "crashed"
            entry["crashes"] = entry.get("crashes", 0) + 1
            self.save()
            self.log(f"{name}: exited with {proc.returncode} (crash {entry['crashes']})")
            self._release(name, workers)
        entry["status"] = "failed"
        self.save()
        self.log(f"{name}: FAILED after {entry['crashes']} crashes")

    def _finished(self, run_id: str | None) -> bool:
        if not run_id:
            return False
        from catan.sim.store import Store

        with Store(self.exp.data_dir / "catan.db") as store:
            return store.find_run(run_id)["status"] == "done"

    def run(self) -> None:
        self.log(f"driver pid {os.getpid()} for {self.exp.cfg.name}")
        if self.state.get("phase") not in ("analysis", "done"):
            self.set_phase("training")
            # Register every unfinished run first, so the queue starts them in file order.
            self.waiting = {n for n in self.exp.names if self.state["runs"][n]["status"] != "done"}
            threads = [
                threading.Thread(target=self.supervise, args=(n, p, c.ppo.num_envs), daemon=True)
                for n, p, c in zip(
                    self.exp.names, self.exp.run_paths, self.exp.train_cfgs, strict=True
                )
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        done = [n for n, e in self.state["runs"].items() if e["status"] == "done"]
        if not done:
            self.set_phase("failed")
            return
        self.set_phase("analysis")
        try:
            analyse(self.exp, self.state, self.log)
        except Exception as e:  # keep the training results; report the failure
            self.log(f"analysis failed: {e!r}")
            self.set_phase("analysis failed")
            raise
        self.set_phase("done")


# ------------------------------------------------------------------ analysis


def _fmt_table(rows: list[dict[str, Any]], cols: list[str]) -> str:
    def cell(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.3f}"
        return "-" if v is None else str(v)

    body = [[cell(r.get(c)) for c in cols] for r in rows]
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    return head + "\n".join("| " + " | ".join(r) + " |" for r in body)


def pick_checkpoints(checkpoints: list[Any], every: int) -> list[Any]:
    """The start, the final policy, and the first checkpoint in each ``every`` steps."""
    out, last = [], -1
    for c in checkpoints:
        bucket = c.steps // every
        if c.label in ("pre", "final") or bucket > last:
            out.append(c)
            if c.label not in ("pre", "final"):
                last = bucket
    return out


PROGRESS = "analysis_progress.json"


def analysis_plan(exp: Experiment, state: dict[str, Any]) -> list[dict[str, Any]]:
    """The analysis tasks in the order ``analyse`` runs them, each with its game count.
    A task's ``key`` is what the driver logs when it finishes ("analysis: <key> done", or
    "analysis: rated <run> <label>" for ratings)."""
    from catan.sim.curve import run_checkpoints
    from catan.sim.store import Store

    a = exp.cfg.analysis
    runs = {n: e["run_id"] for n, e in state["runs"].items() if e["status"] == "done"}
    with Store(exp.data_dir / "catan.db") as store:
        checkpoints = {n: run_checkpoints(store, rid)[1] for n, rid in runs.items()}
    plan: list[dict[str, Any]] = [{"stage": "health", "key": "health", "games": 0}]
    if a.curve is not None:
        for n in runs:
            k = len(pick_checkpoints(checkpoints[n], a.curve.every))
            games = k * len(a.curve.lineups) * a.curve.games
            plan.append({"stage": "curve", "key": f"curve {n}", "games": games})
    if a.rate is not None:
        games = (
            a.rate.games or load_config(exp.cwd / a.rate.reference, ReferenceConfig).gauntlet_games
        )
        for n in runs:
            for c in pick_checkpoints(checkpoints[n], a.rate.every):
                plan.append({"stage": "rate", "key": f"rated {n} {c.label}", "games": games})
    if a.compare is not None:
        ids = {i for p in compare_points(exp, state).values() for m in p.values() for i in m}
        games = len(ids) * len(a.compare.lineups) * a.compare.games
        plan.append({"stage": "compare", "key": "compare", "games": games})
    return plan


class _Progress:
    """Game-level progress of the running analysis task, in ``analysis_progress.json``,
    for the dashboard. Writes at most every couple of seconds. With ``echo``, long tasks
    (1,000+ games) also report each quarter."""

    def __init__(self, exp_dir: Path, echo: Any = None) -> None:
        self.path = exp_dir / PROGRESS
        self.echo = echo
        self.last = 0.0

    def task(self, key: str) -> Any:
        quarter = [0]

        def update(done: int, total: int) -> None:
            now = time.time()
            if done == total or now - self.last >= 2:
                self.last = now
                tmp = self.path.with_suffix(".tmp")
                tmp.write_text(json.dumps({"task": key, "done": done, "total": total, "time": now}))
                tmp.replace(self.path)
            q = 4 * done // max(total, 1)
            if self.echo is not None and total >= 1000 and 0 < q < 4 and q > quarter[0]:
                quarter[0] = q
                self.echo(f"  {key}: {done:,}/{total:,} games")

        update(0, 1)
        return update


def _member_runs(
    exp: Experiment, state: dict[str, Any], group: str | list[str], store: Any
) -> list[tuple[str, str | None]]:
    """(ref, run id or None) for each member of a compare group."""
    if group == "self":
        return [(n, e["run_id"]) for n, e in state["runs"].items() if e["status"] == "done"]
    if isinstance(group, str):  # experiment:<name>
        other = read_state(exp.data_dir / "experiments" / group.removeprefix("experiment:"))
        if not other:
            raise KeyError(f"no experiment {group.removeprefix('experiment:')!r}")
        return [(n, e["run_id"]) for n, e in other["runs"].items() if e["status"] == "done"]
    out: list[tuple[str, str | None]] = []
    for ref in group:
        row = store.conn.execute(  # a training run: newest finished one with that name
            "SELECT id FROM runs WHERE kind = 'train' AND (id = ? OR name = ?)"
            " ORDER BY id = ? DESC, status = 'done' DESC, created_at DESC LIMIT 1",
            (ref, ref, ref),
        ).fetchone()
        out.append((ref, row["id"] if row else None))
    return out


def compare_points(exp: Experiment, state: dict[str, Any]) -> dict[str, dict[str, list[str]]]:
    """Policy ids for each comparison point ("final", then each of ``compare.steps``) and
    group. A run's point is its final policy, or its checkpoint nearest the step count."""
    from catan.sim.curve import run_checkpoints
    from catan.sim.store import Store

    c = exp.cfg.analysis.compare
    assert c is not None
    points: dict[str, dict[str, list[str]]] = {"final": {}, **{str(s): {} for s in c.steps}}
    with Store(exp.data_dir / "catan.db") as store:
        for g, v in c.groups.items():
            for p in points.values():
                p[g] = []
            for ref, run_id in _member_runs(exp, state, v, store):
                if run_id is None:  # a plain policy; for `steps`, use the run it came from
                    pid = _pin(ref, exp.data_dir)
                    points["final"][g].append(pid)
                    if not c.steps:
                        continue
                    run_id = (store.policy(pid) or {}).get("run_id")
                    if run_id is None:
                        raise ValueError(f"group {g!r}: {ref} has no training run for `steps`")
                    ckpts = run_checkpoints(store, run_id)[1]
                else:
                    ckpts = run_checkpoints(store, run_id)[1]
                    points["final"][g].append(
                        next(k.policy_id for k in ckpts if k.label == "final")
                    )
                for s in c.steps:
                    near = min(
                        (k for k in ckpts if k.label != "pre"), key=lambda k: abs(k.steps - s)
                    )
                    if abs(near.steps - s) > 0.1 * s:
                        raise ValueError(f"group {g!r}: {ref} has no checkpoint near {s} steps")
                    points[str(s)][g].append(near.policy_id)
    return points


def _pin(ref: str, data_dir: Path) -> str:
    from catan.sim.artifacts import read_manifest, resolve_policy_path

    return str(read_manifest(resolve_policy_path(ref.removeprefix("policy:"), data_dir))["id"])


def reanalyse(manifest: str | Path, cwd: str | Path | None = None, echo: Any = print) -> Path:
    """Run (or rerun) the analysis of a stopped or finished experiment in this process,
    with the current code: ``catan experiment analyse``. Logs to ``driver.log`` too, so
    ``watch`` follows it."""
    exp = Experiment(manifest, cwd)
    state = exp.state()
    if not state:
        raise RuntimeError(f"{exp.cfg.name} has never been launched")
    if alive(state.get("driver_pid")):
        raise RuntimeError(f"{exp.cfg.name} is running (driver pid {state['driver_pid']})")
    if not any(e["status"] == "done" for e in state["runs"].values()):
        raise RuntimeError(f"{exp.cfg.name} has no finished runs to analyse")

    def log(msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        with open(exp.dir / "driver.log", "a") as f:
            f.write(line + "\n")
        echo(line)

    git = git_state(exp.cwd)
    state.setdefault("analyses", []).append(
        {"time": time.time(), "commit": git.get("commit"), "dirty": git.get("dirty")}
    )
    state["phase"] = "analysis"
    state["driver_pid"] = None
    _write_state(exp.dir, state)
    log("phase: analysis")
    try:
        with contextlib.chdir(exp.cwd):  # stored policy paths are relative to it
            report = analyse(exp, state, log, echo=echo)
    except Exception as e:
        log(f"analysis failed: {e!r}")
        state["phase"] = "analysis failed"
        _write_state(exp.dir, state)
        raise
    state["phase"] = "done"
    _write_state(exp.dir, state)
    log("phase: done")
    return report


def analyse(exp: Experiment, state: dict[str, Any], log: Any = print, echo: Any = None) -> Path:
    """Health, curve, reference rating and comparison for the finished runs -> report.md."""
    from catan.sim.compare import METRICS, group_summary
    from catan.sim.curve import Checkpoint, evaluate_curve, run_checkpoints
    from catan.sim.health import COLUMNS, run_metrics, summarize
    from catan.sim.reference import rate
    from catan.sim.store import Store

    a, out, data = exp.cfg.analysis, exp.dir, exp.data_dir
    progress = _Progress(out, echo)
    runs = {n: e["run_id"] for n, e in state["runs"].items() if e["status"] == "done"}
    md = [f"# {exp.cfg.name}", "", exp.cfg.description, ""]
    if exp.cfg.comparison:
        md += ["**Comparison, set before the run:**", "", exp.cfg.comparison, ""]
    git = state.get("provenance", {}).get("git", {})
    md += [f"Code: `{git.get('commit')}` (dirty: {git.get('dirty')})", ""]
    md += ["| run | run id | status | attempts |", "|---|---|---|---|"]
    md += [
        f"| {n} | {e['run_id']} | {e['status']} | {e['attempts']} |"
        for n, e in state["runs"].items()
    ]

    def flush() -> None:
        (out / "report.md").write_text("\n".join(md) + "\n")

    flush()
    checkpoints: dict[str, list[Any]] = {}
    with Store(data / "catan.db") as store:
        for n, rid in runs.items():
            _, checkpoints[n] = run_checkpoints(store, rid)
        md += [
            "",
            "## Speed",
            "",
            "| run | steps | first / middle / last steps/s |",
            "|---|---|---|",
        ]
        health: list[str] = []
        for n, rid in runs.items():
            _, rows = run_metrics(store, rid, data)
            sps = [r.get("sps_recent", r["sps"]) for r in rows]
            md.append(
                f"| {n} | {rows[-1]['steps']} | "
                f"{sps[0]:.0f} / {sps[len(sps) // 2]:.0f} / {sps[-1]:.0f} |"
            )
            if a.health:
                windows = summarize(rows, 10)
                text = _fmt_table(windows, ["steps", *COLUMNS])
                (out / f"health_{n}.md").write_text(text + "\n")
                warn = [f"{w['steps']}: {m}" for w in windows for m in w["warnings"]]
                health += ["", f"### {n}", "", text, *(f"- {w}" for w in warn)]
    if health:
        md += ["", "## Health", *health]
    flush()
    log("analysis: health done")

    if a.curve is not None:
        md += ["", "## Curve", "", "Cells: VP margin over the best opponent, mean rank, win rate."]
        for n in runs:
            ckpts = pick_checkpoints(checkpoints[n], a.curve.every)
            rows = evaluate_curve(
                ckpts,
                a.curve.lineups,
                a.curve.games,
                a.curve.seed,
                None,
                a.workers,
                data,
                progress=progress.task(f"curve {n}"),
            )
            (out / f"curve_{n}.json").write_text(json.dumps(rows, indent=1))
            table = []
            for c in ckpts:
                row: dict[str, Any] = {"policy": c.label}
                for r in rows:
                    if r["label"] == c.label:
                        row[r["lineup"]] = (
                            f"{r['vp_margin']:+.2f} / {r['mean_rank']:.2f} / {r['win_rate']:.0%}"
                        )
                table.append(row)
            md += ["", f"### {n}", "", _fmt_table(table, ["policy", *a.curve.lineups])]
            flush()
            log(f"analysis: curve {n} done")

    if a.rate is not None:
        ref_cfg = load_config(exp.cwd / a.rate.reference, ReferenceConfig)
        md += ["", "## Reference rating", "", f"Reference set `{ref_cfg.name}`.", ""]
        rated = []
        for n in runs:
            for c in pick_checkpoints(checkpoints[n], a.rate.every):
                r = rate(
                    f"policy:{c.policy_id}",
                    ref_cfg,
                    a.rate.games,
                    a.workers,
                    data,
                    progress=progress.task(f"rated {n} {c.label}"),
                )
                rated.append({"run": n, "policy": c.label, "id": c.policy_id, **r})
                log(f"analysis: rated {n} {c.label}")
        (out / "ratings.json").write_text(json.dumps(rated, indent=1))
        md.append(
            _fmt_table(
                rated, ["run", "policy", "mu", "sigma", "vp_margin", "mean_rank", "win_rate"]
            )
        )
        flush()

    if a.compare is not None:
        points = compare_points(exp, state)
        ids = list(
            dict.fromkeys(i for p in points.values() for members in p.values() for i in members)
        )
        rows = evaluate_curve(
            [Checkpoint(i, i, 0) for i in ids],
            a.compare.lineups,
            a.compare.games,
            a.compare.seed,
            None,
            a.workers,
            data,
            progress=progress.task("compare"),
        )
        (out / "compare.json").write_text(
            json.dumps({"groups": points["final"], "points": points, "rows": rows}, indent=1)
        )

        def ci(t: tuple[float, float, float]) -> str:
            m, lo, hi = t
            return f"{m:.3f}" if lo != lo else f"{m:.3f} [{lo:.3f}, {hi:.3f}]"

        md += ["", "## Comparison (mean and 95% t-interval over each group)"]
        for label, groups in points.items():
            table = [
                {
                    "group": r["group"],
                    "lineup": r["lineup"],
                    "n": r["n"],
                    **{m: ci(r[m]) for m in METRICS},
                }
                for r in group_summary(rows, groups)
            ]
            if len(points) > 1:
                md += [
                    "",
                    "### Final policies" if label == "final" else f"### At {int(label):,} steps",
                ]
            md += ["", _fmt_table(table, ["group", "lineup", "n", *METRICS])]
        flush()
        log("analysis: compare done")
    return out / "report.md"


def main(argv: list[str]) -> None:
    if len(argv) == 3 and argv[0] == "drive":
        _Driver(Experiment(argv[1], argv[2])).run()
    else:
        raise SystemExit("usage: python -m catan.sim.experiment drive <manifest> <cwd>")


if __name__ == "__main__":
    main(sys.argv[1:])

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
        raise RuntimeError(f"{exp.cfg.name} already finished: {exp.dir / 'report.md'}")
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

    def save(self) -> None:
        with self.lock:
            _write_state(self.exp.dir, self.state)

    def log(self, msg: str) -> None:
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)

    def set_phase(self, phase: str) -> None:
        self.state["phase"] = phase
        self.save()
        self.log(f"phase: {phase}")

    def supervise(self, name: str, cfg_path: Path) -> None:
        entry = self.state["runs"][name]
        log_path = self.exp.dir / f"{name}.log"
        if entry["status"] == "done":
            return
        if alive(entry.get("pid")):  # left running by a previous driver: wait for it
            self.log(f"{name}: waiting for running pid {entry['pid']}")
            while alive(entry["pid"]):
                time.sleep(10)
            entry["status"] = "done" if self._finished(entry.get("run_id")) else "crashed"
            self.save()
            if entry["status"] == "done":
                return
        while entry["attempts"] <= self.exp.cfg.max_resumes:
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
                return
            entry["status"] = "crashed"
            self.save()
            self.log(f"{name}: exited with {proc.returncode}")
        entry["status"] = "failed"
        self.save()
        self.log(f"{name}: FAILED after {entry['attempts']} attempts")

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
            threads = [
                threading.Thread(target=self.supervise, args=(n, p), daemon=True)
                for n, p in zip(self.exp.names, self.exp.run_paths, strict=True)
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


def _pin(ref: str, data_dir: Path) -> str:
    from catan.sim.artifacts import read_manifest, resolve_policy_path

    return str(read_manifest(resolve_policy_path(ref.removeprefix("policy:"), data_dir))["id"])


def analyse(exp: Experiment, state: dict[str, Any], log: Any = print) -> Path:
    """Health, curve, reference rating and comparison for the finished runs -> report.md."""
    from catan.sim.compare import METRICS, group_summary
    from catan.sim.curve import Checkpoint, evaluate_curve, run_checkpoints
    from catan.sim.health import COLUMNS, run_metrics, summarize
    from catan.sim.reference import rate
    from catan.sim.store import Store

    a, out, data = exp.cfg.analysis, exp.dir, exp.data_dir
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
    finals: dict[str, str] = {}
    with Store(data / "catan.db") as store:
        for n, rid in runs.items():
            _, checkpoints[n] = run_checkpoints(store, rid)
            finals[n] = next(c.policy_id for c in checkpoints[n] if c.label == "final")
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
                ckpts, a.curve.lineups, a.curve.games, a.curve.seed, None, a.workers, data
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
                r = rate(f"policy:{c.policy_id}", ref_cfg, a.rate.games, a.workers, data)
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
        groups = {
            g: list(finals.values()) if v == "self" else [_pin(x, data) for x in v]
            for g, v in a.compare.groups.items()
        }
        ids = list(dict.fromkeys(i for members in groups.values() for i in members))
        rows = evaluate_curve(
            [Checkpoint(i, i, 0) for i in ids],
            a.compare.lineups,
            a.compare.games,
            a.compare.seed,
            None,
            a.workers,
            data,
        )
        summary = group_summary(rows, groups)
        (out / "compare.json").write_text(json.dumps({"groups": groups, "rows": rows}, indent=1))

        def ci(t: tuple[float, float, float]) -> str:
            m, lo, hi = t
            return f"{m:.3f}" if lo != lo else f"{m:.3f} [{lo:.3f}, {hi:.3f}]"

        table = [
            {
                "group": r["group"],
                "lineup": r["lineup"],
                "n": r["n"],
                **{m: ci(r[m]) for m in METRICS},
            }
            for r in summary
        ]
        md += ["", "## Comparison (mean and 95% t-interval over each group)", ""]
        md.append(_fmt_table(table, ["group", "lineup", "n", *METRICS]))
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

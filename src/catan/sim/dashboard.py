"""Live view of a running experiment (``catan experiment watch`` / ``status``).

Reads only files the experiment already writes (``state.json`` and each run's
``metrics.jsonl``), so watching never affects the runs, and closing the
dashboard leaves them running.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from rich.console import Console, Group
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from catan.sim.experiment import PROGRESS, Experiment, analysis_plan, read_state

LOG_EVERY = 100_000  # one log line per run each time it passes another multiple of this
RECENT = 5  # updates averaged for the current speed, for rows without timestamps
RATE_WINDOW = 900.0  # seconds of recent wall time behind the speed and ETA
WARMUP = 120.0  # skipped after each (re)start: fresh games run fast for the first minute or so
GAP = 300.0  # a longer pause between rows means the run was stopped and resumed


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _rate(rows: list[dict[str, Any]]) -> float | None:
    """Steps/s of wall time (evals included) over the last ``RATE_WINDOW`` seconds of the
    current attempt. Rows before a resume, and the first ``WARMUP`` seconds after it, are
    left out. None until the current attempt has run past its warm-up."""
    timed = [r for r in rows if "time" in r]
    if not timed:
        recent = [float(r["sps_recent"]) for r in rows[-RECENT:] if r.get("sps_recent")]
        return sum(recent) / len(recent) if recent else None
    start = len(timed) - 1
    while start > 0 and timed[start]["time"] - timed[start - 1]["time"] < GAP:
        start -= 1
    attempt = timed[start:]
    end = attempt[-1]["time"]
    window = [
        r for r in attempt if r["time"] >= max(attempt[0]["time"] + WARMUP, end - RATE_WINDOW)
    ]
    if len(window) < 2 or end - window[0]["time"] <= 0:
        return None
    return (window[-1]["steps"] - window[0]["steps"]) / (end - window[0]["time"])


_PLANS: dict[tuple[str, ...], list[dict[str, Any]]] = {}
_STAGES = {"health": "health", "curve": "curves", "rate": "ratings", "compare": "comparison"}


def _driver_log(exp_dir: Path) -> list[tuple[float, str]]:
    """(time, message) for each line of the current analysis in ``driver.log``."""
    path = exp_dir / "driver.log"
    if not path.exists():
        return []
    out: list[tuple[float, str]] = []
    for line in path.read_text().splitlines():
        try:
            when = time.mktime(time.strptime(line[:19], "%Y-%m-%d %H:%M:%S"))
        except ValueError:
            continue
        msg = line[20:]
        if msg == "phase: analysis":
            out = []
        out.append((when, msg))
    return out


def _analysis(exp: Experiment, state: dict[str, Any], exp_dir: Path) -> dict[str, Any] | None:
    """Progress of the analysis phase: per stage and overall, with an ETA."""
    if state.get("phase") not in ("analysis", "done", "analysis failed"):
        return None
    key = (str(exp_dir), *(str(e.get("run_id")) for e in state["runs"].values()))
    if key not in _PLANS:
        _PLANS[key] = analysis_plan(exp, state)
    plan = _PLANS[key]
    lines = _driver_log(exp_dir)
    if not lines:
        return None
    start = lines[0][0]
    finished: dict[str, float] = {}
    for when, msg in lines:
        if msg.startswith("analysis: ") and not msg.startswith("analysis failed"):
            finished[msg.removeprefix("analysis: ").removesuffix(" done")] = when
    current = next((task for task in plan if task["key"] not in finished), None)
    partial = 0
    prog = json.loads((exp_dir / PROGRESS).read_text()) if (exp_dir / PROGRESS).exists() else {}
    if current is not None and prog.get("task") == current["key"] and prog.get("total"):
        partial = round(current["games"] * prog["done"] / prog["total"])
    stages: dict[str, dict[str, Any]] = {}
    for task in plan:
        s = stages.setdefault(
            task["stage"], {"tasks": 0, "done": 0, "games": 0, "played": 0, "current": None}
        )
        s["tasks"] += 1
        s["games"] += task["games"]
        if task["key"] in finished:
            s["done"] += 1
            s["played"] += task["games"]
        elif task is current:
            s["played"] += partial
            s["current"] = task["key"]
    games = sum(task["games"] for task in plan)
    played = sum(s["played"] for s in stages.values())
    now = time.time() if state.get("phase") == "analysis" else (lines[-1][0])
    # Speed over the game-playing part (from the end of the health stage).
    t0 = finished.get("health", start)
    eta = None
    if state.get("phase") == "analysis" and played and now > t0:
        eta = now + (games - played) * (now - t0) / played
    return {
        "stages": [{"stage": _STAGES[k], **v} for k, v in stages.items()],
        "games": games,
        "played": played,
        "elapsed": now - start,
        "eta": eta,
        "finished": [(k, w) for k, w in finished.items()],
        "plan": {task["key"]: task for task in plan},
    }


def _top_processes(n: int = 3) -> list[str]:
    try:
        out = subprocess.run(
            ["ps", "-Ao", "pcpu,comm", "-r"], capture_output=True, text=True, timeout=5
        ).stdout.splitlines()[1 : n + 1]
    except (OSError, subprocess.SubprocessError):
        return []
    return [f"{line.split()[0]}% {Path(line.split(None, 1)[1]).name[:40]}" for line in out]


def snapshot(exp_dir: str | Path) -> dict[str, Any]:
    """Everything the dashboard shows, as plain data."""
    exp_dir = Path(exp_dir)
    state = read_state(exp_dir)
    if not state:
        raise FileNotFoundError(f"no experiment state in {exp_dir}")
    exp = Experiment(state["manifest"], state["cwd"])
    totals = {c.name: c.ppo.total_steps for c in exp.train_cfgs}
    runs = []
    for name, entry in state["runs"].items():
        rows = _rows(exp.data_dir / "runs" / str(entry.get("run_id")) / "metrics.jsonl")
        last = rows[-1] if rows else {}
        sps = _rate(rows)
        steps, total = last.get("steps", 0), totals[name]
        eta = time.time() + (total - steps) / sps if sps and steps < total else None
        runs.append(
            {
                "name": name,
                "status": entry["status"],
                "resumes": max(0, entry["attempts"] - 1),
                "steps": steps,
                "total": total,
                "sps": sps,
                "eta": eta,
                "margin": last.get("eval_vp_margin"),
                "rank": last.get("eval_mean_rank"),
                "win": last.get("eval_win_rate"),
                "ev": last.get("explained_var"),
                "kl": last.get("approx_kl"),
                "clip": last.get("clip_frac"),
            }
        )
    load = os.getloadavg()[0] if hasattr(os, "getloadavg") else None
    cores = os.cpu_count() or 1
    return {
        "name": state["name"],
        "phase": state.get("phase"),
        "started": state.get("created_at"),
        "launched": (state.get("launches") or [state.get("created_at")])[-1],
        "launches": len(state.get("launches", [])),
        "runs": runs,
        "load": load,
        "cores": cores,
        "busy": _top_processes() if load is not None and load > cores else [],
        "report": str(exp_dir / "report.md") if (exp_dir / "report.md").exists() else None,
        "analysis": _analysis(exp, state, exp_dir),
    }


def _steps(n: float) -> str:
    return f"{n / 1e6:.2f}M" if n >= 1e6 else f"{n / 1e3:.0f}k" if n >= 1e3 else str(int(n))


def _num(v: float | None, fmt: str) -> str:
    return "-" if v is None else format(v, fmt)


def _health(v: float | None, fmt: str, bad: bool) -> Text:
    return Text(_num(v, fmt), style="bold red" if v is not None and bad else "")


def render(snap: dict[str, Any]) -> Group:
    elapsed = time.time() - snap["launched"] if snap.get("launched") else 0
    since = ""
    if snap.get("launches", 0) > 1 and snap.get("started"):
        first = (time.time() - snap["started"]) / 3600
        since = f" (launch {snap['launches']}, first {first:.1f} h ago)"
    head = Text.assemble(
        (snap["name"], "bold"),
        f"  phase: {snap['phase']}  elapsed {elapsed / 3600:.1f} h{since}  ",
        (
            f"load {snap['load']:.1f} / {snap['cores']} cores",
            "bold red" if snap["load"] and snap["load"] > snap["cores"] else "",
        )
        if snap["load"] is not None
        else "",
    )
    table = Table(expand=False, pad_edge=False)
    for col in ["run", "status", "progress", "steps", "steps/s", "ETA", "margin", "rank", "win"]:
        table.add_column(
            col, justify="right" if col not in ("run", "status", "progress") else "left"
        )
    for col in ["ev", "kl", "clip", "resumes"]:
        table.add_column(col, justify="right")
    for r in snap["runs"]:
        table.add_row(
            r["name"],
            Text(
                r["status"],
                style={"done": "green", "failed": "bold red", "crashed": "yellow"}.get(
                    r["status"], ""
                ),
            ),
            ProgressBar(total=r["total"], completed=min(r["steps"], r["total"]), width=24),
            f"{_steps(r['steps'])} / {_steps(r['total'])}",
            _num(r["sps"], ".0f"),
            time.strftime("%H:%M", time.localtime(r["eta"])) if r["eta"] else "-",
            _num(r["margin"], "+.2f"),
            _num(r["rank"], ".2f"),
            _num(r["win"], ".0%"),
            _health(r["ev"], ".2f", r["ev"] is not None and r["ev"] < 0.1),
            _health(r["kl"], ".4f", r["kl"] is not None and r["kl"] > 0.05),
            _health(r["clip"], ".2f", r["clip"] is not None and r["clip"] > 0.3),
            str(r["resumes"]),
        )
    parts: list[Any] = [head, table]
    if snap.get("analysis"):
        parts.append(_render_analysis(snap["analysis"]))
    if snap["busy"]:
        parts.append(Text("busiest: " + ", ".join(snap["busy"]), style="yellow"))
    if snap["report"]:
        label = "report" if snap["phase"] == "done" else "report so far"
        parts.append(Text(f"{label}: {snap['report']}", style="green"))
    return Group(*parts)


def _render_analysis(an: dict[str, Any]) -> Group:
    eta = time.strftime("%H:%M", time.localtime(an["eta"])) if an["eta"] else "-"
    pct = an["played"] / an["games"] if an["games"] else 1.0
    head = Text.assemble(
        ("analysis", "bold"),
        f"  {pct:.0%} of {an['games']:,} games  elapsed {an['elapsed'] / 60:.0f} min  ETA {eta}",
    )
    table = Table(expand=False, pad_edge=False)
    for col in ["stage", "progress", "tasks", "games", "now"]:
        table.add_column(col, justify="right" if col in ("tasks", "games") else "left")
    for s in an["stages"]:
        done = s["done"] == s["tasks"]
        table.add_row(
            Text(s["stage"], style="green" if done else ""),
            ProgressBar(
                total=max(s["games"], 1), completed=s["played"] if s["games"] else done, width=24
            ),
            f"{s['done']}/{s['tasks']}",
            f"{s['played']:,}/{s['games']:,}",
            s["current"] or ("done" if done else "-"),
        )
    return Group(head, table)


def analysis_line(key: str, when: float, an: dict[str, Any]) -> str:
    games = an["plan"].get(key, {}).get("games", 0)
    count = f" ({games:,} games)" if games else ""
    pct = an["played"] / an["games"] if an["games"] else 1.0
    return (
        f"{time.strftime('%H:%M', time.localtime(when))} analysis: {key} done{count}  [{pct:.0%}]"
    )


def log_line(r: dict[str, Any]) -> str:
    eta = time.strftime("%H:%M", time.localtime(r["eta"])) if r["eta"] else "-"
    return (
        f"{time.strftime('%H:%M')} {r['name']} {_steps(r['steps'])}  {_num(r['sps'], '.0f')} steps/s"
        f"  ETA {eta}  margin {_num(r['margin'], '+.2f')} rank {_num(r['rank'], '.2f')}"
        f" win {_num(r['win'], '.0%')}  ev {_num(r['ev'], '.2f')} kl {_num(r['kl'], '.4f')}"
    )


def watch(exp_dir: str | Path, interval: float = 10.0, console: Console | None = None) -> None:
    """Redraw every ``interval`` seconds, with a log line per run every 100k steps, until the
    experiment finishes. Ctrl-C closes the view; the experiment keeps running."""
    from rich.live import Live

    console = console or Console()
    logged: dict[str, int] = {}
    snap = snapshot(exp_dir)
    for r in snap["runs"]:
        logged[r["name"]] = r["steps"] // LOG_EVERY
    shown = {k for k, _ in (snap["analysis"] or {}).get("finished", [])}
    announced = snap["analysis"] is not None
    try:
        with Live(render(snap), console=console, refresh_per_second=1) as live:
            while True:
                snap = snapshot(exp_dir)
                for r in snap["runs"]:
                    bucket = r["steps"] // LOG_EVERY
                    if bucket > logged.get(r["name"], -1):
                        logged[r["name"]] = bucket
                        console.print(log_line(r))
                an = snap["analysis"]
                if an and not announced:
                    announced = True
                    console.print(f"{time.strftime('%H:%M')} training done; analysis started")
                for k, when in (an or {}).get("finished", []):
                    if k not in shown:
                        shown.add(k)
                        console.print(analysis_line(k, when, an))
                live.update(render(snap))
                if snap["phase"] in ("done", "failed", "analysis failed"):
                    break
                time.sleep(interval)
    except KeyboardInterrupt:
        console.print("closed the view; the experiment is still running")

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

from catan.sim.experiment import Experiment, read_state

LOG_EVERY = 100_000  # one log line per run each time it passes another multiple of this
RECENT = 5  # updates averaged for the current speed


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


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
        recent = [float(r["sps_recent"]) for r in rows[-RECENT:] if r.get("sps_recent")]
        sps = sum(recent) / len(recent) if recent else last.get("sps")
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
        "runs": runs,
        "load": load,
        "cores": cores,
        "busy": _top_processes() if load is not None and load > cores else [],
        "report": str(exp_dir / "report.md") if (exp_dir / "report.md").exists() else None,
    }


def _steps(n: float) -> str:
    return f"{n / 1e6:.2f}M" if n >= 1e6 else f"{n / 1e3:.0f}k" if n >= 1e3 else str(int(n))


def _num(v: float | None, fmt: str) -> str:
    return "-" if v is None else format(v, fmt)


def _health(v: float | None, fmt: str, bad: bool) -> Text:
    return Text(_num(v, fmt), style="bold red" if v is not None and bad else "")


def render(snap: dict[str, Any]) -> Group:
    elapsed = time.time() - snap["started"] if snap.get("started") else 0
    head = Text.assemble(
        (snap["name"], "bold"),
        f"  phase: {snap['phase']}  elapsed {elapsed / 3600:.1f} h  ",
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
    if snap["busy"]:
        parts.append(Text("busiest: " + ", ".join(snap["busy"]), style="yellow"))
    if snap["report"]:
        parts.append(Text(f"report: {snap['report']}", style="green"))
    return Group(*parts)


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
    try:
        with Live(render(snap), console=console, refresh_per_second=1) as live:
            while True:
                snap = snapshot(exp_dir)
                for r in snap["runs"]:
                    bucket = r["steps"] // LOG_EVERY
                    if bucket > logged.get(r["name"], -1):
                        logged[r["name"]] = bucket
                        console.print(log_line(r))
                live.update(render(snap))
                if snap["phase"] in ("done", "failed", "analysis failed"):
                    break
                time.sleep(interval)
    except KeyboardInterrupt:
        console.print("closed the view; the experiment is still running")

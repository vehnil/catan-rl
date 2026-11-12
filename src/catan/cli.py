"""``catan`` command-line interface."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Catan RL platform.")

AgentsOpt = Annotated[
    str, typer.Option("--agents", "-a", help="Comma-separated agent specs, one per seat (4).")
]
DataOpt = Annotated[
    str, typer.Option("--data-dir", help="Where the DB, policies and replays live.")
]


def _split_agents(agents: str) -> list[str]:
    # Specs can contain commas inside kwargs ("ismcts:iterations=50,rollout_depth=10"),
    # so split on ';' if present, else on ','-separated top-level names.
    parts = [p.strip() for p in (agents.split(";") if ";" in agents else agents.split(","))]
    if len(parts) != 4:
        raise typer.BadParameter("need exactly 4 agents (use ';' to separate specs with kwargs)")
    return parts


def _table(rows: list[dict], cols: list[str]) -> str:
    if not rows:
        return "(empty)"

    def fmt(v: object) -> str:
        if isinstance(v, float):
            return f"{v:.3f}"
        if isinstance(v, tuple | list) and v and isinstance(v[0], float):
            return "[" + ", ".join(f"{x:.2f}" for x in v) + "]"
        return str(v)

    cells = [[fmt(r.get(c, "")) for c in cols] for r in rows]
    widths = [max(len(c), *(len(row[i]) for row in cells)) for i, c in enumerate(cols)]
    line = "  ".join(c.ljust(w) for c, w in zip(cols, widths, strict=True))
    body = ["  ".join(v.ljust(w) for v, w in zip(row, widths, strict=True)) for row in cells]
    return "\n".join([line, "-" * len(line), *body])


def _print_summary(summary: dict) -> None:
    typer.echo(
        f"\ngames={summary['games']} truncated={summary['truncated']} "
        f"mean_turns={summary['mean_turns']:.1f} run={summary.get('run_id', '-')}"
    )
    rows = [{"agent": k, **v} for k, v in summary["agents"].items()]
    typer.echo(
        _table(
            rows, ["agent", "seats", "wins", "win_rate", "win_rate_ci95", "mean_vp", "mean_rank"]
        )
    )


@app.command()
def agents() -> None:
    """List registered agent names."""
    from catan.agents.registry import available

    typer.echo("\n".join(available()))


@app.command()
def play(
    agents: AgentsOpt = "greedy,random,random,random",
    seed: int = 0,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Print every action.")] = False,
    board: str = "random",
    data_dir: DataOpt = "data",
) -> None:
    """Play one game and print the final board."""
    from catan.agents.registry import make_agent
    from catan.engine import RulesConfig, describe
    from catan.engine.render import render
    from catan.sim.replay import rebuild
    from catan.sim.runner import play_game

    specs = _split_agents(agents)
    rec = play_game(
        [make_agent(s, data_dir) for s in specs], seed, RulesConfig(board=board), specs=specs
    )
    if verbose:
        from catan.engine import Game

        g = Game(rec.seed, RulesConfig(**rec.rules))
        for a in rec.actions:
            typer.echo(f"t{g.state.turn:<4} P{g.actor} {describe(a)}")
            g.step(a)
    typer.echo(render(rebuild(rec).state))
    typer.echo(
        f"\nwinner: seat {rec.winner} ({rec.winner_spec}) in {rec.turns} turns, {rec.duration_s:.2f}s"
    )


@app.command()
def sim(
    agents: AgentsOpt = "greedy,random,random,random",
    games: int = 100,
    workers: int = 1,
    seed: int = 0,
    save_replays: bool = False,
    board: str = "random",
    data_dir: DataOpt = "data",
) -> None:
    """Run N games of a fixed lineup (seats rotated) and store the results."""
    from catan.config import RulesModel, TournamentConfig
    from catan.sim.tournament import run_tournament

    cfg = TournamentConfig(
        name="sim",
        format="fixed",
        agents=_split_agents(agents),
        games=games,
        workers=workers,
        seed=seed,
        save_replays=save_replays,
        rules=RulesModel(board=board),
        data_dir=data_dir,
    )
    _print_summary(run_tournament(cfg, progress=_progress))


def _progress(done: int, total: int, rec) -> None:  # noqa: ANN001
    if done == total or done % max(1, total // 20) == 0:
        typer.echo(f"  {done}/{total} games", err=True)


@app.command()
def tournament(config: Path) -> None:
    """Run a tournament described by a YAML file (see experiments/)."""
    from catan.config import TournamentConfig, load_config
    from catan.sim.tournament import run_tournament

    _print_summary(run_tournament(load_config(config, TournamentConfig), progress=_progress))


@app.command()
def train(
    config: Path,
    resume: Annotated[
        str | None,
        typer.Option(help="Continue this stopped run (id or config name) from its last state."),
    ] = None,
) -> None:
    """Train a PPO agent described by a YAML file."""
    from catan.agents.learned.ppo.trainer import train as run_train
    from catan.config import TrainConfig, load_config

    result = run_train(load_config(config, TrainConfig), log=typer.echo, resume=resume)
    typer.echo(json.dumps(result, indent=2, default=str))


@app.command()
def replay(
    path: Path,
    step: Annotated[bool, typer.Option(help="Print every action with the board after it.")] = False,
    seat: Annotated[int | None, typer.Option(help="Render from this seat's point of view.")] = None,
) -> None:
    """Re-simulate a stored replay and verify it reproduces the recorded result."""
    from catan.engine.render import render
    from catan.sim.replay import load_replay, rebuild, step_through, verify

    rec = load_replay(path)
    if step:
        for i, actor, desc, board in step_through(rec, seat):
            typer.echo(f"\n#{i} P{actor} {desc}\n{board}")
    else:
        typer.echo(render(rebuild(rec).state, seat))
    typer.echo(
        f"\nagents={rec.agents} winner={rec.winner} turns={rec.turns} verified={verify(rec)}"
    )


@app.command()
def export(
    policy: str,
    fmt: Annotated[str, typer.Option("--format", "-f", help="json | onnx")] = "json",
    out: Path | None = None,
    data_dir: DataOpt = "data",
) -> None:
    """Export a stored policy (by id, name or path)."""
    from catan.sim.artifacts import export_policy, resolve_policy_path

    typer.echo(export_policy(resolve_policy_path(policy, data_dir), fmt, out))


@app.command()
def timeline(
    path: Path,
    out: Annotated[Path | None, typer.Option("--out", "-o", help="Write here instead of stdout.")] = None,
) -> None:
    """Export a replay as per-step JSON (board, actions, events, state, beliefs)."""
    from catan.sim.replay import load_replay
    from catan.sim.timeline import build_timeline

    text = json.dumps(build_timeline(load_replay(path)))
    if out is None:
        typer.echo(text)
    else:
        out.write_text(text)
        typer.echo(f"wrote {out}")


@app.command()
def view(
    path: Annotated[Path | None, typer.Argument(help="Replay to open (default: the picker).")] = None,
    port: int = 8765,
    open_browser: Annotated[bool, typer.Option("--open/--no-open", help="Open a browser tab.")] = True,
    data_dir: DataOpt = "data",
) -> None:
    """Open the game viewer in a browser."""
    from catan.viewer.server import serve

    try:
        serve(data_dir, path, port, open_browser)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e


@app.command()
def ratings(data_dir: DataOpt = "data") -> None:
    """Show the rating leaderboard (sorted by mu - 3*sigma)."""
    from catan.sim.store import Store

    with Store(Path(data_dir) / "catan.db") as store:
        typer.echo(_table(store.leaderboard(), ["agent", "mu", "sigma", "conservative", "games"]))


@app.command()
def policies(data_dir: DataOpt = "data") -> None:
    """List stored policies."""
    from catan.sim.store import Store

    with Store(Path(data_dir) / "catan.db") as store:
        typer.echo(_table(store.policies(), ["id", "name", "kind", "parent_id", "run_id", "path"]))


@app.command()
def runs(limit: int = 20, data_dir: DataOpt = "data") -> None:
    """List recent runs."""
    from catan.sim.store import Store

    with Store(Path(data_dir) / "catan.db") as store:
        typer.echo(_table(store.runs(limit), ["id", "kind", "name", "status", "created_at"]))


def _pin(spec: str, data_dir: str) -> str:
    """``policy:<ref>`` -> ``policy:<id>``, so reports name exact policies."""
    if not spec.startswith("policy:"):
        return spec
    from catan.sim.artifacts import read_manifest, resolve_policy_path

    ref, hash_, mode = spec.removeprefix("policy:").partition("#")
    return f"policy:{read_manifest(resolve_policy_path(ref, data_dir))['id']}{hash_}{mode}"


def _lineups(vs: list[str], data_dir: str) -> dict[str, list[str]]:
    lineups: dict[str, list[str]] = {}
    for item in vs:
        name, sep, specs = item.partition("=")
        if not sep:
            raise typer.BadParameter(f"--vs needs name=spec,spec,spec, got {item!r}")
        lineups[name.strip()] = [_pin(s.strip(), data_dir) for s in specs.split(",")]
    return lineups


def _cell(r: dict) -> str:
    return f"{r['vp_margin']:+.2f}m {r['mean_rank']:.2f}r {r['win_rate']:.0%}"


@app.command()
def curve(
    run: Annotated[str, typer.Argument(help="Run id, or a training config name (newest run).")],
    vs: Annotated[
        list[str],
        typer.Option("--vs", help="Lineup to play against, as name=spec,spec,spec. Repeatable."),
    ],
    games: int = 200,
    seed: int = 0,
    workers: int = 8,
    deterministic: Annotated[
        bool, typer.Option(help="Play each policy's most likely action (#argmax).")
    ] = False,
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Also write rows as JSON.")
    ] = None,
    data_dir: DataOpt = "data",
) -> None:
    """Evaluate a run's starting policy, checkpoints and final policy against fixed lineups.

    Cells show VP margin over the best opponent (m), mean rank (r) and win rate."""
    from catan.sim.curve import evaluate_curve, run_checkpoints
    from catan.sim.store import Store

    lineups = _lineups(vs, data_dir)
    with Store(Path(data_dir) / "catan.db") as store:
        run_id, checkpoints = run_checkpoints(store, run)
    typer.echo(
        f"run {run_id}: {len(checkpoints)} policies x {len(lineups)} lineups x {games} games"
    )
    for name, lineup in lineups.items():
        typer.echo(f"  {name}: {', '.join(lineup)}")
    rows = evaluate_curve(
        checkpoints, lineups, games, seed, None, workers, data_dir, deterministic=deterministic
    )
    table = []
    for c in checkpoints:
        row: dict[str, object] = {"policy": c.label, "id": c.policy_id}
        for r in rows:
            if r["label"] == c.label:
                row[r["lineup"]] = _cell(r)
        table.append(row)
    typer.echo(_table(table, ["policy", "id", *lineups]))
    if out is not None:
        out.write_text(json.dumps({"run_id": run_id, "lineups": lineups, "rows": rows}, indent=2))
        typer.echo(f"wrote {out}")


@app.command()
def rate(
    candidates: Annotated[
        list[str] | None, typer.Argument(help="Agent specs to rate, e.g. policy:<id>.")
    ] = None,
    reference: Path = Path("experiments/reference.yaml"),
    calibrate: Annotated[
        bool, typer.Option("--calibrate", help="(Re)fix the reference ratings first.")
    ] = False,
    games: Annotated[int | None, typer.Option(help="Gauntlet games per candidate.")] = None,
    workers: int = 8,
    data_dir: DataOpt = "data",
) -> None:
    """Rate agents against a fixed reference set (only the candidate's rating moves)."""
    from catan.config import ReferenceConfig, load_config
    from catan.sim import reference as ref

    cfg = load_config(reference, ReferenceConfig)
    if calibrate:
        cal = ref.calibrate(cfg, workers, data_dir, progress=lambda d, t: _progress(d, t, None))
        rows = [{"agent": k, **v} for k, v in cal["ratings"].items()]
        rows.sort(key=lambda r: -r["mu"])
        typer.echo(f"calibrated {cfg.name} over {cal['games']} games")
        typer.echo(_table(rows, ["agent", "mu", "sigma", "games"]))
    rows = []
    for spec in candidates or []:
        rows.append(
            ref.rate(
                _pin(spec, data_dir), cfg, games, workers, data_dir,
                progress=lambda d, t: _progress(d, t, None),
            )
        )
    if rows:
        cols = ["candidate", "mu", "sigma", "conservative", "games", "vp_margin", "mean_rank"]
        typer.echo(_table(rows, [*cols, "win_rate"]))


@app.command()
def compare(
    group: Annotated[
        list[str],
        typer.Option("--group", help="name=policy,policy,... (e.g. one per seed). Repeatable."),
    ],
    vs: Annotated[
        list[str],
        typer.Option("--vs", help="Lineup to play against, as name=spec,spec,spec. Repeatable."),
    ],
    games: int = 400,
    seed: int = 0,
    workers: int = 8,
    deterministic: bool = False,
    out: Annotated[Path | None, typer.Option("--out", "-o")] = None,
    data_dir: DataOpt = "data",
) -> None:
    """Compare groups of policies on the same games: mean and 95% t-interval per group."""
    from catan.sim.compare import METRICS, group_summary
    from catan.sim.curve import Checkpoint, evaluate_curve

    groups: dict[str, list[str]] = {}
    for item in group:
        name, sep, members = item.partition("=")
        if not sep:
            raise typer.BadParameter(f"--group needs name=policy,policy, got {item!r}")
        groups[name.strip()] = [
            _pin(f"policy:{i.strip().removeprefix('policy:')}", data_dir).removeprefix("policy:")
            for i in members.split(",")
        ]
    lineups = _lineups(vs, data_dir)
    ids = list(dict.fromkeys(i for g in groups.values() for i in g))
    checkpoints = [Checkpoint(i, i, 0) for i in ids]
    rows = evaluate_curve(
        checkpoints, lineups, games, seed, None, workers, data_dir, deterministic=deterministic
    )
    summary = group_summary(rows, groups)

    def fmt(t: tuple[float, float, float]) -> str:
        mean, lo, hi = t
        return f"{mean:.3f}" if lo != lo else f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"

    table = [
        {"group": r["group"], "lineup": r["lineup"], "n": r["n"], **{m: fmt(r[m]) for m in METRICS}}
        for r in summary
    ]
    typer.echo(_table(table, ["group", "lineup", "n", *METRICS]))
    if out is not None:
        out.write_text(json.dumps({"groups": groups, "lineups": lineups, "rows": rows}, indent=2))
        typer.echo(f"wrote {out}")


@app.command()
def health(
    run: Annotated[str, typer.Argument(help="Run id, or a training config name (newest run).")],
    windows: int = 10,
    data_dir: DataOpt = "data",
) -> None:
    """Summarize a training run's PPO health metrics and flag warning signs."""
    from catan.sim.health import COLUMNS, run_metrics, summarize
    from catan.sim.store import Store

    with Store(Path(data_dir) / "catan.db") as store:
        run_id, rows = run_metrics(store, run, data_dir)
    summary = summarize(rows, windows)
    typer.echo(f"run {run_id}: {len(rows)} updates")
    table = [
        {k: ("-" if v is None else v) for k, v in w.items() if k != "warnings"} for w in summary
    ]
    typer.echo(_table(table, ["steps", *COLUMNS]))
    for w in summary:
        for msg in w["warnings"]:
            typer.echo(f"  @ {w['steps']}: {msg}")


experiment_app = typer.Typer(
    no_args_is_help=True, help="Run experiments unattended, and watch them (see experiments/)."
)
app.add_typer(experiment_app, name="experiment")

ExpTarget = Annotated[str, typer.Argument(help="Experiment file (.yaml), or experiment name.")]


def _exp_dir(target: str, data_dir: str) -> Path:
    if target.endswith((".yaml", ".yml")):
        from catan.sim.experiment import Experiment

        return Experiment(target).dir
    return Path(data_dir) / "experiments" / target


@experiment_app.command("run")
def experiment_run(
    manifest: Annotated[Path, typer.Argument(help="Experiment file.")],
    allow_dirty: Annotated[
        bool, typer.Option(help="Run even with uncommitted changes (recorded in provenance).")
    ] = False,
    watch: Annotated[bool, typer.Option(help="Open the live view after launching.")] = True,
) -> None:
    """Start an experiment in the background, or continue a stopped one."""
    from catan.sim import dashboard
    from catan.sim.experiment import Experiment, launch

    try:
        state = launch(manifest, allow_dirty=allow_dirty)
    except RuntimeError as e:
        raise typer.BadParameter(str(e)) from e
    exp_dir = Experiment(manifest).dir
    typer.echo(f"launched {state['name']} (driver pid {state['driver_pid']}) -> {exp_dir}")
    typer.echo("closing this terminal won't stop it; `catan experiment watch` reattaches")
    if watch:
        dashboard.watch(exp_dir)


@experiment_app.command("watch")
def experiment_watch(target: ExpTarget, interval: float = 10.0, data_dir: DataOpt = "data") -> None:
    """Live view: progress, speed, ETA, latest eval and health per run (Ctrl-C to close)."""
    from catan.sim import dashboard

    dashboard.watch(_exp_dir(target, data_dir), interval)


@experiment_app.command("status")
def experiment_status(target: ExpTarget, data_dir: DataOpt = "data") -> None:
    """One-shot view of an experiment."""
    from rich.console import Console

    from catan.sim import dashboard

    Console().print(dashboard.render(dashboard.snapshot(_exp_dir(target, data_dir))))


@experiment_app.command("stop")
def experiment_stop(target: ExpTarget, data_dir: DataOpt = "data") -> None:
    """Stop an experiment and all its runs (`catan experiment run` continues it later)."""
    from catan.sim.experiment import stop

    typer.echo("stopped" if stop(_exp_dir(target, data_dir)) else "not running")


@app.command()
def benchmark(games: int = 50, agent: str = "random", workers: int = 1) -> None:
    """Measure engine + agent throughput (games/min)."""
    from catan.sim.runner import Matchup, run_matchups

    t = time.perf_counter()
    steps = 0
    for rec in run_matchups(
        [Matchup((agent,) * 4, s) for s in range(games)], workers=workers, record_actions=False
    ):
        steps += rec.steps
    dt = time.perf_counter() - t
    typer.echo(
        f"{games} games, {steps} steps in {dt:.1f}s: {60 * games / dt:.0f} games/min, {steps / dt:.0f} steps/s"
    )


if __name__ == "__main__":
    app()

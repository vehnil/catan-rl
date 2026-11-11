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
def train(config: Path) -> None:
    """Train a PPO agent described by a YAML file."""
    from catan.agents.learned.ppo.trainer import train as run_train
    from catan.config import TrainConfig, load_config

    result = run_train(load_config(config, TrainConfig), log=typer.echo)
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


@app.command()
def curve(
    run: Annotated[str, typer.Argument(help="Run id, or a training config name (newest run).")],
    vs: Annotated[
        list[str],
        typer.Option("--vs", help="Lineup to play against, as name=spec,spec,spec. Repeatable."),
    ],
    games: int = 100,
    seed: int = 0,
    workers: int = 8,
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Also write rows as JSON.")
    ] = None,
    data_dir: DataOpt = "data",
) -> None:
    """Evaluate a run's starting policy, checkpoints and final policy against fixed lineups."""
    from catan.sim.artifacts import read_manifest, resolve_policy_path
    from catan.sim.curve import evaluate_curve, run_checkpoints
    from catan.sim.store import Store

    lineups: dict[str, list[str]] = {}
    for item in vs:
        name, sep, specs = item.partition("=")
        if not sep:
            raise typer.BadParameter(f"--vs needs name=spec,spec,spec, got {item!r}")
        pinned = []
        for spec in (s.strip() for s in specs.split(",")):
            if spec.startswith("policy:"):  # pin refs so the report names exact policies
                pid = read_manifest(resolve_policy_path(spec.removeprefix("policy:"), data_dir))
                spec = f"policy:{pid['id']}"
            pinned.append(spec)
        lineups[name.strip()] = pinned
    with Store(Path(data_dir) / "catan.db") as store:
        run_id, checkpoints = run_checkpoints(store, run)
    typer.echo(
        f"run {run_id}: {len(checkpoints)} policies x {len(lineups)} lineups x {games} games"
    )
    for name, lineup in lineups.items():
        typer.echo(f"  {name}: {', '.join(lineup)}")
    rows = evaluate_curve(checkpoints, lineups, games, seed, None, workers, data_dir)
    table = []
    for c in checkpoints:
        row: dict[str, object] = {"policy": c.label, "id": c.policy_id}
        for r in rows:
            if r["label"] == c.label:
                row[r["lineup"]] = f"{r['win_rate']:.0%} {r['mean_vp']:.2f}vp"
        table.append(row)
    typer.echo(_table(table, ["policy", "id", *lineups]))
    if out is not None:
        out.write_text(json.dumps({"run_id": run_id, "lineups": lineups, "rows": rows}, indent=2))
        typer.echo(f"wrote {out}")


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

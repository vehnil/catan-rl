"""Tournament scheduling and execution (results go to the store, replays to disk)."""

from __future__ import annotations

import itertools
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from catan.config.models import TournamentConfig
from catan.engine.constants import NUM_PLAYERS
from catan.sim.ratings import update_ratings, wilson_interval
from catan.sim.replay import save_replay
from catan.sim.runner import GameRecord, Matchup, run_matchups
from catan.sim.store import Store


def _rotations(lineup: tuple[str, ...]) -> list[tuple[str, ...]]:
    """All four cyclic seatings, so nobody keeps the first-placement advantage."""
    return [lineup[i:] + lineup[:i] for i in range(NUM_PLAYERS)]


def schedule_fixed(agents: list[str], games: int, seed: int) -> list[Matchup]:
    rots = _rotations(tuple(agents))
    return [Matchup(rots[g % NUM_PLAYERS], seed + g) for g in range(games)]


def schedule_round_robin(agents: list[str], games: int, seed: int) -> list[Matchup]:
    if len(agents) >= NUM_PLAYERS:
        lineups = list(itertools.combinations(agents, NUM_PLAYERS))
    else:
        lineups = [
            c
            for c in itertools.combinations_with_replacement(agents, NUM_PLAYERS)
            if set(c) == set(agents)
        ]
    seatings = [r for lu in lineups for r in _rotations(lu)]
    return [Matchup(seatings[g % len(seatings)], seed + g) for g in range(games)]


def schedule_gauntlet(candidate: str, pool: list[str], games: int, seed: int) -> list[Matchup]:
    rng = np.random.default_rng(seed)
    out = []
    for g in range(games):
        opp = [str(x) for x in rng.choice(pool, size=NUM_PLAYERS - 1, replace=True)]
        seat = g % NUM_PLAYERS
        out.append(Matchup(tuple(opp[:seat] + [candidate] + opp[seat:]), seed + g))
    return out


def schedule(cfg: TournamentConfig) -> list[Matchup]:
    if cfg.format == "fixed":
        return schedule_fixed(cfg.agents, cfg.games, cfg.seed)
    if cfg.format == "gauntlet":
        assert cfg.candidate
        return schedule_gauntlet(cfg.candidate, cfg.agents, cfg.games, cfg.seed)
    return schedule_round_robin(cfg.agents, cfg.games, cfg.seed)


def summarize(records: list[GameRecord]) -> dict[str, Any]:
    """Per-agent seat-level stats (an agent in two seats counts twice)."""
    stats: dict[str, dict[str, float]] = {}
    for rec in records:
        for seat, spec in enumerate(rec.agents):
            s = stats.setdefault(spec, {"seats": 0, "wins": 0, "vp": 0, "rank": 0})
            s["seats"] += 1
            s["wins"] += seat == rec.winner
            s["vp"] += rec.vp[seat]
            s["rank"] += rec.ranks[seat]
    table: dict[str, dict[str, Any]] = {}
    for spec, s in stats.items():
        lo, hi = wilson_interval(int(s["wins"]), int(s["seats"]))
        table[spec] = {
            "seats": int(s["seats"]),
            "wins": int(s["wins"]),
            "win_rate": s["wins"] / s["seats"],
            "win_rate_ci95": (lo, hi),
            "mean_vp": s["vp"] / s["seats"],
            "mean_rank": s["rank"] / s["seats"],
        }
    n = len(records)
    return {
        "games": n,
        "truncated": sum(r.truncated for r in records),
        "mean_turns": sum(r.turns for r in records) / n if n else 0,
        "agents": dict(sorted(table.items(), key=lambda kv: -float(kv[1]["win_rate"]))),
    }


def run_tournament(
    cfg: TournamentConfig,
    store: Store | None = None,
    progress: Callable[[int, int, GameRecord], None] | None = None,
) -> dict[str, Any]:
    data_dir = Path(cfg.data_dir)
    own_store = store is None
    store = store or Store(data_dir / "catan.db")
    run_id = store.start_run("tournament", cfg.model_dump(), cfg.name)
    matchups = schedule(cfg)
    records: list[GameRecord] = []
    try:
        for i, rec in enumerate(
            run_matchups(
                matchups, cfg.rules.to_rules(), cfg.workers, str(data_dir), cfg.save_replays
            )
        ):
            replay = None
            if cfg.save_replays:
                replay = str(
                    save_replay(rec, data_dir / "replays" / run_id, f"{i:05d}_seed{rec.seed}")
                )
            store.add_match(rec, run_id, replay)
            records.append(rec)
            if progress:
                progress(i + 1, len(matchups), rec)
        summary = summarize(records)
        if cfg.update_ratings:
            agents = {a for r in records for a in r.agents}
            ratings = update_ratings(store.get_ratings(agents), records)
            store.set_ratings(ratings)
            summary["ratings"] = ratings
        summary["run_id"] = run_id
        store.finish_run(run_id, summary)
        return summary
    except BaseException:
        store.finish_run(run_id, {"games_completed": len(records)}, status="failed")
        raise
    finally:
        if own_store:
            store.close()

"""Rate policies on one fixed scale: a reference set of opponents, calibrated once.

1. ``calibrate``: a round-robin among the references fixes their ratings
   (OpenSkill), written to ``data/reference/<name>.json``.
2. ``rate``: a candidate plays a gauntlet against lineups drawn from the
   references; only the candidate's rating moves (``rate_against_fixed``).

Win rate against a strong opponent sits at 0% for a long time; the rating keeps
moving with finishing position against every opponent in the set, so it shows
progress earlier and puts every policy on the same scale.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from catan.config.models import ReferenceConfig
from catan.sim.ratings import rate_against_fixed, update_ratings
from catan.sim.runner import run_matchups
from catan.sim.tournament import schedule_gauntlet, schedule_round_robin


def reference_path(cfg: ReferenceConfig, data_dir: str | Path) -> Path:
    return Path(data_dir) / "reference" / f"{cfg.name}.json"


def calibrate(
    cfg: ReferenceConfig, workers: int = 8, data_dir: str | Path = "data", progress: Any = None
) -> dict[str, Any]:
    """Play the references' round-robin and save their ratings."""
    matchups = schedule_round_robin(cfg.agents, cfg.calibration_games, cfg.seed)
    records = []
    for i, rec in enumerate(
        run_matchups(
            matchups,
            cfg.rules.to_rules(),
            workers=workers,
            data_dir=str(data_dir),
            record_actions=False,
        )
    ):
        records.append(rec)
        if progress is not None:
            progress(i + 1, len(matchups))
    records.sort(key=lambda r: r.seed)  # completion order varies with workers
    out = {
        "name": cfg.name,
        "agents": cfg.agents,
        "games": len(records),
        "seed": cfg.seed,
        "ratings": update_ratings({}, records),
    }
    path = reference_path(cfg, data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    return out


def load_reference(cfg: ReferenceConfig, data_dir: str | Path = "data") -> dict[str, Any]:
    path = reference_path(cfg, data_dir)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run `catan rate --calibrate` first")
    ref = json.loads(path.read_text())
    if ref["agents"] != cfg.agents:
        raise ValueError(f"{path} was calibrated for {ref['agents']}, not {cfg.agents}")
    return ref


def rate(
    candidate: str,
    cfg: ReferenceConfig,
    games: int | None = None,
    workers: int = 8,
    data_dir: str | Path = "data",
    progress: Any = None,
) -> dict[str, Any]:
    """The candidate's rating against the fixed references, plus its gauntlet results."""
    ref = load_reference(cfg, data_dir)
    if candidate in cfg.agents:
        raise ValueError(f"{candidate} is itself a reference")
    matchups = schedule_gauntlet(candidate, cfg.agents, games or cfg.gauntlet_games, cfg.seed)
    records = []
    for i, rec in enumerate(
        run_matchups(
            matchups,
            cfg.rules.to_rules(),
            workers=workers,
            data_dir=str(data_dir),
            record_actions=False,
        )
    ):
        records.append(rec)
        if progress is not None:
            progress(i + 1, len(matchups))
    records.sort(key=lambda r: r.seed)  # same order (and so rating) for any worker count
    r = rate_against_fixed(ref["ratings"], candidate, records)
    seats = [rec.agents.index(candidate) for rec in records]
    n = len(records)
    return {
        "candidate": candidate,
        "mu": r["mu"],
        "sigma": r["sigma"],
        "conservative": r["mu"] - 3 * r["sigma"],
        "games": n,
        "win_rate": sum(rec.winner == s for rec, s in zip(records, seats, strict=True)) / n,
        "mean_rank": sum(rec.ranks[s] for rec, s in zip(records, seats, strict=True)) / n,
        "vp_margin": sum(
            rec.vp[s] - max(v for q, v in enumerate(rec.vp) if q != s)
            for rec, s in zip(records, seats, strict=True)
        )
        / n,
    }

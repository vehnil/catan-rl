"""Multiplayer skill ratings (OpenSkill Plackett-Luce, a TrueSkill-like model) and win-rate CIs."""

from __future__ import annotations

import math
from collections.abc import Iterable

from openskill.models import PlackettLuce

from catan.sim.runner import GameRecord

_MODEL = PlackettLuce()
DEFAULT = {"mu": _MODEL.mu, "sigma": _MODEL.sigma, "games": 0}


def update_ratings(
    ratings: dict[str, dict[str, float]], records: Iterable[GameRecord]
) -> dict[str, dict[str, float]]:
    """Return updated copies of ``ratings`` after the given games.

    If the same agent occupies several seats in one game (self-play), each
    seat is rated as its own team and the agent's rating becomes the mean of
    the seat updates.
    """
    out = {k: dict(v) for k, v in ratings.items()}
    for rec in records:
        teams = []
        for spec in rec.agents:
            r = out.get(spec, DEFAULT)
            teams.append([_MODEL.rating(mu=r["mu"], sigma=r["sigma"], name=spec)])
        new = _MODEL.rate(teams, ranks=list(rec.ranks))
        merged: dict[str, list[tuple[float, float]]] = {}
        for spec, team in zip(rec.agents, new, strict=True):
            merged.setdefault(spec, []).append((team[0].mu, team[0].sigma))
        for spec, vals in merged.items():
            prev = out.get(spec, DEFAULT)
            out[spec] = {
                "mu": sum(v[0] for v in vals) / len(vals),
                "sigma": sum(v[1] for v in vals) / len(vals),
                "games": prev["games"] + 1,
            }
    return out


def wilson_interval(wins: int, games: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a win rate."""
    if games == 0:
        return 0.0, 1.0
    p = wins / games
    denom = 1 + z * z / games
    centre = (p + z * z / (2 * games)) / denom
    half = z * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)

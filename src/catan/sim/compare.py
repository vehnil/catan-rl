"""Compare groups of policies (e.g. one policy per seed, per training recipe).

Every policy plays the same games against each lineup; each group's result is
the mean over its members with a 95% t-interval, so the spread between seeds
is part of the comparison instead of hidden by one lucky seed. With three
seeds the interval is wide; that is the honest answer.
"""

from __future__ import annotations

import math
from typing import Any

# Two-sided 95% Student-t critical values by degrees of freedom.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}

METRICS = ["vp_margin", "mean_rank", "win_rate", "mean_vp"]


def mean_ci(values: list[float]) -> tuple[float, float, float]:
    """(mean, low, high): 95% t-interval over the values; a single value has no interval."""
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, float("nan"), float("nan")
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    half = _T95.get(n - 1, 2.0 if n <= 31 else 1.96) * sd / math.sqrt(n)
    return mean, mean - half, mean + half


def group_summary(
    rows: list[dict[str, Any]], groups: dict[str, list[str]], metrics: list[str] = METRICS
) -> list[dict[str, Any]]:
    """``rows`` from ``evaluate_curve`` (labelled by policy id) -> one row per (group, lineup)."""
    lineups = list(dict.fromkeys(r["lineup"] for r in rows))
    out = []
    for group, ids in groups.items():
        for lineup in lineups:
            members = [r for r in rows if r["lineup"] == lineup and r["policy_id"] in ids]
            row: dict[str, Any] = {"group": group, "lineup": lineup, "n": len(members)}
            for m in metrics:
                row[m] = mean_ci([r[m] for r in members])
                row[f"{m}_each"] = [r[m] for r in members]
            out.append(row)
    return out

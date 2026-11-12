"""Training-health summary of a run's per-update metrics (``runs/<id>/metrics.jsonl``).

PPO can fail quietly: win rate stays flat while the value head predicts nothing
(explained variance near 0, so advantages are noise) or updates are too large
(high KL / clip fraction). This averages the metrics over windows of updates
and flags the usual warning signs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from catan.sim.store import Store

COLUMNS = [
    "win_rate_100",
    "return_100",
    "entropy",
    "explained_var",
    "value_loss",
    "value_mean",
    "approx_kl",
    "clip_frac",
    "grad_norm",
]

# (column, test, message); a window that passes the test gets the message.
WARNINGS = [
    ("explained_var", lambda v: v < 0.1, "explained variance < 0.1: advantages are mostly noise"),
    ("approx_kl", lambda v: v > 0.05, "KL > 0.05: updates may be too large"),
    ("clip_frac", lambda v: v > 0.3, "clip fraction > 0.3: updates may be too large"),
]


def run_metrics(store: Store, run: str, data_dir: str | Path) -> tuple[str, list[dict[str, Any]]]:
    """``run`` is a run id or a config name (newest run with that name)."""
    run_id = store.find_run(run)["id"]
    path = Path(data_dir) / "runs" / run_id / "metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return run_id, rows


def _mean(vals: list[float | None]) -> float | None:
    clean = [v for v in vals if v is not None and not math.isnan(v)]
    return sum(clean) / len(clean) if clean else None


def summarize(rows: list[dict[str, Any]], windows: int = 10) -> list[dict[str, Any]]:
    """Average ``COLUMNS`` over ``windows`` equal runs of updates; each row ends at ``steps``."""
    if not rows:
        return []
    size = max(1, math.ceil(len(rows) / windows))
    out = []
    for i in range(0, len(rows), size):
        chunk = rows[i : i + size]
        w: dict[str, Any] = {"steps": chunk[-1]["steps"]}
        for c in COLUMNS:
            w[c] = _mean([r.get(c) for r in chunk])
        w["warnings"] = [msg for c, test, msg in WARNINGS if w.get(c) is not None and test(w[c])]
        out.append(w)
    return out

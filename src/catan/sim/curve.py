"""Evaluate a training run's checkpoints against fixed lineups, to see overfitting.

For one run: its starting policy ("pre", the warm-start parent, if any), every
checkpoint saved with ``checkpoint_every``, and the final policy ("post"). Each
plays the same games against each named lineup (same seeds and seat rotations
for every policy), so differences between checkpoints come from the policy,
not from the draw. Overfitting shows up as the gap between lineups growing:
better against the training opponents, flat or worse against held-out ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from catan.engine.state import RulesConfig
from catan.sim.artifacts import read_manifest
from catan.sim.runner import run_matchups
from catan.sim.store import Store
from catan.sim.tournament import schedule_fixed


@dataclass(frozen=True)
class Checkpoint:
    label: str  # "pre", "<steps>" or "final"
    policy_id: str
    steps: int


def run_checkpoints(store: Store, run: str) -> tuple[str, list[Checkpoint]]:
    """``run`` is a run id or a training config name (newest run with that name)."""
    run_id = store.find_run(run)["id"]
    rows = [p for p in store.policies() if p["run_id"] == run_id]
    if not rows:
        raise KeyError(f"run {run_id} has no saved policies")
    out: list[Checkpoint] = []
    parent = rows[0]["parent_id"]
    if parent:
        out.append(Checkpoint("pre", parent, 0))
    # A resumed run re-saves checkpoints past its resume point; keep the newest of each.
    newest: dict[tuple[str, int], Checkpoint] = {}
    for p in sorted(rows, key=lambda r: r["created_at"]):
        m = read_manifest(p["path"])
        steps = int(m.get("metrics", {}).get("steps", 0))
        label = str(steps) if m.get("metrics", {}).get("checkpoint") else "final"
        newest[(label, steps)] = Checkpoint(label, p["id"], steps)
    staged = list(newest.values())
    staged.sort(key=lambda c: (c.steps, c.label == "final"))
    final_steps = {c.steps for c in staged if c.label == "final"}
    # The last checkpoint usually lands on the final step too: same weights, so skip it.
    staged = [c for c in staged if c.label == "final" or c.steps not in final_steps]
    return run_id, out + staged


def evaluate_curve(
    checkpoints: list[Checkpoint],
    lineups: dict[str, list[str]],
    games: int = 100,
    seed: int = 0,
    rules: RulesConfig | None = None,
    workers: int = 8,
    data_dir: str | Path = "data",
    progress: Any = None,
) -> list[dict[str, Any]]:
    """One row per (checkpoint, lineup): win rate, mean VP and mean turns over ``games``."""
    for name, lineup in lineups.items():
        if len(lineup) != 3:
            raise ValueError(f"lineup {name!r} needs 3 opponents, got {lineup}")
    matchups, owner = [], {}
    for c in checkpoints:
        cand = f"policy:{c.policy_id}"
        for name, lineup in lineups.items():
            if cand in lineup:
                raise ValueError(f"{cand} also appears in lineup {name!r}")
            for m in schedule_fixed([cand, *lineup], games, seed):
                matchups.append(m)
                owner[(m.specs, m.seed)] = (c, name)
    stats: dict[tuple[str, str], list[tuple[bool, int, int]]] = {}
    for i, rec in enumerate(
        run_matchups(matchups, rules, workers=workers, data_dir=str(data_dir), record_actions=False)
    ):
        c, name = owner[(tuple(rec.agents), rec.seed)]
        seat = rec.agents.index(f"policy:{c.policy_id}")
        stats.setdefault((c.label, name), []).append((rec.winner == seat, rec.vp[seat], rec.turns))
        if progress is not None:
            progress(i + 1, len(matchups))
    rows = []
    for c in checkpoints:
        for name in lineups:
            s = stats[(c.label, name)]
            rows.append(
                {
                    "label": c.label,
                    "steps": c.steps,
                    "policy_id": c.policy_id,
                    "lineup": name,
                    "games": len(s),
                    "win_rate": sum(w for w, _, _ in s) / len(s),
                    "mean_vp": sum(v for _, v, _ in s) / len(s),
                    "mean_turns": sum(t for _, _, t in s) / len(s),
                }
            )
    return rows

"""Replays: a game is fully determined by (seed, rules, actions), so files stay tiny."""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from pathlib import Path

from catan.engine import Game, RulesConfig, describe
from catan.engine.render import render
from catan.sim.runner import GameRecord


def save_replay(rec: GameRecord, directory: str | Path, name: str | None = None) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name or f'game_{rec.seed}'}.json.gz"
    with gzip.open(path, "wt") as f:
        json.dump(rec.to_dict(), f)
    return path


def load_replay(path: str | Path) -> GameRecord:
    with gzip.open(path, "rt") as f:
        return GameRecord(**json.load(f))


def rebuild(rec: GameRecord, upto: int | None = None) -> Game:
    """Re-simulate the game (optionally only the first ``upto`` actions)."""
    actions = rec.actions if upto is None else rec.actions[:upto]
    return Game.replay(rec.seed, RulesConfig(**rec.rules), actions)


def step_through(rec: GameRecord, seat: int | None = None) -> Iterator[tuple[int, int, str, str]]:
    """Yield (step, actor, action description, rendered state after the action)."""
    g = Game(rec.seed, RulesConfig(**rec.rules))
    for i, a in enumerate(rec.actions):
        actor = g.actor
        g.step(a)
        yield i, actor, describe(a), render(g.state, seat)


def verify(rec: GameRecord) -> bool:
    """True if re-simulating reproduces the recorded outcome."""
    s = rebuild(rec).state
    return s.winner == rec.winner and s.turn == rec.turns and s.done

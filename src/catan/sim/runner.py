"""Play games between agents, sequentially or across worker processes."""

from __future__ import annotations

import os
import time
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from catan.engine import Game, RulesConfig
from catan.engine.constants import NUM_PLAYERS
from catan.player.base import Agent
from catan.player.observation import observe


@dataclass
class GameRecord:
    seed: int
    rules: dict[str, Any]
    agents: list[str]  # spec per seat
    winner: int
    turns: int
    truncated: bool
    vp: list[int]
    ranks: list[int]  # rank per seat, 1 = best (ties share a rank)
    steps: int
    duration_s: float
    actions: list[int] = field(default_factory=list)

    @property
    def winner_spec(self) -> str | None:
        return self.agents[self.winner] if self.winner >= 0 else None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ranks(vp: list[int], winner: int) -> list[int]:
    keys = [(p == winner, v) for p, v in enumerate(vp)]
    return [1 + sum(k2 > k for k2 in keys) for k in keys]


def play_game(
    agents: Sequence[Agent],
    seed: int,
    config: RulesConfig | None = None,
    specs: Sequence[str] | None = None,
    record_actions: bool = True,
) -> GameRecord:
    """Play one game. ``agents[i]`` sits in seat ``i`` (seat 0 places first)."""
    assert len(agents) == NUM_PLAYERS
    config = config or RulesConfig()
    t0 = time.perf_counter()
    game = Game(seed, config)
    for i, a in enumerate(agents):
        a.reset(i, np.random.default_rng([seed, i]))
    steps = 0
    while not game.done:
        p = game.actor
        action = agents[p].act(observe(game.state, p))
        for e in game.step(action):
            for i, a in enumerate(agents):
                a.observe(e.view(i))
        steps += 1
    s = game.state
    vp = [s.total_vp(p) for p in range(NUM_PLAYERS)]
    return GameRecord(
        seed=seed,
        rules=asdict(config),
        agents=list(specs) if specs else [getattr(a, "spec", a.name) for a in agents],
        winner=s.winner,
        turns=s.turn,
        truncated=s.truncated,
        vp=vp,
        ranks=_ranks(vp, s.winner),
        steps=steps,
        duration_s=time.perf_counter() - t0,
        actions=game.actions if record_actions else [],
    )


@dataclass(frozen=True)
class Matchup:
    specs: tuple[str, ...]  # one per seat
    seed: int


def _play_matchup(args: tuple[Matchup, dict[str, Any], str, bool]) -> GameRecord:
    from catan.agents.registry import make_agent

    m, rules, data_dir, record = args
    agents = [make_agent(s, data_dir) for s in m.specs]
    return play_game(agents, m.seed, RulesConfig(**rules), specs=m.specs, record_actions=record)


def run_matchups(
    matchups: Iterable[Matchup],
    config: RulesConfig | None = None,
    workers: int = 1,
    data_dir: str = "data",
    record_actions: bool = True,
) -> Iterator[GameRecord]:
    """Yield records as games finish. ``workers > 1`` uses a process pool."""
    rules = asdict(config or RulesConfig())
    jobs = [(m, rules, data_dir, record_actions) for m in matchups]
    if workers <= 1:
        for j in jobs:
            yield _play_matchup(j)
        return
    workers = min(workers, os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        yield from pool.map(_play_matchup, jobs, chunksize=max(1, len(jobs) // (workers * 8)))

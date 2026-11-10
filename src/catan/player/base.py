"""Agent interface.

The simulator drives agents like this::

    agent.reset(seat, rng)            # once per game
    agent.observe(event.view(seat))   # every event, for every seat, in order
    action = agent.act(obs)           # only when obs.actor == seat

The base class keeps a ``BeliefTracker`` up to date from observed events, so
every agent has card counting for free.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from catan.player.belief import BeliefTracker
from catan.player.observation import Observation


class Agent(ABC):
    name: str = "agent"
    track_beliefs: bool = True

    def __init__(self) -> None:
        self.seat = -1
        self.rng = np.random.default_rng()
        self.belief: BeliefTracker | None = None

    def reset(self, seat: int, rng: np.random.Generator) -> None:
        self.seat = seat
        self.rng = rng
        self.belief = BeliefTracker(seat) if self.track_beliefs else None

    def observe(self, event_view: dict[str, Any]) -> None:
        if self.belief is not None:
            self.belief.update(event_view)

    @abstractmethod
    def act(self, obs: Observation) -> int: ...

    def config(self) -> dict[str, Any]:
        """JSON-serializable constructor kwargs (stored in policy manifests)."""
        return {}

    def save_files(self, directory: Path) -> dict[str, str]:
        """Write any state beyond ``config()`` (weights, statistics). Returns {role: filename}."""
        return {}

    def load_files(self, directory: Path, manifest: dict[str, Any]) -> None:  # noqa: B027
        """Inverse of ``save_files``."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.config()})"

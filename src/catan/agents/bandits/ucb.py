"""UCB1 bandit (also the tree policy inside MCTS, via ``ucb1_score``)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def ucb1_score(total: float, visits: int, parent_visits: int, c: float) -> float:
    if visits == 0:
        return math.inf
    return total / visits + c * math.sqrt(math.log(max(parent_visits, 1)) / visits)


class UCB1:
    """Rewards are assumed to lie roughly in [0, 1]."""

    def __init__(self, n_arms: int, c: float = math.sqrt(2)) -> None:
        self.c = c
        self.counts = np.zeros(n_arms, dtype=np.int64)
        self.totals = np.zeros(n_arms)

    @property
    def n_arms(self) -> int:
        return len(self.counts)

    def select(self, rng: np.random.Generator | None = None) -> int:
        untried = np.flatnonzero(self.counts == 0)
        if len(untried):
            return int(untried[0] if rng is None else rng.choice(untried))
        n = self.counts.sum()
        scores = self.totals / self.counts + self.c * np.sqrt(np.log(n) / self.counts)
        return int(np.argmax(scores))

    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        self.totals[arm] += reward

    def means(self) -> np.ndarray:
        return np.divide(self.totals, self.counts, out=np.zeros(self.n_arms), where=self.counts > 0)

    def best(self) -> int:
        """Arm to commit to after exploration (most-visited, ties by mean)."""
        return int(np.lexsort((self.means(), self.counts))[-1])

    def to_dict(self) -> dict[str, Any]:
        return {"c": self.c, "counts": self.counts.tolist(), "totals": self.totals.tolist()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> UCB1:
        b = cls(len(d["counts"]), d["c"])
        b.counts = np.array(d["counts"], dtype=np.int64)
        b.totals = np.array(d["totals"], dtype=float)
        return b

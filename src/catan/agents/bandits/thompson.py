"""Thompson sampling with a Beta posterior (rewards in [0, 1], fractional allowed)."""

from __future__ import annotations

from typing import Any

import numpy as np


class BetaThompson:
    def __init__(self, n_arms: int, prior_a: float = 1.0, prior_b: float = 1.0) -> None:
        self.a = np.full(n_arms, prior_a)
        self.b = np.full(n_arms, prior_b)

    @property
    def n_arms(self) -> int:
        return len(self.a)

    @property
    def counts(self) -> np.ndarray:
        return self.a + self.b - 2

    def select(self, rng: np.random.Generator) -> int:
        return int(np.argmax(rng.beta(self.a, self.b)))

    def update(self, arm: int, reward: float) -> None:
        r = float(np.clip(reward, 0.0, 1.0))
        self.a[arm] += r
        self.b[arm] += 1.0 - r

    def means(self) -> np.ndarray:
        return self.a / (self.a + self.b)

    def best(self) -> int:
        return int(np.argmax(self.means()))

    def to_dict(self) -> dict[str, Any]:
        return {"a": self.a.tolist(), "b": self.b.tolist()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BetaThompson:
        t = cls(len(d["a"]))
        t.a, t.b = np.array(d["a"]), np.array(d["b"])
        return t

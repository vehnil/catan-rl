"""Max^n search with expectimax chance nodes for dice.

Max^n generalizes minimax to N players: each node returns a utility *vector*
and the player to move picks the child that maximizes its own component.
Dice rolls are chance nodes averaged over the 11 sums. Search runs on
determinized states (hidden cards sampled from the belief tracker) and is
depth-limited with heuristic leaves; see ``common.SearchAgent`` for the
beam and the routine-decision default policy.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from catan.agents.search.common import SearchAgent
from catan.engine.actions import ActionType, decode
from catan.engine.constants import DICE_WEIGHTS
from catan.engine.rules import apply, apply_roll
from catan.engine.state import GameState
from catan.player.observation import Observation

DICE_OUTCOMES: list[tuple[int, int, float]] = [
    (max(1, s - 6), s - max(1, s - 6), w / 36) for s, w in DICE_WEIGHTS.items()
]


def chance_children(s: GameState) -> list[tuple[GameState, float]]:
    out = []
    for d1, d2, p in DICE_OUTCOMES:
        c = s.clone()
        apply_roll(c, d1, d2)
        out.append((c, p))
    return out


class MaxNAgent(SearchAgent):
    name = "maxn"

    def __init__(self, depth: int = 2, beam: int = 5, determinizations: int = 1, **kw: Any) -> None:
        super().__init__(beam=beam, determinizations=determinizations, **kw)
        self.depth = depth
        self.nodes = 0

    def config(self) -> dict[str, Any]:
        return {**super().config(), "depth": self.depth}

    def search(self, obs: Observation) -> int:
        totals: dict[int, np.ndarray] = {}
        for _ in range(self.determinizations):
            s = self.sample_state(obs)
            for a in self.candidates(s, obs.legal_mask):
                totals[a] = totals.get(a, 0) + self._action_value(s, a, self.depth)
        return max(totals, key=lambda a: totals[a][obs.seat])

    def _action_value(self, s: GameState, a: int, depth: int) -> np.ndarray:
        self.nodes += 1
        if decode(a).type == ActionType.ROLL:
            return np.sum([p * self._value(c, depth - 1) for c, p in chance_children(s)], axis=0)
        c = s.clone()
        apply(c, a, self.rng, validate=False)
        return self._value(c, depth - 1)

    def _value(self, s: GameState, depth: int) -> np.ndarray:
        self.advance_default(s, self.rng)
        if s.done or depth <= 0:
            return self.leaf_values(s)
        actor = s.actor
        best: np.ndarray | None = None
        for a in self.candidates(s):
            v = self._action_value(s, a, depth)
            if best is None or v[actor] > best[actor]:
                best = v
        assert best is not None
        return best

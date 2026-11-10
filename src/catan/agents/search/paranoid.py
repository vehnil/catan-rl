"""Paranoid search: assume every opponent plays to minimize *my* utility.

This reduces the N-player game to me vs. a coalition, which makes alpha-beta
pruning valid again (Max^n can't prune). It's pessimistic, since opponents in
Catan don't really cooperate, but it's a useful contrast to Max^n. Dice are
expectimax chance nodes; alpha-beta windows are reset below them.
"""

from __future__ import annotations

import math
from typing import Any

from catan.agents.search.common import SearchAgent
from catan.agents.search.maxn import chance_children
from catan.engine.actions import ActionType, decode
from catan.engine.rules import apply
from catan.engine.state import GameState
from catan.player.observation import Observation


class ParanoidAgent(SearchAgent):
    name = "paranoid"

    def __init__(self, depth: int = 3, beam: int = 5, determinizations: int = 1, **kw: Any) -> None:
        super().__init__(beam=beam, determinizations=determinizations, **kw)
        self.depth = depth
        self.nodes = 0

    def config(self) -> dict[str, Any]:
        return {**super().config(), "depth": self.depth}

    def search(self, obs: Observation) -> int:
        me = obs.seat
        totals: dict[int, float] = {}
        for _ in range(self.determinizations):
            s = self.sample_state(obs)
            alpha = -math.inf
            for a in self.candidates(s, obs.legal_mask):
                v = self._action_value(s, a, self.depth, me, alpha, math.inf)
                totals[a] = totals.get(a, 0.0) + v
                if self.determinizations == 1:
                    alpha = max(alpha, v)
        return max(totals, key=lambda a: totals[a])

    def _action_value(
        self, s: GameState, a: int, depth: int, me: int, alpha: float, beta: float
    ) -> float:
        self.nodes += 1
        if decode(a).type == ActionType.ROLL:
            return sum(
                p * self._value(c, depth - 1, me, -math.inf, math.inf)
                for c, p in chance_children(s)
            )
        c = s.clone()
        apply(c, a, self.rng, validate=False)
        return self._value(c, depth - 1, me, alpha, beta)

    def _value(self, s: GameState, depth: int, me: int, alpha: float, beta: float) -> float:
        self.advance_default(s, self.rng)
        if s.done or depth <= 0:
            return float(self.leaf_values(s)[me])
        maximizing = s.actor == me
        best = -math.inf if maximizing else math.inf
        for a in self.candidates(s):
            v = self._action_value(s, a, depth, me, alpha, beta)
            if maximizing:
                best = max(best, v)
                alpha = max(alpha, best)
            else:
                best = min(best, v)
                beta = min(beta, best)
            if alpha >= beta:
                break
        return best

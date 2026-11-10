"""Shared machinery for search agents.

Catan's raw branching factor is large (hundreds of trade offers, dozens of
roads), and many decisions are routine. Search agents therefore:

* branch only at strategic phases (``BRANCH_PHASES``); routine decisions
  (discards, trade responses, steal targets, confirmations) follow the greedy
  default policy and don't consume search depth;
* consider a *beam* of candidate actions ranked by one-step lookahead, with
  domestic trade proposals reduced to the greedy policy's single best offer;
* search over a *determinized* state sampled from the belief tracker.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from catan.agents.baselines.greedy import _LOOKAHEAD_TYPES, GreedyAgent, _after
from catan.agents.heuristics import evaluate, evaluate_all, robber_hex_score, vertex_score
from catan.engine.actions import ActionType, block, decode
from catan.engine.constants import NUM_PLAYERS
from catan.engine.phases import Phase
from catan.engine.rules import apply, legal_mask
from catan.engine.state import GameState
from catan.player.observation import Observation

BRANCH_PHASES = {
    Phase.SETUP_SETTLEMENT,
    Phase.SETUP_ROAD,
    Phase.ROLL,
    Phase.MOVE_ROBBER,
    Phase.MAIN,
    Phase.ROAD_BUILDING,
}
_PROPOSE = block(ActionType.PROPOSE_TRADE)
_RANKABLE = _LOOKAHEAD_TYPES | {ActionType.MOVE_ROBBER}


class SearchAgent(GreedyAgent):
    """Greedy everywhere except strategic decisions, which go to ``search``."""

    def __init__(self, beam: int = 6, determinizations: int = 1, **greedy_kwargs: Any) -> None:
        super().__init__(**greedy_kwargs)
        self.beam = beam
        self.determinizations = determinizations
        self.default_policy = GreedyAgent(weights=self.w)

    def config(self) -> dict[str, Any]:
        return {**super().config(), "beam": self.beam, "determinizations": self.determinizations}

    def act(self, obs: Observation) -> int:
        legal = np.flatnonzero(obs.legal_mask)
        if len(legal) == 1:
            return int(legal[0])
        if obs.phase not in BRANCH_PHASES:
            return super().act(obs)
        self.default_policy.rng = self.rng
        self.default_policy.belief = self.belief
        return self.search(obs)

    def search(self, obs: Observation) -> int:
        raise NotImplementedError

    # ------------------------------------------------------------ helpers

    def sample_state(self, obs: Observation) -> GameState:
        if self.belief is not None:
            return self.belief.determinize(obs.redacted, self.rng)
        return self._guess_state(obs)

    def candidates(self, s: GameState, mask: np.ndarray | None = None) -> list[int]:
        """Beam of promising actions for ``s.actor`` (always includes the greedy choice)."""
        mask = legal_mask(s) if mask is None else mask
        me = s.actor
        legal = [int(a) for a in np.flatnonzero(mask) if not (_PROPOSE.start <= a < _PROPOSE.stop)]
        keep = self.default_policy.act_state(s, mask, me)
        if len(legal) > self.beam:
            ranked = [a for a in legal if decode(a).type in _RANKABLE]
            rest = [a for a in legal if decode(a).type not in _RANKABLE]
            vals = [self._rank(s, a, me) for a in ranked]
            legal = [ranked[i] for i in np.argsort(vals)[::-1][: self.beam]] + rest
        if keep not in legal:
            legal.append(keep)
        return legal

    def _rank(self, s: GameState, a: int, me: int) -> float:
        t, arg = decode(a)
        if t == ActionType.BUILD_SETTLEMENT and s.phase == Phase.SETUP_SETTLEMENT:
            return vertex_score(s, arg, me, self.w)
        if t == ActionType.MOVE_ROBBER:
            return robber_hex_score(s, arg, me)
        return evaluate(_after(s, a), me, self.w)

    def advance_default(self, s: GameState, rng: np.random.Generator) -> None:
        """Play routine (non-branching) decisions with the default policy."""
        while not s.done and s.phase not in BRANCH_PHASES:
            mask = legal_mask(s)
            self.default_policy.seat = s.actor
            apply(s, self.default_policy.act_state(s, mask, s.actor), rng, validate=False)

    def rollout(self, s: GameState, turns: int, rng: np.random.Generator) -> None:
        """Default-policy playout for ``turns`` turns (0 = none)."""
        end = s.turn + turns
        while turns and not s.done and s.turn < end:
            mask = legal_mask(s)
            self.default_policy.seat = s.actor
            apply(s, self.default_policy.act_state(s, mask, s.actor), rng, validate=False)

    def leaf_values(self, s: GameState, temperature: float = 60.0) -> np.ndarray:
        """Per-seat rewards in (0, 1): win = 1, else sigmoid of lead over the best rival."""
        if s.done:
            r = np.zeros(NUM_PLAYERS)
            if s.winner >= 0:
                r[s.winner] = 1.0
            else:
                r[:] = 0.25
            return r
        v = evaluate_all(s, self.w)
        out = np.empty(NUM_PLAYERS)
        for p in range(NUM_PLAYERS):
            out[p] = 1 / (1 + np.exp(-(v[p] - np.delete(v, p).max()) / temperature))
        return out

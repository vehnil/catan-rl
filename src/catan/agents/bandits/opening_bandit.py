"""Opening placement as a best-arm bandit problem.

Each initial settlement choice is an arm. Pulling an arm = one rollout:
sample hidden information, place the settlement, let every seat play greedily
for ``rollout_turns`` turns after setup, and score the result. UCB1 or
Thompson sampling allocates the rollout budget across the top-``top_k``
candidate vertices (pre-ranked by the static heuristic). After setup, play is
delegated to the greedy agent.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from catan.agents.bandits.thompson import BetaThompson
from catan.agents.bandits.ucb import UCB1
from catan.agents.baselines.greedy import GreedyAgent
from catan.agents.heuristics import evaluate_all, vertex_score
from catan.engine.actions import ActionType, block, encode
from catan.engine.phases import Phase
from catan.engine.rules import apply, legal_mask
from catan.engine.state import GameState
from catan.player.observation import Observation


def relative_score(values: np.ndarray, me: int, temperature: float = 60.0) -> float:
    """Squash 'my utility vs. the best opponent' into (0, 1)."""
    others = np.delete(values, me)
    return float(1 / (1 + np.exp(-(values[me] - others.max()) / temperature)))


class OpeningBanditAgent(GreedyAgent):
    name = "opening_bandit"

    def __init__(
        self,
        budget: int = 48,
        top_k: int = 6,
        rollout_turns: int = 8,
        algo: str = "ucb",
        c: float = 0.5,
        **greedy_kwargs: Any,
    ) -> None:
        super().__init__(**greedy_kwargs)
        self.budget = budget
        self.top_k = top_k
        self.rollout_turns = rollout_turns
        self.algo = algo
        self.c = c
        self._rollout_policy = GreedyAgent(epsilon=0.1)
        self.last_bandit: UCB1 | BetaThompson | None = None

    def config(self) -> dict[str, Any]:
        return {
            **super().config(),
            "budget": self.budget,
            "top_k": self.top_k,
            "rollout_turns": self.rollout_turns,
            "algo": self.algo,
            "c": self.c,
        }

    def act(self, obs: Observation) -> int:
        if obs.phase != Phase.SETUP_SETTLEMENT:
            return super().act(obs)
        state = self._guess_state(obs)
        o = block(ActionType.BUILD_SETTLEMENT).start
        legal = np.flatnonzero(obs.legal_mask[block(ActionType.BUILD_SETTLEMENT)])
        ranked = sorted(legal, key=lambda v: -vertex_score(state, int(v), obs.seat, self.w))
        arms = [int(v) for v in ranked[: self.top_k]]
        if len(arms) == 1:
            return o + arms[0]
        bandit: UCB1 | BetaThompson = (
            UCB1(len(arms), self.c) if self.algo == "ucb" else BetaThompson(len(arms))
        )
        self._rollout_policy.rng = self.rng
        for _ in range(self.budget):
            arm = bandit.select(self.rng)
            bandit.update(arm, self._rollout(obs, arms[arm]))
        self.last_bandit = bandit
        return o + arms[bandit.best()]

    def _rollout(self, obs: Observation, vertex: int) -> float:
        s: GameState = (
            self.belief.determinize(obs.redacted, self.rng) if self.belief else obs.redacted.clone()
        )
        apply(s, encode(ActionType.BUILD_SETTLEMENT, vertex), self.rng)
        start_turn = None
        while not s.done:
            if not s.is_setup and start_turn is None:
                start_turn = s.turn
            if start_turn is not None and s.turn - start_turn >= self.rollout_turns:
                break
            mask = legal_mask(s)
            self._rollout_policy.seat = s.actor
            apply(s, self._rollout_policy.act_state(s, mask, s.actor), self.rng, validate=False)
        return relative_score(evaluate_all(s, self.w), obs.seat)

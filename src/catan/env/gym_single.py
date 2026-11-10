"""Single-agent Gymnasium wrapper: one learner seat vs. fixed opponent agents.

Opponents act inside ``step`` until it is the learner's move again (or the
game ends). The legal-action mask is exposed via ``info["action_mask"]`` and
``action_masks()`` (the sb3-contrib MaskablePPO convention).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from catan.engine import Game, RulesConfig
from catan.engine.actions import NUM_ACTIONS
from catan.engine.constants import NUM_PLAYERS
from catan.player.base import Agent
from catan.player.belief import BeliefTracker
from catan.player.features import FEATURE_DIM, encode
from catan.player.observation import Observation, observe

OpponentFactory = Callable[[], Sequence[Agent]]


class CatanSingleAgentEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        opponents: OpponentFactory,
        config: RulesConfig | None = None,
        learner_seat: int | None = None,
        win_reward: float = 1.0,
        loss_reward: float = -1.0,
        vp_reward_scale: float = 0.0,
    ) -> None:
        self.opponents_factory = opponents
        self.config = config or RulesConfig()
        self.fixed_seat = learner_seat
        self.win_reward = win_reward
        self.loss_reward = loss_reward
        self.vp_reward_scale = vp_reward_scale
        self.observation_space = spaces.Box(-np.inf, np.inf, (FEATURE_DIM,), np.float32)
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.game: Game | None = None
        self._obs: Observation | None = None

    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        game_seed = int(self.np_random.integers(2**31)) if seed is None else seed
        self.game = Game(game_seed, self.config)
        self.seat = (
            self.fixed_seat
            if self.fixed_seat is not None
            else int(self.np_random.integers(NUM_PLAYERS))
        )
        self.belief = BeliefTracker(self.seat)
        opps = list(self.opponents_factory())
        assert len(opps) == NUM_PLAYERS - 1
        self.agents: dict[int, Agent] = {}
        for i, s in enumerate(q for q in range(NUM_PLAYERS) if q != self.seat):
            opps[i].reset(s, np.random.default_rng(game_seed + 1 + s))
            self.agents[s] = opps[i]
        self._advance()
        return self._features(), self._info()

    def step(self, action: int):
        assert self.game is not None
        state = self.game.state
        vp_before = state.total_vp(self.seat)
        self._dispatch(self.game.step(int(action)))
        self._advance()
        reward = self.vp_reward_scale * (state.total_vp(self.seat) - vp_before)
        terminated = state.done and not state.truncated
        truncated = state.truncated
        if state.done and state.winner >= 0:
            reward += self.win_reward if state.winner == self.seat else self.loss_reward
        return self._features(), float(reward), terminated, truncated, self._info()

    def action_masks(self) -> np.ndarray:
        assert self._obs is not None
        return self._obs.legal_mask

    def render(self) -> str:
        from catan.engine.render import render

        assert self.game is not None
        return render(self.game.state, seat=self.seat)

    # ------------------------------------------------------------------

    def _advance(self) -> None:
        """Let opponents move until the learner must act or the game ends."""
        assert self.game is not None
        while not self.game.done and self.game.actor != self.seat:
            p = self.game.actor
            a = self.agents[p].act(observe(self.game.state, p))
            self._dispatch(self.game.step(a))
        self._obs = observe(self.game.state, self.seat)

    def _dispatch(self, events) -> None:
        for e in events:
            self.belief.update(e.view(self.seat))
            for s, ag in self.agents.items():
                ag.observe(e.view(s))

    def _features(self) -> np.ndarray:
        assert self._obs is not None
        return encode(self._obs, self.belief)

    def _info(self) -> dict[str, Any]:
        assert self._obs is not None and self.game is not None
        info: dict[str, Any] = {"action_mask": self._obs.legal_mask, "seat": self.seat}
        if self.game.done:
            info.update(winner=self.game.state.winner, turns=self.game.state.turn)
        return info

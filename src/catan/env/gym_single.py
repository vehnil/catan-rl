"""Single-agent Gymnasium wrapper: one learner seat vs. fixed opponent agents.

Opponents act inside ``step`` until it is the learner's move again (or the
game ends). The legal-action mask is exposed via ``info["action_mask"]`` and
``action_masks()`` (the sb3-contrib MaskablePPO convention).

``reset``/``step`` are also available in pieces (``begin_reset``/``begin_step``,
``run_opponents``, ``finish_reset``/``finish_step``) so a driver running many
envs can pause each one when a network opponent must move, batch those
decisions across envs, and resume with ``opponent_move``.
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
        fast_win_bonus: float = 0.0,
        fast_win_turns: int = 200,
    ) -> None:
        self.opponents_factory = opponents
        self.config = config or RulesConfig()
        self.fixed_seat = learner_seat
        self.win_reward = win_reward
        self.loss_reward = loss_reward
        self.vp_reward_scale = vp_reward_scale
        # Extra reward for winning sooner: the full bonus at turn 0, nothing from fast_win_turns on.
        # Only wins get it, so there is no incentive to lose quickly.
        self.fast_win_bonus = fast_win_bonus
        self.fast_win_turns = fast_win_turns
        self.observation_space = spaces.Box(-np.inf, np.inf, (FEATURE_DIM,), np.float32)
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.game: Game | None = None
        self._obs: Observation | None = None

    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self.begin_reset(seed=seed)
        self.run_opponents()
        return self.finish_reset()

    def begin_reset(self, seed: int | None = None) -> None:
        """Start a new game; opponents seated before the learner have not moved yet."""
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
        self._pending: int | None = None

    def finish_reset(self):
        return self._features(), self._info()

    def step(self, action: int):
        self.begin_step(action)
        self.run_opponents()
        return self.finish_step()

    def begin_step(self, action: int) -> None:
        """Apply the learner's action; opponents have not replied yet."""
        assert self.game is not None
        self._vp_before = self.game.state.total_vp(self.seat)
        self._dispatch(self.game.step(int(action)))

    def finish_step(self):
        """Reward and next observation, once ``run_opponents`` has returned None."""
        assert self.game is not None
        state = self.game.state
        reward = self.vp_reward_scale * (state.total_vp(self.seat) - self._vp_before)
        terminated = state.done and not state.truncated
        truncated = state.truncated
        if state.done and state.winner >= 0:
            reward += self.win_reward if state.winner == self.seat else self.loss_reward
            if state.winner == self.seat and self.fast_win_bonus:
                reward += self.fast_win_bonus * max(0.0, 1 - state.turn / self.fast_win_turns)
        return self._features(), float(reward), terminated, truncated, self._info()

    def action_masks(self) -> np.ndarray:
        assert self._obs is not None
        return self._obs.legal_mask

    def render(self) -> str:
        from catan.engine.render import render

        assert self.game is not None
        return render(self.game.state, seat=self.seat)

    # ------------------------------------------------------------------

    def run_opponents(self, pause_for: Callable[[Agent], bool] | None = None) -> Agent | None:
        """Let opponents move until the learner must act or the game ends, and return None.

        With ``pause_for``, stop instead when an opponent it matches must move, and return
        that agent: its observation is ``pending_observation()``, and its move goes in with
        ``opponent_move``; then call this again."""
        assert self.game is not None and self._pending is None
        while not self.game.done and self.game.actor != self.seat:
            p = self.game.actor
            agent = self.agents[p]
            if pause_for is not None and pause_for(agent):
                self._pending = p
                return agent
            a = agent.act(observe(self.game.state, p))
            self._dispatch(self.game.step(a))
        self._obs = observe(self.game.state, self.seat)
        return None

    def pending_observation(self) -> Observation:
        assert self.game is not None and self._pending is not None
        return observe(self.game.state, self._pending)

    def opponent_move(self, action: int) -> None:
        assert self.game is not None and self._pending is not None
        self._pending = None
        self._dispatch(self.game.step(int(action)))

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

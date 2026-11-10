"""PettingZoo AEC (turn-based multi-agent) wrapper.

``agent_selection`` follows the engine's ``actor``, which is not always the
player whose turn it is: discards after a 7 and responses to trade offers are
made out of turn. Observations are dicts with a flat feature vector and the
legal-action mask (the convention used by PettingZoo classic envs).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv

from catan.engine import Game, RulesConfig
from catan.engine.actions import NUM_ACTIONS
from catan.engine.constants import NUM_PLAYERS
from catan.engine.render import render
from catan.player.belief import BeliefTracker
from catan.player.features import FEATURE_DIM, encode
from catan.player.observation import observe


def agent_name(seat: int) -> str:
    return f"player_{seat}"


class CatanAECEnv(AECEnv):
    metadata = {"name": "catan_v0", "render_modes": ["ansi", "human"], "is_parallelizable": False}

    def __init__(
        self,
        config: RulesConfig | None = None,
        render_mode: str | None = None,
        win_reward: float = 1.0,
        loss_reward: float = -1.0,
        vp_reward_scale: float = 0.0,
    ) -> None:
        super().__init__()
        self.config = config or RulesConfig()
        self.render_mode = render_mode
        self.win_reward = win_reward
        self.loss_reward = loss_reward
        self.vp_reward_scale = vp_reward_scale
        self.possible_agents = [agent_name(i) for i in range(NUM_PLAYERS)]
        self._seat = {a: i for i, a in enumerate(self.possible_agents)}
        obs_space = spaces.Dict(
            {
                "observation": spaces.Box(-np.inf, np.inf, (FEATURE_DIM,), np.float32),
                "action_mask": spaces.Box(0, 1, (NUM_ACTIONS,), np.int8),
            }
        )
        self.observation_spaces = {a: obs_space for a in self.possible_agents}
        self.action_spaces = {a: spaces.Discrete(NUM_ACTIONS) for a in self.possible_agents}
        self.game: Game | None = None

    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        self.game = Game(seed, self.config)
        self.beliefs = [BeliefTracker(s) for s in range(NUM_PLAYERS)]
        self.agents = self.possible_agents[:]
        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos: dict[str, dict[str, Any]] = {a: {} for a in self.agents}
        self.agent_selection = agent_name(self.game.actor)

    def observe(self, agent: str) -> dict[str, np.ndarray]:
        assert self.game is not None
        seat = self._seat[agent]
        obs = observe(self.game.state, seat)
        return {
            "observation": encode(obs, self.beliefs[seat]),
            "action_mask": obs.legal_mask.astype(np.int8),
        }

    def step(self, action: int | None) -> None:
        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return
        assert self.game is not None and action is not None
        seat = self._seat[agent]
        state = self.game.state
        vp_before = state.total_vp(seat)
        self._cumulative_rewards[agent] = 0.0
        self.rewards = {a: 0.0 for a in self.agents}

        for e in self.game.step(int(action)):
            for s, tr in enumerate(self.beliefs):
                tr.update(e.view(s))

        if self.vp_reward_scale:
            self.rewards[agent] += self.vp_reward_scale * (state.total_vp(seat) - vp_before)

        if state.done:
            for a in self.agents:
                if state.winner >= 0:
                    self.rewards[a] += (
                        self.win_reward if self._seat[a] == state.winner else self.loss_reward
                    )
                if state.truncated:
                    self.truncations[a] = True
                else:
                    self.terminations[a] = True
                self.infos[a] = {"winner": state.winner, "turns": state.turn}
            self.agent_selection = self.agents[(self.agents.index(agent) + 1) % len(self.agents)]
        else:
            self.agent_selection = agent_name(self.game.actor)
        self._accumulate_rewards()
        if self.render_mode == "human":
            self.render()

    def render(self) -> str | None:
        assert self.game is not None
        text = render(self.game.state)
        if self.render_mode == "human":
            print(text)
            return None
        return text

    def close(self) -> None:
        pass


def env(**kwargs: Any) -> CatanAECEnv:
    return CatanAECEnv(**kwargs)

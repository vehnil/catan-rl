"""Opponent pool for self-play: fixed agents plus frozen snapshots of the learner."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from catan.agents.learned.ppo.agent import PPOAgent
from catan.agents.learned.ppo.model import ActorCritic
from catan.agents.registry import make_agent
from catan.config.models import LeagueConfig
from catan.engine.constants import NUM_PLAYERS
from catan.player.base import Agent


class League:
    def __init__(
        self,
        cfg: LeagueConfig,
        hidden: list[int],
        rng: np.random.Generator,
        data_dir: str | Path = "data",
    ) -> None:
        self.cfg = cfg
        self.hidden = hidden
        self.rng = rng
        self.data_dir = data_dir
        self.snapshots: list[ActorCritic] = []
        # Stored PPO opponents ("policy:<ref>") are loaded once and shared frozen across games.
        self._frozen: dict[str, PPOAgent] = {}

    def add_snapshot(self, model: ActorCritic) -> None:
        snap = copy.deepcopy(model).eval()
        for p in snap.parameters():
            p.requires_grad_(False)
        self.snapshots.append(snap)
        if len(self.snapshots) > self.cfg.max_snapshots:
            self.snapshots.pop(0)

    def _make(self, spec: str) -> Agent:
        if not spec.startswith("policy:"):
            return make_agent(spec)
        proto = self._frozen.get(spec)
        if proto is None:
            agent = make_agent(spec, self.data_dir)
            if not isinstance(agent, PPOAgent):
                return agent  # other stored kinds may carry per-game state; load fresh
            agent.model.eval()
            for p in agent.model.parameters():
                p.requires_grad_(False)
            proto = self._frozen[spec] = agent
        fresh = PPOAgent(proto.hidden, deterministic=proto.deterministic, model=proto.model)
        fresh.spec = proto.spec  # type: ignore[attr-defined]
        return fresh

    def sample_opponents(self) -> list[Agent]:
        lineup = (
            [str(s) for s in self.rng.permutation(self.cfg.fixed_opponents)]
            if self.cfg.lineup
            else None
        )
        out: list[Agent] = []
        for i in range(NUM_PLAYERS - 1):
            if self.snapshots and self.rng.random() < self.cfg.self_play_prob:
                model = self.snapshots[int(self.rng.integers(len(self.snapshots)))]
                out.append(PPOAgent(self.hidden, model=model))
            elif lineup is not None:
                out.append(self._make(lineup[i]))
            else:
                out.append(self._make(str(self.rng.choice(self.cfg.fixed_opponents))))
        return out

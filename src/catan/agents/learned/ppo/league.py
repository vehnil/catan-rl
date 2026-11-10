"""Opponent pool for self-play: fixed agents plus frozen snapshots of the learner."""

from __future__ import annotations

import copy

import numpy as np

from catan.agents.learned.ppo.agent import PPOAgent
from catan.agents.learned.ppo.model import ActorCritic
from catan.agents.registry import make_agent
from catan.config.models import LeagueConfig
from catan.engine.constants import NUM_PLAYERS
from catan.player.base import Agent


class League:
    def __init__(self, cfg: LeagueConfig, hidden: list[int], rng: np.random.Generator) -> None:
        self.cfg = cfg
        self.hidden = hidden
        self.rng = rng
        self.snapshots: list[ActorCritic] = []

    def add_snapshot(self, model: ActorCritic) -> None:
        snap = copy.deepcopy(model).eval()
        for p in snap.parameters():
            p.requires_grad_(False)
        self.snapshots.append(snap)
        if len(self.snapshots) > self.cfg.max_snapshots:
            self.snapshots.pop(0)

    def sample_opponents(self) -> list[Agent]:
        out: list[Agent] = []
        for _ in range(NUM_PLAYERS - 1):
            if self.snapshots and self.rng.random() < self.cfg.self_play_prob:
                model = self.snapshots[int(self.rng.integers(len(self.snapshots)))]
                out.append(PPOAgent(self.hidden, model=model))
            else:
                out.append(make_agent(str(self.rng.choice(self.cfg.fixed_opponents))))
        return out

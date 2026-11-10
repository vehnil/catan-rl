"""``Game``: a GameState plus its RNG and full history. The unit replays are built from."""

from __future__ import annotations

import numpy as np

from catan.engine import rules
from catan.engine.events import Event
from catan.engine.state import GameState, RulesConfig


class Game:
    def __init__(self, seed: int | None = None, config: RulesConfig | None = None) -> None:
        self.seed = seed
        self.config = config or RulesConfig()
        self.rng = np.random.default_rng(seed)
        self.state = GameState.new(self.config, self.rng)
        self.actions: list[int] = []
        self.events: list[Event] = []

    @property
    def done(self) -> bool:
        return self.state.done

    @property
    def actor(self) -> int:
        return self.state.actor

    def legal_mask(self) -> np.ndarray:
        return rules.legal_mask(self.state)

    def legal_actions(self) -> np.ndarray:
        return rules.legal_actions(self.state)

    def step(self, action: int) -> list[Event]:
        new = rules.apply(self.state, int(action), self.rng)
        self.actions.append(int(action))
        self.events.extend(new)
        return new

    @staticmethod
    def replay(seed: int | None, config: RulesConfig, actions: list[int]) -> Game:
        g = Game(seed, config)
        for a in actions:
            g.step(a)
        return g

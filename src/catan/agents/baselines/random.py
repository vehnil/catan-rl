"""Uniform-random legal play. The floor every other agent is measured against."""

from __future__ import annotations

from typing import Any

import numpy as np

from catan.engine.actions import ActionType, block
from catan.player.base import Agent
from catan.player.observation import Observation

_PROPOSE = block(ActionType.PROPOSE_TRADE)


class RandomAgent(Agent):
    """Picks uniformly among legal actions.

    ``trade_weight`` scales the total probability of proposing a domestic trade
    so that the ~230 offer actions don't drown out everything else.
    """

    name = "random"
    track_beliefs = False

    def __init__(self, trade_weight: float = 0.05) -> None:
        super().__init__()
        self.trade_weight = trade_weight

    def act(self, obs: Observation) -> int:
        legal = np.flatnonzero(obs.legal_mask)
        if len(legal) == 1:
            return int(legal[0])
        is_offer = (legal >= _PROPOSE.start) & (legal < _PROPOSE.stop)
        n_offer, n_other = int(is_offer.sum()), int((~is_offer).sum())
        if n_offer == 0 or n_other == 0:
            return int(self.rng.choice(legal))
        w = np.where(is_offer, self.trade_weight / n_offer, (1 - self.trade_weight) / n_other)
        return int(self.rng.choice(legal, p=w / w.sum()))

    def config(self) -> dict[str, Any]:
        return {"trade_weight": self.trade_weight}

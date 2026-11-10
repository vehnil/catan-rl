"""Maritime (bank / port) trade rates and domestic trade helpers."""

from __future__ import annotations

import numpy as np

from catan.engine.actions import OFFER_GET, OFFER_GIVE
from catan.engine.board import NUM_VERTICES
from catan.engine.constants import GENERIC_PORT, NUM_PLAYERS, NUM_RESOURCES
from catan.engine.state import GameState


def maritime_rates(state: GameState, p: int) -> np.ndarray:
    """Cards of each resource ``p`` must give the bank for one card (2, 3 or 4)."""
    rates = np.full(NUM_RESOURCES, 4, dtype=np.int64)
    ports = state.board.vertex_port[:NUM_VERTICES][state.vertex_owner[:NUM_VERTICES] == p]
    ports = ports[ports >= 0]
    if (ports == GENERIC_PORT).any():
        rates[:] = 3
    for r in ports[ports < GENERIC_PORT]:
        rates[r] = 2
    return rates


def can_afford(hand: np.ndarray, cost: np.ndarray) -> bool:
    return bool((hand >= cost).all())


def offer_vectors(offer: int) -> tuple[np.ndarray, np.ndarray]:
    """(give, get) from the proposer's perspective."""
    return OFFER_GIVE[offer], OFFER_GET[offer]


def responders_for(proposer: int) -> list[int]:
    return [(proposer + i) % NUM_PLAYERS for i in range(1, NUM_PLAYERS)]

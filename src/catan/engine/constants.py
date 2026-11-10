"""Static game constants: resources, costs, piece limits, dev-card deck."""

from __future__ import annotations

from enum import IntEnum

import numpy as np

NUM_PLAYERS = 4
NUM_RESOURCES = 5
NUM_DEV_TYPES = 5


class Resource(IntEnum):
    WOOD = 0
    BRICK = 1
    SHEEP = 2
    WHEAT = 3
    ORE = 4


DESERT = -1
"""Terrain id for the desert hex (produces nothing)."""

RESOURCE_NAMES = ["wood", "brick", "sheep", "wheat", "ore"]
RESOURCE_SHORT = ["Wd", "Br", "Sh", "Wh", "Or"]


class DevCard(IntEnum):
    KNIGHT = 0
    VICTORY_POINT = 1
    ROAD_BUILDING = 2
    YEAR_OF_PLENTY = 3
    MONOPOLY = 4


DEV_NAMES = ["knight", "victory_point", "road_building", "year_of_plenty", "monopoly"]
DEV_DECK_COUNTS = np.array([14, 5, 2, 2, 2], dtype=np.int64)  # 25 total
BANK_PER_RESOURCE = 19


def _cost(
    wood: int = 0, brick: int = 0, sheep: int = 0, wheat: int = 0, ore: int = 0
) -> np.ndarray:
    return np.array([wood, brick, sheep, wheat, ore], dtype=np.int64)


COST_ROAD = _cost(wood=1, brick=1)
COST_SETTLEMENT = _cost(wood=1, brick=1, sheep=1, wheat=1)
COST_CITY = _cost(wheat=2, ore=3)
COST_DEV = _cost(sheep=1, wheat=1, ore=1)

MAX_ROADS = 15
MAX_SETTLEMENTS = 5
MAX_CITIES = 4

LONGEST_ROAD_MIN = 5
LARGEST_ARMY_MIN = 3

# Terrain multiset for the base board (19 hexes).
TERRAIN_COUNTS = {
    Resource.WOOD: 4,
    Resource.BRICK: 3,
    Resource.SHEEP: 4,
    Resource.WHEAT: 4,
    Resource.ORE: 3,
    DESERT: 1,
}
NUMBER_TOKENS = [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12]

# Port ids: 0..4 = 2:1 for that resource, GENERIC_PORT = 3:1.
GENERIC_PORT = 5
PORT_TYPES = [GENERIC_PORT] * 4 + [int(r) for r in Resource]

# Probability (out of 36) of each dice sum.
DICE_WEIGHTS = {s: 6 - abs(s - 7) for s in range(2, 13)}

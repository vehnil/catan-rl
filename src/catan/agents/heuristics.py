"""Hand-written position evaluation shared by greedy, search and rollout policies."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from catan.engine.board import NUM_HEXES, NUM_VERTICES, TOPOLOGY, Board
from catan.engine.constants import (
    COST_CITY,
    COST_DEV,
    COST_ROAD,
    COST_SETTLEMENT,
    DESERT,
    DICE_WEIGHTS,
    GENERIC_PORT,
    NUM_PLAYERS,
    NUM_RESOURCES,
    DevCard,
)
from catan.engine.rules import settlement_mask
from catan.engine.state import GameState


@dataclass(frozen=True)
class Weights:
    vp: float = 100.0
    production: float = 3.0
    diversity: float = 6.0
    goal_progress: float = 25.0
    over_limit: float = 6.0
    dev_card: float = 18.0
    knight: float = 8.0
    road: float = 3.0
    best_spot: float = 1.5
    port: float = 4.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


DEFAULT_WEIGHTS = Weights()

# Named strategy styles (arms for the meta-bandit, variety for tournaments).
PRESETS: dict[str, Weights] = {
    "balanced": DEFAULT_WEIGHTS,
    "expansion": Weights(road=6.0, best_spot=3.0, diversity=8.0, dev_card=10.0),
    "cities": Weights(production=4.5, best_spot=0.8, road=1.5, goal_progress=30.0),
    "development": Weights(dev_card=30.0, knight=14.0, road=2.0),
}


_PIPS_CACHE: dict[tuple[int, int], tuple[Board, np.ndarray]] = {}


def _vertex_hex_matrix() -> np.ndarray:
    vh = np.zeros((NUM_VERTICES, NUM_HEXES))
    for v in range(NUM_VERTICES):
        for h in TOPOLOGY.vertex_hexes[v]:
            if h >= 0:
                vh[v, h] = 1
    return vh


_VH = _vertex_hex_matrix()


def vertex_pips(board: Board, robber: int = -1) -> np.ndarray:
    """(54, 5) expected pips per resource for a settlement at each vertex. Cached."""
    key = (id(board), robber)
    hit = _PIPS_CACHE.get(key)
    if hit is not None and hit[0] is board:
        return hit[1]
    pips = np.array([DICE_WEIGHTS.get(int(n), 0) for n in board.numbers], dtype=np.float64)
    if robber >= 0:
        pips[robber] = 0
    hex_res = np.zeros((NUM_HEXES, NUM_RESOURCES))
    land = board.terrain != DESERT
    hex_res[np.flatnonzero(land), board.terrain[land]] = 1
    out = _VH @ (pips[:, None] * hex_res)
    out.setflags(write=False)
    if len(_PIPS_CACHE) > 2048:
        _PIPS_CACHE.clear()
    _PIPS_CACHE[key] = (board, out)
    return out


def production(state: GameState, p: int) -> np.ndarray:
    """(5,) pips per resource earned by ``p``'s buildings (robber respected)."""
    vp = vertex_pips(state.board, state.robber)
    owned = state.vertex_owner[:NUM_VERTICES] == p
    return (vp[owned] * state.vertex_level[:NUM_VERTICES][owned, None]).sum(axis=0)


def vertex_score(state: GameState, v: int, p: int, w: Weights = DEFAULT_WEIGHTS) -> float:
    """How good a settlement at ``v`` would be for ``p``."""
    vp = vertex_pips(state.board, state.robber)[v]
    have = production(state, p)
    new_types = ((vp > 0) & (have == 0)).sum()
    score = vp.sum() + 0.5 * w.diversity * new_types
    port = state.board.vertex_port[v]
    if port == GENERIC_PORT:
        score += 0.5 * w.port
    elif port >= 0:
        score += w.port * min(1.0, (have[port] + vp[port]) / 5)
    return float(score)


def _progress(hand: np.ndarray, cost: np.ndarray) -> float:
    return float(np.minimum(hand, cost).sum() / cost.sum())


def goal_progress(state: GameState, p: int) -> float:
    """Progress (0..1) toward the most valuable thing ``p`` could build next."""
    hand = state.hands[p]
    goals = [(_progress(hand, COST_DEV), 0.6)]
    if (
        state.cities_left[p] > 0
        and (state.vertex_level[:NUM_VERTICES][state.vertex_owner[:NUM_VERTICES] == p] == 1).any()
    ):
        goals.append((_progress(hand, COST_CITY), 1.0))
    if state.settlements_left[p] > 0:
        goals.append(
            (_progress(hand, COST_SETTLEMENT), 1.0 if settlement_mask(state, p).any() else 0.4)
        )
    if state.roads_left[p] > 0:
        goals.append((_progress(hand, COST_ROAD), 0.4))
    return max(prog * weight for prog, weight in goals)


def evaluate(state: GameState, p: int, w: Weights = DEFAULT_WEIGHTS) -> float:
    """Scalar utility of ``state`` for ``p`` (needs full information about ``p`` only)."""
    if state.winner == p:
        return 1e6
    if state.winner >= 0:
        return -1e6
    prod = production(state, p)
    v = w.vp * state.total_vp(p)
    v += w.production * prod.sum() + w.diversity * (prod > 0).sum()
    v += w.goal_progress * goal_progress(state, p)
    n = int(state.hands[p].sum())
    if n > state.config.discard_limit:
        v -= w.over_limit * (n - state.config.discard_limit)
    devs = state.dev_hand[p] + state.dev_new[p]
    v += w.dev_card * (devs.sum() - devs[DevCard.VICTORY_POINT])
    v += w.knight * state.knights[p]
    v += w.road * min(int(state.road_length[p]), 8)
    if state.settlements_left[p] > 0:
        spots = np.flatnonzero(settlement_mask(state, p))
        if len(spots):
            v += w.best_spot * vertex_pips(state.board, state.robber)[spots].sum(axis=1).max()
    return float(v)


def evaluate_all(state: GameState, w: Weights = DEFAULT_WEIGHTS) -> np.ndarray:
    """Utility vector for every seat (used by Max^n)."""
    return np.array([evaluate(state, p, w) for p in range(NUM_PLAYERS)])


def robber_hex_score(state: GameState, h: int, p: int, vp: np.ndarray | None = None) -> float:
    """Damage to opponents minus damage to self from putting the robber on ``h``."""
    pips = DICE_WEIGHTS.get(int(state.board.numbers[h]), 0)
    score = 0.0
    for v in TOPOLOGY.hex_vertices[h]:
        o = state.vertex_owner[v]
        if o < 0:
            continue
        dmg = pips * state.vertex_level[v]
        if o == p:
            score -= 3 * dmg
        else:
            lead = vp[o] if vp is not None else state.public_vp(o)
            score += dmg * (1 + lead / 5) + 0.5 * (state.hands[o].sum() > 0)
    return score

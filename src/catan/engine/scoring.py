"""Longest road and largest army."""

from __future__ import annotations

import numpy as np

from catan.engine.board import NUM_EDGES, NUM_VERTICES, TOPOLOGY
from catan.engine.constants import LARGEST_ARMY_MIN, LONGEST_ROAD_MIN, NUM_PLAYERS
from catan.engine.state import GameState

_EV = TOPOLOGY.edge_vertices


def longest_road(state: GameState, p: int) -> int:
    """Length of ``p``'s longest simple trail of roads.

    A trail may end at, but not pass through, a vertex occupied by an opponent.
    """
    edges = np.flatnonzero(state.edge_owner[:NUM_EDGES] == p)
    if len(edges) == 0:
        return 0
    owner = state.vertex_owner
    own_edges = set(edges.tolist())
    adj: dict[int, list[int]] = {}
    for e in own_edges:
        a, b = int(_EV[e, 0]), int(_EV[e, 1])
        adj.setdefault(a, []).append(e)
        adj.setdefault(b, []).append(e)

    used: set[int] = set()

    def dfs(v: int, length: int) -> int:
        if length > 0 and owner[v] != -1 and owner[v] != p:
            return length
        best = length
        for e in adj[v]:
            if e in used:
                continue
            used.add(e)
            a, b = int(_EV[e, 0]), int(_EV[e, 1])
            best = max(best, dfs(b if a == v else a, length + 1))
            used.discard(e)
        return best

    return max(dfs(v, 0) for v in adj)


def update_longest_road(state: GameState, players: list[int] | None = None) -> bool:
    """Recompute cached road lengths (for ``players`` or all) and the title holder.

    Returns True if the holder changed. The holder keeps the title on ties; if
    the holder is overtaken by a tie, nobody holds it.
    """
    for p in players if players is not None else range(NUM_PLAYERS):
        state.road_length[p] = longest_road(state, p)
    lengths = state.road_length
    old = state.longest_road_holder
    best = int(lengths.max())
    if old >= 0 and lengths[old] == best and best >= LONGEST_ROAD_MIN:
        return False
    if best < LONGEST_ROAD_MIN:
        new = -1
    else:
        leaders = np.flatnonzero(lengths == best)
        new = int(leaders[0]) if len(leaders) == 1 else -1
    state.longest_road_holder = new
    return new != old


def update_largest_army(state: GameState, p: int) -> bool:
    k = state.knights[p]
    holder = state.largest_army_holder
    if holder == p or k < LARGEST_ARMY_MIN:
        return False
    if holder == -1 or k > state.knights[holder]:
        state.largest_army_holder = p
        return True
    return False


def buildings_vp(state: GameState) -> np.ndarray:
    owner = state.vertex_owner[:NUM_VERTICES]
    level = state.vertex_level[:NUM_VERTICES]
    return np.array([int(level[owner == p].sum()) for p in range(NUM_PLAYERS)])

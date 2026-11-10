from __future__ import annotations

import numpy as np
import pytest

from catan.engine import Game, RulesConfig, legal_actions
from catan.engine.board import TOPOLOGY
from catan.engine.phases import Phase
from catan.engine.state import GameState


def place(state: GameState, p: int, vertex: int, level: int = 1) -> None:
    state.vertex_owner[vertex] = p
    state.vertex_level[vertex] = level
    if level == 1:
        state.settlements_left[p] -= 1
    else:
        state.cities_left[p] -= 1


def road(state: GameState, p: int, edge: int) -> None:
    state.edge_owner[edge] = p
    state.roads_left[p] -= 1


def give(state: GameState, p: int, cards: list[int]) -> None:
    """Move cards from bank to player ``p`` (keeps conservation intact)."""
    v = np.array(cards, dtype=np.int64)
    state.hands[p] += v
    state.bank -= v


def edge_between(a: int, b: int) -> int:
    for e, (x, y) in enumerate(TOPOLOGY.edge_vertices):
        if {int(x), int(y)} == {a, b}:
            return e
    raise ValueError("not adjacent")


def path_edges(
    start: int, length: int, avoid: set[int] | None = None
) -> tuple[list[int], list[int]]:
    """A simple vertex path of ``length`` edges starting at ``start`` (greedy walk)."""
    avoid = set(avoid or ())
    verts, edges = [start], []
    cur = start
    for _ in range(length):
        for nb in TOPOLOGY.vertex_vertices[cur]:
            nb = int(nb)
            if nb < len(TOPOLOGY.vertex_pos) and nb not in verts and nb not in avoid:
                edges.append(edge_between(cur, nb))
                verts.append(nb)
                cur = nb
                break
        else:
            raise ValueError("stuck")
    return verts, edges


def main_phase_game(seed: int = 0, config: RulesConfig | None = None) -> Game:
    """A game advanced straight to P0's MAIN phase with an empty board."""
    g = Game(seed, config)
    g.state.phase = Phase.MAIN
    g.state.current = 0
    g.state.setup_step = 8
    return g


def play_random(game: Game, rng: np.random.Generator, max_steps: int = 200_000) -> Game:
    for _ in range(max_steps):
        if game.done:
            break
        game.step(int(rng.choice(legal_actions(game.state))))
    return game


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)

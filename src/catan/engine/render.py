"""Plain-text rendering of a game state, for debugging, the CLI and replays."""

from __future__ import annotations

import numpy as np

from catan.engine.board import NUM_EDGES, NUM_VERTICES
from catan.engine.constants import (
    DESERT,
    DEV_NAMES,
    GENERIC_PORT,
    NUM_PLAYERS,
    RESOURCE_NAMES,
    RESOURCE_SHORT,
)
from catan.engine.state import GameState

_ROW_LENGTHS = [3, 4, 5, 4, 3]


def _hex_label(state: GameState, h: int) -> str:
    t = state.board.terrain[h]
    name = "--" if t == DESERT else RESOURCE_SHORT[t]
    num = int(state.board.numbers[h])
    robber = "R" if h == state.robber else " "
    return f"[{h:>2}:{name}{num:>3}{robber}]"


def render_board(state: GameState) -> str:
    lines = []
    h = 0
    for n in _ROW_LENGTHS:
        indent = " " * (6 * (5 - n))
        lines.append(indent + " ".join(_hex_label(state, h + i) for i in range(n)))
        h += n
    return "\n".join(lines)


def render(state: GameState, seat: int | None = None) -> str:
    """Board + per-player summary. With ``seat`` set, other players' hands are hidden."""
    out = [render_board(state), ""]
    ports = ", ".join(
        f"e{int(e)}:{'3:1' if t == GENERIC_PORT else '2:1 ' + RESOURCE_NAMES[t]}"
        for e, t in zip(state.board.port_edges, state.board.port_types, strict=True)
    )
    out.append(f"ports: {ports}")
    out.append(
        f"turn {state.turn}  phase {state.phase.name}  current P{state.current}  "
        f"actor P{state.actor}  last roll {state.last_roll}  bank {state.bank.tolist()}  "
        f"dev deck {len(state.dev_deck)}"
    )
    owner = state.vertex_owner[:NUM_VERTICES]
    level = state.vertex_level[:NUM_VERTICES]
    for p in range(NUM_PLAYERS):
        settlements = np.flatnonzero((owner == p) & (level == 1)).tolist()
        cities = np.flatnonzero((owner == p) & (level == 2)).tolist()
        roads = np.flatnonzero(state.edge_owner[:NUM_EDGES] == p).tolist()
        if seat is None or seat == p:
            hand = dict(zip(RESOURCE_SHORT, state.hands[p].tolist(), strict=True))
            devs = {
                DEV_NAMES[c]: int(state.dev_hand[p, c] + state.dev_new[p, c])
                for c in range(len(DEV_NAMES))
                if state.dev_hand[p, c] + state.dev_new[p, c]
            }
            vp = state.total_vp(p)
        else:
            hand = {"cards": int(state.hands[p].sum())}
            devs = {"hidden": int(state.dev_hand[p].sum() + state.dev_new[p].sum())}
            vp = state.public_vp(p)
        tags = []
        if state.longest_road_holder == p:
            tags.append("LR")
        if state.largest_army_holder == p:
            tags.append("LA")
        out.append(
            f"P{p}{'*' if p == state.current else ' '} vp={vp:<2} {' '.join(tags):<5} "
            f"hand={hand} dev={devs} knights={int(state.knights[p])} road={int(state.road_length[p])}"
        )
        out.append(f"     settlements={settlements} cities={cities} roads={roads}")
    if state.done:
        out.append(f"GAME OVER winner={state.winner} truncated={state.truncated}")
    return "\n".join(out)

"""Timeline export: a replay re-simulated into per-step JSON for the viewer and game logs.

``build_timeline`` spells out the board geometry, then records, for every step,
the action taken, the events it produced (spectator view plus any seat views
that differ), a compact state snapshot, and each seat's belief tracker summary.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from catan.engine import Game, RulesConfig, describe
from catan.engine.actions import OFFER_GET, OFFER_GIVE
from catan.engine.board import NUM_EDGES, NUM_VERTICES, TOPOLOGY
from catan.engine.constants import (
    DESERT,
    DEV_NAMES,
    GENERIC_PORT,
    NUM_PLAYERS,
    RESOURCE_NAMES,
)
from catan.engine.events import Event
from catan.engine.state import GameState
from catan.player.belief import BeliefTracker
from catan.sim.replay import verify
from catan.sim.runner import GameRecord

_SQ3 = math.sqrt(3)


def _jsonable(x: Any) -> Any:
    """Recursively convert numpy scalars/arrays and tuples to plain JSON types."""
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _ints(a: np.ndarray) -> list[int]:
    return [int(v) for v in a]


def board_json(state: GameState) -> dict[str, Any]:
    """Board geometry in unit-hex coordinates (hex circumradius 1, pointy top)."""
    b, t = state.board, TOPOLOGY
    hexes = [
        {
            "x": _SQ3 * (q + r / 2),
            "y": 1.5 * r,
            "terrain": "desert" if b.terrain[h] == DESERT else RESOURCE_NAMES[b.terrain[h]],
            "number": int(b.numbers[h]),
            "vertices": _ints(t.hex_vertices[h]),
        }
        for h, (q, r) in enumerate(t.hex_coords)
    ]
    vertices = [{"x": kx * _SQ3 / 2, "y": ky / 2} for kx, ky in t.vertex_pos]
    ports = [
        {
            "edge": int(e),
            "type": "3:1" if p == GENERIC_PORT else RESOURCE_NAMES[p],
        }
        for e, p in zip(b.port_edges, b.port_types, strict=True)
    ]
    return {
        "hexes": hexes,
        "vertices": vertices,
        "edges": [_ints(ev) for ev in t.edge_vertices],
        "ports": ports,
    }


def snapshot(state: GameState) -> dict[str, Any]:
    """Everything the viewer draws for one position (true, unredacted state)."""
    players = [
        {
            "hand": _ints(state.hands[p]),
            "devs": _ints(state.dev_hand[p] + state.dev_new[p]),
            "knights": int(state.knights[p]),
            "road_length": int(state.road_length[p]),
            "public_vp": state.public_vp(p),
            "total_vp": state.total_vp(p),
            "longest_road": state.longest_road_holder == p,
            "largest_army": state.largest_army_holder == p,
            "pieces_left": {
                "settlements": int(state.settlements_left[p]),
                "cities": int(state.cities_left[p]),
                "roads": int(state.roads_left[p]),
            },
        }
        for p in range(NUM_PLAYERS)
    ]
    trade = None
    if state.trade is not None:
        tr = state.trade
        trade = {
            "proposer": tr.proposer,
            "give": _ints(OFFER_GIVE[tr.offer]),
            "get": _ints(OFFER_GET[tr.offer]),
            "pending": list(tr.pending),
            "acceptors": list(tr.acceptors),
        }
    return {
        "turn": state.turn,
        "current": state.current,
        "actor": state.actor,
        "phase": state.phase.name,
        "last_roll": state.last_roll,
        "bank": _ints(state.bank),
        "dev_deck": len(state.dev_deck),
        "robber": state.robber,
        "vertex_owner": _ints(state.vertex_owner[:NUM_VERTICES]),
        "vertex_level": _ints(state.vertex_level[:NUM_VERTICES]),
        "edge_owner": _ints(state.edge_owner[:NUM_EDGES]),
        "players": players,
        "trade": trade,
        "winner": state.winner,
        "done": state.done,
    }


def event_json(e: Event) -> dict[str, Any]:
    """Spectator view, plus ``views[seat]`` for each seat whose view differs."""
    out: dict[str, Any] = _jsonable(e.view(None))
    if e.secret:
        out["views"] = {
            str(s): _jsonable(e.view(s)) for s in range(NUM_PLAYERS) if s not in e.visible_to
        }
    return out


def beliefs_json(tracker: BeliefTracker) -> dict[str, Any]:
    return {
        "hands": {str(q): _jsonable(v) for q, v in tracker.summary().items()},
        "dev_type_probs": [round(float(x), 4) for x in tracker.dev_type_probs()],
    }


def build_timeline(rec: GameRecord) -> dict[str, Any]:
    game = Game(rec.seed, RulesConfig(**rec.rules))
    trackers = [BeliefTracker(s) for s in range(NUM_PLAYERS)]
    steps: list[dict[str, Any]] = [
        {
            "action": None,
            "events": [],
            "state": snapshot(game.state),
            "beliefs": [beliefs_json(t) for t in trackers],
        }
    ]
    for a in rec.actions:
        actor = game.actor
        events = game.step(a)
        for e in events:
            for s, t in enumerate(trackers):
                t.update(e.view(s))
        steps.append(
            {
                "action": {"index": int(a), "label": describe(a), "actor": actor},
                "events": [event_json(e) for e in events],
                "state": snapshot(game.state),
                "beliefs": [beliefs_json(t) for t in trackers],
            }
        )
    return {
        "meta": {
            "seed": rec.seed,
            "rules": rec.rules,
            "agents": rec.agents,
            "winner": rec.winner,
            "turns": rec.turns,
            "truncated": rec.truncated,
            "vp": rec.vp,
            "ranks": rec.ranks,
            "verified": verify(rec),
            "resources": RESOURCE_NAMES,
            "dev_cards": DEV_NAMES,
            # (give, get) per trade offer index, from the proposer's side
            "offers": [[_ints(g), _ints(r)] for g, r in zip(OFFER_GIVE, OFFER_GET, strict=True)],
        },
        "board": board_json(game.state),
        "steps": steps,
    }

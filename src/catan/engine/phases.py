"""Turn state machine.

Transitions (see ``rules.apply`` for the full logic)::

    SETUP_SETTLEMENT -> SETUP_ROAD -> (next seat, snake order 0123 3210) ... -> ROLL
    ROLL --(7)--> DISCARD? -> MOVE_ROBBER -> STEAL? -> MAIN
    ROLL --(other)--> MAIN
    ROLL/MAIN --knight--> MOVE_ROBBER -> STEAL? -> (back to ROLL/MAIN)
    ROLL/MAIN --road building--> ROAD_BUILDING -> (back)
    MAIN --propose--> TRADE_RESPOND (each opponent) -> TRADE_CONFIRM? -> MAIN
    MAIN --end turn--> ROLL (next seat) | GAME_OVER (turn cap)
    any --10 VP on own turn--> GAME_OVER
"""

from __future__ import annotations

from enum import IntEnum


class Phase(IntEnum):
    SETUP_SETTLEMENT = 0
    SETUP_ROAD = 1
    ROLL = 2
    DISCARD = 3
    MOVE_ROBBER = 4
    STEAL = 5
    MAIN = 6
    ROAD_BUILDING = 7
    TRADE_RESPOND = 8
    TRADE_CONFIRM = 9
    GAME_OVER = 10


NUM_PHASES = len(Phase)

# Seat order for initial placement: forward then reverse ("snake").
SETUP_ORDER = (0, 1, 2, 3, 3, 2, 1, 0)

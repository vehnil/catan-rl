"""Typed game events with public / private views.

``apply`` returns a list of events describing what happened. ``data`` is
visible to everyone; ``secret`` is visible only to seats in ``visible_to``
(e.g. which card was stolen, which dev card was bought). Belief trackers and
replays consume events through ``view(seat)`` so hidden information can never
leak into a player's observation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Event:
    kind: str
    player: int
    data: dict[str, Any] = field(default_factory=dict)
    secret: dict[str, Any] = field(default_factory=dict)
    visible_to: tuple[int, ...] = ()

    def view(self, seat: int | None) -> dict[str, Any]:
        """Event as seen by ``seat`` (``None`` = omniscient spectator)."""
        out: dict[str, Any] = {"kind": self.kind, "player": self.player, **self.data}
        if self.secret and (seat is None or seat in self.visible_to):
            out.update(self.secret)
        return out


# Event kinds (string constants keep logs/replays human readable).
ROLL = "roll"
PRODUCE = "produce"
DISCARD = "discard"
MOVE_ROBBER = "move_robber"
STEAL = "steal"
BUILD_SETTLEMENT = "build_settlement"
BUILD_CITY = "build_city"
BUILD_ROAD = "build_road"
BUY_DEV = "buy_dev"
PLAY_DEV = "play_dev"
YEAR_OF_PLENTY = "year_of_plenty"
MONOPOLY = "monopoly"
MARITIME_TRADE = "maritime_trade"
PROPOSE_TRADE = "propose_trade"
RESPOND_TRADE = "respond_trade"
TRADE = "trade"
CANCEL_TRADE = "cancel_trade"
SETUP_RESOURCES = "setup_resources"
END_TURN = "end_turn"
LONGEST_ROAD = "longest_road"
LARGEST_ARMY = "largest_army"
GAME_OVER = "game_over"

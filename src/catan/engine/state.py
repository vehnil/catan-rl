"""Mutable game state. Rules live in ``rules.py``; this module only holds data.

State arrays indexed by vertex / edge have one extra sentinel slot at the end
(always empty) so padded adjacency tables from ``board`` can index them
directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from catan.engine.board import NUM_EDGES, NUM_VERTICES, Board
from catan.engine.constants import (
    BANK_PER_RESOURCE,
    DEV_DECK_COUNTS,
    MAX_CITIES,
    MAX_ROADS,
    MAX_SETTLEMENTS,
    NUM_DEV_TYPES,
    NUM_PLAYERS,
    NUM_RESOURCES,
    DevCard,
)
from catan.engine.phases import Phase


@dataclass(frozen=True)
class RulesConfig:
    vp_to_win: int = 10
    discard_limit: int = 7
    board: str = "random"  # "random" (drawn from the game seed) | "standard" (fixed layout)
    avoid_adjacent_red: bool = True
    # Loop / length limits are action counts, never wall-clock (keeps games reproducible).
    max_trade_proposals_per_turn: int = 3
    max_actions_per_turn: int = 50
    max_turns_per_game: int = 500


@dataclass(slots=True)
class TradeState:
    offer: int  # index into actions.TRADE_OFFERS (from the proposer's perspective)
    proposer: int
    pending: list[int]  # responders still to answer, in seat order
    acceptors: list[int] = field(default_factory=list)


@dataclass(slots=True)
class GameState:
    board: Board
    config: RulesConfig
    phase: Phase = Phase.SETUP_SETTLEMENT
    current: int = 0
    turn: int = 0  # completed player-turns after setup
    setup_step: int = 0
    hands: np.ndarray = field(
        default_factory=lambda: np.zeros((NUM_PLAYERS, NUM_RESOURCES), np.int64)
    )
    bank: np.ndarray = field(
        default_factory=lambda: np.full(NUM_RESOURCES, BANK_PER_RESOURCE, np.int64)
    )
    dev_deck: list[int] = field(default_factory=list)  # top of deck = end of list (hidden)
    dev_hand: np.ndarray = field(
        default_factory=lambda: np.zeros((NUM_PLAYERS, NUM_DEV_TYPES), np.int64)
    )
    dev_new: np.ndarray = field(
        default_factory=lambda: np.zeros((NUM_PLAYERS, NUM_DEV_TYPES), np.int64)
    )
    dev_played_this_turn: bool = False
    knights: np.ndarray = field(default_factory=lambda: np.zeros(NUM_PLAYERS, np.int64))
    vertex_owner: np.ndarray = field(
        default_factory=lambda: np.full(NUM_VERTICES + 1, -1, np.int64)
    )
    vertex_level: np.ndarray = field(default_factory=lambda: np.zeros(NUM_VERTICES + 1, np.int64))
    edge_owner: np.ndarray = field(default_factory=lambda: np.full(NUM_EDGES + 1, -1, np.int64))
    robber: int = 0
    settlements_left: np.ndarray = field(
        default_factory=lambda: np.full(NUM_PLAYERS, MAX_SETTLEMENTS, np.int64)
    )
    cities_left: np.ndarray = field(
        default_factory=lambda: np.full(NUM_PLAYERS, MAX_CITIES, np.int64)
    )
    roads_left: np.ndarray = field(
        default_factory=lambda: np.full(NUM_PLAYERS, MAX_ROADS, np.int64)
    )
    road_length: np.ndarray = field(default_factory=lambda: np.zeros(NUM_PLAYERS, np.int64))
    longest_road_holder: int = -1
    largest_army_holder: int = -1
    discard_pending: np.ndarray = field(default_factory=lambda: np.zeros(NUM_PLAYERS, np.int64))
    return_phase: Phase = Phase.MAIN
    free_roads: int = 0
    last_setup_vertex: int = -1
    steal_candidates: list[int] = field(default_factory=list)
    trade: TradeState | None = None
    trade_proposals_this_turn: int = 0
    rejected_offers: set[int] = field(default_factory=set)
    actions_this_turn: int = 0
    last_roll: int = 0
    winner: int = -1
    truncated: bool = False

    @staticmethod
    def new(config: RulesConfig, rng: np.random.Generator) -> GameState:
        if config.board == "random":
            board = Board.random(rng, config.avoid_adjacent_red)
        elif config.board == "standard":
            board = Board.standard()
        else:
            raise ValueError(f"unknown board {config.board!r}")
        deck = [int(c) for c in DevCard for _ in range(int(DEV_DECK_COUNTS[c]))]
        rng.shuffle(deck)
        return GameState(board=board, config=config, dev_deck=deck, robber=board.desert)

    def clone(self) -> GameState:
        t = self.trade
        return GameState(
            board=self.board,
            config=self.config,
            phase=self.phase,
            current=self.current,
            turn=self.turn,
            setup_step=self.setup_step,
            hands=self.hands.copy(),
            bank=self.bank.copy(),
            dev_deck=list(self.dev_deck),
            dev_hand=self.dev_hand.copy(),
            dev_new=self.dev_new.copy(),
            dev_played_this_turn=self.dev_played_this_turn,
            knights=self.knights.copy(),
            vertex_owner=self.vertex_owner.copy(),
            vertex_level=self.vertex_level.copy(),
            edge_owner=self.edge_owner.copy(),
            robber=self.robber,
            settlements_left=self.settlements_left.copy(),
            cities_left=self.cities_left.copy(),
            roads_left=self.roads_left.copy(),
            road_length=self.road_length.copy(),
            longest_road_holder=self.longest_road_holder,
            largest_army_holder=self.largest_army_holder,
            discard_pending=self.discard_pending.copy(),
            return_phase=self.return_phase,
            free_roads=self.free_roads,
            last_setup_vertex=self.last_setup_vertex,
            steal_candidates=list(self.steal_candidates),
            trade=None
            if t is None
            else TradeState(t.offer, t.proposer, list(t.pending), list(t.acceptors)),
            trade_proposals_this_turn=self.trade_proposals_this_turn,
            rejected_offers=set(self.rejected_offers),
            actions_this_turn=self.actions_this_turn,
            last_roll=self.last_roll,
            winner=self.winner,
            truncated=self.truncated,
        )

    # ---- derived quantities -------------------------------------------------

    @property
    def actor(self) -> int:
        """Seat that must act next (not always the player whose turn it is)."""
        if self.phase == Phase.DISCARD:
            for i in range(NUM_PLAYERS):
                p = (self.current + i) % NUM_PLAYERS
                if self.discard_pending[p] > 0:
                    return p
        if self.phase == Phase.TRADE_RESPOND and self.trade is not None:
            return self.trade.pending[0]
        return self.current

    @property
    def done(self) -> bool:
        return self.phase == Phase.GAME_OVER

    @property
    def is_setup(self) -> bool:
        return self.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD)

    def public_vp(self, p: int) -> int:
        owned = self.vertex_owner[:NUM_VERTICES] == p
        vp = int(self.vertex_level[:NUM_VERTICES][owned].sum())
        vp += 2 * (self.longest_road_holder == p) + 2 * (self.largest_army_holder == p)
        return vp

    def total_vp(self, p: int) -> int:
        hidden = self.dev_hand[p, DevCard.VICTORY_POINT] + self.dev_new[p, DevCard.VICTORY_POINT]
        return self.public_vp(p) + int(hidden)

    def ranking(self) -> list[int]:
        """Seats ordered best to worst (winner first, then by total VP)."""
        return sorted(
            range(NUM_PLAYERS),
            key=lambda p: (p != self.winner, -self.total_vp(p)),
        )

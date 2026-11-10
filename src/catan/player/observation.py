"""What one seat is allowed to see.

``observe(state, seat)`` copies every public field plus the seat's own private
cards. Hidden information (opponents' hand composition, opponents' dev-card
types, deck order) is never copied. ``redacted`` is a GameState with those
fields blanked; it exists only so ``BeliefTracker.determinize`` can fill them
back in with sampled guesses for search agents.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from catan.engine.actions import NUM_ACTIONS
from catan.engine.board import Board
from catan.engine.constants import NUM_PLAYERS, DevCard
from catan.engine.phases import Phase
from catan.engine.rules import legal_mask
from catan.engine.state import GameState, RulesConfig


@dataclass(frozen=True)
class Observation:
    seat: int
    phase: Phase
    current: int
    actor: int
    turn: int
    board: Board
    config: RulesConfig
    vertex_owner: np.ndarray
    vertex_level: np.ndarray
    edge_owner: np.ndarray
    robber: int
    bank: np.ndarray
    dev_deck_size: int
    # Own private info.
    hand: np.ndarray
    dev_hand: np.ndarray  # playable
    dev_new: np.ndarray  # bought this turn
    # Public per-player info (indexed by absolute seat).
    hand_sizes: np.ndarray
    dev_counts: np.ndarray  # unplayed dev cards held (types hidden for opponents)
    knights: np.ndarray
    public_vp: np.ndarray
    road_length: np.ndarray
    settlements_left: np.ndarray
    cities_left: np.ndarray
    roads_left: np.ndarray
    discard_pending: np.ndarray
    longest_road_holder: int
    largest_army_holder: int
    # Turn-local public info.
    last_roll: int
    dev_played_this_turn: bool
    trade_offer: int  # -1 if none
    trade_proposer: int
    trade_acceptors: tuple[int, ...]
    trade_proposals_this_turn: int
    actions_this_turn: int
    free_roads: int
    legal_mask: np.ndarray  # all False unless seat == actor
    redacted: GameState

    @property
    def is_my_turn(self) -> bool:
        return self.actor == self.seat

    @property
    def my_vp(self) -> int:
        hidden = self.dev_hand[DevCard.VICTORY_POINT] + self.dev_new[DevCard.VICTORY_POINT]
        return int(self.public_vp[self.seat] + hidden)

    def relative(self, absolute_seat: int) -> int:
        return (absolute_seat - self.seat) % NUM_PLAYERS


def redact(state: GameState, seat: int) -> GameState:
    r = state.clone()
    for q in range(NUM_PLAYERS):
        if q != seat:
            r.hands[q] = 0
            r.dev_hand[q] = 0
            r.dev_new[q] = 0
    r.dev_deck = []
    return r


def observe(state: GameState, seat: int) -> Observation:
    t = state.trade
    mask = legal_mask(state) if state.actor == seat else np.zeros(NUM_ACTIONS, dtype=bool)
    return Observation(
        seat=seat,
        phase=state.phase,
        current=state.current,
        actor=state.actor,
        turn=state.turn,
        board=state.board,
        config=state.config,
        vertex_owner=state.vertex_owner.copy(),
        vertex_level=state.vertex_level.copy(),
        edge_owner=state.edge_owner.copy(),
        robber=state.robber,
        bank=state.bank.copy(),
        dev_deck_size=len(state.dev_deck),
        hand=state.hands[seat].copy(),
        dev_hand=state.dev_hand[seat].copy(),
        dev_new=state.dev_new[seat].copy(),
        hand_sizes=state.hands.sum(axis=1),
        dev_counts=state.dev_hand.sum(axis=1) + state.dev_new.sum(axis=1),
        knights=state.knights.copy(),
        public_vp=np.array([state.public_vp(p) for p in range(NUM_PLAYERS)]),
        road_length=state.road_length.copy(),
        settlements_left=state.settlements_left.copy(),
        cities_left=state.cities_left.copy(),
        roads_left=state.roads_left.copy(),
        discard_pending=state.discard_pending.copy(),
        longest_road_holder=state.longest_road_holder,
        largest_army_holder=state.largest_army_holder,
        last_roll=state.last_roll,
        dev_played_this_turn=state.dev_played_this_turn,
        trade_offer=-1 if t is None else t.offer,
        trade_proposer=-1 if t is None else t.proposer,
        trade_acceptors=() if t is None else tuple(t.acceptors),
        trade_proposals_this_turn=state.trade_proposals_this_turn,
        actions_this_turn=state.actions_this_turn,
        free_roads=state.free_roads,
        legal_mask=mask,
        redacted=redact(state, seat),
    )

"""Rules: legal-action masks and state transitions.

``legal_mask(state)`` returns a boolean vector over the flat action space for
``state.actor``. ``apply(state, action, rng)`` mutates ``state`` in place and
returns the events produced. All randomness (dice, steals, dev draws already
fixed by the shuffled deck) goes through ``rng``, so a seed plus the action
sequence reproduces a game exactly.
"""

from __future__ import annotations

import numpy as np

from catan.engine import events as ev
from catan.engine.actions import (
    MARITIME_PAIRS,
    NUM_ACTIONS,
    OFFER_GET,
    OFFER_GIVE,
    YOP_PAIRS,
    ActionType,
    block,
    decode,
    encode,
)
from catan.engine.board import NUM_EDGES, NUM_VERTICES, TOPOLOGY
from catan.engine.constants import (
    COST_CITY,
    COST_DEV,
    COST_ROAD,
    COST_SETTLEMENT,
    DESERT,
    NUM_PLAYERS,
    NUM_RESOURCES,
    DevCard,
)
from catan.engine.events import Event
from catan.engine.phases import SETUP_ORDER, Phase
from catan.engine.scoring import update_largest_army, update_longest_road
from catan.engine.state import GameState, TradeState
from catan.engine.trading import maritime_rates, responders_for

_HV = TOPOLOGY.hex_vertices
_EV = TOPOLOGY.edge_vertices
_VV = TOPOLOGY.vertex_vertices
_VE = TOPOLOGY.vertex_edges
_VH = TOPOLOGY.vertex_hexes


class IllegalActionError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Placement masks (vectorized over vertices / edges)
# ---------------------------------------------------------------------------


def _distance_ok(state: GameState) -> np.ndarray:
    """Vertices that are empty and have no adjacent building."""
    owner = state.vertex_owner
    empty = owner[:NUM_VERTICES] == -1
    nb_empty = (owner[_VV] == -1).all(axis=1)
    return empty & nb_empty


def settlement_mask(state: GameState, p: int, setup: bool = False) -> np.ndarray:
    ok = _distance_ok(state)
    if not setup:
        ok &= (state.edge_owner[_VE] == p).any(axis=1)
    return ok


def city_mask(state: GameState, p: int) -> np.ndarray:
    return (state.vertex_owner[:NUM_VERTICES] == p) & (state.vertex_level[:NUM_VERTICES] == 1)


def road_mask(state: GameState, p: int) -> np.ndarray:
    owner = state.vertex_owner[:NUM_VERTICES]
    touches_own_road = (state.edge_owner[_VE] == p).any(axis=1)
    # A road can extend from a vertex we own, or an unoccupied vertex our road reaches.
    access = (owner == p) | ((owner == -1) & touches_own_road)
    access = np.append(access, False)
    empty = state.edge_owner[:NUM_EDGES] == -1
    return empty & (access[_EV[:, 0]] | access[_EV[:, 1]])


def setup_road_mask(state: GameState) -> np.ndarray:
    m = np.zeros(NUM_EDGES, dtype=bool)
    for e in _VE[state.last_setup_vertex]:
        if e < NUM_EDGES and state.edge_owner[e] == -1:
            m[e] = True
    return m


# ---------------------------------------------------------------------------
# Legal mask
# ---------------------------------------------------------------------------


def legal_mask(state: GameState) -> np.ndarray:
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    phase = state.phase
    p = state.actor

    if phase == Phase.GAME_OVER:
        return mask
    if phase == Phase.SETUP_SETTLEMENT:
        mask[block(ActionType.BUILD_SETTLEMENT)] = settlement_mask(state, p, setup=True)
    elif phase == Phase.SETUP_ROAD:
        mask[block(ActionType.BUILD_ROAD)] = setup_road_mask(state)
    elif phase == Phase.ROLL:
        mask[encode(ActionType.ROLL)] = True
        _dev_play_mask(state, p, mask)
    elif phase == Phase.DISCARD:
        mask[block(ActionType.DISCARD)] = state.hands[p] > 0
    elif phase == Phase.MOVE_ROBBER:
        m = np.ones(len(_HV), dtype=bool)
        m[state.robber] = False
        mask[block(ActionType.MOVE_ROBBER)] = m
    elif phase == Phase.STEAL:
        for victim in state.steal_candidates:
            mask[encode(ActionType.STEAL, (victim - p) % NUM_PLAYERS - 1)] = True
    elif phase == Phase.ROAD_BUILDING:
        mask[block(ActionType.BUILD_ROAD)] = road_mask(state, p)
    elif phase == Phase.TRADE_RESPOND:
        assert state.trade is not None
        mask[encode(ActionType.REJECT_TRADE)] = True
        if (state.hands[p] >= OFFER_GET[state.trade.offer]).all():
            mask[encode(ActionType.ACCEPT_TRADE)] = True
    elif phase == Phase.TRADE_CONFIRM:
        assert state.trade is not None
        mask[encode(ActionType.CANCEL_TRADE)] = True
        for a in state.trade.acceptors:
            mask[encode(ActionType.CONFIRM_TRADE, (a - p) % NUM_PLAYERS - 1)] = True
    elif phase == Phase.MAIN:
        mask[encode(ActionType.END_TURN)] = True
        if state.actions_this_turn >= state.config.max_actions_per_turn:
            return mask
        _main_mask(state, p, mask)
    return mask


def _dev_play_mask(state: GameState, p: int, mask: np.ndarray) -> None:
    if state.dev_played_this_turn:
        return
    held = state.dev_hand[p]
    if held[DevCard.KNIGHT] > 0:
        mask[encode(ActionType.PLAY_KNIGHT)] = True
    if held[DevCard.ROAD_BUILDING] > 0 and state.roads_left[p] > 0 and road_mask(state, p).any():
        mask[encode(ActionType.PLAY_ROAD_BUILDING)] = True
    if held[DevCard.YEAR_OF_PLENTY] > 0:
        o = block(ActionType.PLAY_YEAR_OF_PLENTY).start
        for i, (r1, r2) in enumerate(YOP_PAIRS):
            need = 2 if r1 == r2 else 1
            if state.bank[r1] >= need and state.bank[r2] >= 1:
                mask[o + i] = True
    if held[DevCard.MONOPOLY] > 0:
        mask[block(ActionType.PLAY_MONOPOLY)] = True


def _main_mask(state: GameState, p: int, mask: np.ndarray) -> None:
    hand = state.hands[p]
    if (hand >= COST_SETTLEMENT).all() and state.settlements_left[p] > 0:
        mask[block(ActionType.BUILD_SETTLEMENT)] = settlement_mask(state, p)
    if (hand >= COST_CITY).all() and state.cities_left[p] > 0:
        mask[block(ActionType.BUILD_CITY)] = city_mask(state, p)
    if (hand >= COST_ROAD).all() and state.roads_left[p] > 0:
        mask[block(ActionType.BUILD_ROAD)] = road_mask(state, p)
    if (hand >= COST_DEV).all() and state.dev_deck:
        mask[encode(ActionType.BUY_DEV)] = True
    _dev_play_mask(state, p, mask)

    rates = maritime_rates(state, p)
    o = block(ActionType.MARITIME_TRADE).start
    for i, (g, r) in enumerate(MARITIME_PAIRS):
        if hand[g] >= rates[g] and state.bank[r] > 0:
            mask[o + i] = True

    if state.trade_proposals_this_turn < state.config.max_trade_proposals_per_turn:
        offers = (hand >= OFFER_GIVE).all(axis=1)
        for i in state.rejected_offers:
            offers[i] = False
        mask[block(ActionType.PROPOSE_TRADE)] = offers


def legal_actions(state: GameState) -> np.ndarray:
    return np.flatnonzero(legal_mask(state))


# ---------------------------------------------------------------------------
# Transitions
# ---------------------------------------------------------------------------


def apply(
    state: GameState, action: int, rng: np.random.Generator, validate: bool = True
) -> list[Event]:
    if validate and not legal_mask(state)[action]:
        raise IllegalActionError(
            f"action {action} ({decode(action)}) illegal in {state.phase.name}"
        )
    t, arg = decode(action)
    p = state.actor
    out: list[Event] = []
    if not state.is_setup:
        state.actions_this_turn += 1

    match t:
        case ActionType.BUILD_SETTLEMENT:
            _build_settlement(state, p, arg, out)
        case ActionType.BUILD_ROAD:
            _build_road(state, p, arg, out)
        case ActionType.BUILD_CITY:
            _pay(state, p, COST_CITY)
            state.vertex_level[arg] = 2
            state.cities_left[p] -= 1
            state.settlements_left[p] += 1
            out.append(Event(ev.BUILD_CITY, p, {"vertex": arg}))
        case ActionType.ROLL:
            _roll(state, p, rng, out)
        case ActionType.DISCARD:
            state.hands[p, arg] -= 1
            state.bank[arg] += 1
            state.discard_pending[p] -= 1
            out.append(Event(ev.DISCARD, p, {"resource": arg}))
            if not state.discard_pending.any():
                state.phase = Phase.MOVE_ROBBER
        case ActionType.MOVE_ROBBER:
            _move_robber(state, p, arg, rng, out)
        case ActionType.STEAL:
            _steal(state, p, (p + arg + 1) % NUM_PLAYERS, rng, out)
            state.phase = state.return_phase
        case ActionType.BUY_DEV:
            _pay(state, p, COST_DEV)
            card = state.dev_deck.pop()
            state.dev_new[p, card] += 1
            out.append(Event(ev.BUY_DEV, p, {}, {"card": card}, (p,)))
        case ActionType.PLAY_KNIGHT:
            _use_dev(state, p, DevCard.KNIGHT, out)
            state.knights[p] += 1
            if update_largest_army(state, p):
                out.append(Event(ev.LARGEST_ARMY, p))
            state.return_phase = state.phase
            state.phase = Phase.MOVE_ROBBER
        case ActionType.PLAY_ROAD_BUILDING:
            _use_dev(state, p, DevCard.ROAD_BUILDING, out)
            state.free_roads = int(min(2, state.roads_left[p]))
            state.return_phase = state.phase
            state.phase = Phase.ROAD_BUILDING
        case ActionType.PLAY_YEAR_OF_PLENTY:
            _use_dev(state, p, DevCard.YEAR_OF_PLENTY, out)
            for r in YOP_PAIRS[arg]:
                state.bank[r] -= 1
                state.hands[p, r] += 1
            out.append(Event(ev.YEAR_OF_PLENTY, p, {"resources": list(YOP_PAIRS[arg])}))
        case ActionType.PLAY_MONOPOLY:
            _use_dev(state, p, DevCard.MONOPOLY, out)
            taken = {}
            for q in range(NUM_PLAYERS):
                if q != p and state.hands[q, arg] > 0:
                    taken[q] = int(state.hands[q, arg])
                    state.hands[p, arg] += state.hands[q, arg]
                    state.hands[q, arg] = 0
            out.append(Event(ev.MONOPOLY, p, {"resource": arg, "taken": taken}))
        case ActionType.MARITIME_TRADE:
            g, r = MARITIME_PAIRS[arg]
            rate = int(maritime_rates(state, p)[g])
            state.hands[p, g] -= rate
            state.bank[g] += rate
            state.bank[r] -= 1
            state.hands[p, r] += 1
            out.append(Event(ev.MARITIME_TRADE, p, {"give": g, "count": rate, "get": r}))
        case ActionType.PROPOSE_TRADE:
            state.trade_proposals_this_turn += 1
            state.trade = TradeState(offer=arg, proposer=p, pending=responders_for(p))
            state.phase = Phase.TRADE_RESPOND
            out.append(Event(ev.PROPOSE_TRADE, p, {"offer": arg}))
            _advance_trade(state, out)
        case ActionType.ACCEPT_TRADE | ActionType.REJECT_TRADE:
            assert state.trade is not None
            accepted = t == ActionType.ACCEPT_TRADE
            state.trade.pending.pop(0)
            if accepted:
                state.trade.acceptors.append(p)
            out.append(Event(ev.RESPOND_TRADE, p, {"accept": accepted}))
            _advance_trade(state, out)
        case ActionType.CONFIRM_TRADE:
            assert state.trade is not None
            partner = (p + arg + 1) % NUM_PLAYERS
            give, get = OFFER_GIVE[state.trade.offer], OFFER_GET[state.trade.offer]
            state.hands[p] += get - give
            state.hands[partner] += give - get
            out.append(Event(ev.TRADE, p, {"partner": partner, "offer": state.trade.offer}))
            state.trade = None
            state.phase = Phase.MAIN
        case ActionType.CANCEL_TRADE:
            out.append(Event(ev.CANCEL_TRADE, p))
            state.trade = None
            state.phase = Phase.MAIN
        case ActionType.END_TURN:
            _end_turn(state, p, out)
            return out

    _check_win(state, out)
    return out


def _pay(state: GameState, p: int, cost: np.ndarray) -> None:
    state.hands[p] -= cost
    state.bank += cost


def _use_dev(state: GameState, p: int, card: DevCard, out: list[Event]) -> None:
    state.dev_hand[p, card] -= 1
    state.dev_played_this_turn = True
    out.append(Event(ev.PLAY_DEV, p, {"card": int(card)}))


def _build_settlement(state: GameState, p: int, v: int, out: list[Event]) -> None:
    setup = state.phase == Phase.SETUP_SETTLEMENT
    if not setup:
        _pay(state, p, COST_SETTLEMENT)
    state.vertex_owner[v] = p
    state.vertex_level[v] = 1
    state.settlements_left[p] -= 1
    out.append(Event(ev.BUILD_SETTLEMENT, p, {"vertex": v, "free": setup}))
    if setup:
        state.last_setup_vertex = v
        state.phase = Phase.SETUP_ROAD
        if state.setup_step >= NUM_PLAYERS:  # second placement round pays out
            gained = np.zeros(NUM_RESOURCES, dtype=np.int64)
            for h in _VH[v]:
                if h >= 0 and state.board.terrain[h] != DESERT:
                    gained[state.board.terrain[h]] += 1
            state.hands[p] += gained
            state.bank -= gained
            out.append(Event(ev.SETUP_RESOURCES, p, {"resources": gained.tolist()}))
    else:
        # A new settlement can cut an opponent's road.
        _refresh_longest_road(state, None, out)


def _build_road(state: GameState, p: int, e: int, out: list[Event]) -> None:
    phase = state.phase
    if phase == Phase.MAIN:
        _pay(state, p, COST_ROAD)
    state.edge_owner[e] = p
    state.roads_left[p] -= 1
    out.append(Event(ev.BUILD_ROAD, p, {"edge": e, "free": phase != Phase.MAIN}))
    _refresh_longest_road(state, [p], out)
    if phase == Phase.SETUP_ROAD:
        state.setup_step += 1
        if state.setup_step == len(SETUP_ORDER):
            state.current = 0
            state.phase = Phase.ROLL
        else:
            state.current = SETUP_ORDER[state.setup_step]
            state.phase = Phase.SETUP_SETTLEMENT
    elif phase == Phase.ROAD_BUILDING:
        state.free_roads -= 1
        if state.free_roads <= 0 or state.roads_left[p] == 0 or not road_mask(state, p).any():
            state.free_roads = 0
            state.phase = state.return_phase


def _refresh_longest_road(state: GameState, players: list[int] | None, out: list[Event]) -> None:
    if update_longest_road(state, players):
        out.append(Event(ev.LONGEST_ROAD, state.longest_road_holder))


def apply_roll(state: GameState, d1: int, d2: int) -> list[Event]:
    """Resolve a ROLL with chosen dice (chance nodes in search, scripted tests)."""
    if state.phase != Phase.ROLL:
        raise IllegalActionError(f"cannot roll in {state.phase.name}")
    state.actions_this_turn += 1
    out: list[Event] = []
    _resolve_roll(state, state.current, d1, d2, out)
    _check_win(state, out)
    return out


def _roll(state: GameState, p: int, rng: np.random.Generator, out: list[Event]) -> None:
    d1, d2 = (int(x) for x in rng.integers(1, 7, size=2))
    _resolve_roll(state, p, d1, d2, out)


def _resolve_roll(state: GameState, p: int, d1: int, d2: int, out: list[Event]) -> None:
    s = d1 + d2
    state.last_roll = s
    out.append(Event(ev.ROLL, p, {"dice": (d1, d2), "sum": s}))
    if s == 7:
        totals = state.hands.sum(axis=1)
        over = totals > state.config.discard_limit
        state.discard_pending[:] = np.where(over, totals // 2, 0)
        state.return_phase = Phase.MAIN
        state.phase = Phase.DISCARD if over.any() else Phase.MOVE_ROBBER
        return
    _produce(state, s, out)
    state.phase = Phase.MAIN


def _produce(state: GameState, s: int, out: list[Event]) -> None:
    owed = np.zeros((NUM_PLAYERS, NUM_RESOURCES), dtype=np.int64)
    board = state.board
    for h in board.hexes_by_number[s]:
        if h == state.robber:
            continue
        r = board.terrain[h]
        for v in _HV[h]:
            o = state.vertex_owner[v]
            if o >= 0:
                owed[o, r] += state.vertex_level[v]
    for r in range(NUM_RESOURCES):
        col = owed[:, r]
        total = col.sum()
        if total == 0 or total <= state.bank[r]:
            continue
        # Bank shortage: one recipient gets what's left, several get nothing.
        recipients = np.flatnonzero(col)
        if len(recipients) == 1:
            col[recipients[0]] = state.bank[r]
        else:
            col[:] = 0
    state.hands += owed
    state.bank -= owed.sum(axis=0)
    if owed.any():
        out.append(Event(ev.PRODUCE, -1, {"gains": owed.tolist()}))


def _move_robber(
    state: GameState, p: int, h: int, rng: np.random.Generator, out: list[Event]
) -> None:
    state.robber = h
    out.append(Event(ev.MOVE_ROBBER, p, {"hex": h}))
    victims = sorted(
        {
            int(o)
            for o in state.vertex_owner[_HV[h]]
            if o >= 0 and o != p and state.hands[o].sum() > 0
        }
    )
    if len(victims) == 0:
        state.phase = state.return_phase
    elif len(victims) == 1:
        _steal(state, p, victims[0], rng, out)
        state.phase = state.return_phase
    else:
        state.steal_candidates = victims
        state.phase = Phase.STEAL


def _steal(
    state: GameState, thief: int, victim: int, rng: np.random.Generator, out: list[Event]
) -> None:
    hand = state.hands[victim]
    r = int(rng.choice(NUM_RESOURCES, p=hand / hand.sum()))
    state.hands[victim, r] -= 1
    state.hands[thief, r] += 1
    state.steal_candidates = []
    out.append(Event(ev.STEAL, thief, {"victim": victim}, {"resource": r}, (thief, victim)))


def _advance_trade(state: GameState, out: list[Event]) -> None:
    """Skip responders who cannot afford the offer; resolve when nobody is left."""
    trade = state.trade
    assert trade is not None
    need = OFFER_GET[trade.offer]
    while trade.pending and not (state.hands[trade.pending[0]] >= need).all():
        q = trade.pending.pop(0)
        # Looks like an ordinary rejection to everyone but q (real players can't see why).
        out.append(Event(ev.RESPOND_TRADE, q, {"accept": False}, {"auto": True}, (q,)))
    if trade.pending:
        return
    if trade.acceptors:
        state.phase = Phase.TRADE_CONFIRM
    else:
        state.rejected_offers.add(trade.offer)
        state.trade = None
        state.phase = Phase.MAIN


def _end_turn(state: GameState, p: int, out: list[Event]) -> None:
    state.dev_hand[p] += state.dev_new[p]
    state.dev_new[p] = 0
    state.dev_played_this_turn = False
    state.trade_proposals_this_turn = 0
    state.rejected_offers.clear()
    state.actions_this_turn = 0
    state.turn += 1
    state.current = (p + 1) % NUM_PLAYERS
    state.phase = Phase.ROLL
    out.append(Event(ev.END_TURN, p))
    if state.turn >= state.config.max_turns_per_game:
        state.truncated = True
        state.phase = Phase.GAME_OVER
        out.append(Event(ev.GAME_OVER, -1, {"truncated": True}))
        return
    _check_win(state, out)  # VP gained off-turn (e.g. longest road via a broken road)


def _check_win(state: GameState, out: list[Event]) -> None:
    if state.phase in (Phase.GAME_OVER,) or state.is_setup:
        return
    p = state.current
    if state.total_vp(p) >= state.config.vp_to_win:
        state.winner = p
        state.phase = Phase.GAME_OVER
        out.append(Event(ev.GAME_OVER, p, {"winner": p, "truncated": False}))

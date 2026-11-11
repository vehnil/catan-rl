"""Greedy one-step lookahead over a hand-tuned evaluation function.

For deterministic actions (builds, bank trades, year of plenty, free roads)
the agent applies the action to its own copy of the public state and keeps the
best strict improvement. Stochastic / special decisions (robber, steals,
discards, dev plays, domestic trades) use dedicated heuristics.

Also serves as the default rollout policy for MCTS.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from catan.agents.heuristics import (
    DEFAULT_WEIGHTS,
    PRESETS,
    Weights,
    evaluate,
    evaluate_hands,
    robber_hex_score,
    vertex_pips,
    vertex_score,
)
from catan.engine.actions import (
    OFFER_GET,
    OFFER_GIVE,
    ActionType,
    block,
    decode,
    encode,
)
from catan.engine.board import NUM_VERTICES, TOPOLOGY
from catan.engine.constants import COST_DEV, NUM_PLAYERS
from catan.engine.phases import Phase
from catan.engine.rules import apply, settlement_mask
from catan.engine.state import GameState
from catan.player.base import Agent
from catan.player.observation import Observation

_DUMMY_RNG = np.random.default_rng(0)
_LOOKAHEAD_TYPES = {
    ActionType.BUILD_SETTLEMENT,
    ActionType.BUILD_CITY,
    ActionType.BUILD_ROAD,
    ActionType.MARITIME_TRADE,
    ActionType.PLAY_YEAR_OF_PLENTY,
}


def _after(state: GameState, action: int) -> GameState:
    s = state.clone()
    apply(s, action, _DUMMY_RNG, validate=False)
    return s


class GreedyAgent(Agent):
    name = "greedy"

    def __init__(
        self,
        weights: Weights | dict[str, float] = DEFAULT_WEIGHTS,
        max_proposals: int = 1,
        accept_margin: float = 2.0,
        epsilon: float = 0.0,
        preset: str | None = None,
    ) -> None:
        super().__init__()
        if preset is not None:
            weights = PRESETS[preset]
        self.w = Weights(**weights) if isinstance(weights, dict) else weights
        self.max_proposals = max_proposals
        self.accept_margin = accept_margin
        self.epsilon = epsilon  # exploration (useful as a rollout policy)

    def config(self) -> dict[str, Any]:
        return {
            "weights": self.w.to_dict(),
            "max_proposals": self.max_proposals,
            "accept_margin": self.accept_margin,
            "epsilon": self.epsilon,
        }

    # ------------------------------------------------------------------ API

    def act(self, obs: Observation) -> int:
        return self.act_state(self._guess_state(obs), obs.legal_mask, obs.seat)

    def _guess_state(self, obs: Observation) -> GameState:
        """Public state with opponents' hands filled in by their most likely value."""
        s = obs.redacted.clone()
        for q in range(NUM_PLAYERS):
            if q == obs.seat:
                continue
            if self.belief is not None:
                s.hands[q] = self.belief.map_hand(q)
            else:  # spread the known hand size evenly, rounding down
                s.hands[q] = obs.hand_sizes[q] // 5
        return s

    def act_state(self, state: GameState, mask: np.ndarray, me: int) -> int:
        """Decide from a (possibly redacted) state. Used directly by search rollouts."""
        legal = np.flatnonzero(mask)
        if len(legal) == 1:
            return int(legal[0])
        if self.epsilon and self.rng.random() < self.epsilon:
            return int(self.rng.choice(legal))
        match state.phase:
            case Phase.SETUP_SETTLEMENT:
                return self._best_vertex(state, legal, me)
            case Phase.SETUP_ROAD | Phase.ROAD_BUILDING:
                return self._best_road(state, legal, me)
            case Phase.ROLL:
                knight = encode(ActionType.PLAY_KNIGHT)
                if mask[knight] and self._robber_hurts_me(state, me):
                    return knight
                return encode(ActionType.ROLL)
            case Phase.DISCARD:
                return self._best_by_eval(state, legal, me)
            case Phase.MOVE_ROBBER:
                return self._best_robber(state, legal, me)
            case Phase.STEAL:
                return self._best_steal(state, legal, me)
            case Phase.TRADE_RESPOND:
                return self._respond(state, me)
            case Phase.TRADE_CONFIRM:
                return self._confirm(state, legal, me)
            case _:
                return self._main(state, mask, legal, me)

    # ------------------------------------------------------------ phases

    def _best_vertex(self, state: GameState, legal: np.ndarray, me: int) -> int:
        o = block(ActionType.BUILD_SETTLEMENT).start
        scores = [vertex_score(state, int(a) - o, me, self.w) for a in legal]
        return int(legal[int(np.argmax(scores))])

    def _best_road(self, state: GameState, legal: np.ndarray, me: int) -> int:
        best, best_score = int(legal[0]), -np.inf
        for a in legal:
            s = _after(state, int(a))
            score = 0.0
            spots = np.flatnonzero(_frontier(s, me))
            if len(spots):
                score = max(vertex_score(s, int(v), me, self.w) for v in spots)
            score += 0.5 * s.road_length[me]
            if score > best_score:
                best, best_score = int(a), score
        return best

    def _best_by_eval(self, state: GameState, legal: np.ndarray, me: int) -> int:
        scores = [evaluate(_after(state, int(a)), me, self.w) for a in legal]
        return int(legal[int(np.argmax(scores))])

    def _best_robber(self, state: GameState, legal: np.ndarray, me: int) -> int:
        o = block(ActionType.MOVE_ROBBER).start
        vp = np.array([state.public_vp(p) for p in range(NUM_PLAYERS)])
        scores = [robber_hex_score(state, int(a) - o, me, vp) for a in legal]
        return int(legal[int(np.argmax(scores))])

    def _best_steal(self, state: GameState, legal: np.ndarray, me: int) -> int:
        def key(a: int) -> tuple[int, float]:
            victim = (me + decode(a).arg + 1) % NUM_PLAYERS
            return state.public_vp(victim), int(state.hands[victim].sum())

        return int(max(legal, key=lambda a: key(int(a))))

    def _respond(self, state: GameState, me: int) -> int:
        reject = encode(ActionType.REJECT_TRADE)
        accept = encode(ActionType.ACCEPT_TRADE)
        t = state.trade
        assert t is not None
        if state.public_vp(t.proposer) >= state.config.vp_to_win - 2:
            return reject
        if not (state.hands[me] >= OFFER_GET[t.offer]).all():
            return reject
        s = state.clone()
        s.hands[me] += OFFER_GIVE[t.offer] - OFFER_GET[t.offer]
        gain = evaluate(s, me, self.w) - evaluate(state, me, self.w)
        return accept if gain > self.accept_margin else reject

    def _confirm(self, state: GameState, legal: np.ndarray, me: int) -> int:
        confirms = [a for a in legal if decode(int(a)).type == ActionType.CONFIRM_TRADE]
        if not confirms:
            return encode(ActionType.CANCEL_TRADE)
        return int(
            min(
                confirms, key=lambda a: state.public_vp((me + decode(int(a)).arg + 1) % NUM_PLAYERS)
            )
        )

    def _main(self, state: GameState, mask: np.ndarray, legal: np.ndarray, me: int) -> int:
        base = evaluate(state, me, self.w)
        best, best_v = encode(ActionType.END_TURN), base + 1e-6
        for a in legal:
            if decode(int(a)).type in _LOOKAHEAD_TYPES:
                v = evaluate(_after(state, int(a)), me, self.w)
                if v > best_v:
                    best, best_v = int(a), v

        buy = encode(ActionType.BUY_DEV)
        if mask[buy]:
            s = state.clone()
            s.hands[me] -= COST_DEV
            v = evaluate(s, me, self.w) + self.w.dev_card
            if v > best_v:
                best, best_v = buy, v

        dev = self._dev_play(state, mask, me)
        if dev is not None and best == encode(ActionType.END_TURN):
            return dev

        if (
            best == encode(ActionType.END_TURN)
            and state.trade_proposals_this_turn < self.max_proposals
        ):
            offer = self._best_offer(state, mask, me, base)
            if offer is not None:
                return offer
        return best

    # ------------------------------------------------------------ helpers

    def _dev_play(self, state: GameState, mask: np.ndarray, me: int) -> int | None:
        knight = encode(ActionType.PLAY_KNIGHT)
        if mask[knight]:
            close_to_army = state.knights[me] + 1 >= 3 and state.largest_army_holder != me
            if self._robber_hurts_me(state, me) or close_to_army:
                return knight
        mono = block(ActionType.PLAY_MONOPOLY)
        if mask[mono].any() and self.belief is not None:
            expected = np.sum([self.belief.hand_marginals(q) for q in self.belief.opponents], axis=0)
            r = int(np.argmax(expected))
            if expected[r] >= 3:
                return mono.start + r
        rb = encode(ActionType.PLAY_ROAD_BUILDING)
        if mask[rb] and not _frontier(state, me).any():
            return rb
        return None

    def _best_offer(self, state: GameState, mask: np.ndarray, me: int, base: float) -> int | None:
        offers = np.flatnonzero(mask[block(ActionType.PROPOSE_TRADE)])
        if len(offers) == 0:
            return None
        # Don't ask for a lopsided deal nobody will take.
        offers = offers[OFFER_GIVE[offers].sum(axis=1) >= OFFER_GET[offers].sum(axis=1)]
        if len(offers) == 0:
            return None
        hands = state.hands[me] + OFFER_GET[offers] - OFFER_GIVE[offers]
        gains = evaluate_hands(state, me, hands, self.w) - base
        i = int(np.argmax(gains))  # first of equal gains, like a strict-improvement scan
        if gains[i] > self.accept_margin * 2:
            return block(ActionType.PROPOSE_TRADE).start + int(offers[i])
        return None

    def _robber_hurts_me(self, state: GameState, me: int) -> bool:
        return any(state.vertex_owner[v] == me for v in TOPOLOGY.hex_vertices[state.robber]) and (
            vertex_pips(state.board)[TOPOLOGY.hex_vertices[state.robber]].sum() > 0
        )


def _frontier(state: GameState, p: int) -> np.ndarray:
    """Settlement spots reachable by ``p``'s existing roads."""
    if state.settlements_left[p] == 0:
        return np.zeros(NUM_VERTICES, dtype=bool)
    return settlement_mask(state, p)

"""Belief tracking over hidden information, from one seat's point of view.

Almost everything in Catan is public: production, builds, trades, discards and
monopolies all reveal exact card movements. Uncertainty about an opponent's
hand comes only from robber steals between *other* players, plus the types of
undisclosed development cards.

Hands are modelled as a hidden Markov process per opponent. Each opponent
keeps a distribution over 5-resource count vectors, updated by:

* **transitions**: known gains/losses shift every hypothesis; an unseen steal
  branches each hypothesis by which card was taken;
* **evidence**: paying a cost, offering or accepting a trade, discarding a
  card, or a monopoly payout filters out hypotheses that couldn't have done it.

Two filters run side by side over the same events:

* the **forward filter** (sum-product) gives the posterior marginal per hand;
* the **Viterbi filter** (max-product) gives the hand at the end of the single
  most likely history of hidden steals.

Opponents are tracked independently (a factored approximation): when one
opponent steals from another, the thief receives the victim's expected
card-type distribution rather than a jointly-correlated update. The exact
per-resource totals held by opponents (known from the bank) are enforced when
sampling determinizations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from catan.engine import events as ev
from catan.engine.actions import OFFER_GET, OFFER_GIVE
from catan.engine.constants import (
    BANK_PER_RESOURCE,
    COST_CITY,
    COST_DEV,
    COST_ROAD,
    COST_SETTLEMENT,
    DEV_DECK_COUNTS,
    NUM_DEV_TYPES,
    NUM_PLAYERS,
    NUM_RESOURCES,
    DevCard,
)
from catan.engine.state import GameState

_EYE = np.eye(NUM_RESOURCES, dtype=np.int64)


@dataclass
class HandDistribution:
    """Weighted set of possible hands. ``mode`` is "sum" (forward) or "max" (Viterbi)."""

    hands: np.ndarray = field(default_factory=lambda: np.zeros((1, NUM_RESOURCES), np.int64))
    weights: np.ndarray = field(default_factory=lambda: np.ones(1))
    mode: str = "sum"
    max_support: int = 4096

    def copy(self) -> HandDistribution:
        return HandDistribution(self.hands.copy(), self.weights.copy(), self.mode, self.max_support)

    # -- transitions -------------------------------------------------------

    def shift(self, delta: np.ndarray) -> None:
        """Add a known vector; hypotheses that would go negative are impossible."""
        self.hands = self.hands + delta
        self._keep((self.hands >= 0).all(axis=1))

    def require(self, need: np.ndarray) -> None:
        """Evidence: the player held at least ``need``."""
        self._keep((self.hands >= need).all(axis=1))

    def require_exact(self, r: int, n: int) -> None:
        self._keep(self.hands[:, r] == n)

    def set_resource(self, r: int, n: int) -> None:
        self.hands[:, r] = n
        self._merge()

    def lose_random_card(self) -> np.ndarray:
        """An unseen card is taken. Returns P(card type) and updates self."""
        totals = self.hands.sum(axis=1, keepdims=True)
        frac = self.hands / np.maximum(totals, 1)
        w = self.weights / self.weights.sum()
        p_type = (w[:, None] * frac).sum(axis=0)
        new_h, new_w = [], []
        for r in range(NUM_RESOURCES):
            ok = self.hands[:, r] > 0
            if ok.any():
                new_h.append(self.hands[ok] - _EYE[r])
                new_w.append(self.weights[ok] * frac[ok, r])
        self.hands = np.concatenate(new_h)
        self.weights = np.concatenate(new_w)
        self._merge()
        return p_type / max(p_type.sum(), 1e-12)

    def gain_random_card(self, p_type: np.ndarray) -> None:
        """Receive one card whose type follows ``p_type``."""
        new_h, new_w = [], []
        for r in range(NUM_RESOURCES):
            if p_type[r] > 0:
                new_h.append(self.hands + _EYE[r])
                new_w.append(self.weights * p_type[r])
        self.hands = np.concatenate(new_h)
        self.weights = np.concatenate(new_w)
        self._merge()

    # -- queries -----------------------------------------------------------

    def probs(self) -> np.ndarray:
        return self.weights / self.weights.sum()

    def expected(self) -> np.ndarray:
        return (self.probs()[:, None] * self.hands).sum(axis=0)

    def best(self) -> np.ndarray:
        return self.hands[int(np.argmax(self.weights))].copy()

    def prob_at_least(self, need: np.ndarray) -> float:
        ok = (self.hands >= need).all(axis=1)
        return float(self.probs()[ok].sum())

    def contains(self, hand: np.ndarray) -> bool:
        return bool((self.hands == hand).all(axis=1).any())

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return self.hands[rng.choice(len(self.weights), p=self.probs())].copy()

    # -- internals ---------------------------------------------------------

    def _keep(self, ok: np.ndarray) -> None:
        if not ok.any():
            # Inconsistent evidence (should not happen with exact event streams):
            # fall back to the nearest non-negative hypothesis rather than crash.
            self.hands = np.maximum(self.hands[:1], 0)
            self.weights = np.ones(1)
            return
        self.hands = self.hands[ok]
        self.weights = self.weights[ok]
        self._normalize()

    def _merge(self) -> None:
        uniq, inv = np.unique(self.hands, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        if self.mode == "sum":
            w = np.bincount(inv, weights=self.weights, minlength=len(uniq))
        else:
            w = np.zeros(len(uniq))
            np.maximum.at(w, inv, self.weights)
        self.hands, self.weights = uniq, w
        if len(w) > self.max_support:
            top = np.argsort(w)[-self.max_support :]
            self.hands, self.weights = self.hands[top], self.weights[top]
        self._normalize()

    def _normalize(self) -> None:
        s = self.weights.sum() if self.mode == "sum" else self.weights.max()
        if s > 0:
            self.weights = self.weights / s


def _monopolized(d: HandDistribution, r: int, n: int) -> None:
    d.require_exact(r, n)
    d.set_resource(r, 0)


class BeliefTracker:
    """Per-seat tracker. Feed it ``event.view(seat)`` dicts in order."""

    def __init__(self, seat: int, max_support: int = 4096) -> None:
        self.seat = seat
        self.forward = {q: HandDistribution(max_support=max_support) for q in self.opponents}
        self.viterbi = {
            q: HandDistribution(mode="max", max_support=max_support) for q in self.opponents
        }
        self.my_hand = np.zeros(NUM_RESOURCES, dtype=np.int64)
        self.my_devs = np.zeros(NUM_DEV_TYPES, dtype=np.int64)  # unplayed (incl. new)
        self.dev_held = np.zeros(NUM_PLAYERS, dtype=np.int64)  # unplayed dev cards per seat
        self.dev_bought_this_turn = np.zeros(NUM_PLAYERS, dtype=np.int64)
        self.dev_played = np.zeros(NUM_DEV_TYPES, dtype=np.int64)  # publicly revealed
        self._offer = -1

    @property
    def opponents(self) -> list[int]:
        return [q for q in range(NUM_PLAYERS) if q != self.seat]

    # ------------------------------------------------------------- update

    def update(self, e: dict) -> None:
        kind, p = e["kind"], e["player"]
        match kind:
            case ev.SETUP_RESOURCES:
                self._shift(p, np.array(e["resources"]))
            case ev.PRODUCE:
                gains = np.array(e["gains"])
                for q in range(NUM_PLAYERS):
                    if gains[q].any():
                        self._shift(q, gains[q])
            case ev.DISCARD:
                self._shift(p, -_EYE[e["resource"]])
            case ev.STEAL:
                self._steal(p, e["victim"], e.get("resource"))
            case ev.BUILD_SETTLEMENT:
                if not e["free"]:
                    self._shift(p, -COST_SETTLEMENT)
            case ev.BUILD_ROAD:
                if not e["free"]:
                    self._shift(p, -COST_ROAD)
            case ev.BUILD_CITY:
                self._shift(p, -COST_CITY)
            case ev.BUY_DEV:
                self._shift(p, -COST_DEV)
                self.dev_held[p] += 1
                self.dev_bought_this_turn[p] += 1
                if p == self.seat:
                    self.my_devs[e["card"]] += 1
            case ev.PLAY_DEV:
                card = e["card"]
                self.dev_held[p] -= 1
                self.dev_played[card] += 1
                if p == self.seat:
                    self.my_devs[card] -= 1
            case ev.YEAR_OF_PLENTY:
                gain = sum(_EYE[r] for r in e["resources"])
                self._shift(p, gain)
            case ev.MONOPOLY:
                r = e["resource"]
                taken = {int(k): v for k, v in e["taken"].items()}
                for q in range(NUM_PLAYERS):
                    if q == p:
                        continue
                    if q == self.seat:
                        self.my_hand[r] = 0
                    else:
                        for d in (self.forward[q], self.viterbi[q]):
                            _monopolized(d, r, taken.get(q, 0))
                self._shift(p, sum(taken.values()) * _EYE[r])
            case ev.MARITIME_TRADE:
                self._shift(p, e["count"] * -_EYE[e["give"]] + _EYE[e["get"]])
            case ev.PROPOSE_TRADE:
                self._offer = e["offer"]
                self._each(p, lambda d: d.require(OFFER_GIVE[self._offer]))
            case ev.RESPOND_TRADE:
                if e["accept"]:
                    self._each(p, lambda d: d.require(OFFER_GET[self._offer]))
            case ev.TRADE:
                give, get = OFFER_GIVE[e["offer"]], OFFER_GET[e["offer"]]
                self._shift(p, get - give)
                self._shift(e["partner"], give - get)
            case ev.END_TURN:
                self.dev_bought_this_turn[p] = 0

    def _each(self, q: int, fn) -> None:
        if q == self.seat:
            return
        fn(self.forward[q])
        fn(self.viterbi[q])

    def _shift(self, q: int, delta: np.ndarray) -> None:
        if q == self.seat:
            self.my_hand = self.my_hand + delta
        else:
            self._each(q, lambda d: d.shift(delta))

    def _steal(self, thief: int, victim: int, resource: int | None) -> None:
        if resource is not None:
            self._shift(victim, -_EYE[resource])
            self._shift(thief, _EYE[resource])
            return
        # Neither party is us: the card type is hidden.
        p_fwd = self.forward[victim].lose_random_card()
        p_vit = self.viterbi[victim].lose_random_card()
        self.forward[thief].gain_random_card(p_fwd)
        self.viterbi[thief].gain_random_card(p_vit)

    # ------------------------------------------------------------- queries

    def hand_marginals(self, q: int) -> np.ndarray:
        """Expected count of each resource in seat ``q``'s hand."""
        return self.my_hand.astype(float) if q == self.seat else self.forward[q].expected()

    def map_hand(self, q: int) -> np.ndarray:
        """Most probable current hand (posterior marginal mode)."""
        return self.my_hand.copy() if q == self.seat else self.forward[q].best()

    def viterbi_hand(self, q: int) -> np.ndarray:
        """Hand at the end of the most likely sequence of hidden steals."""
        return self.my_hand.copy() if q == self.seat else self.viterbi[q].best()

    def prob_can_afford(self, q: int, cost: np.ndarray) -> float:
        if q == self.seat:
            return float((self.my_hand >= cost).all())
        return self.forward[q].prob_at_least(cost)

    def uncertainty(self, q: int) -> int:
        """Number of hand hypotheses still alive for ``q`` (1 = known exactly)."""
        return 1 if q == self.seat else len(self.forward[q].weights)

    def unseen_dev_pool(self) -> np.ndarray:
        """Dev cards whose location is unknown to us (in the deck or opponents' hands)."""
        return DEV_DECK_COUNTS - self.dev_played - self.my_devs

    def dev_type_probs(self) -> np.ndarray:
        """P(type) of any single unseen dev card (exchangeable assumption)."""
        pool = self.unseen_dev_pool().astype(float)
        return pool / pool.sum() if pool.sum() > 0 else pool

    def expected_hidden_vp(self, q: int) -> float:
        if q == self.seat:
            return float(self.my_devs[DevCard.VICTORY_POINT])
        return float(self.dev_held[q] * self.dev_type_probs()[DevCard.VICTORY_POINT])

    def expected_hidden_vps(self) -> list[float]:
        """``expected_hidden_vp`` for every seat, computing the dev-type odds once."""
        p_vp = self.dev_type_probs()[DevCard.VICTORY_POINT]
        return [
            float(self.my_devs[DevCard.VICTORY_POINT])
            if q == self.seat
            else float(self.dev_held[q] * p_vp)
            for q in range(NUM_PLAYERS)
        ]

    # ------------------------------------------------------------- determinization

    def determinize(
        self, redacted: GameState, rng: np.random.Generator, tries: int = 200
    ) -> GameState:
        """Fill the hidden parts of ``redacted`` with a sample consistent with beliefs.

        Opponent hands are sampled from the forward filter, subject to the exact
        per-resource totals implied by the bank; dev cards are dealt from the
        unseen pool.
        """
        s = redacted.clone()
        opp = self.opponents
        remaining = BANK_PER_RESOURCE - s.bank - s.hands[self.seat]
        hands = None
        for _ in range(tries):
            sample = [self.forward[q].sample(rng) for q in opp[:-1]]
            last = remaining - sum(sample)
            if (last >= 0).all():
                hands = [*sample, last]
                break
        if hands is None:  # fall back to MAP for all but one opponent
            first = [self.forward[q].best() for q in opp[:-1]]
            last = np.maximum(remaining - sum(first), 0)
            hands = [*first, last]
        for q, h in zip(opp, hands, strict=True):
            s.hands[q] = h

        pool = [int(c) for c in range(NUM_DEV_TYPES) for _ in range(int(self.unseen_dev_pool()[c]))]
        rng.shuffle(pool)
        for q in opp:
            n_new = int(self.dev_bought_this_turn[q]) if q == s.current else 0
            for i in range(int(self.dev_held[q])):
                card = pool.pop() if pool else int(DevCard.KNIGHT)
                if i < n_new:
                    s.dev_new[q, card] += 1
                else:
                    s.dev_hand[q, card] += 1
        s.dev_deck = pool  # whatever is left over is the (reshuffled) deck
        return s

    # convenience for tests / features
    def summary(self) -> dict:
        return {
            q: {
                "expected": self.hand_marginals(q).round(2).tolist(),
                "map": self.map_hand(q).tolist(),
                "viterbi": self.viterbi_hand(q).tolist(),
                "hypotheses": self.uncertainty(q),
                "dev_cards": int(self.dev_held[q]),
            }
            for q in range(NUM_PLAYERS)
        }

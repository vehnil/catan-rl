import numpy as np

from catan.engine import ActionType, Phase, RulesConfig, encode
from catan.engine.actions import OFFER_GET, OFFER_GIVE, TRADE_OFFERS, block
from catan.engine.rules import legal_mask
from tests.conftest import give, main_phase_game


def offer_index(give_vec, get_vec) -> int:
    for i, (g, r) in enumerate(TRADE_OFFERS):
        if g.tolist() == give_vec and r.tolist() == get_vec:
            return i
    raise KeyError


WOOD_FOR_ORE = offer_index([1, 0, 0, 0, 0], [0, 0, 0, 0, 1])


def test_offer_enumeration_is_disjoint_and_bounded():
    assert ((OFFER_GIVE > 0) & (OFFER_GET > 0)).sum() == 0
    assert set(OFFER_GIVE.sum(1)) == {1, 2} and set(OFFER_GET.sum(1)) == {1, 2}
    assert len({(tuple(g), tuple(r)) for g, r in TRADE_OFFERS}) == len(TRADE_OFFERS)


def test_offer_requires_proposer_holding_cards():
    g = main_phase_game()
    assert not legal_mask(g.state)[block(ActionType.PROPOSE_TRADE)].any()
    give(g.state, 0, [1, 0, 0, 0, 0])
    offers = np.flatnonzero(legal_mask(g.state)[block(ActionType.PROPOSE_TRADE)])
    assert all(OFFER_GIVE[i].tolist() == [1, 0, 0, 0, 0] for i in offers)


def test_full_protocol_accept_and_confirm():
    g = main_phase_game()
    s = g.state
    give(s, 0, [1, 0, 0, 0, 0])
    give(s, 1, [0, 0, 0, 0, 1])
    give(s, 3, [0, 0, 0, 0, 1])
    # P2 can't afford -> auto-rejected and skipped.
    g.step(encode(ActionType.PROPOSE_TRADE, WOOD_FOR_ORE))
    assert s.phase == Phase.TRADE_RESPOND and s.actor == 1
    g.step(encode(ActionType.ACCEPT_TRADE))
    assert s.actor == 3
    g.step(encode(ActionType.ACCEPT_TRADE))
    assert s.phase == Phase.TRADE_CONFIRM and s.actor == 0
    m = legal_mask(s)
    assert m[encode(ActionType.CONFIRM_TRADE, 0)] and m[encode(ActionType.CONFIRM_TRADE, 2)]
    assert not m[encode(ActionType.CONFIRM_TRADE, 1)]
    g.step(encode(ActionType.CONFIRM_TRADE, 2))  # trade with P3
    assert s.hands[0].tolist() == [0, 0, 0, 0, 1]
    assert s.hands[3].tolist() == [1, 0, 0, 0, 0]
    assert s.hands[1].tolist() == [0, 0, 0, 0, 1]
    assert s.phase == Phase.MAIN and s.trade is None


def test_rejected_offer_masked_for_rest_of_turn():
    g = main_phase_game()
    s = g.state
    give(s, 0, [1, 0, 0, 0, 0])
    give(s, 1, [0, 0, 0, 0, 1])
    g.step(encode(ActionType.PROPOSE_TRADE, WOOD_FOR_ORE))
    g.step(encode(ActionType.REJECT_TRADE))
    assert s.phase == Phase.MAIN
    assert not legal_mask(s)[encode(ActionType.PROPOSE_TRADE, WOOD_FOR_ORE)]
    g.step(encode(ActionType.END_TURN))
    assert WOOD_FOR_ORE not in s.rejected_offers


def test_offer_nobody_can_afford_resolves_immediately():
    g = main_phase_game()
    give(g.state, 0, [1, 0, 0, 0, 0])
    g.step(encode(ActionType.PROPOSE_TRADE, WOOD_FOR_ORE))
    assert g.state.phase == Phase.MAIN
    assert WOOD_FOR_ORE in g.state.rejected_offers


def test_proposal_cap_per_turn():
    g = main_phase_game(config=RulesConfig(max_trade_proposals_per_turn=2))
    s = g.state
    give(s, 0, [3, 0, 0, 0, 0])
    give(s, 1, [0, 0, 1, 1, 1])
    for get in ([0, 0, 0, 0, 1], [0, 0, 0, 1, 0]):
        g.step(encode(ActionType.PROPOSE_TRADE, offer_index([1, 0, 0, 0, 0], get)))
        g.step(encode(ActionType.ACCEPT_TRADE))
        g.step(encode(ActionType.CANCEL_TRADE))
    assert not legal_mask(s)[block(ActionType.PROPOSE_TRADE)].any()

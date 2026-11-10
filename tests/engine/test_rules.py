import numpy as np
import pytest

from catan.engine import ActionType, Game, IllegalActionError, Phase, RulesConfig, encode
from catan.engine.actions import MARITIME_PAIRS, block
from catan.engine.board import TOPOLOGY
from catan.engine.constants import DESERT, DevCard, Resource
from catan.engine.rules import _produce, legal_mask
from catan.engine.scoring import update_longest_road
from catan.engine.trading import maritime_rates
from tests.conftest import edge_between, give, main_phase_game, path_edges, place, road

W, B, S, H, O = range(5)


def legal(g: Game, t: ActionType) -> np.ndarray:
    return np.flatnonzero(legal_mask(g.state)[block(t)])


class FixedRng:
    """Stands in for np.random.Generator to force dice rolls."""

    def __init__(self, dice: tuple[int, int]):
        self.dice = dice
        self._real = np.random.default_rng(0)

    def integers(self, *_a, **_k):
        return np.array(self.dice)

    def choice(self, *a, **k):
        return self._real.choice(*a, **k)


# ---------------------------------------------------------------- setup


def test_setup_snake_order_and_second_round_resources():
    g = Game(seed=0)
    order = []
    while g.state.is_setup:
        order.append(g.state.current)
        v = int(legal(g, ActionType.BUILD_SETTLEMENT)[0])
        g.step(encode(ActionType.BUILD_SETTLEMENT, v))
        if len(order) > 4:
            hexes = [h for h in TOPOLOGY.vertex_hexes[v] if h >= 0]
            expected = sum(1 for h in hexes if g.state.board.terrain[h] != DESERT)
            assert g.state.hands[g.state.current].sum() == expected
        else:
            assert g.state.hands[g.state.current].sum() == 0
        e = int(legal(g, ActionType.BUILD_ROAD)[0])
        assert v in TOPOLOGY.edge_vertices[e]
        g.step(encode(ActionType.BUILD_ROAD, e))
    assert order == [0, 1, 2, 3, 3, 2, 1, 0]
    assert g.state.phase == Phase.ROLL and g.state.current == 0
    assert (g.state.settlements_left == 3).all() and (g.state.roads_left == 13).all()


def test_distance_rule():
    g = Game(seed=0)
    g.step(encode(ActionType.BUILD_SETTLEMENT, 20))
    g.step(int(np.flatnonzero(legal_mask(g.state))[0]))
    allowed = legal(g, ActionType.BUILD_SETTLEMENT)
    assert 20 not in allowed
    for nb in TOPOLOGY.vertex_vertices[20]:
        assert nb not in allowed


def test_illegal_action_rejected():
    g = Game(seed=0)
    with pytest.raises(IllegalActionError):
        g.step(encode(ActionType.ROLL))


# ---------------------------------------------------------------- building


def test_settlement_requires_connected_road():
    g = main_phase_game()
    s = g.state
    place(s, 0, 20)
    give(s, 0, [1, 1, 1, 1, 0])
    assert len(legal(g, ActionType.BUILD_SETTLEMENT)) == 0
    verts, edges = path_edges(20, 2)
    for e in edges:
        road(s, 0, e)
    assert legal(g, ActionType.BUILD_SETTLEMENT).tolist() == [verts[2]]
    g.step(encode(ActionType.BUILD_SETTLEMENT, verts[2]))
    assert s.hands[0].sum() == 0 and s.bank.sum() == 95
    assert s.public_vp(0) == 2


def test_road_cannot_pass_through_opponent_settlement():
    g = main_phase_game()
    s = g.state
    verts, edges = path_edges(20, 1)
    place(s, 0, 20)
    road(s, 0, edges[0])
    place(s, 1, verts[1])  # opponent sits at the end of our road
    give(s, 0, [1, 1, 0, 0, 0])
    allowed = set(legal(g, ActionType.BUILD_ROAD).tolist())
    beyond = {int(e) for e in TOPOLOGY.vertex_edges[verts[1]] if e < 72} - {edges[0]}
    assert not allowed & beyond
    near = {int(e) for e in TOPOLOGY.vertex_edges[20] if e < 72} - {edges[0]}
    assert near <= allowed


def test_city_upgrade():
    g = main_phase_game()
    s = g.state
    place(s, 0, 20)
    give(s, 0, [0, 0, 0, 2, 3])
    assert legal(g, ActionType.BUILD_CITY).tolist() == [20]
    g.step(encode(ActionType.BUILD_CITY, 20))
    assert s.vertex_level[20] == 2 and s.public_vp(0) == 2
    assert s.settlements_left[0] == 5 and s.cities_left[0] == 3


# ---------------------------------------------------------------- production & robber


def _single_hex(g: Game, number: int = 12) -> int:
    hexes = g.state.board.hexes_by_number[number]
    assert len(hexes) == 1
    return hexes[0]


def test_production_settlement_and_city():
    g = main_phase_game()
    s = g.state
    h = _single_hex(g)
    r = s.board.terrain[h]
    v1, v2 = int(TOPOLOGY.hex_vertices[h][0]), int(TOPOLOGY.hex_vertices[h][3])
    place(s, 0, v1)
    place(s, 1, v2, level=2)
    _produce(s, 12, [])
    assert s.hands[0, r] == 1 and s.hands[1, r] == 2
    s.robber = h
    _produce(s, 12, [])
    assert s.hands[0, r] == 1 and s.hands[1, r] == 2


def test_bank_shortage_multiple_recipients_get_nothing():
    g = main_phase_game()
    s = g.state
    h = _single_hex(g)
    r = s.board.terrain[h]
    hv = TOPOLOGY.hex_vertices[h]
    place(s, 0, int(hv[0]), level=2)
    place(s, 1, int(hv[3]), level=2)
    give(s, 2, [16 if i == r else 0 for i in range(5)])  # 3 left in bank, 4 owed
    _produce(s, 12, [])
    assert s.hands[0, r] == 0 and s.hands[1, r] == 0


def test_bank_shortage_single_recipient_gets_remainder():
    g = main_phase_game()
    s = g.state
    h = _single_hex(g)
    r = s.board.terrain[h]
    place(s, 0, int(TOPOLOGY.hex_vertices[h][0]), level=2)
    give(s, 2, [18 if i == r else 0 for i in range(5)])  # 1 left, 2 owed
    _produce(s, 12, [])
    assert s.hands[0, r] == 1 and s.bank[r] == 0


def test_seven_triggers_discard_then_robber():
    g = main_phase_game()
    s = g.state
    s.phase = Phase.ROLL
    give(s, 1, [3, 3, 3, 0, 0])  # 9 cards -> discard 4
    give(s, 2, [2, 2, 2, 1, 0])  # 7 cards -> safe
    g.rng = FixedRng((3, 4))
    g.step(encode(ActionType.ROLL))
    assert s.phase == Phase.DISCARD and s.actor == 1
    assert s.discard_pending.tolist() == [0, 4, 0, 0]
    for _ in range(4):
        g.step(encode(ActionType.DISCARD, W) if s.hands[1, W] else encode(ActionType.DISCARD, B))
    assert s.phase == Phase.MOVE_ROBBER and s.actor == 0
    assert s.hands[1].sum() == 5
    assert s.robber not in legal(g, ActionType.MOVE_ROBBER)


def test_robber_steal_single_victim_auto():
    g = main_phase_game()
    s = g.state
    h = 0 if s.robber != 0 else 1
    place(s, 1, int(TOPOLOGY.hex_vertices[h][0]))
    give(s, 1, [0, 0, 2, 0, 0])
    s.phase = Phase.MOVE_ROBBER
    s.return_phase = Phase.MAIN
    events = g.step(encode(ActionType.MOVE_ROBBER, h))
    assert s.phase == Phase.MAIN
    assert s.hands[0, S] == 1 and s.hands[1, S] == 1
    steal = [e for e in events if e.kind == "steal"][0]
    assert "resource" not in steal.view(2)
    assert steal.view(0)["resource"] == S and steal.view(1)["resource"] == S


def test_robber_multiple_victims_requires_choice():
    g = main_phase_game()
    s = g.state
    h = 0 if s.robber != 0 else 1
    hv = TOPOLOGY.hex_vertices[h]
    place(s, 1, int(hv[0]))
    place(s, 2, int(hv[3]))
    give(s, 1, [1, 0, 0, 0, 0])
    give(s, 2, [0, 1, 0, 0, 0])
    s.phase = Phase.MOVE_ROBBER
    g.step(encode(ActionType.MOVE_ROBBER, h))
    assert s.phase == Phase.STEAL
    assert legal(g, ActionType.STEAL).tolist() == [0, 1]  # offsets +1 (P1), +2 (P2)
    g.step(encode(ActionType.STEAL, 1))
    assert s.hands[0, B] == 1 and s.hands[2].sum() == 0


# ---------------------------------------------------------------- dev cards


def test_dev_card_not_playable_on_turn_bought():
    g = main_phase_game()
    s = g.state
    s.dev_deck.append(int(DevCard.KNIGHT))
    give(s, 0, [0, 0, 1, 1, 1])
    g.step(encode(ActionType.BUY_DEV))
    assert s.dev_new[0, DevCard.KNIGHT] == 1
    assert not legal_mask(s)[encode(ActionType.PLAY_KNIGHT)]
    g.step(encode(ActionType.END_TURN))
    assert s.dev_hand[0, DevCard.KNIGHT] == 1


def test_one_dev_card_per_turn_and_knight_before_roll():
    g = main_phase_game()
    s = g.state
    s.dev_hand[0, DevCard.KNIGHT] = 2
    s.phase = Phase.ROLL
    assert legal_mask(s)[encode(ActionType.PLAY_KNIGHT)]
    g.step(encode(ActionType.PLAY_KNIGHT))
    assert s.phase == Phase.MOVE_ROBBER and s.return_phase == Phase.ROLL
    h = int(legal(g, ActionType.MOVE_ROBBER)[0])
    g.step(encode(ActionType.MOVE_ROBBER, h))
    assert s.phase == Phase.ROLL
    assert not legal_mask(s)[encode(ActionType.PLAY_KNIGHT)]


def test_largest_army():
    g = main_phase_game()
    s = g.state
    s.knights[1] = 3
    s.largest_army_holder = 1
    s.knights[0] = 3
    s.dev_hand[0, DevCard.KNIGHT] = 1
    g.step(encode(ActionType.PLAY_KNIGHT))
    assert s.largest_army_holder == 0 and s.public_vp(0) == 2


def test_monopoly_and_year_of_plenty():
    g = main_phase_game()
    s = g.state
    give(s, 1, [0, 0, 0, 3, 0])
    give(s, 2, [0, 0, 0, 2, 1])
    s.dev_hand[0, DevCard.MONOPOLY] = 1
    s.dev_hand[0, DevCard.YEAR_OF_PLENTY] = 1
    g.step(encode(ActionType.PLAY_MONOPOLY, H))
    assert s.hands[0, H] == 5 and s.hands[1:, H].sum() == 0
    assert not legal_mask(s)[block(ActionType.PLAY_YEAR_OF_PLENTY)].any()  # one per turn
    s.dev_played_this_turn = False
    g.step(encode(ActionType.PLAY_YEAR_OF_PLENTY, 0))  # (wood, wood)
    assert s.hands[0, W] == 2


def test_road_building_places_two_free_roads():
    g = main_phase_game()
    s = g.state
    place(s, 0, 20)
    s.dev_hand[0, DevCard.ROAD_BUILDING] = 1
    g.step(encode(ActionType.PLAY_ROAD_BUILDING))
    assert s.phase == Phase.ROAD_BUILDING
    for _ in range(2):
        g.step(encode(ActionType.BUILD_ROAD, int(legal(g, ActionType.BUILD_ROAD)[0])))
    assert s.phase == Phase.MAIN and s.roads_left[0] == 13 and s.bank.sum() == 95


# ---------------------------------------------------------------- scoring


def test_longest_road_awarded_and_broken():
    g = main_phase_game()
    s = g.state
    verts, edges = path_edges(0, 6)
    place(s, 0, verts[0])
    for e in edges[:4]:
        road(s, 0, e)
    give(s, 0, [1, 1, 0, 0, 0])
    g.step(encode(ActionType.BUILD_ROAD, edges[4]))
    assert s.road_length[0] == 5 and s.longest_road_holder == 0
    # Opponent settles in the middle of the road: 5 -> max(2, 3) and title lost.
    place(s, 1, verts[2])
    update_longest_road(s)
    assert s.road_length[0] == 3
    assert s.longest_road_holder == -1


def test_longest_road_holder_keeps_title_on_tie():
    g = main_phase_game()
    s = g.state
    _, e0 = path_edges(0, 5)
    for e in e0:
        road(s, 0, e)
    update_longest_road(s)
    assert s.longest_road_holder == 0
    _, e1 = path_edges(53, 5)
    for e in e1:
        road(s, 1, e)
    update_longest_road(s)
    assert s.road_length[1] == 5 and s.longest_road_holder == 0


def test_maritime_rates_and_trade():
    g = main_phase_game()
    s = g.state
    assert maritime_rates(s, 0).tolist() == [4] * 5
    e = int(s.board.port_edges[list(s.board.port_types).index(int(Resource.ORE))])
    place(s, 0, int(TOPOLOGY.edge_vertices[e][0]))
    rates = maritime_rates(s, 0)
    assert rates[O] == 2 and rates[W] == 4
    give(s, 0, [0, 0, 0, 0, 2])
    allowed = [MARITIME_PAIRS[i] for i in legal(g, ActionType.MARITIME_TRADE)]
    assert set(allowed) == {(O, r) for r in range(4)}
    g.step(encode(ActionType.MARITIME_TRADE, MARITIME_PAIRS.index((O, W))))
    assert s.hands[0].tolist() == [1, 0, 0, 0, 0]


def test_win_on_own_turn():
    g = main_phase_game()
    s = g.state
    for v in [0, 5, 10, 22]:
        place(s, 0, v, level=2)
    s.dev_hand[0, DevCard.VICTORY_POINT] = 1
    give(s, 0, [0, 0, 1, 1, 1])
    s.dev_deck.append(int(DevCard.VICTORY_POINT))
    g.step(encode(ActionType.BUY_DEV))
    assert s.done and s.winner == 0 and not s.truncated


# ---------------------------------------------------------------- limits


def test_turn_cap_truncates():
    g = main_phase_game(config=RulesConfig(max_turns_per_game=3))
    s = g.state
    for _ in range(3):
        s.phase = Phase.MAIN
        g.step(encode(ActionType.END_TURN))
    assert s.done and s.truncated and s.winner == -1


def test_action_cap_forces_end_turn():
    g = main_phase_game(config=RulesConfig(max_actions_per_turn=2))
    s = g.state
    give(s, 0, [8, 0, 0, 0, 0])
    i = MARITIME_PAIRS.index((W, B))
    g.step(encode(ActionType.MARITIME_TRADE, i))
    g.step(encode(ActionType.MARITIME_TRADE, i))
    assert np.flatnonzero(legal_mask(s)).tolist() == [encode(ActionType.END_TURN)]


def test_edge_between_helper():
    a, b = (int(x) for x in TOPOLOGY.edge_vertices[7])
    assert edge_between(a, b) == 7

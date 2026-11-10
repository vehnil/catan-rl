import numpy as np

from catan.engine.board import NUM_EDGES, NUM_VERTICES, TOPOLOGY, Board, _adjacent_red
from catan.engine.constants import DESERT, GENERIC_PORT, NUMBER_TOKENS, TERRAIN_COUNTS


def test_topology_counts():
    assert TOPOLOGY.hex_vertices.shape == (19, 6)
    assert TOPOLOGY.edge_vertices.shape == (NUM_EDGES, 2)
    degrees = (TOPOLOGY.vertex_vertices < NUM_VERTICES).sum(axis=1)
    assert sorted(np.bincount(degrees).tolist()) == [0, 0, 18, 36]
    hexes_per_vertex = (TOPOLOGY.vertex_hexes >= 0).sum(axis=1)
    assert np.bincount(hexes_per_vertex).tolist() == [0, 18, 12, 24]


def test_edges_connect_adjacent_vertices():
    for a, b in TOPOLOGY.edge_vertices:
        assert b in TOPOLOGY.vertex_vertices[a]
        assert a in TOPOLOGY.vertex_vertices[b]


def test_hex_neighbors_symmetric():
    for h, nbs in enumerate(TOPOLOGY.hex_neighbors):
        for nb in nbs:
            if nb >= 0:
                assert h in TOPOLOGY.hex_neighbors[nb]
                # Neighbouring hexes share exactly one edge (two vertices).
                assert len(set(TOPOLOGY.hex_vertices[h]) & set(TOPOLOGY.hex_vertices[nb])) == 2


def _check_layout(b: Board):
    for t, c in TERRAIN_COUNTS.items():
        assert (b.terrain == int(t)).sum() == c
    assert sorted(b.numbers[b.terrain != DESERT].tolist()) == sorted(NUMBER_TOKENS)
    assert b.numbers[b.desert] == 0
    assert len(set(b.port_edges.tolist())) == 9
    assert set(b.port_edges.tolist()) <= set(TOPOLOGY.coastal_edges)
    assert (b.port_types == GENERIC_PORT).sum() == 4
    assert sorted(b.port_types[b.port_types != GENERIC_PORT].tolist()) == [0, 1, 2, 3, 4]
    # Ports never share a vertex.
    assert (b.vertex_port >= 0).sum() == 18


def test_standard_layout_valid():
    _check_layout(Board.standard())


def test_random_layouts_valid():
    rng = np.random.default_rng(0)
    for _ in range(20):
        b = Board.random(rng, avoid_adjacent_red=True)
        _check_layout(b)
        assert not _adjacent_red(b.numbers)

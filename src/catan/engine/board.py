"""Board topology (fixed for every game) and layouts (terrain, numbers, ports).

Hexes use pointy-top axial coordinates (q, r) within radius 2. Vertices and
edges are derived geometrically and given stable integer ids, sorted top to
bottom then left to right.

All per-vertex / per-edge adjacency tables are padded numpy arrays. Padding
points at a sentinel index (``NUM_VERTICES`` / ``NUM_EDGES``) so that state
arrays sized ``N + 1`` can be fancy-indexed without masking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from catan.engine.constants import (
    DESERT,
    GENERIC_PORT,
    NUMBER_TOKENS,
    PORT_TYPES,
    TERRAIN_COUNTS,
    Resource,
)

NUM_HEXES = 19
NUM_VERTICES = 54
NUM_EDGES = 72
NUM_PORTS = 9


def _hex_coords() -> list[tuple[int, int]]:
    coords = [
        (q, r) for r in range(-2, 3) for q in range(-2, 3) if max(abs(q), abs(r), abs(q + r)) <= 2
    ]
    return coords


def _corner_keys(q: int, r: int) -> list[tuple[int, int]]:
    """Integer keys for the 6 corners of a hex, clockwise from the top.

    x is measured in units of sqrt(3)/2, y in units of 1/2, which makes every
    corner land on an exact integer lattice point.
    """
    cx, cy = math.sqrt(3) * (q + r / 2), 1.5 * r
    keys = []
    for i in range(6):
        ang = math.radians(-90 + 60 * i)
        x, y = cx + math.cos(ang), cy + math.sin(ang)
        keys.append((round(x / (math.sqrt(3) / 2)), round(y * 2)))
    return keys


def _pad(rows: list[list[int]], width: int, fill: int) -> np.ndarray:
    out = np.full((len(rows), width), fill, dtype=np.int64)
    for i, row in enumerate(rows):
        out[i, : len(row)] = row
    return out


@dataclass(frozen=True)
class Topology:
    hex_coords: list[tuple[int, int]]
    vertex_pos: list[tuple[int, int]]  # integer lattice keys, useful for rendering
    hex_vertices: np.ndarray  # (19, 6)
    hex_neighbors: np.ndarray  # (19, 6) padded with -1
    edge_vertices: np.ndarray  # (72, 2)
    vertex_vertices: np.ndarray  # (54, 3) padded with NUM_VERTICES
    vertex_edges: np.ndarray  # (54, 3) padded with NUM_EDGES
    vertex_hexes: np.ndarray  # (54, 3) padded with -1
    coastal_edges: list[int]  # ordered clockwise around the island

    @staticmethod
    def build() -> Topology:
        coords = _hex_coords()
        corner_lists = [_corner_keys(q, r) for q, r in coords]
        all_keys = sorted({k for ks in corner_lists for k in ks}, key=lambda k: (k[1], k[0]))
        vid = {k: i for i, k in enumerate(all_keys)}
        hex_vertices = np.array([[vid[k] for k in ks] for ks in corner_lists], dtype=np.int64)

        edge_set: set[tuple[int, int]] = set()
        edge_hex_count: dict[tuple[int, int], int] = {}
        for hv in hex_vertices:
            for i in range(6):
                a, b = int(hv[i]), int(hv[(i + 1) % 6])
                e = (min(a, b), max(a, b))
                edge_set.add(e)
                edge_hex_count[e] = edge_hex_count.get(e, 0) + 1
        edges = sorted(edge_set)
        eid = {e: i for i, e in enumerate(edges)}
        edge_vertices = np.array(edges, dtype=np.int64)

        nv = len(all_keys)
        vv: list[list[int]] = [[] for _ in range(nv)]
        ve: list[list[int]] = [[] for _ in range(nv)]
        vh: list[list[int]] = [[] for _ in range(nv)]
        for (a, b), i in eid.items():
            vv[a].append(b)
            vv[b].append(a)
            ve[a].append(i)
            ve[b].append(i)
        for h, hv in enumerate(hex_vertices):
            for v in hv:
                vh[int(v)].append(h)

        coord_idx = {c: i for i, c in enumerate(coords)}
        dirs = [(1, -1), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1)]
        hex_nb = [
            [coord_idx[(q + dq, r + dr)] for dq, dr in dirs if (q + dq, r + dr) in coord_idx]
            for q, r in coords
        ]

        def edge_angle(e: tuple[int, int]) -> float:
            (x1, y1), (x2, y2) = all_keys[e[0]], all_keys[e[1]]
            mx, my = (x1 + x2) * math.sqrt(3) / 4, (y1 + y2) / 4
            return math.atan2(my, mx)

        coastal = sorted((e for e, c in edge_hex_count.items() if c == 1), key=edge_angle)

        return Topology(
            hex_coords=coords,
            vertex_pos=all_keys,
            hex_vertices=hex_vertices,
            hex_neighbors=_pad(hex_nb, 6, -1),
            edge_vertices=edge_vertices,
            vertex_vertices=_pad(vv, 3, nv),
            vertex_edges=_pad(ve, 3, len(edges)),
            vertex_hexes=_pad(vh, 3, -1),
            coastal_edges=[eid[e] for e in coastal],
        )


TOPOLOGY = Topology.build()
assert len(TOPOLOGY.hex_coords) == NUM_HEXES
assert len(TOPOLOGY.vertex_pos) == NUM_VERTICES
assert len(TOPOLOGY.edge_vertices) == NUM_EDGES
assert len(TOPOLOGY.coastal_edges) == 30

# Indices into the 30 coastal edges where the 9 harbors sit (gaps 3/4/3 repeating).
PORT_SLOTS = [0, 3, 7, 10, 13, 17, 20, 23, 27]

W, B, S, H, O, D = (
    Resource.WOOD,
    Resource.BRICK,
    Resource.SHEEP,
    Resource.WHEAT,
    Resource.ORE,
    DESERT,
)
# Fixed reference layout (row-major, rows of 3/4/5/4/3 hexes).
STANDARD_TERRAIN = [O, S, W, H, B, S, B, H, W, D, W, O, W, O, H, S, B, H, S]
STANDARD_NUMBERS = [10, 2, 9, 12, 6, 4, 10, 9, 11, 0, 3, 8, 8, 3, 4, 5, 5, 6, 11]
STANDARD_PORTS = [GENERIC_PORT, H, O, GENERIC_PORT, S, GENERIC_PORT, GENERIC_PORT, B, W]


@dataclass(frozen=True)
class Board:
    """A concrete layout. Immutable and shared between cloned states."""

    terrain: np.ndarray  # (19,) resource id or DESERT
    numbers: np.ndarray  # (19,) dice number, 0 for desert
    port_edges: np.ndarray  # (9,) edge ids
    port_types: np.ndarray  # (9,) resource id or GENERIC_PORT
    vertex_port: np.ndarray = field(init=False)  # (55,) port type at vertex or -1
    hexes_by_number: dict[int, list[int]] = field(init=False)
    topo: Topology = field(default=TOPOLOGY, repr=False)

    def __post_init__(self) -> None:
        vp = np.full(NUM_VERTICES + 1, -1, dtype=np.int64)
        for e, t in zip(self.port_edges, self.port_types, strict=True):
            for v in self.topo.edge_vertices[e]:
                vp[v] = t
        object.__setattr__(self, "vertex_port", vp)
        by_num: dict[int, list[int]] = {n: [] for n in range(2, 13)}
        for h, n in enumerate(self.numbers):
            if n:
                by_num[int(n)].append(h)
        object.__setattr__(self, "hexes_by_number", by_num)

    @property
    def desert(self) -> int:
        return int(np.flatnonzero(self.terrain == DESERT)[0])

    @staticmethod
    def standard() -> Board:
        coastal = TOPOLOGY.coastal_edges
        return Board(
            terrain=np.array([int(t) for t in STANDARD_TERRAIN], dtype=np.int64),
            numbers=np.array(STANDARD_NUMBERS, dtype=np.int64),
            port_edges=np.array([coastal[i] for i in PORT_SLOTS], dtype=np.int64),
            port_types=np.array([int(p) for p in STANDARD_PORTS], dtype=np.int64),
        )

    @staticmethod
    def random(rng: np.random.Generator, avoid_adjacent_red: bool = True) -> Board:
        terrain = np.array(
            [int(t) for t, c in TERRAIN_COUNTS.items() for _ in range(c)], dtype=np.int64
        )
        for _ in range(1000):
            rng.shuffle(terrain)
            tokens = np.array(NUMBER_TOKENS, dtype=np.int64)
            rng.shuffle(tokens)
            numbers = np.zeros(NUM_HEXES, dtype=np.int64)
            numbers[terrain != DESERT] = tokens
            if not avoid_adjacent_red or not _adjacent_red(numbers):
                break
        ports = np.array(PORT_TYPES, dtype=np.int64)
        rng.shuffle(ports)
        coastal = TOPOLOGY.coastal_edges
        return Board(
            terrain=terrain,
            numbers=numbers,
            port_edges=np.array([coastal[i] for i in PORT_SLOTS], dtype=np.int64),
            port_types=ports,
        )


def _adjacent_red(numbers: np.ndarray) -> bool:
    red = (numbers == 6) | (numbers == 8)
    for h in np.flatnonzero(red):
        for nb in TOPOLOGY.hex_neighbors[h]:
            if nb >= 0 and red[nb]:
                return True
    return False

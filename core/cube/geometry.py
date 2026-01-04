"""
KONOMI Core - Cube Geometry
Cube topology and vertex definitions
"""

from typing import List, Dict
from collections import defaultdict

# 8 vertex positions: North/South, East/West, Up/Down
VERTICES: List[str] = [
    'NEU', 'NED', 'NWU', 'NWD',
    'SEU', 'SED', 'SWU', 'SWD'
]


def are_adjacent(v1: str, v2: str) -> bool:
    """
    Check if two vertices are adjacent (share a cube edge).
    Adjacent vertices differ in exactly 1 axis.
    """
    diffs = sum(c1 != c2 for c1, c2 in zip(v1, v2))
    return diffs == 1


def build_cube_edges() -> Dict[str, List[str]]:
    """
    Build standard cube edge connections.

    Connects:
    1. Vertices that differ by exactly 1 axis (cube edges)
    2. All vertices to central node (hub-spoke)
    """
    edges: Dict[str, List[str]] = defaultdict(list)

    # Connect vertices that differ by exactly 1 axis
    for v1 in VERTICES:
        for v2 in VERTICES:
            if v1 != v2 and are_adjacent(v1, v2):
                if v2 not in edges[v1]:
                    edges[v1].append(v2)

    # Connect all vertices to central node (bidirectional)
    for v in VERTICES:
        edges[v].append('central')
        edges['central'].append(v)

    return edges


def get_valid_nodes() -> List[str]:
    """Get list of all valid node names."""
    return VERTICES + ['central']


def visualize_cube() -> str:
    """
    Create ASCII visualization of the cube graph.

    Returns:
        ASCII art string showing cube structure
    """
    lines = [
        "Cube Graph Topology:",
        "",
        "        NWU -------- NEU",
        "       /|           /|",
        "      / |          / |",
        "    SWU -------- SEU |",
        "     |  |    C   |  |",
        "     | NWD -------|-NED",
        "     | /          | /",
        "     |/           |/",
        "    SWD -------- SED",
        "",
        "C = Central node (connected to all vertices)",
    ]
    return "\n".join(lines)

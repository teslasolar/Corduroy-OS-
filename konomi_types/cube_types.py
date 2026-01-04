"""
KONOMI UDT Layer 1: Cube Types
Type definitions for Cube graph operations
"""

from typing import List, Dict, Any, NamedTuple, Literal


# Vertex identifier type
VertexID = Literal[
    'NEU', 'NED', 'NWU', 'NWD',
    'SEU', 'SED', 'SWU', 'SWD',
    'central'
]

# All vertex positions (8 corners of cube)
VERTICES: List[str] = [
    'NEU', 'NED', 'NWU', 'NWD',
    'SEU', 'SED', 'SWU', 'SWD'
]


class MessageRecord(NamedTuple):
    """Record of a message sent between nodes."""
    from_vertex: str
    to_vertex: str
    message: str
    response: str


class CubeStatus(NamedTuple):
    """Status information for a Cube instance."""
    vertices: int
    central: int
    total_nodes: int
    total_edges: int
    messages_sent: int
    active_llms: int
    edge_list: Dict[str, List[str]]

    @classmethod
    def from_cube(
        cls,
        edges: Dict[str, List[str]],
        message_count: int
    ) -> 'CubeStatus':
        """Create status from cube state."""
        total_edges = sum(len(e) for e in edges.values()) // 2
        return cls(
            vertices=len(VERTICES),
            central=1,
            total_nodes=len(VERTICES) + 1,
            total_edges=total_edges,
            messages_sent=message_count,
            active_llms=len(VERTICES) + 1,
            edge_list=dict(edges)
        )


def are_adjacent(v1: str, v2: str) -> bool:
    """
    Check if two vertices are adjacent (share a cube edge).
    Adjacent vertices differ in exactly 1 axis.
    """
    if len(v1) != 3 or len(v2) != 3:
        return False
    diffs = sum(c1 != c2 for c1, c2 in zip(v1, v2))
    return diffs == 1

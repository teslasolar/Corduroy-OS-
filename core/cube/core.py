"""
KONOMI Core - Cube Core
Cube class with 9-node graph structure
"""

from typing import Dict, List, Any
from collections import defaultdict
from ..femto import Femto
from .geometry import VERTICES, build_cube_edges, get_valid_nodes


class Cube:
    """
    Cube geometry with 9 LLM nodes.

    Structure:
    - 8 vertex nodes (corners of a cube)
    - 1 central node (hub)
    - Edge connections for message passing

    Vertex naming: [N/S][E/W][U/D]

    Targets:
    - 9 concurrent LLM instances
    - <10ms message passing between nodes
    """

    VERTICES = VERTICES

    def __init__(self):
        """Initialize Cube with 9 Femto LLM nodes."""
        # Each vertex has a Femto LLM
        self.v: Dict[str, Femto] = {}
        for i, vertex in enumerate(self.VERTICES):
            self.v[vertex] = Femto(seed=i)

        # Central node (hub)
        self.c = Femto(seed=999)

        # Edges for message passing
        self.edges: Dict[str, List[str]] = defaultdict(list)
        self.edges.update(build_cube_edges())

        # Message history for debugging
        self.message_history: List[Dict[str, Any]] = []

    def connect(self, v1: str, v2: str) -> None:
        """Add custom edge between two nodes."""
        valid_nodes = get_valid_nodes()
        if v1 not in valid_nodes or v2 not in valid_nodes:
            raise ValueError(
                f"Invalid nodes: {v1}, {v2}. "
                f"Valid: {', '.join(valid_nodes)}"
            )
        if v2 not in self.edges[v1]:
            self.edges[v1].append(v2)
        if v1 not in self.edges[v2]:
            self.edges[v2].append(v1)

    def disconnect(self, v1: str, v2: str) -> bool:
        """Remove edge between two nodes."""
        removed = False
        if v2 in self.edges[v1]:
            self.edges[v1].remove(v2)
            removed = True
        if v1 in self.edges[v2]:
            self.edges[v2].remove(v1)
            removed = True
        return removed

    def get_neighbors(self, vertex: str) -> List[str]:
        """Get all nodes connected to a vertex."""
        return self.edges[vertex]

    def __repr__(self):
        total_edges = sum(len(e) for e in self.edges.values()) // 2
        return (
            f"Cube(nodes={len(self.VERTICES) + 1}, "
            f"edges={total_edges}, "
            f"messages={len(self.message_history)})"
        )

    # Import methods from other modules
    from .messaging import send_message, broadcast, process_at_vertex
    from .pathfinding import shortest_path, status, reset_history

    def _are_adjacent(self, v1: str, v2: str) -> bool:
        """Check if two vertices are adjacent (share a cube edge)."""
        from .geometry import are_adjacent
        return are_adjacent(v1, v2)

    def visualize_graph(self) -> str:
        """Create ASCII visualization of the cube graph."""
        from .geometry import visualize_cube
        viz = visualize_cube()
        return viz + f"\nTotal edges: {self.status()['total_edges']}"

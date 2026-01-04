"""
KONOMI Core - Cube Pathfinding
Graph algorithms and analytics
"""

from typing import Dict, List, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Cube

from .geometry import VERTICES


def shortest_path(self: 'Cube', start: str, end: str) -> Optional[List[str]]:
    """
    Find shortest path between two nodes using BFS.

    Args:
        start: Start node
        end: End node

    Returns:
        List of nodes in path, or None if no path exists
    """
    if start == end:
        return [start]

    # BFS
    queue = [(start, [start])]
    visited = {start}

    while queue:
        node, path = queue.pop(0)

        for neighbor in self.edges[node]:
            if neighbor == end:
                return path + [neighbor]

            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return None


def status(self: 'Cube') -> Dict[str, Any]:
    """
    Get Cube status information.

    Returns:
        Dict with node count, edges, and activity stats
    """
    total_edges = sum(len(edges) for edges in self.edges.values()) // 2

    return {
        'vertices': len(VERTICES),
        'central': 1,
        'total_nodes': len(VERTICES) + 1,
        'total_edges': total_edges,
        'messages_sent': len(self.message_history),
        'active_llms': len(self.v) + 1,
        'edge_list': dict(self.edges)
    }


def reset_history(self: 'Cube') -> int:
    """
    Clear message history.

    Returns:
        Number of messages cleared
    """
    count = len(self.message_history)
    self.message_history = []
    return count

"""
Konomi Corduroy-OS - Cube Module
9-node graph geometry with message passing

🎲 Cube: 8 vertices + 1 central node with Femto LLMs
"""

from collections import defaultdict
from typing import List, Dict, Any, Optional
from .femto import Femto


class Cube:
    """
    Cube geometry with 9 LLM nodes.

    Structure:
    - 8 vertex nodes (corners of a cube)
    - 1 central node (hub)
    - Edge connections for message passing

    Vertex naming: [N/S][E/W][U/D]
    - N/S: North/South
    - E/W: East/West
    - U/D: Up/Down

    Examples:
        NEU = North-East-Up (top-front-right corner)
        SWD = South-West-Down (bottom-back-left corner)

    Targets:
    - 9 concurrent LLM instances
    - <10ms message passing between nodes
    - WebSocket communication on port 6789

    Examples:
        >>> cube = Cube()
        >>> cube.connect('NEU', 'SWD')
        >>> msg = cube.send_message('NEU', 'SWD', 'Hello')
        >>> cube.status()
    """

    # 8 vertex positions: North/South, East/West, Up/Down
    VERTICES = ['NEU', 'NED', 'NWU', 'NWD', 'SEU', 'SED', 'SWU', 'SWD']

    def __init__(self):
        """Initialize Cube with 9 Femto LLM nodes."""

        # Each vertex has a Femto LLM
        self.v: Dict[str, Femto] = {}
        for i, vertex in enumerate(self.VERTICES):
            self.v[vertex] = Femto(seed=i)

        # Central node (hub)
        self.c = Femto(seed=999)

        # Edges: defaultdict for message passing
        # Structure: {node: [list of connected nodes]}
        self.edges: Dict[str, List[str]] = defaultdict(list)

        # Build standard cube edge connections
        self._build_cube_edges()

        # Message history for debugging
        self.message_history: List[Dict[str, Any]] = []

    def _build_cube_edges(self) -> None:
        """
        Build standard cube edge connections.

        Connects:
        1. Vertices that differ by exactly 1 axis (cube edges)
        2. All vertices to central node (hub-spoke)
        """
        # Connect vertices that differ by exactly 1 axis
        for v1 in self.VERTICES:
            for v2 in self.VERTICES:
                if v1 != v2 and self._are_adjacent(v1, v2):
                    if v2 not in self.edges[v1]:
                        self.edges[v1].append(v2)

        # Connect all vertices to central node (bidirectional)
        for v in self.VERTICES:
            self.edges[v].append('central')
            self.edges['central'].append(v)

    def _are_adjacent(self, v1: str, v2: str) -> bool:
        """
        Check if two vertices are adjacent (share a cube edge).

        Adjacent vertices differ in exactly 1 axis.

        Args:
            v1, v2: Vertex names

        Returns:
            True if vertices share an edge

        Examples:
            >>> cube = Cube()
            >>> cube._are_adjacent('NEU', 'NED')  # Same N, E, differ in U/D
            True
            >>> cube._are_adjacent('NEU', 'SWD')  # Differ in all 3
            False
        """
        diffs = sum(c1 != c2 for c1, c2 in zip(v1, v2))
        return diffs == 1

    def connect(self, v1: str, v2: str) -> None:
        """
        Add custom edge between two nodes.

        Creates bidirectional connection.

        Args:
            v1, v2: Node names (vertex or 'central')

        Raises:
            ValueError: If node names are invalid

        Examples:
            >>> cube = Cube()
            >>> cube.connect('NEU', 'SWD')  # Connect opposite corners
        """
        # Validate node names
        valid_nodes = self.VERTICES + ['central']
        if v1 not in valid_nodes or v2 not in valid_nodes:
            raise ValueError(
                f"Invalid nodes: {v1}, {v2}. "
                f"Valid: {', '.join(valid_nodes)}"
            )

        # Add bidirectional edges
        if v2 not in self.edges[v1]:
            self.edges[v1].append(v2)
        if v1 not in self.edges[v2]:
            self.edges[v2].append(v1)

    def disconnect(self, v1: str, v2: str) -> bool:
        """
        Remove edge between two nodes.

        Args:
            v1, v2: Node names

        Returns:
            True if edge existed and was removed
        """
        removed = False

        if v2 in self.edges[v1]:
            self.edges[v1].remove(v2)
            removed = True

        if v1 in self.edges[v2]:
            self.edges[v2].remove(v1)
            removed = True

        return removed

    def send_message(
        self,
        from_vertex: str,
        to_vertex: str,
        message: str
    ) -> str:
        """
        Send message from one node to another.

        Message is processed through the receiving LLM.

        Args:
            from_vertex: Source node
            to_vertex: Destination node
            message: Message content

        Returns:
            Processed message from receiving LLM

        Raises:
            ValueError: If no edge exists between nodes

        Examples:
            >>> cube = Cube()
            >>> response = cube.send_message('NEU', 'central', 'Status?')
        """
        # Validate edge exists
        if to_vertex not in self.edges[from_vertex]:
            raise ValueError(
                f"No edge from {from_vertex} to {to_vertex}. "
                f"Available edges: {self.edges[from_vertex]}"
            )

        # Get receiving LLM
        if to_vertex == 'central':
            receiver = self.c
        else:
            receiver = self.v[to_vertex]

        # Process message through LLM
        response = receiver.p(message)

        # Log message
        self.message_history.append({
            'from': from_vertex,
            'to': to_vertex,
            'message': message,
            'response': response
        })

        return response

    def broadcast(
        self,
        from_vertex: str,
        message: str
    ) -> Dict[str, str]:
        """
        Broadcast message from one node to all connected nodes.

        Args:
            from_vertex: Source node
            message: Message to broadcast

        Returns:
            Dict of {node: response} from each connected node

        Examples:
            >>> cube = Cube()
            >>> responses = cube.broadcast('central', 'Ping all')
        """
        responses = {}

        for to_vertex in self.edges[from_vertex]:
            response = self.send_message(from_vertex, to_vertex, message)
            responses[to_vertex] = response

        return responses

    def process_at_vertex(self, vertex: str, data: str) -> str:
        """
        Process data at a specific vertex LLM.

        Args:
            vertex: Vertex name or 'central'
            data: Data to process

        Returns:
            Processed result

        Examples:
            >>> cube = Cube()
            >>> result = cube.process_at_vertex('NEU', 'analyze this')
        """
        if vertex == 'central':
            llm = self.c
        elif vertex in self.v:
            llm = self.v[vertex]
        else:
            raise ValueError(f"Invalid vertex: {vertex}")

        return llm.p(data)

    def get_neighbors(self, vertex: str) -> List[str]:
        """
        Get all nodes connected to a vertex.

        Args:
            vertex: Vertex name or 'central'

        Returns:
            List of connected node names
        """
        return self.edges[vertex]

    def shortest_path(self, start: str, end: str) -> Optional[List[str]]:
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

    def status(self) -> Dict[str, Any]:
        """
        Get Cube status information.

        Returns:
            Dict with node count, edges, and activity stats
        """
        total_edges = sum(len(edges) for edges in self.edges.values()) // 2

        return {
            'vertices': len(self.VERTICES),
            'central': 1,
            'total_nodes': len(self.VERTICES) + 1,
            'total_edges': total_edges,
            'messages_sent': len(self.message_history),
            'active_llms': len(self.v) + 1,
            'edge_list': dict(self.edges)
        }

    def reset_history(self) -> int:
        """
        Clear message history.

        Returns:
            Number of messages cleared
        """
        count = len(self.message_history)
        self.message_history = []
        return count

    def visualize_graph(self) -> str:
        """
        Create ASCII visualization of the cube graph.

        Returns:
            ASCII art string showing cube structure
        """
        viz = []
        viz.append("Cube Graph Topology:")
        viz.append("")
        viz.append("        NWU -------- NEU")
        viz.append("       /|           /|")
        viz.append("      / |          / |")
        viz.append("    SWU -------- SEU |")
        viz.append("     |  |    C   |  |")
        viz.append("     | NWD -------|-NED")
        viz.append("     | /          | /")
        viz.append("     |/           |/")
        viz.append("    SWD -------- SED")
        viz.append("")
        viz.append("C = Central node (connected to all vertices)")
        viz.append(f"Total edges: {self.status()['total_edges']}")

        return "\n".join(viz)

    def __repr__(self):
        status = self.status()
        return (
            f"Cube(nodes={status['total_nodes']}, "
            f"edges={status['total_edges']}, "
            f"messages={status['messages_sent']})"
        )

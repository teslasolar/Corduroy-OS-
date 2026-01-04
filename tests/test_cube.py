"""
Tests for Cube 9-node graph
"""

import pytest
from core.cube import Cube


class TestCube:
    def setup_method(self):
        """Setup for each test."""
        self.cube = Cube()

    def test_initialization(self):
        """Test Cube initialization."""
        assert len(self.cube.v) == 8  # 8 vertices
        assert self.cube.c is not None  # Central node
        assert len(self.cube.VERTICES) == 8

    def test_vertices_named_correctly(self):
        """Test vertex naming."""
        expected = ['NEU', 'NED', 'NWU', 'NWD', 'SEU', 'SED', 'SWU', 'SWD']

        for vertex in expected:
            assert vertex in self.cube.v

    def test_cube_edges_built(self):
        """Test that cube edges are auto-built."""
        # NEU should connect to NED (differ in U/D)
        assert 'NED' in self.cube.edges['NEU']

        # All vertices should connect to central
        for vertex in self.cube.VERTICES:
            assert 'central' in self.cube.edges[vertex]
            assert vertex in self.cube.edges['central']

    def test_are_adjacent(self):
        """Test adjacency checking."""
        # NEU and NED differ only in U/D
        assert self.cube._are_adjacent('NEU', 'NED')

        # NEU and SED differ in N/S and U/D (2 axes)
        assert not self.cube._are_adjacent('NEU', 'SED')

        # Vertex not adjacent to itself
        assert not self.cube._are_adjacent('NEU', 'NEU')

    def test_connect_vertices(self):
        """Test connecting custom vertices."""
        self.cube.connect('NEU', 'SWD')

        assert 'SWD' in self.cube.edges['NEU']
        assert 'NEU' in self.cube.edges['SWD']

    def test_disconnect_vertices(self):
        """Test disconnecting vertices."""
        self.cube.connect('NEU', 'SWD')
        removed = self.cube.disconnect('NEU', 'SWD')

        assert removed
        assert 'SWD' not in self.cube.edges['NEU']
        assert 'NEU' not in self.cube.edges['SWD']

    def test_disconnect_nonexistent(self):
        """Test disconnecting non-existent edge."""
        removed = self.cube.disconnect('NEU', 'SWD')

        # May or may not exist depending on cube structure
        assert isinstance(removed, bool)

    def test_send_message(self):
        """Test message sending."""
        response = self.cube.send_message('NEU', 'central', 'test')

        assert response == "[test]"
        assert len(self.cube.message_history) == 1

    def test_send_message_no_edge(self):
        """Test sending message with no edge raises error."""
        # NEU and SWD are not connected by default
        with pytest.raises(ValueError, match="No edge"):
            self.cube.send_message('NEU', 'SWD', 'test')

    def test_broadcast(self):
        """Test broadcasting message."""
        responses = self.cube.broadcast('central', 'ping')

        # Central is connected to all 8 vertices
        assert len(responses) == 8

        for vertex in self.cube.VERTICES:
            assert vertex in responses
            assert responses[vertex] == "[ping]"

    def test_process_at_vertex(self):
        """Test processing at specific vertex."""
        result = self.cube.process_at_vertex('NEU', 'data')

        assert result == "[data]"

    def test_process_at_central(self):
        """Test processing at central node."""
        result = self.cube.process_at_vertex('central', 'data')

        assert result == "[data]"

    def test_process_at_invalid_vertex(self):
        """Test processing at invalid vertex."""
        with pytest.raises(ValueError, match="Invalid vertex"):
            self.cube.process_at_vertex('INVALID', 'data')

    def test_get_neighbors(self):
        """Test getting neighbors."""
        neighbors = self.cube.get_neighbors('central')

        # Central connects to all 8 vertices
        assert len(neighbors) == 8

    def test_shortest_path_direct(self):
        """Test shortest path for directly connected nodes."""
        # NEU and NED are adjacent
        path = self.cube.shortest_path('NEU', 'NED')

        assert path == ['NEU', 'NED']

    def test_shortest_path_via_central(self):
        """Test shortest path via central node."""
        # All paths can go through central
        path = self.cube.shortest_path('NEU', 'SWD')

        assert path is not None
        assert path[0] == 'NEU'
        assert path[-1] == 'SWD'

    def test_shortest_path_same_node(self):
        """Test shortest path to same node."""
        path = self.cube.shortest_path('NEU', 'NEU')

        assert path == ['NEU']

    def test_status(self):
        """Test status retrieval."""
        status = self.cube.status()

        assert status['vertices'] == 8
        assert status['central'] == 1
        assert status['total_nodes'] == 9
        assert status['active_llms'] == 9
        assert 'total_edges' in status
        assert 'messages_sent' in status

    def test_reset_history(self):
        """Test resetting message history."""
        self.cube.send_message('NEU', 'central', 'test1')
        self.cube.send_message('NED', 'central', 'test2')

        count = self.cube.reset_history()

        assert count == 2
        assert len(self.cube.message_history) == 0

    def test_visualize_graph(self):
        """Test graph visualization."""
        viz = self.cube.visualize_graph()

        assert 'Cube Graph Topology' in viz
        assert 'NWU' in viz
        assert 'SEU' in viz
        assert 'Central' in viz

    def test_repr(self):
        """Test string representation."""
        repr_str = repr(self.cube)

        assert 'Cube' in repr_str
        assert 'nodes=' in repr_str
        assert 'edges=' in repr_str

    def test_message_history(self):
        """Test message history tracking."""
        self.cube.send_message('NEU', 'central', 'msg1')
        self.cube.send_message('central', 'NED', 'msg2')

        assert len(self.cube.message_history) == 2

        msg1 = self.cube.message_history[0]
        assert msg1['from'] == 'NEU'
        assert msg1['to'] == 'central'
        assert msg1['message'] == 'msg1'
        assert msg1['response'] == '[msg1]'

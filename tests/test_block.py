"""
Tests for Block sparse 3D grid
"""

import pytest
from core.block import Block


class TestBlock:
    def setup_method(self):
        """Setup for each test."""
        self.block = Block(dims=(10, 10, 10))

    def test_initialization(self):
        """Test Block initialization."""
        assert self.block.dims == (10, 10, 10)
        assert len(self.block.a) == 0
        assert len(self.block.L) == 0

    def test_set_get(self):
        """Test setting and getting values."""
        self.block.set(5, 5, 5, 42.0)
        value = self.block.get(5, 5, 5)

        assert value == 42.0

    def test_get_default(self):
        """Test getting non-existent value returns default."""
        value = self.block.get(1, 1, 1, default=0.0)

        assert value == 0.0

    def test_set_out_of_bounds(self):
        """Test setting out of bounds raises error."""
        with pytest.raises(ValueError, match="Out of bounds"):
            self.block.set(100, 100, 100, 1.0)

    def test_sparse_storage(self):
        """Test sparse storage (only stores non-zero)."""
        self.block.set(0, 0, 0, 1.0)
        self.block.set(1, 1, 1, 2.0)

        assert len(self.block.a) == 2

        # Setting to zero should remove from storage
        self.block.set(0, 0, 0, 0.0)
        assert len(self.block.a) == 1

    def test_at_creates_llm(self):
        """Test that at() creates LLM at coordinate."""
        llm = self.block.at(3, 3, 3)

        assert (3, 3, 3) in self.block.L
        assert llm is not None

    def test_at_returns_same_llm(self):
        """Test that at() returns same LLM instance."""
        llm1 = self.block.at(5, 5, 5)
        llm2 = self.block.at(5, 5, 5)

        assert llm1 is llm2

    def test_at_different_coordinates(self):
        """Test that different coordinates have different LLMs."""
        llm1 = self.block.at(1, 1, 1)
        llm2 = self.block.at(2, 2, 2)

        assert llm1 is not llm2

    def test_face_ops(self):
        """Test face_ops returns active cell count."""
        self.block.set(0, 0, 0, 1.0)
        self.block.set(1, 1, 1, 2.0)
        self.block.set(2, 2, 2, 3.0)

        assert self.block.face_ops() == 3

    def test_clear_region(self):
        """Test clearing a region."""
        # Set values in region
        for i in range(5):
            self.block.set(i, i, i, float(i))

        # Clear region (0,0,0) to (2,2,2)
        cleared = self.block.clear_region((0, 2), (0, 2), (0, 2))

        assert cleared == 3  # 0,0,0 and 1,1,1 and 2,2,2
        assert self.block.get(0, 0, 0) == 0.0
        assert self.block.get(3, 3, 3) == 3.0  # Outside region

    def test_query_region(self):
        """Test querying a region."""
        self.block.set(1, 1, 1, 1.0)
        self.block.set(2, 2, 2, 2.0)
        self.block.set(5, 5, 5, 5.0)

        region = self.block.query_region((0, 3), (0, 3), (0, 3))

        assert len(region) == 2
        assert region[(1, 1, 1)] == 1.0
        assert region[(2, 2, 2)] == 2.0
        assert (5, 5, 5) not in region

    def test_broadcast_to_llms(self):
        """Test broadcasting message to LLMs."""
        # Create LLMs at coordinates
        self.block.at(1, 1, 1)
        self.block.at(2, 2, 2)
        self.block.at(5, 5, 5)

        # Broadcast to region
        responses = self.block.broadcast_to_llms(
            "test message",
            x_range=(0, 3),
            y_range=(0, 3),
            z_range=(0, 3)
        )

        assert len(responses) == 2  # Only (1,1,1) and (2,2,2) in range
        assert responses[(1, 1, 1)] == "[test message]"

    def test_density(self):
        """Test density calculation."""
        # Empty block
        assert self.block.density() == 0.0

        # Add some cells
        for i in range(10):
            self.block.set(i, 0, 0, 1.0)

        expected_density = 10 / (10 * 10 * 10)
        assert abs(self.block.density() - expected_density) < 0.0001

    def test_memory_estimate(self):
        """Test memory estimation."""
        # Create some cells and LLMs
        self.block.set(0, 0, 0, 1.0)
        self.block.at(0, 0, 0)

        mem = self.block.memory_estimate()

        assert 'cells_mb' in mem
        assert 'llms_mb' in mem
        assert 'total_mb' in mem
        assert mem['total_mb'] >= 0

    def test_info(self):
        """Test info retrieval."""
        self.block.set(1, 1, 1, 1.0)
        self.block.at(2, 2, 2)

        info = self.block.info()

        assert info['dims'] == (10, 10, 10)
        assert info['active_cells'] == 1
        assert info['active_llms'] == 1
        assert 'density' in info
        assert 'memory_mb' in info

    def test_repr(self):
        """Test string representation."""
        repr_str = repr(self.block)

        assert 'Block' in repr_str
        assert 'dims=' in repr_str

    def test_large_grid(self):
        """Test with larger grid."""
        large_block = Block(dims=(1000, 1000, 1000))

        large_block.set(500, 500, 500, 42.0)
        assert large_block.get(500, 500, 500) == 42.0

        # Should still be sparse
        assert len(large_block.a) == 1

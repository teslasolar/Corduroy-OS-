"""
Tests for Femto 16d nano LLM
"""

import pytest
import jax.numpy as jnp
from core.femto import Femto


class TestFemto:
    def setup_method(self):
        """Setup for each test."""
        self.femto = Femto(dim=16, seed=42)

    def test_initialization(self):
        """Test Femto initialization."""
        assert self.femto.dim == 16
        assert self.femto.W.shape == (16, 16)
        assert self.femto.b.shape == (16,)

    def test_p_basic(self):
        """Test basic text processing."""
        result = self.femto.p("Hello world")

        assert result == "[Hello world]"

    def test_p_truncation(self):
        """Test text truncation to 50 chars."""
        long_text = "x" * 100
        result = self.femto.p(long_text)

        assert len(result) == 52  # "[" + 50 chars + "]"
        assert result == f"[{'x' * 50}]"

    def test_forward_pass(self):
        """Test forward pass."""
        x = jnp.ones(16)
        result = self.femto.forward(x)

        assert result.shape == (16,)

    def test_encode_text(self):
        """Test text encoding."""
        vec = self.femto.encode_text("test")

        assert vec.shape == (16,)
        # Vector should be normalized
        norm = jnp.linalg.norm(vec)
        assert jnp.isclose(norm, 1.0) or jnp.isclose(norm, 0.0)

    def test_encode_different_texts(self):
        """Test that different texts produce different encodings."""
        vec1 = self.femto.encode_text("hello")
        vec2 = self.femto.encode_text("world")

        # Vectors should be different
        assert not jnp.array_equal(vec1, vec2)

    def test_encode_same_text(self):
        """Test that same text produces same encoding."""
        vec1 = self.femto.encode_text("test")
        vec2 = self.femto.encode_text("test")

        # Vectors should be identical
        assert jnp.array_equal(vec1, vec2)

    def test_process_full(self):
        """Test full processing pipeline."""
        result = self.femto.process_full("analyze this")

        assert result.shape == (16,)

    def test_similarity_identical(self):
        """Test similarity of identical texts."""
        sim = self.femto.similarity("test", "test")

        assert 0.0 <= sim <= 1.0
        assert sim > 0.5  # Should be high for identical texts

    def test_similarity_different(self):
        """Test similarity of different texts."""
        sim = self.femto.similarity("cat", "dog")

        assert 0.0 <= sim <= 1.0

    def test_update_weights(self):
        """Test weight update."""
        x = jnp.ones(16)
        y = jnp.zeros(16)

        loss = self.femto.update_weights(x, y, lr=0.01)

        assert isinstance(loss, float)
        assert loss >= 0

    def test_info(self):
        """Test info retrieval."""
        info = self.femto.info()

        assert 'id' in info
        assert 'dim' in info
        assert 'params' in info
        assert 'weight_shape' in info
        assert 'memory_estimate_kb' in info

        assert info['dim'] == 16
        assert info['params'] == 256 + 16  # W + b

    def test_repr(self):
        """Test string representation."""
        repr_str = repr(self.femto)

        assert 'Femto' in repr_str
        assert 'dim=16' in repr_str
        assert 'params=' in repr_str

    def test_deterministic_initialization(self):
        """Test that same seed produces same weights."""
        femto1 = Femto(dim=16, seed=123)
        femto2 = Femto(dim=16, seed=123)

        assert jnp.array_equal(femto1.W, femto2.W)
        assert jnp.array_equal(femto1.b, femto2.b)

    def test_different_seeds(self):
        """Test that different seeds produce different weights."""
        femto1 = Femto(dim=16, seed=1)
        femto2 = Femto(dim=16, seed=2)

        assert not jnp.array_equal(femto1.W, femto2.W)

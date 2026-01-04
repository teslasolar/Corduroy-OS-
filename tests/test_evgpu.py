"""
Tests for eVGPU CPU-only tensor operations
"""

import pytest
import jax.numpy as jnp
from core.evgpu import eVGPU


class TestEVGPU:
    def setup_method(self):
        """Setup for each test."""
        self.evgpu = eVGPU()

    def test_matmul(self):
        """Test matrix multiplication."""
        a = jnp.ones((4, 4))
        b = jnp.ones((4, 4))

        result = self.evgpu.t(a, b, '@')

        assert result.shape == (4, 4)
        # Each element should be 4.0 (1+1+1+1)
        assert jnp.allclose(result, jnp.full((4, 4), 4.0))

    def test_addition(self):
        """Test element-wise addition."""
        a = jnp.array([[1, 2], [3, 4]])
        b = jnp.array([[5, 6], [7, 8]])

        result = self.evgpu.t(a, b, '+')

        expected = jnp.array([[6, 8], [10, 12]])
        assert jnp.array_equal(result, expected)

    def test_subtraction(self):
        """Test element-wise subtraction."""
        a = jnp.array([[5, 6], [7, 8]])
        b = jnp.array([[1, 2], [3, 4]])

        result = self.evgpu.t(a, b, '-')

        expected = jnp.array([[4, 4], [4, 4]])
        assert jnp.array_equal(result, expected)

    def test_multiplication(self):
        """Test element-wise multiplication."""
        a = jnp.array([[2, 3], [4, 5]])
        b = jnp.array([[2, 2], [2, 2]])

        result = self.evgpu.t(a, b, '*')

        expected = jnp.array([[4, 6], [8, 10]])
        assert jnp.array_equal(result, expected)

    def test_jit_matmul(self):
        """Test JIT-compiled matmul."""
        a = jnp.ones((3, 3))
        b = jnp.ones((3, 3))

        result = self.evgpu.matmul_jit(a, b)

        assert result.shape == (3, 3)
        assert jnp.allclose(result, jnp.full((3, 3), 3.0))

    def test_jit_add(self):
        """Test JIT-compiled addition."""
        a = jnp.array([1, 2, 3])
        b = jnp.array([4, 5, 6])

        result = self.evgpu.add_jit(a, b)

        expected = jnp.array([5, 7, 9])
        assert jnp.array_equal(result, expected)

    def test_invalid_operation(self):
        """Test invalid operation raises error."""
        a = jnp.ones((2, 2))
        b = jnp.ones((2, 2))

        with pytest.raises(ValueError, match="Unsupported operation"):
            self.evgpu.t(a, b, '/')

    def test_info(self):
        """Test eVGPU info."""
        info = self.evgpu.info()

        assert 'platform' in info
        assert info['platform'] == 'cpu'
        assert 'devices' in info
        assert 'version' in info

    def test_numpy_array_input(self):
        """Test that numpy arrays work as input."""
        import numpy as np

        a = np.ones((2, 2))
        b = np.ones((2, 2))

        result = self.evgpu.t(a, b, '@')

        assert result.shape == (2, 2)
        assert jnp.allclose(result, jnp.full((2, 2), 2.0))

    def test_large_matrix(self):
        """Test with larger matrices."""
        a = jnp.ones((100, 100))
        b = jnp.ones((100, 100))

        result = self.evgpu.t(a, b, '@')

        assert result.shape == (100, 100)
        assert jnp.allclose(result, jnp.full((100, 100), 100.0))

    def test_cpu_only(self):
        """Verify that no GPU is being used."""
        import jax

        # Should be running on CPU
        assert jax.default_backend() == 'cpu'

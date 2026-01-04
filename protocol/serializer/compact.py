"""
KONOMI Protocol - Compact Serializer
Optimized serialization for small tensors (Femto weights)
"""

import struct
import numpy as np
import jax.numpy as jnp
from typing import Any
from .dtypes import DType, DTYPE_MAP, REVERSE_DTYPE_MAP


class CompactSerializer:
    """
    Even more compact serialization for small tensors.

    Used for Femto LLM weights (16x16 = 256 floats).
    Optimized for minimal overhead.

    Format for 16x16 float32:
    [dtype:1][16:1][16:1][data:1024]
    Total: 1027 bytes
    """

    @staticmethod
    def serialize_16x16(array: Any) -> bytes:
        """
        Serialize 16x16 array (Femto weights).

        Args:
            array: 16x16 JAX or NumPy array

        Returns:
            Binary data (1027 bytes for float32)
        """
        if isinstance(array, jnp.ndarray):
            array = np.array(array)

        if array.shape != (16, 16):
            raise ValueError(f"Expected (16, 16), got {array.shape}")

        dtype_code = DTYPE_MAP.get(array.dtype.type, DType.FLOAT32)
        header = struct.pack('!BBB', dtype_code, 16, 16)
        data = np.ascontiguousarray(array).tobytes()

        return header + data

    @staticmethod
    def deserialize_16x16(data: bytes) -> jnp.ndarray:
        """
        Deserialize 16x16 array.

        Args:
            data: Binary data from serialize_16x16()

        Returns:
            16x16 JAX array
        """
        dtype_code, dim1, dim2 = struct.unpack('!BBB', data[:3])

        if (dim1, dim2) != (16, 16):
            raise ValueError(f"Expected 16x16, got {dim1}x{dim2}")

        dtype = REVERSE_DTYPE_MAP.get(dtype_code, np.float32)
        array = np.frombuffer(data[3:], dtype=dtype)
        array = array.reshape((16, 16))

        return jnp.array(array)


def serialize_femto_weights(weights: Any) -> bytes:
    """Serialize Femto 16x16 weights (optimized)."""
    return CompactSerializer.serialize_16x16(weights)


def deserialize_femto_weights(data: bytes) -> jnp.ndarray:
    """Deserialize Femto 16x16 weights."""
    return CompactSerializer.deserialize_16x16(data)

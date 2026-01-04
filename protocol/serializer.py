"""
Konomi Corduroy-OS - Tensor Serializer
Efficient binary serialization for JAX tensors

Binary format optimized for 56k-style packet transmission
"""

import struct
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Any
from enum import IntEnum


class DType(IntEnum):
    """
    Data type encoding for tensors.

    Maps to numpy/JAX dtypes with single-byte encoding.
    """
    FLOAT32 = 0x01
    FLOAT64 = 0x02
    INT32 = 0x03
    INT64 = 0x04
    UINT32 = 0x05
    UINT64 = 0x06
    BOOL = 0x07
    FLOAT16 = 0x08


# Mapping between numpy dtypes and our encoding
DTYPE_MAP = {
    np.float32: DType.FLOAT32,
    np.float64: DType.FLOAT64,
    np.int32: DType.INT32,
    np.int64: DType.INT64,
    np.uint32: DType.UINT32,
    np.uint64: DType.UINT64,
    np.bool_: DType.BOOL,
    np.float16: DType.FLOAT16,
}

REVERSE_DTYPE_MAP = {v: k for k, v in DTYPE_MAP.items()}


class TensorSerializer:
    """
    Efficient binary serialization for JAX/NumPy arrays.

    Format:
    [Header][Shape][Data]

    Header (5 bytes):
    - dtype (1 byte): Data type code
    - ndim (1 byte): Number of dimensions
    - size (3 bytes): Total element count (up to 16M elements)

    Shape (ndim * 2 bytes):
    - Each dimension as uint16 (up to 65535 per dimension)

    Data (size * dtype_size bytes):
    - Raw array data in C order

    Examples:
        >>> serializer = TensorSerializer()
        >>> arr = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        >>> binary = serializer.serialize(arr)
        >>> restored = serializer.deserialize(binary)
    """

    HEADER_SIZE = 5
    MAX_ELEMENTS = 16_777_215  # 3 bytes = 2^24 - 1
    MAX_DIM_SIZE = 65535  # 2 bytes = 2^16 - 1

    @classmethod
    def serialize(cls, array: Any) -> bytes:
        """
        Serialize array to binary format.

        Args:
            array: JAX or NumPy array

        Returns:
            Binary representation

        Raises:
            ValueError: If array is too large or unsupported dtype
        """
        # Convert JAX array to numpy for serialization
        if isinstance(array, jnp.ndarray):
            array = np.array(array)

        # Validate dtype
        dtype = array.dtype.type
        if dtype not in DTYPE_MAP:
            raise ValueError(f"Unsupported dtype: {array.dtype}")

        # Validate size
        if array.size > cls.MAX_ELEMENTS:
            raise ValueError(
                f"Array too large: {array.size} > {cls.MAX_ELEMENTS}"
            )

        # Validate dimensions
        if any(d > cls.MAX_DIM_SIZE for d in array.shape):
            raise ValueError(
                f"Dimension too large: max {cls.MAX_DIM_SIZE}"
            )

        # Pack header
        dtype_code = DTYPE_MAP[dtype]
        ndim = len(array.shape)
        size = array.size

        header = struct.pack(
            '!BB3s',  # Big-endian: byte, byte, 3-byte int
            dtype_code,
            ndim,
            size.to_bytes(3, 'big')
        )

        # Pack shape
        shape_data = struct.pack(f'!{ndim}H', *array.shape)

        # Pack data (ensure C order)
        data = np.ascontiguousarray(array).tobytes()

        return header + shape_data + data

    @classmethod
    def deserialize(cls, data: bytes) -> jnp.ndarray:
        """
        Deserialize binary data to JAX array.

        Args:
            data: Binary data from serialize()

        Returns:
            JAX array

        Raises:
            ValueError: If data is invalid
        """
        if len(data) < cls.HEADER_SIZE:
            raise ValueError("Data too short for header")

        # Unpack header
        dtype_code, ndim, size_bytes = struct.unpack(
            '!BB3s',
            data[:cls.HEADER_SIZE]
        )
        size = int.from_bytes(size_bytes, 'big')

        # Get dtype
        if dtype_code not in REVERSE_DTYPE_MAP:
            raise ValueError(f"Unknown dtype code: {dtype_code}")
        dtype = REVERSE_DTYPE_MAP[dtype_code]

        # Unpack shape
        shape_size = ndim * 2
        shape_end = cls.HEADER_SIZE + shape_size

        if len(data) < shape_end:
            raise ValueError("Data too short for shape")

        shape = struct.unpack(
            f'!{ndim}H',
            data[cls.HEADER_SIZE:shape_end]
        )

        # Unpack data
        array_bytes = data[shape_end:]
        array = np.frombuffer(array_bytes, dtype=dtype)

        # Reshape
        array = array.reshape(shape)

        # Convert to JAX
        return jnp.array(array)

    @classmethod
    def estimate_size(cls, shape: Tuple[int, ...], dtype=np.float32) -> int:
        """
        Estimate serialized size for an array.

        Args:
            shape: Array shape
            dtype: Data type

        Returns:
            Estimated size in bytes
        """
        ndim = len(shape)
        size = np.prod(shape)

        # Header + shape + data
        header_size = cls.HEADER_SIZE
        shape_size = ndim * 2
        data_size = size * np.dtype(dtype).itemsize

        return header_size + shape_size + data_size


class CompactSerializer:
    """
    Even more compact serialization for small tensors.

    Used for Femto LLM weights (16x16 = 256 floats).
    Optimized for minimal overhead.

    Format for 16x16 float32:
    [dtype:1][16:1][16:1][data:1024]
    Total: 1027 bytes (vs 1029 for TensorSerializer)
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

        # Header: dtype + shape
        dtype_code = DTYPE_MAP.get(array.dtype.type, DType.FLOAT32)
        header = struct.pack('!BBB', dtype_code, 16, 16)

        # Data
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
        # Unpack header
        dtype_code, dim1, dim2 = struct.unpack('!BBB', data[:3])

        if (dim1, dim2) != (16, 16):
            raise ValueError(f"Expected 16x16, got {dim1}x{dim2}")

        # Get dtype
        dtype = REVERSE_DTYPE_MAP.get(dtype_code, np.float32)

        # Unpack data
        array = np.frombuffer(data[3:], dtype=dtype)
        array = array.reshape((16, 16))

        return jnp.array(array)


# Convenience functions

def serialize_tensor(array: Any) -> bytes:
    """Serialize tensor (convenience wrapper)."""
    return TensorSerializer.serialize(array)


def deserialize_tensor(data: bytes) -> jnp.ndarray:
    """Deserialize tensor (convenience wrapper)."""
    return TensorSerializer.deserialize(data)


def serialize_femto_weights(weights: Any) -> bytes:
    """Serialize Femto 16x16 weights (optimized)."""
    return CompactSerializer.serialize_16x16(weights)


def deserialize_femto_weights(data: bytes) -> jnp.ndarray:
    """Deserialize Femto 16x16 weights."""
    return CompactSerializer.deserialize_16x16(data)

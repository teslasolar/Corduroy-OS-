"""
KONOMI Protocol - Tensor Serializer
Main serialization for JAX/NumPy arrays
"""

import struct
import numpy as np
import jax.numpy as jnp
from typing import Tuple, Any
from .dtypes import DTYPE_MAP, REVERSE_DTYPE_MAP


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
    """

    HEADER_SIZE = 5
    MAX_ELEMENTS = 16_777_215  # 3 bytes = 2^24 - 1
    MAX_DIM_SIZE = 65535  # 2 bytes = 2^16 - 1

    @classmethod
    def serialize(cls, array: Any) -> bytes:
        """Serialize array to binary format."""
        if isinstance(array, jnp.ndarray):
            array = np.array(array)

        dtype = array.dtype.type
        if dtype not in DTYPE_MAP:
            raise ValueError(f"Unsupported dtype: {array.dtype}")

        if array.size > cls.MAX_ELEMENTS:
            raise ValueError(
                f"Array too large: {array.size} > {cls.MAX_ELEMENTS}"
            )

        if any(d > cls.MAX_DIM_SIZE for d in array.shape):
            raise ValueError(f"Dimension too large: max {cls.MAX_DIM_SIZE}")

        dtype_code = DTYPE_MAP[dtype]
        ndim = len(array.shape)
        size = array.size

        header = struct.pack(
            '!BB3s',
            dtype_code,
            ndim,
            size.to_bytes(3, 'big')
        )

        shape_data = struct.pack(f'!{ndim}H', *array.shape)
        data = np.ascontiguousarray(array).tobytes()

        return header + shape_data + data

    @classmethod
    def deserialize(cls, data: bytes) -> jnp.ndarray:
        """Deserialize binary data to JAX array."""
        if len(data) < cls.HEADER_SIZE:
            raise ValueError("Data too short for header")

        dtype_code, ndim, size_bytes = struct.unpack(
            '!BB3s',
            data[:cls.HEADER_SIZE]
        )
        size = int.from_bytes(size_bytes, 'big')

        if dtype_code not in REVERSE_DTYPE_MAP:
            raise ValueError(f"Unknown dtype code: {dtype_code}")
        dtype = REVERSE_DTYPE_MAP[dtype_code]

        shape_size = ndim * 2
        shape_end = cls.HEADER_SIZE + shape_size

        if len(data) < shape_end:
            raise ValueError("Data too short for shape")

        shape = struct.unpack(
            f'!{ndim}H',
            data[cls.HEADER_SIZE:shape_end]
        )

        array_bytes = data[shape_end:]
        array = np.frombuffer(array_bytes, dtype=dtype)
        array = array.reshape(shape)

        return jnp.array(array)

    @classmethod
    def estimate_size(cls, shape: Tuple[int, ...], dtype=np.float32) -> int:
        """Estimate serialized size for an array."""
        ndim = len(shape)
        size = np.prod(shape)
        header_size = cls.HEADER_SIZE
        shape_size = ndim * 2
        data_size = size * np.dtype(dtype).itemsize
        return header_size + shape_size + data_size


def serialize_tensor(array: Any) -> bytes:
    """Serialize tensor (convenience wrapper)."""
    return TensorSerializer.serialize(array)


def deserialize_tensor(data: bytes) -> jnp.ndarray:
    """Deserialize tensor (convenience wrapper)."""
    return TensorSerializer.deserialize(data)

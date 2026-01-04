"""
KONOMI Protocol - Serializer Module
Efficient binary serialization for JAX tensors

Re-exports main classes and functions for backward compatibility.
"""

from .dtypes import DType, DTYPE_MAP, REVERSE_DTYPE_MAP
from .tensor import TensorSerializer, serialize_tensor, deserialize_tensor
from .compact import CompactSerializer, serialize_femto_weights, deserialize_femto_weights

__all__ = [
    'DType',
    'DTYPE_MAP',
    'REVERSE_DTYPE_MAP',
    'TensorSerializer',
    'serialize_tensor',
    'deserialize_tensor',
    'CompactSerializer',
    'serialize_femto_weights',
    'deserialize_femto_weights',
]

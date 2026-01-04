"""
KONOMI Protocol - DTypes
Data type encoding for tensor serialization
"""

from enum import IntEnum
import numpy as np


class DType(IntEnum):
    """Data type encoding for tensors."""
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

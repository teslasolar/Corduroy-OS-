"""
Konomi Corduroy-OS - Protocol Module

56k dial-up inspired binary protocol for AI operations
"""

from .dialup import DialUpPacket, OpCode, DialUpSession
from .serializer import TensorSerializer, serialize_tensor, deserialize_tensor
from .server import DialUpServer

__all__ = [
    'DialUpPacket',
    'OpCode',
    'DialUpSession',
    'TensorSerializer',
    'serialize_tensor',
    'deserialize_tensor',
    'DialUpServer'
]

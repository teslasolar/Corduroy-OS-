"""
KONOMI Protocol - Dial-Up Module
56k modem-inspired binary protocol for AI operations

Re-exports main classes for backward compatibility.
"""

from .opcodes import OpCode
from .packet import DialUpPacket
from .session import DialUpSession
from .utils import create_handshake, create_ack, verify_packet

__all__ = [
    'OpCode',
    'DialUpPacket',
    'DialUpSession',
    'create_handshake',
    'create_ack',
    'verify_packet',
]

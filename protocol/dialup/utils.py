"""
KONOMI Protocol - Dial-Up Utilities
Convenience functions for packet creation
"""

from typing import Tuple, Optional
from .packet import DialUpPacket
from .session import DialUpSession


def create_handshake() -> bytes:
    """Create handshake packet (convenience function)."""
    session = DialUpSession()
    return session.handshake().pack()


def create_ack(seq: int) -> bytes:
    """Create ACK packet (convenience function)."""
    session = DialUpSession()
    return session.ack(seq).pack()


def verify_packet(data: bytes) -> Tuple[bool, Optional[DialUpPacket]]:
    """
    Verify and unpack packet.

    Args:
        data: Binary packet data

    Returns:
        Tuple of (is_valid, packet or None)
    """
    try:
        packet = DialUpPacket.unpack(data)
        return (True, packet)
    except Exception:
        return (False, None)

"""
KONOMI Protocol - Session
DialUpSession class for connection management
"""

import struct
from .opcodes import OpCode
from .packet import DialUpPacket


class DialUpSession:
    """
    Manages a dial-up protocol session.

    Handles:
    - Sequence number tracking
    - Handshake protocol
    - Keepalive
    """

    def __init__(self):
        """Initialize session."""
        self.seq = 0
        self.connected = False
        self.last_ack = 0

    def create_packet(
        self,
        opcode: OpCode,
        payload: bytes = b'',
        flags: int = 0
    ) -> DialUpPacket:
        """
        Create packet with auto-incrementing sequence number.

        Args:
            opcode: Operation code
            payload: Data payload
            flags: Control flags

        Returns:
            DialUpPacket instance
        """
        packet = DialUpPacket(opcode, payload, self.seq, flags)
        self.seq = (self.seq + 1) % 65536  # 16-bit wraparound
        return packet

    def handshake(self) -> DialUpPacket:
        """
        Create handshake packet.
        Like modem ATD (dial) command.
        """
        payload = b'KONOMI-CORDUROY-OS-V1'
        return self.create_packet(OpCode.HANDSHAKE, payload)

    def ack(self, seq: int) -> DialUpPacket:
        """Create acknowledgment packet."""
        self.last_ack = seq
        payload = struct.pack('!H', seq)
        return self.create_packet(OpCode.ACK, payload)

    def keepalive(self) -> DialUpPacket:
        """Create keepalive packet."""
        return self.create_packet(OpCode.KEEPALIVE)

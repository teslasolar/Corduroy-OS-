"""
Konomi Corduroy-OS - Dial-Up Protocol
56k modem-inspired binary protocol for AI operations

📞 Inspired by V.90 modems: chunked transfer, flow control, error detection
"""

import struct
from enum import IntEnum
from typing import Optional, Tuple


class OpCode(IntEnum):
    """
    Operation codes for protocol messages.

    Inspired by modem AT commands but in binary form.
    """
    # Connection control
    HANDSHAKE = 0x01      # Initial connection (like ATD)
    DISCONNECT = 0x02     # Terminate connection (like ATH)
    KEEPALIVE = 0x03      # Keep connection alive

    # Data operations
    TENSOR_OP = 0x10      # Tensor operation request
    LLM_QUERY = 0x11      # LLM query
    BLOCK_GET = 0x12      # Get BlockArray value
    BLOCK_SET = 0x13      # Set BlockArray value
    CUBE_MSG = 0x14       # Cube message passing
    CUBE_STATUS = 0x15    # Get Cube status

    # Flow control (like XON/XOFF)
    ACK = 0x0F            # Acknowledgment
    NAK = 0x10            # Negative acknowledgment
    RETRY = 0x11          # Request retry

    # Error codes
    ERROR_INVALID = 0xF0  # Invalid request
    ERROR_TIMEOUT = 0xF1  # Operation timeout
    ERROR_OVERLOAD = 0xF2 # System overload


class DialUpPacket:
    """
    56k-inspired packet format.

    Inspired by V.90 modem frames:
    - Maximum 4KB payload (like modem frame size)
    - CRC16 error detection (like modem error correction)
    - Sequence numbers (for packet ordering)
    - Minimal header overhead

    Packet structure:
    [Header: 8 bytes][Payload: 0-4096 bytes]

    Header format:
    - opcode (1 byte): Operation code
    - length (2 bytes): Payload length (big-endian)
    - seq (2 bytes): Sequence number (big-endian)
    - crc (2 bytes): CRC16 checksum (big-endian)
    - flags (1 byte): Control flags
    """

    HEADER_SIZE = 8
    MAX_PAYLOAD = 4096  # 4KB like V.90 frame

    # Flags
    FLAG_COMPRESSED = 0x01
    FLAG_ENCRYPTED = 0x02
    FLAG_FRAGMENTED = 0x04
    FLAG_URGENT = 0x08

    def __init__(
        self,
        opcode: OpCode,
        payload: bytes = b'',
        seq: int = 0,
        flags: int = 0
    ):
        """
        Initialize dial-up packet.

        Args:
            opcode: Operation code
            payload: Data payload (max 4KB)
            seq: Sequence number
            flags: Control flags

        Raises:
            ValueError: If payload exceeds maximum size
        """
        if len(payload) > self.MAX_PAYLOAD:
            raise ValueError(
                f"Payload too large: {len(payload)} > {self.MAX_PAYLOAD}"
            )

        self.opcode = opcode
        self.payload = payload
        self.seq = seq
        self.flags = flags

    def pack(self) -> bytes:
        """
        Pack packet into binary format.

        Returns:
            Binary packet data

        Format:
        [opcode:1][length:2][seq:2][crc:2][flags:1][payload:N]
        """
        payload_len = len(self.payload)
        crc = self._crc16(self.payload)

        # Pack header (big-endian for network byte order)
        header = struct.pack(
            '!BHHBB',
            self.opcode,
            payload_len,
            self.seq,
            crc,
            self.flags
        )

        return header + self.payload

    @classmethod
    def unpack(cls, data: bytes) -> 'DialUpPacket':
        """
        Unpack binary data into packet.

        Args:
            data: Binary packet data

        Returns:
            DialUpPacket instance

        Raises:
            ValueError: If packet is invalid or CRC fails
        """
        if len(data) < cls.HEADER_SIZE:
            raise ValueError(
                f"Packet too short: {len(data)} < {cls.HEADER_SIZE}"
            )

        # Unpack header
        opcode, payload_len, seq, crc, flags = struct.unpack(
            '!BHHBB',
            data[:cls.HEADER_SIZE]
        )

        # Extract payload
        payload_end = cls.HEADER_SIZE + payload_len
        if len(data) < payload_end:
            raise ValueError(
                f"Incomplete payload: expected {payload_len}, "
                f"got {len(data) - cls.HEADER_SIZE}"
            )

        payload = data[cls.HEADER_SIZE:payload_end]

        # Verify CRC
        computed_crc = cls._crc16(payload)
        if computed_crc != crc:
            raise ValueError(
                f"CRC mismatch: expected {crc:04x}, "
                f"got {computed_crc:04x}"
            )

        return cls(OpCode(opcode), payload, seq, flags)

    @staticmethod
    def _crc16(data: bytes) -> int:
        """
        Compute CRC16 checksum.

        Uses CRC16-ANSI (also known as CRC16-IBM).
        Polynomial: 0xA001 (reversed 0x8005)

        This is the same algorithm used in many modem protocols.

        Args:
            data: Input bytes

        Returns:
            16-bit CRC value
        """
        crc = 0xFFFF

        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 0x0001:
                    crc = (crc >> 1) ^ 0xA001
                else:
                    crc >>= 1

        return crc & 0xFFFF

    def is_valid(self) -> bool:
        """
        Validate packet integrity.

        Returns:
            True if packet is valid
        """
        try:
            # Check payload size
            if len(self.payload) > self.MAX_PAYLOAD:
                return False

            # Verify CRC
            computed_crc = self._crc16(self.payload)
            packed = self.pack()
            unpacked = self.unpack(packed)

            return unpacked.opcode == self.opcode

        except Exception:
            return False

    def __repr__(self):
        return (
            f"DialUpPacket(opcode={self.opcode.name}, "
            f"len={len(self.payload)}, "
            f"seq={self.seq}, "
            f"flags={self.flags:02x})"
        )


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

        Returns:
            Handshake packet
        """
        payload = b'KONOMI-CORDUROY-OS-V1'
        return self.create_packet(OpCode.HANDSHAKE, payload)

    def ack(self, seq: int) -> DialUpPacket:
        """
        Create acknowledgment packet.

        Args:
            seq: Sequence number being acknowledged

        Returns:
            ACK packet
        """
        self.last_ack = seq
        payload = struct.pack('!H', seq)
        return self.create_packet(OpCode.ACK, payload)

    def keepalive(self) -> DialUpPacket:
        """
        Create keepalive packet.

        Prevents connection timeout.

        Returns:
            Keepalive packet
        """
        return self.create_packet(OpCode.KEEPALIVE)


# Utility functions

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

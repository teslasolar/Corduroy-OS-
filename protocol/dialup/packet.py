"""
KONOMI Protocol - Packet
DialUpPacket class for binary packet format
"""

import struct
from .opcodes import OpCode
from .checksum import crc16


class DialUpPacket:
    """
    56k-inspired packet format.

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
        """Initialize dial-up packet."""
        if len(payload) > self.MAX_PAYLOAD:
            raise ValueError(
                f"Payload too large: {len(payload)} > {self.MAX_PAYLOAD}"
            )
        self.opcode = opcode
        self.payload = payload
        self.seq = seq
        self.flags = flags

    def pack(self) -> bytes:
        """Pack packet into binary format."""
        payload_len = len(self.payload)
        checksum = crc16(self.payload)

        header = struct.pack(
            '!BHHHB',
            self.opcode,
            payload_len,
            self.seq,
            checksum,
            self.flags
        )
        return header + self.payload

    @classmethod
    def unpack(cls, data: bytes) -> 'DialUpPacket':
        """Unpack binary data into packet."""
        if len(data) < cls.HEADER_SIZE:
            raise ValueError(
                f"Packet too short: {len(data)} < {cls.HEADER_SIZE}"
            )

        opcode, payload_len, seq, checksum, flags = struct.unpack(
            '!BHHHB',
            data[:cls.HEADER_SIZE]
        )

        payload_end = cls.HEADER_SIZE + payload_len
        if len(data) < payload_end:
            raise ValueError(
                f"Incomplete payload: expected {payload_len}, "
                f"got {len(data) - cls.HEADER_SIZE}"
            )

        payload = data[cls.HEADER_SIZE:payload_end]

        # Verify CRC
        computed_crc = crc16(payload)
        if computed_crc != checksum:
            raise ValueError(
                f"CRC mismatch: expected {checksum:04x}, "
                f"got {computed_crc:04x}"
            )

        return cls(OpCode(opcode), payload, seq, flags)

    def is_valid(self) -> bool:
        """Validate packet integrity."""
        try:
            if len(self.payload) > self.MAX_PAYLOAD:
                return False
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

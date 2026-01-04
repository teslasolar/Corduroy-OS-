"""
KONOMI UDT Layer 1: Protocol Types
Type definitions for 56k dial-up protocol
"""

from typing import NamedTuple, Optional
from enum import IntEnum


class DTypeCode(IntEnum):
    """Data type encoding for tensor serialization."""
    FLOAT32 = 0x01
    FLOAT64 = 0x02
    INT32 = 0x03
    INT64 = 0x04
    UINT32 = 0x05
    UINT64 = 0x06
    BOOL = 0x07
    FLOAT16 = 0x08


class PacketHeader(NamedTuple):
    """Header structure for dial-up packets."""
    opcode: int
    payload_len: int
    seq: int
    crc: int
    flags: int

    HEADER_SIZE: int = 8
    MAX_PAYLOAD: int = 4096

    # Flag constants
    FLAG_COMPRESSED: int = 0x01
    FLAG_ENCRYPTED: int = 0x02
    FLAG_FRAGMENTED: int = 0x04
    FLAG_URGENT: int = 0x08


class SessionState(NamedTuple):
    """State of a dial-up session."""
    seq: int
    connected: bool
    last_ack: int

    @classmethod
    def new(cls) -> 'SessionState':
        """Create new session state."""
        return cls(seq=0, connected=False, last_ack=0)

    def next_seq(self) -> 'SessionState':
        """Return state with incremented sequence."""
        return self._replace(seq=(self.seq + 1) % 65536)

    def set_connected(self) -> 'SessionState':
        """Return state marked as connected."""
        return self._replace(connected=True)

    def set_ack(self, ack: int) -> 'SessionState':
        """Return state with updated last ack."""
        return self._replace(last_ack=ack)


class ServerPorts(NamedTuple):
    """Port configuration for the dial-up server."""
    api: int = 3001
    stream: int = 3002
    mesh: int = 6789

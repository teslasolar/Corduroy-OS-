"""
KONOMI Protocol - Checksum
CRC16 checksum for packet validation
"""


def crc16(data: bytes) -> int:
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

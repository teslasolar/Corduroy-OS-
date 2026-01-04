"""
KONOMI Protocol - OpCodes
Operation codes for dial-up protocol messages
"""

from enum import IntEnum


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

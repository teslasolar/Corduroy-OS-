# UDT: OpCode

## Description
Operation codes for dial-up protocol messages

## Connection Control
| Code | Value | Description |
|------|-------|-------------|
| HANDSHAKE | 0x01 | Initial connection |
| DISCONNECT | 0x02 | Terminate connection |
| KEEPALIVE | 0x03 | Keep connection alive |

## Data Operations
| Code | Value | Description |
|------|-------|-------------|
| TENSOR_OP | 0x10 | Tensor operation request |
| LLM_QUERY | 0x11 | LLM query |
| BLOCK_GET | 0x12 | Get BlockArray value |
| BLOCK_SET | 0x13 | Set BlockArray value |
| CUBE_MSG | 0x14 | Cube message passing |
| CUBE_STATUS | 0x15 | Get Cube status |

## Flow Control
| Code | Value | Description |
|------|-------|-------------|
| ACK | 0x0F | Acknowledgment |
| NAK | 0x10 | Negative acknowledgment |
| RETRY | 0x11 | Request retry |

## Error Codes
| Code | Value | Description |
|------|-------|-------------|
| ERROR_INVALID | 0xF0 | Invalid request |
| ERROR_TIMEOUT | 0xF1 | Operation timeout |
| ERROR_OVERLOAD | 0xF2 | System overload |

## ISA-95 Layer
Layer 1: Protocol Types

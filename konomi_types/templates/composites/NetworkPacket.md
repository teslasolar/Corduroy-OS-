# NetworkPacket UDT Template

ISA-95 Layer 2 Composite Type

## Overview
Complete network packet with header, session context,
and data type information. Bundles all protocol state
needed for tensor transmission over 56k dial-up.

## Nested Types
| Field | Type | Layer | Description |
|-------|------|-------|-------------|
| header | PacketHeader | L1 | Binary packet header |
| session | SessionState | L1 | Connection state |
| dtype | DTypeCode | L1 | Payload data type |

## Structure
```
NetworkPacket
├── header: PacketHeader
│   ├── opcode: int
│   ├── payload_len: int
│   ├── seq: int
│   ├── crc: int
│   └── flags: int
├── session: SessionState
│   ├── seq: int
│   ├── connected: bool
│   └── last_ack: int
└── dtype: DTypeCode
    └── (enum value: FLOAT32, FLOAT64, etc.)
```

## Use Cases
- Packet inspection/debugging
- Protocol logging
- Transmission auditing

## Instance Format
```
NetworkPacket:
  header:
    opcode: TENSOR_OP (0x10)
    payload_len: 256
    seq: 42
    crc: 0xA3F1
    flags: 0x00
  session:
    seq: 42
    connected: true
    last_ack: 41
  dtype: FLOAT32 (0x01)
```

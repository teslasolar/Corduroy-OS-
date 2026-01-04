# ProtocolSession UDT Template

ISA-95 Layer 2 Composite Type

## Overview
Complete protocol session state bundling connection
state, last packet header, and server port config.
Full context for dial-up protocol operations.

## Nested Types
| Field | Type | Layer | Description |
|-------|------|-------|-------------|
| state | SessionState | L1 | Connection state |
| last_header | PacketHeader? | L1 | Last received header |
| ports | ServerPorts | L1 | Port configuration |

## Structure
```
ProtocolSession
├── state: SessionState
│   ├── seq: int
│   ├── connected: bool
│   └── last_ack: int
├── last_header: Optional[PacketHeader]
│   ├── opcode: int
│   ├── payload_len: int
│   ├── seq: int
│   ├── crc: int
│   └── flags: int
└── ports: ServerPorts
    ├── api: int (default: 3001)
    ├── stream: int (default: 3002)
    └── mesh: int (default: 6789)
```

## Use Cases
- Session debugging
- Connection auditing
- Protocol state snapshots

## Instance Format
```
ProtocolSession:
  state:
    seq: 100
    connected: true
    last_ack: 99
  last_header:
    opcode: ACK (0x02)
    payload_len: 0
    seq: 99
    crc: 0x0000
    flags: 0x00
  ports:
    api: 3001
    stream: 3002
    mesh: 6789
```

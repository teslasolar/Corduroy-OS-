# UDT: PacketHeader

## Description
Header structure for 56k dial-up protocol packets

## Fields
| Name | Type | Size | Description |
|------|------|------|-------------|
| opcode | uint8 | 1 byte | Operation code |
| payload_len | uint16 | 2 bytes | Payload length |
| seq | uint16 | 2 bytes | Sequence number |
| crc | uint16 | 2 bytes | CRC16 checksum |
| flags | uint8 | 1 byte | Control flags |

## Constants
- HEADER_SIZE = 8 bytes
- MAX_PAYLOAD = 4096 bytes

## Flags
| Flag | Value | Description |
|------|-------|-------------|
| FLAG_COMPRESSED | 0x01 | Payload is compressed |
| FLAG_ENCRYPTED | 0x02 | Payload is encrypted |
| FLAG_FRAGMENTED | 0x04 | Packet is fragmented |
| FLAG_URGENT | 0x08 | High priority |

## ISA-95 Layer
Layer 1: Protocol Types

# CubeGraph UDT Template

ISA-95 Layer 2 Composite Type

## Overview
Cube status with full message history. Captures the
complete state of a 9-node LLM graph including all
inter-node communication records.

## Nested Types
| Field | Type | Layer | Description |
|-------|------|-------|-------------|
| status | CubeStatus | L1 | Node and edge counts |
| messages | List[MessageRecord] | L1 | Communication history |

## Structure
```
CubeGraph
├── status: CubeStatus
│   ├── vertices: int (always 8)
│   ├── central: int (always 1)
│   ├── total_nodes: int
│   ├── total_edges: int
│   ├── messages_sent: int
│   ├── active_llms: int
│   └── edge_list: Dict[str, List[str]]
└── messages: List[MessageRecord]
    └── [i]: MessageRecord
        ├── from_vertex: str
        ├── to_vertex: str
        ├── message: str
        └── response: str
```

## Use Cases
- Debugging LLM communication
- Audit trails
- Message replay

## Instance Format
```
CubeGraph:
  status:
    vertices: 8
    central: 1
    total_nodes: 9
    total_edges: 20
    messages_sent: 3
    active_llms: 9
  messages:
    - from: NEU, to: central, msg: "query", resp: "[query]"
    - from: central, to: SWD, msg: "relay", resp: "[relay]"
    - from: NED, to: SEU, msg: "sync", resp: "[sync]"
```

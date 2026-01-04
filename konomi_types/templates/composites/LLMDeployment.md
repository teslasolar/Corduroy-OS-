# LLMDeployment UDT Template

ISA-95 Layer 2 Composite Type

## Overview
Deployment record for a Femto LLM instance within the
spatial grid. Tracks location, host block, optional
cube membership, and resource usage.

## Nested Types
| Field | Type | Layer | Description |
|-------|------|-------|-------------|
| location | Coord3D | L1 | Grid coordinates |
| host_block | BlockMetadata | L1 | Parent block info |
| host_cube | CubeStatus? | L1 | Optional cube membership |
| memory | MemoryEstimate | L1 | Resource footprint |

## Structure
```
LLMDeployment
├── location: Coord3D
│   └── (x, y, z): Tuple[int, int, int]
├── host_block: BlockMetadata
│   ├── dims: Dim3D
│   ├── capacity: int
│   ├── active_cells: int
│   ├── active_llms: int
│   └── active_cubes: int
├── host_cube: Optional[CubeStatus]
│   ├── vertices: int
│   ├── central: int
│   ├── total_nodes: int
│   ├── total_edges: int
│   ├── messages_sent: int
│   ├── active_llms: int
│   └── edge_list: Dict
└── memory: MemoryEstimate
    ├── cells_mb: float
    ├── llms_mb: float
    ├── cubes_mb: float
    └── total_mb: float
```

## Use Cases
- LLM inventory tracking
- Resource planning
- Cube membership audits

## Instance Format
```
LLMDeployment:
  location: (50, 50, 50)
  host_block:
    dims: (100, 100, 100)
    capacity: 1000000
    active_cells: 100
    active_llms: 10
    active_cubes: 1
  host_cube:
    vertices: 8
    total_nodes: 9
    total_edges: 20
  memory:
    cells_mb: 0.0
    llms_mb: 0.001
    cubes_mb: 0.0
    total_mb: 0.001
```

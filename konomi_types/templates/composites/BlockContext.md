# BlockContext UDT Template

ISA-95 Layer 2 Composite Type

## Overview
Full block state combining metadata, bounds, and memory usage.
Nests 3 Layer 1 types into a single context object.

## Nested Types
| Field | Type | Layer | Description |
|-------|------|-------|-------------|
| metadata | BlockMetadata | L1 | Dimensions and counts |
| bounds | RegionBounds | L1 | Active region bounds |
| memory | MemoryEstimate | L1 | Memory breakdown |

## Structure
```
BlockContext
├── metadata: BlockMetadata
│   ├── dims: Dim3D
│   ├── capacity: int
│   ├── active_cells: int
│   ├── active_llms: int
│   └── active_cubes: int
├── bounds: RegionBounds
│   ├── x_min, x_max: int
│   ├── y_min, y_max: int
│   └── z_min, z_max: int
└── memory: MemoryEstimate
    ├── cells_mb: float
    ├── llms_mb: float
    ├── cubes_mb: float
    └── total_mb: float
```

## Use Cases
- Block status reporting
- Resource allocation planning
- Spatial indexing

## Instance Format
```
BlockContext:
  metadata:
    dims: (100, 100, 100)
    capacity: 1000000
    active_cells: 1500
    active_llms: 25
    active_cubes: 3
  bounds:
    x: [0, 99]
    y: [10, 80]
    z: [5, 95]
  memory:
    cells_mb: 0.143
    llms_mb: 0.024
    cubes_mb: 0.026
    total_mb: 0.193
```

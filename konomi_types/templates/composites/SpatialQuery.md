# SpatialQuery UDT Template

ISA-95 Layer 2 Composite Type

## Overview
Spatial query combining origin point, search range,
and optional result metadata. Used for BlockArray
region queries and LLM lookups.

## Nested Types
| Field | Type | Layer | Description |
|-------|------|-------|-------------|
| origin | Coord3D | L1 | Query origin point |
| range | Range3D | L1 | Search bounds |
| result | BlockMetadata? | L1 | Optional result data |

## Structure
```
SpatialQuery
├── origin: Coord3D
│   └── (x, y, z): Tuple[int, int, int]
├── range: Range3D
│   ├── x_range: Tuple[int, int]
│   ├── y_range: Tuple[int, int]
│   └── z_range: Tuple[int, int]
└── result: Optional[BlockMetadata]
    ├── dims: Dim3D
    ├── capacity: int
    ├── active_cells: int
    ├── active_llms: int
    └── active_cubes: int
```

## Use Cases
- Region-based cell queries
- LLM broadcast targeting
- Cube placement searches

## Instance Format
```
SpatialQuery:
  origin: (50, 50, 50)
  range:
    x: [0, 100]
    y: [0, 100]
    z: [0, 100]
  result:
    dims: (100, 100, 100)
    capacity: 1000000
    active_cells: 42
    active_llms: 5
    active_cubes: 1
```

# UDT: BlockMetadata

## Description
Metadata for a BlockArray instance

## Fields
| Name | Type | Description |
|------|------|-------------|
| dims | Dim3D | Grid dimensions (x, y, z) |
| capacity | int | Total cell capacity (x * y * z) |
| active_cells | int | Count of non-zero cells |
| active_llms | int | Count of Femto LLM instances |
| active_cubes | int | Count of nested Cube instances |

## Computed
- `density` = active_cells / capacity

## ISA-95 Layer
Layer 1: Block Types

## Related
- Coord3D
- Dim3D
- MemoryEstimate

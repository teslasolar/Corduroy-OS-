# UDT: Coord3D

## Description
3D coordinate for spatial addressing in BlockArrays

## Fields
| Name | Type | Description |
|------|------|-------------|
| x | int | X-axis position (0 to dims[0]-1) |
| y | int | Y-axis position (0 to dims[1]-1) |
| z | int | Z-axis position (0 to dims[2]-1) |

## Validation
- All values must be non-negative integers
- Must be within BlockArray bounds

## Example
```
x: 10
y: 20
z: 30
```

## ISA-95 Layer
Layer 1: Base Primitives

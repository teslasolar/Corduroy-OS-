# UDT: Range3D

## Description
3D region bounds for spatial queries

## Fields
| Name | Type | Description |
|------|------|-------------|
| x_min | int | Minimum X (inclusive) |
| x_max | int | Maximum X (inclusive) |
| y_min | int | Minimum Y (inclusive) |
| y_max | int | Maximum Y (inclusive) |
| z_min | int | Minimum Z (inclusive) |
| z_max | int | Maximum Z (inclusive) |

## Methods
- `contains(x, y, z)` → bool

## Validation
- min ≤ max for all axes
- All values non-negative

## Example
```
x_min: 0
x_max: 100
y_min: 0
y_max: 100
z_min: 0
z_max: 100
```

## ISA-95 Layer
Layer 1: Base Primitives

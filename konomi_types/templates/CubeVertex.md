# UDT: CubeVertex

## Description
Vertex identifier for Cube graph nodes

## Valid Values
| ID | Position |
|----|----------|
| NEU | North-East-Up |
| NED | North-East-Down |
| NWU | North-West-Up |
| NWD | North-West-Down |
| SEU | South-East-Up |
| SED | South-East-Down |
| SWU | South-West-Up |
| SWD | South-West-Down |
| central | Hub node (connected to all) |

## Adjacency Rule
Two vertices are adjacent if they differ in exactly 1 axis character.

## Example
```
vertex: NEU
adjacent: [NED, NWU, SEU, central]
```

## ISA-95 Layer
Layer 1: Cube Types

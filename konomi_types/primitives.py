"""
KONOMI UDT Layer 1: Primitive Types
Base type definitions for the entire system
"""

from typing import Tuple, NamedTuple, Dict, Any


# Coordinate type aliases
Coord3D = Tuple[int, int, int]
Dim3D = Tuple[int, int, int]


class Range3D(NamedTuple):
    """3D region bounds for spatial queries."""
    x_range: Tuple[int, int]
    y_range: Tuple[int, int]
    z_range: Tuple[int, int]

    def contains(self, x: int, y: int, z: int) -> bool:
        """Check if coordinate is within range."""
        return (
            self.x_range[0] <= x <= self.x_range[1] and
            self.y_range[0] <= y <= self.y_range[1] and
            self.z_range[0] <= z <= self.z_range[1]
        )


class MemoryEstimate(NamedTuple):
    """Memory usage breakdown in MB."""
    cells_mb: float
    llms_mb: float
    cubes_mb: float
    total_mb: float

    @classmethod
    def from_counts(
        cls,
        cell_count: int,
        llm_count: int,
        cube_count: int
    ) -> 'MemoryEstimate':
        """Calculate memory from object counts."""
        # Each dict entry ~100 bytes
        cells_mb = (cell_count * 100) / (1024 * 1024)
        # Each Femto ~1KB (16x16 float32 + overhead)
        llms_mb = (llm_count * 1024) / (1024 * 1024)
        # Each Cube: 9 Femtos ~9KB + overhead
        cubes_mb = (cube_count * 9 * 1024) / (1024 * 1024)
        return cls(
            cells_mb=cells_mb,
            llms_mb=llms_mb,
            cubes_mb=cubes_mb,
            total_mb=cells_mb + llms_mb + cubes_mb
        )

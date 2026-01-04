"""
KONOMI UDT Layer 1: Block Types
Type definitions for BlockArray operations
"""

from typing import Tuple, Dict, Any, NamedTuple, Optional
from .primitives import Dim3D


class BlockMetadata(NamedTuple):
    """Metadata for a BlockArray instance."""
    dims: Dim3D
    capacity: int
    active_cells: int
    active_llms: int
    active_cubes: int

    @classmethod
    def create(cls, dims: Dim3D) -> 'BlockMetadata':
        """Create initial metadata for new block."""
        return cls(
            dims=dims,
            capacity=dims[0] * dims[1] * dims[2],
            active_cells=0,
            active_llms=0,
            active_cubes=0
        )

    def with_cell_delta(self, delta: int) -> 'BlockMetadata':
        """Return new metadata with cell count changed."""
        return self._replace(active_cells=self.active_cells + delta)

    def with_llm_delta(self, delta: int) -> 'BlockMetadata':
        """Return new metadata with LLM count changed."""
        return self._replace(active_llms=self.active_llms + delta)

    def with_cube_delta(self, delta: int) -> 'BlockMetadata':
        """Return new metadata with cube count changed."""
        return self._replace(active_cubes=self.active_cubes + delta)


class BlockQuery(NamedTuple):
    """Query parameters for block operations."""
    block_name: str
    x: int
    y: int
    z: int
    value: Optional[float] = None
    text: Optional[str] = None


class RegionBounds(NamedTuple):
    """Bounds for region-based operations."""
    x_min: int
    x_max: int
    y_min: int
    y_max: int
    z_min: int
    z_max: int

    def contains(self, x: int, y: int, z: int) -> bool:
        """Check if coordinate is within bounds."""
        return (
            self.x_min <= x <= self.x_max and
            self.y_min <= y <= self.y_max and
            self.z_min <= z <= self.z_max
        )

    @classmethod
    def from_tuples(
        cls,
        x_range: Tuple[int, int],
        y_range: Tuple[int, int],
        z_range: Tuple[int, int]
    ) -> 'RegionBounds':
        """Create from tuple ranges."""
        return cls(
            x_min=x_range[0], x_max=x_range[1],
            y_min=y_range[0], y_max=y_range[1],
            z_min=z_range[0], z_max=z_range[1]
        )

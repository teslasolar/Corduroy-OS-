"""
KONOMI Core - Block Core
Block class with basic operations: __init__, set, get, at
"""

from typing import Dict, Tuple, TYPE_CHECKING
from ..femto import Femto

if TYPE_CHECKING:
    from ..cube import Cube


class Block:
    """
    Sparse 3D grid (BlockArray) with LLM nodes at coordinates.

    Each coordinate (x, y, z) can store:
    1. A scalar value (in the sparse array `a`)
    2. A Femto LLM instance (in the LLM map `L`)
    3. A Cube (9-node graph) at the coordinate

    Targets:
    - 1 billion sparse cells capacity
    - <100MB memory for active regions
    - O(1) coordinate access
    """

    def __init__(self, dims: Tuple[int, int, int] = (1000, 1000, 1000)):
        """Initialize BlockArray with dimensions."""
        self.dims = dims
        self.a: Dict[Tuple[int, int, int], float] = {}
        self.L: Dict[Tuple[int, int, int], Femto] = {}
        self.C: Dict[Tuple[int, int, int], 'Cube'] = {}
        self.metadata = {
            'dims': dims,
            'capacity': dims[0] * dims[1] * dims[2],
            'active_cells': 0,
            'active_llms': 0,
            'active_cubes': 0
        }

    def _in_bounds(self, x: int, y: int, z: int) -> bool:
        """Check if coordinate is within grid bounds."""
        return (
            0 <= x < self.dims[0] and
            0 <= y < self.dims[1] and
            0 <= z < self.dims[2]
        )

    def set(self, x: int, y: int, z: int, value: float) -> None:
        """Set scalar value at coordinate."""
        if not self._in_bounds(x, y, z):
            raise ValueError(
                f"Coordinate ({x}, {y}, {z}) out of bounds {self.dims}"
            )
        if value != 0:
            was_empty = (x, y, z) not in self.a
            self.a[(x, y, z)] = value
            if was_empty:
                self.metadata['active_cells'] += 1
        else:
            if (x, y, z) in self.a:
                del self.a[(x, y, z)]
                self.metadata['active_cells'] -= 1

    def get(self, x: int, y: int, z: int, default: float = 0.0) -> float:
        """Get scalar value at coordinate."""
        return self.a.get((x, y, z), default)

    def at(self, x: int, y: int, z: int) -> Femto:
        """Get or create Femto LLM at coordinate."""
        if not self._in_bounds(x, y, z):
            raise ValueError(
                f"Coordinate ({x}, {y}, {z}) out of bounds {self.dims}"
            )
        if (x, y, z) not in self.L:
            seed = hash((x, y, z)) % (2**31)
            self.L[(x, y, z)] = Femto(seed=seed)
            self.metadata['active_llms'] += 1
        return self.L[(x, y, z)]

    def face_ops(self) -> int:
        """Get number of active cells."""
        return len(self.a)

    def __repr__(self):
        return (
            f"Block(dims={self.dims}, "
            f"active_cells={len(self.a)}, "
            f"active_llms={len(self.L)})"
        )

    # Import methods from other modules
    from .cube_nest import cube_at, broadcast_to_cubes
    from .regions import clear_region, query_region, broadcast_to_llms
    from .analytics import density, memory_estimate, info

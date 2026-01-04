"""
Konomi Corduroy-OS - Block Module
Sparse 3D grid with embedded LLM nodes and nested Cubes

🧊 BlockArray: Sparse voxel storage with Femto LLMs and Cube nodes at coordinates
"""

import jax.numpy as jnp
from typing import Tuple, Optional, Dict, Any, TYPE_CHECKING
from .femto import Femto

if TYPE_CHECKING:
    from .cube import Cube


class Block:
    """
    Sparse 3D grid (BlockArray) with LLM nodes at coordinates.

    Each coordinate (x, y, z) can store:
    1. A scalar value (in the sparse array `a`)
    2. A Femto LLM instance (in the LLM map `L`)

    This allows the grid to act as both a spatial data structure
    and a distributed AI compute fabric.

    Targets:
    - 1 billion sparse cells capacity
    - <100MB memory for active regions
    - O(1) coordinate access
    - Face operations on 1M cells

    Examples:
        >>> block = Block(dims=(1000, 1000, 1000))
        >>> block.set(0, 0, 0, 1.5)
        >>> block.get(0, 0, 0)
        1.5
        >>> llm = block.at(0, 0, 0)
        >>> llm.p("Hello")
        '[Hello]'
    """

    def __init__(self, dims: Tuple[int, int, int] = (1000, 1000, 1000)):
        """
        Initialize BlockArray.

        Args:
            dims: Dimensions (x, y, z) of the 3D grid
        """
        self.dims = dims

        # Sparse storage: only store non-zero values
        # Dict is memory-efficient for sparse data
        self.a: Dict[Tuple[int, int, int], float] = {}

        # LLM storage: Femto instances at coordinates
        self.L: Dict[Tuple[int, int, int], Femto] = {}

        # Cube storage: Cube instances nested at coordinates
        self.C: Dict[Tuple[int, int, int], 'Cube'] = {}

        # Metadata
        self.metadata = {
            'dims': dims,
            'capacity': dims[0] * dims[1] * dims[2],
            'active_cells': 0,
            'active_llms': 0,
            'active_cubes': 0
        }

    def set(self, x: int, y: int, z: int, value: float) -> None:
        """
        Set scalar value at coordinate.

        Args:
            x, y, z: Coordinate
            value: Scalar value to store

        Raises:
            ValueError: If coordinate is out of bounds
        """
        if not self._in_bounds(x, y, z):
            raise ValueError(
                f"Coordinate ({x}, {y}, {z}) out of bounds {self.dims}"
            )

        # Store in sparse dict
        if value != 0:
            was_empty = (x, y, z) not in self.a
            self.a[(x, y, z)] = value
            if was_empty:
                self.metadata['active_cells'] += 1
        else:
            # Remove zero values to save memory
            if (x, y, z) in self.a:
                del self.a[(x, y, z)]
                self.metadata['active_cells'] -= 1

    def get(self, x: int, y: int, z: int, default: float = 0.0) -> float:
        """
        Get scalar value at coordinate.

        Args:
            x, y, z: Coordinate
            default: Default value if coordinate is empty

        Returns:
            Scalar value at coordinate
        """
        return self.a.get((x, y, z), default)

    def at(self, x: int, y: int, z: int) -> Femto:
        """
        Get or create Femto LLM at coordinate.

        Each coordinate can have its own LLM instance. This allows
        distributed AI processing across the 3D grid.

        Args:
            x, y, z: Coordinate

        Returns:
            Femto LLM instance at this coordinate

        Examples:
            >>> block = Block()
            >>> llm = block.at(5, 10, 15)
            >>> result = llm.p("Process this data")
        """
        if not self._in_bounds(x, y, z):
            raise ValueError(
                f"Coordinate ({x}, {y}, {z}) out of bounds {self.dims}"
            )

        # Create LLM if it doesn't exist
        if (x, y, z) not in self.L:
            # Use coordinate as seed for reproducible LLM
            seed = hash((x, y, z)) % (2**31)
            self.L[(x, y, z)] = Femto(seed=seed)
            self.metadata['active_llms'] += 1

        return self.L[(x, y, z)]

    def cube_at(self, x: int, y: int, z: int) -> 'Cube':
        """
        Get or create Cube (9-node LLM graph) at coordinate.

        This allows you to nest Cubes in the 3D BlockArray, creating
        a hierarchy of distributed AI agents with lots of space between
        them for data storage.

        Args:
            x, y, z: Coordinate

        Returns:
            Cube instance at this coordinate

        Examples:
            >>> block = Block(dims=(100, 100, 100))
            >>> cube1 = block.cube_at(10, 10, 10)
            >>> cube2 = block.cube_at(50, 50, 50)
            >>> # Now you have 2 Cubes (18 LLMs total) in a 100³ grid
        """
        # Import here to avoid circular dependency
        from .cube import Cube

        if not self._in_bounds(x, y, z):
            raise ValueError(
                f"Coordinate ({x}, {y}, {z}) out of bounds {self.dims}"
            )

        # Create Cube if it doesn't exist
        if (x, y, z) not in self.C:
            self.C[(x, y, z)] = Cube()
            self.metadata['active_cubes'] += 1

        return self.C[(x, y, z)]

    def _in_bounds(self, x: int, y: int, z: int) -> bool:
        """Check if coordinate is within grid bounds."""
        return (
            0 <= x < self.dims[0] and
            0 <= y < self.dims[1] and
            0 <= z < self.dims[2]
        )

    def face_ops(self) -> int:
        """
        Get number of active cells (face operations).

        In 3D graphics, "face operations" typically refers to
        operations on surface faces. Here, it returns the count
        of active (non-zero) cells.

        Returns:
            Number of active cells

        Target: Handle 1M+ cells efficiently
        """
        return len(self.a)

    def clear_region(
        self,
        x_range: Tuple[int, int],
        y_range: Tuple[int, int],
        z_range: Tuple[int, int]
    ) -> int:
        """
        Clear all values in a region.

        Args:
            x_range: (min_x, max_x) inclusive
            y_range: (min_y, max_y) inclusive
            z_range: (min_z, max_z) inclusive

        Returns:
            Number of cells cleared
        """
        cleared = 0
        coords_to_remove = []

        for (x, y, z) in self.a.keys():
            if (x_range[0] <= x <= x_range[1] and
                y_range[0] <= y <= y_range[1] and
                z_range[0] <= z <= z_range[1]):
                coords_to_remove.append((x, y, z))

        for coord in coords_to_remove:
            del self.a[coord]
            cleared += 1

        self.metadata['active_cells'] -= cleared
        return cleared

    def query_region(
        self,
        x_range: Tuple[int, int],
        y_range: Tuple[int, int],
        z_range: Tuple[int, int]
    ) -> Dict[Tuple[int, int, int], float]:
        """
        Query all values in a region.

        Args:
            x_range: (min_x, max_x) inclusive
            y_range: (min_y, max_y) inclusive
            z_range: (min_z, max_z) inclusive

        Returns:
            Dict of {(x, y, z): value} for region
        """
        result = {}

        for (x, y, z), value in self.a.items():
            if (x_range[0] <= x <= x_range[1] and
                y_range[0] <= y <= y_range[1] and
                z_range[0] <= z <= z_range[1]):
                result[(x, y, z)] = value

        return result

    def broadcast_to_llms(
        self,
        message: str,
        x_range: Optional[Tuple[int, int]] = None,
        y_range: Optional[Tuple[int, int]] = None,
        z_range: Optional[Tuple[int, int]] = None
    ) -> Dict[Tuple[int, int, int], str]:
        """
        Broadcast message to all LLMs in a region.

        Args:
            message: Message to send to LLMs
            x_range, y_range, z_range: Region bounds (None = all)

        Returns:
            Dict of {(x, y, z): response} from each LLM
        """
        results = {}

        # Default to all LLMs if no range specified
        if x_range is None:
            x_range = (0, self.dims[0] - 1)
        if y_range is None:
            y_range = (0, self.dims[1] - 1)
        if z_range is None:
            z_range = (0, self.dims[2] - 1)

        for (x, y, z), llm in self.L.items():
            if (x_range[0] <= x <= x_range[1] and
                y_range[0] <= y <= y_range[1] and
                z_range[0] <= z <= z_range[1]):
                results[(x, y, z)] = llm.p(message)

        return results

    def density(self) -> float:
        """
        Calculate grid density (active cells / capacity).

        Returns:
            Density ratio between 0 and 1
        """
        return len(self.a) / self.metadata['capacity']

    def broadcast_to_cubes(
        self,
        message: str,
        vertex: str = 'central',
        x_range: Optional[Tuple[int, int]] = None,
        y_range: Optional[Tuple[int, int]] = None,
        z_range: Optional[Tuple[int, int]] = None
    ) -> Dict[Tuple[int, int, int], str]:
        """
        Broadcast message to all Cubes in a region.

        Args:
            message: Message to send
            vertex: Which vertex of each Cube to send to
            x_range, y_range, z_range: Region bounds (None = all)

        Returns:
            Dict of {(x, y, z): response} from each Cube
        """
        results = {}

        # Default to all Cubes if no range specified
        if x_range is None:
            x_range = (0, self.dims[0] - 1)
        if y_range is None:
            y_range = (0, self.dims[1] - 1)
        if z_range is None:
            z_range = (0, self.dims[2] - 1)

        for (x, y, z), cube in self.C.items():
            if (x_range[0] <= x <= x_range[1] and
                y_range[0] <= y <= y_range[1] and
                z_range[0] <= z <= z_range[1]):
                results[(x, y, z)] = cube.process_at_vertex(vertex, message)

        return results

    def memory_estimate(self) -> Dict[str, float]:
        """
        Estimate memory usage.

        Returns:
            Dict with memory breakdown in MB
        """
        # Each dict entry ~100 bytes (key + value + overhead)
        cells_mb = (len(self.a) * 100) / (1024 * 1024)

        # Each Femto ~1KB (16x16 float32 weights + overhead)
        llms_mb = (len(self.L) * 1024) / (1024 * 1024)

        # Each Cube: 9 Femtos ~9KB + overhead
        cubes_mb = (len(self.C) * 9 * 1024) / (1024 * 1024)

        return {
            'cells_mb': cells_mb,
            'llms_mb': llms_mb,
            'cubes_mb': cubes_mb,
            'total_mb': cells_mb + llms_mb + cubes_mb
        }

    def info(self) -> Dict[str, Any]:
        """
        Get BlockArray information.

        Returns:
            Dict with metadata and stats
        """
        mem = self.memory_estimate()

        return {
            **self.metadata,
            'density': self.density(),
            'memory_mb': mem['total_mb'],
            'memory_breakdown': mem
        }

    def __repr__(self):
        return (
            f"Block(dims={self.dims}, "
            f"active_cells={len(self.a)}, "
            f"active_llms={len(self.L)})"
        )

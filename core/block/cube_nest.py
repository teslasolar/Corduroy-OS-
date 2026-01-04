"""
KONOMI Core - Block Cube Nesting
Methods for nesting Cubes inside BlockArrays
"""

from typing import Dict, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Block
    from ..cube import Cube


def cube_at(self: 'Block', x: int, y: int, z: int) -> 'Cube':
    """
    Get or create Cube (9-node LLM graph) at coordinate.

    This allows nesting Cubes in the 3D BlockArray, creating
    a hierarchy of distributed AI agents with space between
    them for data storage.
    """
    from ..cube import Cube as CubeClass

    if not self._in_bounds(x, y, z):
        raise ValueError(
            f"Coordinate ({x}, {y}, {z}) out of bounds {self.dims}"
        )
    if (x, y, z) not in self.C:
        self.C[(x, y, z)] = CubeClass()
        self.metadata['active_cubes'] += 1
    return self.C[(x, y, z)]


def broadcast_to_cubes(
    self: 'Block',
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

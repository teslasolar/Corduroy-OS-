"""
KONOMI Core - Block Regions
Region-based operations for BlockArrays
"""

from typing import Dict, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Block


def clear_region(
    self: 'Block',
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
    self: 'Block',
    x_range: Tuple[int, int],
    y_range: Tuple[int, int],
    z_range: Tuple[int, int]
) -> Dict[Tuple[int, int, int], float]:
    """
    Query all values in a region.

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
    self: 'Block',
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

"""
KONOMI Core - Cube Management
Cube creation and management
"""

from typing import Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Konomi
    from ..cube import Cube


def cube(self: 'Konomi', name: str) -> 'Cube':
    """
    Create or get Cube instance.

    Args:
        name: Cube identifier

    Returns:
        Cube instance
    """
    from ..cube import Cube as CubeClass

    if name not in self.cubes:
        self.cubes[name] = CubeClass()

    return self.cubes[name]


def delete_cube(self: 'Konomi', name: str) -> bool:
    """
    Delete a Cube instance.

    Returns:
        True if deleted, False if not found
    """
    if name in self.cubes:
        del self.cubes[name]
        return True
    return False


def list_cubes(self: 'Konomi') -> Dict[str, Dict[str, Any]]:
    """
    List all Cube instances with stats.

    Returns:
        Dict of {name: status} for each cube
    """
    return {
        name: cube.status()
        for name, cube in self.cubes.items()
    }

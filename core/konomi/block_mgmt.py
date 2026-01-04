"""
KONOMI Core - Block Management
BlockArray creation and management
"""

from typing import Dict, Tuple, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Konomi
    from ..block import Block


def block(
    self: 'Konomi',
    dims: Tuple[int, int, int] = (10, 10, 10),
    name: Optional[str] = None
) -> 'Block':
    """
    Create or get BlockArray instance.

    Args:
        dims: Dimensions (x, y, z) of the grid
        name: Optional custom name (auto-generated if None)

    Returns:
        Block instance
    """
    from ..block import Block as BlockClass

    if name is None:
        name = f"block_{self._block_counter}"
        self._block_counter += 1

    if name not in self.blocks:
        self.blocks[name] = BlockClass(dims)

    return self.blocks[name]


def delete_block(self: 'Konomi', name: str) -> bool:
    """
    Delete a BlockArray instance.

    Returns:
        True if deleted, False if not found
    """
    if name in self.blocks:
        del self.blocks[name]
        return True
    return False


def list_blocks(self: 'Konomi') -> Dict[str, Dict[str, Any]]:
    """
    List all BlockArray instances with stats.

    Returns:
        Dict of {name: info} for each block
    """
    return {
        name: block.info()
        for name, block in self.blocks.items()
    }

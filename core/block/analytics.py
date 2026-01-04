"""
KONOMI Core - Block Analytics
Analytics and metrics for BlockArrays
"""

from typing import Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Block


def density(self: 'Block') -> float:
    """
    Calculate grid density (active cells / capacity).

    Returns:
        Density ratio between 0 and 1
    """
    return len(self.a) / self.metadata['capacity']


def memory_estimate(self: 'Block') -> Dict[str, float]:
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


def info(self: 'Block') -> Dict[str, Any]:
    """
    Get BlockArray information.

    Returns:
        Dict with metadata and stats
    """
    mem = memory_estimate(self)

    return {
        **self.metadata,
        'density': density(self),
        'memory_mb': mem['total_mb'],
        'memory_breakdown': mem
    }

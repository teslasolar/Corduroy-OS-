"""
KONOMI Core - Konomi Status
System-wide status and metrics
"""

from typing import Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Konomi


def status(self: 'Konomi') -> Dict[str, Any]:
    """
    Get system-wide status.

    Returns:
        Dict with comprehensive system information
    """
    # Calculate memory usage
    total_block_memory = sum(
        block.memory_estimate()['total_mb']
        for block in self.blocks.values()
    )

    # Count total LLMs
    total_llms = sum(
        block.metadata['active_llms']
        for block in self.blocks.values()
    )
    total_llms += sum(
        cube.status()['active_llms']
        for cube in self.cubes.values()
    )

    # eVGPU info
    evgpu_info = self.evgpu.info()

    return {
        'metadata': self.metadata,
        'evgpu': evgpu_info,
        'blocks': {
            'count': len(self.blocks),
            'total_memory_mb': total_block_memory,
            'names': list(self.blocks.keys())
        },
        'cubes': {
            'count': len(self.cubes),
            'names': list(self.cubes.keys())
        },
        'llms': {
            'total_instances': total_llms
        }
    }

"""
KONOMI Core - Konomi Core
Main Konomi class with initialization
"""

from typing import Dict, Tuple, Any, Optional
from ..evgpu import eVGPU
from ..block import Block
from ..cube import Cube
import jax.numpy as jnp


class Konomi:
    """
    Main orchestrator for Konomi Corduroy-OS.

    Manages:
    - eVGPU tensor operations
    - BlockArray instances (sparse 3D grids)
    - Cube instances (9-node LLM graphs)
    - Global system state
    """

    def __init__(self):
        """Initialize Konomi Corduroy-OS orchestrator."""
        self.evgpu = eVGPU()
        self.blocks: Dict[str, Block] = {}
        self._block_counter = 0
        self.cubes: Dict[str, Cube] = {}
        self.metadata = {
            'version': '0.1.0',
            'name': 'Konomi Corduroy-OS',
            'description': 'Hybrid 3D tensor + LLM + XP10 OS system',
            'components': {
                'evgpu': 'JAX CPU-only tensor operations',
                'block': 'Sparse 3D grid with LLM nodes',
                'cube': '9-node LLM graph',
                'xp10': 'Custom NT kernel (in development)'
            }
        }

    def tensor_op(self, a, b, op: str = '@'):
        """Convenience wrapper for eVGPU tensor operations."""
        return self.evgpu.t(a, b, op)

    def __repr__(self):
        return (
            f"Konomi(blocks={len(self.blocks)}, "
            f"cubes={len(self.cubes)}, "
            f"evgpu={self.evgpu.info()['platform']})"
        )

    # Import methods from other modules
    from .block_mgmt import block, delete_block, list_blocks
    from .cube_mgmt import cube, delete_cube, list_cubes
    from .status import status
    from .demo import demo, export_state

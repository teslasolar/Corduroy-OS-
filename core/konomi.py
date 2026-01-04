"""
Konomi Corduroy-OS - Main Orchestrator

🪟 Konomi: Central orchestrator for the entire Corduroy-OS system
"""

from typing import Dict, Tuple, Any, Optional
from .evgpu import eVGPU
from .block import Block
from .cube import Cube
from .femto import Femto
import jax.numpy as jnp


class Konomi:
    """
    Main orchestrator for Konomi Corduroy-OS.

    Manages:
    - eVGPU tensor operations
    - BlockArray instances (sparse 3D grids)
    - Cube instances (9-node LLM graphs)
    - Global system state

    Examples:
        >>> K = Konomi()
        >>> B = K.block((10, 10, 10))
        >>> B.set(0, 0, 0, 1.0)
        >>> C = K.cube("c1")
        >>> r = K.evgpu.t(jnp.ones((4, 4)), jnp.ones((4, 4)))
    """

    def __init__(self):
        """Initialize Konomi Corduroy-OS orchestrator."""

        # eVGPU instance for tensor operations
        self.evgpu = eVGPU()

        # BlockArray storage
        self.blocks: Dict[str, Block] = {}
        self._block_counter = 0

        # Cube storage
        self.cubes: Dict[str, Cube] = {}

        # Global metadata
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

    def block(
        self,
        dims: Tuple[int, int, int] = (10, 10, 10),
        name: Optional[str] = None
    ) -> Block:
        """
        Create or get BlockArray instance.

        Args:
            dims: Dimensions (x, y, z) of the grid
            name: Optional custom name (auto-generated if None)

        Returns:
            Block instance

        Examples:
            >>> K = Konomi()
            >>> B1 = K.block((100, 100, 100))
            >>> B2 = K.block((10, 10, 10), name="small_grid")
        """
        # Generate name if not provided
        if name is None:
            name = f"block_{self._block_counter}"
            self._block_counter += 1

        # Create new block if doesn't exist
        if name not in self.blocks:
            self.blocks[name] = Block(dims)

        return self.blocks[name]

    def cube(self, name: str) -> Cube:
        """
        Create or get Cube instance.

        Args:
            name: Cube identifier

        Returns:
            Cube instance

        Examples:
            >>> K = Konomi()
            >>> C = K.cube("main_cube")
            >>> C.connect('NEU', 'SWD')
        """
        if name not in self.cubes:
            self.cubes[name] = Cube()

        return self.cubes[name]

    def delete_block(self, name: str) -> bool:
        """
        Delete a BlockArray instance.

        Args:
            name: Block name

        Returns:
            True if deleted, False if not found
        """
        if name in self.blocks:
            del self.blocks[name]
            return True
        return False

    def delete_cube(self, name: str) -> bool:
        """
        Delete a Cube instance.

        Args:
            name: Cube name

        Returns:
            True if deleted, False if not found
        """
        if name in self.cubes:
            del self.cubes[name]
            return True
        return False

    def list_blocks(self) -> Dict[str, Dict[str, Any]]:
        """
        List all BlockArray instances with stats.

        Returns:
            Dict of {name: info} for each block
        """
        return {
            name: block.info()
            for name, block in self.blocks.items()
        }

    def list_cubes(self) -> Dict[str, Dict[str, Any]]:
        """
        List all Cube instances with stats.

        Returns:
            Dict of {name: status} for each cube
        """
        return {
            name: cube.status()
            for name, cube in self.cubes.items()
        }

    def tensor_op(self, a, b, op: str = '@'):
        """
        Convenience wrapper for eVGPU tensor operations.

        Args:
            a, b: Tensors
            op: Operation ('@', '+', '-', '*')

        Returns:
            Result tensor

        Examples:
            >>> K = Konomi()
            >>> result = K.tensor_op(
            ...     jnp.ones((3, 3)),
            ...     jnp.ones((3, 3)),
            ...     op='@'
            ... )
        """
        return self.evgpu.t(a, b, op)

    def status(self) -> Dict[str, Any]:
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

    def demo(self) -> Dict[str, Any]:
        """
        Run a quick demonstration of all components.

        Returns:
            Dict with demo results
        """
        results = {}

        # Test eVGPU
        from jax import random
        key = random.PRNGKey(0)
        a = random.normal(key, (4, 4))
        b = random.normal(key, (4, 4))
        tensor_result = self.evgpu.t(a, b, '@')
        results['evgpu'] = {
            'input_shape': a.shape,
            'output_shape': tensor_result.shape,
            'operation': 'matmul'
        }

        # Test Block
        demo_block = self.block((10, 10, 10), name='demo_block')
        demo_block.set(5, 5, 5, 42.0)
        llm_at_555 = demo_block.at(5, 5, 5)
        llm_response = llm_at_555.p("Hello from Konomi")
        results['block'] = {
            'value_at_555': demo_block.get(5, 5, 5),
            'llm_response': llm_response,
            'info': demo_block.info()
        }

        # Test Cube
        demo_cube = self.cube('demo_cube')
        demo_cube.connect('NEU', 'SWD')
        msg = demo_cube.send_message('NEU', 'central', 'Status check')
        results['cube'] = {
            'message_response': msg,
            'status': demo_cube.status()
        }

        return results

    def export_state(self) -> Dict[str, Any]:
        """
        Export entire system state for serialization.

        Returns:
            Dict containing all state information
        """
        return {
            'metadata': self.metadata,
            'blocks': {
                name: {
                    'dims': block.dims,
                    'data': dict(block.a),
                    'llm_count': len(block.L)
                }
                for name, block in self.blocks.items()
            },
            'cubes': {
                name: {
                    'edges': dict(cube.edges),
                    'messages': cube.message_history
                }
                for name, cube in self.cubes.items()
            }
        }

    def __repr__(self):
        return (
            f"Konomi(blocks={len(self.blocks)}, "
            f"cubes={len(self.cubes)}, "
            f"evgpu={self.evgpu.info()['platform']})"
        )


# Convenience instance for quick imports
konomi = Konomi()

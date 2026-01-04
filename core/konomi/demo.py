"""
KONOMI Core - Konomi Demo
Demo and export functionality
"""

from typing import Dict, Any, TYPE_CHECKING
from jax import random
import jax.numpy as jnp

if TYPE_CHECKING:
    from .core import Konomi


def demo(self: 'Konomi') -> Dict[str, Any]:
    """
    Run a quick demonstration of all components.

    Returns:
        Dict with demo results
    """
    results = {}

    # Test eVGPU
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
    from .block_mgmt import block
    demo_block = block(self, (10, 10, 10), name='demo_block')
    demo_block.set(5, 5, 5, 42.0)
    llm_at_555 = demo_block.at(5, 5, 5)
    llm_response = llm_at_555.p("Hello from Konomi")
    results['block'] = {
        'value_at_555': demo_block.get(5, 5, 5),
        'llm_response': llm_response,
        'info': demo_block.info()
    }

    # Test Cube
    from .cube_mgmt import cube
    demo_cube = cube(self, 'demo_cube')
    demo_cube.connect('NEU', 'SWD')
    msg = demo_cube.send_message('NEU', 'central', 'Status check')
    results['cube'] = {
        'message_response': msg,
        'status': demo_cube.status()
    }

    return results


def export_state(self: 'Konomi') -> Dict[str, Any]:
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

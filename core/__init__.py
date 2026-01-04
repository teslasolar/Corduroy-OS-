"""
Konomi Corduroy-OS - Core Module

Main components:
- eVGPU: CPU-only tensor operations (JAX)
- Femto: 16d nano LLM
- Block: Sparse 3D grid with LLM nodes
- Cube: 9-node graph geometry
- Konomi: Main orchestrator
"""

from .evgpu import eVGPU, evgpu
from .femto import Femto
from .block import Block
from .cube import Cube
from .konomi import Konomi, konomi

__all__ = ['eVGPU', 'evgpu', 'Femto', 'Block', 'Cube', 'Konomi', 'konomi']

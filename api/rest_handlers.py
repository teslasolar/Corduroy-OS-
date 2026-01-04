"""
Konomi Corduroy-OS - REST API Handlers
REST-style endpoints over custom dial-up protocol
"""

import json
from typing import Dict, Any


class RESTHandlers:
    """
    REST-style API handlers for BlockArray and LLM operations.

    Endpoints:
    - POST /template - Create BlockArray template
    - POST /instance - Instantiate Block with dimensions
    - GET /val - Get value at (x,y,z)
    - POST /llm - Query Femto at coordinate
    """

    def __init__(self, konomi):
        """
        Initialize handlers.

        Args:
            konomi: Konomi instance
        """
        self.konomi = konomi

    def create_template(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create BlockArray template.

        Request:
            {
                "name": "my_block",
                "dims": [100, 100, 100]
            }

        Returns:
            {
                "success": true,
                "name": "my_block",
                "dims": [100, 100, 100]
            }
        """
        name = request.get('name', f'block_{len(self.konomi.blocks)}')
        dims = tuple(request.get('dims', [10, 10, 10]))

        block = self.konomi.block(dims=dims, name=name)

        return {
            'success': True,
            'name': name,
            'dims': list(block.dims),
            'info': block.info()
        }

    def create_instance(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Instantiate Block with dimensions.

        Request:
            {
                "name": "large_grid",
                "dims": [1000, 1000, 1000]
            }

        Returns:
            {
                "success": true,
                "name": "large_grid"
            }
        """
        return self.create_template(request)

    def get_value(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get value at coordinate.

        Request:
            {
                "block": "my_block",
                "x": 5,
                "y": 10,
                "z": 15
            }

        Returns:
            {
                "value": 42.0
            }
        """
        block_name = request.get('block', 'default')
        x = request.get('x', 0)
        y = request.get('y', 0)
        z = request.get('z', 0)

        if block_name not in self.konomi.blocks:
            return {'error': f'Block not found: {block_name}'}

        block = self.konomi.blocks[block_name]
        value = block.get(x, y, z)

        return {'value': value}

    def query_llm(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Query Femto LLM at coordinate.

        Request:
            {
                "block": "my_block",
                "x": 5,
                "y": 10,
                "z": 15,
                "text": "Process this data"
            }

        Returns:
            {
                "response": "[Process this data]"
            }
        """
        block_name = request.get('block', 'default')
        x = request.get('x', 0)
        y = request.get('y', 0)
        z = request.get('z', 0)
        text = request.get('text', '')

        block = self.konomi.block(name=block_name)
        llm = block.at(x, y, z)
        response = llm.p(text)

        return {'response': response}

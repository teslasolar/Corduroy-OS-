"""
Konomi Corduroy-OS - WebSocket API Handlers
WebSocket-style handlers for Cube operations
"""

import json
from typing import Dict, Any


class WebSocketHandlers:
    """
    WebSocket-style API handlers for Cube operations.

    Actions:
    - init: Initialize Cube
    - process: Process data at vertex
    - connect: Add edge between vertices
    - status: Get Cube status
    """

    def __init__(self, konomi):
        """
        Initialize handlers.

        Args:
            konomi: Konomi instance
        """
        self.konomi = konomi

    def handle_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle WebSocket message.

        Args:
            message: Message dict with 'action' field

        Returns:
            Response dict
        """
        action = message.get('action')

        if action == 'init':
            return self.init_cube(message)
        elif action == 'process':
            return self.process_vertex(message)
        elif action == 'connect':
            return self.connect_vertices(message)
        elif action == 'status':
            return self.cube_status(message)
        else:
            return {'error': f'Unknown action: {action}'}

    def init_cube(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Initialize Cube.

        Message:
            {
                "action": "init",
                "name": "my_cube"
            }

        Returns:
            {
                "success": true,
                "name": "my_cube",
                "status": {...}
            }
        """
        name = message.get('name', 'default')
        cube = self.konomi.cube(name)

        return {
            'success': True,
            'name': name,
            'status': cube.status()
        }

    def process_vertex(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process data at vertex.

        Message:
            {
                "action": "process",
                "cube": "my_cube",
                "vertex": "NEU",
                "data": "analyze this"
            }

        Returns:
            {
                "result": "[analyze this]"
            }
        """
        cube_name = message.get('cube', 'default')
        vertex = message.get('vertex', 'NEU')
        data = message.get('data', '')

        cube = self.konomi.cube(cube_name)
        result = cube.process_at_vertex(vertex, data)

        return {'result': result}

    def connect_vertices(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Connect two vertices.

        Message:
            {
                "action": "connect",
                "cube": "my_cube",
                "from": "NEU",
                "to": "SWD"
            }

        Returns:
            {
                "success": true
            }
        """
        cube_name = message.get('cube', 'default')
        from_vertex = message.get('from')
        to_vertex = message.get('to')

        cube = self.konomi.cube(cube_name)
        cube.connect(from_vertex, to_vertex)

        return {'success': True}

    def cube_status(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get Cube status.

        Message:
            {
                "action": "status",
                "cube": "my_cube"
            }

        Returns:
            {
                "status": {...}
            }
        """
        cube_name = message.get('cube', 'default')
        cube = self.konomi.cube(cube_name)

        return {'status': cube.status()}

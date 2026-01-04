"""
KONOMI Protocol - Server Core
DialUpServer class initialization
"""

from typing import Dict, Callable
from ..dialup import DialUpSession, OpCode


class DialUpServer:
    """
    Async server for the dial-up protocol.

    Listens on:
    - Port 3001: API operations (BlockArray, Cube)
    - Port 3002: WebSocket-style streaming
    - Port 6789: Cube mesh communication
    """

    def __init__(self, konomi):
        """
        Initialize server.

        Args:
            konomi: Konomi instance to handle requests
        """
        self.konomi = konomi
        self.sessions: Dict[str, DialUpSession] = {}
        self.handlers: Dict[OpCode, Callable] = {}

        self.ports = {
            'api': 3001,
            'stream': 3002,
            'mesh': 6789
        }

        self.servers = []

        # Register handlers
        self._register_handlers()

    def _register_handlers(self):
        """Register opcode handlers."""
        from .handlers import (
            handle_handshake,
            handle_tensor_op,
            handle_llm_query,
            handle_block_get,
            handle_block_set,
            handle_cube_msg,
            handle_cube_status,
        )

        self.handlers = {
            OpCode.HANDSHAKE: handle_handshake,
            OpCode.TENSOR_OP: handle_tensor_op,
            OpCode.LLM_QUERY: handle_llm_query,
            OpCode.BLOCK_GET: handle_block_get,
            OpCode.BLOCK_SET: handle_block_set,
            OpCode.CUBE_MSG: handle_cube_msg,
            OpCode.CUBE_STATUS: handle_cube_status,
        }

    # Import lifecycle methods
    from .lifecycle import handle_client, start, stop

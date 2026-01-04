"""
Konomi Corduroy-OS - Protocol Server
Async server for dial-up protocol

Listens on multiple ports for different services
"""

import asyncio
import json
from typing import Dict, Optional, Callable, Any
from .dialup import DialUpPacket, OpCode, DialUpSession, verify_packet
from .serializer import serialize_tensor, deserialize_tensor
import jax.numpy as jnp


class DialUpServer:
    """
    Async server for the dial-up protocol.

    Listens on:
    - Port 3001: API operations (BlockArray, Cube)
    - Port 3002: WebSocket-style streaming
    - Port 6789: Cube mesh communication

    Examples:
        >>> server = DialUpServer(konomi_instance)
        >>> await server.start()
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
        self._register_handlers()

        # Server configuration
        self.ports = {
            'api': 3001,
            'stream': 3002,
            'mesh': 6789
        }

        self.servers = []

    def _register_handlers(self):
        """Register opcode handlers."""
        self.handlers = {
            OpCode.HANDSHAKE: self._handle_handshake,
            OpCode.TENSOR_OP: self._handle_tensor_op,
            OpCode.LLM_QUERY: self._handle_llm_query,
            OpCode.BLOCK_GET: self._handle_block_get,
            OpCode.BLOCK_SET: self._handle_block_set,
            OpCode.CUBE_MSG: self._handle_cube_msg,
            OpCode.CUBE_STATUS: self._handle_cube_status,
        }

    async def _handle_handshake(
        self,
        packet: DialUpPacket,
        session_id: str
    ) -> bytes:
        """Handle handshake request."""
        # Create or update session
        if session_id not in self.sessions:
            self.sessions[session_id] = DialUpSession()

        session = self.sessions[session_id]
        session.connected = True

        # Send ACK
        response = session.ack(packet.seq)
        return response.pack()

    async def _handle_tensor_op(
        self,
        packet: DialUpPacket,
        session_id: str
    ) -> bytes:
        """Handle tensor operation."""
        try:
            # Parse payload: operation type + two tensors
            payload = packet.payload
            op_code = chr(payload[0])  # '@', '+', '-', '*'

            # Deserialize tensors (simplified for now)
            # TODO: Implement proper tensor deserialization from payload
            a = jnp.ones((4, 4))
            b = jnp.ones((4, 4))

            # Perform operation
            result = self.konomi.evgpu.t(a, b, op_code)

            # Serialize result
            result_data = serialize_tensor(result)

            # Create response packet
            session = self.sessions.get(session_id, DialUpSession())
            response = session.create_packet(OpCode.ACK, result_data)
            return response.pack()

        except Exception as e:
            # Error response
            error_msg = str(e).encode('utf-8')[:1024]
            session = self.sessions.get(session_id, DialUpSession())
            response = session.create_packet(OpCode.ERROR_INVALID, error_msg)
            return response.pack()

    async def _handle_llm_query(
        self,
        packet: DialUpPacket,
        session_id: str
    ) -> bytes:
        """Handle LLM query."""
        try:
            # Parse payload: block_name + coords + text
            payload_str = packet.payload.decode('utf-8')
            data = json.loads(payload_str)

            block_name = data.get('block', 'default')
            x = data.get('x', 0)
            y = data.get('y', 0)
            z = data.get('z', 0)
            text = data.get('text', '')

            # Get or create block
            block = self.konomi.block(name=block_name)

            # Query LLM at coordinates
            llm = block.at(x, y, z)
            result = llm.p(text)

            # Create response
            response_data = json.dumps({'result': result}).encode('utf-8')
            session = self.sessions.get(session_id, DialUpSession())
            response = session.create_packet(OpCode.ACK, response_data)
            return response.pack()

        except Exception as e:
            error_msg = str(e).encode('utf-8')[:1024]
            session = self.sessions.get(session_id, DialUpSession())
            response = session.create_packet(OpCode.ERROR_INVALID, error_msg)
            return response.pack()

    async def _handle_block_get(
        self,
        packet: DialUpPacket,
        session_id: str
    ) -> bytes:
        """Handle block value get."""
        try:
            payload_str = packet.payload.decode('utf-8')
            data = json.loads(payload_str)

            block_name = data.get('block', 'default')
            x = data.get('x', 0)
            y = data.get('y', 0)
            z = data.get('z', 0)

            # Get block
            block = self.konomi.block(name=block_name)
            value = block.get(x, y, z)

            # Create response
            response_data = json.dumps({'value': value}).encode('utf-8')
            session = self.sessions.get(session_id, DialUpSession())
            response = session.create_packet(OpCode.ACK, response_data)
            return response.pack()

        except Exception as e:
            error_msg = str(e).encode('utf-8')[:1024]
            session = self.sessions.get(session_id, DialUpSession())
            response = session.create_packet(OpCode.ERROR_INVALID, error_msg)
            return response.pack()

    async def _handle_block_set(
        self,
        packet: DialUpPacket,
        session_id: str
    ) -> bytes:
        """Handle block value set."""
        try:
            payload_str = packet.payload.decode('utf-8')
            data = json.loads(payload_str)

            block_name = data.get('block', 'default')
            x = data.get('x', 0)
            y = data.get('y', 0)
            z = data.get('z', 0)
            value = data.get('value', 0.0)

            # Get block and set value
            block = self.konomi.block(name=block_name)
            block.set(x, y, z, value)

            # Create response
            response_data = json.dumps({'success': True}).encode('utf-8')
            session = self.sessions.get(session_id, DialUpSession())
            response = session.create_packet(OpCode.ACK, response_data)
            return response.pack()

        except Exception as e:
            error_msg = str(e).encode('utf-8')[:1024]
            session = self.sessions.get(session_id, DialUpSession())
            response = session.create_packet(OpCode.ERROR_INVALID, error_msg)
            return response.pack()

    async def _handle_cube_msg(
        self,
        packet: DialUpPacket,
        session_id: str
    ) -> bytes:
        """Handle cube message passing."""
        try:
            payload_str = packet.payload.decode('utf-8')
            data = json.loads(payload_str)

            cube_name = data.get('cube', 'default')
            from_vertex = data.get('from')
            to_vertex = data.get('to')
            message = data.get('message', '')

            # Get cube
            cube = self.konomi.cube(cube_name)

            # Send message
            response_msg = cube.send_message(from_vertex, to_vertex, message)

            # Create response
            response_data = json.dumps({'response': response_msg}).encode('utf-8')
            session = self.sessions.get(session_id, DialUpSession())
            response = session.create_packet(OpCode.ACK, response_data)
            return response.pack()

        except Exception as e:
            error_msg = str(e).encode('utf-8')[:1024]
            session = self.sessions.get(session_id, DialUpSession())
            response = session.create_packet(OpCode.ERROR_INVALID, error_msg)
            return response.pack()

    async def _handle_cube_status(
        self,
        packet: DialUpPacket,
        session_id: str
    ) -> bytes:
        """Handle cube status request."""
        try:
            payload_str = packet.payload.decode('utf-8')
            data = json.loads(payload_str)

            cube_name = data.get('cube', 'default')

            # Get cube status
            cube = self.konomi.cube(cube_name)
            status = cube.status()

            # Create response
            response_data = json.dumps(status).encode('utf-8')
            session = self.sessions.get(session_id, DialUpSession())
            response = session.create_packet(OpCode.ACK, response_data)
            return response.pack()

        except Exception as e:
            error_msg = str(e).encode('utf-8')[:1024]
            session = self.sessions.get(session_id, DialUpSession())
            response = session.create_packet(OpCode.ERROR_INVALID, error_msg)
            return response.pack()

    async def handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter
    ):
        """
        Handle client connection.

        Args:
            reader: Stream reader
            writer: Stream writer
        """
        addr = writer.get_extra_info('peername')
        session_id = f"{addr[0]}:{addr[1]}"

        print(f"[DialUp] Client connected: {session_id}")

        try:
            while True:
                # Read packet header first
                header_data = await reader.read(DialUpPacket.HEADER_SIZE)

                if not header_data:
                    break

                # Parse header to get payload length
                # Read full packet
                # (For now, read up to max size)
                remaining = await reader.read(DialUpPacket.MAX_PAYLOAD)
                packet_data = header_data + remaining

                # Verify and unpack packet
                is_valid, packet = verify_packet(packet_data)

                if not is_valid:
                    print(f"[DialUp] Invalid packet from {session_id}")
                    continue

                # Handle packet based on opcode
                handler = self.handlers.get(packet.opcode)

                if handler:
                    response_data = await handler(packet, session_id)
                    writer.write(response_data)
                    await writer.drain()
                else:
                    print(f"[DialUp] Unknown opcode: {packet.opcode}")

        except Exception as e:
            print(f"[DialUp] Error with {session_id}: {e}")
        finally:
            print(f"[DialUp] Client disconnected: {session_id}")
            writer.close()
            await writer.wait_closed()

            # Clean up session
            if session_id in self.sessions:
                del self.sessions[session_id]

    async def start(self):
        """Start all servers."""
        print("[DialUp] Starting Konomi Corduroy-OS Protocol Server")

        # Start API server (port 3001)
        api_server = await asyncio.start_server(
            self.handle_client,
            '0.0.0.0',
            self.ports['api']
        )
        self.servers.append(api_server)
        print(f"[DialUp] API server listening on port {self.ports['api']}")

        # Start stream server (port 3002)
        stream_server = await asyncio.start_server(
            self.handle_client,
            '0.0.0.0',
            self.ports['stream']
        )
        self.servers.append(stream_server)
        print(f"[DialUp] Stream server listening on port {self.ports['stream']}")

        # Start mesh server (port 6789)
        mesh_server = await asyncio.start_server(
            self.handle_client,
            '0.0.0.0',
            self.ports['mesh']
        )
        self.servers.append(mesh_server)
        print(f"[DialUp] Mesh server listening on port {self.ports['mesh']}")

        # Serve forever
        async with api_server, stream_server, mesh_server:
            await asyncio.gather(
                api_server.serve_forever(),
                stream_server.serve_forever(),
                mesh_server.serve_forever()
            )

    async def stop(self):
        """Stop all servers."""
        print("[DialUp] Stopping servers...")
        for server in self.servers:
            server.close()
            await server.wait_closed()


# Main entry point
async def main():
    """Run server."""
    from core.konomi import Konomi

    # Create Konomi instance
    konomi = Konomi()

    # Create and start server
    server = DialUpServer(konomi)
    await server.start()


if __name__ == '__main__':
    asyncio.run(main())

"""
KONOMI Protocol - Server Lifecycle
Connection handling and server lifecycle
"""

import asyncio
from typing import TYPE_CHECKING
from ..dialup import DialUpPacket, verify_packet

if TYPE_CHECKING:
    from .core import DialUpServer


async def handle_client(
    self: 'DialUpServer',
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter
):
    """Handle client connection."""
    addr = writer.get_extra_info('peername')
    session_id = f"{addr[0]}:{addr[1]}"

    print(f"[DialUp] Client connected: {session_id}")

    try:
        while True:
            header_data = await reader.read(DialUpPacket.HEADER_SIZE)
            if not header_data:
                break

            remaining = await reader.read(DialUpPacket.MAX_PAYLOAD)
            packet_data = header_data + remaining

            is_valid, packet = verify_packet(packet_data)

            if not is_valid:
                print(f"[DialUp] Invalid packet from {session_id}")
                continue

            handler = self.handlers.get(packet.opcode)
            if handler:
                response_data = await handler(self, packet, session_id)
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

        if session_id in self.sessions:
            del self.sessions[session_id]


async def start(self: 'DialUpServer'):
    """Start all servers."""
    print("[DialUp] Starting Konomi Corduroy-OS Protocol Server")

    api_server = await asyncio.start_server(
        lambda r, w: handle_client(self, r, w),
        '0.0.0.0',
        self.ports['api']
    )
    self.servers.append(api_server)
    print(f"[DialUp] API server listening on port {self.ports['api']}")

    stream_server = await asyncio.start_server(
        lambda r, w: handle_client(self, r, w),
        '0.0.0.0',
        self.ports['stream']
    )
    self.servers.append(stream_server)
    print(f"[DialUp] Stream server listening on port {self.ports['stream']}")

    mesh_server = await asyncio.start_server(
        lambda r, w: handle_client(self, r, w),
        '0.0.0.0',
        self.ports['mesh']
    )
    self.servers.append(mesh_server)
    print(f"[DialUp] Mesh server listening on port {self.ports['mesh']}")

    async with api_server, stream_server, mesh_server:
        await asyncio.gather(
            api_server.serve_forever(),
            stream_server.serve_forever(),
            mesh_server.serve_forever()
        )


async def stop(self: 'DialUpServer'):
    """Stop all servers."""
    print("[DialUp] Stopping servers...")
    for server in self.servers:
        server.close()
        await server.wait_closed()

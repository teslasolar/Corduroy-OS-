"""
KONOMI Protocol - Server Handlers
Request handlers for each opcode
"""

import json
from typing import TYPE_CHECKING
import jax.numpy as jnp
from ..dialup import DialUpPacket, DialUpSession, OpCode
from ..serializer import serialize_tensor

if TYPE_CHECKING:
    from .core import DialUpServer


async def handle_handshake(
    server: 'DialUpServer',
    packet: DialUpPacket,
    session_id: str
) -> bytes:
    """Handle handshake request."""
    if session_id not in server.sessions:
        server.sessions[session_id] = DialUpSession()

    session = server.sessions[session_id]
    session.connected = True

    response = session.ack(packet.seq)
    return response.pack()


async def handle_tensor_op(
    server: 'DialUpServer',
    packet: DialUpPacket,
    session_id: str
) -> bytes:
    """Handle tensor operation."""
    try:
        payload = packet.payload
        op_code = chr(payload[0])

        a = jnp.ones((4, 4))
        b = jnp.ones((4, 4))
        result = server.konomi.evgpu.t(a, b, op_code)

        result_data = serialize_tensor(result)
        session = server.sessions.get(session_id, DialUpSession())
        response = session.create_packet(OpCode.ACK, result_data)
        return response.pack()
    except Exception as e:
        return _error_response(server, session_id, str(e))


async def handle_llm_query(
    server: 'DialUpServer',
    packet: DialUpPacket,
    session_id: str
) -> bytes:
    """Handle LLM query."""
    try:
        data = json.loads(packet.payload.decode('utf-8'))
        block = server.konomi.block(name=data.get('block', 'default'))
        llm = block.at(data.get('x', 0), data.get('y', 0), data.get('z', 0))
        result = llm.p(data.get('text', ''))

        response_data = json.dumps({'result': result}).encode('utf-8')
        session = server.sessions.get(session_id, DialUpSession())
        response = session.create_packet(OpCode.ACK, response_data)
        return response.pack()
    except Exception as e:
        return _error_response(server, session_id, str(e))


async def handle_block_get(
    server: 'DialUpServer',
    packet: DialUpPacket,
    session_id: str
) -> bytes:
    """Handle block value get."""
    try:
        data = json.loads(packet.payload.decode('utf-8'))
        block = server.konomi.block(name=data.get('block', 'default'))
        value = block.get(data.get('x', 0), data.get('y', 0), data.get('z', 0))

        response_data = json.dumps({'value': value}).encode('utf-8')
        session = server.sessions.get(session_id, DialUpSession())
        response = session.create_packet(OpCode.ACK, response_data)
        return response.pack()
    except Exception as e:
        return _error_response(server, session_id, str(e))


async def handle_block_set(
    server: 'DialUpServer',
    packet: DialUpPacket,
    session_id: str
) -> bytes:
    """Handle block value set."""
    try:
        data = json.loads(packet.payload.decode('utf-8'))
        block = server.konomi.block(name=data.get('block', 'default'))
        block.set(
            data.get('x', 0),
            data.get('y', 0),
            data.get('z', 0),
            data.get('value', 0.0)
        )

        response_data = json.dumps({'success': True}).encode('utf-8')
        session = server.sessions.get(session_id, DialUpSession())
        response = session.create_packet(OpCode.ACK, response_data)
        return response.pack()
    except Exception as e:
        return _error_response(server, session_id, str(e))


async def handle_cube_msg(
    server: 'DialUpServer',
    packet: DialUpPacket,
    session_id: str
) -> bytes:
    """Handle cube message passing."""
    try:
        data = json.loads(packet.payload.decode('utf-8'))
        cube = server.konomi.cube(data.get('cube', 'default'))
        response_msg = cube.send_message(
            data.get('from'),
            data.get('to'),
            data.get('message', '')
        )

        response_data = json.dumps({'response': response_msg}).encode('utf-8')
        session = server.sessions.get(session_id, DialUpSession())
        response = session.create_packet(OpCode.ACK, response_data)
        return response.pack()
    except Exception as e:
        return _error_response(server, session_id, str(e))


async def handle_cube_status(
    server: 'DialUpServer',
    packet: DialUpPacket,
    session_id: str
) -> bytes:
    """Handle cube status request."""
    try:
        data = json.loads(packet.payload.decode('utf-8'))
        cube = server.konomi.cube(data.get('cube', 'default'))
        status = cube.status()

        response_data = json.dumps(status).encode('utf-8')
        session = server.sessions.get(session_id, DialUpSession())
        response = session.create_packet(OpCode.ACK, response_data)
        return response.pack()
    except Exception as e:
        return _error_response(server, session_id, str(e))


def _error_response(server: 'DialUpServer', session_id: str, error: str) -> bytes:
    """Create error response packet."""
    error_msg = error.encode('utf-8')[:1024]
    session = server.sessions.get(session_id, DialUpSession())
    response = session.create_packet(OpCode.ERROR_INVALID, error_msg)
    return response.pack()

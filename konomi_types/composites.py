"""
KONOMI UDT Layer 2: Composite Types
Nested types that bundle Layer 1 primitives
"""

from typing import List, Optional, NamedTuple
from .primitives import Coord3D, Range3D, MemoryEstimate
from .block_types import BlockMetadata, RegionBounds
from .cube_types import CubeStatus, MessageRecord
from .protocol_types import PacketHeader, SessionState, DTypeCode, ServerPorts


class BlockContext(NamedTuple):
    """
    Full block state with metadata, bounds, and memory.
    ISA-95 Layer 2 composite.
    """
    metadata: BlockMetadata
    bounds: RegionBounds
    memory: MemoryEstimate

    @classmethod
    def from_block(cls, block) -> 'BlockContext':
        """Create context from a Block instance."""
        info = block.info()
        mem = block.memory_estimate()

        # Find active bounds
        coords = list(block.a.keys())
        if coords:
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            zs = [c[2] for c in coords]
            bounds = RegionBounds(
                min(xs), max(xs),
                min(ys), max(ys),
                min(zs), max(zs)
            )
        else:
            bounds = RegionBounds(0, 0, 0, 0, 0, 0)

        return cls(
            metadata=BlockMetadata(
                dims=info['dims'],
                capacity=info['dims'][0] * info['dims'][1] * info['dims'][2],
                active_cells=info['active_cells'],
                active_llms=info['active_llms'],
                active_cubes=info.get('active_cubes', 0)
            ),
            bounds=bounds,
            memory=MemoryEstimate(
                cells_mb=mem['cells_mb'],
                llms_mb=mem['llms_mb'],
                cubes_mb=mem.get('cubes_mb', 0.0),
                total_mb=mem['total_mb']
            )
        )


class CubeGraph(NamedTuple):
    """
    Cube status with message history.
    ISA-95 Layer 2 composite.
    """
    status: CubeStatus
    messages: List[MessageRecord]

    @classmethod
    def from_cube(cls, cube) -> 'CubeGraph':
        """Create graph from a Cube instance."""
        status = CubeStatus.from_cube(cube.edges, len(cube.message_history))
        messages = [
            MessageRecord(
                from_vertex=m['from'],
                to_vertex=m['to'],
                message=m['message'],
                response=m['response']
            )
            for m in cube.message_history
        ]
        return cls(status=status, messages=messages)


class NetworkPacket(NamedTuple):
    """
    Complete packet with header, session, and data type.
    ISA-95 Layer 2 composite.
    """
    header: PacketHeader
    session: SessionState
    dtype: DTypeCode

    @classmethod
    def create(
        cls,
        opcode: int,
        payload_len: int,
        seq: int,
        crc: int,
        flags: int,
        connected: bool,
        dtype: DTypeCode = DTypeCode.FLOAT32
    ) -> 'NetworkPacket':
        """Create packet with full context."""
        return cls(
            header=PacketHeader(opcode, payload_len, seq, crc, flags),
            session=SessionState(seq, connected, seq - 1 if seq > 0 else 0),
            dtype=dtype
        )


class SpatialQuery(NamedTuple):
    """
    Spatial query with origin, range, and optional result.
    ISA-95 Layer 2 composite.
    """
    origin: Coord3D
    range: Range3D
    result: Optional[BlockMetadata]

    @classmethod
    def create(
        cls,
        origin: Coord3D,
        x_range: tuple,
        y_range: tuple,
        z_range: tuple
    ) -> 'SpatialQuery':
        """Create query without result."""
        return cls(
            origin=origin,
            range=Range3D(x_range, y_range, z_range),
            result=None
        )

    def with_result(self, metadata: BlockMetadata) -> 'SpatialQuery':
        """Return query with result attached."""
        return self._replace(result=metadata)


class LLMDeployment(NamedTuple):
    """
    LLM deployment with location, host, and resources.
    ISA-95 Layer 2 composite.
    """
    location: Coord3D
    host_block: BlockMetadata
    host_cube: Optional[CubeStatus]
    memory: MemoryEstimate

    @classmethod
    def standalone(
        cls,
        location: Coord3D,
        block_dims: tuple
    ) -> 'LLMDeployment':
        """Create standalone LLM deployment (no cube)."""
        return cls(
            location=location,
            host_block=BlockMetadata.create(block_dims),
            host_cube=None,
            memory=MemoryEstimate.from_counts(0, 1, 0)
        )


class ProtocolSession(NamedTuple):
    """
    Full protocol session state.
    ISA-95 Layer 2 composite.
    """
    state: SessionState
    last_header: Optional[PacketHeader]
    ports: ServerPorts

    @classmethod
    def new(cls, ports: ServerPorts = None) -> 'ProtocolSession':
        """Create new session."""
        return cls(
            state=SessionState.new(),
            last_header=None,
            ports=ports or ServerPorts()
        )

    def with_header(self, header: PacketHeader) -> 'ProtocolSession':
        """Return session with updated header."""
        return self._replace(
            last_header=header,
            state=self.state.next_seq().set_ack(header.seq)
        )

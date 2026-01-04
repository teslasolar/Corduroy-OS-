"""
Tests for dial-up protocol and serialization
"""

import pytest
import jax.numpy as jnp
import numpy as np
from protocol.dialup import DialUpPacket, OpCode, DialUpSession, verify_packet
from protocol.serializer import serialize_tensor, deserialize_tensor, TensorSerializer


class TestDialUpPacket:
    def test_packet_creation(self):
        """Test creating a packet."""
        packet = DialUpPacket(OpCode.HANDSHAKE, b'test', seq=1)

        assert packet.opcode == OpCode.HANDSHAKE
        assert packet.payload == b'test'
        assert packet.seq == 1

    def test_packet_pack_unpack(self):
        """Test packing and unpacking."""
        original = DialUpPacket(OpCode.TENSOR_OP, b'data123', seq=42)

        # Pack
        binary = original.pack()

        # Unpack
        restored = DialUpPacket.unpack(binary)

        assert restored.opcode == original.opcode
        assert restored.payload == original.payload
        assert restored.seq == original.seq

    def test_crc16_validation(self):
        """Test CRC16 checksum validation."""
        packet = DialUpPacket(OpCode.ACK, b'test data')
        binary = packet.pack()

        # Corrupt payload
        corrupted = binary[:8] + b'X' + binary[9:]

        with pytest.raises(ValueError, match="CRC mismatch"):
            DialUpPacket.unpack(corrupted)

    def test_max_payload_size(self):
        """Test maximum payload size enforcement."""
        large_payload = b'x' * (DialUpPacket.MAX_PAYLOAD + 1)

        with pytest.raises(ValueError, match="Payload too large"):
            DialUpPacket(OpCode.TENSOR_OP, large_payload)

    def test_is_valid(self):
        """Test packet validation."""
        valid_packet = DialUpPacket(OpCode.HANDSHAKE, b'test')
        assert valid_packet.is_valid()

    def test_verify_packet_utility(self):
        """Test verify_packet utility function."""
        packet = DialUpPacket(OpCode.LLM_QUERY, b'query')
        binary = packet.pack()

        is_valid, restored = verify_packet(binary)

        assert is_valid
        assert restored.opcode == OpCode.LLM_QUERY

    def test_empty_payload(self):
        """Test packet with empty payload."""
        packet = DialUpPacket(OpCode.KEEPALIVE, b'')

        binary = packet.pack()
        restored = DialUpPacket.unpack(binary)

        assert restored.payload == b''

    def test_repr(self):
        """Test string representation."""
        packet = DialUpPacket(OpCode.TENSOR_OP, b'test', seq=5)
        repr_str = repr(packet)

        assert 'DialUpPacket' in repr_str
        assert 'TENSOR_OP' in repr_str


class TestDialUpSession:
    def test_session_creation(self):
        """Test creating a session."""
        session = DialUpSession()

        assert session.seq == 0
        assert not session.connected

    def test_create_packet_increments_seq(self):
        """Test that creating packets increments sequence."""
        session = DialUpSession()

        p1 = session.create_packet(OpCode.HANDSHAKE, b'test1')
        p2 = session.create_packet(OpCode.HANDSHAKE, b'test2')

        assert p1.seq == 0
        assert p2.seq == 1

    def test_handshake(self):
        """Test handshake packet creation."""
        session = DialUpSession()
        packet = session.handshake()

        assert packet.opcode == OpCode.HANDSHAKE
        assert b'KONOMI-CORDUROY-OS' in packet.payload

    def test_ack(self):
        """Test ACK packet creation."""
        session = DialUpSession()
        packet = session.ack(42)

        assert packet.opcode == OpCode.ACK
        assert session.last_ack == 42

    def test_keepalive(self):
        """Test keepalive packet creation."""
        session = DialUpSession()
        packet = session.keepalive()

        assert packet.opcode == OpCode.KEEPALIVE


class TestTensorSerializer:
    def test_serialize_1d(self):
        """Test serializing 1D array."""
        arr = jnp.array([1.0, 2.0, 3.0, 4.0])

        binary = serialize_tensor(arr)
        restored = deserialize_tensor(binary)

        assert jnp.array_equal(restored, arr)

    def test_serialize_2d(self):
        """Test serializing 2D array."""
        arr = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        binary = serialize_tensor(arr)
        restored = deserialize_tensor(binary)

        assert jnp.array_equal(restored, arr)

    def test_serialize_3d(self):
        """Test serializing 3D array."""
        arr = jnp.ones((3, 4, 5))

        binary = serialize_tensor(arr)
        restored = deserialize_tensor(binary)

        assert restored.shape == arr.shape
        assert jnp.allclose(restored, arr)

    def test_different_dtypes(self):
        """Test serializing different data types."""
        # Float32
        arr_f32 = jnp.array([1.0, 2.0], dtype=jnp.float32)
        binary = serialize_tensor(arr_f32)
        restored = deserialize_tensor(binary)
        assert jnp.array_equal(restored, arr_f32)

        # Int32
        arr_i32 = jnp.array([1, 2, 3], dtype=jnp.int32)
        binary = serialize_tensor(arr_i32)
        restored = deserialize_tensor(binary)
        assert jnp.array_equal(restored, arr_i32)

    def test_estimate_size(self):
        """Test size estimation."""
        size = TensorSerializer.estimate_size((10, 10), np.float32)

        # Header (5) + shape (2*2=4) + data (10*10*4=400)
        expected = 5 + 4 + 400
        assert size == expected

    def test_round_trip_preserves_data(self):
        """Test that serialization round-trip preserves data."""
        original = jnp.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]])

        binary = serialize_tensor(original)
        restored = deserialize_tensor(binary)

        assert jnp.allclose(restored, original)
        assert restored.shape == original.shape

    def test_numpy_array_input(self):
        """Test that numpy arrays work."""
        arr = np.array([1, 2, 3, 4])

        binary = serialize_tensor(arr)
        restored = deserialize_tensor(binary)

        assert jnp.array_equal(restored, arr)

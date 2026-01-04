#!/usr/bin/env python3
"""
Konomi Corduroy-OS Demo
Quick demonstration of all system components
"""

import jax.numpy as jnp
from core.konomi import Konomi


def banner(text):
    """Print a fancy banner."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")


def main():
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║          🪟 Konomi Corduroy-OS Demo                     ║
    ║                                                          ║
    ║   Hybrid 3D Tensor + Distributed LLM + XP10 OS          ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """)

    # Initialize Konomi
    banner("🎭 Initializing Konomi Orchestrator")
    K = Konomi()
    print(f"✓ {K}")

    # eVGPU Demo
    banner("⚡ eVGPU: CPU-only Tensor Operations")
    print("Matrix multiply: 4×4 @ 4×4")
    a = jnp.ones((4, 4))
    b = jnp.ones((4, 4))
    result = K.evgpu.t(a, b, '@')
    print(f"✓ Result shape: {result.shape}")
    print(f"✓ Result[0,0]: {result[0,0]}")

    # eVGPU info
    evgpu_info = K.evgpu.info()
    print(f"✓ Platform: {evgpu_info['platform']}")
    print(f"✓ JAX version: {evgpu_info['version']}")

    # BlockArray Demo
    banner("🧊 BlockArray: Sparse 3D Grid with LLMs")
    print("Creating 10×10×10 block...")
    B = K.block((10, 10, 10), name="demo_block")
    print(f"✓ Block created: {B.dims}")

    print("\nSetting values...")
    B.set(0, 0, 0, 1.0)
    B.set(5, 5, 5, 42.0)
    B.set(9, 9, 9, 100.0)
    print(f"✓ Active cells: {B.face_ops()}")

    print("\nQuerying LLM at (5, 5, 5)...")
    llm = B.at(5, 5, 5)
    response = llm.p("Hello from Konomi Corduroy-OS!")
    print(f"✓ LLM response: {response}")

    print("\nBlock info:")
    info = B.info()
    print(f"  Capacity: {info['capacity']:,} cells")
    print(f"  Active: {info['active_cells']} cells")
    print(f"  Active LLMs: {info['active_llms']}")
    print(f"  Density: {info['density']:.6f}")
    print(f"  Memory: {info['memory_mb']:.2f} MB")

    # Cube Demo
    banner("🎲 Cube: 9-Node LLM Graph")
    print("Creating Cube...")
    C = K.cube("demo_cube")
    print(f"✓ Cube created: {C}")

    print("\nConnecting custom edge: NEU ↔ SWD")
    C.connect('NEU', 'SWD')
    print("✓ Edge added")

    print("\nSending message: NEU → central")
    msg = C.send_message('NEU', 'central', "Status check from NEU")
    print(f"✓ Response: {msg}")

    print("\nBroadcasting from central to all vertices...")
    responses = C.broadcast('central', "Ping from central")
    print(f"✓ Received {len(responses)} responses")

    print("\nFinding shortest path: NEU → SWD")
    path = C.shortest_path('NEU', 'SWD')
    print(f"✓ Path: {' → '.join(path)}")

    print("\nCube status:")
    status = C.status()
    print(f"  Nodes: {status['total_nodes']}")
    print(f"  Edges: {status['total_edges']}")
    print(f"  Messages sent: {status['messages_sent']}")

    # Cube visualization
    print("\nCube topology:")
    print(C.visualize_graph())

    # System Status
    banner("📊 System Status")
    system_status = K.status()

    print("eVGPU:")
    print(f"  Platform: {system_status['evgpu']['platform']}")
    print(f"  Devices: {system_status['evgpu']['device_count']}")

    print("\nBlockArrays:")
    print(f"  Count: {system_status['blocks']['count']}")
    print(f"  Total memory: {system_status['blocks']['total_memory_mb']:.2f} MB")
    print(f"  Names: {', '.join(system_status['blocks']['names'])}")

    print("\nCubes:")
    print(f"  Count: {system_status['cubes']['count']}")
    print(f"  Names: {', '.join(system_status['cubes']['names'])}")

    print("\nLLMs:")
    print(f"  Total instances: {system_status['llms']['total_instances']}")

    # Protocol Demo
    banner("📞 56k Dial-Up Protocol")
    from protocol.dialup import DialUpSession, OpCode

    print("Creating session...")
    session = DialUpSession()
    print(f"✓ Session initialized")

    print("\nCreating handshake packet...")
    handshake = session.handshake()
    print(f"✓ Opcode: {handshake.opcode.name}")
    print(f"✓ Payload: {handshake.payload[:30]}...")

    print("\nPacking packet to binary...")
    binary = handshake.pack()
    print(f"✓ Packet size: {len(binary)} bytes")

    # Tensor Serialization Demo
    banner("🔄 Tensor Serialization")
    from protocol.serializer import serialize_tensor, deserialize_tensor

    print("Serializing 4×4 tensor...")
    tensor = jnp.array([[1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16]], dtype=jnp.float32)

    serialized = serialize_tensor(tensor)
    print(f"✓ Serialized size: {len(serialized)} bytes")

    print("\nDeserializing...")
    restored = deserialize_tensor(serialized)
    print(f"✓ Restored shape: {restored.shape}")
    print(f"✓ Data preserved: {jnp.array_equal(tensor, restored)}")

    # XP10 Info
    banner("🪟 XP10 Custom Kernel")
    print("Status: Architecture design phase")
    print("Roadmap: 2-year development plan")
    print("\nFeatures (planned):")
    print("  ✓ Windows XP Luna theme")
    print("  ✓ Windows 10 security (ASLR, DEP, CFG)")
    print("  ✓ TLS 1.3 support")
    print("  ✓ Version spoofing: GetVersionEx → 10.0.19041")
    print("\nSee: xp10/docs/kernel_plan.md")

    # Final Message
    banner("✅ Demo Complete!")
    print("Konomi Corduroy-OS is fully operational.\n")
    print("Next steps:")
    print("  - Run tests: pytest tests/")
    print("  - Start server: python -m protocol.server")
    print("  - Open frontend: frontend/index.html")
    print("  - Docker: cd docker && docker-compose up -d")
    print("\n🎉 Happy hacking with Konomi Corduroy-OS!\n")


if __name__ == '__main__':
    main()

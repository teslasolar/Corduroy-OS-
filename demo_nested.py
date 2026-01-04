#!/usr/bin/env python3
"""
Konomi Corduroy-OS - Nested Architecture Demo
Demonstrates Cubes nested inside BlockArrays with spatial distribution
"""

import jax.numpy as jnp
from core.konomi import Konomi


def banner(text):
    """Print a fancy banner."""
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}\n")


def main():
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║     🪟 Konomi Corduroy-OS - Nested Architecture Demo           ║
    ║                                                                  ║
    ║        Cubes Nested in BlockArrays with Spatial Distribution    ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

    # Initialize Konomi
    banner("🎭 Initializing Konomi Orchestrator")
    K = Konomi()

    # Create a large BlockArray
    banner("🧊 Creating 100×100×100 BlockArray")
    B = K.block((100, 100, 100), name="spatial_grid")
    print(f"✓ Grid capacity: {B.metadata['capacity']:,} cells")
    print(f"✓ Dimensions: {B.dims}")

    # Nest Cubes at specific coordinates (with lots of space between them)
    banner("🎲 Nesting Cubes in 3D Space")

    print("Placing Cube #1 at (10, 10, 10)...")
    cube1 = B.cube_at(10, 10, 10)
    print(f"✓ Cube #1: {cube1.status()['total_nodes']} nodes")

    print("\nPlacing Cube #2 at (50, 50, 50)...")
    cube2 = B.cube_at(50, 50, 50)
    print(f"✓ Cube #2: {cube2.status()['total_nodes']} nodes")

    print("\nPlacing Cube #3 at (90, 90, 90)...")
    cube3 = B.cube_at(90, 90, 90)
    print(f"✓ Cube #3: {cube3.status()['total_nodes']} nodes")

    print(f"\n✓ Total Cubes nested: {B.metadata['active_cubes']}")
    print(f"✓ Total LLM nodes: {B.metadata['active_cubes'] * 9}")

    # Store data in the space between Cubes
    banner("📦 Storing Data Between Cubes")

    print("Storing values in the grid...")
    # Store data around Cube #1
    for i in range(5):
        for j in range(5):
            B.set(10 + i, 10 + j, 10, float(i + j))

    # Store data in mid-region (between Cube #1 and #2)
    for i in range(30, 35):
        B.set(i, i, i, float(i))

    # Store data around Cube #3
    for i in range(5):
        B.set(90 + i, 90, 90, float(i * 10))

    print(f"✓ Active storage cells: {B.face_ops()}")
    print(f"✓ Grid density: {B.density():.6f}")

    # Demonstrate spatial queries
    banner("🔍 Spatial Queries")

    print("Query region around Cube #1 (5-15, 5-15, 5-15)...")
    region1 = B.query_region((5, 15), (5, 15), (5, 15))
    print(f"✓ Found {len(region1)} data points")

    print("\nQuery region around Cube #2 (45-55, 45-55, 45-55)...")
    region2 = B.query_region((45, 55), (45, 55), (45, 55))
    print(f"✓ Found {len(region2)} data points")

    # Demonstrate Cube operations within BlockArray
    banner("🎯 Cube Operations in 3D Space")

    print("Connecting vertices in Cube #1...")
    cube1.connect('NEU', 'SWD')
    print("✓ Edge added: NEU ↔ SWD")

    print("\nSending message through Cube #1...")
    msg = cube1.send_message('NEU', 'central', "Message from spatial coord (10,10,10)")
    print(f"✓ Response: {msg}")

    print("\nSending message through Cube #3...")
    msg = cube3.send_message('central', 'NEU', "Message from spatial coord (90,90,90)")
    print(f"✓ Response: {msg}")

    # Broadcast to all Cubes
    banner("📡 Broadcasting to All Nested Cubes")

    print("Broadcasting message to all Cubes in grid...")
    responses = B.broadcast_to_cubes("System-wide ping", vertex='central')

    print(f"✓ Responses from {len(responses)} Cubes:")
    for coord, response in responses.items():
        print(f"  Cube at {coord}: {response}")

    # Demonstrate mixing LLMs and Cubes
    banner("🧠 Mixing Individual LLMs with Cubes")

    print("Placing individual Femto LLMs between Cubes...")

    # Place LLMs in the space between Cube #1 and #2
    llm1 = B.at(25, 25, 25)
    llm2 = B.at(30, 30, 30)
    llm3 = B.at(35, 35, 35)

    print(f"✓ LLM at (25, 25, 25): {llm1}")
    print(f"✓ LLM at (30, 30, 30): {llm2}")
    print(f"✓ LLM at (35, 35, 35): {llm3}")

    print("\nQuerying individual LLMs...")
    r1 = llm1.p("Process data at waypoint 25")
    r2 = llm2.p("Process data at waypoint 30")
    r3 = llm3.p("Process data at waypoint 35")

    print(f"✓ LLM #1: {r1}")
    print(f"✓ LLM #2: {r2}")
    print(f"✓ LLM #3: {r3}")

    # Memory usage analysis
    banner("💾 Memory Usage Analysis")

    mem = B.memory_estimate()

    print(f"Storage cells: {mem['cells_mb']:.2f} MB")
    print(f"Individual LLMs: {mem['llms_mb']:.2f} MB")
    print(f"Nested Cubes: {mem['cubes_mb']:.2f} MB")
    print(f"Total: {mem['total_mb']:.2f} MB")

    # Final system info
    banner("📊 Final System State")

    info = B.info()

    print(f"Grid dimensions: {info['dims']}")
    print(f"Grid capacity: {info['capacity']:,} cells")
    print(f"Active storage cells: {info['active_cells']}")
    print(f"Individual LLMs: {info['active_llms']}")
    print(f"Nested Cubes: {info['active_cubes']}")
    print(f"Total LLM nodes: {info['active_llms'] + (info['active_cubes'] * 9)}")
    print(f"Density: {info['density']:.6f}")
    print(f"Memory: {info['memory_mb']:.2f} MB")

    # Visualization
    banner("🗺️  Spatial Distribution Map")

    print("""
    100³ BlockArray with nested Cubes:

    (0,0,0) ─────────────────────────────────────────────────────── (100,100,100)
       │                                                                │
       │   🎲 Cube #1 @ (10,10,10)                                    │
       │      ├─ 9 LLM nodes                                          │
       │      └─ Data storage around: ████                            │
       │                                                               │
       │      🧠 Individual LLMs @ (25-35, 25-35, 25-35)             │
       │         └─ Waypoint processors                               │
       │                                                               │
       │         🎲 Cube #2 @ (50,50,50)                              │
       │            ├─ 9 LLM nodes                                    │
       │            └─ Hub node                                       │
       │                                                               │
       │                                                               │
       │                   🎲 Cube #3 @ (90,90,90)                    │
       │                      ├─ 9 LLM nodes                          │
       │                      └─ Data storage around: ████            │
       │                                                               │
       └───────────────────────────────────────────────────────────────┘

    Total Resources:
    - 3 Cubes (27 nodes) at strategic coordinates
    - 3 Individual LLMs at waypoints
    - {B.face_ops()} storage cells with data
    - Lots of empty space for future expansion
    """)

    # Architecture benefits
    banner("✨ Architecture Benefits")

    print("""
    ✓ Spatial Distribution:
      - Cubes are distributed across 3D space
      - Lots of room between them for data storage
      - Can store local context near each agent cluster

    ✓ Hierarchical Processing:
      - Individual Femto LLMs for lightweight tasks
      - 9-node Cubes for complex distributed processing
      - Mix and match based on needs

    ✓ Scalability:
      - 1000³ grid can hold thousands of Cubes
      - Sparse storage keeps memory low
      - Only pay for what you use

    ✓ Spatial Queries:
      - Query data by region
      - Broadcast to agents in specific areas
      - Efficient coordinate-based access

    ✓ Future Potential:
      - Connect Cubes across space (inter-Cube messaging)
      - Implement spatial routing algorithms
      - Add physics simulations between agents
      - Create hierarchies (Cubes containing Cubes)
    """)

    banner("🎉 Nested Architecture Demo Complete!")
    print("Konomi Corduroy-OS: Where every coordinate can be an agent.\n")


if __name__ == '__main__':
    main()

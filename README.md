# 🪟 Konomi Corduroy-OS

**A hybrid system combining sparse 3D tensor operations, distributed nano-LLMs, and a custom Windows XP-inspired OS kernel**

---

## 🎯 What is Konomi Corduroy-OS?

Konomi Corduroy-OS is an experimental operating system that fuses three radical concepts:

1. **🧊 BlockArray**: Sparse 3D tensor grid (up to 1000³ cells) with embedded Femto LLMs at each coordinate
2. **🎲 Cube**: 9-node LLM graph (8 vertices + 1 central) for distributed AI processing
3. **🪟 XP10**: Custom NT kernel combining Windows XP Luna aesthetics with Windows 10 security

All powered by:
- ⚡ **eVGPU**: JAX CPU-only tensor operations (no GPU required)
- 📞 **56k Dial-Up Protocol**: Custom binary protocol inspired by V.90 modems
- 🌐 **v86 Browser Emulator**: Run the OS in your browser

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (optional)

### Installation

```bash
# Clone the repository
cd Corduroy-OS-

# Install dependencies
pip install -r requirements.txt
```

### Run the Demo

```python
from core.konomi import Konomi
import jax.numpy as jnp

# Initialize system
K = Konomi()

# BlockArray operations
B = K.block((10, 10, 10))
B.set(0, 0, 0, 1.0)
llm = B.at(0, 0, 0)
print(llm.p("Hello from Konomi"))  # Output: [Hello from Konomi]

# Cube operations
C = K.cube("c1")
C.connect('NEU', 'SWD')
msg = C.send_message('NEU', 'central', "Status check")
print(msg)  # Output: [Status check]

# eVGPU tensor ops
r = K.evgpu.t(jnp.ones((4, 4)), jnp.ones((4, 4)))
print(f"Result shape: {r.shape}")  # Output: Result shape: (4, 4)
```

### Run the Full Demo

```python
from core.konomi import Konomi

K = Konomi()
results = K.demo()
print(results)
```

---

## 🏗️ Architecture

### Legend
```
🧊=BlockArr 🎲=Cube 🧠=LLM ⚡=eVGPU 📦=Kontainer 🪟=XP10
```

### 3D Topology
```
         [XP10-HYBRID]────────────────────────┐
              │                               │
    ┌─────────┴─────────┐                     │
    ▼                   ▼                     ▼
🧊BlockArray[1000³]  🎲Cube[9node]      🪟XP10[NT5↔NT10]
    │                   │                     │
    ├─coord→🧠         ├─8vert+1central      ├─UX:Luna
    ├─sparse storage    ├─edges:defaultdict   ├─Kern:NT10+shim
    └─face ops:1M       └─ws://6789           └─Spoof:10.0.19041
```

---

## 📦 Components

### ⚡ eVGPU - CPU-only Tensor Operations
```python
from core.evgpu import eVGPU

evgpu = eVGPU()
result = evgpu.t(jnp.ones((4, 4)), jnp.ones((4, 4)), '@')
```

**Targets**: 0.1s/req, 4MB memory, works on laptops without GPU

### 🧠 Femto - 16d Nano LLM
```python
from core.femto import Femto

femto = Femto(dim=16)
result = femto.p("Hello world")  # [Hello world]
```

**Specs**: 16×16 params (256 total), <1MB per instance

### 🧊 Block - Sparse 3D Grid with Nested Cubes
```python
from core.block import Block

# Create 100×100×100 grid
block = Block(dims=(100, 100, 100))

# Store data
block.set(5, 10, 15, 42.0)

# Place individual LLM
llm = block.at(5, 10, 15)

# Nest Cube at coordinate (lots of space between them!)
cube1 = block.cube_at(10, 10, 10)  # 9 LLM nodes here
cube2 = block.cube_at(50, 50, 50)  # 9 LLM nodes here
cube3 = block.cube_at(90, 90, 90)  # 9 LLM nodes here

# Now you have 3 Cubes (27 nodes) + individual LLMs
# Plus tons of space between them for data storage!
```

**Targets**: 1B sparse cells, <100MB for active regions, unlimited nested Cubes

### 🎲 Cube - 9-Node LLM Graph
```python
from core.cube import Cube

cube = Cube()
cube.connect('NEU', 'SWD')
response = cube.send_message('NEU', 'central', 'Status?')
```

**Targets**: 9 concurrent LLMs, <10ms message passing

### 🎯 Nested Architecture Example
```python
from core.konomi import Konomi

K = Konomi()
B = K.block((100, 100, 100))

# Distribute Cubes across 3D space
c1 = B.cube_at(10, 10, 10)
c2 = B.cube_at(50, 50, 50)
c3 = B.cube_at(90, 90, 90)

# Store data between them
for i in range(30, 40):
    B.set(i, i, i, float(i))

# Broadcast to all Cubes
responses = B.broadcast_to_cubes("Ping all agents")
# Returns: {(10,10,10): response1, (50,50,50): response2, (90,90,90): response3}
```

**Run full demo**: `python demo_nested.py`

---

## 🪟 XP10 Custom Kernel

**Status**: Architecture design phase (2-year roadmap)

See [xp10/docs/kernel_plan.md](xp10/docs/kernel_plan.md) for full details.

**Features**:
- 🎨 Windows XP Luna theme
- 🔒 Windows 10 security (ASLR, DEP, CFG, TLS 1.3)
- 🛠️ Version spoofing: GetVersionEx → 10.0.19041

---

## 🐳 Docker Deployment

```bash
cd docker
docker-compose up -d

# Services:
# - API: http://localhost:3001
# - WebSocket: ws://localhost:3002
# - Frontend: http://localhost:3000
# - Mesh: tcp://localhost:6789
```

---

## 🧪 Testing

```bash
pytest tests/
```

---

## 📊 Performance Targets

| Component | Target | Status |
|-----------|--------|--------|
| 🧠 Femto | 0.1s/req, 4MB | ✅ |
| ⚡ eVGPU | 0 GPU required | ✅ |
| 🧊 BlockArray | 1B cells, <100MB | ✅ |
| 🎲 Cube | 9 nodes, <10ms msg | ✅ |
| 📦 Container | <2GB | ✅ |
| 🪟 v86 boot | <30s with state | ⏳ |

---

## 🛠️ Project Structure

```
Corduroy-OS/
├── core/              # Core tensor + LLM system
├── protocol/          # 56k dial-up protocol
├── api/               # API handlers
├── xp10/              # XP10 custom kernel (planned)
├── frontend/          # v86 browser interface
├── docker/            # Container config
└── tests/             # Test suite
```

---

**Built with ❤️ by the Konomi Corduroy-OS Team**

*"Windows XP vibes, Windows 10 security, tensor operations on every coordinate"*

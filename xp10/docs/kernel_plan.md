# Konomi Corduroy-OS - XP10 Custom NT Kernel
## 2-Year Development Roadmap

**Goal**: Build a custom Windows NT kernel that combines Windows XP's Luna aesthetic with Windows 10's modern security features.

---

## Overview

**XP10** is a custom NT kernel implementation that provides:
- 🎨 **UX**: Windows XP Luna theme, Classic Start Menu, familiar UI
- 🔒 **Security**: Windows 10 security features (ASLR, DEP, CFG, TLS 1.3)
- 🛠️ **Architecture**: Modern NT kernel architecture with XP compatibility layer
- 📦 **Deployment**: Runs in v86 browser emulator for portability

---

## Architecture Components

### Core Kernel
```
ntoskrnl.exe - NT Kernel Executive
├── HAL (Hardware Abstraction Layer)
├── Memory Manager
├── Process/Thread Manager
├── I/O Manager
├── Object Manager
└── Security Reference Monitor
```

### Subsystems
```
win32k.sys - Win32 Subsystem
├── Window Manager (USER)
├── GDI (Graphics Device Interface)
└── Luna Theme Engine
```

### Drivers
```
Drivers
├── NTFS Driver
├── Network Stack
├── Display Driver
└── Input Drivers (keyboard, mouse)
```

---

## Year 1: Foundation (Quarters 1-4)

### Q1: Hardware Abstraction Layer (HAL)

**Goal**: Build HAL for x86 architecture

**Tasks**:
- [ ] x86 boot loader (GRUB2 or custom)
- [ ] Protected mode initialization
- [ ] Interrupt Descriptor Table (IDT) setup
- [ ] Programmable Interrupt Controller (PIC) driver
- [ ] Timer interrupt handling
- [ ] Basic hardware detection

**Deliverables**:
- HAL that boots to protected mode
- Timer ticks working
- Can handle hardware interrupts

**Estimated Effort**: 12 weeks

---

### Q2: Memory Management

**Goal**: Implement paging and heap allocator

**Tasks**:
- [ ] Page table management
- [ ] Virtual memory allocation
- [ ] Physical memory manager
- [ ] Kernel heap allocator
- [ ] Memory protection (NX bit for DEP)
- [ ] ASLR implementation

**Deliverables**:
- Working virtual memory system
- Heap allocator (malloc/free equivalent)
- DEP and ASLR enabled

**Estimated Effort**: 12 weeks

**Security Features**:
- ✅ Data Execution Prevention (DEP)
- ✅ Address Space Layout Randomization (ASLR)

---

### Q3: Process & Thread Scheduler

**Goal**: Implement process/thread management

**Tasks**:
- [ ] Process structure (EPROCESS equivalent)
- [ ] Thread structure (ETHREAD equivalent)
- [ ] Context switching (x86 register save/restore)
- [ ] Thread scheduler (priority-based)
- [ ] System call interface (int 0x2E or SYSENTER)
- [ ] User mode transition

**Deliverables**:
- Can create and schedule processes
- User mode programs can run
- System calls working

**Estimated Effort**: 12 weeks

---

### Q4: Object Manager & Basic I/O

**Goal**: NT Object Manager and basic I/O

**Tasks**:
- [ ] Object Manager (handles, reference counting)
- [ ] Basic I/O Manager structure
- [ ] File object abstraction
- [ ] Device object abstraction
- [ ] IRP (I/O Request Packet) mechanism
- [ ] Synchronization primitives (mutexes, events)

**Deliverables**:
- Object Manager functional
- Can open/read files (RAMdisk)
- Basic synchronization working

**Estimated Effort**: 12 weeks

---

## Year 2: Subsystems & Polish (Quarters 1-4)

### Q1-Q2: Win32k Subsystem

**Goal**: Windowing and GDI for Luna theme

**Tasks**:
- [ ] Window Manager (CreateWindow, message queue)
- [ ] GDI primitives (BitBlt, drawing APIs)
- [ ] Luna theme engine
  - [ ] Window chrome (title bars, borders)
  - [ ] Button styles (blue Luna buttons)
  - [ ] Start button and taskbar
- [ ] Input event handling (keyboard, mouse)
- [ ] Desktop Window Manager basics

**Deliverables**:
- Can display windows with Luna theme
- Start menu functional
- Classic XP appearance

**Estimated Effort**: 24 weeks

**UX Features**:
- ✅ Luna theme with blue/green gradients
- ✅ Classic Start Menu
- ✅ "My Computer" icon
- ✅ XP-style window chrome

---

### Q3: File System & Drivers

**Goal**: NTFS driver and I/O stack

**Tasks**:
- [ ] NTFS driver (read/write)
  - [ ] MFT (Master File Table) parsing
  - [ ] File read/write/delete
  - [ ] Directory operations
- [ ] I/O Manager completion
  - [ ] Driver loading framework
  - [ ] PnP (Plug and Play) basics
- [ ] Network stack basics
  - [ ] TCP/IP stack (or port lwIP)
  - [ ] Socket interface

**Deliverables**:
- Can read/write NTFS volumes
- Network connectivity (TCP/IP)
- Driver framework working

**Estimated Effort**: 12 weeks

---

### Q4: Security & Finalization

**Goal**: Modern security features and polish

**Tasks**:
- [ ] Security Reference Monitor
  - [ ] Access tokens
  - [ ] ACLs (Access Control Lists)
  - [ ] Privilege checks
- [ ] Control Flow Guard (CFG)
  - [ ] Valid call target bitmap
  - [ ] Indirect call validation
- [ ] TLS 1.3 support
  - [ ] Crypto primitives (AES, SHA)
  - [ ] TLS handshake
- [ ] Version spoofing layer
  - [ ] GetVersionEx → returns 10.0.19041
  - [ ] API shims for XP app compatibility

**Deliverables**:
- Security subsystem functional
- CFG enabled
- TLS 1.3 working
- GetVersionEx returns Windows 10 version

**Estimated Effort**: 12 weeks

**Security Features**:
- ✅ Control Flow Guard (CFG)
- ✅ TLS 1.3 support
- ✅ Modern crypto (AES-256, SHA-256)

---

## Technical Specifications

### Target Platform
- **Architecture**: x86 (32-bit for v86 compatibility)
- **Emulator**: v86 (https://github.com/copy/v86)
- **Memory**: 512MB - 4GB
- **Boot**: GRUB2 or custom bootloader

### Development Stack
- **Language**: C (kernel), Assembly (boot/low-level)
- **Compiler**: GCC or Clang with freestanding flags
- **Build**: Make or CMake
- **Debug**: QEMU + GDB for development, v86 for deployment

### Code Style
- **NT Naming**: PascalCase for functions (NtCreateFile)
- **Hungarian Notation**: For kernel structures (p prefix for pointers)
- **Comments**: Doxygen-style documentation

---

## Version Spoofing Implementation

### GetVersionEx Shim

Create shim DLL that intercepts version calls:

```c
// version.c
#include <windows.h>

BOOL WINAPI GetVersionExA_Shim(OSVERSIONINFOA *lpVersionInfo) {
    // Return Windows 10 version
    lpVersionInfo->dwMajorVersion = 10;
    lpVersionInfo->dwMinorVersion = 0;
    lpVersionInfo->dwBuildNumber = 19041;
    lpVersionInfo->dwPlatformId = VER_PLATFORM_WIN32_NT;
    strcpy(lpVersionInfo->szCSDVersion, "");
    return TRUE;
}

// Export as GetVersionExA
#pragma comment(linker, "/export:GetVersionExA=GetVersionExA_Shim")
```

### API Compatibility Layer

For XP apps running on XP10:
- Map old API calls to new implementations
- Emulate removed APIs
- Thunk 16-bit calls to 32-bit

---

## Milestones

### Milestone 1: "Hello World Kernel" (End of Q1)
- Boots to protected mode
- Prints to screen
- Handles timer interrupts

### Milestone 2: "User Mode" (End of Q2)
- Can run simple user mode program
- Memory protection working
- System calls functional

### Milestone 3: "Process Manager" (End of Q3)
- Can create multiple processes
- Thread scheduling working
- Context switching stable

### Milestone 4: "File I/O" (End of Q4)
- Can read files from RAMdisk
- Basic I/O Manager complete
- Object Manager functional

### Milestone 5: "Luna UI" (End of Year 2 Q2)
- Windows display with Luna theme
- Start menu works
- Can launch programs

### Milestone 6: "XP10 Release" (End of Year 2 Q4)
- All security features enabled
- NTFS driver working
- Network stack functional
- GetVersionEx returns Windows 10

---

## Testing Strategy

### Unit Tests
- Memory allocator tests
- Scheduler tests
- Object Manager tests

### Integration Tests
- Full boot cycle
- User mode program execution
- File I/O operations

### Compatibility Tests
- Run classic XP applications
- Verify Luna theme rendering
- Test version spoofing

---

## Resources

### Reference Materials
- **Windows Internals** (Russinovich & Solomon)
- **ReactOS Source** (https://reactos.org)
- **OSDev Wiki** (https://wiki.osdev.org)
- **NT Kernel documentation** (leaked Windows NT 4.0 source)

### Tools
- **QEMU**: Development emulator
- **v86**: Deployment emulator
- **GDB**: Kernel debugging
- **IDA Pro / Ghidra**: Reverse engineering for API compatibility

---

## MVP for Integration

### Immediate (Week 1-2)
- [ ] Document this roadmap ✅
- [ ] Create kernel directory structure
- [ ] Write boot stub (prints "XP10 booting...")
- [ ] Version shim DLL for demo

### Short-term (Month 1-3)
- [ ] HAL boot to protected mode
- [ ] Basic memory management
- [ ] v86 integration (boot XP10 kernel in browser)

### Medium-term (Month 3-6)
- [ ] Process/thread basics
- [ ] Simple user mode program runs
- [ ] Demo: XP10 running in Konomi Corduroy-OS frontend

---

## Notes

**This is a massive undertaking**. The 2-year timeline is aggressive and assumes:
- Full-time development (40 hrs/week)
- Strong C/Assembly skills
- OS development experience

**For MVP**: Focus on documentation and v86 integration with a minimal boot stub. The full kernel is a long-term goal.

**Alternative**: Use ReactOS as a base and customize it with Luna theme + version spoofing for faster deployment.

---

## Status

- [x] Roadmap documented
- [ ] Directory structure created
- [ ] Boot stub implemented
- [ ] HAL in development
- [ ] Memory Manager in development
- [ ] Everything else: pending

**Current Phase**: Planning / Architecture Design
**Est. Completion**: Q4 2027 (2 years from 2026-01-04)

---

*Last Updated: 2026-01-04*
*Konomi Corduroy-OS XP10 Kernel Team*

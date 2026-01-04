# XP10 Custom NT Kernel

**Status**: Architecture design phase

This directory will contain the custom NT kernel implementation for Konomi Corduroy-OS.

## Directory Structure (Planned)

```
kernel/
├── hal/              # Hardware Abstraction Layer
│   ├── boot/        # Boot loader
│   ├── interrupts/  # Interrupt handling
│   └── timer/       # Timer drivers
├── ntoskrnl/        # NT Kernel Executive
│   ├── mm/          # Memory Manager
│   ├── ps/          # Process/Thread Manager
│   ├── io/          # I/O Manager
│   ├── ob/          # Object Manager
│   └── se/          # Security Reference Monitor
└── win32k/          # Win32 Subsystem
    ├── user/        # Window Manager
    ├── gdi/         # Graphics Device Interface
    └── themes/      # Luna theme engine
```

## Architecture

See [kernel_plan.md](../docs/kernel_plan.md) for the full 2-year development roadmap.

### Key Features

- **XP Feel**: Luna theme, Classic Start Menu, familiar Windows XP UI
- **Win10 Security**: ASLR, DEP, CFG, TLS 1.3
- **Version Spoofing**: GetVersionEx returns Windows 10 (10.0.19041)
- **v86 Compatible**: Runs in browser via v86 emulator

## Current Status

This is currently **architecture documentation only**. The actual kernel implementation is a 2-year project.

### Immediate Next Steps

1. Create boot stub that prints "XP10 Booting..."
2. Set up build system (Makefile)
3. Integrate with v86 emulator
4. Implement HAL for x86 protected mode

## References

- **Windows Internals** by Russinovich & Solomon
- **ReactOS** (open-source NT kernel): https://reactos.org
- **OSDev Wiki**: https://wiki.osdev.org

## For Contributors

If you want to contribute to XP10 kernel development:

1. Read [kernel_plan.md](../docs/kernel_plan.md)
2. Familiarity with C, x86 Assembly, OS development required
3. Start with HAL boot stub (Q1 milestone)

---

*Konomi Corduroy-OS - XP10 Kernel Project*

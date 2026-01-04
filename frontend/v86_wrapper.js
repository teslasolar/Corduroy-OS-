/**
 * Konomi Corduroy-OS - v86 Wrapper
 * v86 emulator initialization and serial I/O
 *
 * Note: This is a stub. Full v86 integration requires:
 * 1. Download v86 library (https://github.com/copy/v86)
 * 2. Prepare disk image with XP10/Luna theme
 * 3. Configure serial port for agent communication
 */

class V86Wrapper {
    constructor(screenElement) {
        this.screenElement = screenElement;
        this.emulator = null;
        this.serialBuffer = '';
    }

    async boot(options = {}) {
        console.log('[v86] Booting virtual machine...');

        // TODO: Initialize v86 emulator
        // This requires:
        // 1. v86 library loaded
        // 2. BIOS files
        // 3. Disk image (Windows XP/10 with Luna theme)

        /*
        this.emulator = new V86Starter({
            screen_container: this.screenElement,
            bios: { url: "bios/seabios.bin" },
            vga_bios: { url: "bios/vgabios.bin" },
            hda: { url: "images/xp10-luna.img", async: true },
            memory_size: 512 * 1024 * 1024, // 512MB
            vga_memory_size: 8 * 1024 * 1024, // 8MB
            boot_order: 0x132, // Hard disk

            // Serial port configuration for agent I/O
            serial_container_xtermjs: true,

            // State save/restore
            initial_state: options.state,

            autostart: true
        });

        // Set up serial communication
        this.emulator.add_listener("serial0-output-char", (char) => {
            this.onSerialOutput(char);
        });
        */

        console.log('[v86] Boot stub - full implementation pending');
        console.log('[v86] To implement:');
        console.log('  1. Download v86: https://github.com/copy/v86');
        console.log('  2. Create XP10 disk image with Luna theme');
        console.log('  3. Configure serial port for agent communication');

        return Promise.resolve();
    }

    onSerialOutput(char) {
        // Accumulate serial output
        this.serialBuffer += char;

        // Look for complete commands/messages
        if (char === '\n') {
            console.log('[v86] Serial output:', this.serialBuffer);

            // Parse and handle commands from OS instance
            this.handleSerialCommand(this.serialBuffer.trim());

            this.serialBuffer = '';
        }
    }

    handleSerialCommand(command) {
        console.log('[v86] Command from OS:', command);

        // Commands could be:
        // - STATUS: Get system status
        // - QUERY: Query LLM
        // - TENSOR: Tensor operation
        // etc.

        // Forward to backend via WebSocket
        if (window.konomiClient && window.konomiClient.connected) {
            konomiClient.send({
                type: 'serial',
                command: command
            });
        }
    }

    sendToSerial(data) {
        if (this.emulator) {
            for (let char of data) {
                this.emulator.serial0_send(char);
            }
        } else {
            console.log('[v86] Emulator not running, cannot send:', data);
        }
    }

    saveState() {
        if (this.emulator) {
            return new Promise((resolve) => {
                this.emulator.save_state((error, state) => {
                    if (error) {
                        console.error('[v86] Save state error:', error);
                        resolve(null);
                    } else {
                        console.log('[v86] State saved:', state.byteLength, 'bytes');
                        resolve(state);
                    }
                });
            });
        }
        return Promise.resolve(null);
    }

    async restoreState(state) {
        if (this.emulator && state) {
            await this.emulator.restore_state(state);
            console.log('[v86] State restored');
        }
    }

    stop() {
        if (this.emulator) {
            this.emulator.stop();
            console.log('[v86] Emulator stopped');
        }
    }
}

// Create global v86 wrapper
const v86wrapper = new V86Wrapper(document.getElementById('v86-screen'));

// Boot function (called from UI)
async function bootV86() {
    await v86wrapper.boot();
}

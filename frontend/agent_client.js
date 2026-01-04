/**
 * Konomi Corduroy-OS - Agent Client
 * WebSocket client for backend communication
 */

class KonomiClient {
    constructor(url = 'ws://localhost:3002') {
        this.url = url;
        this.ws = null;
        this.connected = false;
        this.messageQueue = [];
    }

    connect() {
        console.log(`[Konomi] Connecting to ${this.url}...`);

        this.ws = new WebSocket(this.url);

        this.ws.onopen = () => {
            console.log('[Konomi] Connected to backend');
            this.connected = true;
            this.onConnect();

            // Send queued messages
            while (this.messageQueue.length > 0) {
                const msg = this.messageQueue.shift();
                this.send(msg);
            }
        };

        this.ws.onmessage = (event) => {
            console.log('[Konomi] Message received:', event.data);
            this.onMessage(event.data);
        };

        this.ws.onerror = (error) => {
            console.error('[Konomi] WebSocket error:', error);
            this.onError(error);
        };

        this.ws.onclose = () => {
            console.log('[Konomi] Disconnected from backend');
            this.connected = false;
            this.onDisconnect();

            // Auto-reconnect after 5 seconds
            setTimeout(() => this.connect(), 5000);
        };
    }

    send(message) {
        if (this.connected && this.ws.readyState === WebSocket.OPEN) {
            const data = typeof message === 'string' ? message : JSON.stringify(message);
            this.ws.send(data);
            console.log('[Konomi] Sent:', data);
        } else {
            console.log('[Konomi] Not connected, queueing message');
            this.messageQueue.push(message);
        }
    }

    // Cube operations
    initCube(name = 'default') {
        this.send({
            action: 'init',
            name: name
        });
    }

    processVertex(cube, vertex, data) {
        this.send({
            action: 'process',
            cube: cube,
            vertex: vertex,
            data: data
        });
    }

    connectVertices(cube, from, to) {
        this.send({
            action: 'connect',
            cube: cube,
            from: from,
            to: to
        });
    }

    cubeStatus(cube) {
        this.send({
            action: 'status',
            cube: cube
        });
    }

    // Hooks for UI integration
    onConnect() {
        console.log('[Konomi] Connection established');
    }

    onDisconnect() {
        console.log('[Konomi] Connection closed');
    }

    onMessage(data) {
        console.log('[Konomi] Processing message:', data);
    }

    onError(error) {
        console.error('[Konomi] Error:', error);
    }

    disconnect() {
        if (this.ws) {
            this.ws.close();
        }
    }
}

// Create global client instance
const konomiClient = new KonomiClient();

// Auto-connect when page loads
window.addEventListener('DOMContentLoaded', () => {
    // Uncomment to auto-connect on load
    // konomiClient.connect();
});

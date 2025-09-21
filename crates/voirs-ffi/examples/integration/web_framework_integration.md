# Web Framework Integration Examples

This document provides comprehensive examples for integrating VoiRS FFI with popular web frameworks for server-side voice synthesis, real-time audio processing, and web applications.

## Table of Contents

1. [Express.js (Node.js) Integration](#expressjs-nodejs-integration)
2. [FastAPI (Python) Integration](#fastapi-python-integration)
3. [Flask (Python) Integration](#flask-python-integration)
4. [Django (Python) Integration](#django-python-integration)
5. [Actix Web (Rust) Integration](#actix-web-rust-integration)
6. [WebAssembly Frontend Integration](#webassembly-frontend-integration)

## Express.js (Node.js) Integration

### Basic REST API Server

#### package.json

```json
{
  "name": "voirs-express-server",
  "version": "1.0.0",
  "description": "VoiRS FFI Express.js integration server",
  "main": "server.js",
  "scripts": {
    "start": "node server.js",
    "dev": "nodemon server.js",
    "test": "jest"
  },
  "dependencies": {
    "express": "^4.18.2",
    "multer": "^1.4.5",
    "cors": "^2.8.5",
    "helmet": "^7.0.0",
    "compression": "^1.7.4",
    "rate-limiter-flexible": "^3.0.0",
    "ws": "^8.13.0",
    "uuid": "^9.0.0",
    "voirs-ffi": "^0.1.0"
  },
  "devDependencies": {
    "nodemon": "^3.0.1",
    "jest": "^29.5.0"
  }
}
```

#### Express Server (server.js)

```javascript
const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const compression = require('compression');
const { RateLimiterMemory } = require('rate-limiter-flexible');
const WebSocket = require('ws');
const { v4: uuidv4 } = require('uuid');
const VoirsFFI = require('voirs-ffi');

class VoiRSServer {
    constructor() {
        this.app = express();
        this.server = null;
        this.wss = null;
        this.voirsEngine = null;
        this.activeSynthesis = new Map();
        
        this.setupMiddleware();
        this.setupRoutes();
        this.setupWebSocket();
        this.setupErrorHandling();
    }
    
    setupMiddleware() {
        // Security middleware
        this.app.use(helmet());
        this.app.use(cors({
            origin: process.env.ALLOWED_ORIGINS?.split(',') || ['http://localhost:3000'],
            credentials: true
        }));
        
        // Performance middleware
        this.app.use(compression());
        this.app.use(express.json({ limit: '10mb' }));
        this.app.use(express.urlencoded({ extended: true }));
        
        // Rate limiting
        this.rateLimiter = new RateLimiterMemory({
            points: 100, // Number of requests
            duration: 60, // Per 60 seconds
        });
        
        this.app.use(async (req, res, next) => {
            try {
                await this.rateLimiter.consume(req.ip);
                next();
            } catch (rateLimiterRes) {
                res.status(429).json({
                    error: 'Rate limit exceeded',
                    retryAfter: rateLimiterRes.msBeforeNext
                });
            }
        });
    }
    
    setupRoutes() {
        // Health check
        this.app.get('/health', (req, res) => {
            res.json({
                status: 'healthy',
                timestamp: new Date().toISOString(),
                voirs_initialized: this.voirsEngine?.isInitialized() || false
            });
        });
        
        // Get available voices
        this.app.get('/api/voices', (req, res) => {
            try {
                const voices = VoirsFFI.getAvailableVoices();
                res.json({ voices });
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });
        
        // Single text synthesis
        this.app.post('/api/synthesize', async (req, res) => {
            try {
                const { text, voice_id, quality, speed, format } = req.body;
                
                if (!text || text.trim().length === 0) {
                    return res.status(400).json({ error: 'Text is required' });
                }
                
                if (text.length > 10000) {
                    return res.status(400).json({ 
                        error: 'Text too long (max 10,000 characters)' 
                    });
                }
                
                const config = {
                    quality: quality || VoirsFFI.Quality.HIGH,
                    speed: speed || 1.0,
                    voice_id: voice_id || 'default',
                    outputFormat: format || VoirsFFI.Format.MP3
                };
                
                const result = await this.voirsEngine.synthesize(text, config);
                
                if (result.success) {
                    res.set({
                        'Content-Type': this.getContentType(config.outputFormat),
                        'Content-Length': result.audioData.length,
                        'X-Synthesis-Duration': result.duration,
                        'X-Audio-Length': result.audioLength
                    });
                    res.send(result.audioData);
                } else {
                    res.status(500).json({ error: 'Synthesis failed' });
                }
            } catch (error) {
                console.error('Synthesis error:', error);
                res.status(500).json({ error: error.message });
            }
        });
        
        // Batch synthesis
        this.app.post('/api/synthesize/batch', async (req, res) => {
            try {
                const { texts, config } = req.body;
                
                if (!Array.isArray(texts) || texts.length === 0) {
                    return res.status(400).json({ error: 'Texts array is required' });
                }
                
                if (texts.length > 50) {
                    return res.status(400).json({ 
                        error: 'Too many texts (max 50)' 
                    });
                }
                
                const batchId = uuidv4();
                this.activeSynthesis.set(batchId, {
                    status: 'processing',
                    total: texts.length,
                    completed: 0,
                    results: []
                });
                
                // Start async processing
                this.processBatch(batchId, texts, config || {});
                
                res.json({ 
                    batchId,
                    status: 'processing',
                    total: texts.length
                });
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });
        
        // Check batch status
        this.app.get('/api/synthesize/batch/:batchId', (req, res) => {
            const { batchId } = req.params;
            const batch = this.activeSynthesis.get(batchId);
            
            if (!batch) {
                return res.status(404).json({ error: 'Batch not found' });
            }
            
            res.json({
                batchId,
                status: batch.status,
                progress: {
                    completed: batch.completed,
                    total: batch.total,
                    percentage: Math.round((batch.completed / batch.total) * 100)
                },
                results: batch.status === 'completed' ? batch.results : undefined
            });
        });
        
        // Stream synthesis (Server-Sent Events)
        this.app.get('/api/synthesize/stream', (req, res) => {
            const { text, voice_id, quality } = req.query;
            
            if (!text) {
                return res.status(400).json({ error: 'Text query parameter is required' });
            }
            
            res.writeHead(200, {
                'Content-Type': 'text/event-stream',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*'
            });
            
            this.streamSynthesis(text, { voice_id, quality }, res);
        });
    }
    
    setupWebSocket() {
        this.wss = new WebSocket.Server({ noServer: true });
        
        this.wss.on('connection', (ws, req) => {
            console.log('WebSocket client connected');
            
            ws.on('message', async (message) => {
                try {
                    const data = JSON.parse(message);
                    await this.handleWebSocketMessage(ws, data);
                } catch (error) {
                    ws.send(JSON.stringify({
                        type: 'error',
                        error: error.message
                    }));
                }
            });
            
            ws.on('close', () => {
                console.log('WebSocket client disconnected');
            });
            
            // Send welcome message
            ws.send(JSON.stringify({
                type: 'connected',
                message: 'VoiRS WebSocket connected'
            }));
        });
    }
    
    async handleWebSocketMessage(ws, data) {
        switch (data.type) {
            case 'synthesize':
                await this.handleWebSocketSynthesis(ws, data);
                break;
                
            case 'stream_text':
                await this.handleWebSocketStream(ws, data);
                break;
                
            case 'cancel':
                this.handleWebSocketCancel(ws, data);
                break;
                
            default:
                ws.send(JSON.stringify({
                    type: 'error',
                    error: 'Unknown message type'
                }));
        }
    }
    
    async handleWebSocketSynthesis(ws, data) {
        const { text, config, requestId } = data;
        
        try {
            ws.send(JSON.stringify({
                type: 'synthesis_started',
                requestId
            }));
            
            const result = await this.voirsEngine.synthesize(text, config || {});
            
            if (result.success) {
                // Send audio data in base64
                const audioBase64 = result.audioData.toString('base64');
                ws.send(JSON.stringify({
                    type: 'synthesis_complete',
                    requestId,
                    audioData: audioBase64,
                    duration: result.duration,
                    format: config?.outputFormat || 'mp3'
                }));
            } else {
                ws.send(JSON.stringify({
                    type: 'synthesis_error',
                    requestId,
                    error: 'Synthesis failed'
                }));
            }
        } catch (error) {
            ws.send(JSON.stringify({
                type: 'synthesis_error',
                requestId,
                error: error.message
            }));
        }
    }
    
    async processBatch(batchId, texts, config) {
        const batch = this.activeSynthesis.get(batchId);
        if (!batch) return;
        
        try {
            for (let i = 0; i < texts.length; i++) {
                const result = await this.voirsEngine.synthesize(texts[i], config);
                
                batch.results.push({
                    index: i,
                    text: texts[i],
                    success: result.success,
                    audioData: result.success ? result.audioData.toString('base64') : null,
                    duration: result.duration,
                    error: result.success ? null : 'Synthesis failed'
                });
                
                batch.completed++;
                
                // Check if batch was cancelled
                if (!this.activeSynthesis.has(batchId)) {
                    return;
                }
            }
            
            batch.status = 'completed';
            
            // Clean up after 1 hour
            setTimeout(() => {
                this.activeSynthesis.delete(batchId);
            }, 3600000);
            
        } catch (error) {
            batch.status = 'error';
            batch.error = error.message;
        }
    }
    
    setupErrorHandling() {
        this.app.use((err, req, res, next) => {
            console.error('Express error:', err);
            res.status(500).json({
                error: 'Internal server error',
                timestamp: new Date().toISOString()
            });
        });
        
        // 404 handler
        this.app.use((req, res) => {
            res.status(404).json({
                error: 'Endpoint not found',
                path: req.path
            });
        });
    }
    
    getContentType(format) {
        switch (format) {
            case VoirsFFI.Format.MP3:
                return 'audio/mpeg';
            case VoirsFFI.Format.WAV:
                return 'audio/wav';
            case VoirsFFI.Format.FLAC:
                return 'audio/flac';
            default:
                return 'audio/mpeg';
        }
    }
    
    async start(port = 3000) {
        try {
            // Initialize VoiRS
            this.voirsEngine = new VoirsFFI.Engine();
            const initialized = await this.voirsEngine.initialize({
                quality: VoirsFFI.Quality.HIGH,
                threadCount: 4
            });
            
            if (!initialized) {
                throw new Error('Failed to initialize VoiRS engine');
            }
            
            // Start HTTP server
            this.server = this.app.listen(port, () => {
                console.log(`VoiRS Express server running on port ${port}`);
            });
            
            // Handle WebSocket upgrade
            this.server.on('upgrade', (request, socket, head) => {
                if (request.url === '/ws') {
                    this.wss.handleUpgrade(request, socket, head, (ws) => {
                        this.wss.emit('connection', ws, request);
                    });
                } else {
                    socket.destroy();
                }
            });
            
        } catch (error) {
            console.error('Failed to start server:', error);
            process.exit(1);
        }
    }
    
    async stop() {
        console.log('Shutting down VoiRS server...');
        
        if (this.wss) {
            this.wss.close();
        }
        
        if (this.server) {
            this.server.close();
        }
        
        if (this.voirsEngine) {
            await this.voirsEngine.shutdown();
        }
        
        console.log('Server shutdown complete');
    }
}

// Server startup
const server = new VoiRSServer();
const port = process.env.PORT || 3000;

server.start(port);

// Graceful shutdown
process.on('SIGTERM', async () => {
    await server.stop();
    process.exit(0);
});

process.on('SIGINT', async () => {
    await server.stop();
    process.exit(0);
});

module.exports = VoiRSServer;
```

#### Frontend Integration (client.js)

```javascript
class VoiRSClient {
    constructor(baseURL = 'http://localhost:3000') {
        this.baseURL = baseURL;
        this.ws = null;
        this.requestCallbacks = new Map();
    }
    
    // REST API methods
    async synthesize(text, options = {}) {
        const response = await fetch(`${this.baseURL}/api/synthesize`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                text,
                ...options
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Synthesis failed');
        }
        
        const audioBlob = await response.blob();
        return {
            audioBlob,
            duration: response.headers.get('X-Synthesis-Duration'),
            audioLength: response.headers.get('X-Audio-Length')
        };
    }
    
    async synthesizeBatch(texts, config = {}) {
        const response = await fetch(`${this.baseURL}/api/synthesize/batch`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ texts, config })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Batch synthesis failed');
        }
        
        return await response.json();
    }
    
    async getBatchStatus(batchId) {
        const response = await fetch(`${this.baseURL}/api/synthesize/batch/${batchId}`);
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Failed to get batch status');
        }
        
        return await response.json();
    }
    
    // WebSocket methods
    connectWebSocket() {
        return new Promise((resolve, reject) => {
            const wsURL = this.baseURL.replace('http', 'ws') + '/ws';
            this.ws = new WebSocket(wsURL);
            
            this.ws.onopen = () => {
                console.log('WebSocket connected');
                resolve();
            };
            
            this.ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handleWebSocketMessage(data);
            };
            
            this.ws.onclose = () => {
                console.log('WebSocket disconnected');
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                reject(error);
            };
        });
    }
    
    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'connected':
                console.log('WebSocket:', data.message);
                break;
                
            case 'synthesis_started':
            case 'synthesis_complete':
            case 'synthesis_error':
                const callback = this.requestCallbacks.get(data.requestId);
                if (callback) {
                    callback(data);
                    if (data.type !== 'synthesis_started') {
                        this.requestCallbacks.delete(data.requestId);
                    }
                }
                break;
        }
    }
    
    synthesizeWebSocket(text, config = {}) {
        return new Promise((resolve, reject) => {
            if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
                reject(new Error('WebSocket not connected'));
                return;
            }
            
            const requestId = Math.random().toString(36).substr(2, 9);
            
            this.requestCallbacks.set(requestId, (data) => {
                switch (data.type) {
                    case 'synthesis_started':
                        console.log('Synthesis started...');
                        break;
                        
                    case 'synthesis_complete':
                        // Convert base64 to blob
                        const audioData = atob(data.audioData);
                        const audioBytes = new Uint8Array(audioData.length);
                        for (let i = 0; i < audioData.length; i++) {
                            audioBytes[i] = audioData.charCodeAt(i);
                        }
                        
                        const audioBlob = new Blob([audioBytes], {
                            type: `audio/${data.format}`
                        });
                        
                        resolve({
                            audioBlob,
                            duration: data.duration,
                            format: data.format
                        });
                        break;
                        
                    case 'synthesis_error':
                        reject(new Error(data.error));
                        break;
                }
            });
            
            this.ws.send(JSON.stringify({
                type: 'synthesize',
                text,
                config,
                requestId
            }));
        });
    }
    
    disconnectWebSocket() {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }
}

// Usage example
async function example() {
    const client = new VoiRSClient();
    
    try {
        // REST API synthesis
        const result = await client.synthesize("Hello, world!", {
            voice_id: "female_young",
            quality: "high",
            format: "mp3"
        });
        
        // Play audio
        const audio = new Audio(URL.createObjectURL(result.audioBlob));
        audio.play();
        
        // WebSocket synthesis
        await client.connectWebSocket();
        const wsResult = await client.synthesizeWebSocket("WebSocket synthesis test");
        const wsAudio = new Audio(URL.createObjectURL(wsResult.audioBlob));
        wsAudio.play();
        
    } catch (error) {
        console.error('Error:', error);
    }
}

// Export for use in modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = VoiRSClient;
}
```

## FastAPI (Python) Integration

### Modern Async Python API

#### requirements.txt

```
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6
websockets==12.0
redis==5.0.1
celery==5.3.4
pydantic==2.5.0
aiofiles==23.2.1
voirs-ffi==0.1.0
```

#### FastAPI Server (main.py)

```python
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import asyncio
import uuid
import time
import json
import base64
from contextlib import asynccontextmanager
import voirs_ffi

# Pydantic models
class SynthesisRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    voice_id: Optional[str] = "default"
    quality: Optional[str] = "high"
    speed: Optional[float] = Field(1.0, ge=0.1, le=3.0)
    volume: Optional[float] = Field(1.0, ge=0.0, le=2.0)
    format: Optional[str] = "mp3"

class BatchSynthesisRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=50)
    config: Optional[SynthesisRequest] = None

class SynthesisResponse(BaseModel):
    success: bool
    audio_data: Optional[str] = None  # base64 encoded
    duration: Optional[float] = None
    error: Optional[str] = None

class BatchStatus(BaseModel):
    batch_id: str
    status: str
    progress: Dict[str, Any]
    results: Optional[List[SynthesisResponse]] = None

# Global state
class AppState:
    def __init__(self):
        self.voirs_engine = None
        self.active_batches: Dict[str, Dict] = {}
        self.websocket_connections: Dict[str, WebSocket] = {}

app_state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Initializing VoiRS engine...")
    app_state.voirs_engine = voirs_ffi.Engine()
    
    config = voirs_ffi.SynthesisConfig(
        quality=voirs_ffi.Quality.HIGH,
        thread_count=4
    )
    
    if not app_state.voirs_engine.initialize(config):
        raise RuntimeError("Failed to initialize VoiRS engine")
    
    print("VoiRS engine initialized successfully")
    yield
    
    # Shutdown
    print("Shutting down VoiRS engine...")
    if app_state.voirs_engine:
        app_state.voirs_engine.shutdown()
    print("Shutdown complete")

app = FastAPI(
    title="VoiRS FFI API",
    description="Real-time voice synthesis API using VoiRS FFI",
    version="1.0.0",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Routes
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "voirs_initialized": app_state.voirs_engine is not None
    }

@app.get("/api/voices")
async def get_voices():
    """Get list of available voices"""
    try:
        voices = voirs_ffi.get_available_voices()
        return {"voices": voices}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/synthesize")
async def synthesize_text(request: SynthesisRequest):
    """Synthesize text to speech"""
    if not app_state.voirs_engine:
        raise HTTPException(status_code=503, detail="VoiRS engine not initialized")
    
    try:
        # Convert request to VoiRS config
        config = voirs_ffi.SynthesisConfig(
            quality=getattr(voirs_ffi.Quality, request.quality.upper()),
            speed=request.speed,
            volume=request.volume,
            voice_id=request.voice_id,
            output_format=getattr(voirs_ffi.Format, request.format.upper())
        )
        
        # Perform synthesis
        result = await asyncio.get_event_loop().run_in_executor(
            None, app_state.voirs_engine.synthesize, request.text, config
        )
        
        if result.success:
            # Return appropriate content type
            content_type = get_content_type(request.format)
            
            return StreamingResponse(
                io.BytesIO(result.audio_data),
                media_type=content_type,
                headers={
                    "X-Synthesis-Duration": str(result.duration),
                    "X-Audio-Length": str(len(result.audio_data))
                }
            )
        else:
            raise HTTPException(status_code=500, detail="Synthesis failed")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/synthesize/batch")
async def synthesize_batch(request: BatchSynthesisRequest):
    """Start batch synthesis job"""
    if not app_state.voirs_engine:
        raise HTTPException(status_code=503, detail="VoiRS engine not initialized")
    
    batch_id = str(uuid.uuid4())
    
    # Initialize batch tracking
    app_state.active_batches[batch_id] = {
        "status": "processing",
        "total": len(request.texts),
        "completed": 0,
        "results": [],
        "start_time": time.time()
    }
    
    # Start background processing
    asyncio.create_task(process_batch(batch_id, request.texts, request.config))
    
    return {
        "batch_id": batch_id,
        "status": "processing",
        "total": len(request.texts)
    }

@app.get("/api/synthesize/batch/{batch_id}")
async def get_batch_status(batch_id: str):
    """Get batch synthesis status"""
    if batch_id not in app_state.active_batches:
        raise HTTPException(status_code=404, detail="Batch not found")
    
    batch = app_state.active_batches[batch_id]
    
    return BatchStatus(
        batch_id=batch_id,
        status=batch["status"],
        progress={
            "completed": batch["completed"],
            "total": batch["total"],
            "percentage": round((batch["completed"] / batch["total"]) * 100, 2)
        },
        results=batch["results"] if batch["status"] == "completed" else None
    )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time synthesis"""
    await websocket.accept()
    
    connection_id = str(uuid.uuid4())
    app_state.websocket_connections[connection_id] = websocket
    
    try:
        await websocket.send_text(json.dumps({
            "type": "connected",
            "connection_id": connection_id,
            "message": "VoiRS WebSocket connected"
        }))
        
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            await handle_websocket_message(websocket, message)
            
    except WebSocketDisconnect:
        print(f"WebSocket {connection_id} disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "error": str(e)
        }))
    finally:
        if connection_id in app_state.websocket_connections:
            del app_state.websocket_connections[connection_id]

async def handle_websocket_message(websocket: WebSocket, message: dict):
    """Handle incoming WebSocket messages"""
    message_type = message.get("type")
    
    if message_type == "synthesize":
        await handle_websocket_synthesis(websocket, message)
    elif message_type == "stream_text":
        await handle_websocket_stream(websocket, message)
    else:
        await websocket.send_text(json.dumps({
            "type": "error",
            "error": f"Unknown message type: {message_type}"
        }))

async def handle_websocket_synthesis(websocket: WebSocket, message: dict):
    """Handle WebSocket synthesis request"""
    request_id = message.get("request_id", str(uuid.uuid4()))
    text = message.get("text", "")
    config = message.get("config", {})
    
    if not text:
        await websocket.send_text(json.dumps({
            "type": "synthesis_error",
            "request_id": request_id,
            "error": "Text is required"
        }))
        return
    
    try:
        # Send start notification
        await websocket.send_text(json.dumps({
            "type": "synthesis_started",
            "request_id": request_id
        }))
        
        # Create synthesis config
        voirs_config = voirs_ffi.SynthesisConfig(
            quality=getattr(voirs_ffi.Quality, config.get("quality", "HIGH")),
            speed=config.get("speed", 1.0),
            volume=config.get("volume", 1.0),
            voice_id=config.get("voice_id", "default"),
            output_format=getattr(voirs_ffi.Format, config.get("format", "MP3"))
        )
        
        # Perform synthesis
        result = await asyncio.get_event_loop().run_in_executor(
            None, app_state.voirs_engine.synthesize, text, voirs_config
        )
        
        if result.success:
            # Encode audio data as base64
            audio_base64 = base64.b64encode(result.audio_data).decode('utf-8')
            
            await websocket.send_text(json.dumps({
                "type": "synthesis_complete",
                "request_id": request_id,
                "audio_data": audio_base64,
                "duration": result.duration,
                "format": config.get("format", "mp3")
            }))
        else:
            await websocket.send_text(json.dumps({
                "type": "synthesis_error",
                "request_id": request_id,
                "error": "Synthesis failed"
            }))
            
    except Exception as e:
        await websocket.send_text(json.dumps({
            "type": "synthesis_error",
            "request_id": request_id,
            "error": str(e)
        }))

async def process_batch(batch_id: str, texts: List[str], config: Optional[SynthesisRequest]):
    """Process batch synthesis in background"""
    batch = app_state.active_batches[batch_id]
    
    try:
        # Create synthesis config
        if config:
            voirs_config = voirs_ffi.SynthesisConfig(
                quality=getattr(voirs_ffi.Quality, config.quality.upper()),
                speed=config.speed,
                volume=config.volume,
                voice_id=config.voice_id,
                output_format=getattr(voirs_ffi.Format, config.format.upper())
            )
        else:
            voirs_config = voirs_ffi.SynthesisConfig()
        
        # Process each text
        for i, text in enumerate(texts):
            # Check if batch was cancelled
            if batch_id not in app_state.active_batches:
                return
            
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, app_state.voirs_engine.synthesize, text, voirs_config
                )
                
                batch["results"].append(SynthesisResponse(
                    success=result.success,
                    audio_data=base64.b64encode(result.audio_data).decode('utf-8') if result.success else None,
                    duration=result.duration if result.success else None,
                    error=None if result.success else "Synthesis failed"
                ))
                
            except Exception as e:
                batch["results"].append(SynthesisResponse(
                    success=False,
                    error=str(e)
                ))
            
            batch["completed"] += 1
        
        batch["status"] = "completed"
        batch["end_time"] = time.time()
        
        # Clean up after 1 hour
        asyncio.create_task(cleanup_batch(batch_id, 3600))
        
    except Exception as e:
        batch["status"] = "error"
        batch["error"] = str(e)

async def cleanup_batch(batch_id: str, delay: int):
    """Clean up batch data after delay"""
    await asyncio.sleep(delay)
    if batch_id in app_state.active_batches:
        del app_state.active_batches[batch_id]

def get_content_type(format_name: str) -> str:
    """Get MIME type for audio format"""
    formats = {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "flac": "audio/flac",
        "ogg": "audio/ogg"
    }
    return formats.get(format_name.lower(), "audio/mpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        access_log=True
    )
```

## Flask (Python) Integration

### Traditional Python Web Framework

#### Flask Application (app.py)

```python
from flask import Flask, request, jsonify, send_file, stream_template
from flask_cors import CORS
from werkzeug.exceptions import BadRequest, InternalServerError
import voirs_ffi
import io
import base64
import uuid
import time
import threading
from queue import Queue
import json

class VoiRSFlaskApp:
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)
        
        self.voirs_engine = None
        self.active_batches = {}
        self.batch_lock = threading.Lock()
        
        self.setup_routes()
        self.setup_error_handlers()
    
    def setup_routes(self):
        @self.app.route('/health', methods=['GET'])
        def health():
            return jsonify({
                'status': 'healthy',
                'timestamp': time.time(),
                'voirs_initialized': self.voirs_engine is not None
            })
        
        @self.app.route('/api/voices', methods=['GET'])
        def get_voices():
            try:
                voices = voirs_ffi.get_available_voices()
                return jsonify({'voices': voices})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/synthesize', methods=['POST'])
        def synthesize():
            if not self.voirs_engine:
                return jsonify({'error': 'VoiRS engine not initialized'}), 503
            
            data = request.get_json()
            if not data or 'text' not in data:
                return jsonify({'error': 'Text is required'}), 400
            
            text = data['text']
            if len(text) > 10000:
                return jsonify({'error': 'Text too long (max 10,000 characters)'}), 400
            
            try:
                config = voirs_ffi.SynthesisConfig(
                    quality=getattr(voirs_ffi.Quality, data.get('quality', 'HIGH')),
                    speed=data.get('speed', 1.0),
                    volume=data.get('volume', 1.0),
                    voice_id=data.get('voice_id', 'default'),
                    output_format=getattr(voirs_ffi.Format, data.get('format', 'MP3'))
                )
                
                result = self.voirs_engine.synthesize(text, config)
                
                if result.success:
                    audio_io = io.BytesIO(result.audio_data)
                    content_type = self.get_content_type(data.get('format', 'mp3'))
                    
                    return send_file(
                        audio_io,
                        mimetype=content_type,
                        as_attachment=True,
                        download_name=f'synthesis_{int(time.time())}.{data.get("format", "mp3")}'
                    )
                else:
                    return jsonify({'error': 'Synthesis failed'}), 500
                    
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/synthesize/json', methods=['POST'])
        def synthesize_json():
            """Return synthesis result as JSON with base64 audio"""
            if not self.voirs_engine:
                return jsonify({'error': 'VoiRS engine not initialized'}), 503
            
            data = request.get_json()
            if not data or 'text' not in data:
                return jsonify({'error': 'Text is required'}), 400
            
            try:
                config = voirs_ffi.SynthesisConfig(
                    quality=getattr(voirs_ffi.Quality, data.get('quality', 'HIGH')),
                    speed=data.get('speed', 1.0),
                    volume=data.get('volume', 1.0),
                    voice_id=data.get('voice_id', 'default'),
                    output_format=getattr(voirs_ffi.Format, data.get('format', 'MP3'))
                )
                
                result = self.voirs_engine.synthesize(data['text'], config)
                
                if result.success:
                    audio_base64 = base64.b64encode(result.audio_data).decode('utf-8')
                    return jsonify({
                        'success': True,
                        'audio_data': audio_base64,
                        'duration': result.duration,
                        'format': data.get('format', 'mp3'),
                        'size': len(result.audio_data)
                    })
                else:
                    return jsonify({'success': False, 'error': 'Synthesis failed'}), 500
                    
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/synthesize/batch', methods=['POST'])
        def synthesize_batch():
            if not self.voirs_engine:
                return jsonify({'error': 'VoiRS engine not initialized'}), 503
            
            data = request.get_json()
            if not data or 'texts' not in data:
                return jsonify({'error': 'Texts array is required'}), 400
            
            texts = data['texts']
            if not isinstance(texts, list) or len(texts) == 0:
                return jsonify({'error': 'Texts must be a non-empty array'}), 400
            
            if len(texts) > 50:
                return jsonify({'error': 'Too many texts (max 50)'}), 400
            
            batch_id = str(uuid.uuid4())
            
            with self.batch_lock:
                self.active_batches[batch_id] = {
                    'status': 'processing',
                    'total': len(texts),
                    'completed': 0,
                    'results': [],
                    'start_time': time.time()
                }
            
            # Start background processing
            thread = threading.Thread(
                target=self.process_batch,
                args=(batch_id, texts, data.get('config', {}))
            )
            thread.daemon = True
            thread.start()
            
            return jsonify({
                'batch_id': batch_id,
                'status': 'processing',
                'total': len(texts)
            })
        
        @self.app.route('/api/synthesize/batch/<batch_id>', methods=['GET'])
        def get_batch_status(batch_id):
            with self.batch_lock:
                if batch_id not in self.active_batches:
                    return jsonify({'error': 'Batch not found'}), 404
                
                batch = self.active_batches[batch_id].copy()
            
            return jsonify({
                'batch_id': batch_id,
                'status': batch['status'],
                'progress': {
                    'completed': batch['completed'],
                    'total': batch['total'],
                    'percentage': round((batch['completed'] / batch['total']) * 100, 2)
                },
                'results': batch['results'] if batch['status'] == 'completed' else None,
                'processing_time': time.time() - batch['start_time']
            })
        
        @self.app.route('/api/synthesize/stream')
        def synthesize_stream():
            """Server-sent events stream for real-time synthesis"""
            text = request.args.get('text')
            if not text:
                return jsonify({'error': 'Text query parameter is required'}), 400
            
            def generate():
                try:
                    yield f"data: {json.dumps({'type': 'start', 'message': 'Starting synthesis'})}\n\n"
                    
                    config = voirs_ffi.SynthesisConfig(
                        quality=getattr(voirs_ffi.Quality, request.args.get('quality', 'HIGH')),
                        voice_id=request.args.get('voice_id', 'default')
                    )
                    
                    result = self.voirs_engine.synthesize(text, config)
                    
                    if result.success:
                        audio_base64 = base64.b64encode(result.audio_data).decode('utf-8')
                        yield f"data: {json.dumps({
                            'type': 'complete',
                            'audio_data': audio_base64,
                            'duration': result.duration
                        })}\n\n"
                    else:
                        yield f"data: {json.dumps({'type': 'error', 'error': 'Synthesis failed'})}\n\n"
                        
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
            
            return self.app.response_class(
                generate(),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive'
                }
            )
    
    def setup_error_handlers(self):
        @self.app.errorhandler(400)
        def bad_request(error):
            return jsonify({'error': 'Bad request', 'message': str(error)}), 400
        
        @self.app.errorhandler(404)
        def not_found(error):
            return jsonify({'error': 'Not found', 'path': request.path}), 404
        
        @self.app.errorhandler(500)
        def internal_error(error):
            return jsonify({'error': 'Internal server error', 'message': str(error)}), 500
    
    def process_batch(self, batch_id, texts, config):
        """Process batch synthesis in background thread"""
        try:
            voirs_config = voirs_ffi.SynthesisConfig(
                quality=getattr(voirs_ffi.Quality, config.get('quality', 'HIGH')),
                speed=config.get('speed', 1.0),
                volume=config.get('volume', 1.0),
                voice_id=config.get('voice_id', 'default'),
                output_format=getattr(voirs_ffi.Format, config.get('format', 'MP3'))
            )
            
            for i, text in enumerate(texts):
                # Check if batch still exists
                with self.batch_lock:
                    if batch_id not in self.active_batches:
                        return
                
                try:
                    result = self.voirs_engine.synthesize(text, voirs_config)
                    
                    result_data = {
                        'index': i,
                        'text': text,
                        'success': result.success,
                        'duration': result.duration if result.success else None,
                        'error': None if result.success else 'Synthesis failed'
                    }
                    
                    if result.success:
                        result_data['audio_data'] = base64.b64encode(result.audio_data).decode('utf-8')
                    
                    with self.batch_lock:
                        if batch_id in self.active_batches:
                            self.active_batches[batch_id]['results'].append(result_data)
                            self.active_batches[batch_id]['completed'] += 1
                        
                except Exception as e:
                    with self.batch_lock:
                        if batch_id in self.active_batches:
                            self.active_batches[batch_id]['results'].append({
                                'index': i,
                                'text': text,
                                'success': False,
                                'error': str(e)
                            })
                            self.active_batches[batch_id]['completed'] += 1
            
            # Mark as completed
            with self.batch_lock:
                if batch_id in self.active_batches:
                    self.active_batches[batch_id]['status'] = 'completed'
                    self.active_batches[batch_id]['end_time'] = time.time()
            
            # Schedule cleanup
            cleanup_timer = threading.Timer(3600, self.cleanup_batch, args=[batch_id])
            cleanup_timer.start()
            
        except Exception as e:
            with self.batch_lock:
                if batch_id in self.active_batches:
                    self.active_batches[batch_id]['status'] = 'error'
                    self.active_batches[batch_id]['error'] = str(e)
    
    def cleanup_batch(self, batch_id):
        """Clean up batch data"""
        with self.batch_lock:
            if batch_id in self.active_batches:
                del self.active_batches[batch_id]
    
    def get_content_type(self, format_name):
        """Get MIME type for audio format"""
        formats = {
            'mp3': 'audio/mpeg',
            'wav': 'audio/wav',
            'flac': 'audio/flac',
            'ogg': 'audio/ogg'
        }
        return formats.get(format_name.lower(), 'audio/mpeg')
    
    def initialize_voirs(self):
        """Initialize VoiRS engine"""
        try:
            self.voirs_engine = voirs_ffi.Engine()
            config = voirs_ffi.SynthesisConfig(
                quality=voirs_ffi.Quality.HIGH,
                thread_count=4
            )
            
            if self.voirs_engine.initialize(config):
                print("VoiRS engine initialized successfully")
                return True
            else:
                print("Failed to initialize VoiRS engine")
                return False
                
        except Exception as e:
            print(f"Error initializing VoiRS: {e}")
            return False
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the Flask application"""
        if not self.initialize_voirs():
            raise RuntimeError("Failed to initialize VoiRS engine")
        
        try:
            self.app.run(host=host, port=port, debug=debug, threaded=True)
        finally:
            if self.voirs_engine:
                self.voirs_engine.shutdown()
                print("VoiRS engine shutdown complete")

# Create and run the application
if __name__ == '__main__':
    app = VoiRSFlaskApp()
    app.run(debug=True)
```

## Django (Python) Integration

### Enterprise Python Web Framework

#### Django Settings (settings.py)

```python
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# VoiRS Configuration
VOIRS_CONFIG = {
    'ENGINE_CONFIG': {
        'quality': 'HIGH',
        'thread_count': 4,
        'cache_size': 1024 * 1024,  # 1MB
    },
    'DEFAULT_SYNTHESIS_CONFIG': {
        'quality': 'HIGH',
        'speed': 1.0,
        'volume': 1.0,
        'voice_id': 'default',
        'format': 'MP3'
    },
    'BATCH_LIMITS': {
        'max_items': 50,
        'max_text_length': 10000,
        'cleanup_timeout': 3600  # 1 hour
    }
}

# ... other Django settings ...

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'corsheaders',
    'voirs_app',  # Our VoiRS Django app
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

REST_FRAMEWORK = {
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 20,
    'DEFAULT_THROTTLE_CLASSES': [
        'rest_framework.throttling.AnonRateThrottle',
        'rest_framework.throttling.UserRateThrottle'
    ],
    'DEFAULT_THROTTLE_RATES': {
        'anon': '100/hour',
        'user': '1000/hour'
    }
}

CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
```

#### Django Models (voirs_app/models.py)

```python
from django.db import models
from django.contrib.auth.models import User
import uuid

class SynthesisJob(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
        ('cancelled', 'Cancelled'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    text = models.TextField(max_length=10000)
    voice_id = models.CharField(max_length=100, default='default')
    quality = models.CharField(max_length=20, default='HIGH')
    speed = models.FloatField(default=1.0)
    volume = models.FloatField(default=1.0)
    output_format = models.CharField(max_length=10, default='MP3')
    
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    # Results
    audio_data = models.BinaryField(null=True, blank=True)
    duration = models.FloatField(null=True, blank=True)
    error_message = models.TextField(null=True, blank=True)
    
    # Metadata
    file_size = models.IntegerField(null=True, blank=True)
    processing_time = models.FloatField(null=True, blank=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Synthesis Job {self.id} - {self.status}"

class BatchSynthesisJob(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
        ('cancelled', 'Cancelled'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    total_items = models.IntegerField()
    completed_items = models.IntegerField(default=0)
    failed_items = models.IntegerField(default=0)
    
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    # Configuration
    config_json = models.JSONField(default=dict)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Batch Job {self.id} - {self.status}"
    
    @property
    def progress_percentage(self):
        if self.total_items == 0:
            return 0
        return round((self.completed_items / self.total_items) * 100, 2)

class BatchSynthesisItem(models.Model):
    batch_job = models.ForeignKey(BatchSynthesisJob, on_delete=models.CASCADE, related_name='items')
    index = models.IntegerField()
    text = models.TextField(max_length=10000)
    
    # Results
    status = models.CharField(max_length=20, choices=SynthesisJob.STATUS_CHOICES, default='pending')
    audio_data = models.BinaryField(null=True, blank=True)
    duration = models.FloatField(null=True, blank=True)
    error_message = models.TextField(null=True, blank=True)
    processing_time = models.FloatField(null=True, blank=True)
    
    class Meta:
        ordering = ['index']
        unique_together = ['batch_job', 'index']
    
    def __str__(self):
        return f"Batch Item {self.batch_job.id}[{self.index}]"
```

#### Django Views (voirs_app/views.py)

```python
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.http import HttpResponse, StreamingHttpResponse
from django.shortcuts import get_object_or_404
from django.conf import settings
from django.utils import timezone
import voirs_ffi
import json
import base64
import threading
import time

from .models import SynthesisJob, BatchSynthesisJob, BatchSynthesisItem
from .serializers import SynthesisJobSerializer, BatchSynthesisJobSerializer
from .voirs_manager import VoiRSManager

class SynthesisJobViewSet(viewsets.ModelViewSet):
    queryset = SynthesisJob.objects.all()
    serializer_class = SynthesisJobSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        return SynthesisJob.objects.filter(user=self.request.user)
    
    def perform_create(self, serializer):
        synthesis_job = serializer.save(user=self.request.user)
        
        # Start async processing
        thread = threading.Thread(
            target=self.process_synthesis_job,
            args=[synthesis_job.id]
        )
        thread.daemon = True
        thread.start()
    
    def process_synthesis_job(self, job_id):
        """Process synthesis job in background thread"""
        try:
            job = SynthesisJob.objects.get(id=job_id)
            job.status = 'processing'
            job.started_at = timezone.now()
            job.save()
            
            start_time = time.time()
            
            # Create VoiRS config
            config = voirs_ffi.SynthesisConfig(
                quality=getattr(voirs_ffi.Quality, job.quality),
                speed=job.speed,
                volume=job.volume,
                voice_id=job.voice_id,
                output_format=getattr(voirs_ffi.Format, job.output_format)
            )
            
            # Perform synthesis
            result = VoiRSManager.get_instance().synthesize(job.text, config)
            
            processing_time = time.time() - start_time
            
            if result.success:
                job.status = 'completed'
                job.audio_data = result.audio_data
                job.duration = result.duration
                job.file_size = len(result.audio_data)
                job.processing_time = processing_time
            else:
                job.status = 'failed'
                job.error_message = 'Synthesis failed'
            
            job.completed_at = timezone.now()
            job.save()
            
        except Exception as e:
            job.status = 'failed'
            job.error_message = str(e)
            job.completed_at = timezone.now()
            job.save()
    
    @action(detail=True, methods=['get'])
    def download(self, request, pk=None):
        """Download synthesized audio"""
        job = get_object_or_404(SynthesisJob, pk=pk, user=request.user)
        
        if job.status != 'completed' or not job.audio_data:
            return Response({'error': 'Audio not available'}, 
                          status=status.HTTP_404_NOT_FOUND)
        
        content_type = self.get_content_type(job.output_format)
        filename = f'synthesis_{job.id}.{job.output_format.lower()}'
        
        response = HttpResponse(job.audio_data, content_type=content_type)
        response['Content-Disposition'] = f'attachment; filename="{filename}"'
        response['X-Synthesis-Duration'] = str(job.duration)
        response['X-Processing-Time'] = str(job.processing_time)
        
        return response
    
    @action(detail=False, methods=['post'])
    def synthesize_immediate(self, request):
        """Immediate synthesis without database storage"""
        text = request.data.get('text')
        if not text:
            return Response({'error': 'Text is required'}, 
                          status=status.HTTP_400_BAD_REQUEST)
        
        try:
            config = voirs_ffi.SynthesisConfig(
                quality=getattr(voirs_ffi.Quality, 
                              request.data.get('quality', 'HIGH')),
                speed=request.data.get('speed', 1.0),
                volume=request.data.get('volume', 1.0),
                voice_id=request.data.get('voice_id', 'default'),
                output_format=getattr(voirs_ffi.Format, 
                                    request.data.get('format', 'MP3'))
            )
            
            result = VoiRSManager.get_instance().synthesize(text, config)
            
            if result.success:
                audio_base64 = base64.b64encode(result.audio_data).decode('utf-8')
                return Response({
                    'success': True,
                    'audio_data': audio_base64,
                    'duration': result.duration,
                    'format': request.data.get('format', 'MP3'),
                    'size': len(result.audio_data)
                })
            else:
                return Response({'error': 'Synthesis failed'}, 
                              status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                
        except Exception as e:
            return Response({'error': str(e)}, 
                          status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def get_content_type(self, format_name):
        formats = {
            'MP3': 'audio/mpeg',
            'WAV': 'audio/wav',
            'FLAC': 'audio/flac',
            'OGG': 'audio/ogg'
        }
        return formats.get(format_name, 'audio/mpeg')

class BatchSynthesisJobViewSet(viewsets.ModelViewSet):
    queryset = BatchSynthesisJob.objects.all()
    serializer_class = BatchSynthesisJobSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        return BatchSynthesisJob.objects.filter(user=self.request.user)
    
    def create(self, request):
        """Create batch synthesis job"""
        texts = request.data.get('texts', [])
        config = request.data.get('config', {})
        
        if not texts or not isinstance(texts, list):
            return Response({'error': 'Texts array is required'}, 
                          status=status.HTTP_400_BAD_REQUEST)
        
        if len(texts) > settings.VOIRS_CONFIG['BATCH_LIMITS']['max_items']:
            return Response({'error': f'Too many texts (max {settings.VOIRS_CONFIG["BATCH_LIMITS"]["max_items"]})'}, 
                          status=status.HTTP_400_BAD_REQUEST)
        
        # Create batch job
        batch_job = BatchSynthesisJob.objects.create(
            user=request.user,
            total_items=len(texts),
            config_json=config
        )
        
        # Create batch items
        for i, text in enumerate(texts):
            BatchSynthesisItem.objects.create(
                batch_job=batch_job,
                index=i,
                text=text
            )
        
        # Start processing
        thread = threading.Thread(
            target=self.process_batch_job,
            args=[batch_job.id]
        )
        thread.daemon = True
        thread.start()
        
        return Response({
            'batch_id': str(batch_job.id),
            'status': 'processing',
            'total': len(texts)
        }, status=status.HTTP_201_CREATED)
    
    def process_batch_job(self, batch_id):
        """Process batch synthesis job"""
        try:
            batch_job = BatchSynthesisJob.objects.get(id=batch_id)
            batch_job.status = 'processing'
            batch_job.started_at = timezone.now()
            batch_job.save()
            
            config_data = batch_job.config_json
            config = voirs_ffi.SynthesisConfig(
                quality=getattr(voirs_ffi.Quality, config_data.get('quality', 'HIGH')),
                speed=config_data.get('speed', 1.0),
                volume=config_data.get('volume', 1.0),
                voice_id=config_data.get('voice_id', 'default'),
                output_format=getattr(voirs_ffi.Format, config_data.get('format', 'MP3'))
            )
            
            # Process each item
            for item in batch_job.items.all():
                try:
                    item.status = 'processing'
                    item.save()
                    
                    start_time = time.time()
                    result = VoiRSManager.get_instance().synthesize(item.text, config)
                    processing_time = time.time() - start_time
                    
                    if result.success:
                        item.status = 'completed'
                        item.audio_data = result.audio_data
                        item.duration = result.duration
                        item.processing_time = processing_time
                        
                        batch_job.completed_items += 1
                    else:
                        item.status = 'failed'
                        item.error_message = 'Synthesis failed'
                        batch_job.failed_items += 1
                    
                    item.save()
                    batch_job.save()
                    
                except Exception as e:
                    item.status = 'failed'
                    item.error_message = str(e)
                    item.save()
                    
                    batch_job.failed_items += 1
                    batch_job.save()
            
            batch_job.status = 'completed'
            batch_job.completed_at = timezone.now()
            batch_job.save()
            
        except Exception as e:
            batch_job.status = 'failed'
            batch_job.save()
    
    @action(detail=True, methods=['get'])
    def results(self, request, pk=None):
        """Get batch synthesis results"""
        batch_job = get_object_or_404(BatchSynthesisJob, pk=pk, user=request.user)
        
        results = []
        for item in batch_job.items.all():
            result_data = {
                'index': item.index,
                'text': item.text,
                'status': item.status,
                'duration': item.duration,
                'processing_time': item.processing_time,
                'error': item.error_message
            }
            
            if item.audio_data and request.query_params.get('include_audio') == 'true':
                result_data['audio_data'] = base64.b64encode(item.audio_data).decode('utf-8')
            
            results.append(result_data)
        
        return Response({
            'batch_id': str(batch_job.id),
            'status': batch_job.status,
            'progress': {
                'completed': batch_job.completed_items,
                'failed': batch_job.failed_items,
                'total': batch_job.total_items,
                'percentage': batch_job.progress_percentage
            },
            'results': results
        })
```

## Actix Web (Rust) Integration

### High-Performance Rust Web Framework

#### Cargo.toml

```toml
[package]
name = "voirs-actix-server"
version = "0.1.0"
edition = "2021"

[dependencies]
actix-web = "4.4"
actix-cors = "0.6"
tokio = { version = "1.0", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
uuid = { version = "1.0", features = ["v4", "serde"] }
base64 = "0.21"
futures = "0.3"
log = "0.4"
env_logger = "0.10"
thiserror = "1.0"
anyhow = "1.0"
voirs-ffi = { path = "../voirs-ffi" }
```

#### Actix Web Server (src/main.rs)

```rust
use actix_web::{
    web, App, HttpServer, HttpResponse, Result, middleware::Logger,
    http::header::ContentType,
};
use actix_cors::Cors;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::{Arc, Mutex}};
use uuid::Uuid;
use voirs_ffi::{Engine, SynthesisConfig, Quality, Format};

#[derive(Debug, Serialize, Deserialize)]
pub struct SynthesisRequest {
    pub text: String,
    pub voice_id: Option<String>,
    pub quality: Option<String>,
    pub speed: Option<f32>,
    pub volume: Option<f32>,
    pub format: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct SynthesisResponse {
    pub success: bool,
    pub audio_data: Option<String>, // base64 encoded
    pub duration: Option<f32>,
    pub error: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BatchRequest {
    pub texts: Vec<String>,
    pub config: Option<SynthesisRequest>,
}

#[derive(Debug, Serialize)]
pub struct BatchResponse {
    pub batch_id: Uuid,
    pub status: String,
    pub total: usize,
}

#[derive(Debug, Serialize)]
pub struct BatchStatus {
    pub batch_id: Uuid,
    pub status: String,
    pub progress: BatchProgress,
    pub results: Option<Vec<SynthesisResponse>>,
}

#[derive(Debug, Serialize)]
pub struct BatchProgress {
    pub completed: usize,
    pub total: usize,
    pub percentage: f32,
}

#[derive(Debug)]
pub struct BatchJob {
    pub status: String,
    pub total: usize,
    pub completed: usize,
    pub results: Vec<SynthesisResponse>,
}

pub struct AppState {
    pub engine: Arc<Mutex<Engine>>,
    pub batches: Arc<Mutex<HashMap<Uuid, BatchJob>>>,
}

async fn health() -> Result<HttpResponse> {
    Ok(HttpResponse::Ok().json(serde_json::json!({
        "status": "healthy",
        "timestamp": chrono::Utc::now().timestamp(),
        "service": "voirs-actix-server"
    })))
}

async fn get_voices() -> Result<HttpResponse> {
    // In a real implementation, this would call voirs_ffi::get_available_voices()
    let voices = vec!["default", "male_young", "female_young", "male_deep", "female_warm"];
    Ok(HttpResponse::Ok().json(serde_json::json!({
        "voices": voices
    })))
}

async fn synthesize(
    request: web::Json<SynthesisRequest>,
    data: web::Data<AppState>,
) -> Result<HttpResponse> {
    if request.text.is_empty() {
        return Ok(HttpResponse::BadRequest().json(serde_json::json!({
            "error": "Text is required"
        })));
    }

    if request.text.len() > 10000 {
        return Ok(HttpResponse::BadRequest().json(serde_json::json!({
            "error": "Text too long (max 10,000 characters)"
        })));
    }

    let engine = data.engine.lock().unwrap();
    
    // Create synthesis config
    let quality = match request.quality.as_deref() {
        Some("low") => Quality::Low,
        Some("medium") => Quality::Medium,
        Some("high") => Quality::High,
        Some("ultra") => Quality::Ultra,
        _ => Quality::High,
    };

    let format = match request.format.as_deref() {
        Some("wav") => Format::WAV,
        Some("mp3") => Format::MP3,
        Some("flac") => Format::FLAC,
        _ => Format::MP3,
    };

    let config = SynthesisConfig {
        quality,
        speed: request.speed.unwrap_or(1.0),
        volume: request.volume.unwrap_or(1.0),
        voice_id: request.voice_id.clone().unwrap_or_else(|| "default".to_string()),
        output_format: format,
    };

    match engine.synthesize(&request.text, &config) {
        Ok(result) => {
            if result.success {
                let audio_base64 = base64::encode(&result.audio_data);
                Ok(HttpResponse::Ok().json(SynthesisResponse {
                    success: true,
                    audio_data: Some(audio_base64),
                    duration: Some(result.duration),
                    error: None,
                }))
            } else {
                Ok(HttpResponse::InternalServerError().json(SynthesisResponse {
                    success: false,
                    audio_data: None,
                    duration: None,
                    error: Some("Synthesis failed".to_string()),
                }))
            }
        }
        Err(e) => Ok(HttpResponse::InternalServerError().json(SynthesisResponse {
            success: false,
            audio_data: None,
            duration: None,
            error: Some(e.to_string()),
        })),
    }
}

async fn synthesize_batch(
    request: web::Json<BatchRequest>,
    data: web::Data<AppState>,
) -> Result<HttpResponse> {
    if request.texts.is_empty() {
        return Ok(HttpResponse::BadRequest().json(serde_json::json!({
            "error": "Texts array is required"
        })));
    }

    if request.texts.len() > 50 {
        return Ok(HttpResponse::BadRequest().json(serde_json::json!({
            "error": "Too many texts (max 50)"
        })));
    }

    let batch_id = Uuid::new_v4();
    let total = request.texts.len();

    // Initialize batch job
    {
        let mut batches = data.batches.lock().unwrap();
        batches.insert(batch_id, BatchJob {
            status: "processing".to_string(),
            total,
            completed: 0,
            results: Vec::new(),
        });
    }

    // Clone data for async processing
    let texts = request.texts.clone();
    let config = request.config.clone();
    let app_data = data.clone();

    // Process batch asynchronously
    tokio::spawn(async move {
        process_batch(batch_id, texts, config, app_data).await;
    });

    Ok(HttpResponse::Ok().json(BatchResponse {
        batch_id,
        status: "processing".to_string(),
        total,
    }))
}

async fn process_batch(
    batch_id: Uuid,
    texts: Vec<String>,
    config: Option<SynthesisRequest>,
    data: web::Data<AppState>,
) {
    let default_config = config.unwrap_or_else(|| SynthesisRequest {
        text: String::new(),
        voice_id: Some("default".to_string()),
        quality: Some("high".to_string()),
        speed: Some(1.0),
        volume: Some(1.0),
        format: Some("mp3".to_string()),
    });

    for (i, text) in texts.iter().enumerate() {
        // Check if batch still exists
        if !data.batches.lock().unwrap().contains_key(&batch_id) {
            return;
        }

        let synthesis_request = SynthesisRequest {
            text: text.clone(),
            voice_id: default_config.voice_id.clone(),
            quality: default_config.quality.clone(),
            speed: default_config.speed,
            volume: default_config.volume,
            format: default_config.format.clone(),
        };

        // Perform synthesis
        let result = {
            let engine = data.engine.lock().unwrap();
            let quality = match synthesis_request.quality.as_deref() {
                Some("low") => Quality::Low,
                Some("medium") => Quality::Medium,
                Some("high") => Quality::High,
                Some("ultra") => Quality::Ultra,
                _ => Quality::High,
            };

            let format = match synthesis_request.format.as_deref() {
                Some("wav") => Format::WAV,
                Some("mp3") => Format::MP3,
                Some("flac") => Format::FLAC,
                _ => Format::MP3,
            };

            let config = SynthesisConfig {
                quality,
                speed: synthesis_request.speed.unwrap_or(1.0),
                volume: synthesis_request.volume.unwrap_or(1.0),
                voice_id: synthesis_request.voice_id.unwrap_or_else(|| "default".to_string()),
                output_format: format,
            };

            engine.synthesize(&synthesis_request.text, &config)
        };

        let synthesis_response = match result {
            Ok(audio_result) => {
                if audio_result.success {
                    SynthesisResponse {
                        success: true,
                        audio_data: Some(base64::encode(&audio_result.audio_data)),
                        duration: Some(audio_result.duration),
                        error: None,
                    }
                } else {
                    SynthesisResponse {
                        success: false,
                        audio_data: None,
                        duration: None,
                        error: Some("Synthesis failed".to_string()),
                    }
                }
            }
            Err(e) => SynthesisResponse {
                success: false,
                audio_data: None,
                duration: None,
                error: Some(e.to_string()),
            },
        };

        // Update batch job
        {
            let mut batches = data.batches.lock().unwrap();
            if let Some(batch) = batches.get_mut(&batch_id) {
                batch.results.push(synthesis_response);
                batch.completed += 1;
            }
        }
    }

    // Mark batch as completed
    {
        let mut batches = data.batches.lock().unwrap();
        if let Some(batch) = batches.get_mut(&batch_id) {
            batch.status = "completed".to_string();
        }
    }

    // Schedule cleanup after 1 hour
    let cleanup_data = data.clone();
    tokio::spawn(async move {
        tokio::time::sleep(tokio::time::Duration::from_secs(3600)).await;
        cleanup_data.batches.lock().unwrap().remove(&batch_id);
    });
}

async fn get_batch_status(
    path: web::Path<Uuid>,
    data: web::Data<AppState>,
) -> Result<HttpResponse> {
    let batch_id = path.into_inner();
    
    let batches = data.batches.lock().unwrap();
    
    if let Some(batch) = batches.get(&batch_id) {
        let percentage = if batch.total > 0 {
            (batch.completed as f32 / batch.total as f32) * 100.0
        } else {
            0.0
        };

        Ok(HttpResponse::Ok().json(BatchStatus {
            batch_id,
            status: batch.status.clone(),
            progress: BatchProgress {
                completed: batch.completed,
                total: batch.total,
                percentage,
            },
            results: if batch.status == "completed" {
                Some(batch.results.clone())
            } else {
                None
            },
        }))
    } else {
        Ok(HttpResponse::NotFound().json(serde_json::json!({
            "error": "Batch not found"
        })))
    }
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::init();

    // Initialize VoiRS engine
    let mut engine = Engine::new();
    let config = SynthesisConfig {
        quality: Quality::High,
        speed: 1.0,
        volume: 1.0,
        voice_id: "default".to_string(),
        output_format: Format::MP3,
    };

    if !engine.initialize(&config) {
        panic!("Failed to initialize VoiRS engine");
    }

    let app_state = web::Data::new(AppState {
        engine: Arc::new(Mutex::new(engine)),
        batches: Arc::new(Mutex::new(HashMap::new())),
    });

    log::info!("Starting VoiRS Actix Web server on 0.0.0.0:8080");

    HttpServer::new(move || {
        let cors = Cors::default()
            .allow_any_origin()
            .allow_any_method()
            .allow_any_header();

        App::new()
            .app_data(app_state.clone())
            .wrap(cors)
            .wrap(Logger::default())
            .route("/health", web::get().to(health))
            .route("/api/voices", web::get().to(get_voices))
            .route("/api/synthesize", web::post().to(synthesize))
            .route("/api/synthesize/batch", web::post().to(synthesize_batch))
            .route("/api/synthesize/batch/{batch_id}", web::get().to(get_batch_status))
    })
    .bind("0.0.0.0:8080")?
    .run()
    .await
}
```

## WebAssembly Frontend Integration

### Browser-Based Voice Synthesis

#### Frontend HTML/JavaScript

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VoiRS WebAssembly Demo</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        .container {
            background: #f5f5f5;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }
        .controls {
            margin: 10px 0;
        }
        label {
            display: inline-block;
            width: 120px;
            font-weight: bold;
        }
        input, select, textarea {
            margin: 5px;
            padding: 8px;
        }
        textarea {
            width: 100%;
            height: 100px;
            resize: vertical;
        }
        button {
            background: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            margin: 5px;
        }
        button:hover {
            background: #0056b3;
        }
        button:disabled {
            background: #6c757d;
            cursor: not-allowed;
        }
        .progress {
            width: 100%;
            height: 20px;
            background: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
        }
        .progress-bar {
            height: 100%;
            background: #28a745;
            transition: width 0.3s ease;
        }
        .audio-player {
            margin: 10px 0;
        }
        .error {
            color: #dc3545;
            background: #f8d7da;
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
        }
        .success {
            color: #155724;
            background: #d4edda;
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <h1>VoiRS WebAssembly Voice Synthesis</h1>
    
    <div class="container">
        <h2>Single Text Synthesis</h2>
        
        <div class="controls">
            <label for="textInput">Text:</label>
            <textarea id="textInput" placeholder="Enter text to synthesize...">Hello, this is a VoiRS WebAssembly demonstration!</textarea>
        </div>
        
        <div class="controls">
            <label for="voiceSelect">Voice:</label>
            <select id="voiceSelect">
                <option value="default">Default</option>
                <option value="male_young">Male Young</option>
                <option value="female_young">Female Young</option>
                <option value="male_deep">Male Deep</option>
                <option value="female_warm">Female Warm</option>
            </select>
        </div>
        
        <div class="controls">
            <label for="qualitySelect">Quality:</label>
            <select id="qualitySelect">
                <option value="low">Low</option>
                <option value="medium">Medium</option>
                <option value="high" selected>High</option>
                <option value="ultra">Ultra</option>
            </select>
        </div>
        
        <div class="controls">
            <label for="speedRange">Speed:</label>
            <input type="range" id="speedRange" min="0.5" max="2.0" step="0.1" value="1.0">
            <span id="speedValue">1.0</span>
        </div>
        
        <div class="controls">
            <label for="volumeRange">Volume:</label>
            <input type="range" id="volumeRange" min="0.0" max="2.0" step="0.1" value="1.0">
            <span id="volumeValue">1.0</span>
        </div>
        
        <button id="synthesizeBtn">Synthesize</button>
        <button id="stopBtn" disabled>Stop</button>
        
        <div id="progressContainer" style="display: none;">
            <div class="progress">
                <div class="progress-bar" id="progressBar" style="width: 0%"></div>
            </div>
            <div id="progressText">Processing...</div>
        </div>
        
        <div id="result"></div>
        <div id="audioContainer"></div>
    </div>
    
    <div class="container">
        <h2>Batch Processing</h2>
        
        <div class="controls">
            <label for="batchTexts">Texts (one per line):</label>
            <textarea id="batchTexts" placeholder="Enter multiple texts, one per line...">Hello world!
This is the second sentence.
And this is the third one.</textarea>
        </div>
        
        <button id="batchSynthesizeBtn">Process Batch</button>
        
        <div id="batchProgress" style="display: none;">
            <h3>Batch Progress</h3>
            <div class="progress">
                <div class="progress-bar" id="batchProgressBar" style="width: 0%"></div>
            </div>
            <div id="batchProgressText">0 / 0 completed</div>
        </div>
        
        <div id="batchResults"></div>
    </div>

    <script type="module">
        // VoiRS WebAssembly wrapper
        class VoiRSWasm {
            constructor() {
                this.module = null;
                this.initialized = false;
            }
            
            async initialize() {
                try {
                    // Load VoiRS WebAssembly module
                    this.module = await import('./voirs_wasm.js');
                    await this.module.default();
                    
                    // Initialize VoiRS engine
                    const config = {
                        quality: 'high',
                        thread_count: 2 // Limited for browser
                    };
                    
                    this.initialized = this.module.initialize(config);
                    return this.initialized;
                } catch (error) {
                    console.error('Failed to initialize VoiRS WASM:', error);
                    return false;
                }
            }
            
            synthesize(text, config = {}) {
                if (!this.initialized) {
                    throw new Error('VoiRS not initialized');
                }
                
                return new Promise((resolve, reject) => {
                    try {
                        const result = this.module.synthesize(text, config);
                        
                        if (result.success) {
                            // Convert Uint8Array to AudioBuffer
                            const audioContext = new AudioContext();
                            audioContext.decodeAudioData(result.audio_data.buffer)
                                .then(audioBuffer => {
                                    resolve({
                                        success: true,
                                        audioBuffer,
                                        duration: result.duration,
                                        audioData: result.audio_data
                                    });
                                })
                                .catch(reject);
                        } else {
                            reject(new Error(result.error || 'Synthesis failed'));
                        }
                    } catch (error) {
                        reject(error);
                    }
                });
            }
            
            async synthesizeBatch(texts, config = {}) {
                const results = [];
                
                for (let i = 0; i < texts.length; i++) {
                    try {
                        const result = await this.synthesize(texts[i], config);
                        results.push({
                            index: i,
                            text: texts[i],
                            success: true,
                            audioBuffer: result.audioBuffer,
                            duration: result.duration
                        });
                    } catch (error) {
                        results.push({
                            index: i,
                            text: texts[i],
                            success: false,
                            error: error.message
                        });
                    }
                    
                    // Yield control to browser
                    await new Promise(resolve => setTimeout(resolve, 0));
                }
                
                return results;
            }
        }
        
        // Application class
        class VoiRSApp {
            constructor() {
                this.voirs = new VoiRSWasm();
                this.currentAudio = null;
                this.synthesisCancelled = false;
                
                this.setupUI();
            }
            
            async initialize() {
                const initialized = await this.voirs.initialize();
                if (!initialized) {
                    this.showError('Failed to initialize VoiRS WebAssembly module');
                    return false;
                }
                return true;
            }
            
            setupUI() {
                // Speed and volume range updates
                document.getElementById('speedRange').addEventListener('input', (e) => {
                    document.getElementById('speedValue').textContent = e.target.value;
                });
                
                document.getElementById('volumeRange').addEventListener('input', (e) => {
                    document.getElementById('volumeValue').textContent = e.target.value;
                });
                
                // Synthesis button
                document.getElementById('synthesizeBtn').addEventListener('click', () => {
                    this.handleSynthesize();
                });
                
                // Stop button
                document.getElementById('stopBtn').addEventListener('click', () => {
                    this.handleStop();
                });
                
                // Batch synthesis button
                document.getElementById('batchSynthesizeBtn').addEventListener('click', () => {
                    this.handleBatchSynthesize();
                });
            }
            
            async handleSynthesize() {
                const text = document.getElementById('textInput').value.trim();
                if (!text) {
                    this.showError('Please enter text to synthesize');
                    return;
                }
                
                const config = {
                    voice_id: document.getElementById('voiceSelect').value,
                    quality: document.getElementById('qualitySelect').value,
                    speed: parseFloat(document.getElementById('speedRange').value),
                    volume: parseFloat(document.getElementById('volumeRange').value),
                    format: 'wav' // WebAssembly works best with WAV
                };
                
                this.showProgress(true);
                this.setSynthesizeButtons(false, true);
                this.synthesisCancelled = false;
                
                try {
                    const result = await this.voirs.synthesize(text, config);
                    
                    if (!this.synthesisCancelled) {
                        this.showAudioResult(result);
                        this.showSuccess(`Synthesis completed in ${result.duration.toFixed(2)}s`);
                    }
                } catch (error) {
                    if (!this.synthesisCancelled) {
                        this.showError(`Synthesis failed: ${error.message}`);
                    }
                } finally {
                    this.showProgress(false);
                    this.setSynthesizeButtons(true, false);
                }
            }
            
            handleStop() {
                this.synthesisCancelled = true;
                if (this.currentAudio) {
                    this.currentAudio.pause();
                    this.currentAudio.currentTime = 0;
                }
                this.showProgress(false);
                this.setSynthesizeButtons(true, false);
            }
            
            async handleBatchSynthesize() {
                const textsInput = document.getElementById('batchTexts').value.trim();
                if (!textsInput) {
                    this.showError('Please enter texts for batch processing');
                    return;
                }
                
                const texts = textsInput.split('\n').filter(line => line.trim());
                if (texts.length === 0) {
                    this.showError('No valid texts found');
                    return;
                }
                
                const config = {
                    voice_id: document.getElementById('voiceSelect').value,
                    quality: document.getElementById('qualitySelect').value,
                    speed: parseFloat(document.getElementById('speedRange').value),
                    volume: parseFloat(document.getElementById('volumeRange').value),
                    format: 'wav'
                };
                
                this.showBatchProgress(true, 0, texts.length);
                document.getElementById('batchSynthesizeBtn').disabled = true;
                
                try {
                    const results = [];
                    
                    for (let i = 0; i < texts.length; i++) {
                        if (this.synthesisCancelled) break;
                        
                        try {
                            const result = await this.voirs.synthesize(texts[i], config);
                            results.push({
                                index: i,
                                text: texts[i],
                                success: true,
                                audioBuffer: result.audioBuffer,
                                duration: result.duration
                            });
                        } catch (error) {
                            results.push({
                                index: i,
                                text: texts[i],
                                success: false,
                                error: error.message
                            });
                        }
                        
                        this.showBatchProgress(true, i + 1, texts.length);
                    }
                    
                    this.showBatchResults(results);
                    
                } catch (error) {
                    this.showError(`Batch processing failed: ${error.message}`);
                } finally {
                    this.showBatchProgress(false);
                    document.getElementById('batchSynthesizeBtn').disabled = false;
                }
            }
            
            showAudioResult(result) {
                const container = document.getElementById('audioContainer');
                
                // Create audio blob from buffer
                const audioContext = new AudioContext();
                const source = audioContext.createBufferSource();
                source.buffer = result.audioBuffer;
                
                // Convert to playable audio
                const canvas = document.createElement('canvas');
                const audioBlob = this.audioBufferToBlob(result.audioBuffer);
                const audioUrl = URL.createObjectURL(audioBlob);
                
                container.innerHTML = `
                    <div class="audio-player">
                        <h3>Generated Audio</h3>
                        <audio controls>
                            <source src="${audioUrl}" type="audio/wav">
                            Your browser does not support audio playback.
                        </audio>
                        <div>Duration: ${result.duration.toFixed(2)}s</div>
                        <button onclick="this.previousElementSibling.previousElementSibling.download='synthesis.wav'">Download</button>
                    </div>
                `;
                
                this.currentAudio = container.querySelector('audio');
            }
            
            audioBufferToBlob(audioBuffer) {
                const length = audioBuffer.length * audioBuffer.numberOfChannels * 2;
                const buffer = new ArrayBuffer(44 + length);
                const view = new DataView(buffer);
                
                // WAV header
                const writeString = (offset, string) => {
                    for (let i = 0; i < string.length; i++) {
                        view.setUint8(offset + i, string.charCodeAt(i));
                    }
                };
                
                writeString(0, 'RIFF');
                view.setUint32(4, 36 + length, true);
                writeString(8, 'WAVE');
                writeString(12, 'fmt ');
                view.setUint32(16, 16, true);
                view.setUint16(20, 1, true);
                view.setUint16(22, audioBuffer.numberOfChannels, true);
                view.setUint32(24, audioBuffer.sampleRate, true);
                view.setUint32(28, audioBuffer.sampleRate * audioBuffer.numberOfChannels * 2, true);
                view.setUint16(32, audioBuffer.numberOfChannels * 2, true);
                view.setUint16(34, 16, true);
                writeString(36, 'data');
                view.setUint32(40, length, true);
                
                // Audio data
                let offset = 44;
                for (let i = 0; i < audioBuffer.length; i++) {
                    for (let channel = 0; channel < audioBuffer.numberOfChannels; channel++) {
                        const sample = Math.max(-1, Math.min(1, audioBuffer.getChannelData(channel)[i]));
                        view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
                        offset += 2;
                    }
                }
                
                return new Blob([buffer], { type: 'audio/wav' });
            }
            
            showBatchResults(results) {
                const container = document.getElementById('batchResults');
                let html = '<h3>Batch Results</h3>';
                
                results.forEach((result, index) => {
                    html += `<div class="batch-result">`;
                    html += `<h4>Result ${index + 1}</h4>`;
                    html += `<p><strong>Text:</strong> ${result.text}</p>`;
                    
                    if (result.success) {
                        const audioBlob = this.audioBufferToBlob(result.audioBuffer);
                        const audioUrl = URL.createObjectURL(audioBlob);
                        
                        html += `<div class="success">✓ Success (${result.duration.toFixed(2)}s)</div>`;
                        html += `<audio controls><source src="${audioUrl}" type="audio/wav"></audio>`;
                    } else {
                        html += `<div class="error">✗ Failed: ${result.error}</div>`;
                    }
                    
                    html += `</div>`;
                });
                
                container.innerHTML = html;
            }
            
            showProgress(show, percentage = 0) {
                const container = document.getElementById('progressContainer');
                const bar = document.getElementById('progressBar');
                const text = document.getElementById('progressText');
                
                container.style.display = show ? 'block' : 'none';
                if (show) {
                    bar.style.width = `${percentage}%`;
                    text.textContent = percentage > 0 ? `${percentage}%` : 'Processing...';
                }
            }
            
            showBatchProgress(show, completed = 0, total = 0) {
                const container = document.getElementById('batchProgress');
                const bar = document.getElementById('batchProgressBar');
                const text = document.getElementById('batchProgressText');
                
                container.style.display = show ? 'block' : 'none';
                if (show && total > 0) {
                    const percentage = Math.round((completed / total) * 100);
                    bar.style.width = `${percentage}%`;
                    text.textContent = `${completed} / ${total} completed`;
                }
            }
            
            setSynthesizeButtons(synthesizeEnabled, stopEnabled) {
                document.getElementById('synthesizeBtn').disabled = !synthesizeEnabled;
                document.getElementById('stopBtn').disabled = !stopEnabled;
            }
            
            showError(message) {
                const container = document.getElementById('result');
                container.innerHTML = `<div class="error">${message}</div>`;
            }
            
            showSuccess(message) {
                const container = document.getElementById('result');
                container.innerHTML = `<div class="success">${message}</div>`;
            }
        }
        
        // Initialize application
        const app = new VoiRSApp();
        
        window.addEventListener('load', async () => {
            const initialized = await app.initialize();
            if (initialized) {
                console.log('VoiRS WebAssembly application ready');
            }
        });
    </script>
</body>
</html>
```

These web framework integration examples demonstrate how to build production-ready voice synthesis services using VoiRS FFI across different platforms and architectures, from high-performance Rust servers to browser-based WebAssembly applications.
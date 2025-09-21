# Real-Time Applications Integration Examples

This document provides comprehensive examples for integrating VoiRS FFI with real-time applications including live streaming, telecommunications, assistive technology, and interactive systems.

## Table of Contents

1. [Live Audio Streaming](#live-audio-streaming)
2. [Voice over IP (VoIP) Integration](#voice-over-ip-voip-integration)
3. [Assistive Technology](#assistive-technology)
4. [Interactive Voice Response (IVR)](#interactive-voice-response-ivr)
5. [Real-Time Broadcasting](#real-time-broadcasting)
6. [Edge Computing Integration](#edge-computing-integration)

## Live Audio Streaming

### WebRTC Real-Time Synthesis

#### WebRTC Streaming Server (Node.js)

```javascript
const WebSocket = require('ws');
const wrtc = require('wrtc');
const VoirsFFI = require('voirs-ffi');

class VoiRSWebRTCServer {
    constructor() {
        this.voirsEngine = null;
        this.peers = new Map();
        this.synthesisQueue = [];
        this.isProcessing = false;
        
        this.initializeVoiRS();
        this.setupWebSocketServer();
    }
    
    async initializeVoiRS() {
        try {
            this.voirsEngine = new VoirsFFI.Engine();
            
            // Optimize for real-time performance
            const config = {
                quality: VoirsFFI.Quality.MEDIUM, // Balance quality vs speed
                threadCount: 2,
                useSimd: true,
                cacheSize: 256 * 1024, // 256KB for low latency
                outputFormat: VoirsFFI.Format.WAV,
                sampleRate: 16000 // Lower sample rate for streaming
            };
            
            const initialized = await this.voirsEngine.initialize(config);
            if (initialized) {
                console.log('VoiRS engine initialized for real-time streaming');
                this.startProcessingLoop();
            } else {
                throw new Error('VoiRS initialization failed');
            }
        } catch (error) {
            console.error('VoiRS initialization error:', error);
        }
    }
    
    setupWebSocketServer() {
        this.wss = new WebSocket.Server({ port: 8080 });
        
        this.wss.on('connection', (ws) => {
            console.log('Client connected');
            
            ws.on('message', async (message) => {
                try {
                    const data = JSON.parse(message);
                    await this.handleMessage(ws, data);
                } catch (error) {
                    console.error('Message handling error:', error);
                    ws.send(JSON.stringify({
                        type: 'error',
                        error: error.message
                    }));
                }
            });
            
            ws.on('close', () => {
                console.log('Client disconnected');
                this.cleanupPeer(ws);
            });
        });
        
        console.log('WebRTC signaling server running on port 8080');
    }
    
    async handleMessage(ws, data) {
        switch (data.type) {
            case 'offer':
                await this.handleOffer(ws, data);
                break;
                
            case 'answer':
                await this.handleAnswer(ws, data);
                break;
                
            case 'ice-candidate':
                await this.handleIceCandidate(ws, data);
                break;
                
            case 'text-synthesis':
                await this.handleTextSynthesis(ws, data);
                break;
                
            case 'stream-text':
                await this.handleStreamText(ws, data);
                break;
        }
    }
    
    async handleOffer(ws, data) {
        const peer = new wrtc.RTCPeerConnection({
            iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
        });
        
        this.peers.set(ws, peer);
        
        // Create audio track for synthesis output
        const audioTrack = this.createAudioTrack();
        peer.addTrack(audioTrack);
        
        // Set up data channel for text input
        const dataChannel = peer.createDataChannel('textInput');
        dataChannel.onmessage = (event) => {
            this.queueSynthesis(event.data, ws);
        };
        
        peer.onicecandidate = (event) => {
            if (event.candidate) {
                ws.send(JSON.stringify({
                    type: 'ice-candidate',
                    candidate: event.candidate
                }));
            }
        };
        
        await peer.setRemoteDescription(data.offer);
        const answer = await peer.createAnswer();
        await peer.setLocalDescription(answer);
        
        ws.send(JSON.stringify({
            type: 'answer',
            answer: answer
        }));
    }
    
    createAudioTrack() {
        // Create a MediaStreamTrack that we can write audio data to
        const source = new wrtc.MediaStreamTrackAudioSource();
        const track = source.createTrack();
        
        // Store the source for later use
        track._voirsSource = source;
        
        return track;
    }
    
    queueSynthesis(text, ws) {
        this.synthesisQueue.push({
            text: text,
            ws: ws,
            timestamp: Date.now()
        });
    }
    
    async startProcessingLoop() {
        this.isProcessing = true;
        
        while (this.isProcessing) {
            if (this.synthesisQueue.length > 0) {
                const request = this.synthesisQueue.shift();
                await this.processSynthesisRequest(request);
            } else {
                // Short delay to prevent busy waiting
                await new Promise(resolve => setTimeout(resolve, 10));
            }
        }
    }
    
    async processSynthesisRequest(request) {
        try {
            const startTime = Date.now();
            
            // Perform synthesis with real-time optimizations
            const result = await this.voirsEngine.synthesize(request.text, {
                quality: VoirsFFI.Quality.MEDIUM,
                speed: 1.1, // Slightly faster for real-time feel
                outputFormat: VoirsFFI.Format.WAV
            });
            
            const synthesisTime = Date.now() - startTime;
            
            if (result.success) {
                // Convert audio to streaming format
                const audioChunks = this.chunkAudioForStreaming(result.audioData, 160); // 10ms chunks at 16kHz
                
                // Stream audio to peer
                const peer = this.peers.get(request.ws);
                if (peer) {
                    await this.streamAudioToPeer(peer, audioChunks);
                }
                
                // Send metadata
                request.ws.send(JSON.stringify({
                    type: 'synthesis-complete',
                    text: request.text,
                    duration: result.duration,
                    synthesisTime: synthesisTime,
                    realTimeFactor: result.duration / (synthesisTime / 1000)
                }));
                
            } else {
                request.ws.send(JSON.stringify({
                    type: 'synthesis-error',
                    text: request.text,
                    error: 'Synthesis failed'
                }));
            }
            
        } catch (error) {
            console.error('Synthesis processing error:', error);
            request.ws.send(JSON.stringify({
                type: 'synthesis-error',
                text: request.text,
                error: error.message
            }));
        }
    }
    
    chunkAudioForStreaming(audioData, samplesPerChunk) {
        const chunks = [];
        const sampleSize = 2; // 16-bit samples
        const bytesPerChunk = samplesPerChunk * sampleSize;
        
        for (let i = 0; i < audioData.length; i += bytesPerChunk) {
            const chunk = audioData.slice(i, i + bytesPerChunk);
            chunks.push(chunk);
        }
        
        return chunks;
    }
    
    async streamAudioToPeer(peer, audioChunks) {
        // Get the audio track and its source
        const sender = peer.getSenders().find(s => s.track && s.track.kind === 'audio');
        if (!sender || !sender.track._voirsSource) {
            return;
        }
        
        const source = sender.track._voirsSource;
        
        // Stream chunks with proper timing
        for (const chunk of audioChunks) {
            // Convert chunk to the format expected by WebRTC
            const audioFrame = this.createAudioFrame(chunk);
            source.onData(audioFrame);
            
            // Wait for chunk duration (10ms for 160 samples at 16kHz)
            await new Promise(resolve => setTimeout(resolve, 10));
        }
    }
    
    createAudioFrame(audioData) {
        // Convert raw audio data to WebRTC audio frame format
        const samples = new Int16Array(audioData.buffer);
        
        return {
            data: samples,
            sampleRate: 16000,
            channelCount: 1,
            numberOfFrames: samples.length
        };
    }
    
    cleanupPeer(ws) {
        const peer = this.peers.get(ws);
        if (peer) {
            peer.close();
            this.peers.delete(ws);
        }
    }
    
    stop() {
        this.isProcessing = false;
        
        // Cleanup all peers
        for (const [ws, peer] of this.peers) {
            peer.close();
        }
        this.peers.clear();
        
        // Close WebSocket server
        if (this.wss) {
            this.wss.close();
        }
        
        // Shutdown VoiRS engine
        if (this.voirsEngine) {
            this.voirsEngine.shutdown();
        }
    }
}

// Usage
const server = new VoiRSWebRTCServer();

// Graceful shutdown
process.on('SIGTERM', () => {
    console.log('Shutting down VoiRS WebRTC server...');
    server.stop();
    process.exit(0);
});
```

#### WebRTC Client (JavaScript)

```html
<!DOCTYPE html>
<html>
<head>
    <title>VoiRS Real-Time Synthesis</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .controls { margin: 20px 0; }
        .status { padding: 10px; border-radius: 5px; margin: 10px 0; }
        .status.connected { background: #d4edda; color: #155724; }
        .status.error { background: #f8d7da; color: #721c24; }
        .metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
        .metric { padding: 10px; background: #f8f9fa; border-radius: 5px; text-align: center; }
    </style>
</head>
<body>
    <h1>VoiRS Real-Time Text-to-Speech</h1>
    
    <div class="controls">
        <button id="connectBtn">Connect</button>
        <button id="disconnectBtn" disabled>Disconnect</button>
    </div>
    
    <div id="status" class="status">Disconnected</div>
    
    <div class="controls">
        <input type="text" id="textInput" placeholder="Enter text to synthesize..." style="width: 300px;">
        <button id="synthesizeBtn" disabled>Synthesize</button>
    </div>
    
    <div class="controls">
        <label>
            <input type="checkbox" id="streamingMode"> Streaming Mode
        </label>
    </div>
    
    <audio id="audioPlayer" controls style="width: 100%; margin: 20px 0;"></audio>
    
    <div class="metrics">
        <div class="metric">
            <div>Synthesis Time</div>
            <div id="synthesisTime">--</div>
        </div>
        <div class="metric">
            <div>Audio Duration</div>
            <div id="audioDuration">--</div>
        </div>
        <div class="metric">
            <div>Real-time Factor</div>
            <div id="realTimeFactor">--</div>
        </div>
    </div>
    
    <div id="log" style="margin-top: 20px; height: 200px; overflow-y: scroll; border: 1px solid #ccc; padding: 10px;"></div>

    <script>
        class VoiRSRealTimeClient {
            constructor() {
                this.ws = null;
                this.peerConnection = null;
                this.dataChannel = null;
                this.audioContext = null;
                this.connected = false;
                
                this.setupUI();
            }
            
            setupUI() {
                document.getElementById('connectBtn').onclick = () => this.connect();
                document.getElementById('disconnectBtn').onclick = () => this.disconnect();
                document.getElementById('synthesizeBtn').onclick = () => this.synthesizeText();
                
                document.getElementById('textInput').onkeypress = (e) => {
                    if (e.key === 'Enter') {
                        this.synthesizeText();
                    }
                };
            }
            
            async connect() {
                try {
                    this.log('Connecting to VoiRS server...');
                    
                    // Initialize audio context
                    this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
                    
                    // Connect to signaling server
                    this.ws = new WebSocket('ws://localhost:8080');
                    
                    this.ws.onopen = () => {
                        this.log('WebSocket connected');
                        this.setupWebRTC();
                    };
                    
                    this.ws.onmessage = (event) => {
                        const data = JSON.parse(event.data);
                        this.handleSignalingMessage(data);
                    };
                    
                    this.ws.onclose = () => {
                        this.log('WebSocket disconnected');
                        this.setConnected(false);
                    };
                    
                    this.ws.onerror = (error) => {
                        this.log('WebSocket error: ' + error, 'error');
                    };
                    
                } catch (error) {
                    this.log('Connection error: ' + error.message, 'error');
                }
            }
            
            async setupWebRTC() {
                try {
                    this.peerConnection = new RTCPeerConnection({
                        iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
                    });
                    
                    // Handle incoming audio stream
                    this.peerConnection.ontrack = (event) => {
                        this.log('Received audio track');
                        const audioPlayer = document.getElementById('audioPlayer');
                        audioPlayer.srcObject = event.streams[0];
                        audioPlayer.play();
                    };
                    
                    // Handle data channel
                    this.peerConnection.ondatachannel = (event) => {
                        this.dataChannel = event.channel;
                        this.log('Data channel established');
                    };
                    
                    this.peerConnection.onicecandidate = (event) => {
                        if (event.candidate) {
                            this.ws.send(JSON.stringify({
                                type: 'ice-candidate',
                                candidate: event.candidate
                            }));
                        }
                    };
                    
                    // Create offer
                    const offer = await this.peerConnection.createOffer();
                    await this.peerConnection.setLocalDescription(offer);
                    
                    this.ws.send(JSON.stringify({
                        type: 'offer',
                        offer: offer
                    }));
                    
                } catch (error) {
                    this.log('WebRTC setup error: ' + error.message, 'error');
                }
            }
            
            async handleSignalingMessage(data) {
                try {
                    switch (data.type) {
                        case 'answer':
                            await this.peerConnection.setRemoteDescription(data.answer);
                            this.setConnected(true);
                            this.log('WebRTC connection established');
                            break;
                            
                        case 'ice-candidate':
                            await this.peerConnection.addIceCandidate(data.candidate);
                            break;
                            
                        case 'synthesis-complete':
                            this.updateMetrics(data);
                            this.log(`Synthesis complete: "${data.text}" (${data.duration.toFixed(2)}s)`);
                            break;
                            
                        case 'synthesis-error':
                            this.log(`Synthesis error: ${data.error}`, 'error');
                            break;
                    }
                } catch (error) {
                    this.log('Signaling message error: ' + error.message, 'error');
                }
            }
            
            synthesizeText() {
                const text = document.getElementById('textInput').value.trim();
                if (!text) {
                    this.log('Please enter text to synthesize', 'error');
                    return;
                }
                
                if (!this.connected) {
                    this.log('Not connected to server', 'error');
                    return;
                }
                
                const streamingMode = document.getElementById('streamingMode').checked;
                
                if (streamingMode && this.dataChannel) {
                    // Send via data channel for real-time streaming
                    this.dataChannel.send(text);
                    this.log(`Streaming synthesis: "${text}"`);
                } else {
                    // Send via WebSocket for standard synthesis
                    this.ws.send(JSON.stringify({
                        type: 'text-synthesis',
                        text: text
                    }));
                    this.log(`Standard synthesis: "${text}"`);
                }
                
                // Clear input
                document.getElementById('textInput').value = '';
            }
            
            updateMetrics(data) {
                document.getElementById('synthesisTime').textContent = `${data.synthesisTime}ms`;
                document.getElementById('audioDuration').textContent = `${data.duration.toFixed(2)}s`;
                document.getElementById('realTimeFactor').textContent = `${data.realTimeFactor.toFixed(2)}x`;
            }
            
            setConnected(connected) {
                this.connected = connected;
                
                document.getElementById('connectBtn').disabled = connected;
                document.getElementById('disconnectBtn').disabled = !connected;
                document.getElementById('synthesizeBtn').disabled = !connected;
                
                const statusEl = document.getElementById('status');
                if (connected) {
                    statusEl.textContent = 'Connected';
                    statusEl.className = 'status connected';
                } else {
                    statusEl.textContent = 'Disconnected';
                    statusEl.className = 'status error';
                }
            }
            
            disconnect() {
                if (this.peerConnection) {
                    this.peerConnection.close();
                    this.peerConnection = null;
                }
                
                if (this.ws) {
                    this.ws.close();
                    this.ws = null;
                }
                
                if (this.audioContext) {
                    this.audioContext.close();
                    this.audioContext = null;
                }
                
                this.setConnected(false);
                this.log('Disconnected from server');
            }
            
            log(message, type = 'info') {
                const logEl = document.getElementById('log');
                const timestamp = new Date().toLocaleTimeString();
                const logEntry = document.createElement('div');
                logEntry.style.color = type === 'error' ? 'red' : 'black';
                logEntry.textContent = `[${timestamp}] ${message}`;
                logEl.appendChild(logEntry);
                logEl.scrollTop = logEl.scrollHeight;
            }
        }
        
        // Initialize client
        const client = new VoiRSRealTimeClient();
    </script>
</body>
</html>
```

## Voice over IP (VoIP) Integration

### SIP Server Integration

#### Asterisk AGI Script (Python)

```python
#!/usr/bin/env python3
"""
VoiRS AGI Script for Asterisk
Provides text-to-speech synthesis for Asterisk PBX using VoiRS FFI
"""

import sys
import os
import voirs_ffi
import tempfile
import re
from asterisk.agi import AGI

class VoiRSAsteriskAGI:
    def __init__(self):
        self.agi = AGI()
        self.engine = None
        self.temp_dir = tempfile.mkdtemp(prefix='voirs_asterisk_')
        self.initialize_voirs()
    
    def initialize_voirs(self):
        """Initialize VoiRS engine with telephony-optimized settings"""
        try:
            self.engine = voirs_ffi.Engine()
            
            # Telephony optimization: 8kHz, μ-law compatible quality
            config = voirs_ffi.SynthesisConfig(
                quality=voirs_ffi.Quality.MEDIUM,  # Good quality for phone
                sample_rate=8000,  # Telephony standard
                output_format=voirs_ffi.Format.WAV,
                thread_count=1,  # Conservative for telephony server
                use_simd=True
            )
            
            if self.engine.initialize(config):
                self.agi.verbose("VoiRS engine initialized for telephony")
                return True
            else:
                self.agi.verbose("VoiRS initialization failed", 1)
                return False
                
        except Exception as e:
            self.agi.verbose(f"VoiRS init error: {e}", 1)
            return False
    
    def text_to_speech(self, text, voice_id='default'):
        """Convert text to speech and return audio file path"""
        try:
            # Clean text for speech synthesis
            clean_text = self.clean_text_for_speech(text)
            
            if not clean_text.strip():
                return None
            
            # Synthesize with telephony settings
            config = voirs_ffi.SynthesisConfig(
                voice_id=voice_id,
                quality=voirs_ffi.Quality.MEDIUM,
                speed=0.9,  # Slightly slower for phone clarity
                volume=1.0
            )
            
            result = self.engine.synthesize(clean_text, config)
            
            if result.success:
                # Save to temporary file
                audio_file = os.path.join(self.temp_dir, f"tts_{os.getpid()}.wav")
                
                with open(audio_file, 'wb') as f:
                    f.write(result.audio_data)
                
                self.agi.verbose(f"TTS generated: {audio_file} ({result.duration:.2f}s)")
                return audio_file
            else:
                self.agi.verbose("TTS synthesis failed", 1)
                return None
                
        except Exception as e:
            self.agi.verbose(f"TTS error: {e}", 1)
            return None
    
    def clean_text_for_speech(self, text):
        """Clean text for better speech synthesis"""
        # Remove or expand common abbreviations
        replacements = {
            r'\bDr\.': 'Doctor',
            r'\bMr\.': 'Mister', 
            r'\bMrs\.': 'Missus',
            r'\bMs\.': 'Miss',
            r'\bProf\.': 'Professor',
            r'\bInc\.': 'Incorporated',
            r'\bLtd\.': 'Limited',
            r'\bCorp\.': 'Corporation',
            r'\bCo\.': 'Company',
            r'\bSt\.': 'Street',
            r'\bAve\.': 'Avenue',
            r'\bBlvd\.': 'Boulevard',
            r'\bRd\.': 'Road',
            r'\betc\.': 'etcetera',
            r'\bi\.e\.': 'that is',
            r'\be\.g\.': 'for example',
            r'\bvs\.': 'versus',
            r'\bUS\b': 'United States',
            r'\bUSA\b': 'United States of America',
            r'\bUK\b': 'United Kingdom',
            r'\bCEO\b': 'chief executive officer',
            r'\bCTO\b': 'chief technology officer',
            r'\bCFO\b': 'chief financial officer'
        }
        
        clean_text = text
        for pattern, replacement in replacements.items():
            clean_text = re.sub(pattern, replacement, clean_text, flags=re.IGNORECASE)
        
        # Normalize whitespace
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        # Handle numbers (basic implementation)
        clean_text = self.expand_numbers(clean_text)
        
        return clean_text
    
    def expand_numbers(self, text):
        """Basic number expansion for speech"""
        # Simple number to word conversion (extend as needed)
        number_words = {
            '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
            '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
            '10': 'ten', '11': 'eleven', '12': 'twelve', '13': 'thirteen',
            '14': 'fourteen', '15': 'fifteen', '16': 'sixteen', '17': 'seventeen',
            '18': 'eighteen', '19': 'nineteen', '20': 'twenty'
        }
        
        # Replace standalone numbers
        for num, word in number_words.items():
            text = re.sub(rf'\b{num}\b', word, text)
        
        return text
    
    def handle_command(self, command, *args):
        """Handle AGI commands from Asterisk"""
        try:
            if command == 'say':
                # Basic TTS playback
                text = ' '.join(args) if args else ''
                if not text:
                    text = self.agi.get_variable('TTS_TEXT') or ''
                
                voice = self.agi.get_variable('TTS_VOICE') or 'default'
                
                audio_file = self.text_to_speech(text, voice)
                if audio_file:
                    # Play the generated audio
                    self.agi.stream_file(audio_file.replace('.wav', ''))
                    # Clean up
                    try:
                        os.unlink(audio_file)
                    except:
                        pass
                else:
                    self.agi.verbose("Failed to generate TTS audio", 1)
            
            elif command == 'say_digits':
                # Speak individual digits
                digits = args[0] if args else ''
                digit_text = ' '.join(list(digits))
                
                audio_file = self.text_to_speech(digit_text)
                if audio_file:
                    self.agi.stream_file(audio_file.replace('.wav', ''))
                    try:
                        os.unlink(audio_file)
                    except:
                        pass
            
            elif command == 'say_number':
                # Speak a number
                number = args[0] if args else '0'
                try:
                    num_int = int(number)
                    text = self.number_to_text(num_int)
                    
                    audio_file = self.text_to_speech(text)
                    if audio_file:
                        self.agi.stream_file(audio_file.replace('.wav', ''))
                        try:
                            os.unlink(audio_file)
                        except:
                            pass
                except ValueError:
                    self.agi.verbose(f"Invalid number: {number}", 1)
            
            elif command == 'say_time':
                # Speak current time
                import datetime
                now = datetime.datetime.now()
                time_text = now.strftime("The current time is %I %M %p")
                
                audio_file = self.text_to_speech(time_text)
                if audio_file:
                    self.agi.stream_file(audio_file.replace('.wav', ''))
                    try:
                        os.unlink(audio_file)
                    except:
                        pass
            
            elif command == 'cleanup':
                # Clean up temporary files
                self.cleanup()
                
            else:
                self.agi.verbose(f"Unknown command: {command}", 1)
                
        except Exception as e:
            self.agi.verbose(f"Command error: {e}", 1)
    
    def number_to_text(self, number):
        """Convert number to text (basic implementation)"""
        # This is a simplified version - extend for full number support
        if number == 0:
            return "zero"
        elif 1 <= number <= 20:
            ones = ["", "one", "two", "three", "four", "five", "six", "seven", 
                   "eight", "nine", "ten", "eleven", "twelve", "thirteen", 
                   "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", 
                   "nineteen", "twenty"]
            return ones[number]
        elif 21 <= number <= 99:
            tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", 
                   "seventy", "eighty", "ninety"]
            ones = ["", "one", "two", "three", "four", "five", "six", "seven", 
                   "eight", "nine"]
            
            ten_digit = number // 10
            one_digit = number % 10
            
            if one_digit == 0:
                return tens[ten_digit]
            else:
                return f"{tens[ten_digit]} {ones[one_digit]}"
        else:
            # For larger numbers, fall back to digit-by-digit
            return ' '.join(list(str(number)))
    
    def cleanup(self):
        """Clean up resources"""
        try:
            # Remove temporary directory
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
            
            # Shutdown VoiRS engine
            if self.engine:
                self.engine.shutdown()
                
        except Exception as e:
            self.agi.verbose(f"Cleanup error: {e}", 1)

def main():
    """Main AGI entry point"""
    try:
        voirs_agi = VoiRSAsteriskAGI()
        
        # Get command from AGI environment or command line
        if len(sys.argv) > 1:
            command = sys.argv[1]
            args = sys.argv[2:] if len(sys.argv) > 2 else []
        else:
            # Read from AGI channel
            command = voirs_agi.agi.get_variable('TTS_COMMAND') or 'say'
            args = []
        
        voirs_agi.handle_command(command, *args)
        voirs_agi.cleanup()
        
    except Exception as e:
        agi = AGI()
        agi.verbose(f"VoiRS AGI error: {e}", 1)
        sys.exit(1)

if __name__ == "__main__":
    main()
```

#### Asterisk Dialplan Configuration

```ini
; extensions.conf - VoiRS TTS integration

[voirs-tts]
; Basic TTS
exten => _X.,1,Answer()
same => n,Set(TTS_TEXT=${ARG1})
same => n,Set(TTS_VOICE=${ARG2})
same => n,AGI(voirs_agi.py,say)
same => n,Return()

; TTS with voice selection
exten => _X.,1,Answer()
same => n,Set(TTS_TEXT=${ARG1})
same => n,Set(TTS_VOICE=${IF($["${ARG2}" != ""]?${ARG2}:default)})
same => n,AGI(voirs_agi.py,say)
same => n,Return()

[main-menu]
; Example IVR menu with TTS
exten => s,1,Answer()
same => n,Set(TTS_TEXT=Welcome to our automated system. Press 1 for sales, 2 for support, or 3 for billing.)
same => n,Gosub(voirs-tts,s,1(${TTS_TEXT},female_warm))
same => n,WaitExten(10)
same => n,Set(TTS_TEXT=I'm sorry, I didn't receive your selection. Please try again.)
same => n,Gosub(voirs-tts,s,1(${TTS_TEXT},female_warm))
same => n,Goto(s,1)

exten => 1,1,Set(TTS_TEXT=You have selected sales. Please hold while we connect you.)
same => n,Gosub(voirs-tts,s,1(${TTS_TEXT},female_warm))
same => n,Dial(SIP/sales,30)
same => n,Hangup()

exten => 2,1,Set(TTS_TEXT=You have selected technical support. Please hold.)
same => n,Gosub(voirs-tts,s,1(${TTS_TEXT},male_young))
same => n,Dial(SIP/support,30)
same => n,Hangup()

exten => 3,1,Set(TTS_TEXT=You have selected billing. Please hold.)
same => n,Gosub(voirs-tts,s,1(${TTS_TEXT},default))
same => n,Dial(SIP/billing,30)
same => n,Hangup()

exten => i,1,Set(TTS_TEXT=Invalid selection. Please press 1, 2, or 3.)
same => n,Gosub(voirs-tts,s,1(${TTS_TEXT},female_warm))
same => n,Goto(s,6)

exten => t,1,Set(TTS_TEXT=Thank you for calling. Goodbye.)
same => n,Gosub(voirs-tts,s,1(${TTS_TEXT},default))
same => n,Hangup()

[time-announcement]
; Time announcement service
exten => 8,1,Answer()
same => n,AGI(voirs_agi.py,say_time)
same => n,Hangup()

[number-reader]
; Read back numbers
exten => _9X.,1,Answer()
same => n,Set(NUMBER=${EXTEN:1})
same => n,AGI(voirs_agi.py,say_number,${NUMBER})
same => n,Hangup()
```

## Assistive Technology

### Screen Reader Integration

#### NVDA Add-on (Python)

```python
"""
VoiRS NVDA Add-on
Provides enhanced text-to-speech for NVDA screen reader using VoiRS FFI
"""

import globalPluginHandler
import speech
import config
import wx
import gui
import addonHandler
import voirs_ffi
import threading
import queue
import tempfile
import os
from scriptHandler import script

addonHandler.initTranslation()

class VoiRSSpeechDriver:
    """VoiRS speech driver for NVDA"""
    
    name = "voirs"
    description = "VoiRS High-Quality TTS"
    
    def __init__(self):
        self.engine = None
        self.voice_id = "default"
        self.rate = 50  # 0-100 scale
        self.volume = 100  # 0-100 scale
        self.pitch = 50  # 0-100 scale
        self.initialized = False
        self.speech_queue = queue.Queue()
        self.worker_thread = None
        self.stop_speech = threading.Event()
        
        self.initialize()
    
    def initialize(self):
        """Initialize VoiRS engine"""
        try:
            self.engine = voirs_ffi.Engine()
            
            # Configure for screen reader use
            config = voirs_ffi.SynthesisConfig(
                quality=voirs_ffi.Quality.HIGH,
                thread_count=1,  # Single thread for responsiveness
                cache_size=512 * 1024,  # 512KB cache
                output_format=voirs_ffi.Format.WAV
            )
            
            if self.engine.initialize(config):
                self.initialized = True
                self.start_worker()
                return True
            else:
                return False
                
        except Exception as e:
            log.error(f"VoiRS initialization failed: {e}")
            return False
    
    def start_worker(self):
        """Start speech processing worker thread"""
        self.worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.worker_thread.start()
    
    def _speech_worker(self):
        """Worker thread for processing speech requests"""
        while not self.stop_speech.is_set():
            try:
                # Get speech request with timeout
                request = self.speech_queue.get(timeout=0.1)
                
                if request is None:  # Shutdown signal
                    break
                
                text, priority = request
                
                # Check if speech should be interrupted
                if self.stop_speech.is_set():
                    continue
                
                # Synthesize text
                config = voirs_ffi.SynthesisConfig(
                    voice_id=self.voice_id,
                    speed=self.rate_to_speed(self.rate),
                    volume=self.volume / 100.0,
                    quality=voirs_ffi.Quality.MEDIUM  # Balance quality/speed
                )
                
                result = self.engine.synthesize(text, config)
                
                if result.success and not self.stop_speech.is_set():
                    # Play audio using NVDA's audio system
                    self.play_audio(result.audio_data, result.sample_rate)
                
            except queue.Empty:
                continue
            except Exception as e:
                log.error(f"Speech worker error: {e}")
    
    def speak(self, text, priority=speech.priorities.SPRI_NORMAL):
        """Queue text for speech synthesis"""
        if not self.initialized:
            return
        
        # Clear queue for high priority speech
        if priority >= speech.priorities.SPRI_HIGH:
            self.cancel()
        
        # Add to queue
        self.speech_queue.put((text, priority))
    
    def cancel(self):
        """Cancel current and queued speech"""
        # Clear queue
        while not self.speech_queue.empty():
            try:
                self.speech_queue.get_nowait()
            except queue.Empty:
                break
        
        # Stop current speech
        self.stop_speech.set()
        # Reset immediately for next speech
        self.stop_speech.clear()
    
    def rate_to_speed(self, rate):
        """Convert NVDA rate (0-100) to VoiRS speed (0.1-3.0)"""
        # Map 0-100 to 0.3-2.0 (reasonable speed range)
        return 0.3 + (rate / 100.0) * 1.7
    
    def play_audio(self, audio_data, sample_rate):
        """Play audio through NVDA's audio system"""
        try:
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_file.write(audio_data)
            temp_file.close()
            
            # Use NVDA's audio system
            import nvwave
            nvwave.playWaveFile(temp_file.name, asynchronous=True)
            
            # Schedule cleanup
            def cleanup():
                try:
                    os.unlink(temp_file.name)
                except:
                    pass
            
            # Cleanup after a delay
            threading.Timer(5.0, cleanup).start()
            
        except Exception as e:
            log.error(f"Audio playback error: {e}")
    
    @property
    def availableVoices(self):
        """Get available voices"""
        try:
            voices = voirs_ffi.get_available_voices()
            return [VoiceInfo(voice, voice) for voice in voices]
        except:
            return [VoiceInfo("default", "Default Voice")]
    
    @property
    def voice(self):
        """Get current voice ID"""
        return self.voice_id
    
    @voice.setter
    def voice(self, voice_id):
        """Set current voice ID"""
        self.voice_id = voice_id
    
    def terminate(self):
        """Cleanup resources"""
        self.stop_speech.set()
        
        # Signal worker to stop
        self.speech_queue.put(None)
        
        if self.worker_thread:
            self.worker_thread.join(timeout=1.0)
        
        if self.engine:
            self.engine.shutdown()

class VoiceInfo:
    """Voice information class"""
    def __init__(self, id, name):
        self.id = id
        self.name = name

class GlobalPlugin(globalPluginHandler.GlobalPlugin):
    """VoiRS NVDA Global Plugin"""
    
    def __init__(self):
        super().__init__()
        
        # Add VoiRS to available speech drivers
        if hasattr(speech, 'synthDriverHandler'):
            speech.synthDriverHandler.synthDrivers['voirs'] = VoiRSSpeechDriver
        
        # Create settings panel
        self.create_settings_panel()
    
    def create_settings_panel(self):
        """Create VoiRS settings panel"""
        try:
            # Add to NVDA settings
            from gui.settingsDialogs import SettingsPanel
            
            class VoiRSSettingsPanel(SettingsPanel):
                title = _("VoiRS")
                
                def makeSettings(self, settingsSizer):
                    # Voice selection
                    voice_label = wx.StaticText(self, label=_("Voice:"))
                    settingsSizer.Add(voice_label)
                    
                    self.voice_choice = wx.Choice(self)
                    try:
                        voices = voirs_ffi.get_available_voices()
                        for voice in voices:
                            self.voice_choice.Append(voice)
                    except:
                        self.voice_choice.Append("default")
                    
                    settingsSizer.Add(self.voice_choice)
                    
                    # Quality selection
                    quality_label = wx.StaticText(self, label=_("Quality:"))
                    settingsSizer.Add(quality_label)
                    
                    self.quality_choice = wx.Choice(self)
                    qualities = ["Low", "Medium", "High", "Ultra"]
                    for quality in qualities:
                        self.quality_choice.Append(quality)
                    self.quality_choice.SetSelection(2)  # Default to High
                    
                    settingsSizer.Add(self.quality_choice)
                    
                    # Enable VoiRS checkbox
                    self.enable_checkbox = wx.CheckBox(self, label=_("Use VoiRS as default speech synthesizer"))
                    settingsSizer.Add(self.enable_checkbox)
                
                def onSave(self):
                    # Save VoiRS settings
                    config.conf["speech"]["synth"] = "voirs"
                    
            # Register settings panel
            gui.settingsDialogs.NVDASettingsDialog.categoryClasses.append(VoiRSSettingsPanel)
            
        except Exception as e:
            log.error(f"Settings panel creation failed: {e}")
    
    @script(
        description=_("Toggle VoiRS speech synthesizer"),
        category=_("VoiRS"),
    )
    def script_toggleVoiRS(self, gesture):
        """Toggle between VoiRS and default synthesizer"""
        current_synth = speech.getSynth().name
        
        if current_synth == "voirs":
            # Switch to default
            speech.setSynth(config.conf["speech"]["defaultSynth"])
            speech.speakMessage(_("Switched to default synthesizer"))
        else:
            # Switch to VoiRS
            speech.setSynth("voirs")
            speech.speakMessage(_("Switched to VoiRS high-quality synthesis"))
    
    @script(
        description=_("Announce VoiRS status"),
        category=_("VoiRS"),
    )
    def script_announceStatus(self, gesture):
        """Announce VoiRS status"""
        try:
            current_synth = speech.getSynth()
            if current_synth.name == "voirs":
                voice = getattr(current_synth, 'voice_id', 'unknown')
                speech.speakMessage(f"VoiRS active with voice {voice}")
            else:
                speech.speakMessage("VoiRS not currently active")
        except Exception as e:
            speech.speakMessage("VoiRS status unavailable")
    
    def terminate(self):
        """Plugin cleanup"""
        try:
            # Remove from speech drivers
            if hasattr(speech, 'synthDriverHandler') and 'voirs' in speech.synthDriverHandler.synthDrivers:
                del speech.synthDriverHandler.synthDrivers['voirs']
        except:
            pass
        
        super().terminate()
```

#### Installation Script (install.py)

```python
"""
VoiRS NVDA Add-on Installation Script
"""

import os
import sys
import shutil
import tempfile
import zipfile
from pathlib import Path

def create_addon_package():
    """Create NVDA add-on package"""
    
    # Add-on manifest
    manifest = """name = VoiRS
summary = High-Quality Text-to-Speech using VoiRS FFI
description = Provides enhanced text-to-speech synthesis for NVDA using the VoiRS FFI library for improved audio quality and naturalness.
version = 1.0.0
author = VoiRS Team
url = https://github.com/voirs/voirs-ffi
minimumNVDAVersion = 2020.1
lastTestedNVDAVersion = 2023.1
"""
    
    # Create temporary directory for add-on
    temp_dir = tempfile.mkdtemp()
    addon_dir = os.path.join(temp_dir, "voirs")
    os.makedirs(addon_dir)
    
    # Copy files
    globalPlugins_dir = os.path.join(addon_dir, "globalPlugins")
    os.makedirs(globalPlugins_dir)
    
    # Write manifest
    with open(os.path.join(addon_dir, "manifest.ini"), "w") as f:
        f.write(manifest)
    
    # Copy VoiRS library (you'll need to provide the actual library path)
    # shutil.copy("path/to/voirs_ffi.dll", os.path.join(addon_dir, "lib"))
    
    # Create .nvda-addon file
    addon_file = "voirs.nvda-addon"
    with zipfile.ZipFile(addon_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(addon_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arc_path = os.path.relpath(file_path, temp_dir)
                zf.write(file_path, arc_path)
    
    # Cleanup
    shutil.rmtree(temp_dir)
    
    print(f"NVDA add-on created: {addon_file}")
    print("Install by opening this file in NVDA or copying to add-ons folder")

if __name__ == "__main__":
    create_addon_package()
```

## Interactive Voice Response (IVR)

### Advanced IVR System

#### IVR Engine (Python)

```python
"""
Advanced IVR System with VoiRS Integration
Provides interactive voice response with high-quality TTS
"""

import asyncio
import json
import logging
import re
import time
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Any
from enum import Enum

import voirs_ffi
from telephony_interface import TelephonyInterface  # Hypothetical telephony library

class IVRState(Enum):
    INITIALIZING = "initializing"
    PLAYING_PROMPT = "playing_prompt"
    WAITING_INPUT = "waiting_input"
    PROCESSING = "processing"
    TRANSFERRING = "transferring"
    ENDING = "ending"

@dataclass
class IVRPrompt:
    text: str
    voice_id: str = "default"
    timeout: float = 10.0
    retries: int = 2
    valid_inputs: Optional[List[str]] = None
    input_pattern: Optional[str] = None

@dataclass
class IVRMenuOption:
    key: str
    prompt: IVRPrompt
    action: str
    parameters: Dict[str, Any] = None

class VoiRSIVREngine:
    """Advanced IVR engine with VoiRS TTS integration"""
    
    def __init__(self, config_file: str):
        self.voirs_engine = None
        self.telephony = TelephonyInterface()
        self.sessions = {}
        self.menus = {}
        self.variables = {}
        self.call_flows = {}
        
        self.load_configuration(config_file)
        self.initialize_voirs()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def load_configuration(self, config_file: str):
        """Load IVR configuration from JSON file"""
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Load menus
        for menu_id, menu_config in config.get('menus', {}).items():
            self.menus[menu_id] = self.parse_menu_config(menu_config)
        
        # Load call flows
        self.call_flows = config.get('call_flows', {})
        
        # Load global variables
        self.variables = config.get('variables', {})
    
    def parse_menu_config(self, menu_config: Dict) -> Dict[str, IVRMenuOption]:
        """Parse menu configuration"""
        menu = {}
        
        for option_key, option_config in menu_config.items():
            prompt = IVRPrompt(
                text=option_config['text'],
                voice_id=option_config.get('voice_id', 'default'),
                timeout=option_config.get('timeout', 10.0),
                retries=option_config.get('retries', 2),
                valid_inputs=option_config.get('valid_inputs'),
                input_pattern=option_config.get('input_pattern')
            )
            
            menu[option_key] = IVRMenuOption(
                key=option_key,
                prompt=prompt,
                action=option_config['action'],
                parameters=option_config.get('parameters', {})
            )
        
        return menu
    
    def initialize_voirs(self):
        """Initialize VoiRS engine for IVR use"""
        try:
            self.voirs_engine = voirs_ffi.Engine()
            
            # IVR-optimized configuration
            config = voirs_ffi.SynthesisConfig(
                quality=voirs_ffi.Quality.HIGH,
                sample_rate=8000,  # Telephony standard
                thread_count=4,    # Handle multiple calls
                cache_size=2 * 1024 * 1024,  # 2MB cache for frequently used prompts
                output_format=voirs_ffi.Format.WAV
            )
            
            if self.voirs_engine.initialize(config):
                self.logger.info("VoiRS engine initialized for IVR")
            else:
                raise Exception("VoiRS initialization failed")
                
        except Exception as e:
            self.logger.error(f"VoiRS initialization error: {e}")
            raise
    
    async def handle_call(self, call_id: str, caller_number: str):
        """Handle incoming call"""
        session = IVRSession(call_id, caller_number, self)
        self.sessions[call_id] = session
        
        try:
            await session.start()
        except Exception as e:
            self.logger.error(f"Call handling error for {call_id}: {e}")
        finally:
            # Cleanup session
            if call_id in self.sessions:
                del self.sessions[call_id]
    
    async def synthesize_prompt(self, text: str, voice_id: str = "default") -> bytes:
        """Synthesize TTS prompt with caching"""
        # Create cache key
        cache_key = f"{text}_{voice_id}"
        
        # Check cache first (implement caching as needed)
        
        try:
            config = voirs_ffi.SynthesisConfig(
                voice_id=voice_id,
                quality=voirs_ffi.Quality.HIGH,
                speed=0.9,  # Slightly slower for phone clarity
                volume=1.0
            )
            
            result = self.voirs_engine.synthesize(text, config)
            
            if result.success:
                return result.audio_data
            else:
                raise Exception("TTS synthesis failed")
                
        except Exception as e:
            self.logger.error(f"TTS synthesis error: {e}")
            raise
    
    def get_current_time_prompt(self) -> str:
        """Generate current time announcement"""
        now = datetime.now()
        return f"The current time is {now.strftime('%I:%M %p')}"
    
    def get_business_hours_prompt(self) -> str:
        """Generate business hours information"""
        now = datetime.now()
        
        # Simple business hours logic
        if now.weekday() < 5 and 9 <= now.hour < 17:
            return "Our office is currently open. Business hours are Monday through Friday, 9 AM to 5 PM."
        else:
            return "Our office is currently closed. Business hours are Monday through Friday, 9 AM to 5 PM. Please call back during business hours or leave a message."

class IVRSession:
    """Individual IVR call session"""
    
    def __init__(self, call_id: str, caller_number: str, engine: VoiRSIVREngine):
        self.call_id = call_id
        self.caller_number = caller_number
        self.engine = engine
        self.state = IVRState.INITIALIZING
        self.current_menu = None
        self.input_buffer = ""
        self.variables = {}
        self.call_start_time = time.time()
        
        self.logger = logging.getLogger(f"{__name__}.{call_id}")
    
    async def start(self):
        """Start IVR session"""
        self.logger.info(f"Starting IVR session for {self.caller_number}")
        
        try:
            # Answer call
            await self.engine.telephony.answer_call(self.call_id)
            
            # Start with main menu
            await self.go_to_menu("main_menu")
            
            # Main session loop
            while self.state != IVRState.ENDING:
                await self.process_session()
                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
                
        except Exception as e:
            self.logger.error(f"Session error: {e}")
        finally:
            await self.end_session()
    
    async def process_session(self):
        """Process current session state"""
        if self.state == IVRState.PLAYING_PROMPT:
            # Wait for prompt to finish
            if not await self.engine.telephony.is_playing_audio(self.call_id):
                self.state = IVRState.WAITING_INPUT
        
        elif self.state == IVRState.WAITING_INPUT:
            # Check for DTMF input
            input_data = await self.engine.telephony.get_dtmf_input(self.call_id)
            if input_data:
                await self.handle_input(input_data)
        
        elif self.state == IVRState.PROCESSING:
            # Processing is handled by specific action methods
            pass
    
    async def go_to_menu(self, menu_id: str):
        """Navigate to specified menu"""
        if menu_id not in self.engine.menus:
            self.logger.error(f"Menu {menu_id} not found")
            await self.end_session()
            return
        
        self.current_menu = menu_id
        menu = self.engine.menus[menu_id]
        
        # Build menu prompt
        if menu_id == "main_menu":
            prompt_text = self.build_main_menu_prompt(menu)
        else:
            prompt_text = self.build_menu_prompt(menu)
        
        await self.play_prompt(prompt_text)
    
    def build_main_menu_prompt(self, menu: Dict[str, IVRMenuOption]) -> str:
        """Build main menu prompt"""
        greeting = f"Thank you for calling. Your call is important to us."
        
        # Add business hours info if after hours
        if not self.is_business_hours():
            greeting += " " + self.engine.get_business_hours_prompt()
        
        # Build options
        options = []
        for key, option in sorted(menu.items()):
            if key.isdigit():
                options.append(f"Press {key} {option.prompt.text}")
        
        return greeting + " " + " ".join(options) + " Please make your selection."
    
    def build_menu_prompt(self, menu: Dict[str, IVRMenuOption]) -> str:
        """Build standard menu prompt"""
        options = []
        for key, option in sorted(menu.items()):
            if key.isdigit():
                options.append(f"Press {key} {option.prompt.text}")
        
        return " ".join(options) + " Please make your selection."
    
    async def play_prompt(self, text: str, voice_id: str = "default"):
        """Play TTS prompt"""
        try:
            self.state = IVRState.PLAYING_PROMPT
            
            # Synthesize audio
            audio_data = await self.engine.synthesize_prompt(text, voice_id)
            
            # Play audio
            await self.engine.telephony.play_audio(self.call_id, audio_data)
            
            self.logger.info(f"Playing prompt: {text[:50]}...")
            
        except Exception as e:
            self.logger.error(f"Prompt playback error: {e}")
            await self.end_session()
    
    async def handle_input(self, input_data: str):
        """Handle DTMF input"""
        self.state = IVRState.PROCESSING
        
        # Add to input buffer
        self.input_buffer += input_data
        
        # Check if we have a valid menu selection
        if self.current_menu and input_data in self.engine.menus[self.current_menu]:
            option = self.engine.menus[self.current_menu][input_data]
            await self.execute_action(option.action, option.parameters)
        else:
            # Invalid input
            await self.handle_invalid_input()
    
    async def execute_action(self, action: str, parameters: Dict[str, Any]):
        """Execute menu action"""
        try:
            if action == "transfer":
                await self.transfer_call(parameters.get("destination"))
            elif action == "play_message":
                await self.play_message(parameters.get("message"))
            elif action == "go_to_menu":
                await self.go_to_menu(parameters.get("menu_id"))
            elif action == "collect_input":
                await self.collect_input(parameters)
            elif action == "say_time":
                await self.say_current_time()
            elif action == "say_business_hours":
                await self.say_business_hours()
            elif action == "end_call":
                await self.end_session()
            else:
                self.logger.warning(f"Unknown action: {action}")
                await self.handle_invalid_input()
                
        except Exception as e:
            self.logger.error(f"Action execution error: {e}")
            await self.handle_invalid_input()
    
    async def transfer_call(self, destination: str):
        """Transfer call to destination"""
        self.state = IVRState.TRANSFERRING
        
        transfer_message = f"Please hold while I transfer your call to {destination}."
        await self.play_prompt(transfer_message)
        
        # Perform transfer
        success = await self.engine.telephony.transfer_call(self.call_id, destination)
        
        if success:
            self.logger.info(f"Call transferred to {destination}")
            await self.end_session()
        else:
            error_message = "I'm sorry, the transfer failed. Please try again or hold for an operator."
            await self.play_prompt(error_message)
            await self.go_to_menu("main_menu")
    
    async def play_message(self, message: str):
        """Play custom message"""
        await self.play_prompt(message)
        self.state = IVRState.WAITING_INPUT
    
    async def collect_input(self, parameters: Dict[str, Any]):
        """Collect multi-digit input"""
        prompt = parameters.get("prompt", "Please enter your input followed by the pound key.")
        max_digits = parameters.get("max_digits", 10)
        timeout = parameters.get("timeout", 30)
        
        await self.play_prompt(prompt)
        
        # Collect input until # or timeout
        collected_input = ""
        start_time = time.time()
        
        while (time.time() - start_time) < timeout:
            input_data = await self.engine.telephony.get_dtmf_input(self.call_id, timeout=1)
            
            if input_data == "#":
                break
            elif input_data and input_data.isdigit():
                collected_input += input_data
                if len(collected_input) >= max_digits:
                    break
        
        # Process collected input
        if collected_input:
            await self.process_collected_input(collected_input, parameters)
        else:
            await self.handle_timeout()
    
    async def process_collected_input(self, input_data: str, parameters: Dict[str, Any]):
        """Process collected input data"""
        # Store in session variables
        variable_name = parameters.get("variable", "collected_input")
        self.variables[variable_name] = input_data
        
        # Execute next action
        next_action = parameters.get("next_action")
        if next_action:
            await self.execute_action(next_action["action"], next_action.get("parameters", {}))
        else:
            await self.go_to_menu("main_menu")
    
    async def say_current_time(self):
        """Announce current time"""
        time_prompt = self.engine.get_current_time_prompt()
        await self.play_prompt(time_prompt)
        self.state = IVRState.WAITING_INPUT
    
    async def say_business_hours(self):
        """Announce business hours"""
        hours_prompt = self.engine.get_business_hours_prompt()
        await self.play_prompt(hours_prompt)
        self.state = IVRState.WAITING_INPUT
    
    async def handle_invalid_input(self):
        """Handle invalid input"""
        error_message = "I'm sorry, that's not a valid selection. Please try again."
        await self.play_prompt(error_message)
        
        # Return to current menu
        if self.current_menu:
            await self.go_to_menu(self.current_menu)
        else:
            await self.go_to_menu("main_menu")
    
    async def handle_timeout(self):
        """Handle input timeout"""
        timeout_message = "I didn't receive your selection. Please try again."
        await self.play_prompt(timeout_message)
        await self.go_to_menu("main_menu")
    
    def is_business_hours(self) -> bool:
        """Check if current time is within business hours"""
        now = datetime.now()
        return now.weekday() < 5 and 9 <= now.hour < 17
    
    async def end_session(self):
        """End IVR session"""
        self.state = IVRState.ENDING
        
        goodbye_message = "Thank you for calling. Goodbye."
        await self.play_prompt(goodbye_message)
        
        # Hangup call
        await self.engine.telephony.hangup_call(self.call_id)
        
        # Log session stats
        duration = time.time() - self.call_start_time
        self.logger.info(f"Session ended for {self.caller_number}, duration: {duration:.2f}s")

# Example configuration file (ivr_config.json)
ivr_config_example = {
    "menus": {
        "main_menu": {
            "1": {
                "text": "for sales",
                "action": "transfer",
                "parameters": {"destination": "sales@company.com"},
                "voice_id": "female_warm"
            },
            "2": {
                "text": "for technical support",
                "action": "go_to_menu",
                "parameters": {"menu_id": "support_menu"},
                "voice_id": "male_young"
            },
            "3": {
                "text": "for billing inquiries",
                "action": "transfer",
                "parameters": {"destination": "billing@company.com"}
            },
            "4": {
                "text": "to hear our business hours",
                "action": "say_business_hours"
            },
            "5": {
                "text": "to hear the current time",
                "action": "say_time"
            },
            "0": {
                "text": "to speak with an operator",
                "action": "transfer",
                "parameters": {"destination": "operator@company.com"}
            }
        },
        "support_menu": {
            "1": {
                "text": "for password reset",
                "action": "go_to_menu",
                "parameters": {"menu_id": "password_reset"}
            },
            "2": {
                "text": "for technical issues",
                "action": "transfer",
                "parameters": {"destination": "tech@company.com"}
            },
            "9": {
                "text": "to return to the main menu",
                "action": "go_to_menu", 
                "parameters": {"menu_id": "main_menu"}
            }
        }
    },
    "variables": {
        "company_name": "Acme Corporation",
        "support_hours": "Monday through Friday, 8 AM to 6 PM"
    }
}

async def main():
    """Main IVR application"""
    # Create configuration file
    with open('ivr_config.json', 'w') as f:
        json.dump(ivr_config_example, f, indent=2)
    
    # Initialize IVR engine
    ivr_engine = VoiRSIVREngine('ivr_config.json')
    
    # Start telephony interface
    await ivr_engine.telephony.start()
    
    # Handle incoming calls
    async def call_handler(call_id: str, caller_number: str):
        await ivr_engine.handle_call(call_id, caller_number)
    
    ivr_engine.telephony.set_call_handler(call_handler)
    
    print("VoiRS IVR system started. Press Ctrl+C to stop.")
    
    try:
        # Keep running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down IVR system...")
    finally:
        await ivr_engine.telephony.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

## Real-Time Broadcasting

### Live Streaming Integration

#### OBS Studio Plugin (C++)

```cpp
// obs_voirs_plugin.cpp - OBS Studio plugin for VoiRS TTS
#include <obs-module.h>
#include <obs-frontend-api.h>
#include <util/platform.h>
#include <QWidget>
#include <QVBoxLayout>
#include <QTextEdit>
#include <QPushButton>
#include <QComboBox>
#include <QSlider>
#include <QLabel>
#include <QTimer>
#include <queue>
#include <mutex>
#include <thread>

extern "C" {
    // VoiRS FFI declarations
    void* voirs_pipeline_create(const struct VoiRSSynthesisConfig* config);
    void* voirs_synthesize(void* pipeline, const char* text, const struct VoiRSSynthesisConfig* config);
    void voirs_synthesis_result_destroy(void* result);
    void voirs_pipeline_destroy(void* pipeline);
    const void* voirs_synthesis_result_get_audio_data(void* result);
    int32_t voirs_synthesis_result_get_audio_size(void* result);
    int32_t voirs_synthesis_result_get_sample_rate(void* result);
}

struct VoiRSSynthesisConfig {
    int32_t quality;
    float speed;
    float volume;
    int32_t output_format;
    const char* voice_id;
};

class VoiRSSource {
private:
    obs_source_t* source;
    void* voirs_pipeline;
    std::queue<std::string> text_queue;
    std::mutex queue_mutex;
    std::thread synthesis_thread;
    bool running;
    
    // Audio output
    obs_source_audio audio_output;
    std::vector<uint8_t> audio_buffer;
    
public:
    VoiRSSource(obs_source_t* source, obs_data_t* settings) 
        : source(source), voirs_pipeline(nullptr), running(false) {
        
        // Initialize VoiRS
        if (initialize_voirs()) {
            running = true;
            synthesis_thread = std::thread(&VoiRSSource::synthesis_worker, this);
        }
        
        // Setup audio output format
        audio_output.format = AUDIO_FORMAT_16BIT;
        audio_output.samples_per_sec = 16000; // Streaming quality
        audio_output.speakers = SPEAKERS_MONO;
    }
    
    ~VoiRSSource() {
        running = false;
        if (synthesis_thread.joinable()) {
            synthesis_thread.join();
        }
        
        if (voirs_pipeline) {
            voirs_pipeline_destroy(voirs_pipeline);
        }
    }
    
    bool initialize_voirs() {
        VoiRSSynthesisConfig config;
        config.quality = 1; // Medium quality for streaming
        config.speed = 1.0f;
        config.volume = 1.0f;
        config.output_format = 0; // WAV
        config.voice_id = "default";
        
        voirs_pipeline = voirs_pipeline_create(&config);
        return voirs_pipeline != nullptr;
    }
    
    void queue_text(const std::string& text) {
        std::lock_guard<std::mutex> lock(queue_mutex);
        text_queue.push(text);
    }
    
    void synthesis_worker() {
        while (running) {
            std::string text;
            
            // Get text from queue
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                if (!text_queue.empty()) {
                    text = text_queue.front();
                    text_queue.pop();
                }
            }
            
            if (!text.empty()) {
                synthesize_and_output(text);
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    
    void synthesize_and_output(const std::string& text) {
        void* result = voirs_synthesize(voirs_pipeline, text.c_str(), nullptr);
        
        if (result) {
            const void* audio_data = voirs_synthesis_result_get_audio_data(result);
            int32_t audio_size = voirs_synthesis_result_get_audio_size(result);
            int32_t sample_rate = voirs_synthesis_result_get_sample_rate(result);
            
            // Convert to OBS audio format
            audio_buffer.resize(audio_size);
            memcpy(audio_buffer.data(), audio_data, audio_size);
            
            // Setup audio output
            audio_output.data[0] = audio_buffer.data();
            audio_output.frames = audio_size / sizeof(int16_t);
            audio_output.timestamp = os_gettime_ns();
            
            // Send to OBS
            obs_source_output_audio(source, &audio_output);
            
            voirs_synthesis_result_destroy(result);
        }
    }
    
    void update_settings(obs_data_t* settings) {
        // Update VoiRS settings based on OBS properties
        const char* voice_id = obs_data_get_string(settings, "voice_id");
        double speed = obs_data_get_double(settings, "speed");
        double volume = obs_data_get_double(settings, "volume");
        
        // Apply settings (would need to recreate pipeline or add config update)
    }
};

// OBS Plugin Interface
static const char* voirs_source_get_name(void* unused) {
    return "VoiRS Text-to-Speech";
}

static void* voirs_source_create(obs_data_t* settings, obs_source_t* source) {
    return new VoiRSSource(source, settings);
}

static void voirs_source_destroy(void* data) {
    delete static_cast<VoiRSSource*>(data);
}

static void voirs_source_update(void* data, obs_data_t* settings) {
    static_cast<VoiRSSource*>(data)->update_settings(settings);
}

static obs_properties_t* voirs_source_properties(void* data) {
    obs_properties_t* props = obs_properties_create();
    
    // Voice selection
    obs_property_t* voice_prop = obs_properties_add_list(props, "voice_id", "Voice",
                                                        OBS_COMBO_TYPE_LIST,
                                                        OBS_COMBO_FORMAT_STRING);
    obs_property_list_add_string(voice_prop, "Default", "default");
    obs_property_list_add_string(voice_prop, "Male Young", "male_young");
    obs_property_list_add_string(voice_prop, "Female Young", "female_young");
    obs_property_list_add_string(voice_prop, "Male Deep", "male_deep");
    obs_property_list_add_string(voice_prop, "Female Warm", "female_warm");
    
    // Speed control
    obs_properties_add_float_slider(props, "speed", "Speed", 0.5, 2.0, 0.1);
    
    // Volume control
    obs_properties_add_float_slider(props, "volume", "Volume", 0.0, 2.0, 0.1);
    
    return props;
}

// VoiRS Control Widget for OBS
class VoiRSWidget : public QWidget {
    Q_OBJECT
    
private:
    QTextEdit* text_input;
    QPushButton* speak_button;
    QPushButton* clear_button;
    QComboBox* voice_combo;
    QSlider* speed_slider;
    QSlider* volume_slider;
    obs_source_t* voirs_source;
    
public:
    VoiRSWidget(QWidget* parent = nullptr) : QWidget(parent) {
        setupUI();
        connectSignals();
        findVoiRSSource();
    }
    
private slots:
    void onSpeakClicked() {
        QString text = text_input->toPlainText().trimmed();
        if (!text.isEmpty() && voirs_source) {
            VoiRSSource* source_data = static_cast<VoiRSSource*>(
                obs_source_get_type_data(voirs_source));
            
            if (source_data) {
                source_data->queue_text(text.toStdString());
            }
        }
    }
    
    void onClearClicked() {
        text_input->clear();
    }
    
private:
    void setupUI() {
        setWindowTitle("VoiRS TTS Control");
        setFixedSize(400, 300);
        
        QVBoxLayout* layout = new QVBoxLayout(this);
        
        // Text input
        text_input = new QTextEdit();
        text_input->setPlaceholderText("Enter text to synthesize...");
        text_input->setMaximumHeight(100);
        layout->addWidget(text_input);
        
        // Voice selection
        layout->addWidget(new QLabel("Voice:"));
        voice_combo = new QComboBox();
        voice_combo->addItems({"Default", "Male Young", "Female Young", "Male Deep", "Female Warm"});
        layout->addWidget(voice_combo);
        
        // Speed control
        layout->addWidget(new QLabel("Speed:"));
        speed_slider = new QSlider(Qt::Horizontal);
        speed_slider->setRange(50, 200);
        speed_slider->setValue(100);
        layout->addWidget(speed_slider);
        
        // Volume control
        layout->addWidget(new QLabel("Volume:"));
        volume_slider = new QSlider(Qt::Horizontal);
        volume_slider->setRange(0, 200);
        volume_slider->setValue(100);
        layout->addWidget(volume_slider);
        
        // Buttons
        QHBoxLayout* button_layout = new QHBoxLayout();
        speak_button = new QPushButton("Speak");
        clear_button = new QPushButton("Clear");
        button_layout->addWidget(speak_button);
        button_layout->addWidget(clear_button);
        layout->addLayout(button_layout);
    }
    
    void connectSignals() {
        connect(speak_button, &QPushButton::clicked, this, &VoiRSWidget::onSpeakClicked);
        connect(clear_button, &QPushButton::clicked, this, &VoiRSWidget::onClearClicked);
        
        // Enable Enter key for speaking
        connect(text_input, &QTextEdit::textChanged, [this]() {
            speak_button->setEnabled(!text_input->toPlainText().trimmed().isEmpty());
        });
    }
    
    void findVoiRSSource() {
        // Find VoiRS source in current scene
        obs_source_t* scene = obs_frontend_get_current_scene();
        if (scene) {
            obs_scene_t* obs_scene = obs_scene_from_source(scene);
            obs_scene_enum_items(obs_scene, [](obs_scene_t*, obs_sceneitem_t* item, void* param) {
                VoiRSWidget* widget = static_cast<VoiRSWidget*>(param);
                obs_source_t* source = obs_sceneitem_get_source(item);
                
                if (strcmp(obs_source_get_id(source), "voirs_source") == 0) {
                    widget->voirs_source = source;
                    return false; // Stop enumeration
                }
                return true; // Continue enumeration
            }, this);
            
            obs_source_release(scene);
        }
    }
};

// Plugin module info
static struct obs_source_info voirs_source_info = {
    .id = "voirs_source",
    .type = OBS_SOURCE_TYPE_INPUT,
    .output_flags = OBS_SOURCE_AUDIO,
    .get_name = voirs_source_get_name,
    .create = voirs_source_create,
    .destroy = voirs_source_destroy,
    .update = voirs_source_update,
    .get_properties = voirs_source_properties,
};

bool obs_module_load(void) {
    obs_register_source(&voirs_source_info);
    
    // Add dock widget
    obs_frontend_add_dock(new VoiRSWidget());
    
    return true;
}

void obs_module_unload(void) {
    // Cleanup handled by destructors
}

MODULE_EXPORT const char* obs_module_description(void) {
    return "VoiRS Text-to-Speech integration for OBS Studio";
}

MODULE_EXPORT const char* obs_module_name(void) {
    return "VoiRS TTS Plugin";
}

#include "obs_voirs_plugin.moc"
```

## Edge Computing Integration

### Raspberry Pi Real-Time System

#### Embedded Real-Time Controller (Rust)

```rust
// src/main.rs - Raspberry Pi real-time VoiRS controller
use std::sync::{Arc, Mutex};
use std::collections::VecDeque;
use std::time::{Duration, Instant};
use tokio::time::sleep;
use serde::{Deserialize, Serialize};

use voirs_ffi::{Engine, SynthesisConfig, Quality, Format};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeTTSRequest {
    pub text: String,
    pub voice_id: Option<String>,
    pub priority: Priority,
    pub timeout_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Priority {
    Critical,  // Emergency announcements
    High,      // Important notifications
    Normal,    // Standard messages
    Low,       // Background information
}

pub struct EdgeVoiRSController {
    engine: Engine,
    audio_device: cpal::Device,
    request_queue: Arc<Mutex<VecDeque<EdgeTTSRequest>>>,
    current_stream: Option<cpal::Stream>,
    performance_monitor: PerformanceMonitor,
}

#[derive(Debug)]
pub struct PerformanceMonitor {
    synthesis_times: VecDeque<Duration>,
    queue_depths: VecDeque<usize>,
    last_update: Instant,
}

impl EdgeVoiRSController {
    pub async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        // Initialize VoiRS engine with edge-optimized settings
        let mut engine = Engine::new();
        let config = SynthesisConfig {
            quality: Quality::Medium, // Balance quality vs. performance
            thread_count: 2,          // Conservative for Pi
            use_simd: true,           // Enable ARM NEON if available
            cache_size: 256 * 1024,   // 256KB cache
            output_format: Format::WAV,
            sample_rate: 16000,       // Lower rate for embedded use
        };
        
        if !engine.initialize(&config) {
            return Err("Failed to initialize VoiRS engine".into());
        }
        
        // Setup audio device
        let host = cpal::default_host();
        let audio_device = host.default_output_device()
            .ok_or("No audio output device available")?;
        
        println!("Using audio device: {}", audio_device.name()?);
        
        Ok(Self {
            engine,
            audio_device,
            request_queue: Arc::new(Mutex::new(VecDeque::new())),
            current_stream: None,
            performance_monitor: PerformanceMonitor::new(),
        })
    }
    
    pub async fn start(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("Starting VoiRS Edge Controller...");
        
        // Start HTTP server for requests
        let queue_clone = Arc::clone(&self.request_queue);
        tokio::spawn(async move {
            Self::start_http_server(queue_clone).await;
        });
        
        // Start GPIO monitoring (for physical buttons)
        tokio::spawn(async move {
            Self::monitor_gpio().await;
        });
        
        // Main processing loop
        self.processing_loop().await;
        
        Ok(())
    }
    
    async fn processing_loop(&mut self) {
        println!("VoiRS processing loop started");
        
        let mut last_performance_update = Instant::now();
        
        loop {
            // Process queued requests
            let request = {
                let mut queue = self.request_queue.lock().unwrap();
                queue.pop_front()
            };
            
            if let Some(req) = request {
                let start_time = Instant::now();
                
                match self.process_request(req).await {
                    Ok(synthesis_time) => {
                        self.performance_monitor.record_synthesis(synthesis_time);
                    }
                    Err(e) => {
                        eprintln!("Request processing error: {}", e);
                    }
                }
            }
            
            // Update performance metrics
            if last_performance_update.elapsed() > Duration::from_secs(5) {
                self.update_performance_metrics().await;
                last_performance_update = Instant::now();
            }
            
            // Short sleep to prevent busy waiting
            sleep(Duration::from_millis(10)).await;
        }
    }
    
    async fn process_request(&mut self, request: EdgeTTSRequest) -> Result<Duration, Box<dyn std::error::Error>> {
        let start_time = Instant::now();
        
        // Apply timeout for low-priority requests
        let timeout = match request.priority {
            Priority::Critical => None,
            Priority::High => Some(Duration::from_millis(2000)),
            Priority::Normal => Some(Duration::from_millis(5000)),
            Priority::Low => Some(Duration::from_millis(10000)),
        };
        
        // Prepare synthesis config
        let voice_id = request.voice_id.unwrap_or_else(|| "default".to_string());
        let config = SynthesisConfig {
            voice_id,
            quality: match request.priority {
                Priority::Critical | Priority::High => Quality::High,
                Priority::Normal => Quality::Medium,
                Priority::Low => Quality::Low,
            },
            speed: 1.0,
            volume: 1.0,
            ..Default::default()
        };
        
        // Perform synthesis with timeout
        let synthesis_result = if let Some(timeout_duration) = timeout {
            tokio::time::timeout(timeout_duration, async {
                self.engine.synthesize(&request.text, &config)
            }).await??
        } else {
            self.engine.synthesize(&request.text, &config)?
        };
        
        let synthesis_time = start_time.elapsed();
        
        if synthesis_result.success {
            // Play audio
            self.play_audio(&synthesis_result.audio_data, synthesis_result.sample_rate).await?;
            
            println!("Synthesized and played: '{}' ({:.2}ms)", 
                     request.text.chars().take(50).collect::<String>(),
                     synthesis_time.as_millis());
        } else {
            return Err("Synthesis failed".into());
        }
        
        Ok(synthesis_time)
    }
    
    async fn play_audio(&mut self, audio_data: &[u8], sample_rate: u32) -> Result<(), Box<dyn std::error::Error>> {
        // Convert raw audio data to f32 samples
        let samples: Vec<f32> = audio_data
            .chunks_exact(2)
            .map(|chunk| {
                let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                sample as f32 / 32768.0
            })
            .collect();
        
        // Setup audio stream
        let config = cpal::StreamConfig {
            channels: 1,
            sample_rate: cpal::SampleRate(sample_rate),
            buffer_size: cpal::BufferSize::Default,
        };
        
        let samples_arc = Arc::new(Mutex::new(samples));
        let sample_index = Arc::new(Mutex::new(0));
        
        let samples_clone = Arc::clone(&samples_arc);
        let index_clone = Arc::clone(&sample_index);
        
        let stream = self.audio_device.build_output_stream(
            &config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                let samples = samples_clone.lock().unwrap();
                let mut index = index_clone.lock().unwrap();
                
                for sample in data.iter_mut() {
                    if *index < samples.len() {
                        *sample = samples[*index];
                        *index += 1;
                    } else {
                        *sample = 0.0;
                    }
                }
            },
            move |err| {
                eprintln!("Audio stream error: {}", err);
            },
        )?;
        
        stream.play()?;
        
        // Wait for playback to complete
        let duration = Duration::from_secs_f64(samples_arc.lock().unwrap().len() as f64 / sample_rate as f64);
        sleep(duration).await;
        
        drop(stream);
        
        Ok(())
    }
    
    async fn start_http_server(queue: Arc<Mutex<VecDeque<EdgeTTSRequest>>>) {
        use warp::Filter;
        
        let tts_route = warp::path("tts")
            .and(warp::post())
            .and(warp::body::json())
            .and_then(move |request: EdgeTTSRequest| {
                let queue = Arc::clone(&queue);
                async move {
                    {
                        let mut q = queue.lock().unwrap();
                        q.push_back(request);
                    }
                    
                    Result::<_, warp::Rejection>::Ok(warp::reply::json(&serde_json::json!({
                        "status": "queued"
                    })))
                }
            });
        
        let status_route = warp::path("status")
            .and(warp::get())
            .map(|| {
                warp::reply::json(&serde_json::json!({
                    "status": "running",
                    "timestamp": chrono::Utc::now().to_rfc3339()
                }))
            });
        
        let routes = tts_route.or(status_route);
        
        println!("HTTP server starting on port 3030");
        warp::serve(routes)
            .run(([0, 0, 0, 0], 3030))
            .await;
    }
    
    async fn monitor_gpio() {
        // GPIO monitoring for physical buttons (placeholder)
        println!("GPIO monitoring started");
        
        // In a real implementation, you would use a GPIO library like rppal
        // to monitor physical buttons connected to the Raspberry Pi
        
        loop {
            sleep(Duration::from_millis(100)).await;
        }
    }
    
    async fn update_performance_metrics(&mut self) {
        let queue_depth = {
            let queue = self.request_queue.lock().unwrap();
            queue.len()
        };
        
        self.performance_monitor.record_queue_depth(queue_depth);
        
        let avg_synthesis_time = self.performance_monitor.average_synthesis_time();
        let avg_queue_depth = self.performance_monitor.average_queue_depth();
        
        println!("Performance: avg_synthesis={:.2}ms, avg_queue={:.1}", 
                 avg_synthesis_time.as_millis(),
                 avg_queue_depth);
        
        // Log to system if performance degrades
        if avg_synthesis_time > Duration::from_millis(1000) {
            eprintln!("WARNING: High synthesis latency detected: {:.2}ms", 
                     avg_synthesis_time.as_millis());
        }
        
        if avg_queue_depth > 5.0 {
            eprintln!("WARNING: High queue depth detected: {:.1}", avg_queue_depth);
        }
    }
}

impl PerformanceMonitor {
    fn new() -> Self {
        Self {
            synthesis_times: VecDeque::new(),
            queue_depths: VecDeque::new(),
            last_update: Instant::now(),
        }
    }
    
    fn record_synthesis(&mut self, duration: Duration) {
        self.synthesis_times.push_back(duration);
        if self.synthesis_times.len() > 100 {
            self.synthesis_times.pop_front();
        }
    }
    
    fn record_queue_depth(&mut self, depth: usize) {
        self.queue_depths.push_back(depth);
        if self.queue_depths.len() > 100 {
            self.queue_depths.pop_front();
        }
    }
    
    fn average_synthesis_time(&self) -> Duration {
        if self.synthesis_times.is_empty() {
            return Duration::from_millis(0);
        }
        
        let total: Duration = self.synthesis_times.iter().sum();
        total / self.synthesis_times.len() as u32
    }
    
    fn average_queue_depth(&self) -> f64 {
        if self.queue_depths.is_empty() {
            return 0.0;
        }
        
        let sum: usize = self.queue_depths.iter().sum();
        sum as f64 / self.queue_depths.len() as f64
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("VoiRS Edge Controller starting...");
    
    let mut controller = EdgeVoiRSController::new().await?;
    controller.start().await?;
    
    Ok(())
}

// Example systemd service file
/*
[Unit]
Description=VoiRS Edge TTS Controller
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/voirs-edge
ExecStart=/home/pi/voirs-edge/target/release/voirs-edge-controller
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
*/
```

These real-time application integration examples demonstrate how VoiRS FFI can be integrated into time-critical systems including live streaming, telecommunications, assistive technology, and edge computing scenarios, providing high-quality voice synthesis with optimized performance for real-time requirements.
#!/usr/bin/env node

/**
 * Advanced VoiRS Node.js Demo
 * 
 * This comprehensive example demonstrates advanced usage of the VoiRS Node.js bindings
 * including real-world scenarios like web server integration, audio processing pipelines,
 * performance optimization, and production-ready error handling.
 */

const express = require('express');
const multer = require('multer');
const fs = require('fs');
const path = require('path');
const { performance } = require('perf_hooks');
const { EventEmitter } = require('events');

// Import VoiRS bindings
const { VoirsPipeline, synthesizeStreaming, utils, ASRModel, AudioAnalyzer } = require('../index.js');

class VoiRSService extends EventEmitter {
    constructor(options = {}) {
        super();
        this.options = {
            maxConcurrentSynthesis: options.maxConcurrentSynthesis || 5,
            defaultQuality: options.defaultQuality || 'medium',
            cacheEnabled: options.cacheEnabled !== false,
            performanceMonitoring: options.performanceMonitoring !== false,
            ...options
        };
        
        this.pipeline = null;
        this.asrModel = null;
        this.audioAnalyzer = null;
        this.activeSynthesis = new Set();
        this.cache = new Map();
        this.stats = {
            totalRequests: 0,
            totalSynthesisTime: 0,
            cacheHits: 0,
            errors: 0
        };
        
        this.initializeService();
    }
    
    async initializeService() {
        try {
            console.log('🚀 Initializing VoiRS Service...');
            
            // Initialize synthesis pipeline
            this.pipeline = new VoirsPipeline({
                useGpu: this.options.useGpu || false,
                numThreads: this.options.numThreads || 4,
                cacheDir: this.options.cacheDir || './cache',
                device: this.options.device || 'cpu'
            });
            
            // Initialize ASR model if recognition is enabled
            if (this.options.enableRecognition) {
                console.log('🎤 Loading ASR model...');
                this.asrModel = ASRModel.whisper(this.options.asrModelSize || 'base');
                console.log('✅ ASR model loaded');
            }
            
            // Initialize audio analyzer
            this.audioAnalyzer = new AudioAnalyzer();
            
            // Load available voices
            const voices = await this.pipeline.listVoices();
            console.log(`🎭 Found ${voices.length} available voices`);
            
            // Set default voice
            if (voices.length > 0) {
                await this.pipeline.setVoice(voices[0].id);
                console.log(`🎵 Default voice set to: ${voices[0].name}`);
            }
            
            // Start performance monitoring
            if (this.options.performanceMonitoring) {
                this.startPerformanceMonitoring();
            }
            
            console.log('✅ VoiRS Service initialized successfully');
            this.emit('ready');
            
        } catch (error) {
            console.error('❌ Failed to initialize VoiRS Service:', error);
            this.emit('error', error);
        }
    }
    
    startPerformanceMonitoring() {
        setInterval(async () => {
            const perfInfo = await this.pipeline.getPerformanceInfo();
            const memoryUsage = process.memoryUsage();
            
            this.emit('performance', {
                timestamp: new Date().toISOString(),
                activeSynthesis: this.activeSynthesis.size,
                memoryUsage: {
                    rss: memoryUsage.rss / 1024 / 1024,
                    heapUsed: memoryUsage.heapUsed / 1024 / 1024,
                    heapTotal: memoryUsage.heapTotal / 1024 / 1024
                },
                voirsMemory: perfInfo.memoryUsageMb,
                cpuCores: perfInfo.cpuCores,
                threadCount: perfInfo.threadCount,
                stats: { ...this.stats }
            });
        }, 10000); // Every 10 seconds
    }
    
    generateCacheKey(text, options = {}) {
        const key = JSON.stringify({ text, options });
        return require('crypto').createHash('md5').update(key).digest('hex');
    }
    
    async synthesizeText(text, options = {}) {
        const startTime = performance.now();
        const requestId = Math.random().toString(36).substring(7);
        
        try {
            this.stats.totalRequests++;
            this.activeSynthesis.add(requestId);
            
            // Check cache first
            const cacheKey = this.generateCacheKey(text, options);
            if (this.options.cacheEnabled && this.cache.has(cacheKey)) {
                this.stats.cacheHits++;
                return this.cache.get(cacheKey);
            }
            
            // Validate options
            const validatedOptions = this.validateAndNormalizeOptions(options);
            
            // Perform synthesis with metrics
            const result = await this.pipeline.synthesizeWithMetrics(text, validatedOptions);
            
            // Cache result
            if (this.options.cacheEnabled) {
                this.cache.set(cacheKey, result);
                
                // Implement cache size limit
                if (this.cache.size > 1000) {
                    const firstKey = this.cache.keys().next().value;
                    this.cache.delete(firstKey);
                }
            }
            
            const totalTime = performance.now() - startTime;
            this.stats.totalSynthesisTime += totalTime;
            
            this.emit('synthesis', {
                requestId,
                text: text.substring(0, 100) + (text.length > 100 ? '...' : ''),
                duration: result.audio.duration,
                processingTime: totalTime,
                realTimeFactor: result.metrics.realTimeFactor,
                cached: false
            });
            
            return result;
            
        } catch (error) {
            this.stats.errors++;
            this.emit('error', { requestId, error: error.message });
            throw error;
        } finally {
            this.activeSynthesis.delete(requestId);
        }
    }
    
    validateAndNormalizeOptions(options) {
        const normalized = {
            speakingRate: Math.max(0.1, Math.min(3.0, options.speakingRate || 1.0)),
            pitchShift: Math.max(-12, Math.min(12, options.pitchShift || 0)),
            volumeGain: Math.max(-20, Math.min(20, options.volumeGain || 0)),
            quality: ['low', 'medium', 'high', 'ultra'].includes(options.quality) 
                ? options.quality 
                : this.options.defaultQuality,
            outputFormat: ['wav', 'flac', 'mp3', 'opus', 'ogg'].includes(options.outputFormat)
                ? options.outputFormat
                : 'wav',
            sampleRate: [8000, 16000, 22050, 44100, 48000].includes(options.sampleRate)
                ? options.sampleRate
                : 22050,
            enableEnhancement: options.enableEnhancement !== false
        };
        
        return normalized;
    }
    
    async synthesizeStreaming(text, options = {}) {
        const requestId = Math.random().toString(36).substring(7);
        const chunks = [];
        
        return new Promise((resolve, reject) => {
            synthesizeStreaming(
                this.pipeline,
                text,
                (chunk) => {
                    chunks.push(chunk);
                    this.emit('streamingChunk', { requestId, chunk });
                },
                (progress) => {
                    this.emit('streamingProgress', { requestId, progress });
                },
                this.validateAndNormalizeOptions(options)
            ).then(() => {
                resolve(Buffer.concat(chunks));
            }).catch(reject);
        });
    }
    
    async batchSynthesize(texts, options = {}) {
        const batchId = Math.random().toString(36).substring(7);
        
        const results = await this.pipeline.batchSynthesize(
            texts,
            this.validateAndNormalizeOptions(options),
            (current, total, progress) => {
                this.emit('batchProgress', { batchId, current, total, progress });
            }
        );
        
        this.emit('batchComplete', { batchId, results: results.length });
        return results;
    }
    
    async analyzeAudio(audioBuffer) {
        return await this.pipeline.analyzeAudio(audioBuffer);
    }
    
    async recognizeSpeech(audioBuffer) {
        if (!this.asrModel) {
            throw new Error('ASR model not initialized');
        }
        
        return await this.asrModel.recognize(audioBuffer);
    }
    
    async getVoices() {
        return await this.pipeline.listVoices();
    }
    
    async setVoice(voiceId) {
        await this.pipeline.setVoice(voiceId);
        this.emit('voiceChanged', { voiceId });
    }
    
    getStats() {
        return {
            ...this.stats,
            activeSynthesis: this.activeSynthesis.size,
            cacheSize: this.cache.size,
            uptime: process.uptime()
        };
    }
}

// Express.js Web Server Integration
class VoiRSWebServer {
    constructor(voirsService, port = 3000) {
        this.voirsService = voirsService;
        this.port = port;
        this.app = express();
        this.setupMiddleware();
        this.setupRoutes();
        this.setupEventListeners();
    }
    
    setupMiddleware() {
        this.app.use(express.json({ limit: '10mb' }));
        this.app.use(express.urlencoded({ extended: true }));
        
        // CORS
        this.app.use((req, res, next) => {
            res.header('Access-Control-Allow-Origin', '*');
            res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
            res.header('Access-Control-Allow-Headers', 'Content-Type, Authorization');
            if (req.method === 'OPTIONS') {
                res.sendStatus(200);
            } else {
                next();
            }
        });
        
        // Request logging
        this.app.use((req, res, next) => {
            console.log(`${new Date().toISOString()} ${req.method} ${req.path}`);
            next();
        });
    }
    
    setupRoutes() {
        // Health check
        this.app.get('/health', (req, res) => {
            res.json({
                status: 'healthy',
                timestamp: new Date().toISOString(),
                uptime: process.uptime(),
                stats: this.voirsService.getStats()
            });
        });
        
        // Synthesize text to speech
        this.app.post('/synthesize', async (req, res) => {
            try {
                const { text, options = {} } = req.body;
                
                if (!text) {
                    return res.status(400).json({ error: 'Text is required' });
                }
                
                const result = await this.voirsService.synthesizeText(text, options);
                
                res.json({
                    success: true,
                    audio: {
                        duration: result.audio.duration,
                        sampleRate: result.audio.sampleRate,
                        channels: result.audio.channels,
                        size: result.audio.samples.length
                    },
                    metrics: result.metrics,
                    audioData: result.audio.samples.toString('base64')
                });
                
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });
        
        // Synthesize and return audio file
        this.app.post('/synthesize/audio', async (req, res) => {
            try {
                const { text, options = {} } = req.body;
                const format = options.outputFormat || 'wav';
                
                if (!text) {
                    return res.status(400).json({ error: 'Text is required' });
                }
                
                const result = await this.voirsService.synthesizeText(text, options);
                
                res.setHeader('Content-Type', `audio/${format}`);
                res.setHeader('Content-Disposition', `attachment; filename="synthesis.${format}"`);
                
                if (format === 'wav') {
                    const wavBuffer = utils.toWav(result.audio);
                    res.send(wavBuffer);
                } else {
                    res.send(result.audio.samples);
                }
                
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });
        
        // Streaming synthesis
        this.app.post('/synthesize/stream', async (req, res) => {
            try {
                const { text, options = {} } = req.body;
                
                if (!text) {
                    return res.status(400).json({ error: 'Text is required' });
                }
                
                res.setHeader('Content-Type', 'application/octet-stream');
                res.setHeader('Transfer-Encoding', 'chunked');
                
                const requestId = Math.random().toString(36).substring(7);
                
                const cleanup = () => {
                    this.voirsService.removeAllListeners(`streamingChunk-${requestId}`);
                    this.voirsService.removeAllListeners(`streamingProgress-${requestId}`);
                };
                
                this.voirsService.on(`streamingChunk-${requestId}`, (data) => {
                    res.write(data.chunk);
                });
                
                this.voirsService.on(`streamingProgress-${requestId}`, (data) => {
                    // Could send progress via Server-Sent Events
                    console.log(`Streaming progress: ${(data.progress * 100).toFixed(1)}%`);
                });
                
                await this.voirsService.synthesizeStreaming(text, options);
                
                res.end();
                cleanup();
                
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });
        
        // Batch synthesis
        this.app.post('/synthesize/batch', async (req, res) => {
            try {
                const { texts, options = {} } = req.body;
                
                if (!texts || !Array.isArray(texts)) {
                    return res.status(400).json({ error: 'Texts array is required' });
                }
                
                const results = await this.voirsService.batchSynthesize(texts, options);
                
                res.json({
                    success: true,
                    results: results.map(r => ({
                        audio: {
                            duration: r.audio.duration,
                            sampleRate: r.audio.sampleRate,
                            channels: r.audio.channels,
                            size: r.audio.samples.length
                        },
                        metrics: r.metrics,
                        audioData: r.audio.samples.toString('base64')
                    }))
                });
                
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });
        
        // Voice management
        this.app.get('/voices', async (req, res) => {
            try {
                const voices = await this.voirsService.getVoices();
                res.json({ voices });
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });
        
        this.app.post('/voices/:voiceId', async (req, res) => {
            try {
                const { voiceId } = req.params;
                await this.voirsService.setVoice(voiceId);
                res.json({ success: true, voiceId });
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });
        
        // Audio analysis
        this.app.post('/analyze', async (req, res) => {
            try {
                const { audioData } = req.body;
                
                if (!audioData) {
                    return res.status(400).json({ error: 'Audio data is required' });
                }
                
                const audioBuffer = {
                    samples: Buffer.from(audioData, 'base64'),
                    sampleRate: req.body.sampleRate || 22050,
                    channels: req.body.channels || 1,
                    duration: req.body.duration || 0
                };
                
                const analysis = await this.voirsService.analyzeAudio(audioBuffer);
                res.json({ analysis });
                
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });
        
        // Speech recognition
        this.app.post('/recognize', async (req, res) => {
            try {
                const { audioData } = req.body;
                
                if (!audioData) {
                    return res.status(400).json({ error: 'Audio data is required' });
                }
                
                const audioBuffer = {
                    samples: Buffer.from(audioData, 'base64'),
                    sampleRate: req.body.sampleRate || 22050,
                    channels: req.body.channels || 1,
                    duration: req.body.duration || 0
                };
                
                const recognition = await this.voirsService.recognizeSpeech(audioBuffer);
                res.json({ recognition });
                
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });
        
        // Statistics
        this.app.get('/stats', (req, res) => {
            res.json(this.voirsService.getStats());
        });
        
        // Utility endpoints
        this.app.get('/formats', (req, res) => {
            res.json({
                formats: utils.getSupportedFormats(),
                qualities: utils.getSupportedQualities()
            });
        });
    }
    
    setupEventListeners() {
        this.voirsService.on('synthesis', (data) => {
            console.log(`✅ Synthesis completed: ${data.text} (${data.duration}s, RTF: ${data.realTimeFactor.toFixed(3)})`);
        });
        
        this.voirsService.on('error', (data) => {
            console.error(`❌ Synthesis error: ${data.error}`);
        });
        
        this.voirsService.on('performance', (data) => {
            if (data.activeSynthesis > 0) {
                console.log(`📊 Performance: ${data.activeSynthesis} active, ${data.memoryUsage.heapUsed.toFixed(1)}MB heap`);
            }
        });
    }
    
    start() {
        this.app.listen(this.port, () => {
            console.log(`🌐 VoiRS Web Server running on port ${this.port}`);
            console.log(`📖 API Documentation available at http://localhost:${this.port}/`);
        });
    }
}

// CLI Interface
class VoiRSCLI {
    constructor() {
        this.voirsService = null;
    }
    
    async initialize() {
        this.voirsService = new VoiRSService({
            enableRecognition: false,
            performanceMonitoring: true
        });
        
        return new Promise((resolve) => {
            this.voirsService.on('ready', resolve);
        });
    }
    
    async runInteractiveMode() {
        const readline = require('readline');
        const rl = readline.createInterface({
            input: process.stdin,
            output: process.stdout
        });
        
        console.log('\\n🎤 VoiRS Interactive Mode');
        console.log('Commands: synthesize, voices, set-voice, analyze, stats, help, exit');
        
        const askQuestion = (question) => {
            return new Promise((resolve) => {
                rl.question(question, resolve);
            });
        };
        
        while (true) {
            const command = await askQuestion('\\nvoirs> ');
            
            try {
                const [cmd, ...args] = command.trim().split(' ');
                
                switch (cmd) {
                    case 'synthesize':
                        const text = args.join(' ') || await askQuestion('Enter text to synthesize: ');
                        const result = await this.voirsService.synthesizeText(text);
                        console.log(`✅ Synthesized ${result.audio.duration}s of audio`);
                        console.log(`📊 Metrics: RTF=${result.metrics.realTimeFactor.toFixed(3)}, Memory=${result.metrics.memoryUsageMb.toFixed(1)}MB`);
                        break;
                        
                    case 'voices':
                        const voices = await this.voirsService.getVoices();
                        console.log('Available voices:');
                        voices.forEach((voice, i) => {
                            console.log(`  ${i + 1}. ${voice.name} (${voice.id}) - ${voice.language}`);
                        });
                        break;
                        
                    case 'set-voice':
                        const voiceId = args[0] || await askQuestion('Enter voice ID: ');
                        await this.voirsService.setVoice(voiceId);
                        console.log(`✅ Voice set to: ${voiceId}`);
                        break;
                        
                    case 'stats':
                        const stats = this.voirsService.getStats();
                        console.log('Service Statistics:');
                        console.log(`  Total requests: ${stats.totalRequests}`);
                        console.log(`  Cache hits: ${stats.cacheHits}`);
                        console.log(`  Errors: ${stats.errors}`);
                        console.log(`  Average synthesis time: ${(stats.totalSynthesisTime / stats.totalRequests).toFixed(2)}ms`);
                        break;
                        
                    case 'help':
                        console.log('Available commands:');
                        console.log('  synthesize [text] - Synthesize text to speech');
                        console.log('  voices - List available voices');
                        console.log('  set-voice [id] - Set active voice');
                        console.log('  stats - Show service statistics');
                        console.log('  help - Show this help');
                        console.log('  exit - Exit interactive mode');
                        break;
                        
                    case 'exit':
                        rl.close();
                        return;
                        
                    default:
                        console.log(`Unknown command: ${cmd}. Type 'help' for available commands.`);
                }
                
            } catch (error) {
                console.error('❌ Error:', error.message);
            }
        }
    }
}

// Main execution
async function main() {
    const args = process.argv.slice(2);
    const mode = args[0] || 'interactive';
    
    switch (mode) {
        case 'server':
            console.log('🌐 Starting VoiRS Web Server...');
            const voirsService = new VoiRSService({
                enableRecognition: args.includes('--with-recognition'),
                performanceMonitoring: true,
                useGpu: args.includes('--gpu')
            });
            
            voirsService.on('ready', () => {
                const server = new VoiRSWebServer(voirsService, 3000);
                server.start();
            });
            break;
            
        case 'interactive':
            console.log('🎤 Starting VoiRS Interactive CLI...');
            const cli = new VoiRSCLI();
            await cli.initialize();
            await cli.runInteractiveMode();
            break;
            
        case 'demo':
            console.log('🎬 Running VoiRS Demo...');
            await runDemo();
            break;
            
        default:
            console.log('Usage: node advanced_demo.js [server|interactive|demo]');
            console.log('  server - Start web server');
            console.log('  interactive - Start interactive CLI');
            console.log('  demo - Run feature demonstration');
    }
}

async function runDemo() {
    const service = new VoiRSService({
        performanceMonitoring: true
    });
    
    await new Promise((resolve) => {
        service.on('ready', resolve);
    });
    
    console.log('\\n🎬 VoiRS Advanced Demo');
    console.log('='.repeat(50));
    
    // Demo 1: Basic synthesis
    console.log('\\n1. Basic Synthesis Demo');
    const basicResult = await service.synthesizeText(
        "Welcome to the VoiRS advanced demonstration!"
    );
    console.log(`✅ Generated ${basicResult.audio.duration}s of audio`);
    console.log(`📊 Real-time factor: ${basicResult.metrics.realTimeFactor.toFixed(3)}`);
    
    // Demo 2: Batch synthesis
    console.log('\\n2. Batch Synthesis Demo');
    const batchTexts = [
        "First sentence in the batch.",
        "Second sentence with different content.",
        "Third sentence for batch processing demo."
    ];
    
    const batchResults = await service.batchSynthesize(batchTexts);
    console.log(`✅ Batch synthesis completed: ${batchResults.length} items`);
    
    // Demo 3: Voice switching
    console.log('\\n3. Voice Management Demo');
    const voices = await service.getVoices();
    console.log(`📋 Available voices: ${voices.length}`);
    
    if (voices.length > 1) {
        await service.setVoice(voices[1].id);
        const voiceResult = await service.synthesizeText("Testing different voice.");
        console.log(`✅ Voice switched and tested`);
    }
    
    // Demo 4: Performance stats
    console.log('\\n4. Performance Statistics');
    const stats = service.getStats();
    console.log(`📊 Statistics:`);
    console.log(`   Total requests: ${stats.totalRequests}`);
    console.log(`   Cache hits: ${stats.cacheHits}`);
    console.log(`   Active synthesis: ${stats.activeSynthesis}`);
    console.log(`   Uptime: ${Math.floor(stats.uptime)}s`);
    
    console.log('\\n🎉 Demo completed successfully!');
}

// Handle graceful shutdown
process.on('SIGINT', () => {
    console.log('\\n🛑 Shutting down gracefully...');
    process.exit(0);
});

process.on('SIGTERM', () => {
    console.log('\\n🛑 Received SIGTERM, shutting down...');
    process.exit(0);
});

if (require.main === module) {
    main().catch(console.error);
}

module.exports = { VoiRSService, VoiRSWebServer, VoiRSCLI };
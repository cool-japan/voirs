#!/usr/bin/env node

/**
 * VoiRS Recognizer - Node.js Example
 * 
 * This example demonstrates how to use the VoiRS speech recognition
 * WASM module in a Node.js application.
 */

const fs = require('fs').promises;
const path = require('path');

// Import the WASM module (adjust path as needed)
const wasmModule = require('../pkg/nodejs/voirs_recognizer.js');

class VoirsRecognizerDemo {
    constructor() {
        this.recognizer = null;
        this.initialized = false;
    }

    async initialize() {
        try {
            console.log('Initializing VoiRS Recognizer...');
            
            // Initialize the WASM module
            await wasmModule.default();
            
            // Initialize logger
            wasmModule.init_wasm_logger();
            
            // Check capabilities
            const compatibility = wasmModule.check_browser_compatibility();
            console.log('Environment compatibility:', JSON.stringify(compatibility, null, 2));
            
            // Create recognizer instance
            this.recognizer = new wasmModule.WasmVoirsRecognizer();
            
            // Configure recognizer
            const config = {
                model_name: 'whisper-base',
                language: 'en',
                sample_rate: 16000,
                enable_vad: true,
                confidence_threshold: 0.5,
                beam_size: 5,
                temperature: 0.0
            };
            
            await this.recognizer.initialize(config);
            
            console.log('Recognizer initialized successfully!');
            
            // Show capabilities
            const capabilities = this.recognizer.get_capabilities();
            console.log('Recognizer capabilities:', JSON.stringify(capabilities, null, 2));
            
            // Show supported models and languages
            const models = await this.recognizer.get_supported_models();
            const languages = await this.recognizer.get_supported_languages();
            
            console.log('Supported models:', models);
            console.log('Supported languages:', languages);
            
            this.initialized = true;
            
        } catch (error) {
            console.error('Failed to initialize recognizer:', error);
            throw error;
        }
    }

    async recognizeAudioFile(filePath) {
        if (!this.initialized) {
            throw new Error('Recognizer not initialized');
        }

        try {
            console.log(`\nProcessing audio file: ${filePath}`);
            
            // Read audio file
            const audioData = await fs.readFile(filePath);
            const audioBytes = new Uint8Array(audioData);
            
            console.log(`Audio file size: ${audioBytes.length} bytes`);
            
            // Perform recognition
            const startTime = Date.now();
            const result = await this.recognizer.recognize_audio(audioBytes);
            const endTime = Date.now();
            
            const processingTime = endTime - startTime;
            
            // Display results
            console.log('\n=== Recognition Results ===');
            console.log(`Text: "${result.text}"`);
            console.log(`Confidence: ${(result.confidence * 100).toFixed(1)}%`);
            console.log(`Language: ${result.language || 'Unknown'}`);
            console.log(`Processing Time: ${processingTime}ms`);
            console.log(`Segments: ${result.segments.length}`);
            
            if (result.segments.length > 0) {
                console.log('\n=== Segments ===');
                result.segments.forEach((segment, index) => {
                    console.log(`${index + 1}. [${segment.start_time.toFixed(2)}s - ${segment.end_time.toFixed(2)}s] "${segment.text}" (${(segment.confidence * 100).toFixed(1)}%)`);
                });
            }
            
            console.log('\n=== Metadata ===');
            console.log(JSON.stringify(result.metadata, null, 2));
            
            return result;
            
        } catch (error) {
            console.error('Recognition failed:', error);
            throw error;
        }
    }

    async demonstrateStreaming() {
        if (!this.initialized) {
            throw new Error('Recognizer not initialized');
        }

        try {
            console.log('\n=== Streaming Recognition Demo ===');
            
            // Configure streaming
            const streamingConfig = {
                chunk_duration: 1.0, // 1 second chunks
                overlap_duration: 0.1, // 100ms overlap
                vad_threshold: 0.5,
                silence_duration: 2.0,
                max_chunk_size: 16000 // 1 second at 16kHz
            };
            
            await this.recognizer.start_streaming(streamingConfig);
            console.log('Streaming started');
            
            // Simulate streaming chunks (in a real app, this would come from a microphone or live audio source)
            const chunkSize = 8000; // 0.5 seconds at 16kHz
            const totalChunks = 5;
            
            for (let i = 0; i < totalChunks; i++) {
                // Generate dummy audio data (in practice, this would be real audio)
                const audioChunk = new Uint8Array(chunkSize * 2); // 16-bit samples
                
                // Fill with some basic sine wave data for demonstration
                for (let j = 0; j < chunkSize; j++) {
                    const sample = Math.sin(2 * Math.PI * 440 * j / 16000) * 0.1; // 440Hz tone
                    const intSample = Math.round(sample * 32767);
                    audioChunk[j * 2] = intSample & 0xFF;
                    audioChunk[j * 2 + 1] = (intSample >> 8) & 0xFF;
                }
                
                console.log(`Processing chunk ${i + 1}/${totalChunks}...`);
                
                try {
                    const chunkResult = await this.recognizer.stream_audio(audioChunk);
                    
                    if (chunkResult.text && chunkResult.text.trim().length > 0) {
                        console.log(`Chunk ${i + 1} result: "${chunkResult.text}" (${(chunkResult.confidence * 100).toFixed(1)}%)`);
                    }
                } catch (chunkError) {
                    console.warn(`Chunk ${i + 1} processing failed:`, chunkError.message);
                }
                
                // Simulate real-time processing delay
                await new Promise(resolve => setTimeout(resolve, 100));
            }
            
            this.recognizer.stop_streaming();
            console.log('Streaming stopped');
            
        } catch (error) {
            console.error('Streaming demo failed:', error);
            throw error;
        }
    }

    async demonstrateModelSwitching() {
        if (!this.initialized) {
            throw new Error('Recognizer not initialized');
        }

        try {
            console.log('\n=== Model Switching Demo ===');
            
            const models = ['whisper-tiny', 'whisper-base', 'whisper-small'];
            
            for (const model of models) {
                console.log(`\nSwitching to model: ${model}`);
                
                try {
                    await this.recognizer.switch_model(model);
                    console.log(`Successfully switched to ${model}`);
                    
                    const config = this.recognizer.get_current_config();
                    console.log('Current config:', JSON.stringify(config, null, 2));
                    
                } catch (error) {
                    console.warn(`Failed to switch to ${model}:`, error.message);
                }
            }
            
        } catch (error) {
            console.error('Model switching demo failed:', error);
            throw error;
        }
    }

    getMemoryUsage() {
        const memoryUsage = wasmModule.get_wasm_memory_usage();
        console.log('\n=== Memory Usage ===');
        console.log(JSON.stringify(memoryUsage, null, 2));
        
        if (process.memoryUsage) {
            const nodeMemory = process.memoryUsage();
            console.log('\n=== Node.js Memory Usage ===');
            console.log(`RSS: ${(nodeMemory.rss / 1024 / 1024).toFixed(2)} MB`);
            console.log(`Heap Used: ${(nodeMemory.heapUsed / 1024 / 1024).toFixed(2)} MB`);
            console.log(`Heap Total: ${(nodeMemory.heapTotal / 1024 / 1024).toFixed(2)} MB`);
            console.log(`External: ${(nodeMemory.external / 1024 / 1024).toFixed(2)} MB`);
        }
    }
}

async function main() {
    const demo = new VoirsRecognizerDemo();
    
    try {
        // Initialize
        await demo.initialize();
        
        // Get command line arguments
        const args = process.argv.slice(2);
        
        if (args.length > 0) {
            // Process audio files provided as arguments
            for (const filePath of args) {
                try {
                    const fullPath = path.resolve(filePath);
                    await demo.recognizeAudioFile(fullPath);
                } catch (error) {
                    console.error(`Failed to process ${filePath}:`, error.message);
                }
            }
        } else {
            console.log('\nNo audio files provided. Running demonstrations...');
            
            // Run demonstrations
            await demo.demonstrateStreaming();
            await demo.demonstrateModelSwitching();
        }
        
        // Show memory usage
        demo.getMemoryUsage();
        
        console.log('\n=== Demo completed successfully! ===');
        
    } catch (error) {
        console.error('Demo failed:', error);
        process.exit(1);
    }
}

// Usage information
if (require.main === module) {
    if (process.argv.includes('--help') || process.argv.includes('-h')) {
        console.log('VoiRS Recognizer - Node.js Example');
        console.log('');
        console.log('Usage:');
        console.log('  node nodejs-example.js [audio-files...]');
        console.log('');
        console.log('Examples:');
        console.log('  node nodejs-example.js                    # Run demonstrations');
        console.log('  node nodejs-example.js audio.wav         # Process single file');
        console.log('  node nodejs-example.js *.wav             # Process multiple files');
        console.log('');
        console.log('Supported formats: WAV, MP3, FLAC, OGG, M4A');
        process.exit(0);
    }
    
    main().catch(error => {
        console.error('Unhandled error:', error);
        process.exit(1);
    });
}

module.exports = VoirsRecognizerDemo;

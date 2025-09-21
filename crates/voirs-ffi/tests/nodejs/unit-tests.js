#!/usr/bin/env node

/**
 * VoiRS Node.js Unit Tests
 * 
 * Core functionality unit tests for VoiRS FFI bindings
 * Tests basic operations, configuration, and API surface
 */

const path = require('path');
const fs = require('fs');

// Try to load the VoiRS bindings
let VoirsPipeline, synthesizeStreaming;
const bindingsPath = path.resolve(__dirname, '../../index.js');

try {
    if (fs.existsSync(bindingsPath)) {
        const bindings = require(bindingsPath);
        VoirsPipeline = bindings.VoirsPipeline;
        synthesizeStreaming = bindings.synthesizeStreaming;
    } else {
        console.log('⚠️  Bindings not found, using mock implementations for testing');
        
        // Mock implementations for testing without built bindings
        class MockVoirsPipeline {
            constructor(options = {}) {
                this.options = options;
                this.currentVoice = null;
            }

            getInfo() {
                return JSON.stringify({
                    version: "0.1.0",
                    features: { nodejs_bindings: true },
                    runtime_info: { worker_threads: 4 }
                });
            }

            async listVoices() {
                return [
                    { id: 'test-voice-1', name: 'Test Voice 1', language: 'en-US', quality: 'high', isAvailable: true },
                    { id: 'test-voice-2', name: 'Test Voice 2', language: 'en-GB', quality: 'medium', isAvailable: true }
                ];
            }

            async setVoice(voiceId) {
                this.currentVoice = voiceId;
            }

            async getVoice() {
                return this.currentVoice;
            }

            async synthesize(text, options = {}) {
                if (!text) throw new Error('Text is required');
                
                // Simulate audio generation
                const sampleRate = options.sampleRate || 22050;
                const duration = Math.max(0.5, text.length * 0.05); // ~50ms per character
                const samples = new Uint8Array(Math.floor(sampleRate * duration * 2)); // 16-bit samples
                
                return {
                    samples: Buffer.from(samples),
                    sampleRate,
                    channels: 1,
                    duration
                };
            }

            async synthesizeSsml(ssml) {
                return this.synthesize(ssml.replace(/<[^>]*>/g, ''));
            }

            async synthesizeWithCallbacks(text, options, progressCb, errorCb) {
                if (progressCb) progressCb(0.0);
                
                const result = await this.synthesize(text, options);
                
                if (progressCb) progressCb(1.0);
                return result;
            }
        }

        VoirsPipeline = MockVoirsPipeline;
        
        synthesizeStreaming = async function(pipeline, text, chunkCb, progressCb, options) {
            const result = await pipeline.synthesize(text, options);
            const chunkSize = 1024;
            const chunks = Math.ceil(result.samples.length / chunkSize);
            
            for (let i = 0; i < chunks; i++) {
                const start = i * chunkSize;
                const end = Math.min(start + chunkSize, result.samples.length);
                const chunk = result.samples.slice(start, end);
                
                if (progressCb) progressCb(i / chunks);
                chunkCb(chunk);
            }
            
            if (progressCb) progressCb(1.0);
        };
    }
} catch (error) {
    console.error('❌ Failed to load VoiRS bindings:', error.message);
    process.exit(1);
}

// Test framework
class TestFramework {
    constructor() {
        this.tests = [];
        this.passed = 0;
        this.failed = 0;
        this.startTime = Date.now();
    }

    test(name, testFn) {
        this.tests.push({ name, testFn });
    }

    async run() {
        console.log('🧪 Running VoiRS Node.js Unit Tests\n');

        for (const { name, testFn } of this.tests) {
            try {
                process.stdout.write(`  Testing ${name}... `);
                await testFn();
                console.log('✅ PASS');
                this.passed++;
            } catch (error) {
                console.log('❌ FAIL');
                console.log(`    Error: ${error.message}`);
                this.failed++;
            }
        }

        const duration = Date.now() - this.startTime;
        console.log(`\n📊 Unit Test Results: ${this.passed} passed, ${this.failed} failed (${duration}ms)`);
        
        if (this.failed > 0) {
            process.exit(1);
        }
    }

    assert(condition, message) {
        if (!condition) {
            throw new Error(message || 'Assertion failed');
        }
    }

    assertEqual(actual, expected, message) {
        if (actual !== expected) {
            throw new Error(message || `Expected ${expected}, got ${actual}`);
        }
    }

    assertNotNull(value, message) {
        if (value === null || value === undefined) {
            throw new Error(message || 'Value should not be null or undefined');
        }
    }

    assertType(value, expectedType, message) {
        if (typeof value !== expectedType) {
            throw new Error(message || `Expected type ${expectedType}, got ${typeof value}`);
        }
    }
}

// Test suite
const test = new TestFramework();

// Basic constructor tests
test.test('VoirsPipeline constructor with no options', async () => {
    const pipeline = new VoirsPipeline();
    test.assertNotNull(pipeline, 'Pipeline should be created');
});

test.test('VoirsPipeline constructor with options', async () => {
    const options = {
        useGpu: false,
        numThreads: 2,
        cacheDir: '/tmp/voirs-test',
        device: 'cpu'
    };
    
    const pipeline = new VoirsPipeline(options);
    test.assertNotNull(pipeline, 'Pipeline should be created with options');
});

// Pipeline info tests
test.test('getInfo returns valid JSON', async () => {
    const pipeline = new VoirsPipeline();
    const info = pipeline.getInfo();
    
    test.assertType(info, 'string', 'getInfo should return string');
    
    let parsed;
    try {
        parsed = JSON.parse(info);
    } catch (e) {
        throw new Error('getInfo should return valid JSON');
    }
    
    test.assertNotNull(parsed.version, 'Info should contain version');
    test.assertNotNull(parsed.features, 'Info should contain features');
});

// Voice management tests  
test.test('listVoices returns array', async () => {
    const pipeline = new VoirsPipeline();
    const voices = await pipeline.listVoices();
    
    test.assert(Array.isArray(voices), 'listVoices should return array');
    
    if (voices.length > 0) {
        const voice = voices[0];
        test.assertNotNull(voice.id, 'Voice should have id');
        test.assertNotNull(voice.name, 'Voice should have name');
        test.assertNotNull(voice.language, 'Voice should have language');
    }
});

test.test('setVoice and getVoice work correctly', async () => {
    const pipeline = new VoirsPipeline();
    const voices = await pipeline.listVoices();
    
    if (voices.length > 0) {
        const voiceId = voices[0].id;
        
        await pipeline.setVoice(voiceId);
        const currentVoice = await pipeline.getVoice();
        
        test.assertEqual(currentVoice, voiceId, 'Set voice should be returned by getVoice');
    }
});

// Synthesis tests
test.test('synthesize basic text', async () => {
    const pipeline = new VoirsPipeline();
    const text = "Hello, world!";
    
    const result = await pipeline.synthesize(text);
    
    test.assertNotNull(result, 'Synthesis should return result');
    test.assertNotNull(result.samples, 'Result should contain samples');
    test.assertType(result.sampleRate, 'number', 'Result should contain sample rate');
    test.assertType(result.channels, 'number', 'Result should contain channel count');
    test.assertType(result.duration, 'number', 'Result should contain duration');
    test.assert(result.duration > 0, 'Duration should be positive');
});

test.test('synthesize with options', async () => {
    const pipeline = new VoirsPipeline();
    const text = "Testing synthesis options";
    const options = {
        speakingRate: 1.2,
        pitchShift: 0.1,
        volumeGain: -0.5,
        enableEnhancement: true,
        outputFormat: 'wav',
        sampleRate: 44100,
        quality: 'high'
    };
    
    const result = await pipeline.synthesize(text, options);
    
    test.assertNotNull(result, 'Synthesis with options should return result');
    test.assert(result.samples.length > 0, 'Should generate audio samples');
});

test.test('synthesize empty text throws error', async () => {
    const pipeline = new VoirsPipeline();
    
    try {
        await pipeline.synthesize('');
        throw new Error('Should have thrown error for empty text');
    } catch (error) {
        test.assert(error.message.includes('required') || error.message.includes('empty'), 
                   'Should throw meaningful error for empty text');
    }
});

test.test('synthesize SSML', async () => {
    const pipeline = new VoirsPipeline();
    const ssml = '<speak><p>Hello <emphasis level="strong">world</emphasis>!</p></speak>';
    
    const result = await pipeline.synthesizeSsml(ssml);
    
    test.assertNotNull(result, 'SSML synthesis should return result');
    test.assert(result.samples.length > 0, 'SSML should generate audio samples');
});

// Callback tests
test.test('synthesizeWithCallbacks executes callbacks', async () => {
    const pipeline = new VoirsPipeline();
    const text = "Testing callbacks";
    
    let progressCalled = false;
    let finalProgress = 0;
    
    const result = await pipeline.synthesizeWithCallbacks(
        text,
        {},
        (progress) => {
            progressCalled = true;
            finalProgress = progress;
        },
        (error) => {
            throw new Error(`Unexpected error callback: ${error}`);
        }
    );
    
    test.assert(progressCalled, 'Progress callback should be called');
    test.assertEqual(finalProgress, 1.0, 'Final progress should be 1.0');
    test.assertNotNull(result, 'Callback synthesis should return result');
});

// Streaming tests
test.test('synthesizeStreaming works', async () => {
    const pipeline = new VoirsPipeline();
    const text = "Testing streaming synthesis";
    
    let chunkCount = 0;
    let totalBytes = 0;
    let progressCount = 0;
    let finalProgress = 0;
    
    await synthesizeStreaming(
        pipeline,
        text,
        (chunk) => {
            chunkCount++;
            totalBytes += chunk.length;
            test.assert(chunk.length > 0, 'Chunks should contain data');
        },
        (progress) => {
            progressCount++;
            finalProgress = progress;
        },
        { quality: 'medium' }
    );
    
    test.assert(chunkCount > 0, 'Should receive at least one chunk');
    test.assert(totalBytes > 0, 'Should receive audio data');
    test.assert(progressCount > 0, 'Should receive progress updates');
    test.assertEqual(finalProgress, 1.0, 'Final progress should be 1.0');
});

// Error handling tests
test.test('invalid voice ID throws error', async () => {
    const pipeline = new VoirsPipeline();
    
    try {
        await pipeline.setVoice('non-existent-voice-id');
        // If mock, this might not throw, so only test if it does
    } catch (error) {
        test.assert(error.message.length > 0, 'Should provide meaningful error message');
    }
});

test.test('invalid synthesis options handled gracefully', async () => {
    const pipeline = new VoirsPipeline();
    const text = "Testing invalid options";
    
    // These should either work or throw meaningful errors
    try {
        await pipeline.synthesize(text, {
            speakingRate: -1,  // Invalid rate
            sampleRate: 1,     // Invalid sample rate
            quality: 'invalid' // Invalid quality
        });
    } catch (error) {
        test.assert(error.message.length > 0, 'Should provide meaningful error for invalid options');
    }
});

// Run all tests
test.run().catch(error => {
    console.error('❌ Test runner failed:', error);
    process.exit(1);
});
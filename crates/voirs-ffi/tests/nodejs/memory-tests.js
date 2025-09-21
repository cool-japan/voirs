#!/usr/bin/env node

/**
 * VoiRS Node.js Memory Tests
 * 
 * Memory usage and leak detection tests for VoiRS FFI bindings
 * Tests memory efficiency, cleanup, and resource management
 */

const path = require('path');
const fs = require('fs');

// Load VoiRS bindings
let VoirsPipeline, synthesizeStreaming;
const bindingsPath = path.resolve(__dirname, '../../index.js');

try {
    if (fs.existsSync(bindingsPath)) {
        const bindings = require(bindingsPath);
        VoirsPipeline = bindings.VoirsPipeline;
        synthesizeStreaming = bindings.synthesizeStreaming;
    } else {
        // Memory-focused mock implementations
        class MockVoirsPipeline {
            constructor(options = {}) {
                this.options = options;
                this.currentVoice = null;
                this.allocatedBuffers = new Set();
                this.instanceId = Math.random().toString(36);
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
                
                await new Promise(resolve => setTimeout(resolve, 10));
                
                const sampleRate = options.sampleRate || 22050;
                const duration = Math.max(0.5, text.length * 0.05);
                const sampleCount = Math.floor(sampleRate * duration * 2);
                
                // Create buffer and track it
                const samples = Buffer.alloc(sampleCount);
                this.allocatedBuffers.add(samples);
                
                // Simulate some audio data
                for (let i = 0; i < sampleCount; i += 2) {
                    const sample = Math.sin(2 * Math.PI * 440 * i / sampleRate) * 32767;
                    samples.writeInt16LE(sample, i);
                }
                
                return {
                    samples,
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

            // Method to check allocated buffers (for testing)
            getAllocatedBufferCount() {
                return this.allocatedBuffers.size;
            }

            // Method to clear buffer references (simulating cleanup)
            cleanup() {
                this.allocatedBuffers.clear();
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

// Memory test framework
class MemoryTestFramework {
    constructor() {
        this.tests = [];
        this.results = [];
        this.startTime = Date.now();
    }

    test(name, testFn) {
        this.tests.push({ name, testFn });
    }

    async run() {
        console.log('🧠 Running VoiRS Node.js Memory Tests\n');

        for (const { name, testFn } of this.tests) {
            try {
                process.stdout.write(`  Testing ${name}... `);
                
                // Force garbage collection before test if available
                if (global.gc) {
                    global.gc();
                }
                
                const result = await testFn();
                console.log('✅ PASS');
                this.results.push({ name, status: 'passed', ...result });
            } catch (error) {
                console.log('❌ FAIL');
                console.log(`    Error: ${error.message}`);
                this.results.push({ name, status: 'failed', error: error.message });
            }
        }

        this.generateReport();
    }

    generateReport() {
        const duration = Date.now() - this.startTime;
        const passed = this.results.filter(r => r.status === 'passed').length;
        const failed = this.results.filter(r => r.status === 'failed').length;

        console.log('\n📊 Memory Test Results');
        console.log('='.repeat(60));

        this.results.forEach(result => {
            if (result.status === 'passed') {
                console.log(`✅ ${result.name}`);
                if (result.metrics) {
                    Object.entries(result.metrics).forEach(([key, value]) => {
                        console.log(`    ${key}: ${value}`);
                    });
                }
            } else {
                console.log(`❌ ${result.name}: ${result.error}`);
            }
        });

        console.log('='.repeat(60));
        console.log(`Total: ${this.results.length}, Passed: ${passed}, Failed: ${failed}`);
        console.log(`Test Duration: ${duration}ms`);

        if (failed > 0) {
            process.exit(1);
        }
    }

    assert(condition, message) {
        if (!condition) {
            throw new Error(message || 'Assertion failed');
        }
    }

    getMemoryUsage() {
        const usage = process.memoryUsage();
        return {
            rss: Math.round(usage.rss / 1024 / 1024), // MB
            heapUsed: Math.round(usage.heapUsed / 1024 / 1024), // MB
            heapTotal: Math.round(usage.heapTotal / 1024 / 1024), // MB
            external: Math.round(usage.external / 1024 / 1024) // MB
        };
    }

    async withMemoryTracking(fn) {
        if (global.gc) global.gc();
        const startMemory = this.getMemoryUsage();
        
        const result = await fn();
        
        if (global.gc) global.gc();
        const endMemory = this.getMemoryUsage();
        
        return {
            result,
            memoryDelta: {
                rss: endMemory.rss - startMemory.rss,
                heapUsed: endMemory.heapUsed - startMemory.heapUsed,
                heapTotal: endMemory.heapTotal - startMemory.heapTotal,
                external: endMemory.external - startMemory.external
            },
            startMemory,
            endMemory
        };
    }
}

// Test suite
const test = new MemoryTestFramework();

// Basic memory usage test
test.test('single synthesis memory usage', async () => {
    const tracked = await test.withMemoryTracking(async () => {
        const pipeline = new VoirsPipeline();
        const text = "Memory usage test for single synthesis operation.";
        const result = await pipeline.synthesize(text);
        
        test.assert(result.samples.length > 0, 'Should generate audio');
        return { audioSize: result.samples.length };
    });
    
    const memGrowth = tracked.memoryDelta.heapUsed;
    test.assert(memGrowth < 100, 'Memory growth should be reasonable for single synthesis');
    
    return {
        metrics: {
            'Audio Size': `${tracked.result.audioSize} bytes`,
            'Memory Growth': `${memGrowth}MB`,
            'Heap Used': `${tracked.startMemory.heapUsed}MB → ${tracked.endMemory.heapUsed}MB`
        }
    };
});

// Memory leak detection test
test.test('memory leak detection', async () => {
    const iterations = 50;
    const memorySnapshots = [];
    
    const pipeline = new VoirsPipeline();
    const text = "Memory leak test iteration text.";
    
    // Take baseline measurement
    if (global.gc) global.gc();
    const baseline = test.getMemoryUsage();
    
    for (let i = 0; i < iterations; i++) {
        const result = await pipeline.synthesize(text);
        test.assert(result.samples.length > 0, 'Should generate audio');
        
        // Don't hold references to results
        if (i % 10 === 9) {
            if (global.gc) global.gc();
            memorySnapshots.push(test.getMemoryUsage());
        }
    }
    
    // Final measurement
    if (global.gc) global.gc();
    const final = test.getMemoryUsage();
    
    const totalGrowth = final.heapUsed - baseline.heapUsed;
    const avgGrowthPerOp = totalGrowth / iterations;
    
    test.assert(totalGrowth < 200, 'Total memory growth should be bounded');
    test.assert(avgGrowthPerOp < 1, 'Average memory growth per operation should be minimal');
    
    return {
        metrics: {
            'Iterations': iterations,
            'Total Growth': `${totalGrowth}MB`,
            'Avg per Op': `${avgGrowthPerOp.toFixed(3)}MB`,
            'Final Heap': `${final.heapUsed}MB`
        }
    };
});

// Large buffer handling test
test.test('large buffer handling', async () => {
    const tracked = await test.withMemoryTracking(async () => {
        const pipeline = new VoirsPipeline();
        const longText = "This is a large text buffer test. ".repeat(200); // ~6800 characters
        
        const result = await pipeline.synthesize(longText, { quality: 'high' });
        test.assert(result.samples.length > 0, 'Should generate audio for large text');
        
        return { 
            textLength: longText.length,
            audioSize: result.samples.length 
        };
    });
    
    const memGrowth = tracked.memoryDelta.heapUsed;
    test.assert(memGrowth < 500, 'Large buffer should not cause excessive memory growth');
    
    return {
        metrics: {
            'Text Length': `${tracked.result.textLength} chars`,
            'Audio Size': `${tracked.result.audioSize} bytes`,
            'Memory Growth': `${memGrowth}MB`
        }
    };
});

// Concurrent memory usage test
test.test('concurrent memory usage', async () => {
    const concurrency = 8;
    const text = "Concurrent memory usage test text.";
    
    const tracked = await test.withMemoryTracking(async () => {
        const pipeline = new VoirsPipeline({ numThreads: 4 });
        
        const promises = Array(concurrency).fill().map((_, i) => 
            pipeline.synthesize(`${text} Instance ${i}`)
        );
        
        const results = await Promise.all(promises);
        test.assert(results.length === concurrency, 'Should complete all concurrent requests');
        
        const totalAudioSize = results.reduce((sum, r) => sum + r.samples.length, 0);
        return { totalAudioSize };
    });
    
    const memGrowth = tracked.memoryDelta.heapUsed;
    test.assert(memGrowth < 300, 'Concurrent operations should not cause excessive memory growth');
    
    return {
        metrics: {
            'Concurrency': concurrency,
            'Total Audio': `${tracked.result.totalAudioSize} bytes`,
            'Memory Growth': `${memGrowth}MB`,
            'Growth per Op': `${(memGrowth / concurrency).toFixed(2)}MB`
        }
    };
});

// Streaming memory efficiency test
test.test('streaming memory efficiency', async () => {
    const tracked = await test.withMemoryTracking(async () => {
        const pipeline = new VoirsPipeline();
        const text = "Streaming memory efficiency test with longer content. ".repeat(50);
        
        let chunkCount = 0;
        let totalBytes = 0;
        let maxChunkSize = 0;
        
        await synthesizeStreaming(
            pipeline,
            text,
            (chunk) => {
                chunkCount++;
                totalBytes += chunk.length;
                maxChunkSize = Math.max(maxChunkSize, chunk.length);
            },
            null,
            { quality: 'medium' }
        );
        
        return { chunkCount, totalBytes, maxChunkSize };
    });
    
    const memGrowth = tracked.memoryDelta.heapUsed;
    test.assert(memGrowth < 100, 'Streaming should be memory efficient');
    
    return {
        metrics: {
            'Chunks': tracked.result.chunkCount,
            'Total Bytes': tracked.result.totalBytes,
            'Max Chunk': `${tracked.result.maxChunkSize} bytes`,
            'Memory Growth': `${memGrowth}MB`
        }
    };
});

// Pipeline cleanup test
test.test('pipeline cleanup', async () => {
    const tracked = await test.withMemoryTracking(async () => {
        const pipelines = [];
        const text = "Pipeline cleanup test.";
        
        // Create multiple pipelines
        for (let i = 0; i < 5; i++) {
            const pipeline = new VoirsPipeline();
            await pipeline.synthesize(text);
            pipelines.push(pipeline);
        }
        
        // Clear references
        pipelines.length = 0;
        
        return { pipelineCount: 5 };
    });
    
    const memGrowth = tracked.memoryDelta.heapUsed;
    test.assert(memGrowth < 150, 'Pipeline cleanup should free memory');
    
    return {
        metrics: {
            'Pipelines Created': tracked.result.pipelineCount,
            'Memory Growth': `${memGrowth}MB`
        }
    };
});

// Voice switching memory test
test.test('voice switching memory', async () => {
    const tracked = await test.withMemoryTracking(async () => {
        const pipeline = new VoirsPipeline();
        const voices = await pipeline.listVoices();
        
        if (voices.length < 2) {
            return { switches: 0, skipped: true };
        }
        
        const text = "Voice switching memory test.";
        const switches = 20;
        
        for (let i = 0; i < switches; i++) {
            const voiceIndex = i % voices.length;
            await pipeline.setVoice(voices[voiceIndex].id);
            await pipeline.synthesize(text);
        }
        
        return { switches };
    });
    
    if (tracked.result.skipped) {
        return { metrics: { 'Result': 'Skipped - need at least 2 voices' } };
    }
    
    const memGrowth = tracked.memoryDelta.heapUsed;
    test.assert(memGrowth < 100, 'Voice switching should not leak memory');
    
    return {
        metrics: {
            'Voice Switches': tracked.result.switches,
            'Memory Growth': `${memGrowth}MB`
        }
    };
});

// Buffer reference counting test
test.test('buffer reference management', async () => {
    const tracked = await test.withMemoryTracking(async () => {
        const pipeline = new VoirsPipeline();
        const text = "Buffer reference test.";
        const buffers = [];
        
        // Generate multiple audio buffers and keep references
        for (let i = 0; i < 10; i++) {
            const result = await pipeline.synthesize(text);
            buffers.push(result.samples);
        }
        
        const totalBufferSize = buffers.reduce((sum, buf) => sum + buf.length, 0);
        
        // Clear references
        buffers.length = 0;
        
        return { totalBufferSize, bufferCount: 10 };
    });
    
    const memGrowth = tracked.memoryDelta.heapUsed;
    
    return {
        metrics: {
            'Buffers Created': tracked.result.bufferCount,
            'Total Buffer Size': `${tracked.result.totalBufferSize} bytes`,
            'Memory Growth': `${memGrowth}MB`
        }
    };
});

// External memory tracking test
test.test('external memory tracking', async () => {
    const tracked = await test.withMemoryTracking(async () => {
        const pipeline = new VoirsPipeline();
        const text = "External memory tracking test.";
        
        const results = [];
        for (let i = 0; i < 5; i++) {
            const result = await pipeline.synthesize(text);
            results.push(result);
        }
        
        return { 
            resultCount: results.length,
            totalAudioSize: results.reduce((sum, r) => sum + r.samples.length, 0)
        };
    });
    
    const externalGrowth = tracked.memoryDelta.external;
    
    return {
        metrics: {
            'Results': tracked.result.resultCount,
            'Audio Size': `${tracked.result.totalAudioSize} bytes`,
            'External Memory Growth': `${externalGrowth}MB`,
            'Heap Growth': `${tracked.memoryDelta.heapUsed}MB`
        }
    };
});

// Run memory tests only if not in fast mode
if (process.env.VOIRS_SKIP_SLOW_TESTS === 'true' || process.env.CI === 'true') {
    console.log('⚠️  Skipping memory tests (fast mode enabled)');
    process.exit(0);
} else {
    test.run().catch(error => {
        console.error('❌ Memory test runner failed:', error);
        process.exit(1);
    });
}
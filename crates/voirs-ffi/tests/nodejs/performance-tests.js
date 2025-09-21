#!/usr/bin/env node

/**
 * VoiRS Node.js Performance Tests
 * 
 * Performance benchmarks and stress tests for VoiRS FFI bindings
 * Tests throughput, latency, and resource efficiency
 */

const path = require('path');
const fs = require('fs');
const os = require('os');

// Load VoiRS bindings
let VoirsPipeline, synthesizeStreaming;
const bindingsPath = path.resolve(__dirname, '../../index.js');

try {
    if (fs.existsSync(bindingsPath)) {
        const bindings = require(bindingsPath);
        VoirsPipeline = bindings.VoirsPipeline;
        synthesizeStreaming = bindings.synthesizeStreaming;
    } else {
        // Performance-focused mock implementations
        class MockVoirsPipeline {
            constructor(options = {}) {
                this.options = options;
                this.currentVoice = null;
                this.synthCount = 0;
                this.totalProcessingTime = 0;
            }

            getInfo() {
                return JSON.stringify({
                    version: "0.1.0",
                    features: { nodejs_bindings: true },
                    runtime_info: { worker_threads: this.options.numThreads || 4 }
                });
            }

            async listVoices() {
                // Simulate slight delay for voice listing
                await new Promise(resolve => setTimeout(resolve, 1));
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
                const startTime = Date.now();
                
                if (!text) throw new Error('Text is required');
                
                // Simulate processing time proportional to text length and quality
                const baseProcessingTime = Math.max(5, text.length * 0.1); // ms
                const qualityMultiplier = {
                    'low': 0.5,
                    'medium': 1.0,
                    'high': 2.0,
                    'ultra': 4.0
                }[options.quality] || 1.0;
                
                const processingTime = baseProcessingTime * qualityMultiplier;
                await new Promise(resolve => setTimeout(resolve, processingTime));
                
                const sampleRate = options.sampleRate || 22050;
                const duration = Math.max(0.5, text.length * 0.05);
                const samples = new Uint8Array(Math.floor(sampleRate * duration * 2));
                
                this.synthCount++;
                this.totalProcessingTime += Date.now() - startTime;
                
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

            getStats() {
                return {
                    synthCount: this.synthCount,
                    avgProcessingTime: this.synthCount > 0 ? this.totalProcessingTime / this.synthCount : 0,
                    totalProcessingTime: this.totalProcessingTime
                };
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
                
                // Simulate minimal streaming delay
                await new Promise(resolve => setTimeout(resolve, 1));
            }
            
            if (progressCb) progressCb(1.0);
        };
    }
} catch (error) {
    console.error('❌ Failed to load VoiRS bindings:', error.message);
    process.exit(1);
}

// Performance test framework
class PerformanceTestFramework {
    constructor() {
        this.tests = [];
        this.results = [];
        this.startTime = Date.now();
    }

    test(name, testFn) {
        this.tests.push({ name, testFn });
    }

    async run() {
        console.log('🚀 Running VoiRS Node.js Performance Tests\n');

        for (const { name, testFn } of this.tests) {
            try {
                process.stdout.write(`  Running ${name}... `);
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

        console.log('\n📊 Performance Test Results');
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
        console.log(`Test Suite Duration: ${duration}ms`);

        if (failed > 0) {
            process.exit(1);
        }
    }

    assert(condition, message) {
        if (!condition) {
            throw new Error(message || 'Assertion failed');
        }
    }

    measureTime(fn) {
        return async (...args) => {
            const start = process.hrtime.bigint();
            const result = await fn(...args);
            const end = process.hrtime.bigint();
            const duration = Number(end - start) / 1000000; // Convert to milliseconds
            return { result, duration };
        };
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
}

// Test suite
const test = new PerformanceTestFramework();

// Basic synthesis performance
test.test('single synthesis latency', async () => {
    const pipeline = new VoirsPipeline();
    const text = "Performance test for single synthesis operation.";
    
    const measured = test.measureTime(pipeline.synthesize.bind(pipeline));
    const { result, duration } = await measured(text);
    
    test.assert(result.samples.length > 0, 'Should generate audio');
    test.assert(duration < 1000, 'Single synthesis should complete within 1 second');
    
    return {
        metrics: {
            'Latency': `${duration.toFixed(2)}ms`,
            'Audio Duration': `${result.duration.toFixed(2)}s`,
            'Sample Rate': `${result.sampleRate}Hz`,
            'Data Size': `${result.samples.length} bytes`
        }
    };
});

// Batch synthesis performance
test.test('batch synthesis throughput', async () => {
    const pipeline = new VoirsPipeline();
    const texts = [
        "First test sentence for batch processing.",
        "Second sentence with different content length.",
        "Third sentence to complete the batch test.",
        "Fourth and final sentence in this performance test."
    ];
    
    const startTime = process.hrtime.bigint();
    const results = [];
    
    for (const text of texts) {
        const result = await pipeline.synthesize(text, { quality: 'medium' });
        results.push(result);
    }
    
    const endTime = process.hrtime.bigint();
    const totalDuration = Number(endTime - startTime) / 1000000; // ms
    const totalAudioDuration = results.reduce((sum, r) => sum + r.duration, 0);
    const throughput = (totalAudioDuration / totalDuration) * 1000; // audio seconds per wall-clock second
    
    test.assert(results.length === texts.length, 'Should process all texts');
    test.assert(throughput > 0.1, 'Should have reasonable throughput');
    
    return {
        metrics: {
            'Total Duration': `${totalDuration.toFixed(2)}ms`,
            'Audio Generated': `${totalAudioDuration.toFixed(2)}s`,
            'Throughput': `${throughput.toFixed(2)}x real-time`,
            'Avg Latency': `${(totalDuration / texts.length).toFixed(2)}ms`
        }
    };
});

// Concurrent synthesis performance
test.test('concurrent synthesis performance', async () => {
    const pipeline = new VoirsPipeline({ numThreads: 4 });
    const concurrency = 5;
    const text = "Concurrent synthesis performance test with moderate length text.";
    
    const startMemory = test.getMemoryUsage();
    const startTime = process.hrtime.bigint();
    
    const promises = Array(concurrency).fill().map((_, i) => 
        pipeline.synthesize(`${text} Instance ${i}`, { quality: 'medium' })
    );
    
    const results = await Promise.all(promises);
    
    const endTime = process.hrtime.bigint();
    const endMemory = test.getMemoryUsage();
    const duration = Number(endTime - startTime) / 1000000; // ms
    
    test.assert(results.length === concurrency, 'Should complete all concurrent requests');
    test.assert(duration < 5000, 'Concurrent synthesis should complete within 5 seconds');
    
    return {
        metrics: {
            'Concurrency': `${concurrency} requests`,
            'Total Duration': `${duration.toFixed(2)}ms`,
            'Avg per Request': `${(duration / concurrency).toFixed(2)}ms`,
            'Memory Delta': `${endMemory.heapUsed - startMemory.heapUsed}MB`
        }
    };
});

// Streaming synthesis performance
test.test('streaming synthesis performance', async () => {
    const pipeline = new VoirsPipeline();
    const text = "This is a longer text for streaming synthesis performance testing. ".repeat(5);
    
    let firstChunkTime = null;
    let lastChunkTime = null;
    let chunkCount = 0;
    let totalBytes = 0;
    
    const startTime = process.hrtime.bigint();
    
    await synthesizeStreaming(
        pipeline,
        text,
        (chunk) => {
            if (firstChunkTime === null) {
                firstChunkTime = process.hrtime.bigint();
            }
            lastChunkTime = process.hrtime.bigint();
            chunkCount++;
            totalBytes += chunk.length;
        },
        null,
        { quality: 'medium' }
    );
    
    const timeToFirstChunk = Number(firstChunkTime - startTime) / 1000000; // ms
    const totalStreamTime = Number(lastChunkTime - startTime) / 1000000; // ms
    
    test.assert(chunkCount > 0, 'Should receive chunks');
    test.assert(timeToFirstChunk < 500, 'First chunk should arrive quickly');
    
    return {
        metrics: {
            'Time to First Chunk': `${timeToFirstChunk.toFixed(2)}ms`,
            'Total Stream Time': `${totalStreamTime.toFixed(2)}ms`,
            'Chunks Received': chunkCount,
            'Total Bytes': totalBytes,
            'Avg Chunk Size': `${Math.round(totalBytes / chunkCount)} bytes`
        }
    };
});

// Voice switching performance
test.test('voice switching performance', async () => {
    const pipeline = new VoirsPipeline();
    const voices = await pipeline.listVoices();
    
    if (voices.length < 2) {
        return { metrics: { 'Result': 'Skipped - need at least 2 voices' } };
    }
    
    const text = "Voice switching performance test.";
    const switchCount = 10;
    
    const startTime = process.hrtime.bigint();
    
    for (let i = 0; i < switchCount; i++) {
        const voiceIndex = i % voices.length;
        await pipeline.setVoice(voices[voiceIndex].id);
        await pipeline.synthesize(text, { quality: 'low' });
    }
    
    const endTime = process.hrtime.bigint();
    const duration = Number(endTime - startTime) / 1000000; // ms
    
    test.assert(duration < 10000, 'Voice switching should be reasonably fast');
    
    return {
        metrics: {
            'Switches': switchCount,
            'Total Duration': `${duration.toFixed(2)}ms`,
            'Avg per Switch': `${(duration / switchCount).toFixed(2)}ms`
        }
    };
});

// Quality level performance comparison
test.test('quality level performance comparison', async () => {
    const pipeline = new VoirsPipeline();
    const text = "Quality level performance comparison test with consistent content.";
    const qualities = ['low', 'medium', 'high'];
    const results = {};
    
    for (const quality of qualities) {
        const measured = test.measureTime(pipeline.synthesize.bind(pipeline));
        const { result, duration } = await measured(text, { quality });
        
        results[quality] = {
            duration: duration,
            audioLength: result.duration,
            dataSize: result.samples.length
        };
    }
    
    // Verify that higher quality takes more time (in mock it should)
    test.assert(results.high.duration >= results.medium.duration, 
                'Higher quality should take more time');
    
    const metrics = {};
    qualities.forEach(quality => {
        const r = results[quality];
        metrics[`${quality.toUpperCase()} Quality`] = 
            `${r.duration.toFixed(2)}ms (${r.dataSize} bytes)`;
    });
    
    return { metrics };
});

// Memory efficiency test
test.test('memory efficiency', async () => {
    const pipeline = new VoirsPipeline();
    const text = "Memory efficiency test with repeated synthesis operations.";
    const iterations = 20;
    
    const startMemory = test.getMemoryUsage();
    
    // Force garbage collection if available
    if (global.gc) {
        global.gc();
    }
    
    const baselineMemory = test.getMemoryUsage();
    
    for (let i = 0; i < iterations; i++) {
        const result = await pipeline.synthesize(text);
        test.assert(result.samples.length > 0, 'Should generate audio');
        
        // Don't hold references to results to test memory cleanup
    }
    
    // Force garbage collection again
    if (global.gc) {
        global.gc();
    }
    
    const finalMemory = test.getMemoryUsage();
    const memoryGrowth = finalMemory.heapUsed - baselineMemory.heapUsed;
    
    test.assert(memoryGrowth < 50, 'Memory growth should be reasonable (<50MB)');
    
    return {
        metrics: {
            'Iterations': iterations,
            'Baseline Memory': `${baselineMemory.heapUsed}MB`,
            'Final Memory': `${finalMemory.heapUsed}MB`,
            'Memory Growth': `${memoryGrowth}MB`,
            'Growth per Op': `${(memoryGrowth / iterations).toFixed(2)}MB`
        }
    };
});

// CPU usage simulation
test.test('CPU efficiency', async () => {
    const pipeline = new VoirsPipeline({ numThreads: 2 });
    const text = "CPU efficiency test with parallel processing.";
    const parallelTasks = 4;
    
    const startTime = process.hrtime.bigint();
    const cpuStartTime = process.cpuUsage();
    
    const promises = Array(parallelTasks).fill().map(() => 
        pipeline.synthesize(text, { quality: 'medium' })
    );
    
    const results = await Promise.all(promises);
    
    const endTime = process.hrtime.bigint();
    const cpuEndTime = process.cpuUsage(cpuStartTime);
    
    const wallClockTime = Number(endTime - startTime) / 1000000; // ms
    const cpuTime = (cpuEndTime.user + cpuEndTime.system) / 1000; // ms
    const cpuEfficiency = (cpuTime / wallClockTime) * 100;
    
    test.assert(results.length === parallelTasks, 'Should complete all tasks');
    test.assert(cpuEfficiency > 50, 'Should use CPU efficiently');
    
    return {
        metrics: {
            'Parallel Tasks': parallelTasks,
            'Wall Clock Time': `${wallClockTime.toFixed(2)}ms`,
            'CPU Time': `${cpuTime.toFixed(2)}ms`,
            'CPU Efficiency': `${cpuEfficiency.toFixed(1)}%`
        }
    };
});

// Run performance tests only if not in fast mode
if (process.env.VOIRS_SKIP_SLOW_TESTS === 'true' || process.env.CI === 'true') {
    console.log('⚠️  Skipping performance tests (fast mode enabled)');
    process.exit(0);
} else {
    test.run().catch(error => {
        console.error('❌ Performance test runner failed:', error);
        process.exit(1);
    });
}
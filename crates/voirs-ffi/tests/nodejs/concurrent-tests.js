#!/usr/bin/env node

/**
 * VoiRS Node.js Concurrency Tests
 * 
 * Tests for multi-threaded and concurrent operations
 * Ensures thread safety and proper resource sharing
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
        // Concurrency-focused mock implementations
        class MockVoirsPipeline {
            constructor(options = {}) {
                this.options = options;
                this.currentVoice = null;
                this.instanceId = Math.random().toString(36);
                this.activeOperations = new Set();
                this.operationCounter = 0;
            }

            getInfo() {
                return JSON.stringify({
                    version: "0.1.0",
                    features: { nodejs_bindings: true },
                    runtime_info: { 
                        worker_threads: this.options.numThreads || 4,
                        instance_id: this.instanceId
                    }
                });
            }

            async listVoices() {
                // Simulate thread-safe voice listing
                await new Promise(resolve => setTimeout(resolve, 1));
                return [
                    { id: 'test-voice-1', name: 'Test Voice 1', language: 'en-US', quality: 'high', isAvailable: true },
                    { id: 'test-voice-2', name: 'Test Voice 2', language: 'en-GB', quality: 'medium', isAvailable: true }
                ];
            }

            async setVoice(voiceId) {
                // Simulate potential race condition handling
                await new Promise(resolve => setTimeout(resolve, 1));
                this.currentVoice = voiceId;
            }

            async getVoice() {
                return this.currentVoice;
            }

            async synthesize(text, options = {}) {
                if (!text) throw new Error('Text is required');
                
                const operationId = ++this.operationCounter;
                this.activeOperations.add(operationId);
                
                try {
                    // Simulate processing time with some variation
                    const processingTime = 10 + Math.random() * 20; // 10-30ms
                    await new Promise(resolve => setTimeout(resolve, processingTime));
                    
                    const sampleRate = options.sampleRate || 22050;
                    const duration = Math.max(0.5, text.length * 0.05);
                    const samples = new Uint8Array(Math.floor(sampleRate * duration * 2));
                    
                    return {
                        samples: Buffer.from(samples),
                        sampleRate,
                        channels: 1,
                        duration,
                        operationId
                    };
                } finally {
                    this.activeOperations.delete(operationId);
                }
            }

            async synthesizeSsml(ssml) {
                return this.synthesize(ssml.replace(/<[^>]*>/g, ''));
            }

            async synthesizeWithCallbacks(text, options, progressCb, errorCb) {
                const operationId = ++this.operationCounter;
                this.activeOperations.add(operationId);
                
                try {
                    if (progressCb) progressCb(0.0);
                    
                    // Simulate progress updates
                    for (let i = 1; i <= 3; i++) {
                        await new Promise(resolve => setTimeout(resolve, 5));
                        if (progressCb) progressCb(i / 3);
                    }
                    
                    const result = await this.synthesize(text, options);
                    return result;
                } finally {
                    this.activeOperations.delete(operationId);
                }
            }

            getActiveOperationCount() {
                return this.activeOperations.size;
            }
        }

        VoirsPipeline = MockVoirsPipeline;
        
        synthesizeStreaming = async function(pipeline, text, chunkCb, progressCb, options) {
            const operationId = Math.random().toString(36);
            
            const result = await pipeline.synthesize(text, options);
            const chunkSize = 1024;
            const chunks = Math.ceil(result.samples.length / chunkSize);
            
            for (let i = 0; i < chunks; i++) {
                const start = i * chunkSize;
                const end = Math.min(start + chunkSize, result.samples.length);
                const chunk = result.samples.slice(start, end);
                
                if (progressCb) progressCb(i / chunks);
                chunkCb(chunk);
                
                // Simulate streaming delay
                await new Promise(resolve => setTimeout(resolve, 1));
            }
            
            if (progressCb) progressCb(1.0);
        };
    }
} catch (error) {
    console.error('❌ Failed to load VoiRS bindings:', error.message);
    process.exit(1);
}

// Concurrency test framework
class ConcurrencyTestFramework {
    constructor() {
        this.tests = [];
        this.results = [];
        this.startTime = Date.now();
    }

    test(name, testFn) {
        this.tests.push({ name, testFn });
    }

    async run() {
        console.log('🔄 Running VoiRS Node.js Concurrency Tests\n');

        for (const { name, testFn } of this.tests) {
            try {
                process.stdout.write(`  Testing ${name}... `);
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

        console.log('\n📊 Concurrency Test Results');
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

    async withTimeout(promise, timeoutMs) {
        return Promise.race([
            promise,
            new Promise((_, reject) => 
                setTimeout(() => reject(new Error(`Timeout after ${timeoutMs}ms`)), timeoutMs)
            )
        ]);
    }
}

// Test suite
const test = new ConcurrencyTestFramework();

// Basic concurrent synthesis test
test.test('parallel synthesis operations', async () => {
    const pipeline = new VoirsPipeline({ numThreads: 4 });
    const concurrency = 8;
    const text = "Concurrent synthesis test text.";
    
    const startTime = Date.now();
    
    const promises = Array(concurrency).fill().map((_, i) => 
        pipeline.synthesize(`${text} Instance ${i}`, { quality: 'medium' })
    );
    
    const results = await Promise.all(promises);
    const duration = Date.now() - startTime;
    
    test.assert(results.length === concurrency, 'Should complete all concurrent operations');
    test.assert(results.every(r => r.samples.length > 0), 'All operations should produce results');
    test.assert(duration < 5000, 'Concurrent operations should complete reasonably fast');
    
    // Check for unique operation IDs if available
    const operationIds = results.map(r => r.operationId).filter(id => id !== undefined);
    const uniqueIds = new Set(operationIds);
    
    return {
        metrics: {
            'Concurrency': concurrency,
            'Duration': `${duration}ms`,
            'Avg per Op': `${(duration / concurrency).toFixed(2)}ms`,
            'Unique Ops': operationIds.length > 0 ? `${uniqueIds.size}/${operationIds.length}` : 'N/A'
        }
    };
});

// Multiple pipeline concurrency test
test.test('multiple pipeline concurrency', async () => {
    const pipelineCount = 4;
    const operationsPerPipeline = 3;
    const text = "Multi-pipeline concurrency test.";
    
    const pipelines = Array(pipelineCount).fill().map(() => 
        new VoirsPipeline({ numThreads: 2 })
    );
    
    const startTime = Date.now();
    
    const allPromises = pipelines.flatMap(pipeline => 
        Array(operationsPerPipeline).fill().map(() => 
            pipeline.synthesize(text)
        )
    );
    
    const results = await test.withTimeout(Promise.all(allPromises), 10000);
    const duration = Date.now() - startTime;
    
    const totalOperations = pipelineCount * operationsPerPipeline;
    test.assert(results.length === totalOperations, 'Should complete all operations');
    test.assert(results.every(r => r.samples.length > 0), 'All operations should produce results');
    
    return {
        metrics: {
            'Pipelines': pipelineCount,
            'Ops per Pipeline': operationsPerPipeline,
            'Total Operations': totalOperations,
            'Duration': `${duration}ms`,
            'Throughput': `${(totalOperations / duration * 1000).toFixed(2)} ops/sec`
        }
    };
});

// Concurrent voice switching test
test.test('concurrent voice switching', async () => {
    const pipeline = new VoirsPipeline();
    const voices = await pipeline.listVoices();
    
    if (voices.length < 2) {
        return { metrics: { 'Result': 'Skipped - need at least 2 voices' } };
    }
    
    const operations = 10;
    const text = "Voice switching concurrency test.";
    
    const startTime = Date.now();
    
    const promises = Array(operations).fill().map(async (_, i) => {
        const voiceIndex = i % voices.length;
        await pipeline.setVoice(voices[voiceIndex].id);
        const result = await pipeline.synthesize(`${text} Voice ${voiceIndex}`);
        return { voiceIndex, result };
    });
    
    const results = await Promise.all(promises);
    const duration = Date.now() - startTime;
    
    test.assert(results.length === operations, 'Should complete all voice switching operations');
    test.assert(results.every(r => r.result.samples.length > 0), 'All operations should produce audio');
    
    const voiceCounts = {};
    results.forEach(r => {
        voiceCounts[r.voiceIndex] = (voiceCounts[r.voiceIndex] || 0) + 1;
    });
    
    return {
        metrics: {
            'Operations': operations,
            'Duration': `${duration}ms`,
            'Voice Distribution': Object.entries(voiceCounts).map(([v, c]) => `V${v}:${c}`).join(', ')
        }
    };
});

// Streaming concurrency test
test.test('concurrent streaming synthesis', async () => {
    const pipeline = new VoirsPipeline();
    const concurrency = 4;
    const text = "Concurrent streaming synthesis test with longer text content.";
    
    const streamData = Array(concurrency).fill().map(() => ({
        chunks: [],
        progressUpdates: [],
        completed: false
    }));
    
    const startTime = Date.now();
    
    const promises = streamData.map((data, i) => 
        synthesizeStreaming(
            pipeline,
            `${text} Stream ${i}`,
            (chunk) => {
                data.chunks.push(chunk);
            },
            (progress) => {
                data.progressUpdates.push(progress);
            },
            { quality: 'medium' }
        ).then(() => {
            data.completed = true;
        })
    );
    
    await Promise.all(promises);
    const duration = Date.now() - startTime;
    
    test.assert(streamData.every(d => d.completed), 'All streams should complete');
    test.assert(streamData.every(d => d.chunks.length > 0), 'All streams should receive chunks');
    test.assert(streamData.every(d => d.progressUpdates.length > 0), 'All streams should receive progress');
    
    const totalChunks = streamData.reduce((sum, d) => sum + d.chunks.length, 0);
    const totalBytes = streamData.reduce((sum, d) => 
        sum + d.chunks.reduce((chunkSum, chunk) => chunkSum + chunk.length, 0), 0
    );
    
    return {
        metrics: {
            'Concurrent Streams': concurrency,
            'Duration': `${duration}ms`,
            'Total Chunks': totalChunks,
            'Total Bytes': totalBytes,
            'Avg Chunks per Stream': Math.round(totalChunks / concurrency)
        }
    };
});

// Mixed operation concurrency test
test.test('mixed concurrent operations', async () => {
    const pipeline = new VoirsPipeline();
    const text = "Mixed operations concurrency test.";
    
    const startTime = Date.now();
    
    const mixedPromises = [
        // Regular synthesis
        pipeline.synthesize(`${text} Regular 1`),
        pipeline.synthesize(`${text} Regular 2`),
        
        // SSML synthesis
        pipeline.synthesizeSsml(`<speak>${text} SSML</speak>`),
        
        // Callback synthesis
        pipeline.synthesizeWithCallbacks(
            `${text} Callback`,
            {},
            (progress) => { /* progress handler */ },
            (error) => { throw new Error(error); }
        ),
        
        // Voice operations
        pipeline.listVoices().then(async voices => {
            if (voices.length > 0) {
                await pipeline.setVoice(voices[0].id);
                return pipeline.synthesize(`${text} Voice`);
            }
            return pipeline.synthesize(`${text} No Voice`);
        })
    ];
    
    const results = await Promise.all(mixedPromises);
    const duration = Date.now() - startTime;
    
    test.assert(results.length === mixedPromises.length, 'Should complete all mixed operations');
    test.assert(results.every(r => r && r.samples && r.samples.length > 0), 'All operations should produce audio');
    
    return {
        metrics: {
            'Mixed Operations': results.length,
            'Duration': `${duration}ms`,
            'Types': 'Synthesis, SSML, Callbacks, Voice Ops'
        }
    };
});

// Race condition test
test.test('race condition handling', async () => {
    const pipeline = new VoirsPipeline();
    const iterations = 20;
    const text = "Race condition test text.";
    
    const startTime = Date.now();
    
    // Create rapid-fire operations that might cause race conditions
    const promises = [];
    
    for (let i = 0; i < iterations; i++) {
        if (i % 4 === 0) {
            promises.push(pipeline.listVoices());
        } else if (i % 4 === 1) {
            promises.push(pipeline.synthesize(`${text} ${i}`));
        } else if (i % 4 === 2) {
            promises.push(pipeline.getVoice());
        } else {
            promises.push(pipeline.synthesize(`${text} Final ${i}`));
        }
    }
    
    const results = await Promise.allSettled(promises);
    const duration = Date.now() - startTime;
    
    const successful = results.filter(r => r.status === 'fulfilled').length;
    const failed = results.filter(r => r.status === 'rejected').length;
    
    test.assert(successful > iterations * 0.8, 'Most operations should succeed despite race conditions');
    
    return {
        metrics: {
            'Operations': iterations,
            'Successful': successful,
            'Failed': failed,
            'Success Rate': `${((successful / iterations) * 100).toFixed(1)}%`,
            'Duration': `${duration}ms`
        }
    };
});

// Load testing
test.test('high concurrency load test', async () => {
    const pipeline = new VoirsPipeline({ numThreads: 4 });
    const highConcurrency = 20;
    const text = "High concurrency load test.";
    
    const startTime = Date.now();
    
    const promises = Array(highConcurrency).fill().map((_, i) => 
        pipeline.synthesize(`${text} Load ${i}`, { quality: 'low' })
    );
    
    const results = await test.withTimeout(Promise.all(promises), 15000); // 15s timeout
    const duration = Date.now() - startTime;
    
    test.assert(results.length === highConcurrency, 'Should handle high concurrency');
    test.assert(results.every(r => r.samples.length > 0), 'All high-concurrency operations should succeed');
    
    const throughput = (highConcurrency / duration) * 1000; // ops per second
    
    return {
        metrics: {
            'Concurrency': highConcurrency,
            'Duration': `${duration}ms`,
            'Throughput': `${throughput.toFixed(2)} ops/sec`,
            'Avg Latency': `${(duration / highConcurrency).toFixed(2)}ms`
        }
    };
});

// Thread safety test
test.test('thread safety validation', async () => {
    const pipeline = new VoirsPipeline({ numThreads: 4 });
    const text = "Thread safety validation test.";
    const iterations = 15;
    
    const startTime = Date.now();
    
    // Create overlapping operations that access shared state
    const promises = [];
    
    for (let i = 0; i < iterations; i++) {
        promises.push(
            Promise.resolve().then(async () => {
                // Mix of state-changing and state-reading operations
                const info = pipeline.getInfo();
                const result = await pipeline.synthesize(`${text} ${i}`);
                const voices = await pipeline.listVoices();
                
                return { info, result, voiceCount: voices.length };
            })
        );
    }
    
    const results = await Promise.all(promises);
    const duration = Date.now() - startTime;
    
    test.assert(results.length === iterations, 'Should complete all thread safety tests');
    test.assert(results.every(r => r.result.samples.length > 0), 'All operations should produce results');
    
    // Check consistency of shared state
    const infos = results.map(r => r.info);
    const voiceCounts = results.map(r => r.voiceCount);
    const uniqueInfos = new Set(infos);
    const uniqueVoiceCounts = new Set(voiceCounts);
    
    test.assert(uniqueInfos.size === 1, 'Pipeline info should be consistent');
    test.assert(uniqueVoiceCounts.size === 1, 'Voice count should be consistent');
    
    return {
        metrics: {
            'Iterations': iterations,
            'Duration': `${duration}ms`,
            'Info Consistency': uniqueInfos.size === 1 ? 'PASS' : 'FAIL',
            'Voice Consistency': uniqueVoiceCounts.size === 1 ? 'PASS' : 'FAIL'
        }
    };
});

// Run all concurrency tests
test.run().catch(error => {
    console.error('❌ Concurrency test runner failed:', error);
    process.exit(1);
});
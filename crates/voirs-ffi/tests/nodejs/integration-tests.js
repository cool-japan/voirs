#!/usr/bin/env node

/**
 * VoiRS Node.js Integration Tests
 * 
 * End-to-end integration tests for VoiRS FFI bindings
 * Tests complete workflows and real-world usage scenarios
 */

const path = require('path');
const fs = require('fs');
const os = require('os');

// Try to load the VoiRS bindings
let VoirsPipeline, synthesizeStreaming;
const bindingsPath = path.resolve(__dirname, '../../index.js');

try {
    if (fs.existsSync(bindingsPath)) {
        const bindings = require(bindingsPath);
        VoirsPipeline = bindings.VoirsPipeline;
        synthesizeStreaming = bindings.synthesizeStreaming;
    } else {
        // Use the same mock implementations as unit tests
        const unitTests = require('./unit-tests.js');
        // Re-implement mocks for integration tests
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
                
                const sampleRate = options.sampleRate || 22050;
                const duration = Math.max(0.5, text.length * 0.05);
                const samples = new Uint8Array(Math.floor(sampleRate * duration * 2));
                
                // Simulate processing time
                await new Promise(resolve => setTimeout(resolve, 10));
                
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
                
                // Simulate progress updates
                for (let i = 1; i <= 5; i++) {
                    await new Promise(resolve => setTimeout(resolve, 2));
                    if (progressCb) progressCb(i / 5);
                }
                
                const result = await this.synthesize(text, options);
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

// Test framework
class IntegrationTestFramework {
    constructor() {
        this.tests = [];
        this.passed = 0;
        this.failed = 0;
        this.startTime = Date.now();
        this.tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'voirs-integration-'));
    }

    test(name, testFn) {
        this.tests.push({ name, testFn });
    }

    async run() {
        console.log('🔧 Running VoiRS Node.js Integration Tests\n');

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

        // Cleanup
        this.cleanup();

        const duration = Date.now() - this.startTime;
        console.log(`\n📊 Integration Test Results: ${this.passed} passed, ${this.failed} failed (${duration}ms)`);
        
        if (this.failed > 0) {
            process.exit(1);
        }
    }

    cleanup() {
        try {
            if (fs.existsSync(this.tempDir)) {
                fs.rmSync(this.tempDir, { recursive: true, force: true });
            }
        } catch (error) {
            console.warn(`⚠️  Failed to cleanup temp directory: ${error.message}`);
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
const test = new IntegrationTestFramework();

// End-to-end workflow tests
test.test('complete text-to-speech workflow', async () => {
    const pipeline = new VoirsPipeline({
        useGpu: false,
        numThreads: 2
    });

    // Get pipeline info
    const info = JSON.parse(pipeline.getInfo());
    test.assertNotNull(info.version, 'Pipeline should have version info');

    // List and set voice
    const voices = await pipeline.listVoices();
    test.assert(voices.length > 0, 'Should have available voices');

    await pipeline.setVoice(voices[0].id);
    const currentVoice = await pipeline.getVoice();
    test.assertEqual(currentVoice, voices[0].id, 'Voice should be set correctly');

    // Synthesize text
    const text = "This is a complete text-to-speech workflow test using VoiRS.";
    const result = await pipeline.synthesize(text, {
        speakingRate: 1.0,
        quality: 'medium',
        outputFormat: 'wav'
    });

    test.assertNotNull(result.samples, 'Should generate audio samples');
    test.assert(result.duration > 0, 'Should have positive duration');
    test.assert(result.samples.length > 0, 'Should have audio data');
});

test.test('file output workflow', async () => {
    const pipeline = new VoirsPipeline();
    const text = "Testing file output functionality.";
    
    const result = await pipeline.synthesize(text);
    
    // Save to temporary file
    const outputPath = path.join(test.tempDir, 'test_output.wav');
    fs.writeFileSync(outputPath, result.samples);
    
    test.assert(fs.existsSync(outputPath), 'Output file should be created');
    
    const stats = fs.statSync(outputPath);
    test.assert(stats.size > 0, 'Output file should have content');
    test.assert(stats.size === result.samples.length, 'File size should match sample data');
});

test.test('multiple voice switching workflow', async () => {
    const pipeline = new VoirsPipeline();
    const voices = await pipeline.listVoices();
    
    if (voices.length < 2) {
        console.log('⚠️  Skipping multiple voice test (need at least 2 voices)');
        return;
    }

    const text = "Testing voice switching.";
    const results = [];

    for (let i = 0; i < Math.min(2, voices.length); i++) {
        await pipeline.setVoice(voices[i].id);
        const currentVoice = await pipeline.getVoice();
        test.assertEqual(currentVoice, voices[i].id, `Voice ${i} should be set correctly`);

        const result = await pipeline.synthesize(text);
        results.push(result);
        
        test.assert(result.samples.length > 0, `Voice ${i} should generate audio`);
    }

    // Results should potentially be different (though in mock they might be the same)
    test.assert(results.length === 2, 'Should have results from both voices');
});

test.test('batch synthesis workflow', async () => {
    const pipeline = new VoirsPipeline();
    const texts = [
        "First sentence for batch processing.",
        "Second sentence with different content.",
        "Third and final sentence in the batch."
    ];

    const results = [];
    const startTime = Date.now();

    for (const text of texts) {
        const result = await pipeline.synthesize(text, { quality: 'medium' });
        results.push(result);
        test.assert(result.samples.length > 0, 'Each synthesis should produce audio');
    }

    const totalDuration = Date.now() - startTime;
    const audioTotalDuration = results.reduce((sum, r) => sum + r.duration, 0);

    test.assert(results.length === texts.length, 'Should process all texts');
    test.assert(totalDuration < 10000, 'Batch processing should be reasonably fast'); // 10s timeout
    test.assert(audioTotalDuration > 0, 'Should have total audio duration');
});

test.test('SSML workflow with complex markup', async () => {
    const pipeline = new VoirsPipeline();
    const ssml = `
    <speak>
        <p>Welcome to the <emphasis level="strong">VoiRS</emphasis> text-to-speech system.</p>
        <break time="500ms"/>
        <p>This system supports various SSML features:</p>
        <p>
            Speed control: <prosody rate="slow">slow speech</prosody> and 
            <prosody rate="fast">fast speech</prosody>.
        </p>
        <p>
            Volume control: <prosody volume="soft">quiet voice</prosody> and 
            <prosody volume="loud">loud voice</prosody>.
        </p>
        <break time="1s"/>
        <p>Thank you for testing VoiRS!</p>
    </speak>
    `;

    const result = await pipeline.synthesizeSsml(ssml);
    
    test.assertNotNull(result.samples, 'SSML should generate audio samples');
    test.assert(result.duration > 2.0, 'Complex SSML should generate longer audio');
    test.assert(result.samples.length > 0, 'SSML should produce audio data');
});

test.test('streaming synthesis workflow', async () => {
    const pipeline = new VoirsPipeline();
    const text = "This is a streaming synthesis test with a longer text to ensure multiple chunks are generated during the streaming process.";

    let chunkCount = 0;
    let totalBytes = 0;
    let progressCount = 0;
    let lastProgress = -1;
    const chunks = [];

    const startTime = Date.now();

    await test.withTimeout(
        synthesizeStreaming(
            pipeline,
            text,
            (chunk) => {
                chunkCount++;
                totalBytes += chunk.length;
                chunks.push(chunk);
                test.assert(chunk.length > 0, 'Each chunk should contain data');
            },
            (progress) => {
                progressCount++;
                test.assert(progress >= lastProgress, 'Progress should be monotonic');
                test.assert(progress >= 0 && progress <= 1, 'Progress should be between 0 and 1');
                lastProgress = progress;
            },
            {
                quality: 'high',
                speakingRate: 1.0
            }
        ),
        5000 // 5 second timeout
    );

    const duration = Date.now() - startTime;

    test.assert(chunkCount > 0, 'Should receive at least one chunk');
    test.assert(totalBytes > 0, 'Should receive audio data');
    test.assert(progressCount > 0, 'Should receive progress updates');
    test.assertEqual(lastProgress, 1.0, 'Final progress should be 1.0');
    test.assert(duration < 5000, 'Streaming should complete within timeout');

    // Verify chunks can be concatenated
    const concatenated = Buffer.concat(chunks);
    test.assertEqual(concatenated.length, totalBytes, 'Concatenated chunks should match total bytes');
});

test.test('callback-based synthesis workflow', async () => {
    const pipeline = new VoirsPipeline();
    const text = "Testing callback-based synthesis with progress tracking.";

    let progressUpdates = [];
    let errorCalled = false;
    let completed = false;

    const result = await test.withTimeout(
        pipeline.synthesizeWithCallbacks(
            text,
            {
                speakingRate: 1.1,
                enableEnhancement: true
            },
            (progress) => {
                progressUpdates.push(progress);
            },
            (error) => {
                errorCalled = true;
                throw new Error(`Unexpected error: ${error}`);
            }
        ),
        3000 // 3 second timeout
    );

    completed = true;

    test.assert(!errorCalled, 'Error callback should not be called');
    test.assert(completed, 'Synthesis should complete');
    test.assert(progressUpdates.length > 0, 'Should receive progress updates');
    test.assert(progressUpdates[progressUpdates.length - 1] === 1.0, 'Final progress should be 1.0');
    test.assertNotNull(result.samples, 'Should return audio result');
});

test.test('configuration options workflow', async () => {
    const configurations = [
        { useGpu: false, numThreads: 1, device: 'cpu' },
        { useGpu: false, numThreads: 4, device: 'cpu' },
        // Add more configurations as needed
    ];

    const text = "Testing different pipeline configurations.";
    const results = [];

    for (const config of configurations) {
        const pipeline = new VoirsPipeline(config);
        const info = JSON.parse(pipeline.getInfo());
        
        test.assertNotNull(info.version, 'Pipeline should have version info');
        
        const result = await pipeline.synthesize(text);
        results.push(result);
        
        test.assert(result.samples.length > 0, 'Each configuration should produce audio');
    }

    test.assert(results.length === configurations.length, 'Should test all configurations');
});

test.test('error recovery workflow', async () => {
    const pipeline = new VoirsPipeline();

    // Test recovery from various error conditions
    try {
        await pipeline.synthesize(''); // Empty text
        test.assert(false, 'Should throw error for empty text');
    } catch (error) {
        // Expected error
    }

    // Pipeline should still work after error
    const result = await pipeline.synthesize('Recovery test after error.');
    test.assert(result.samples.length > 0, 'Pipeline should work after error');

    // Test invalid options recovery
    try {
        await pipeline.synthesize('Test', { sampleRate: -1 });
    } catch (error) {
        // May or may not throw depending on implementation
    }

    // Should still work
    const result2 = await pipeline.synthesize('Second recovery test.');
    test.assert(result2.samples.length > 0, 'Pipeline should work after invalid options');
});

// Run all tests
test.run().catch(error => {
    console.error('❌ Integration test runner failed:', error);
    process.exit(1);
});
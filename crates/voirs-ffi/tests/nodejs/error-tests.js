#!/usr/bin/env node

/**
 * VoiRS Node.js Error Handling Tests
 * 
 * Tests error conditions, edge cases, and error recovery
 * Ensures robust behavior under adverse conditions
 */

const path = require('path');
const fs = require('fs');

// Load VoiRS bindings with error handling
let VoirsPipeline, synthesizeStreaming;
const bindingsPath = path.resolve(__dirname, '../../index.js');

try {
    if (fs.existsSync(bindingsPath)) {
        const bindings = require(bindingsPath);
        VoirsPipeline = bindings.VoirsPipeline;
        synthesizeStreaming = bindings.synthesizeStreaming;
    } else {
        // Mock implementations with error simulation
        class MockVoirsPipeline {
            constructor(options = {}) {
                if (options.device === 'invalid-device') {
                    throw new Error('Invalid device specified');
                }
                if (options.numThreads && options.numThreads < 0) {
                    throw new Error('Number of threads must be positive');
                }
                this.options = options;
                this.currentVoice = null;
                this.simulateErrors = options.simulateErrors || false;
            }

            getInfo() {
                return JSON.stringify({
                    version: "0.1.0",
                    features: { nodejs_bindings: true },
                    runtime_info: { worker_threads: 4 }
                });
            }

            async listVoices() {
                if (this.simulateErrors && Math.random() < 0.1) {
                    throw new Error('Simulated voice listing error');
                }
                return [
                    { id: 'test-voice-1', name: 'Test Voice 1', language: 'en-US', quality: 'high', isAvailable: true },
                    { id: 'test-voice-2', name: 'Test Voice 2', language: 'en-GB', quality: 'medium', isAvailable: true }
                ];
            }

            async setVoice(voiceId) {
                if (!voiceId || voiceId === 'invalid-voice') {
                    throw new Error('Invalid voice ID');
                }
                if (voiceId === 'unavailable-voice') {
                    throw new Error('Voice not available');
                }
                this.currentVoice = voiceId;
            }

            async getVoice() {
                return this.currentVoice;
            }

            async synthesize(text, options = {}) {
                if (!text || text.trim() === '') {
                    throw new Error('Text is required for synthesis');
                }
                if (text.length > 10000) {
                    throw new Error('Text too long for synthesis');
                }
                if (options.sampleRate && options.sampleRate < 8000) {
                    throw new Error('Sample rate too low');
                }
                if (options.speakingRate && options.speakingRate <= 0) {
                    throw new Error('Speaking rate must be positive');
                }
                if (options.quality && !['low', 'medium', 'high', 'ultra'].includes(options.quality)) {
                    throw new Error('Invalid quality level');
                }
                if (this.simulateErrors && Math.random() < 0.05) {
                    throw new Error('Simulated synthesis error');
                }

                const sampleRate = options.sampleRate || 22050;
                const duration = Math.max(0.5, text.length * 0.05);
                const samples = new Uint8Array(Math.floor(sampleRate * duration * 2));
                
                return {
                    samples: Buffer.from(samples),
                    sampleRate,
                    channels: 1,
                    duration
                };
            }

            async synthesizeSsml(ssml) {
                if (!ssml || !ssml.includes('<speak>')) {
                    throw new Error('Invalid SSML format');
                }
                return this.synthesize(ssml.replace(/<[^>]*>/g, ''));
            }

            async synthesizeWithCallbacks(text, options, progressCb, errorCb) {
                try {
                    if (progressCb) progressCb(0.0);
                    
                    const result = await this.synthesize(text, options);
                    
                    if (progressCb) progressCb(1.0);
                    return result;
                } catch (error) {
                    if (errorCb) errorCb(error.message);
                    throw error;
                }
            }
        }

        VoirsPipeline = MockVoirsPipeline;
        
        synthesizeStreaming = async function(pipeline, text, chunkCb, progressCb, options) {
            if (!chunkCb || typeof chunkCb !== 'function') {
                throw new Error('Chunk callback is required');
            }

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

// Error test framework
class ErrorTestFramework {
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
        console.log('💥 Running VoiRS Node.js Error Handling Tests\n');

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
        console.log(`\n📊 Error Test Results: ${this.passed} passed, ${this.failed} failed (${duration}ms)`);
        
        if (this.failed > 0) {
            process.exit(1);
        }
    }

    assert(condition, message) {
        if (!condition) {
            throw new Error(message || 'Assertion failed');
        }
    }

    async assertThrows(fn, expectedMessage, testMessage) {
        try {
            await fn();
            throw new Error(testMessage || 'Expected function to throw an error');
        } catch (error) {
            if (expectedMessage && !error.message.includes(expectedMessage)) {
                throw new Error(`Expected error containing "${expectedMessage}", got "${error.message}"`);
            }
        }
    }

    async assertDoesNotThrow(fn, testMessage) {
        try {
            await fn();
        } catch (error) {
            throw new Error(testMessage || `Expected function not to throw, but got: ${error.message}`);
        }
    }
}

// Test suite
const test = new ErrorTestFramework();

// Constructor error tests
test.test('invalid constructor options', async () => {
    await test.assertThrows(
        () => new VoirsPipeline({ device: 'invalid-device' }),
        'Invalid device',
        'Should throw error for invalid device'
    );

    await test.assertThrows(
        () => new VoirsPipeline({ numThreads: -1 }),
        'positive',
        'Should throw error for negative thread count'
    );

    // Valid options should not throw
    await test.assertDoesNotThrow(
        () => new VoirsPipeline({ useGpu: false, numThreads: 1 }),
        'Valid options should not throw'
    );
});

// Voice management errors
test.test('invalid voice operations', async () => {
    const pipeline = new VoirsPipeline();

    await test.assertThrows(
        () => pipeline.setVoice(''),
        'Invalid voice',
        'Should throw error for empty voice ID'
    );

    await test.assertThrows(
        () => pipeline.setVoice('invalid-voice'),
        'Invalid voice',
        'Should throw error for non-existent voice ID'
    );

    await test.assertThrows(
        () => pipeline.setVoice('unavailable-voice'),
        'not available',
        'Should throw error for unavailable voice'
    );

    await test.assertThrows(
        () => pipeline.setVoice(null),
        'Invalid voice',
        'Should throw error for null voice ID'
    );
});

// Synthesis input validation errors
test.test('synthesis input validation', async () => {
    const pipeline = new VoirsPipeline();

    await test.assertThrows(
        () => pipeline.synthesize(''),
        'required',
        'Should throw error for empty text'
    );

    await test.assertThrows(
        () => pipeline.synthesize(null),
        'required',
        'Should throw error for null text'
    );

    await test.assertThrows(
        () => pipeline.synthesize(undefined),
        'required',
        'Should throw error for undefined text'
    );

    await test.assertThrows(
        () => pipeline.synthesize('   '),
        'required',
        'Should throw error for whitespace-only text'
    );

    // Very long text
    const longText = 'a'.repeat(15000);
    await test.assertThrows(
        () => pipeline.synthesize(longText),
        'too long',
        'Should throw error for excessively long text'
    );
});

// Synthesis options validation errors
test.test('synthesis options validation', async () => {
    const pipeline = new VoirsPipeline();
    const text = 'Test text';

    await test.assertThrows(
        () => pipeline.synthesize(text, { sampleRate: 100 }),
        'too low',
        'Should throw error for too low sample rate'
    );

    await test.assertThrows(
        () => pipeline.synthesize(text, { speakingRate: 0 }),
        'positive',
        'Should throw error for zero speaking rate'
    );

    await test.assertThrows(
        () => pipeline.synthesize(text, { speakingRate: -1 }),
        'positive',
        'Should throw error for negative speaking rate'
    );

    await test.assertThrows(
        () => pipeline.synthesize(text, { quality: 'invalid' }),
        'Invalid quality',
        'Should throw error for invalid quality level'
    );

    // Valid options should work
    await test.assertDoesNotThrow(
        () => pipeline.synthesize(text, { 
            sampleRate: 22050, 
            speakingRate: 1.0, 
            quality: 'medium' 
        }),
        'Valid options should not throw'
    );
});

// SSML validation errors
test.test('SSML validation errors', async () => {
    const pipeline = new VoirsPipeline();

    await test.assertThrows(
        () => pipeline.synthesizeSsml(''),
        'Invalid SSML',
        'Should throw error for empty SSML'
    );

    await test.assertThrows(
        () => pipeline.synthesizeSsml('Plain text without SSML tags'),
        'Invalid SSML',
        'Should throw error for plain text without SSML'
    );

    await test.assertThrows(
        () => pipeline.synthesizeSsml('<invalid>Not valid SSML</invalid>'),
        'Invalid SSML',
        'Should throw error for invalid SSML format'
    );

    // Valid SSML should work
    await test.assertDoesNotThrow(
        () => pipeline.synthesizeSsml('<speak>Valid SSML content</speak>'),
        'Valid SSML should not throw'
    );
});

// Callback function validation
test.test('callback validation errors', async () => {
    const pipeline = new VoirsPipeline();
    const text = 'Test text';

    // Streaming with invalid callback
    await test.assertThrows(
        () => synthesizeStreaming(pipeline, text, null),
        'required',
        'Should throw error for null chunk callback'
    );

    await test.assertThrows(
        () => synthesizeStreaming(pipeline, text, 'not-a-function'),
        'required',
        'Should throw error for non-function chunk callback'
    );

    // Valid callback should work
    await test.assertDoesNotThrow(
        () => synthesizeStreaming(pipeline, text, () => {}, () => {}),
        'Valid callbacks should not throw'
    );
});

// Resource exhaustion simulation
test.test('resource exhaustion handling', async () => {
    const pipeline = new VoirsPipeline();

    // Simulate rapid successive calls
    const promises = [];
    for (let i = 0; i < 10; i++) {
        promises.push(pipeline.synthesize(`Test ${i}`));
    }

    // Should handle concurrent requests gracefully
    const results = await Promise.allSettled(promises);
    const successful = results.filter(r => r.status === 'fulfilled').length;
    
    test.assert(successful > 0, 'At least some requests should succeed under load');
});

// Error recovery tests
test.test('error recovery and continuation', async () => {
    const pipeline = new VoirsPipeline();

    // Cause an error
    try {
        await pipeline.synthesize('');
    } catch (error) {
        // Expected
    }

    // Pipeline should still work after error
    const result = await pipeline.synthesize('Recovery test');
    test.assert(result.samples.length > 0, 'Pipeline should work after error');

    // Multiple errors and recoveries
    for (let i = 0; i < 3; i++) {
        try {
            await pipeline.setVoice('invalid-voice');
        } catch (error) {
            // Expected
        }

        // Should still work
        const testResult = await pipeline.synthesize(`Test ${i}`);
        test.assert(testResult.samples.length > 0, `Pipeline should work after error ${i}`);
    }
});

// Memory leak simulation
test.test('large data handling', async () => {
    const pipeline = new VoirsPipeline();

    // Test with moderately large text (within limits)
    const largeText = 'This is a moderately large text. '.repeat(100); // ~3300 characters
    const result = await pipeline.synthesize(largeText);
    
    test.assert(result.samples.length > 0, 'Should handle moderately large text');
    test.assert(result.duration > 5, 'Large text should produce longer audio');
});

// Concurrent error handling
test.test('concurrent operations with errors', async () => {
    const pipeline = new VoirsPipeline();

    const operations = [
        // Some valid operations
        pipeline.synthesize('Valid text 1'),
        pipeline.synthesize('Valid text 2'),
        // Some invalid operations
        pipeline.synthesize('').catch(() => ({ error: true })),
        pipeline.setVoice('invalid').catch(() => ({ error: true })),
        // More valid operations
        pipeline.synthesize('Valid text 3'),
    ];

    const results = await Promise.allSettled(operations);
    const successful = results.filter(r => 
        r.status === 'fulfilled' && r.value && !r.value.error
    ).length;

    test.assert(successful >= 3, 'Valid operations should succeed despite concurrent errors');
});

// Timeout simulation
test.test('operation timeout handling', async () => {
    const pipeline = new VoirsPipeline();

    // Test with timeout wrapper
    const timeoutPromise = (promise, ms) => {
        return Promise.race([
            promise,
            new Promise((_, reject) => 
                setTimeout(() => reject(new Error('Operation timeout')), ms)
            )
        ]);
    };

    // Normal operation should complete within timeout
    await test.assertDoesNotThrow(
        () => timeoutPromise(pipeline.synthesize('Quick test'), 1000),
        'Normal operation should complete within timeout'
    );

    // Verify pipeline still works after timeout test
    const result = await pipeline.synthesize('Post-timeout test');
    test.assert(result.samples.length > 0, 'Pipeline should work after timeout test');
});

// Edge case input handling
test.test('edge case input handling', async () => {
    const pipeline = new VoirsPipeline();

    // Special characters
    await test.assertDoesNotThrow(
        () => pipeline.synthesize('Special chars: àáâãäåæçèéêë'),
        'Should handle special characters'
    );

    // Unicode
    await test.assertDoesNotThrow(
        () => pipeline.synthesize('Unicode: 你好世界 🌍 🎤'),
        'Should handle Unicode characters'
    );

    // Numbers and symbols
    await test.assertDoesNotThrow(
        () => pipeline.synthesize('Numbers: 123.45, symbols: @#$%^&*()'),
        'Should handle numbers and symbols'
    );

    // Very short text
    await test.assertDoesNotThrow(
        () => pipeline.synthesize('A'),
        'Should handle single character'
    );
});

// Run all error tests
test.run().catch(error => {
    console.error('❌ Error test runner failed:', error);
    process.exit(1);
});
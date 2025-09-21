#!/usr/bin/env node

/**
 * Comprehensive Performance Benchmarking Suite for VoiRS Node.js Bindings
 * 
 * This suite provides extensive performance testing, profiling, and benchmarking
 * capabilities for the VoiRS Node.js FFI bindings across different scenarios.
 */

const path = require('path');
const fs = require('fs');
const os = require('os');
const { performance } = require('perf_hooks');

// Try to load the VoiRS bindings
let VoirsPipeline, synthesizeStreaming, utils;
const bindingsPath = path.resolve(__dirname, '../../index.js');

try {
    if (fs.existsSync(bindingsPath)) {
        const bindings = require(bindingsPath);
        VoirsPipeline = bindings.VoirsPipeline;
        synthesizeStreaming = bindings.synthesizeStreaming;
        utils = bindings.utils;
    } else {
        console.log('⚠️  Bindings not found, using mock implementations for performance testing');
        
        // Enhanced mock implementations for performance testing
        class MockVoirsPipeline {
            constructor(options = {}) {
                this.options = options;
                this.currentVoice = null;
                this.synthCount = 0;
                this.totalProcessingTime = 0;
                this.memoryUsage = 0;
            }

            async synthesize(text, options = {}) {
                const startTime = performance.now();
                
                // Simulate processing time based on text length and quality
                const baseTime = Math.max(10, text.length * 0.5);
                const qualityMultiplier = {
                    'low': 0.5,
                    'medium': 1.0,
                    'high': 2.0,
                    'ultra': 4.0
                }[options.quality] || 1.0;
                
                const processingTime = baseTime * qualityMultiplier;
                await new Promise(resolve => setTimeout(resolve, processingTime));
                
                const sampleRate = options.sampleRate || 22050;
                const duration = Math.max(0.5, text.length * 0.05);
                const samples = new Uint8Array(Math.floor(sampleRate * duration * 2));
                
                const processingTimeMs = performance.now() - startTime;
                this.synthCount++;
                this.totalProcessingTime += processingTimeMs;
                this.memoryUsage += samples.length;
                
                return {
                    samples: Buffer.from(samples),
                    sampleRate,
                    channels: 1,
                    duration
                };
            }

            async synthesizeWithMetrics(text, options = {}) {
                const startTime = performance.now();
                const startMemory = this.memoryUsage;
                
                const audio = await this.synthesize(text, options);
                
                const processingTimeMs = performance.now() - startTime;
                const audioDurationMs = audio.duration * 1000;
                const realTimeFactor = processingTimeMs / audioDurationMs;
                const memoryUsageMb = (this.memoryUsage - startMemory) / 1024 / 1024;
                
                return {
                    audio,
                    metrics: {
                        processingTimeMs,
                        audioDurationMs,
                        realTimeFactor,
                        memoryUsageMb,
                        cacheHitRate: Math.random() * 0.3 + 0.7 // Simulate cache hit rate
                    }
                };
            }

            async batchSynthesize(texts, options = {}, progressCallback = null) {
                const results = [];
                
                for (let i = 0; i < texts.length; i++) {
                    if (progressCallback) {
                        progressCallback(i, texts.length, i / texts.length);
                    }
                    
                    const result = await this.synthesizeWithMetrics(texts[i], options);
                    results.push(result);
                }
                
                if (progressCallback) {
                    progressCallback(texts.length, texts.length, 1.0);
                }
                
                return results;
            }

            async analyzeAudio(audioBuffer) {
                // Simulate audio analysis
                await new Promise(resolve => setTimeout(resolve, 5));
                
                return {
                    durationSeconds: audioBuffer.duration,
                    sampleRate: audioBuffer.sampleRate,
                    channels: audioBuffer.channels,
                    rmsEnergy: Math.random() * 0.5 + 0.1,
                    zeroCrossingRate: Math.random() * 0.1 + 0.05,
                    spectralCentroid: Math.random() * 2000 + 1000,
                    silenceRegions: []
                };
            }

            async getPerformanceInfo() {
                return {
                    cpuCores: os.cpus().length,
                    memoryUsageMb: this.memoryUsage / 1024 / 1024,
                    gpuAvailable: false,
                    cacheSizeMb: 512,
                    threadCount: this.options.numThreads || 4
                };
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

        utils = {
            validateSynthesisOptions: (options) => true,
            getSupportedFormats: () => ['wav', 'flac', 'mp3', 'opus', 'ogg'],
            getSupportedQualities: () => ['low', 'medium', 'high', 'ultra'],
            resampleAudio: async (audioBuffer, targetRate) => ({
                ...audioBuffer,
                sampleRate: targetRate
            }),
            mixAudio: async (audio1, audio2, ratio = 0.5) => audio1
        };
    }
} catch (error) {
    console.error('Failed to load bindings:', error);
    process.exit(1);
}

class PerformanceBenchmarkSuite {
    constructor() {
        this.results = {};
        this.systemInfo = this.getSystemInfo();
        this.startTime = Date.now();
    }

    getSystemInfo() {
        return {
            platform: os.platform(),
            arch: os.arch(),
            nodeVersion: process.version,
            cpuModel: os.cpus()[0].model,
            cpuCores: os.cpus().length,
            totalMemory: Math.round(os.totalmem() / 1024 / 1024 / 1024 * 100) / 100, // GB
            freeMemory: Math.round(os.freemem() / 1024 / 1024 / 1024 * 100) / 100, // GB
            loadAverage: os.loadavg(),
            uptime: os.uptime()
        };
    }

    async measureMemoryUsage() {
        const memUsage = process.memoryUsage();
        return {
            rss: memUsage.rss / 1024 / 1024, // MB
            heapUsed: memUsage.heapUsed / 1024 / 1024, // MB
            heapTotal: memUsage.heapTotal / 1024 / 1024, // MB
            external: memUsage.external / 1024 / 1024, // MB
            arrayBuffers: memUsage.arrayBuffers / 1024 / 1024 // MB
        };
    }

    async measureCpuUsage() {
        const startUsage = process.cpuUsage();
        await new Promise(resolve => setTimeout(resolve, 100));
        const endUsage = process.cpuUsage(startUsage);
        
        return {
            user: endUsage.user / 1000, // ms
            system: endUsage.system / 1000, // ms
            total: (endUsage.user + endUsage.system) / 1000 // ms
        };
    }

    async benchmarkBasicSynthesis() {
        console.log('📊 Benchmarking Basic Synthesis...');
        
        const pipeline = new VoirsPipeline();
        const testTexts = [
            "Hello world",
            "This is a medium length test sentence for benchmarking.",
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.",
            "A" * 1000 // Very long text
        ];
        
        const results = [];
        
        for (const text of testTexts) {
            const iterations = 10;
            const times = [];
            let totalMemoryDelta = 0;
            
            for (let i = 0; i < iterations; i++) {
                const startMemory = await this.measureMemoryUsage();
                const startTime = performance.now();
                
                await pipeline.synthesize(text);
                
                const endTime = performance.now();
                const endMemory = await this.measureMemoryUsage();
                
                times.push(endTime - startTime);
                totalMemoryDelta += endMemory.heapUsed - startMemory.heapUsed;
            }
            
            const avgTime = times.reduce((a, b) => a + b, 0) / times.length;
            const minTime = Math.min(...times);
            const maxTime = Math.max(...times);
            const stdDev = Math.sqrt(times.reduce((sq, n) => sq + Math.pow(n - avgTime, 2), 0) / times.length);
            
            results.push({
                textLength: text.length,
                iterations,
                avgTimeMs: avgTime,
                minTimeMs: minTime,
                maxTimeMs: maxTime,
                stdDevMs: stdDev,
                avgMemoryDeltaMb: totalMemoryDelta / iterations,
                throughputCharsPerSec: text.length / (avgTime / 1000)
            });
        }
        
        return results;
    }

    async benchmarkSynthesisWithMetrics() {
        console.log('📊 Benchmarking Synthesis with Metrics...');
        
        const pipeline = new VoirsPipeline();
        const testText = "This is a test sentence for metrics benchmarking.";
        const iterations = 20;
        
        const results = [];
        
        for (let i = 0; i < iterations; i++) {
            const startTime = performance.now();
            const result = await pipeline.synthesizeWithMetrics(testText);
            const endTime = performance.now();
            
            results.push({
                iteration: i + 1,
                totalTimeMs: endTime - startTime,
                processingTimeMs: result.metrics.processingTimeMs,
                realTimeFactor: result.metrics.realTimeFactor,
                memoryUsageMb: result.metrics.memoryUsageMb,
                cacheHitRate: result.metrics.cacheHitRate,
                audioDurationMs: result.metrics.audioDurationMs
            });
        }
        
        // Calculate statistics
        const avgRealTimeFactor = results.reduce((sum, r) => sum + r.realTimeFactor, 0) / results.length;
        const avgMemoryUsage = results.reduce((sum, r) => sum + r.memoryUsageMb, 0) / results.length;
        const avgCacheHitRate = results.reduce((sum, r) => sum + r.cacheHitRate, 0) / results.length;
        
        return {
            iterations,
            results,
            statistics: {
                avgRealTimeFactor,
                avgMemoryUsage,
                avgCacheHitRate,
                bestRealTimeFactor: Math.min(...results.map(r => r.realTimeFactor)),
                worstRealTimeFactor: Math.max(...results.map(r => r.realTimeFactor))
            }
        };
    }

    async benchmarkBatchSynthesis() {
        console.log('📊 Benchmarking Batch Synthesis...');
        
        const pipeline = new VoirsPipeline();
        const batchSizes = [1, 5, 10, 20, 50];
        const testTexts = [
            "First test sentence for batch processing.",
            "Second test sentence with different content.",
            "Third test sentence for comprehensive testing.",
            "Fourth test sentence with varied length and complexity.",
            "Fifth test sentence to complete the batch."
        ];
        
        const results = [];
        
        for (const batchSize of batchSizes) {
            const texts = [];
            for (let i = 0; i < batchSize; i++) {
                texts.push(testTexts[i % testTexts.length]);
            }
            
            const startTime = performance.now();
            const startMemory = await this.measureMemoryUsage();
            
            const batchResults = await pipeline.batchSynthesize(texts);
            
            const endTime = performance.now();
            const endMemory = await this.measureMemoryUsage();
            
            const totalTime = endTime - startTime;
            const memoryDelta = endMemory.heapUsed - startMemory.heapUsed;
            const avgRealTimeFactor = batchResults.reduce((sum, r) => sum + r.metrics.realTimeFactor, 0) / batchResults.length;
            const totalAudioDuration = batchResults.reduce((sum, r) => sum + r.metrics.audioDurationMs, 0);
            
            results.push({
                batchSize,
                totalTimeMs: totalTime,
                avgTimePerItemMs: totalTime / batchSize,
                memoryDeltaMb: memoryDelta,
                avgRealTimeFactor,
                totalAudioDurationMs: totalAudioDuration,
                throughputItemsPerSec: batchSize / (totalTime / 1000)
            });
        }
        
        return results;
    }

    async benchmarkStreamingSynthesis() {
        console.log('📊 Benchmarking Streaming Synthesis...');
        
        const pipeline = new VoirsPipeline();
        const testTexts = [
            "Short text for streaming test.",
            "Medium length text for streaming synthesis evaluation and performance testing.",
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur."
        ];
        
        const results = [];
        
        for (const text of testTexts) {
            const chunks = [];
            const progressUpdates = [];
            let firstChunkTime = null;
            let lastChunkTime = null;
            
            const startTime = performance.now();
            
            await synthesizeStreaming(
                pipeline,
                text,
                (chunk) => {
                    const now = performance.now();
                    if (firstChunkTime === null) firstChunkTime = now;
                    lastChunkTime = now;
                    chunks.push({
                        timestamp: now,
                        size: chunk.length
                    });
                },
                (progress) => {
                    progressUpdates.push({
                        timestamp: performance.now(),
                        progress
                    });
                },
                { quality: 'medium' }
            );
            
            const endTime = performance.now();
            const totalTime = endTime - startTime;
            const timeToFirstChunk = firstChunkTime - startTime;
            const streamingDuration = lastChunkTime - firstChunkTime;
            
            results.push({
                textLength: text.length,
                totalTimeMs: totalTime,
                timeToFirstChunkMs: timeToFirstChunk,
                streamingDurationMs: streamingDuration,
                chunkCount: chunks.length,
                avgChunkSize: chunks.reduce((sum, c) => sum + c.size, 0) / chunks.length,
                progressUpdateCount: progressUpdates.length,
                chunksPerSecond: chunks.length / (streamingDuration / 1000),
                bytesPerSecond: chunks.reduce((sum, c) => sum + c.size, 0) / (streamingDuration / 1000)
            });
        }
        
        return results;
    }

    async benchmarkAudioAnalysis() {
        console.log('📊 Benchmarking Audio Analysis...');
        
        const pipeline = new VoirsPipeline();
        const testTexts = [
            "Short audio for analysis.",
            "Medium length audio for analysis testing.",
            "Long audio content for comprehensive analysis benchmarking and performance evaluation."
        ];
        
        const results = [];
        
        for (const text of testTexts) {
            const iterations = 5;
            const analysisResults = [];
            
            // First synthesize the audio
            const audioResult = await pipeline.synthesize(text);
            
            for (let i = 0; i < iterations; i++) {
                const startTime = performance.now();
                const analysis = await pipeline.analyzeAudio(audioResult);
                const endTime = performance.now();
                
                analysisResults.push({
                    timeMs: endTime - startTime,
                    rmsEnergy: analysis.rmsEnergy,
                    zeroCrossingRate: analysis.zeroCrossingRate,
                    spectralCentroid: analysis.spectralCentroid
                });
            }
            
            const avgTime = analysisResults.reduce((sum, r) => sum + r.timeMs, 0) / iterations;
            const avgRmsEnergy = analysisResults.reduce((sum, r) => sum + r.rmsEnergy, 0) / iterations;
            const avgZeroCrossingRate = analysisResults.reduce((sum, r) => sum + r.zeroCrossingRate, 0) / iterations;
            const avgSpectralCentroid = analysisResults.reduce((sum, r) => sum + r.spectralCentroid, 0) / iterations;
            
            results.push({
                textLength: text.length,
                audioDurationMs: audioResult.duration * 1000,
                iterations,
                avgAnalysisTimeMs: avgTime,
                realTimeFactorAnalysis: avgTime / (audioResult.duration * 1000),
                avgRmsEnergy,
                avgZeroCrossingRate,
                avgSpectralCentroid,
                analysisSpeedRatio: (audioResult.duration * 1000) / avgTime
            });
        }
        
        return results;
    }

    async benchmarkConcurrentSynthesis() {
        console.log('📊 Benchmarking Concurrent Synthesis...');
        
        const concurrencyLevels = [1, 2, 4, 8, 16];
        const testText = "This is a test sentence for concurrent synthesis benchmarking.";
        
        const results = [];
        
        for (const concurrency of concurrencyLevels) {
            const pipeline = new VoirsPipeline({ numThreads: concurrency });
            const promises = [];
            
            const startTime = performance.now();
            const startMemory = await this.measureMemoryUsage();
            
            for (let i = 0; i < concurrency; i++) {
                promises.push(pipeline.synthesizeWithMetrics(testText));
            }
            
            const synthesisResults = await Promise.all(promises);
            
            const endTime = performance.now();
            const endMemory = await this.measureMemoryUsage();
            
            const totalTime = endTime - startTime;
            const memoryDelta = endMemory.heapUsed - startMemory.heapUsed;
            const avgRealTimeFactor = synthesisResults.reduce((sum, r) => sum + r.metrics.realTimeFactor, 0) / synthesisResults.length;
            const throughput = concurrency / (totalTime / 1000);
            
            results.push({
                concurrency,
                totalTimeMs: totalTime,
                avgTimePerTaskMs: totalTime / concurrency,
                memoryDeltaMb: memoryDelta,
                avgRealTimeFactor,
                throughputTasksPerSec: throughput,
                efficiency: throughput / concurrency // Should be close to 1 for good scaling
            });
        }
        
        return results;
    }

    async benchmarkMemoryUsage() {
        console.log('📊 Benchmarking Memory Usage...');
        
        const pipeline = new VoirsPipeline();
        const testText = "This is a test sentence for memory usage benchmarking.";
        const iterations = 100;
        
        const memorySnapshots = [];
        
        // Baseline memory
        const baselineMemory = await this.measureMemoryUsage();
        memorySnapshots.push({ iteration: 0, ...baselineMemory });
        
        for (let i = 1; i <= iterations; i++) {
            await pipeline.synthesize(testText);
            
            if (i % 10 === 0) {
                const memoryUsage = await this.measureMemoryUsage();
                memorySnapshots.push({ iteration: i, ...memoryUsage });
            }
        }
        
        // Force garbage collection if available
        if (global.gc) {
            global.gc();
            const postGcMemory = await this.measureMemoryUsage();
            memorySnapshots.push({ iteration: iterations + 1, ...postGcMemory, postGc: true });
        }
        
        // Calculate memory trends
        const memoryGrowth = memorySnapshots[memorySnapshots.length - 1].heapUsed - baselineMemory.heapUsed;
        const memoryGrowthRate = memoryGrowth / iterations;
        
        return {
            iterations,
            baselineMemoryMb: baselineMemory,
            memorySnapshots,
            memoryGrowthMb: memoryGrowth,
            memoryGrowthRateMbPerIteration: memoryGrowthRate,
            potentialMemoryLeak: memoryGrowthRate > 0.1 // MB per iteration
        };
    }

    async runComprehensiveBenchmarks() {
        console.log('🚀 Starting Comprehensive Performance Benchmarking Suite');
        console.log('=' * 60);
        console.log('System Information:');
        console.log(`  Platform: ${this.systemInfo.platform} (${this.systemInfo.arch})`);
        console.log(`  Node.js: ${this.systemInfo.nodeVersion}`);
        console.log(`  CPU: ${this.systemInfo.cpuModel} (${this.systemInfo.cpuCores} cores)`);
        console.log(`  Memory: ${this.systemInfo.totalMemory} GB total, ${this.systemInfo.freeMemory} GB free`);
        console.log(`  Load Average: ${this.systemInfo.loadAverage.map(l => l.toFixed(2)).join(', ')}`);
        console.log('=' * 60);
        
        const benchmarks = [
            { name: 'Basic Synthesis', fn: () => this.benchmarkBasicSynthesis() },
            { name: 'Synthesis with Metrics', fn: () => this.benchmarkSynthesisWithMetrics() },
            { name: 'Batch Synthesis', fn: () => this.benchmarkBatchSynthesis() },
            { name: 'Streaming Synthesis', fn: () => this.benchmarkStreamingSynthesis() },
            { name: 'Audio Analysis', fn: () => this.benchmarkAudioAnalysis() },
            { name: 'Concurrent Synthesis', fn: () => this.benchmarkConcurrentSynthesis() },
            { name: 'Memory Usage', fn: () => this.benchmarkMemoryUsage() }
        ];
        
        for (const benchmark of benchmarks) {
            try {
                const startTime = performance.now();
                this.results[benchmark.name] = await benchmark.fn();
                const endTime = performance.now();
                
                console.log(`✅ ${benchmark.name} completed in ${(endTime - startTime).toFixed(2)}ms`);
            } catch (error) {
                console.error(`❌ ${benchmark.name} failed:`, error.message);
                this.results[benchmark.name] = { error: error.message };
            }
        }
        
        return this.generateReport();
    }

    generateReport() {
        const report = {
            timestamp: new Date().toISOString(),
            systemInfo: this.systemInfo,
            testDuration: Date.now() - this.startTime,
            results: this.results,
            summary: this.generateSummary()
        };
        
        return report;
    }

    generateSummary() {
        const summary = {
            totalTests: Object.keys(this.results).length,
            passedTests: Object.values(this.results).filter(r => !r.error).length,
            failedTests: Object.values(this.results).filter(r => r.error).length,
            keyMetrics: {}
        };
        
        // Extract key metrics
        if (this.results['Basic Synthesis'] && !this.results['Basic Synthesis'].error) {
            const basicResults = this.results['Basic Synthesis'];
            summary.keyMetrics.avgSynthesisTime = basicResults.reduce((sum, r) => sum + r.avgTimeMs, 0) / basicResults.length;
            summary.keyMetrics.avgThroughput = basicResults.reduce((sum, r) => sum + r.throughputCharsPerSec, 0) / basicResults.length;
        }
        
        if (this.results['Synthesis with Metrics'] && !this.results['Synthesis with Metrics'].error) {
            const metricsResults = this.results['Synthesis with Metrics'];
            summary.keyMetrics.avgRealTimeFactor = metricsResults.statistics.avgRealTimeFactor;
            summary.keyMetrics.avgMemoryUsage = metricsResults.statistics.avgMemoryUsage;
            summary.keyMetrics.avgCacheHitRate = metricsResults.statistics.avgCacheHitRate;
        }
        
        if (this.results['Memory Usage'] && !this.results['Memory Usage'].error) {
            const memoryResults = this.results['Memory Usage'];
            summary.keyMetrics.memoryGrowthRate = memoryResults.memoryGrowthRateMbPerIteration;
            summary.keyMetrics.potentialMemoryLeak = memoryResults.potentialMemoryLeak;
        }
        
        return summary;
    }
}

async function main() {
    const suite = new PerformanceBenchmarkSuite();
    
    try {
        const report = await suite.runComprehensiveBenchmarks();
        
        // Generate detailed report
        console.log('\n📊 Performance Benchmark Report');
        console.log('=' * 60);
        console.log(`Test Duration: ${report.testDuration / 1000}s`);
        console.log(`Tests Passed: ${report.summary.passedTests}/${report.summary.totalTests}`);
        
        if (report.summary.keyMetrics.avgSynthesisTime) {
            console.log(`Average Synthesis Time: ${report.summary.keyMetrics.avgSynthesisTime.toFixed(2)}ms`);
        }
        
        if (report.summary.keyMetrics.avgThroughput) {
            console.log(`Average Throughput: ${report.summary.keyMetrics.avgThroughput.toFixed(2)} chars/sec`);
        }
        
        if (report.summary.keyMetrics.avgRealTimeFactor) {
            console.log(`Average Real-time Factor: ${report.summary.keyMetrics.avgRealTimeFactor.toFixed(3)}`);
        }
        
        if (report.summary.keyMetrics.memoryGrowthRate) {
            console.log(`Memory Growth Rate: ${report.summary.keyMetrics.memoryGrowthRate.toFixed(4)} MB/iteration`);
        }
        
        if (report.summary.keyMetrics.potentialMemoryLeak) {
            console.log('⚠️  Potential memory leak detected!');
        }
        
        // Save detailed report
        const reportPath = path.join(__dirname, 'performance_report.json');
        fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
        console.log(`\nDetailed report saved to: ${reportPath}`);
        
        // Exit with appropriate code
        if (report.summary.failedTests === 0) {
            console.log('\n✅ All performance benchmarks completed successfully!');
            process.exit(0);
        } else {
            console.log(`\n❌ ${report.summary.failedTests} benchmark(s) failed.`);
            process.exit(1);
        }
        
    } catch (error) {
        console.error('❌ Benchmark suite failed:', error);
        process.exit(1);
    }
}

if (require.main === module) {
    main();
}

module.exports = { PerformanceBenchmarkSuite };
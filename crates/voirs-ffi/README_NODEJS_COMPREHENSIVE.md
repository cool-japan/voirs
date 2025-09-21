# VoiRS Node.js Bindings - Comprehensive Guide

## Overview

The VoiRS Node.js bindings provide a complete, production-ready interface for integrating high-quality speech synthesis, recognition, and audio analysis capabilities into Node.js applications. Built using NAPI-RS for optimal performance and safety.

## Features

### 🎯 Core Synthesis Features
- **High-quality text-to-speech synthesis** with multiple voice options
- **SSML support** for advanced speech markup
- **Real-time streaming synthesis** with chunked audio delivery
- **Batch processing** for multiple texts with progress tracking
- **Comprehensive audio format support** (WAV, FLAC, MP3, Opus, OGG)

### 📊 Performance & Monitoring
- **Detailed synthesis metrics** including real-time factor, memory usage, and processing time
- **Performance profiling** with comprehensive benchmarking tools
- **Memory usage tracking** and leak detection
- **CPU utilization monitoring** for optimization

### 🔧 Audio Processing
- **Audio analysis** with spectral analysis, energy detection, and silence detection
- **Audio resampling** for different sample rates
- **Audio mixing** capabilities for multi-source audio
- **Format conversion** and validation utilities

### 🎤 Recognition & Analysis (Optional)
- **Speech recognition** using Whisper and other ASR models
- **Audio quality analysis** with comprehensive metrics
- **Phoneme alignment** for pronunciation assessment
- **Language detection** capabilities

### 🔍 Quality Assurance
- **Cross-language consistency testing** with C and Python bindings
- **Comprehensive test suite** covering all functionality
- **Performance benchmarking** for optimization
- **Error handling** with structured error information

## Installation

### Prerequisites
- Node.js 16.0 or higher
- Rust 1.70 or higher (for building from source)

### From NPM (Recommended)
```bash
npm install voirs-ffi
```

### From Source
```bash
git clone https://github.com/cool-japan/voirs.git
cd voirs/crates/voirs-ffi
npm install
npm run build
```

## Quick Start

### Basic Text-to-Speech
```javascript
const { VoirsPipeline } = require('voirs-ffi');

async function basicSynthesis() {
    // Create a pipeline with default settings
    const pipeline = new VoirsPipeline();
    
    // Synthesize text to audio
    const result = await pipeline.synthesize(
        "Hello! This is VoiRS speech synthesis."
    );
    
    console.log(`Generated ${result.duration}s of audio`);
    console.log(`Sample rate: ${result.sampleRate}Hz`);
    console.log(`Channels: ${result.channels}`);
    console.log(`Audio data: ${result.samples.length} bytes`);
}

basicSynthesis().catch(console.error);
```

### Advanced Synthesis with Options
```javascript
const { VoirsPipeline } = require('voirs-ffi');

async function advancedSynthesis() {
    // Create pipeline with custom configuration
    const pipeline = new VoirsPipeline({
        useGpu: true,
        numThreads: 4,
        device: 'cuda'
    });
    
    // Advanced synthesis with options
    const result = await pipeline.synthesize(
        "This is advanced speech synthesis with custom settings.",
        {
            speakingRate: 1.2,
            pitchShift: 0.5,
            volumeGain: 2.0,
            outputFormat: 'wav',
            sampleRate: 44100,
            quality: 'ultra'
        }
    );
    
    console.log(`High-quality audio generated: ${result.duration}s`);
}

advancedSynthesis().catch(console.error);
```

### Synthesis with Performance Metrics
```javascript
const { VoirsPipeline } = require('voirs-ffi');

async function synthesisWithMetrics() {
    const pipeline = new VoirsPipeline();
    
    const result = await pipeline.synthesizeWithMetrics(
        "Measuring performance during synthesis."
    );
    
    console.log('Audio:', {
        duration: result.audio.duration,
        sampleRate: result.audio.sampleRate,
        size: result.audio.samples.length
    });
    
    console.log('Metrics:', {
        processingTime: result.metrics.processingTimeMs,
        realTimeFactor: result.metrics.realTimeFactor,
        memoryUsage: result.metrics.memoryUsageMb,
        cacheHitRate: result.metrics.cacheHitRate
    });
}

synthesisWithMetrics().catch(console.error);
```

## Comprehensive Examples

### 1. Batch Processing with Progress Tracking
```javascript
const { VoirsPipeline } = require('voirs-ffi');

async function batchSynthesis() {
    const pipeline = new VoirsPipeline();
    
    const texts = [
        "First text to synthesize.",
        "Second text with different content.",
        "Third text for batch processing.",
        "Fourth text with longer content for testing.",
        "Fifth and final text in the batch."
    ];
    
    const results = await pipeline.batchSynthesize(
        texts,
        { quality: 'high' },
        (current, total, progress) => {
            console.log(`Progress: ${current}/${total} (${(progress * 100).toFixed(1)}%)`);
        }
    );
    
    // Process results
    const totalDuration = results.reduce((sum, r) => sum + r.audio.duration, 0);
    const avgRealTimeFactor = results.reduce((sum, r) => sum + r.metrics.realTimeFactor, 0) / results.length;
    
    console.log(`Batch completed: ${totalDuration}s total audio`);
    console.log(`Average real-time factor: ${avgRealTimeFactor.toFixed(3)}`);
}

batchSynthesis().catch(console.error);
```

### 2. Streaming Synthesis with Real-time Processing
```javascript
const { VoirsPipeline, synthesizeStreaming } = require('voirs-ffi');
const fs = require('fs');

async function streamingSynthesis() {
    const pipeline = new VoirsPipeline();
    
    const longText = `
        This is a long text that will be synthesized in streaming mode.
        The audio will be delivered in real-time chunks as it's being processed.
        This enables low-latency applications and real-time audio playback.
    `;
    
    const audioChunks = [];
    let chunkCount = 0;
    
    await synthesizeStreaming(
        pipeline,
        longText,
        // Chunk callback - called for each audio chunk
        (chunk) => {
            chunkCount++;
            audioChunks.push(chunk);
            console.log(`Received chunk ${chunkCount}: ${chunk.length} bytes`);
            
            // Process chunk in real-time (e.g., stream to audio device)
            // playAudioChunk(chunk);
        },
        // Progress callback - called with synthesis progress
        (progress) => {
            console.log(`Synthesis progress: ${(progress * 100).toFixed(1)}%`);
        },
        // Options
        { quality: 'medium', sampleRate: 22050 }
    );
    
    // Combine all chunks
    const fullAudio = Buffer.concat(audioChunks);
    fs.writeFileSync('streaming_output.wav', fullAudio);
    console.log(`Streaming synthesis complete: ${chunkCount} chunks, ${fullAudio.length} bytes`);
}

streamingSynthesis().catch(console.error);
```

### 3. Audio Analysis and Processing
```javascript
const { VoirsPipeline, utils } = require('voirs-ffi');

async function audioAnalysisExample() {
    const pipeline = new VoirsPipeline();
    
    // Synthesize test audio
    const audioResult = await pipeline.synthesize(
        "This is test audio for analysis and processing."
    );
    
    // Analyze audio characteristics
    const analysis = await pipeline.analyzeAudio(audioResult);
    
    console.log('Audio Analysis:', {
        duration: analysis.durationSeconds,
        rmsEnergy: analysis.rmsEnergy,
        zeroCrossingRate: analysis.zeroCrossingRate,
        spectralCentroid: analysis.spectralCentroid,
        silenceRegions: analysis.silenceRegions.length
    });
    
    // Audio processing operations
    console.log('\\nAudio Processing:');
    
    // Resample audio
    const resampledAudio = await utils.resampleAudio(audioResult, 44100);
    console.log(`Resampled to ${resampledAudio.sampleRate}Hz`);
    
    // Convert to WAV format
    const wavBuffer = utils.toWav(audioResult);
    require('fs').writeFileSync('output.wav', wavBuffer);
    console.log(`WAV file saved: ${wavBuffer.length} bytes`);
    
    // Validate synthesis options
    const validOptions = {
        speakingRate: 1.0,
        pitchShift: 0.0,
        quality: 'high'
    };
    
    const isValid = utils.validateSynthesisOptions(validOptions);
    console.log(`Options valid: ${isValid}`);
}

audioAnalysisExample().catch(console.error);
```

### 4. Voice Management and SSML
```javascript
const { VoirsPipeline } = require('voirs-ffi');

async function voiceManagement() {
    const pipeline = new VoirsPipeline();
    
    // List available voices
    const voices = await pipeline.listVoices();
    console.log('Available voices:');
    voices.forEach((voice, index) => {
        console.log(`  ${index + 1}. ${voice.name} (${voice.id})`);
        console.log(`     Language: ${voice.language}`);
        console.log(`     Quality: ${voice.quality}`);
        console.log(`     Available: ${voice.isAvailable}`);
    });
    
    // Select and use a specific voice
    if (voices.length > 0) {
        await pipeline.setVoice(voices[0].id);
        console.log(`\\nSelected voice: ${voices[0].name}`);
        
        // Verify voice selection
        const currentVoice = await pipeline.getVoice();
        console.log(`Current voice: ${currentVoice}`);
    }
    
    // SSML synthesis with voice and prosody control
    const ssmlContent = `
        <speak>
            <p>This is an example of <emphasis level="strong">SSML synthesis</emphasis>.</p>
            <break time="500ms"/>
            <p>
                I can speak at <prosody rate="slow">different speeds</prosody>,
                with <prosody pitch="high">different pitches</prosody>,
                and <prosody volume="loud">different volumes</prosody>.
            </p>
            <p>
                I can also add <break time="1s"/> dramatic pauses.
            </p>
        </speak>
    `;
    
    const ssmlResult = await pipeline.synthesizeSsml(ssmlContent);
    console.log(`\\nSSML synthesis complete: ${ssmlResult.duration}s`);
}

voiceManagement().catch(console.error);
```

### 5. Error Handling and Performance Monitoring
```javascript
const { VoirsPipeline } = require('voirs-ffi');

async function errorHandlingExample() {
    const pipeline = new VoirsPipeline();
    
    // Synthesis with error handling
    try {
        const result = await pipeline.synthesizeWithCallbacks(
            "This is synthesis with comprehensive error handling.",
            { quality: 'high' },
            // Progress callback
            (progress) => {
                console.log(`Progress: ${(progress * 100).toFixed(1)}%`);
            },
            // Error callback
            (error) => {
                console.error('Synthesis error:', error);
            }
        );
        
        console.log('Synthesis successful:', result.duration);
        
    } catch (error) {
        console.error('Synthesis failed:', error.message);
        
        // Handle specific error types
        if (error.code === 'VOICE_NOT_FOUND') {
            console.log('Suggestion: Check available voices');
        } else if (error.code === 'INVALID_SSML') {
            console.log('Suggestion: Validate SSML syntax');
        }
    }
    
    // Performance monitoring
    const performanceInfo = await pipeline.getPerformanceInfo();
    console.log('\\nPerformance Information:', {
        cpuCores: performanceInfo.cpuCores,
        memoryUsage: performanceInfo.memoryUsageMb,
        gpuAvailable: performanceInfo.gpuAvailable,
        threadCount: performanceInfo.threadCount
    });
}

errorHandlingExample().catch(console.error);
```

### 6. Speech Recognition (Optional Feature)
```javascript
const { ASRModel, AudioAnalyzer } = require('voirs-ffi');

async function speechRecognitionExample() {
    // Create ASR model
    const asrModel = ASRModel.whisper('base');
    
    // Create audio analyzer
    const analyzer = new AudioAnalyzer();
    
    // Synthesize audio for recognition test
    const { VoirsPipeline } = require('voirs-ffi');
    const pipeline = new VoirsPipeline();
    
    const audioResult = await pipeline.synthesize(
        "This is a test of speech recognition capabilities."
    );
    
    // Recognize speech
    const recognition = await asrModel.recognize(audioResult);
    console.log('Recognition Result:', {
        text: recognition.text,
        confidence: recognition.confidence,
        language: recognition.language,
        processingTime: recognition.processingTimeMs
    });
    
    // Analyze audio for recognition quality
    const analysis = await analyzer.analyze(audioResult);
    console.log('Audio Analysis:', {
        duration: analysis.durationSeconds,
        rmsEnergy: analysis.rmsEnergy,
        zeroCrossingRate: analysis.zeroCrossingRate,
        spectralCentroid: analysis.spectralCentroid
    });
    
    // Get supported languages
    const languages = asrModel.supportedLanguages();
    console.log('Supported languages:', languages);
}

speechRecognitionExample().catch(console.error);
```

## API Reference

### VoirsPipeline Class

#### Constructor
```javascript
new VoirsPipeline(options?: PipelineOptions)
```

#### Methods

**Basic Synthesis**
- `synthesize(text: string, options?: SynthesisOptions): Promise<AudioBufferResult>`
- `synthesizeSync(text: string, options?: SynthesisOptions): AudioBufferResult`
- `synthesizeSsml(ssml: string): Promise<AudioBufferResult>`

**Advanced Synthesis**
- `synthesizeWithMetrics(text: string, options?: SynthesisOptions): Promise<SynthesisResult>`
- `synthesizeSsmlWithMetrics(ssml: string): Promise<SynthesisResult>`
- `batchSynthesize(texts: string[], options?: SynthesisOptions, progressCallback?: Function): Promise<SynthesisResult[]>`

**Voice Management**
- `listVoices(): Promise<VoiceInfo[]>`
- `setVoice(voiceId: string): Promise<void>`
- `getVoice(): Promise<string | null>`

**Audio Analysis**
- `analyzeAudio(audioBuffer: AudioBufferResult): Promise<AudioAnalysis>`

**Performance & Monitoring**
- `getPerformanceInfo(): Promise<PerformanceInfo>`
- `getInfo(): string`

### Utility Functions

```javascript
const { utils } = require('voirs-ffi');

// Audio format conversion
utils.toWav(audioBuffer: AudioBufferResult): Buffer

// Audio processing
utils.resampleAudio(audioBuffer: AudioBufferResult, targetSampleRate: number): Promise<AudioBufferResult>
utils.mixAudio(audio1: AudioBufferResult, audio2: AudioBufferResult, mixRatio?: number): Promise<AudioBufferResult>

// Validation
utils.validateSynthesisOptions(options: SynthesisOptions): boolean
utils.getSupportedFormats(): string[]
utils.getSupportedQualities(): string[]
```

### Streaming Function

```javascript
synthesizeStreaming(
    pipeline: VoirsPipeline,
    text: string,
    chunkCallback: (chunk: Buffer) => void,
    progressCallback?: (progress: number) => void,
    options?: SynthesisOptions
): Promise<void>
```

## Performance Optimization

### Best Practices

1. **Pipeline Reuse**: Create one pipeline instance and reuse it
2. **Batch Processing**: Use `batchSynthesize()` for multiple texts
3. **Streaming**: Use streaming synthesis for long texts
4. **Quality Settings**: Choose appropriate quality for your use case
5. **Memory Management**: Monitor memory usage with performance info

### Performance Monitoring

```javascript
// Monitor synthesis performance
const result = await pipeline.synthesizeWithMetrics(text);
console.log('Real-time factor:', result.metrics.realTimeFactor);
console.log('Memory usage:', result.metrics.memoryUsageMb);

// Monitor system performance
const perfInfo = await pipeline.getPerformanceInfo();
console.log('System metrics:', perfInfo);
```

### Benchmarking

Run comprehensive performance benchmarks:

```bash
cd tests/nodejs
node comprehensive-performance-tests.js
```

## Error Handling

### Error Types

The bindings provide structured error information:

```javascript
try {
    await pipeline.synthesize(text);
} catch (error) {
    console.log('Error code:', error.code);
    console.log('Error message:', error.message);
    console.log('Details:', error.details);
    console.log('Suggestion:', error.suggestion);
}
```

### Common Error Codes

- `SYNTHESIS_FAILED`: General synthesis error
- `VOICE_NOT_FOUND`: Requested voice not available
- `INVALID_SSML`: SSML syntax error
- `INVALID_OPTIONS`: Invalid synthesis options
- `MEMORY_ERROR`: Memory allocation error
- `DEVICE_ERROR`: Audio device error

## Testing

### Unit Tests
```bash
cd tests/nodejs
npm test
```

### Integration Tests
```bash
cd tests/nodejs
node integration-tests.js
```

### Performance Tests
```bash
cd tests/nodejs
node performance-tests.js
```

### Cross-Language Consistency Tests
```bash
cd tests/cross_lang
python comprehensive_consistency_test.py
```

## Building from Source

### Prerequisites
- Rust 1.70+
- Node.js 16+
- NAPI-RS CLI

### Build Steps
```bash
# Install dependencies
npm install

# Build the native module
npm run build

# Run tests
npm test

# Build for specific platforms
npm run build -- --target x86_64-unknown-linux-gnu
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

- GitHub Issues: https://github.com/cool-japan/voirs/issues
- Documentation: https://voirs.cool-japan.org/
- Examples: https://github.com/cool-japan/voirs/tree/main/examples
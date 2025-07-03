# VoiRS Node.js Bindings

High-performance Node.js bindings for VoiRS (Voice Recognition & Synthesis) library, built with NAPI-RS for optimal performance and cross-platform compatibility.

## Features

- 🚀 **High Performance**: Native Rust implementation with minimal JavaScript overhead
- 🌐 **Cross-platform**: Works on Windows, macOS, and Linux (x64, ARM64)
- ⚡ **Async/Await Support**: Full async/await support with Promise-based APIs
- 🔄 **Streaming**: Real-time audio streaming with chunked callbacks
- 🎛️ **Configurable**: Extensive configuration options for synthesis parameters
- 🎭 **Multiple Voices**: Support for various voices and languages
- 🏷️ **SSML Support**: Advanced Speech Synthesis Markup Language support
- 📊 **Progress Tracking**: Real-time progress callbacks for long syntheses
- 💾 **Multiple Formats**: Output in WAV, FLAC, MP3, Opus, and OGG formats
- 🧵 **Threading**: Configurable multi-threading for optimal performance

## Installation

```bash
npm install voirs-ffi
```

Or with yarn:
```bash
yarn add voirs-ffi
```

## Quick Start

```javascript
const { VoirsPipeline } = require('voirs-ffi');

async function quickExample() {
    // Create a pipeline with default settings
    const pipeline = new VoirsPipeline();
    
    // Synthesize text to audio
    const result = await pipeline.synthesize("Hello, world!");
    
    // Save to file
    require('fs').writeFileSync('output.wav', result.samples);
    
    console.log(`Generated ${result.duration}s of audio at ${result.sampleRate}Hz`);
}

quickExample();
```

## API Reference

### VoirsPipeline

The main class for text-to-speech synthesis.

#### Constructor

```javascript
const pipeline = new VoirsPipeline(options?)
```

**Options:**
- `useGpu?: boolean` - Enable GPU acceleration (default: false)
- `numThreads?: number` - Number of worker threads (default: auto)
- `cacheDir?: string` - Cache directory path
- `device?: string` - Device type: 'cpu', 'cuda', 'metal', 'vulkan'

#### Methods

##### synthesize(text, options?)

Synthesize text to audio asynchronously.

```javascript
const result = await pipeline.synthesize("Hello world", {
    speakingRate: 1.0,        // 0.5-2.0
    pitchShift: 0.0,          // -12.0 to 12.0 semitones
    volumeGain: 0.0,          // -20.0 to 20.0 dB
    enableEnhancement: true,
    outputFormat: 'wav',      // 'wav', 'flac', 'mp3', 'opus', 'ogg'
    sampleRate: 22050,        // 8000, 16000, 22050, 44100, 48000
    quality: 'high'           // 'low', 'medium', 'high', 'ultra'
});
```

##### synthesizeSync(text, options?)

Synthesize text to audio synchronously (blocks the event loop).

```javascript
const result = pipeline.synthesizeSync("Hello world");
```

##### synthesizeSsml(ssml)

Synthesize SSML content to audio.

```javascript
const result = await pipeline.synthesizeSsml(`
    <speak>
        <p>Hello <emphasis>world</emphasis>!</p>
        <break time="1s"/>
        <p>This is <prosody rate="slow">slow speech</prosody>.</p>
    </speak>
`);
```

##### synthesizeWithCallbacks(text, options?, progressCallback?, errorCallback?)

Synthesize with progress and error callbacks.

```javascript
const result = await pipeline.synthesizeWithCallbacks(
    "Long text to synthesize...",
    { quality: 'high' },
    (progress) => console.log(`Progress: ${progress * 100}%`),
    (error) => console.error('Error:', error)
);
```

##### setVoice(voiceId)

Set the active voice for synthesis.

```javascript
await pipeline.setVoice('en-US-neural-jenny');
```

##### getVoice()

Get the current active voice.

```javascript
const currentVoice = await pipeline.getVoice();
console.log('Current voice:', currentVoice);
```

##### listVoices()

List all available voices.

```javascript
const voices = await pipeline.listVoices();
voices.forEach(voice => {
    console.log(`${voice.name} (${voice.id}) - ${voice.language}`);
});
```

##### getInfo()

Get pipeline and runtime information.

```javascript
const info = pipeline.getInfo();
console.log('Pipeline info:', info);
```

### Streaming Synthesis

For real-time audio streaming:

```javascript
const { synthesizeStreaming } = require('voirs-ffi');

await synthesizeStreaming(
    pipeline,
    "Text to stream...",
    (chunk) => {
        // Handle audio chunk (Buffer)
        console.log(`Received ${chunk.length} bytes`);
    },
    (progress) => {
        // Handle progress (0.0 to 1.0)
        console.log(`Progress: ${progress * 100}%`);
    },
    { quality: 'medium' }
);
```

## Advanced Usage

### Error Handling

```javascript
const { VoirsError, SynthesisError, VoiceError } = require('voirs-ffi');

try {
    const result = await pipeline.synthesize("Hello world");
} catch (error) {
    if (error instanceof SynthesisError) {
        console.error('Synthesis failed:', error.message);
    } else if (error instanceof VoiceError) {
        console.error('Voice error:', error.message, 'Voice ID:', error.voiceId);
    } else {
        console.error('Unknown error:', error);
    }
}
```

### Multiple Pipelines

You can create multiple pipelines with different configurations:

```javascript
const fastPipeline = new VoirsPipeline({
    numThreads: 2,
    device: 'cpu'
});

const qualityPipeline = new VoirsPipeline({
    useGpu: true,
    numThreads: 8,
    device: 'cuda'
});

// Use different pipelines for different use cases
const quickResult = await fastPipeline.synthesize("Quick message", {
    quality: 'medium'
});

const highQualityResult = await qualityPipeline.synthesize("Important announcement", {
    quality: 'ultra',
    enableEnhancement: true
});
```

### Batch Processing

Process multiple texts efficiently:

```javascript
async function batchSynthesize(texts) {
    const pipeline = new VoirsPipeline({ numThreads: 8 });
    
    const results = await Promise.all(
        texts.map(text => pipeline.synthesize(text, { quality: 'high' }))
    );
    
    return results;
}

const texts = ["Hello", "World", "From", "VoiRS"];
const audioResults = await batchSynthesize(texts);
```

## TypeScript Support

Full TypeScript definitions are included:

```typescript
import { VoirsPipeline, SynthesisOptions, AudioBufferResult } from 'voirs-ffi';

const pipeline = new VoirsPipeline({
    useGpu: false,
    numThreads: 4
});

const options: SynthesisOptions = {
    speakingRate: 1.2,
    quality: 'high',
    outputFormat: 'wav'
};

const result: AudioBufferResult = await pipeline.synthesize("Hello TypeScript!", options);
```

## Performance Tips

1. **Reuse Pipelines**: Create pipelines once and reuse them for multiple syntheses
2. **Threading**: Set `numThreads` to match your CPU cores for optimal performance
3. **Quality vs Speed**: Use lower quality settings for real-time applications
4. **GPU Acceleration**: Enable GPU if available for faster processing
5. **Streaming**: Use streaming for long texts to get partial results early

## Platform Support

| Platform | x64 | ARM64 | x86 |
|----------|-----|-------|-----|
| Windows  | ✅  | ✅    | ✅  |
| macOS    | ✅  | ✅    | ❌  |
| Linux    | ✅  | ✅    | ❌  |

## Examples

See the `examples/` directory for more comprehensive examples:

- `examples/nodejs_example.js` - Complete examples demonstrating all features
- `examples/streaming_demo.js` - Advanced streaming examples
- `examples/batch_processing.js` - Efficient batch processing patterns

## Troubleshooting

### Common Issues

1. **"Cannot find module 'voirs-ffi'"**
   - Ensure the package is installed: `npm install voirs-ffi`
   - Check that you're importing correctly: `const { VoirsPipeline } = require('voirs-ffi');`

2. **"Pipeline creation failed"**
   - Check that you have sufficient memory available
   - Verify GPU drivers if using GPU acceleration
   - Try reducing the number of threads

3. **"Voice not found"**
   - Use `listVoices()` to see available voices
   - Ensure the voice ID is correct
   - Some voices may require additional downloads

4. **Performance issues**
   - Increase `numThreads` for CPU-bound tasks
   - Enable GPU acceleration if available
   - Use appropriate quality settings for your use case

### Debug Mode

Enable debug logging:

```javascript
process.env.RUST_LOG = 'debug';
const pipeline = new VoirsPipeline();
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please see the main repository for contribution guidelines.
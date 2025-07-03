# voirs-cli

[![Crates.io](https://img.shields.io/crates/v/voirs-cli.svg)](https://crates.io/crates/voirs-cli)
[![Documentation](https://docs.rs/voirs-cli/badge.svg)](https://docs.rs/voirs-cli)

**Command-line interface for VoiRS speech synthesis framework.**

A powerful, user-friendly CLI tool for converting text to speech using the VoiRS framework. Features batch processing, real-time synthesis, voice management, and comprehensive output format support.

## Features

- **Text-to-Speech Synthesis**: Convert text files or direct input to high-quality audio
- **SSML Support**: Full Speech Synthesis Markup Language processing
- **Voice Management**: Download, list, and manage voices and models
- **Batch Processing**: Process multiple files efficiently with progress tracking
- **Real-time Synthesis**: Interactive mode with live audio playback
- **Multiple Formats**: Output to WAV, FLAC, MP3, Opus, and streaming audio
- **Quality Control**: Configurable quality settings and audio enhancement
- **Cross-platform**: Windows, macOS, and Linux support

## Installation

### Pre-built Binaries

Download the latest release for your platform from [GitHub Releases](https://github.com/cool-japan/voirs/releases).

### From Source

```bash
cargo install voirs-cli
```

### Package Managers

```bash
# Homebrew (macOS/Linux)
brew install voirs

# Scoop (Windows)
scoop install voirs

# Chocolatey (Windows)
choco install voirs
```

## Quick Start

```bash
# Basic text synthesis
voirs synth "Hello, world!" output.wav

# Use specific voice
voirs synth "Hello, world!" output.wav --voice en-US-female-calm

# SSML synthesis
voirs synth '<speak><emphasis level="strong">Hello</emphasis> world!</speak>' output.wav --ssml

# Interactive mode
voirs interactive

# List available voices
voirs voices list
```

## Commands

### `synth` - Text to Speech Synthesis

Convert text to speech audio.

```bash
voirs synth [OPTIONS] <TEXT> <OUTPUT>

# Examples
voirs synth "Hello world" hello.wav
voirs synth "Hello world" hello.wav --voice en-US-male-news
voirs synth "Bonjour le monde" bonjour.wav --voice fr-FR-female-casual
voirs synth "Hello world" hello.flac --quality high
voirs synth "Hello world" hello.mp3 --bitrate 320
```

#### Options

```bash
-v, --voice <VOICE>          Voice to use for synthesis [default: auto]
-q, --quality <QUALITY>      Synthesis quality [low|medium|high|ultra] [default: high]
-r, --sample-rate <RATE>     Output sample rate [default: 22050]
-f, --format <FORMAT>        Output format [wav|flac|mp3|opus] [default: auto]
-s, --ssml                   Input is SSML markup
    --speed <SPEED>          Speaking rate multiplier [default: 1.0]
    --pitch <PITCH>          Pitch shift in semitones [default: 0.0]
    --volume <VOLUME>        Volume adjustment in dB [default: 0.0]
    --enhance                Enable audio enhancement
    --no-normalize           Skip audio normalization
    --gpu                    Use GPU acceleration if available
    --streaming              Enable streaming synthesis for large texts
    --chunk-size <SIZE>      Chunk size for streaming [default: 256]
```

### `batch` - Batch Processing

Process multiple texts or files efficiently.

```bash
voirs batch [OPTIONS] <INPUT> <OUTPUT_DIR>

# Examples
voirs batch texts.txt ./audio/
voirs batch sentences.csv ./output/ --format flac
voirs batch book.txt ./chapters/ --split-sentences
```

#### Input Formats

```bash
# Text file (one sentence per line)
sentences.txt

# CSV file with columns: text,output_name,voice,speed
metadata.csv

# JSON file with array of synthesis requests
requests.json
```

#### Options

```bash
-f, --format <FORMAT>        Output format for all files
-v, --voice <VOICE>          Default voice for all texts
    --split-sentences        Split long texts into sentences
    --split-paragraphs       Split texts into paragraphs
    --max-length <LENGTH>    Maximum text length per file [default: 1000]
    --parallel <N>           Number of parallel synthesis jobs [default: 4]
    --resume                 Resume interrupted batch processing
    --progress               Show detailed progress information
```

### `interactive` - Interactive Mode

Start an interactive synthesis session.

```bash
voirs interactive [OPTIONS]

# Examples
voirs interactive
voirs interactive --voice en-US-female-calm --auto-play
```

#### Interactive Commands

```
> Hello, this is a test.                    # Synthesize text
> :voice en-GB-male-formal                  # Change voice
> :speed 1.2                                # Adjust speaking rate
> :pitch +0.5                               # Adjust pitch
> :quality ultra                            # Change quality
> :save last_synthesis.wav                  # Save last synthesis
> :play                                     # Replay last synthesis
> :ssml <speak><emphasis>Hello</emphasis></speak>  # SSML mode
> :help                                     # Show help
> :quit                                     # Exit
```

### `voices` - Voice Management

Manage available voices and models.

```bash
voirs voices <SUBCOMMAND>

# Subcommands
voirs voices list              # List available voices
voirs voices search <QUERY>    # Search for voices
voirs voices info <VOICE>      # Show voice details
voirs voices download <VOICE>  # Download voice model
voirs voices remove <VOICE>    # Remove voice model
voirs voices update            # Update voice database
```

#### Examples

```bash
# List all voices
voirs voices list

# List voices by language
voirs voices list --language en-US

# Search for female voices
voirs voices search female

# Get voice information
voirs voices info en-US-female-calm

# Download a voice
voirs voices download en-GB-male-formal

# Remove unused voices
voirs voices remove --unused
```

### `models` - Model Management

Manage synthesis models and backends.

```bash
voirs models <SUBCOMMAND>

# Subcommands
voirs models list              # List available models
voirs models info <MODEL>      # Show model details  
voirs models download <MODEL>  # Download model
voirs models remove <MODEL>    # Remove model
voirs models benchmark         # Benchmark models
voirs models optimize         # Optimize models for current hardware
```

#### Examples

```bash
# List installed models
voirs models list

# Download VITS model
voirs models download vits-en-us-female

# Benchmark all models
voirs models benchmark --output benchmark.json

# Optimize for current GPU
voirs models optimize --device cuda:0
```

### `config` - Configuration Management

Manage VoiRS configuration and preferences.

```bash
voirs config <SUBCOMMAND>

# Subcommands
voirs config show             # Show current configuration
voirs config set <KEY> <VALUE>  # Set configuration value
voirs config reset            # Reset to defaults
voirs config export <FILE>    # Export configuration
voirs config import <FILE>    # Import configuration
```

#### Examples

```bash
# Show configuration
voirs config show

# Set default voice
voirs config set default.voice en-US-female-calm

# Set output directory
voirs config set paths.output ~/Downloads/voirs/

# Reset configuration
voirs config reset --confirm

# Export settings
voirs config export my-settings.toml
```

### `server` - HTTP Server Mode

Start VoiRS as an HTTP API server.

```bash
voirs server [OPTIONS]

# Examples
voirs server --port 8080
voirs server --host 0.0.0.0 --port 3000 --workers 4
```

#### Options

```bash
-p, --port <PORT>           Port to listen on [default: 8080]
-h, --host <HOST>           Host to bind to [default: 127.0.0.1]
-w, --workers <N>           Number of worker threads [default: 4]
    --max-text-length <N>   Maximum text length [default: 5000]
    --rate-limit <N>        Requests per minute per IP [default: 60]
    --cors                  Enable CORS headers
    --api-key <KEY>         Require API key authentication
```

#### API Endpoints

```bash
POST /synthesize              # Synthesize text to audio
GET  /voices                  # List available voices
GET  /voices/{id}             # Get voice information
GET  /health                  # Health check
```

### `benchmark` - Performance Testing

Run performance benchmarks and quality tests.

```bash
voirs benchmark [OPTIONS]

# Examples
voirs benchmark --voices en-US-female-calm,en-GB-male-formal
voirs benchmark --output benchmark.json --detailed
```

#### Options

```bash
-v, --voices <VOICES>       Comma-separated list of voices to test
-o, --output <FILE>         Output results to file
    --detailed              Include detailed metrics
    --quality               Run quality tests (requires reference audio)
    --rtf                   Measure real-time factor
    --memory                Monitor memory usage
    --gpu-usage             Monitor GPU utilization
```

## Configuration

VoiRS uses a hierarchical configuration system with the following precedence:

1. Command-line arguments
2. Environment variables  
3. User configuration file (`~/.voirs/config.toml`)
4. System configuration file (`/etc/voirs/config.toml`)
5. Default values

### Configuration File

```toml
# ~/.voirs/config.toml

[default]
voice = "en-US-female-calm"
quality = "high"
sample_rate = 22050
format = "wav"

[paths]
models = "~/.voirs/models/"
cache = "~/.voirs/cache/"
output = "~/Downloads/"

[synthesis]
gpu_acceleration = true
streaming = false
chunk_size = 256
enhance_audio = true
normalize_output = true

[voices]
auto_download = true
preferred_languages = ["en-US", "en-GB"]
fallback_voice = "en-US-female-neutral"

[server]
host = "127.0.0.1"
port = 8080
workers = 4
max_text_length = 5000
rate_limit = 60

[batch]
parallel_jobs = 4
progress_reporting = true
resume_enabled = true
auto_split = true

[advanced]
backend = "candle"              # candle, onnx
device = "auto"                 # auto, cpu, cuda:0, metal
precision = "fp32"              # fp16, fp32
memory_limit = "4GB"
log_level = "info"              # error, warn, info, debug, trace
```

### Environment Variables

```bash
# Override configuration with environment variables
export VOIRS_DEFAULT_VOICE="en-US-male-news"
export VOIRS_SYNTHESIS_GPU_ACCELERATION="true"
export VOIRS_PATHS_MODELS="/custom/models/path"
export VOIRS_LOG_LEVEL="debug"
```

## Output Formats

### WAV (Uncompressed)
```bash
voirs synth "Hello" output.wav --sample-rate 44100 --bit-depth 24
```

### FLAC (Lossless Compression)
```bash
voirs synth "Hello" output.flac --compression-level 8
```

### MP3 (Lossy Compression)
```bash
voirs synth "Hello" output.mp3 --bitrate 320 --quality high
```

### Opus (Modern Codec)
```bash
voirs synth "Hello" output.opus --bitrate 128 --application audio
```

### Streaming Audio
```bash
# Stream to system audio output
voirs synth "Hello world" --play

# Stream to file while playing
voirs synth "Hello world" output.wav --play --streaming
```

## SSML Support

VoiRS supports Speech Synthesis Markup Language (SSML) for advanced speech control.

### Basic SSML

```bash
voirs synth '<speak>Hello <emphasis level="strong">world</emphasis>!</speak>' output.wav --ssml
```

### Advanced SSML Examples

```xml
<!-- Prosody control -->
<speak>
  <prosody rate="slow" pitch="low" volume="soft">
    This is spoken slowly, in a low pitch, and softly.
  </prosody>
</speak>

<!-- Pauses and breaks -->
<speak>
  Step 1. <break time="1s"/> Step 2. <break time="500ms"/> Step 3.
</speak>

<!-- Phonetic pronunciation -->
<speak>
  You say <phoneme alphabet="ipa" ph="təˈmeɪtoʊ">tomato</phoneme>,
  I say <phoneme alphabet="ipa" ph="təˈmɑːtoʊ">tomato</phoneme>.
</speak>

<!-- Voice selection -->
<speak>
  <voice name="en-US-female-calm">This is a calm female voice.</voice>
  <voice name="en-US-male-energetic">This is an energetic male voice!</voice>
</speak>

<!-- Language switching -->
<speak xml:lang="en-US">
  Hello! <span xml:lang="es-ES">¡Hola!</span> 
  <span xml:lang="fr-FR">Bonjour!</span>
</speak>
```

## Batch Processing

### Text File Input

```
# sentences.txt
Hello, this is the first sentence.
This is the second sentence.
And this is the third sentence.
```

```bash
voirs batch sentences.txt ./output/ --voice en-US-female-calm
```

### CSV Input with Metadata

```csv
text,output_name,voice,speed,pitch
"Hello world",hello,en-US-female-calm,1.0,0.0
"Bonjour le monde",bonjour,fr-FR-female-casual,1.1,0.5
"Hola mundo",hola,es-ES-male-news,0.9,-0.2
```

```bash
voirs batch metadata.csv ./output/ --format flac
```

### JSON Input with Full Control

```json
[
  {
    "text": "Hello, world!",
    "output": "hello.wav",
    "voice": "en-US-female-calm",
    "quality": "high",
    "ssml": false,
    "effects": {
      "speed": 1.0,
      "pitch": 0.0,
      "volume": 0.0
    }
  },
  {
    "text": "<speak><emphasis>Important</emphasis> announcement!</speak>",
    "output": "announcement.wav", 
    "voice": "en-US-male-formal",
    "quality": "ultra",
    "ssml": true
  }
]
```

```bash
voirs batch requests.json ./output/
```

## Performance Optimization

### GPU Acceleration

```bash
# Use GPU if available
voirs synth "Hello world" output.wav --gpu

# Specify GPU device
CUDA_VISIBLE_DEVICES=0 voirs synth "Hello world" output.wav --gpu

# Benchmark GPU performance
voirs benchmark --gpu-usage --voices en-US-female-calm
```

### Streaming for Long Texts

```bash
# Enable streaming for reduced latency
voirs synth "Very long text..." output.wav --streaming --chunk-size 512

# Interactive streaming
echo "Long text content" | voirs synth - output.wav --streaming
```

### Parallel Batch Processing

```bash
# Process with 8 parallel jobs
voirs batch large_dataset.txt ./output/ --parallel 8

# Monitor resource usage
voirs batch large_dataset.txt ./output/ --parallel 4 --progress
```

## Audio Quality Enhancement

### Basic Enhancement

```bash
voirs synth "Hello world" output.wav --enhance
```

### Advanced Audio Processing

```bash
# Custom quality settings
voirs synth "Hello world" output.wav \
  --quality ultra \
  --enhance \
  --volume +3.0 \
  --sample-rate 48000

# Professional audio settings
voirs synth "Hello world" broadcast.wav \
  --quality ultra \
  --enhance \
  --format wav \
  --sample-rate 48000 \
  --bit-depth 24 \
  --no-normalize  # Skip normalization for professional workflow
```

## Troubleshooting

### Common Issues

**Voice not found:**
```bash
# List available voices
voirs voices list

# Download missing voice
voirs voices download en-US-female-calm
```

**GPU not working:**
```bash
# Check GPU support
voirs config show | grep gpu

# Force CPU mode
voirs synth "Hello" output.wav --device cpu
```

**Poor audio quality:**
```bash
# Try higher quality settings
voirs synth "Hello" output.wav --quality ultra --enhance

# Check sample rate
voirs synth "Hello" output.wav --sample-rate 48000
```

**Memory issues:**
```bash
# Enable streaming for large texts
voirs synth "$(cat large_text.txt)" output.wav --streaming

# Reduce chunk size
voirs synth "$(cat large_text.txt)" output.wav --streaming --chunk-size 128
```

### Debug Mode

```bash
# Enable verbose logging
VOIRS_LOG_LEVEL=debug voirs synth "Hello" output.wav

# Save debug information
voirs synth "Hello" output.wav --debug --debug-output debug.json
```

### Performance Issues

```bash
# Profile synthesis performance
voirs benchmark --voices en-US-female-calm --detailed

# Check system resources
voirs benchmark --memory --gpu-usage

# Optimize models for your hardware
voirs models optimize --device auto
```

## Integration Examples

### Shell Scripts

```bash
#!/bin/bash
# text_to_speech.sh - Convert text files to audio

for file in *.txt; do
    echo "Processing $file..."
    voirs synth "$(cat "$file")" "${file%.txt}.wav" \
        --voice en-US-female-calm \
        --quality high \
        --progress
done
```

### Python Integration

```python
import subprocess
import json

def synthesize_text(text, output_file, voice="en-US-female-calm"):
    """Synthesize text using VoiRS CLI"""
    cmd = [
        "voirs", "synth", text, output_file,
        "--voice", voice,
        "--quality", "high"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Synthesis failed: {result.stderr}")
    
    return output_file

# Usage
synthesize_text("Hello, world!", "greeting.wav")
```

### Web Integration

```javascript
// Node.js example using child_process
const { exec } = require('child_process');

function synthesizeText(text, outputFile) {
    return new Promise((resolve, reject) => {
        const cmd = `voirs synth "${text}" "${outputFile}" --quality high`;
        
        exec(cmd, (error, stdout, stderr) => {
            if (error) {
                reject(error);
            } else {
                resolve(outputFile);
            }
        });
    });
}

// Usage
synthesizeText("Hello from Node.js!", "greeting.wav")
    .then(file => console.log(`Audio saved to ${file}`))
    .catch(err => console.error(`Error: ${err.message}`));
```

## Contributing

We welcome contributions! Please see the [main repository](https://github.com/cool-japan/voirs) for contribution guidelines.

### Development Setup

```bash
git clone https://github.com/cool-japan/voirs.git
cd voirs/crates/voirs-cli

# Install development dependencies
cargo install cargo-nextest

# Run tests
cargo nextest run

# Run CLI locally
cargo run -- synth "Hello world" test.wav

# Build release version
cargo build --release
```

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE))
- MIT license ([LICENSE-MIT](../../LICENSE-MIT))

at your option.
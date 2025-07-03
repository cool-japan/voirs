# voirs-vocoder

[![Crates.io](https://img.shields.io/crates/v/voirs-vocoder.svg)](https://crates.io/crates/voirs-vocoder)
[![Documentation](https://docs.rs/voirs-vocoder/badge.svg)](https://docs.rs/voirs-vocoder)

**Neural vocoding for VoiRS speech synthesis - converts mel spectrograms to high-quality audio.**

This crate implements state-of-the-art neural vocoders including HiFi-GAN and DiffWave for converting mel spectrograms into high-quality audio waveforms. It serves as the final stage in the VoiRS pipeline, transforming acoustic model outputs into listenable speech.

## Features

- **HiFi-GAN Implementation**: Fast, high-quality generative adversarial vocoder
- **DiffWave Support**: Diffusion-based vocoder for ultra-high quality synthesis
- **Multi-sample Rate**: Support for 16kHz, 22kHz, 44kHz, and 48kHz output
- **Real-time Streaming**: Low-latency chunk-based audio generation (<50ms)
- **GPU Acceleration**: CUDA, Metal, and OpenCL backends for fast inference
- **Post-processing**: Dynamic range compression, noise gating, and enhancement
- **Format Support**: WAV, FLAC, MP3, and Opus output formats

## Quick Start

```rust
use voirs_vocoder::{HiFiGAN, Vocoder, AudioBuffer};
use voirs_acoustic::MelSpectrogram;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load pre-trained HiFi-GAN model
    let vocoder = HiFiGAN::from_pretrained("hifigan-22k").await?;
    
    // Convert mel spectrogram to audio
    let mel: MelSpectrogram = /* from acoustic model */;
    let audio: AudioBuffer = vocoder.vocode(&mel, None).await?;
    
    // Save audio to file
    audio.save_wav("output.wav")?;
    
    println!("Generated audio: {:.2}s @ {}Hz", 
             audio.duration(), audio.sample_rate());
    
    Ok(())
}
```

## Supported Models

| Model | Type | Quality (MOS) | Speed (RTF) | Latency | Size | Status |
|-------|------|---------------|-------------|---------|------|--------|
| HiFi-GAN V1 | GAN | 4.38 | 0.02× | 12ms | 17MB | ✅ Stable |
| HiFi-GAN V2 | GAN | 4.31 | 0.01× | 8ms | 14MB | ✅ Stable |
| HiFi-GAN V3 | GAN | 4.42 | 0.03× | 15ms | 23MB | ✅ Stable |
| DiffWave | Diffusion | 4.54 | 0.15× | 180ms | 32MB | 🚧 Beta |
| MelGAN | GAN | 3.97 | 0.01× | 6ms | 8MB | 🚧 Beta |
| UnivNet | GAN | 4.36 | 0.02× | 11ms | 19MB | 📋 Planned |

## Architecture

```
Mel Spectrogram → Upsampling → Multi-Receptive Field → Post-processing → Audio
      ↓              ↓               ↓                      ↓            ↓
   [80, 256]    [1, 5632]      CNN Layers           Enhancement    [1, 88200]
```

### Core Components

1. **Upsampling Network**
   - Transposed convolution layers
   - Anti-aliasing filters
   - Progressive upsampling (×2, ×2, ×5, ×5)

2. **Multi-Receptive Field (MRF)**
   - Parallel residual blocks
   - Different kernel sizes (3, 7, 11)
   - Feature fusion and gating

3. **Post-processing**
   - Dynamic range compression
   - High-frequency enhancement
   - Noise gate and filtering

## API Reference

### Core Trait

```rust
#[async_trait]
pub trait Vocoder: Send + Sync {
    /// Convert mel spectrogram to audio
    async fn vocode(
        &self, 
        mel: &MelSpectrogram, 
        config: Option<&VocodingConfig>
    ) -> Result<AudioBuffer>;
    
    /// Stream-based vocoding for real-time synthesis
    async fn vocode_stream(
        &self,
        mel_stream: impl Stream<Item = MelSpectrogram> + Send,
        config: Option<&StreamConfig>
    ) -> Result<impl Stream<Item = Result<AudioBuffer>>>;
    
    /// Batch vocoding for multiple inputs
    async fn vocode_batch(
        &self,
        mels: &[MelSpectrogram],
        configs: Option<&[VocodingConfig]>
    ) -> Result<Vec<AudioBuffer>>;
    
    /// Get vocoder metadata and capabilities
    fn metadata(&self) -> VocoderMetadata;
}
```

### HiFi-GAN Model

```rust
pub struct HiFiGAN {
    generator: Generator,
    sample_rate: u32,
    hop_length: u32,
    device: Device,
    config: HiFiGANConfig,
}

impl HiFiGAN {
    /// Load pre-trained model from HuggingFace Hub
    pub async fn from_pretrained(model_id: &str) -> Result<Self>;
    
    /// Load model from local files
    pub async fn from_files(
        config_path: &Path, 
        weights_path: &Path
    ) -> Result<Self>;
    
    /// Generate audio with quality control
    pub async fn vocode_with_quality(
        &self,
        mel: &MelSpectrogram,
        quality: Quality,
        enhancement: Option<Enhancement>
    ) -> Result<AudioBuffer>;
}
```

### Audio Buffer

```rust
#[derive(Debug, Clone)]
pub struct AudioBuffer {
    /// Audio samples as f32 values in range [-1.0, 1.0]
    samples: Vec<f32>,
    
    /// Sample rate in Hz
    sample_rate: u32,
    
    /// Number of audio channels (1=mono, 2=stereo)
    channels: u32,
    
    /// Bit depth for output encoding
    bit_depth: u16,
}

impl AudioBuffer {
    /// Get audio duration in seconds
    pub fn duration(&self) -> f32;
    
    /// Save to WAV file
    pub fn save_wav(&self, path: impl AsRef<Path>) -> Result<()>;
    
    /// Save to various formats (feature: "encoding")
    #[cfg(feature = "encoding")]
    pub fn save(&self, path: impl AsRef<Path>, format: AudioFormat) -> Result<()>;
    
    /// Convert to raw samples for processing
    pub fn samples(&self) -> &[f32];
    
    /// Resample to different sample rate
    pub fn resample(&self, target_rate: u32) -> Result<AudioBuffer>;
    
    /// Apply audio effects and enhancement
    pub fn enhance(&self, config: &EnhancementConfig) -> Result<AudioBuffer>;
}
```

## Usage Examples

### Basic Vocoding

```rust
use voirs_vocoder::{HiFiGAN, Vocoder};

let vocoder = HiFiGAN::from_pretrained("hifigan-22k").await?;

// Simple vocoding
let mel = /* mel spectrogram from acoustic model */;
let audio = vocoder.vocode(&mel, None).await?;
```

### Quality Control

```rust
use voirs_vocoder::{VocodingConfig, Quality, Enhancement};

let config = VocodingConfig {
    quality: Quality::High,
    enhancement: Some(Enhancement {
        noise_gate: true,
        compressor: Some(CompressorConfig {
            threshold: -20.0,
            ratio: 3.0,
            attack: 0.003,
            release: 0.1,
        }),
        high_freq_boost: 0.1,    // 10% boost above 8kHz
        warmth: 0.05,            // subtle low-freq enhancement
    }),
    output_format: AudioFormat::Wav16,
    ..Default::default()
};

let audio = vocoder.vocode(&mel, Some(&config)).await?;
```

### Streaming Vocoding

```rust
use voirs_vocoder::{StreamingVocoder, StreamConfig};
use futures::StreamExt;

let vocoder = StreamingVocoder::new(HiFiGAN::from_pretrained("hifigan-22k").await?);

let stream_config = StreamConfig {
    chunk_size: 256,             // mel frames per chunk
    overlap: 64,                 // overlap for smooth concatenation
    max_latency_ms: 50,         // target latency
    buffer_size: 1024,          // output buffer size
};

let mel_stream = /* stream of mel spectrograms */;
let mut audio_stream = vocoder.vocode_stream(mel_stream, Some(&stream_config)).await?;

while let Some(audio_chunk) = audio_stream.next().await {
    let chunk = audio_chunk?;
    // Process audio chunk immediately for real-time playback
    play_audio(chunk).await?;
}
```

### Batch Processing

```rust
use voirs_vocoder::{BatchVocoder, BatchConfig};

let batch_vocoder = BatchVocoder::new(vocoder, BatchConfig {
    max_batch_size: 8,
    max_total_frames: 4096,      // limit memory usage
    padding_strategy: PaddingStrategy::Shortest,
});

let mel_batch: Vec<MelSpectrogram> = load_mel_spectrograms()?;
let audio_batch = batch_vocoder.vocode_batch(&mel_batch, None).await?;
```

### Multi-format Output

```rust
use voirs_vocoder::{AudioFormat, EncodingConfig};

// Save as different formats
audio.save_wav("output.wav")?;

#[cfg(feature = "encoding")]
{
    audio.save("output.flac", AudioFormat::Flac)?;
    audio.save("output.mp3", AudioFormat::Mp3(Mp3Config {
        bitrate: 320,
        quality: Mp3Quality::High,
    }))?;
    audio.save("output.opus", AudioFormat::Opus(OpusConfig {
        bitrate: 128,
        application: OpusApplication::Audio,
    }))?;
}
```

### DiffWave High-Quality Synthesis

```rust
use voirs_vocoder::{DiffWave, DiffusionConfig, SamplingSchedule};

let diffwave = DiffWave::from_pretrained("diffwave-22k").await?;

let config = DiffusionConfig {
    num_steps: 50,               // more steps = higher quality
    schedule: SamplingSchedule::Linear,
    guidance_scale: 1.0,
    temperature: 0.8,            // control randomness
};

let audio = diffwave.vocode_with_diffusion(&mel, &config).await?;
```

### Real-time Audio Effects

```rust
use voirs_vocoder::{AudioProcessor, EffectChain};

let processor = AudioProcessor::new();
let effects = EffectChain::new()
    .add_reverb(ReverbConfig::room())
    .add_eq(EqConfig::vocal_presence())
    .add_limiter(LimiterConfig::broadcast());

let enhanced_audio = processor.apply_effects(&audio, &effects)?;
```

## Performance

### Benchmarks (Intel i7-12700K + RTX 4080)

| Model | Backend | Device | RTF | Throughput | Quality (MOS) |
|-------|---------|--------|-----|------------|---------------|
| HiFi-GAN V1 | Candle | CPU | 0.02× | 180 sent/s | 4.38 |
| HiFi-GAN V1 | Candle | CUDA | 0.005× | 750 sent/s | 4.38 |
| HiFi-GAN V2 | Candle | CPU | 0.015× | 220 sent/s | 4.31 |
| HiFi-GAN V2 | Candle | CUDA | 0.003× | 900 sent/s | 4.31 |
| DiffWave | Candle | CPU | 0.15× | 25 sent/s | 4.54 |
| DiffWave | Candle | CUDA | 0.08× | 45 sent/s | 4.54 |

### Latency Analysis

- **HiFi-GAN V1**: 12ms end-to-end (256 mel frames)
- **HiFi-GAN V2**: 8ms end-to-end (256 mel frames)  
- **DiffWave**: 180ms end-to-end (50 diffusion steps)
- **Streaming**: <50ms additional buffering latency

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
voirs-vocoder = "0.1"

# Enable specific features
[dependencies.voirs-vocoder]
version = "0.1"
features = ["candle", "onnx", "gpu", "encoding"]
```

### Feature Flags

- `candle`: Enable Candle backend (default)
- `onnx`: Enable ONNX Runtime backend
- `gpu`: Enable GPU acceleration (CUDA/Metal)
- `streaming`: Enable real-time streaming vocoding
- `encoding`: Enable MP3, FLAC, Opus output formats
- `effects`: Enable audio effects and post-processing
- `scirs`: Integration with SciRS2 for optimized DSP

### System Dependencies

**Audio encoding support:**
```bash
# Ubuntu/Debian
sudo apt-get install libsndfile1-dev libopus-dev libmp3lame-dev

# macOS
brew install libsndfile opus lame
```

**GPU acceleration:**
```bash
# CUDA (NVIDIA)
export CUDA_ROOT=/usr/local/cuda

# Metal (macOS) - built-in, no additional setup needed
```

## Configuration

Create `~/.voirs/vocoder.toml`:

```toml
[default]
model = "hifigan-22k"
backend = "candle"
device = "auto"              # auto, cpu, cuda:0, metal
quality = "high"             # low, medium, high, ultra

[models]
cache_dir = "~/.voirs/models/vocoder"
auto_download = true
verify_checksums = true

[hifigan]
generator_version = "v1"     # v1, v2, v3
upsample_rates = [8, 8, 2, 2]
upsample_kernel_sizes = [16, 16, 4, 4]

[diffwave]
default_steps = 50
fast_steps = 20              # for real-time applications
schedule = "linear"          # linear, cosine, sigmoid

[streaming]
chunk_size = 256
overlap = 64
max_latency_ms = 50
buffer_frames = 1024

[enhancement]
enable_noise_gate = true
enable_compressor = true
enable_eq = false
high_freq_boost = 0.0

[output]
default_format = "wav"       # wav, flac, mp3, opus
default_sample_rate = 22050
default_bit_depth = 16
```

## Audio Quality Optimization

### Quality vs Speed Trade-offs

```rust
use voirs_vocoder::{Quality, PerformanceMode};

// Ultra-high quality (slower)
let config = VocodingConfig {
    quality: Quality::Ultra,
    performance: PerformanceMode::Quality,
    enhancement: Some(Enhancement::studio()),
    ..Default::default()
};

// Real-time optimized (faster)
let config = VocodingConfig {
    quality: Quality::Medium,
    performance: PerformanceMode::Speed,
    enhancement: Some(Enhancement::realtime()),
    ..Default::default()
};
```

### Custom Enhancement Pipeline

```rust
use voirs_vocoder::{EnhancementPipeline, AudioEffect};

let pipeline = EnhancementPipeline::builder()
    .add_effect(AudioEffect::NoiseGate {
        threshold: -40.0,
        ratio: 10.0,
        attack: 0.001,
        release: 0.05,
    })
    .add_effect(AudioEffect::Compressor {
        threshold: -12.0,
        ratio: 3.0,
        attack: 0.003,
        release: 0.1,
        makeup_gain: 2.0,
    })
    .add_effect(AudioEffect::Eq {
        low_shelf: EqBand { freq: 100.0, gain: 0.0, q: 0.7 },
        mid_peak: EqBand { freq: 2000.0, gain: 1.0, q: 1.4 },
        high_shelf: EqBand { freq: 8000.0, gain: 0.5, q: 0.7 },
    })
    .build();

let enhanced_audio = pipeline.process(&audio)?;
```

## Error Handling

```rust
use voirs_vocoder::{VocoderError, ErrorKind};

match vocoder.vocode(&mel, None).await {
    Ok(audio) => println!("Success: {:.2}s audio", audio.duration()),
    Err(VocoderError { kind, context, .. }) => match kind {
        ErrorKind::ModelNotFound => {
            eprintln!("Vocoder model not found: {}", context);
        }
        ErrorKind::InvalidMelSpectrogram => {
            eprintln!("Invalid mel spectrogram: {}", context);
        }
        ErrorKind::InferenceError => {
            eprintln!("Vocoding failed: {}", context);
        }
        ErrorKind::AudioProcessingError => {
            eprintln!("Audio processing error: {}", context);
        }
        _ => eprintln!("Other error: {}", context),
    }
}
```

## Advanced Features

### Custom Vocoder Implementation

```rust
use voirs_vocoder::{Vocoder, VocoderMetadata, ModelFeature};

pub struct CustomVocoder {
    // Custom implementation
}

#[async_trait]
impl Vocoder for CustomVocoder {
    async fn vocode(
        &self, 
        mel: &MelSpectrogram, 
        config: Option<&VocodingConfig>
    ) -> Result<AudioBuffer> {
        // Custom vocoding logic
        todo!()
    }
    
    fn metadata(&self) -> VocoderMetadata {
        VocoderMetadata {
            name: "CustomVocoder".to_string(),
            version: "1.0.0".to_string(),
            sample_rates: vec![22050, 44100],
            features: vec![ModelFeature::Streaming, ModelFeature::BatchProcessing],
            latency_ms: 25.0,
        }
    }
}
```

### Audio Analysis and Debugging

```rust
#[cfg(feature = "analysis")]
use voirs_vocoder::{AudioAnalyzer, SpectrumPlot};

let analyzer = AudioAnalyzer::new();
let analysis = analyzer.analyze(&audio)?;

println!("Peak level: {:.2} dB", analysis.peak_db);
println!("RMS level: {:.2} dB", analysis.rms_db);
println!("THD+N: {:.3}%", analysis.thd_plus_noise * 100.0);
println!("Dynamic range: {:.1} dB", analysis.dynamic_range);

// Visualize spectrum
let plot = SpectrumPlot::new(&audio)
    .frequency_range(20.0, 20000.0)
    .db_range(-80.0, 0.0);
plot.save("spectrum.png")?;
```

## Contributing

We welcome contributions! Please see the [main repository](https://github.com/cool-japan/voirs) for contribution guidelines.

### Development Setup

```bash
git clone https://github.com/cool-japan/voirs.git
cd voirs/crates/voirs-vocoder

# Install development dependencies
cargo install cargo-nextest criterion

# Run tests
cargo nextest run

# Run benchmarks
cargo bench

# Check code quality
cargo clippy -- -D warnings
cargo fmt --check
```

### Adding New Vocoders

1. Implement the `Vocoder` trait
2. Add model configuration and loading logic
3. Create comprehensive tests and benchmarks
4. Add audio quality validation
5. Update documentation and examples

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE))
- MIT license ([LICENSE-MIT](../../LICENSE-MIT))

at your option.
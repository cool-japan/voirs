# voirs-acoustic

[![Crates.io](https://img.shields.io/crates/v/voirs-acoustic.svg)](https://crates.io/crates/voirs-acoustic)
[![Documentation](https://docs.rs/voirs-acoustic/badge.svg)](https://docs.rs/voirs-acoustic)

**Neural acoustic modeling for VoiRS speech synthesis - converts phonemes to mel spectrograms.**

This crate implements state-of-the-art neural acoustic models including VITS (Variational Inference Text-to-Speech) and FastSpeech2. It serves as the core component in the VoiRS pipeline, transforming phonetic representations into mel spectrograms that can be converted to audio by vocoders.

## Features

- **VITS Implementation**: End-to-end variational autoencoder for high-quality synthesis
- **FastSpeech2 Support**: Non-autoregressive transformer for fast and controllable synthesis  
- **Multi-backend**: Candle (native Rust) and ONNX Runtime support
- **GPU Acceleration**: CUDA, Metal, and OpenCL backends for fast inference
- **Speaker Control**: Multi-speaker models with speaker embedding and voice morphing
- **Prosody Control**: Fine-grained control over pitch, duration, and energy
- **Streaming Inference**: Real-time synthesis with low latency

## Quick Start

```rust
use voirs_acoustic::{VitsModel, AcousticModel, MelSpectrogram};
use voirs_g2p::Phoneme;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load pre-trained VITS model
    let model = VitsModel::from_pretrained("vits-en-us-female").await?;
    
    // Convert phonemes to mel spectrogram
    let phonemes = vec![/* phonemes from G2P */];
    let mel: MelSpectrogram = model.synthesize(&phonemes, None).await?;
    
    // Mel spectrogram is ready for vocoder
    println!("Generated mel: {}x{}", mel.n_mels(), mel.n_frames());
    
    Ok(())
}
```

## Supported Models

| Model | Type | Quality (MOS) | Speed (RTF) | Size | Status |
|-------|------|---------------|-------------|------|--------|
| VITS-EN-US | VITS | 4.42 | 0.28× | 89MB | ✅ Stable |
| VITS-EN-UK | VITS | 4.38 | 0.28× | 89MB | ✅ Stable |
| VITS-JP | VITS | 4.35 | 0.31× | 92MB | ✅ Stable |
| VITS-Multilingual | VITS | 4.15 | 0.35× | 156MB | 🚧 Beta |
| FastSpeech2-EN | Non-AR | 4.21 | 0.15× | 67MB | 🚧 Beta |

## Architecture

```
Phonemes → Text Encoder → Posterior Encoder → Flow → Decoder → Mel Spectrogram
    ↓           ↓              ↓             ↓       ↓            ↓
 [P, H, OW]  Transformer    CNN Features   Flows   CNN Gen   [80, 256]
```

### Core Components

1. **Text Encoder**
   - Transformer-based phoneme embedding
   - Positional encoding and attention
   - Language-specific encoding layers

2. **Posterior Encoder** (VITS only)
   - CNN-based feature extraction
   - Variational posterior estimation
   - KL divergence regularization

3. **Normalizing Flows** (VITS only)
   - Invertible transformations
   - Gaussian to complex distribution mapping
   - Stochastic duration modeling

4. **Decoder/Generator**
   - CNN-based mel generation
   - Multi-scale feature fusion
   - Residual and gated convolutions

## API Reference

### Core Trait

```rust
#[async_trait]
pub trait AcousticModel: Send + Sync {
    /// Generate mel spectrogram from phonemes
    async fn synthesize(
        &self, 
        phonemes: &[Phoneme], 
        config: Option<&SynthesisConfig>
    ) -> Result<MelSpectrogram>;
    
    /// Batch synthesis for multiple inputs
    async fn synthesize_batch(
        &self,
        inputs: &[&[Phoneme]],
        configs: Option<&[SynthesisConfig]>
    ) -> Result<Vec<MelSpectrogram>>;
    
    /// Get model metadata and capabilities
    fn metadata(&self) -> ModelMetadata;
    
    /// Check if model supports specific features
    fn supports(&self, feature: ModelFeature) -> bool;
}
```

### VITS Model

```rust
pub struct VitsModel {
    text_encoder: TextEncoder,
    posterior_encoder: PosteriorEncoder,
    decoder: Decoder,
    flow: NormalizingFlows,
    device: Device,
    config: VitsConfig,
}

impl VitsModel {
    /// Load pre-trained model from HuggingFace Hub
    pub async fn from_pretrained(model_id: &str) -> Result<Self>;
    
    /// Load model from local files
    pub async fn from_files(
        config_path: &Path, 
        weights_path: &Path
    ) -> Result<Self>;
    
    /// Generate with speaker control
    pub async fn synthesize_with_speaker(
        &self,
        phonemes: &[Phoneme],
        speaker_id: Option<u32>,
        emotion: Option<EmotionVector>
    ) -> Result<MelSpectrogram>;
}
```

### Mel Spectrogram

```rust
#[derive(Debug, Clone)]
pub struct MelSpectrogram {
    /// Mel filterbank features [n_mels, n_frames]
    data: Tensor,
    
    /// Sample rate in Hz
    sample_rate: u32,
    
    /// Hop length in samples
    hop_length: u32,
    
    /// Number of mel channels
    n_mels: u32,
}

impl MelSpectrogram {
    /// Get mel values as ndarray for processing
    pub fn to_array(&self) -> Array2<f32>;
    
    /// Convert to raw tensor for vocoder input
    pub fn to_tensor(&self) -> &Tensor;
    
    /// Get duration in seconds
    pub fn duration(&self) -> f32;
    
    /// Visualize mel spectrogram (feature: "plotting")
    #[cfg(feature = "plotting")]
    pub fn plot(&self) -> Plot;
}
```

## Usage Examples

### Basic Synthesis

```rust
use voirs_acoustic::{VitsModel, AcousticModel};

let model = VitsModel::from_pretrained("vits-en-us-female").await?;

// Simple synthesis
let phonemes = vec![/* phonemes from G2P */];
let mel = model.synthesize(&phonemes, None).await?;
```

### Multi-Speaker Synthesis

```rust
use voirs_acoustic::{VitsModel, SynthesisConfig, SpeakerConfig};

let model = VitsModel::from_pretrained("vits-multilingual").await?;

let config = SynthesisConfig {
    speaker: Some(SpeakerConfig {
        speaker_id: Some(42),
        emotion: Some(EmotionVector::happy(0.8)),
        age: Some(25.0),
        gender: Some(Gender::Female),
    }),
    ..Default::default()
};

let mel = model.synthesize(&phonemes, Some(&config)).await?;
```

### Prosody Control

```rust
use voirs_acoustic::{ProsodyConfig, DurationControl, PitchControl};

let config = SynthesisConfig {
    prosody: Some(ProsodyConfig {
        speaking_rate: 1.2,           // 20% faster
        pitch_shift: 0.1,             // 10% higher pitch
        energy_scale: 1.1,            // 10% more energy
        duration_control: DurationControl::Predictive,
        pitch_control: PitchControl::Neural,
    }),
    ..Default::default()
};

let mel = model.synthesize(&phonemes, Some(&config)).await?;
```

### Streaming Synthesis

```rust
use voirs_acoustic::{StreamingVits, StreamConfig};
use futures::StreamExt;

let model = StreamingVits::from_pretrained("vits-en-us-female").await?;

let stream_config = StreamConfig {
    chunk_size: 256,              // frames per chunk
    overlap: 64,                  // overlap between chunks
    max_latency_ms: 50,          // maximum acceptable latency
};

let mut stream = model.synthesize_stream(&phonemes, stream_config).await?;

while let Some(mel_chunk) = stream.next().await {
    let chunk = mel_chunk?;
    // Process chunk immediately for low latency
    send_to_vocoder(chunk).await?;
}
```

### Batch Processing

```rust
use voirs_acoustic::{BatchProcessor, BatchConfig};

let processor = BatchProcessor::new(model, BatchConfig {
    max_batch_size: 16,
    max_sequence_length: 1000,
    padding_strategy: PaddingStrategy::Longest,
});

let phoneme_batches: Vec<Vec<Phoneme>> = load_phoneme_data()?;
let mel_batches = processor.process_batches(&phoneme_batches).await?;
```

### Custom Model Loading

```rust
use voirs_acoustic::{VitsConfig, ModelLoader};

// Load from custom configuration
let config = VitsConfig {
    text_encoder: TextEncoderConfig {
        n_vocab: 512,
        hidden_channels: 384,
        filter_channels: 1536,
        n_heads: 2,
        n_layers: 6,
        kernel_size: 3,
        p_dropout: 0.1,
    },
    // ... other config
};

let model = VitsModel::from_config(config, "path/to/weights.safetensors").await?;
```

## Performance

### Benchmarks (Intel i7-12700K + RTX 4080)

| Model | Backend | Device | RTF | Throughput | Memory |
|-------|---------|--------|-----|------------|--------|
| VITS-EN | Candle | CPU | 0.28× | 45 sent/s | 512MB |
| VITS-EN | Candle | CUDA | 0.04× | 320 sent/s | 2.1GB |
| VITS-EN | ONNX | CPU | 0.31× | 42 sent/s | 480MB |
| VITS-EN | ONNX | CUDA | 0.05× | 290 sent/s | 1.8GB |
| FastSpeech2 | Candle | CPU | 0.15× | 78 sent/s | 384MB |
| FastSpeech2 | Candle | CUDA | 0.02× | 450 sent/s | 1.2GB |

### Quality Metrics

- **Naturalness (MOS)**: 4.42 ± 0.15 (22kHz, VITS-EN-US)
- **Speaker Similarity**: 0.89 ± 0.08 Si-SDR (multi-speaker models)
- **Intelligibility**: 98.7% word accuracy (ASR evaluation)
- **Prosody Correlation**: 0.83 with human-rated naturalness

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
voirs-acoustic = "0.1"

# Enable specific backends
[dependencies.voirs-acoustic]
version = "0.1"
features = ["candle", "onnx", "gpu"]
```

### Feature Flags

- `candle`: Enable Candle backend (default)
- `onnx`: Enable ONNX Runtime backend
- `gpu`: Enable GPU acceleration (CUDA/Metal)
- `streaming`: Enable streaming synthesis
- `training`: Enable model training capabilities
- `plotting`: Enable mel spectrogram visualization
- `scirs`: Integration with SciRS2 for optimized DSP
- `numrs`: Integration with NumRS2 for linear algebra

### System Dependencies

**CUDA backend:**
```bash
# Ensure CUDA 11.8+ is installed
export CUDA_ROOT=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_ROOT/lib64:$LD_LIBRARY_PATH
```

**ONNX backend:**
```bash
# Ubuntu/Debian
sudo apt-get install libonnxruntime-dev

# macOS
brew install onnxruntime
```

## Configuration

Create `~/.voirs/acoustic.toml`:

```toml
[default]
backend = "candle"          # candle, onnx
device = "auto"             # auto, cpu, cuda:0, metal
precision = "fp32"          # fp32, fp16

[models]
cache_dir = "~/.voirs/models/acoustic"
auto_download = true
verify_checksums = true

[candle]
enable_flash_attention = true
use_memory_pool = true
optimize_for_inference = true

[onnx]
provider = "cuda"           # cpu, cuda, tensorrt
inter_op_threads = 4
intra_op_threads = 8
enable_profiling = false

[synthesis]
default_sample_rate = 22050
default_hop_length = 256
default_n_mels = 80
max_sequence_length = 1000

[streaming]
chunk_size = 256
overlap = 64
max_latency_ms = 50
buffer_size = 1024
```

## Model Training

### Training a VITS Model

```rust
use voirs_acoustic::{VitsTrainer, TrainingConfig, DataLoader};

let config = TrainingConfig {
    model: VitsConfig::default(),
    optimizer: OptimizerConfig::adam(1e-4),
    scheduler: SchedulerConfig::exponential(0.999),
    batch_size: 32,
    gradient_accumulation: 4,
    max_epochs: 1000,
    ..Default::default()
};

let trainer = VitsTrainer::new(config);
let dataloader = DataLoader::from_manifest("train_manifest.json").await?;

trainer.train(dataloader).await?;
```

### Fine-tuning for New Speaker

```rust
use voirs_acoustic::{FineTuner, SpeakerConfig};

let base_model = VitsModel::from_pretrained("vits-en-us-base").await?;
let fine_tuner = FineTuner::new(base_model);

let speaker_config = SpeakerConfig {
    speaker_data: "path/to/speaker/audio",
    target_hours: 2.0,           // 2 hours of data
    learning_rate: 1e-5,
    freeze_encoder: true,        // Only fine-tune decoder
};

let custom_model = fine_tuner.fine_tune(speaker_config).await?;
```

## Error Handling

```rust
use voirs_acoustic::{AcousticError, ErrorKind};

match model.synthesize(&phonemes, None).await {
    Ok(mel) => println!("Success: {} frames", mel.n_frames()),
    Err(AcousticError { kind, context, .. }) => match kind {
        ErrorKind::ModelNotFound => {
            eprintln!("Model not found: {}", context);
        }
        ErrorKind::InvalidInput => {
            eprintln!("Invalid phoneme sequence: {}", context);
        }
        ErrorKind::InferenceError => {
            eprintln!("Model inference failed: {}", context);
        }
        ErrorKind::DeviceError => {
            eprintln!("GPU/device error: {}", context);
        }
        _ => eprintln!("Other error: {}", context),
    }
}
```

## Advanced Features

### Voice Morphing

```rust
use voirs_acoustic::{VoiceMorpher, MorphingConfig};

let morpher = VoiceMorpher::new();
let config = MorphingConfig {
    source_speaker: 0,
    target_speaker: 1,
    interpolation_factor: 0.5,   // 50% blend
    preserve_prosody: true,
};

let morphed_mel = morpher.morph(&base_mel, &config).await?;
```

### Attention Visualization

```rust
#[cfg(feature = "plotting")]
use voirs_acoustic::{AttentionVisualizer, VisualizationConfig};

let visualizer = AttentionVisualizer::new();
let attention_weights = model.get_attention_weights(&phonemes).await?;

let plot = visualizer.plot_attention(
    &attention_weights,
    &phonemes,
    &mel,
    VisualizationConfig::default()
);

plot.save("attention.png")?;
```

## Examples and Benchmarks

### Available Examples

The crate includes several comprehensive examples demonstrating different features:

- **`simple_synthesis_demo`**: Basic synthesis workflow with minimal setup
- **`advanced_features_demo`**: Comprehensive demo of advanced features (requires `candle` feature)
- **`fusion_optimization_demo`**: Kernel fusion optimization demonstrations
- **`profiling_demo`**: Performance profiling and tracing examples
- **`production_monitoring_demo`**: Production monitoring and metrics tracking
- **`scirs2_optimization_demo`**: SciRS2-Core integration and SIMD optimizations
- **`model_optimization`**: Model optimization techniques (requires `candle` feature)
- **`pretrained_models`**: Working with pre-trained models from HuggingFace Hub (requires `candle` feature)

Run examples with:
```bash
# Basic example (no features required)
cargo run --example simple_synthesis_demo

# Advanced features (requires candle)
cargo run --example advanced_features_demo --features candle

# Fusion optimization demo
cargo run --example fusion_optimization_demo
```

### Performance Benchmarks

The crate includes comprehensive benchmark suites for performance analysis:

- **`simple_benchmarks`**: Basic performance regression testing
- **`performance_validation`**: Performance target validation across different scenarios
- **`advanced_features_benchmarks`**: Advanced features performance testing
- **`profiling_benchmarks`**: Profiling system overhead measurements
- **`scirs2_benchmarks`**: SciRS2-Core optimization benchmarks (SIMD, parallel, etc.)
- **`fusion_benchmarks`**: Kernel fusion performance improvements
- **`production_benchmarks`**: Production-realistic workload benchmarks

Run benchmarks with:
```bash
# Run all benchmarks (generates HTML reports in target/criterion/)
cargo bench

# Run specific benchmark suite
cargo bench --bench fusion_benchmarks

# Run with baseline comparison
cargo bench --bench performance_validation -- --save-baseline main
```

### Property-Based Testing

The crate includes comprehensive property-based tests using `proptest`:

```bash
# Run property-based tests
cargo test --test property_tests

# Run with more iterations for thorough validation
PROPTEST_CASES=10000 cargo test --test property_tests
```

### Integration Tests

Run all integration tests including fusion system validation:

```bash
# Run all integration tests
cargo test --tests

# Run specific integration test
cargo test --test fusion_integration
```

## Developer Resources

### Getting Started Guide

For a comprehensive guide to getting started with voirs-acoustic, see [GETTING_STARTED.md](GETTING_STARTED.md). This guide covers:

- Quick start and hello world examples
- Core concepts and architecture
- Basic and advanced usage patterns
- Performance optimization strategies
- Production deployment best practices
- Troubleshooting common issues

### Production-Ready Features

#### Error Handling and Diagnostics

The crate provides sophisticated error handling with structured diagnostics:

```rust
use voirs_acoustic::error::ErrorContextBuilder;

// Get detailed error context with recovery suggestions
let ctx = ErrorContextBuilder::inference_error(
    "synthesis",
    "[100 phonemes]",
    "Shape mismatch",
);

// Print comprehensive error report
println!("{}", ctx.format_report());
// Output includes:
// - Error severity and category
// - Context information
// - Performance impact estimate
// - Prioritized recovery suggestions
// - Related error chains
```

#### Performance Monitoring

Built-in production monitoring for tracking synthesis performance:

```bash
# Run production monitoring example
cargo run --example production_monitoring_simple --features candle
```

Features:
- Real-time performance metrics (RTF, latency, throughput)
- Success/failure rate tracking
- Performance alert system
- Detailed error diagnostics with recovery actions

#### Memory Optimization

Advanced memory management strategies for different deployment scenarios:

```bash
# Run memory optimization examples
cargo run --example memory_optimization_advanced --features candle
```

Strategies demonstrated:
- Memory pooling and reuse
- Streaming synthesis for long sequences
- Adaptive batch sizing
- Memory pressure handling

### Testing Infrastructure

The crate includes comprehensive testing:

- **746 total tests** across all test suites:
  - 605 unit tests
  - 55 property-based tests (robustness validation) - **10 NEW optimization property tests!**
  - 29 fusion integration tests
  - 16 multi-language integration tests
  - 12 optimization integration tests (NEW!)
  - 10 SciRS2 integration tests
  - 10 conditioning tests
  - 9 production integration tests

Run property-based tests with custom iteration counts:

```bash
# Run with more iterations for thorough testing
PROPTEST_CASES=10000 cargo test --test property_tests
```

### Code Quality Standards

- ✅ **Zero warnings policy**: All code must compile without warnings
- ✅ **100% SciRS2 compliance**: No direct use of rand/ndarray/etc.
- ✅ **Refactoring compliance**: No single file exceeds 2000 lines
- ✅ **Comprehensive documentation**: All public APIs documented with examples

## Contributing

We welcome contributions! Please see the [main repository](https://github.com/cool-japan/voirs) for contribution guidelines.

### Development Setup

```bash
git clone https://github.com/cool-japan/voirs.git
cd voirs/crates/voirs-acoustic

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

### Adding New Models

1. Implement the `AcousticModel` trait
2. Add model configuration structures
3. Create model loading and inference logic
4. Add comprehensive tests and benchmarks
5. Update documentation and examples

## License

Licensed under the Apache License, Version 2.0 ([LICENSE](../../LICENSE)).
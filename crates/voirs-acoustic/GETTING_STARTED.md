# Getting Started with voirs-acoustic

Welcome to voirs-acoustic! This guide will help you get up and running quickly with neural acoustic modeling for text-to-speech synthesis.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Concepts](#core-concepts)
3. [Basic Usage](#basic-usage)
4. [Advanced Features](#advanced-features)
5. [Performance Optimization](#performance-optimization)
6. [Production Deployment](#production-deployment)
7. [Examples](#examples)
8. [Troubleshooting](#troubleshooting)

## Quick Start

### Installation

Add voirs-acoustic to your `Cargo.toml`:

```toml
[dependencies]
voirs-acoustic = "0.1.0"
voirs-g2p = "0.1.0"
tokio = { version = "1.47", features = ["full"] }
```

### Hello World

```rust
use voirs_acoustic::{Phoneme, SynthesisConfig, MelSpectrogram};
use voirs_g2p::{G2p, OpenJTalkG2p}; // For Japanese

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Convert text to phonemes using G2P
    let g2p = OpenJTalkG2p::new()?;
    let phonemes = g2p.convert("こんにちは").await?;

    // 2. Create synthesis configuration
    let config = SynthesisConfig::default();

    // 3. Synthesize mel spectrogram (would use actual model in production)
    // let model = VitsModel::from_pretrained("vits-ja-jp").await?;
    // let mel: MelSpectrogram = model.synthesize(&phonemes, Some(&config)).await?;

    println!("Ready to synthesize {} phonemes!", phonemes.len());
    Ok(())
}
```

## Core Concepts

### Architecture Overview

```
Text → G2P → Phonemes → Acoustic Model → Mel Spectrogram → Vocoder → Audio
                          ↑ (voirs-acoustic handles this)
```

### Key Components

1. **Phonemes**: Linguistic representation of sounds
2. **Acoustic Models**: Neural networks that generate mel spectrograms (VITS, FastSpeech2)
3. **Mel Spectrograms**: Frequency-domain audio representations
4. **Synthesis Configuration**: Controls for speed, pitch, energy, emotion, etc.

### Supported Backends

- **Candle** (default): Pure Rust ML framework with GPU support
- **ONNX Runtime**: Cross-platform ML runtime with broad hardware support

## Basic Usage

### Creating Phoneme Sequences

```rust
use voirs_acoustic::Phoneme;

// Manual phoneme creation
let phonemes = vec![
    Phoneme::new("HH"),
    Phoneme::new("EH"),
    Phoneme::new("L"),
    Phoneme::new("OW"),
];

// With duration and features
let mut phoneme = Phoneme::new("AH");
phoneme.duration = Some(0.15); // 150ms
phoneme.features = Some(vec![0.5, 0.3, 0.8]); // Custom features
```

### Synthesis Configuration

```rust
use voirs_acoustic::SynthesisConfig;

// Default configuration
let config = SynthesisConfig::default();

// Custom configuration
let config = SynthesisConfig {
    speed: 1.2,           // 20% faster
    pitch_shift: 2.0,     // 2 semitones higher
    energy: 1.3,          // 30% louder
    ..Default::default()
};

// Emotion control (if model supports it)
let config = SynthesisConfig {
    emotion_type: Some("happy".to_string()),
    emotion_intensity: 0.8,
    ..Default::default()
};
```

### Working with Mel Spectrograms

```rust
use voirs_acoustic::MelSpectrogram;

// Create a mel spectrogram
let n_mels = 80;
let n_frames = 256;
let data = vec![vec![0.0; n_frames]; n_mels];
let mel = MelSpectrogram::new(data, 22050, 256);

// Access properties
println!("Dimensions: {}x{}", mel.n_mels(), mel.n_frames());
println!("Duration: {:.2}s", mel.duration());
println!("Sample rate: {}", mel.sample_rate());

// Convert to audio samples (if vocoder is available)
// let audio: Vec<f32> = vocoder.synthesize(&mel).await?;
```

## Advanced Features

### Emotion Control

```rust
use voirs_acoustic::SynthesisConfig;

// Basic emotion control
let happy_config = SynthesisConfig {
    emotion_type: Some("happy".to_string()),
    emotion_intensity: 0.8,
    ..Default::default()
};

// Blended emotions
let config = SynthesisConfig {
    emotion_type: Some("happy+excited".to_string()),
    emotion_intensity: 0.6,
    ..Default::default()
};
```

### Multi-Speaker Synthesis

```rust
// Speaker selection
let config = SynthesisConfig {
    speaker_id: Some(42),
    ..Default::default()
};

// Speaker morphing (blend between speakers)
let config = SynthesisConfig {
    speaker_id: Some(10),
    speaker_mix: vec![(10, 0.7), (20, 0.3)], // 70% speaker 10, 30% speaker 20
    ..Default::default()
};
```

### Streaming Synthesis

```rust
use voirs_acoustic::streaming::StreamingConfig;

// Configure streaming
let stream_config = StreamingConfig {
    chunk_frames: 256,
    overlap_frames: 64,
    max_latency_ms: 100,
    enable_realtime_streaming: true,
    ..Default::default()
};

// Process in streaming mode
// let stream = model.synthesize_streaming(&phonemes, &stream_config).await?;
// while let Some(chunk) = stream.next().await {
//     // Process each chunk as it becomes available
//     play_audio(&chunk);
// }
```

### Voice Cloning

```rust
use voirs_acoustic::speaker::cloning::VoiceCloningConfig;

// Configure voice cloning
let cloning_config = VoiceCloningConfig {
    num_reference_samples: 10,
    adaptation_strength: 0.7,
    enable_cross_lingual: true,
    ..Default::default()
};

// Clone voice from reference audio
// let cloned_speaker_id = model.clone_voice(&reference_audio, &cloning_config).await?;
```

## Performance Optimization

### Memory Optimization

```rust
use voirs_acoustic::memory::{MemoryPoolConfig};

// Configure memory pooling
let pool_config = MemoryPoolConfig {
    initial_pool_size_mb: 128,
    max_pool_size_mb: 512,
    enable_dynamic_growth: true,
    growth_factor: 1.5,
    ..Default::default()
};

// For systems with limited memory (<2GB RAM)
let conservative_config = MemoryPoolConfig {
    initial_pool_size_mb: 64,
    max_pool_size_mb: 256,
    enable_dynamic_growth: true,
    growth_factor: 1.2,
    ..Default::default()
};
```

### Result Caching

```rust
use voirs_acoustic::synthesis_cache::{SynthesisCache, SynthesisCacheConfig};

// Create cache
let cache_config = SynthesisCacheConfig {
    max_entries: 1000,
    max_size_mb: 512,
    ttl: std::time::Duration::from_secs(3600),
    eviction_policy: EvictionPolicy::LRU,
};

let cache = SynthesisCache::new(cache_config);

// Cache synthesis results
// let mel = model.synthesize(&phonemes, &config).await?;
// cache.insert(cache_key, mel.clone())?;
```

### GPU Acceleration

```toml
# Enable GPU support in Cargo.toml
[dependencies]
voirs-acoustic = { version = "0.1.0", features = ["gpu", "metal"] } # macOS
# Or for CUDA:
# voirs-acoustic = { version = "0.1.0", features = ["gpu", "cuda"] }
```

```rust
use voirs_acoustic::config::RuntimeConfig;

// Configure GPU usage
let runtime_config = RuntimeConfig {
    device: "cuda:0".to_string(), // Or "metal", "cpu"
    num_threads: 8,
    enable_gpu: true,
    ..Default::default()
};
```

## Production Deployment

### Error Handling

```rust
use voirs_acoustic::error::{ErrorContextBuilder, AcousticError};

async fn synthesize_with_error_handling(
    phonemes: &[Phoneme],
) -> Result<MelSpectrogram, AcousticError> {
    // Validate input
    if phonemes.is_empty() {
        let ctx = ErrorContextBuilder::input_validation_error(
            "phonemes",
            "0",
            "at least 1 phoneme",
        );
        eprintln!("{}", ctx.format_report());
        return Err(AcousticError::InputError {
            message: "Empty phoneme sequence".to_string(),
        });
    }

    // Attempt synthesis with recovery
    match model.synthesize(phonemes, None).await {
        Ok(mel) => Ok(mel),
        Err(e) => {
            let ctx = ErrorContextBuilder::inference_error(
                "synthesis",
                &format!("[{} phonemes]", phonemes.len()),
                &e.to_string(),
            );
            eprintln!("Error Report:\n{}", ctx.format_report());
            Err(e)
        }
    }
}
```

### Performance Monitoring

```rust
use voirs_acoustic::production_monitoring::{ProductionMetrics, PerformanceAlert};
use std::time::Instant;

// Track synthesis performance
let start = Instant::now();
let mel = model.synthesize(&phonemes, &config).await?;
let duration = start.elapsed();

// Calculate RTF (Real-Time Factor)
let audio_duration = mel.duration();
let rtf = duration.as_secs_f32() / audio_duration;

if rtf > 0.5 {
    eprintln!("WARNING: RTF ({:.2}x) exceeds target (< 0.3x)", rtf);
}

// Log metrics
println!("Synthesis: {:.2}ms (RTF: {:.3}x)",
         duration.as_millis(), rtf);
```

### Batch Processing

```rust
use voirs_acoustic::batching::{BatchProcessor, BatchConfig};

// Configure batch processing
let batch_config = BatchConfig {
    max_batch_size: 32,
    max_sequence_length: 500,
    enable_dynamic_batching: true,
    ..Default::default()
};

// Process multiple inputs efficiently
let inputs = vec![phonemes1, phonemes2, phonemes3];
let results = batch_processor.process_batch(&inputs, &config).await?;
```

## Examples

The crate includes comprehensive examples demonstrating various use cases:

### Run Examples

```bash
# Production monitoring and error handling
cargo run --example production_monitoring_simple --features candle

# Advanced memory optimization strategies
cargo run --example memory_optimization_advanced --features candle

# Fusion optimization for performance
cargo run --example fusion_optimization_demo --features candle

# Performance profiling
cargo run --example profiling_demo --features candle
```

### Available Examples

1. **production_monitoring_simple** - Error handling and metrics tracking
2. **memory_optimization_advanced** - Memory-efficient synthesis techniques
3. **fusion_optimization_demo** - Kernel fusion for GPU optimization
4. **profiling_demo** - Performance profiling and analysis
5. **scirs2_optimization_demo** - SciRS2-specific optimizations
6. **advanced_features_demo** - Emotion, cloning, and multi-speaker features

## Troubleshooting

### Common Issues

#### Issue: Out of Memory Errors

**Solution:**
- Reduce batch size
- Enable streaming synthesis for long sequences
- Use memory pooling with conservative settings
- Process shorter sequences

```rust
// Example: Reduce batch size
let config = BatchConfig {
    max_batch_size: 8, // Reduced from 32
    ..Default::default()
};
```

#### Issue: Slow Synthesis (High RTF)

**Solution:**
- Enable GPU acceleration if available
- Use result caching for repeated inputs
- Enable kernel fusion optimization
- Reduce quality settings if acceptable

```rust
// Example: Enable caching
let cache = SynthesisCache::with_capacity(1000);
```

#### Issue: Model Loading Fails

**Solution:**
- Verify model file exists and is accessible
- Check model format (.safetensors or .onnx)
- Ensure sufficient memory for model loading
- Verify file permissions

```rust
// Example: Check file before loading
if !std::path::Path::new("/path/to/model.safetensors").exists() {
    eprintln!("Model file not found!");
}
```

### Performance Tips

1. **Use GPU when available**: Can provide 5-10x speedup
2. **Enable caching**: Eliminates redundant computation
3. **Batch similar requests**: Improves GPU utilization
4. **Use streaming for long sequences**: Reduces memory usage
5. **Profile before optimizing**: Identify actual bottlenecks

### Debug Logging

```rust
// Enable detailed logging
use tracing_subscriber;

tracing_subscriber::fmt()
    .with_max_level(tracing::Level::DEBUG)
    .with_target(false)
    .init();
```

## Next Steps

- Explore the [API documentation](https://docs.rs/voirs-acoustic)
- Check out the [examples directory](examples/)
- Run the [benchmarks](benches/) to understand performance
- Read the [advanced features guide](README.md#advanced-features)
- Join the community and contribute!

## Getting Help

- **Documentation**: https://docs.rs/voirs-acoustic
- **Examples**: See `examples/` directory
- **Issues**: Report bugs and request features on GitHub
- **Discussions**: Join community discussions

## Contributing

We welcome contributions! Areas where you can help:

- Adding support for new acoustic models
- Improving performance optimizations
- Writing more examples and tutorials
- Enhancing documentation
- Reporting bugs and issues

---

**Happy Synthesizing!** 🎤🔊

For more information, see the main [README.md](README.md) and [API documentation](https://docs.rs/voirs-acoustic).

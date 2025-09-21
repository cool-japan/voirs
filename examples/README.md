# VoiRS Examples

> **Comprehensive Examples and Tutorials for the VoiRS Neural Speech Synthesis Ecosystem**

This directory contains practical examples demonstrating the capabilities of VoiRS across all major use cases, from basic text-to-speech to advanced real-time applications.

## 🎯 Overview

The examples are organized by complexity and use case, making it easy to find the right starting point for your application. Each example includes detailed comments, error handling, and performance considerations.

## 🚀 Quick Start Examples

### Basic Text-to-Speech

- **[`simple_synthesis.rs`](simple_synthesis.rs)** - Basic text-to-speech synthesis
  - Single-voice synthesis
  - Audio file output
  - Error handling
  - Perfect for getting started

- **[`ssml_synthesis.rs`](ssml_synthesis.rs)** - SSML markup support
  - Speech Synthesis Markup Language
  - Prosody control
  - Voice selection
  - Advanced text processing

### Batch Processing

- **[`batch_synthesis.rs`](batch_synthesis.rs)** - Process multiple texts efficiently
  - Parallel processing
  - Progress reporting
  - Quality metrics
  - Output management

- **[`batch_evaluation_comparison.rs`](batch_evaluation_comparison.rs)** - Compare synthesis quality
  - A/B testing framework
  - Quality metrics comparison
  - Statistical analysis
  - Report generation

## 🎪 Advanced Features

### Real-time Applications

- **[`streaming_synthesis.rs`](streaming_synthesis.rs)** - Real-time streaming synthesis
  - Low-latency processing
  - Chunk-based synthesis
  - Buffer management
  - Live audio output

- **[`realtime_voice_coach.rs`](realtime_voice_coach.rs)** - Interactive voice coaching
  - Real-time feedback
  - Progress tracking
  - Adaptive learning
  - Gamification elements

### Emotion and Expression

- **[`emotion_control_example.rs`](emotion_control_example.rs)** - Emotion-based synthesis
  - Emotion state control
  - Prosody modification
  - Expression interpolation
  - Dynamic emotion changes

- **[`emotion_control_example_fixed.rs`](emotion_control_example_fixed.rs)** - Enhanced emotion control
  - Bug fixes and improvements
  - Advanced emotion mapping
  - Real-time emotion adaptation
  - Quality optimizations

### Voice Cloning and Conversion

- **[`voice_cloning_example.rs`](voice_cloning_example.rs)** - Basic voice cloning
  - Few-shot learning
  - Speaker adaptation
  - Quality assessment
  - Ethical considerations

- **[`voice_cloning_example_fixed.rs`](voice_cloning_example_fixed.rs)** - Enhanced cloning
  - Improved algorithms
  - Cross-lingual support
  - Real-time adaptation
  - Quality metrics

- **[`voice_conversion_example.rs`](voice_conversion_example.rs)** - Voice-to-voice conversion
  - Real-time conversion
  - Age/gender transformation
  - Style transfer
  - Quality preservation

### Specialized Synthesis

- **[`singing_synthesis_example.rs`](singing_synthesis_example.rs)** - Singing voice synthesis
  - Musical note processing
  - MIDI integration
  - Vibrato and effects
  - Multi-voice harmony

- **[`spatial_audio_example.rs`](spatial_audio_example.rs)** - 3D spatial audio
  - 3D positioning
  - HRTF processing
  - Room acoustics
  - VR/AR integration

## 🔍 Production Examples

### Complete Pipelines

- **[`complete_pipeline.rs`](complete_pipeline.rs)** - End-to-end synthesis pipeline
  - Full processing chain
  - Quality control
  - Performance monitoring
  - Production deployment

- **[`complete_voice_pipeline.rs`](complete_voice_pipeline.rs)** - Complete voice processing
  - Multi-stage processing
  - Quality assurance
  - Monitoring and logging
  - Error recovery

### Speech Recognition Integration

- **[`production_whisper_example.rs`](production_whisper_example.rs)** - Production-ready ASR
  - Whisper integration
  - Real-time transcription
  - Quality optimization
  - Performance tuning

## 📚 Usage Instructions

### Running Examples

```bash
# Run a specific example
cargo run --example simple_synthesis

# Run with release optimizations
cargo run --release --example streaming_synthesis

# Run with specific features
cargo run --example voice_cloning_example --features="cloning,gpu"

# Run with environment variables
VOIRS_LOG_LEVEL=debug cargo run --example realtime_voice_coach
```

### Example Categories

#### 🔰 Beginner (Green)
- `simple_synthesis.rs`
- `ssml_synthesis.rs`
- `batch_synthesis.rs`

#### 🔶 Intermediate (Yellow)
- `streaming_synthesis.rs`
- `emotion_control_example.rs`
- `voice_cloning_example.rs`
- `singing_synthesis_example.rs`

#### 🔴 Advanced (Red)
- `realtime_voice_coach.rs`
- `complete_pipeline.rs`
- `spatial_audio_example.rs`
- `production_whisper_example.rs`

## 🔧 Configuration Examples

### Basic Configuration

```rust
// Simple setup for most examples
use voirs::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize with default configuration
    let pipeline = VoirsPipeline::builder()
        .with_voice("en-US-female-natural")
        .with_quality(QualityLevel::Standard)
        .build()
        .await?;

    let audio = pipeline
        .synthesize("Hello from VoiRS!")
        .await?;

    audio.save("output.wav")?;
    Ok(())
}
```

### Advanced Configuration

```rust
// Production-ready setup
use voirs::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Advanced configuration with monitoring
    let pipeline = VoirsPipeline::builder()
        .with_voice(VoiceConfig {
            name: "en-US-female-natural".to_string(),
            emotion: EmotionState::neutral(),
            speed: 1.0,
            pitch: 1.0,
        })
        .with_quality(QualityLevel::Production)
        .with_gpu_acceleration(true)
        .with_real_time_factor_target(0.1)
        .with_monitoring(true)
        .build()
        .await?;

    let result = pipeline
        .synthesize_with_options(SynthesisOptions {
            text: "Advanced VoiRS synthesis".to_string(),
            output_format: AudioFormat::WAV,
            sample_rate: 48000,
            streaming: false,
        })
        .await?;

    println!("Synthesis metrics: {:?}", result.metrics);
    Ok(())
}
```

## 🎪 Feature Demonstrations

### Emotion Control

```bash
# Try different emotions
cargo run --example emotion_control_example -- --emotion happy --intensity high
cargo run --example emotion_control_example -- --emotion sad --intensity medium
cargo run --example emotion_control_example -- --emotion excited --intensity low
```

### Voice Cloning

```bash
# Clone a voice from samples
cargo run --example voice_cloning_example -- \
  --samples "sample1.wav,sample2.wav,sample3.wav" \
  --target-text "Hello, this is my cloned voice" \
  --output "cloned_output.wav"
```

### Real-time Synthesis

```bash
# Start real-time synthesis server
cargo run --example streaming_synthesis -- --port 8080 --buffer-size 256

# Then connect with:
# echo "Hello world" | curl -X POST -d @- http://localhost:8080/synthesize
```

## 📊 Performance Examples

### Benchmarking

```rust
// Performance measurement example
use std::time::Instant;
use voirs::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let pipeline = VoirsPipeline::default().await?;
    
    let start = Instant::now();
    let audio = pipeline.synthesize("Performance test").await?;
    let duration = start.elapsed();
    
    let rtf = duration.as_secs_f32() / audio.duration_seconds();
    println!("Real-time factor: {:.3}×", rtf);
    
    Ok(())
}
```

### Memory Usage

```rust
// Memory monitoring example
use voirs::monitoring::*;

#[tokio::main]
async fn main() -> Result<()> {
    let monitor = MemoryMonitor::new();
    let pipeline = VoirsPipeline::default().await?;
    
    monitor.start_monitoring();
    let audio = pipeline.synthesize("Memory test").await?;
    let memory_stats = monitor.get_stats();
    
    println!("Peak memory usage: {:.1} MB", memory_stats.peak_mb);
    println!("Average memory usage: {:.1} MB", memory_stats.average_mb);
    
    Ok(())
}
```

## 🧪 Testing Examples

### Running Example Tests

```bash
# Test all examples
cargo test --examples

# Test specific example
cargo test --example simple_synthesis

# Test with specific features
cargo test --examples --features="gpu,all-models"

# Integration tests for examples
cargo test --test examples_integration
```

### Quality Validation

```bash
# Validate example outputs
cd examples
python validate_outputs.py --check-quality --check-format

# Generate reference outputs
cargo run --example batch_synthesis -- --generate-references
```

## 🔗 Integration Examples

### Web Integration

```bash
# WebAssembly example
cargo build --example web_synthesis --target wasm32-unknown-unknown
wasm-pack build --target web --out-dir www examples/web_synthesis
```

### Python Integration

```bash
# Python binding example
cd examples/python
pip install maturin
maturin develop
python python_synthesis_example.py
```

### C Integration

```bash
# C FFI example
cd examples/c
make
./c_synthesis_example
```

## 📝 Documentation

Each example includes:

- **Inline documentation** - Detailed code comments
- **README sections** - Usage instructions and explanations
- **Error handling** - Robust error handling patterns
- **Performance notes** - Optimization tips and considerations
- **Related examples** - Links to similar or advanced examples

## 🎓 Learning Path

### Recommended Learning Sequence

1. **Start Simple** - `simple_synthesis.rs`
2. **Add Markup** - `ssml_synthesis.rs`
3. **Batch Processing** - `batch_synthesis.rs`
4. **Real-time** - `streaming_synthesis.rs`
5. **Advanced Features** - `emotion_control_example.rs`
6. **Specialization** - Choose voice cloning, singing, or spatial audio
7. **Production** - `complete_pipeline.rs`

### Skills Development

- **Basic TTS** - Synthesis fundamentals
- **Quality Control** - Evaluation and optimization
- **Real-time Processing** - Streaming and performance
- **Advanced Features** - Emotion, cloning, spatial audio
- **Production Deployment** - Monitoring, scaling, reliability

## ⚠️ Prerequisites

### System Requirements

```bash
# Install required dependencies
sudo apt install libasound2-dev  # Linux audio
brew install portaudio           # macOS audio

# For GPU examples
cuda-toolkit                     # NVIDIA CUDA
rocm                            # AMD ROCm
```

### Rust Dependencies

```toml
[dependencies]
voirs = { version = "0.1", features = ["full"] }
tokio = { version = "1.0", features = ["full"] }
anyhow = "1.0"
tracing = "0.1"
```

## 🚑 Troubleshooting

### Common Issues

1. **Audio Device Not Found**
   ```bash
   # List available audio devices
   cargo run --example list_audio_devices
   ```

2. **Model Download Failures**
   ```bash
   # Manually download models
   cargo run --example download_models
   ```

3. **Memory Issues**
   ```bash
   # Run with memory profiling
   VOIRS_MEMORY_PROFILE=1 cargo run --example memory_test
   ```

4. **Performance Issues**
   ```bash
   # Run with performance monitoring
   VOIRS_PERF_MONITOR=1 cargo run --example performance_test
   ```

## 📝 License

All examples are licensed under either of Apache License 2.0 or MIT License at your option, consistent with the main VoiRS project.

---

*Part of the [VoiRS](../README.md) neural speech synthesis ecosystem.*
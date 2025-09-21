# VoiRS Recognizer

[![Crates.io](https://img.shields.io/crates/v/voirs-recognizer)](https://crates.io/crates/voirs-recognizer)
[![Documentation](https://docs.rs/voirs-recognizer/badge.svg)](https://docs.rs/voirs-recognizer)
[![License](https://img.shields.io/crates/l/voirs-recognizer)](LICENSE)

**Automatic Speech Recognition (ASR) and phoneme alignment for VoiRS**

VoiRS Recognizer provides comprehensive speech recognition capabilities for the VoiRS ecosystem, enabling accurate transcription, phoneme alignment, and audio analysis for speech synthesis evaluation and training.

## Features

### 🎤 Multi-Model ASR Support
- **OpenAI Whisper**: State-of-the-art multilingual speech recognition
- **Mozilla DeepSpeech**: Privacy-focused local speech recognition
- **Facebook Wav2Vec2**: Self-supervised speech representation learning
- **Custom Models**: Plugin architecture for additional ASR backends

### 🔤 Phoneme Recognition & Alignment
- **Forced Alignment**: Precise time-aligned phoneme segmentation
- **Montreal Forced Alignment (MFA)**: Professional-grade phoneme alignment
- **Custom Phoneme Sets**: Support for multiple languages and dialects
- **Confidence Scoring**: Reliability metrics for recognition results

### 📊 Audio Analysis
- **Quality Assessment**: SNR, THD, and spectral analysis
- **Prosody Analysis**: Pitch, rhythm, stress, and intonation
- **Speaker Characteristics**: Gender, age, emotion detection
- **Artifact Detection**: Clipping, distortion, and noise identification

## Quick Start

Add VoiRS Recognizer to your `Cargo.toml`:

```toml
[dependencies]
voirs-recognizer = "0.1.0"

# Enable specific ASR models
voirs-recognizer = { version = "0.1.0", features = ["whisper", "forced-align"] }
```

### Basic Speech Recognition

```rust
use voirs_recognizer::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize ASR system with Whisper
    let asr = WhisperASR::new().await?;
    
    // Load audio file
    let audio = AudioBuffer::from_file("speech.wav")?;
    
    // Recognize speech
    let transcript = asr.recognize(&audio, None).await?;
    println!("Transcript: {}", transcript.text);
    
    // Get word-level timestamps
    for word in &transcript.words {
        println!("{}: {:.2}s - {:.2}s", word.word, word.start, word.end);
    }
    
    Ok(())
}
```

### Phoneme Alignment

```rust
use voirs_recognizer::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize phoneme recognizer
    let recognizer = MFARecognizer::new().await?;
    
    // Align audio with text
    let audio = AudioBuffer::from_file("speech.wav")?;
    let text = "Hello world, this is a test.";
    
    let alignment = recognizer.align_phonemes(&audio, text, None).await?;
    
    // Print phoneme-level alignment
    for phoneme in &alignment.phonemes {
        println!("{}: {:.3}s - {:.3}s (confidence: {:.2})", 
                 phoneme.phoneme.symbol, 
                 phoneme.start_time, 
                 phoneme.end_time,
                 phoneme.confidence);
    }
    
    Ok(())
}
```

### Audio Quality Analysis

```rust
use voirs_recognizer::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize audio analyzer
    let analyzer = AudioAnalyzer::new().await?;
    
    // Analyze audio quality
    let audio = AudioBuffer::from_file("speech.wav")?;
    let analysis = analyzer.analyze_quality(&audio, None).await?;
    
    println!("SNR: {:.1} dB", analysis.snr);
    println!("THD: {:.2}%", analysis.thd);
    println!("Spectral centroid: {:.1} Hz", analysis.spectral_centroid);
    
    // Check for artifacts
    if analysis.clipping_detected {
        println!("⚠️  Audio clipping detected");
    }
    
    Ok(())
}
```

## Supported ASR Models

### Whisper
- **Languages**: 99+ languages supported
- **Model Sizes**: tiny, base, small, medium, large
- **Features**: Multilingual, robust to noise, timestamp accuracy
- **Use Case**: General-purpose, multilingual applications

### DeepSpeech
- **Languages**: English (primary), with community models for other languages  
- **Features**: Local processing, privacy-focused, customizable
- **Use Case**: Privacy-sensitive applications, offline deployment

### Wav2Vec2
- **Languages**: English, with multilingual variants available
- **Features**: Self-supervised learning, fine-tunable
- **Use Case**: Research applications, custom domain adaptation

## Feature Flags

Enable specific functionality through feature flags:

```toml
[dependencies]
voirs-recognizer = { 
    version = "0.1.0", 
    features = [
        "whisper",      # OpenAI Whisper support
        "deepspeech",   # Mozilla DeepSpeech support  
        "wav2vec2",     # Facebook Wav2Vec2 support
        "forced-align", # Basic forced alignment
        "mfa",          # Montreal Forced Alignment
        "all-models",   # Enable all ASR models
        "gpu",          # GPU acceleration support
    ]
}
```

## Performance Optimization

VoiRS Recognizer is designed to meet strict performance requirements:
- **Real-time factor (RTF) < 0.3** on modern CPUs
- **Memory usage < 2GB** for largest models
- **Startup time < 5 seconds**
- **Streaming latency < 200ms**

### Performance Validation

Use the built-in performance validator to ensure your configuration meets requirements:

```rust
use voirs_recognizer::prelude::*;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let validator = PerformanceValidator::new().with_verbose(true);
    
    // Set custom requirements if needed
    let requirements = PerformanceRequirements {
        max_rtf: 0.25,                    // Stricter than default 0.3
        max_memory_usage: 1_500_000_000,  // 1.5GB instead of 2GB
        max_startup_time_ms: 3000,        // 3 seconds instead of 5
        max_streaming_latency_ms: 150,    // 150ms instead of 200ms
    };
    
    let custom_validator = PerformanceValidator::with_requirements(requirements);
    
    // Validate your ASR system
    let audio = AudioBuffer::from_file("test_audio.wav")?;
    let startup_fn = || async {
        let _asr = WhisperASR::new().await?;
        Ok(())
    };
    
    let processing_time = Duration::from_millis(150); // Your actual processing time
    let streaming_latency = Some(Duration::from_millis(120));
    
    let validation = validator
        .validate_comprehensive(&audio, startup_fn, processing_time, streaming_latency)
        .await?;
    
    if validation.passed {
        println!("✅ All performance requirements met!");
        println!("RTF: {:.3}", validation.metrics.rtf);
        println!("Memory: {:.1} MB", validation.metrics.memory_usage as f64 / (1024.0 * 1024.0));
        println!("Throughput: {:.0} samples/sec", validation.metrics.throughput_samples_per_sec);
    } else {
        println!("❌ Performance requirements not met");
        for (test, passed) in &validation.test_results {
            println!("{}: {}", test, if *passed { "PASS" } else { "FAIL" });
        }
    }
    
    Ok(())
}
```

### Model Selection for Performance

Choose the appropriate model size based on your performance requirements:

```rust
use voirs_recognizer::prelude::*;

// Ultra-fast processing (RTF ~0.1, lower accuracy)
let fast_config = WhisperConfig {
    model_size: "tiny".to_string(),
    compute_type: ComputeType::Int8,     // Quantized for speed
    beam_size: 1,                        // Greedy decoding
    ..Default::default()
};

// Balanced performance and accuracy (RTF ~0.3)
let balanced_config = WhisperConfig {
    model_size: "base".to_string(),
    compute_type: ComputeType::Float16,
    beam_size: 3,
    ..Default::default()
};

// High accuracy (RTF ~0.8, higher latency)
let accurate_config = WhisperConfig {
    model_size: "small".to_string(),
    compute_type: ComputeType::Float32,
    beam_size: 5,
    temperature: 0.0,                    // Deterministic output
    ..Default::default()
};
```

### GPU Acceleration

Enable GPU acceleration for significant performance gains:

```rust
use voirs_recognizer::prelude::*;

// NVIDIA GPU acceleration
let cuda_config = WhisperConfig {
    model_size: "base".to_string(),
    device: Device::Cuda(0),             // Use first GPU
    compute_type: ComputeType::Float16,  // FP16 for memory efficiency
    memory_fraction: 0.8,                // Use 80% of GPU memory
    ..Default::default()
};

// Apple Silicon GPU acceleration  
let metal_config = WhisperConfig {
    model_size: "base".to_string(),
    device: Device::Metal,
    compute_type: ComputeType::Float16,
    ..Default::default()
};

// CPU with optimizations
let optimized_cpu_config = WhisperConfig {
    model_size: "tiny".to_string(),
    device: Device::Cpu,
    num_threads: num_cpus::get(),        // Use all CPU cores
    compute_type: ComputeType::Int8,     // Quantization for speed
    ..Default::default()
};
```

### Memory Optimization

Reduce memory usage with these techniques:

```rust
use voirs_recognizer::prelude::*;

// Memory-efficient configuration
let memory_config = WhisperConfig {
    model_size: "tiny".to_string(),      // Smallest model
    compute_type: ComputeType::Int8,     // 8-bit quantization
    enable_memory_pooling: true,         // Reuse memory allocations
    cache_size_mb: 100,                  // Limit cache size
    ..Default::default()
};

// Enable dynamic quantization for further memory savings
let quantized_config = WhisperConfig {
    model_size: "base".to_string(),
    compute_type: ComputeType::Dynamic,  // Dynamic quantization
    quantization_mode: QuantizationMode::Int8,
    ..Default::default()
};
```

### Real-time Processing Optimization

Configure for ultra-low latency streaming:

```rust
use voirs_recognizer::prelude::*;
use voirs_recognizer::integration::config::{StreamingConfig, LatencyMode};

// Ultra-low latency configuration
let streaming_config = StreamingConfig {
    latency_mode: LatencyMode::UltraLow,
    chunk_size: 1600,                    // 100ms chunks at 16kHz
    overlap: 400,                        // 25ms overlap
    buffer_duration: 2.0,                // 2 second buffer
    vad_enabled: true,                   // Voice activity detection
    noise_suppression: true,             // Real-time noise reduction
    echo_cancellation: false,            // Disable for lowest latency
};

// Balanced latency/accuracy configuration
let balanced_streaming = StreamingConfig {
    latency_mode: LatencyMode::Balanced,
    chunk_size: 4800,                    // 300ms chunks
    overlap: 800,                        // 50ms overlap
    buffer_duration: 5.0,                // 5 second buffer
    vad_enabled: true,
    noise_suppression: true,
    echo_cancellation: true,
};

// Create streaming ASR with optimized config
let streaming_asr = StreamingASR::with_config(streaming_config).await?;
```

### Batch Processing

Process multiple files efficiently:

```rust
use voirs_recognizer::prelude::*;

// Optimal batch processing
let batch_config = BatchProcessingConfig {
    batch_size: 8,                       // Process 8 files at once
    max_memory_usage: 1_500_000_000,     // 1.5GB memory limit
    parallel_workers: 4,                 // Use 4 worker threads
    enable_gpu_batching: true,           // Batch GPU operations
    ..Default::default()
};

let audio_files = vec![
    "file1.wav", "file2.wav", "file3.wav", "file4.wav",
    "file5.wav", "file6.wav", "file7.wav", "file8.wav",
];

// Load audio files
let audio_buffers: Vec<AudioBuffer> = audio_files
    .iter()
    .map(|path| AudioBuffer::from_file(path))
    .collect::<Result<Vec<_>, _>>()?;

// Process batch efficiently
let batch_processor = BatchProcessor::with_config(batch_config).await?;
let transcripts = batch_processor.process_batch(&audio_buffers).await?;

// Results are returned in the same order as input
for (i, transcript) in transcripts.iter().enumerate() {
    println!("File {}: {}", audio_files[i], transcript.text);
}
```

### Performance Monitoring

Monitor your application's performance in real-time:

```rust
use voirs_recognizer::prelude::*;
use std::time::Instant;

// Enable performance monitoring
let monitor = PerformanceMonitor::new()
    .with_metrics_collection(true)
    .with_real_time_reporting(true);

// Monitor ASR performance
let start = Instant::now();
let audio = AudioBuffer::from_file("speech.wav")?;

let transcript = asr.recognize(&audio, None).await?;
let processing_time = start.elapsed();

// Validate performance
let validator = PerformanceValidator::new().with_verbose(true);
let (rtf, rtf_passed) = validator.validate_rtf(&audio, processing_time);
let (memory_usage, memory_passed) = validator.estimate_memory_usage()?;

println!("Performance Metrics:");
println!("  RTF: {:.3} ({})", rtf, if rtf_passed { "✅" } else { "❌" });
println!("  Memory: {:.1} MB ({})", 
         memory_usage as f64 / (1024.0 * 1024.0),
         if memory_passed { "✅" } else { "❌" });
println!("  Processing time: {:?}", processing_time);
println!("  Audio duration: {:.2}s", audio.duration());

// Log performance metrics for analysis
monitor.log_performance_metrics(&PerformanceMetrics {
    rtf,
    memory_usage,
    startup_time_ms: 0, // Already started
    streaming_latency_ms: 0, // Not applicable for batch
    throughput_samples_per_sec: validator.calculate_throughput(audio.samples().len(), processing_time),
    cpu_utilization: (processing_time.as_secs_f32() / audio.duration() * 100.0).min(100.0),
});
```

### Platform-Specific Optimizations

#### SIMD Acceleration (Automatic)
VoiRS automatically detects and uses SIMD instructions:
- **Intel/AMD**: AVX2, AVX-512 when available
- **ARM**: NEON instructions on ARM64
- **Apple**: Apple Silicon optimizations

No manual configuration required - optimizations are applied automatically.

#### Multi-threading
Optimize thread usage for your hardware:

```rust
use voirs_recognizer::prelude::*;

// Automatic thread optimization
let thread_config = ThreadingConfig::auto_detect();

// Manual thread configuration
let manual_config = ThreadingConfig {
    num_inference_threads: num_cpus::get(),     // All cores for inference
    num_preprocessing_threads: 2,               // 2 cores for preprocessing
    enable_thread_affinity: true,               // Pin threads to cores
    prefer_performance_cores: true,             // Use P-cores on hybrid CPUs
};

let asr = WhisperASR::with_threading_config(manual_config).await?;
```

### Troubleshooting Performance Issues

Common performance problems and solutions:

#### High Memory Usage
```rust
// If memory usage exceeds limits, try:
let low_memory_config = WhisperConfig {
    model_size: "tiny".to_string(),      // Use smaller model
    compute_type: ComputeType::Int8,     // Enable quantization
    cache_size_mb: 50,                   // Reduce cache size
    enable_memory_pooling: true,         // Reuse allocations
    gradient_checkpointing: true,        // Trade compute for memory
    ..Default::default()
};
```

#### Poor RTF Performance
```rust
// If RTF is too high, try:
let fast_config = WhisperConfig {
    model_size: "tiny".to_string(),      // Smallest/fastest model
    beam_size: 1,                        // Greedy decoding
    compute_type: ComputeType::Int8,     // Quantization for speed
    enable_cuda: true,                   // GPU acceleration if available
    num_threads: num_cpus::get(),        // Use all CPU cores
    ..Default::default()
};
```

#### High Streaming Latency
```rust
// If streaming latency is too high, try:
let low_latency_config = StreamingConfig {
    latency_mode: LatencyMode::UltraLow,
    chunk_size: 800,                     // Smaller chunks (50ms at 16kHz)
    overlap: 160,                        // Minimal overlap (10ms)
    buffer_duration: 1.0,                // Smaller buffer
    preprocessing_enabled: false,        // Disable preprocessing
    ..Default::default()
};
```

## Language Support

VoiRS Recognizer supports multiple languages through its ASR backends:

| Language | Whisper | DeepSpeech | Wav2Vec2 | MFA |
|----------|---------|------------|----------|-----|
| English  | ✅      | ✅         | ✅       | ✅  |
| Spanish  | ✅      | ❌         | ❌       | ✅  |
| French   | ✅      | ❌         | ❌       | ✅  |
| German   | ✅      | ❌         | ❌       | ✅  |
| Japanese | ✅      | ❌         | ❌       | ❌  |
| Chinese  | ✅      | ❌         | ❌       | ❌  |
| Korean   | ✅      | ❌         | ❌       | ❌  |

## Configuration

### Custom ASR Configuration
```rust
use voirs_recognizer::prelude::*;

let config = ASRConfig {
    // Model selection
    preferred_models: vec![ASRBackend::Whisper, ASRBackend::DeepSpeech],
    
    // Language settings
    language: Some(LanguageCode::EnUs),
    auto_detect_language: true,
    
    // Quality settings
    enable_vad: true,           // Voice Activity Detection
    noise_suppression: true,    // Noise reduction
    
    // Performance settings
    chunk_duration_ms: 30000,  // 30 second chunks
    overlap_duration_ms: 1000, // 1 second overlap
    
    // Output settings
    include_word_timestamps: true,
    include_confidence_scores: true,
    normalize_text: true,
};

let asr = ASRSystem::with_config(config).await?;
```

### Phoneme Alignment Configuration
```rust
use voirs_recognizer::prelude::*;

let config = PhonemeConfig {
    // Alignment precision
    time_resolution_ms: 10,    // 10ms resolution
    confidence_threshold: 0.5, // Minimum confidence
    
    // Language model
    acoustic_model: "english_us".to_string(),
    pronunciation_dict: "cmudict".to_string(),
    
    // Processing options
    enable_speaker_adaptation: true,
    enable_pronunciation_variants: true,
};

let recognizer = PhonemeRecognizer::with_config(config).await?;
```

## Error Handling

VoiRS Recognizer provides comprehensive error handling:

```rust
use voirs_recognizer::prelude::*;

match asr.recognize(&audio, None).await {
    Ok(transcript) => {
        println!("Success: {}", transcript.text);
    }
    Err(ASRError::ModelNotFound { model }) => {
        eprintln!("Model not available: {}", model);
    }
    Err(ASRError::AudioTooShort { duration }) => {
        eprintln!("Audio too short: {:.1}s", duration);
    }
    Err(ASRError::LanguageNotSupported { language }) => {
        eprintln!("Language not supported: {:?}", language);
    }
    Err(e) => {
        eprintln!("Recognition failed: {}", e);
    }
}
```

## Examples

Check out the [examples](examples/) directory for more comprehensive usage examples:

- [`basic_recognition.rs`](examples/basic_recognition.rs) - Simple speech recognition
- [`phoneme_alignment.rs`](examples/phoneme_alignment.rs) - Detailed phoneme alignment
- [`audio_analysis.rs`](examples/audio_analysis.rs) - Audio quality analysis
- [`batch_processing.rs`](examples/batch_processing.rs) - Efficient batch processing
- [`streaming_recognition.rs`](examples/streaming_recognition.rs) - Real-time recognition
- [`multilingual.rs`](examples/multilingual.rs) - Multi-language support

## Benchmarks

Performance benchmarks on common datasets:

| Model | Dataset | WER | RTF | Memory |
|-------|---------|-----|-----|--------|
| Whisper-base | LibriSpeech | 5.2% | 0.3x | 1.2GB |
| DeepSpeech | CommonVoice | 8.1% | 0.8x | 800MB |
| Wav2Vec2-base | LibriSpeech | 6.4% | 0.5x | 1.0GB |

*RTF = Real Time Factor (processing time / audio duration)*

## Community Support

### 🆘 Getting Help

- **GitHub Issues**: For bug reports and feature requests
  - 🐛 [Bug Report](https://github.com/cool-japan/voirs/issues/new?template=bug_report.md)
  - ✨ [Feature Request](https://github.com/cool-japan/voirs/issues/new?template=feature_request.md)
  - 📖 [Documentation Request](https://github.com/cool-japan/voirs/issues/new?template=documentation.md)

- **GitHub Discussions**: For questions, ideas, and community chat
  - 💬 [General Discussion](https://github.com/cool-japan/voirs/discussions)
  - 🤝 [Help & Questions](https://github.com/cool-japan/voirs/discussions/categories/q-a)
  - 🎯 [Ideas & Suggestions](https://github.com/cool-japan/voirs/discussions/categories/ideas)

### 🌟 Connect with the Community

- **Discord Server**: Real-time chat and support
  - 🔗 [Join VoiRS Community Discord](https://discord.gg/voirs-community)
  - Channels: `#general`, `#help`, `#showcase`, `#development`

- **Matrix**: Bridged with Discord for matrix users
  - 🔗 [#voirs:matrix.org](https://matrix.to/#/#voirs:matrix.org)

### 📚 Learning Resources

- **Documentation**: [docs.rs/voirs-recognizer](https://docs.rs/voirs-recognizer)
- **Examples**: [GitHub Examples](https://github.com/cool-japan/voirs/tree/main/crates/voirs-recognizer/examples)
- **Tutorials**: [VoiRS Learning Hub](https://github.com/cool-japan/voirs/wiki)
- **Blog**: [Medium @voirs-dev](https://medium.com/@voirs-dev)

### 🚀 Professional Support

For commercial deployments and enterprise support:
- **Email**: support@voirs.dev
- **Consulting**: Available for integration assistance, performance optimization, and custom development

### 🎯 Roadmap & Planning

- **Project Board**: [GitHub Projects](https://github.com/orgs/cool-japan/projects/voirs)
- **Milestones**: [GitHub Milestones](https://github.com/cool-japan/voirs/milestones)
- **Changelog**: [CHANGELOG.md](../../CHANGELOG.md)

### 🏆 Recognition

- **Contributors**: [All Contributors](https://github.com/cool-japan/voirs/graphs/contributors)
- **Sponsors**: [GitHub Sponsors](https://github.com/sponsors/cool-japan)
- **Citations**: See [Citation](#citation) section for academic references

## Contributing

We welcome contributions! Please see our [Contributing Guide](../../CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/cool-japan/voirs.git
cd voirs/crates/voirs-recognizer

# Install dependencies
cargo build --all-features

# Run tests
cargo test --all-features

# Run benchmarks
cargo bench --all-features
```

## License

This project is licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](../../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](../../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Citation

If you use VoiRS Recognizer in your research, please cite:

```bibtex
@software{voirs_recognizer,
  title = {VoiRS Recognizer: Advanced Speech Recognition for Neural TTS},
  author = {Tetsuya Kitahata},
  organization = {Cool Japan Co., Ltd.},
  year = {2024},
  url = {https://github.com/cool-japan/voirs}
}
```
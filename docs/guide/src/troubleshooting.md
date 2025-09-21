# Troubleshooting

This guide helps you diagnose and resolve common issues with VoiRS Recognizer.

## Common Issues

### Installation Problems

#### "could not find system library 'alsa'" (Linux)

**Problem**: Missing ALSA development libraries on Linux.

**Solution**:
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install libasound2-dev pkg-config

# CentOS/RHEL/Fedora
sudo yum install alsa-lib-devel pkgconfig
# or
sudo dnf install alsa-lib-devel pkgconfig
```

#### "linking with 'cc' failed" (macOS)

**Problem**: Missing Xcode command line tools.

**Solution**:
```bash
xcode-select --install
```

#### "failed to run custom build command for 'some-sys'"

**Problem**: Missing C compiler or build tools.

**Solution**:
```bash
# Install build essentials
# Ubuntu/Debian
sudo apt install build-essential

# macOS
xcode-select --install

# Windows
# Install Visual Studio Build Tools
```

### Runtime Errors

#### `RecognitionError::ModelLoadError`

**Problem**: Failed to load recognition model.

**Possible Causes**:
- Insufficient memory
- Corrupted model files
- Incompatible model format

**Solutions**:
```rust
// Try a smaller model size
let config = RecognitionConfig::default()
    .with_model_size(ModelSize::Tiny);

// Check available memory
println!("Available memory: {} GB", 
         sys_info::mem_info().unwrap().avail / 1024 / 1024);

// Clear model cache
let config = RecognitionConfig::default()
    .with_clear_cache(true);
```

#### `RecognitionError::AudioError`

**Problem**: Audio processing failed.

**Common Causes**:
- Unsupported audio format
- Corrupted audio file
- Invalid sample rate

**Solutions**:
```rust
// Check audio format
let format = voirs_recognizer::audio_formats::detect_format("audio.wav")?;
println!("Detected format: {:?}", format);

// Convert to supported format
let config = AudioLoadConfig {
    target_sample_rate: Some(16000),
    force_mono: true,
    normalize: true,
    ..Default::default()
};
```

#### `RecognitionError::InferenceError`

**Problem**: Model inference failed.

**Solutions**:
```rust
// Reduce input size
let config = RecognitionConfig::default()
    .with_max_duration_seconds(30.0);

// Try different precision
let config = RecognitionConfig::default()
    .with_precision(Precision::Float16);

// Enable fallback backends
let config = RecognitionConfig::default()
    .with_fallback_enabled(true);
```

### Performance Issues

#### High Memory Usage

**Symptoms**: System becomes unresponsive, out-of-memory errors.

**Diagnostics**:
```rust
// Monitor memory usage
let monitor = MemoryMonitor::new();
monitor.start();

// Use memory profiling
let config = RecognitionConfig::default()
    .with_memory_profiling(true);
```

**Solutions**:
```rust
// Use smaller models
let config = RecognitionConfig::default()
    .with_model_size(ModelSize::Tiny);

// Enable memory compression
let config = RecognitionConfig::default()
    .with_memory_compression(true);

// Limit batch size
let config = RecognitionConfig::default()
    .with_batch_size(1);

// Enable garbage collection
let config = RecognitionConfig::default()
    .with_gc_enabled(true);
```

#### Slow Performance (High RTF)

**Symptoms**: Processing takes longer than real-time.

**Diagnostics**:
```rust
// Measure performance
let validator = PerformanceValidator::new();
let metrics = validator.benchmark(&recognizer, &audio).await?;
println!("RTF: {:.3}", metrics.rtf);
```

**Solutions**:
```rust
// Use GPU acceleration
let config = RecognitionConfig::default()
    .with_gpu_acceleration(true);

// Optimize threading
let config = RecognitionConfig::default()
    .with_cpu_threads(num_cpus::get());

// Reduce model size
let config = RecognitionConfig::default()
    .with_model_size(ModelSize::Base);

// Enable quantization
let config = RecognitionConfig::default()
    .with_quantization(Quantization::Int8);
```

#### High Latency in Real-time Processing

**Symptoms**: Delays in streaming recognition results.

**Solutions**:
```rust
// Reduce chunk size
let config = RecognitionConfig::default()
    .with_chunk_duration_ms(50);

// Disable heavy analysis
let config = AudioAnalysisConfig {
    prosody_analysis: false,
    speaker_analysis: false,
    ..Default::default()
};

// Use low-latency mode
let config = RecognitionConfig::default()
    .with_streaming_mode(StreamingMode::LowLatency);
```

### Audio Quality Issues

#### Poor Recognition Accuracy

**Symptoms**: Low confidence scores, incorrect transcriptions.

**Diagnostics**:
```rust
// Analyze audio quality
let analyzer = AudioAnalyzer::new(AudioAnalysisConfig::default()).await?;
let analysis = analyzer.analyze(&audio).await?;

if let Some(snr) = analysis.quality_metrics.get("snr") {
    if *snr < 10.0 {
        println!("Warning: Low SNR ({:.1} dB)", snr);
    }
}
```

**Solutions**:
```rust
// Enable preprocessing
let config = RecognitionConfig::default()
    .with_noise_suppression(true)
    .with_auto_gain_control(true);

// Use larger models
let config = RecognitionConfig::default()
    .with_model_size(ModelSize::Large);

// Improve audio quality
let load_config = AudioLoadConfig {
    normalize: true,
    remove_dc: true,
    ..Default::default()
};
```

#### "No speech detected" in VAD

**Problem**: Voice Activity Detection not finding speech.

**Solutions**:
```rust
// Adjust VAD sensitivity
let config = RecognitionConfig::default()
    .with_vad_sensitivity(0.3); // Lower threshold

// Check energy levels
let analysis = analyzer.analyze(&audio).await?;
if let Some(energy) = analysis.quality_metrics.get("energy") {
    println!("Audio energy: {:.3}", energy);
}

// Disable VAD if problematic
let config = RecognitionConfig::default()
    .with_vad(false);
```

### Multi-language Issues

#### Wrong Language Detection

**Problem**: Auto-detection chooses incorrect language.

**Solutions**:
```rust
// Explicitly set language
let config = RecognitionConfig::default()
    .with_language(LanguageCode::EnUs);

// Use language hints
let config = RecognitionConfig::default()
    .with_language_hints(vec![
        LanguageCode::EnUs,
        LanguageCode::EsEs,
    ]);
```

#### Mixing Languages in Same Audio

**Problem**: Audio contains multiple languages.

**Solutions**:
```rust
// Enable language switching
let config = RecognitionConfig::default()
    .with_auto_language_switching(true);

// Use multilingual models
let config = RecognitionConfig::default()
    .with_multilingual_model(true);
```

### GPU Issues

#### GPU Acceleration Not Working

**Problem**: CUDA/Metal acceleration not being used.

**Diagnostics**:
```rust
// Check GPU availability
if !voirs_recognizer::gpu::is_available() {
    println!("GPU acceleration not available");
}

// List available devices
let devices = voirs_recognizer::gpu::list_devices();
for device in devices {
    println!("GPU: {}", device.name);
}
```

**Solutions**:
```bash
# Install CUDA (NVIDIA)
# Check NVIDIA drivers
nvidia-smi

# Install CUDA toolkit
sudo apt install nvidia-cuda-toolkit

# For Metal (Apple Silicon)
# Ensure you're on macOS with Apple Silicon
system_profiler SPHardwareDataType | grep "Chip"
```

#### Out of GPU Memory

**Problem**: GPU runs out of memory during processing.

**Solutions**:
```rust
// Reduce GPU memory usage
let config = RecognitionConfig::default()
    .with_gpu_memory_fraction(0.5);

// Use smaller batch sizes
let config = RecognitionConfig::default()
    .with_batch_size(1);

// Enable memory clearing
let config = RecognitionConfig::default()
    .with_clear_gpu_cache(true);
```

## Debugging Tools

### Enable Debug Logging

```rust
// Enable detailed logging
env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug")).init();

// Or set environment variable
// RUST_LOG=debug cargo run
```

### Performance Profiling

```rust
// Built-in profiler
let config = RecognitionConfig::default()
    .with_profiling(true);

let recognizer = Recognizer::new(config).await?;
// ... use recognizer ...

let report = recognizer.profiler().generate_report();
println!("{}", report);
```

### Memory Profiling

```bash
# Using Valgrind (Linux)
valgrind --tool=massif ./your_app
ms_print massif.out.* > memory_profile.txt

# Using heaptrack (Linux)
heaptrack ./your_app
heaptrack_gui heaptrack.your_app.*

# Using Instruments (macOS)
# Run through Xcode Instruments
```

### Performance Monitoring

```rust
// Real-time monitoring
let monitor = PerformanceMonitor::new()
    .with_rtf_threshold(1.0)
    .with_memory_threshold(2_000_000_000);

monitor.start(&recognizer);

// Check alerts
if let Some(alert) = monitor.get_alerts().pop() {
    println!("Performance Alert: {}", alert.message);
}
```

## Configuration Validation

### Validate Your Configuration

```rust
// Check configuration validity
let config = RecognitionConfig::default()
    .with_model_size(ModelSize::Large)
    .with_gpu_acceleration(true);

if let Err(e) = config.validate() {
    println!("Configuration error: {}", e);
}
```

### Test Your Setup

```rust
// Run system tests
let test_suite = SystemTestSuite::new();
let results = test_suite.run_all().await;

for (test_name, result) in results {
    match result {
        Ok(_) => println!("✅ {}", test_name),
        Err(e) => println!("❌ {}: {}", test_name, e),
    }
}
```

## Getting Help

### Collect Diagnostic Information

```rust
// Generate diagnostic report
let diagnostics = voirs_recognizer::diagnostics::collect_system_info().await;
println!("System Diagnostics:\n{}", diagnostics);
```

### Enable Crash Reporting

```rust
// Enable crash reporting (optional)
let config = RecognitionConfig::default()
    .with_crash_reporting(true);
```

### Community Support

1. **GitHub Issues**: [Report bugs and request features](https://github.com/cool-japan/voirs/issues)
2. **Discussions**: [Get help from the community](https://github.com/cool-japan/voirs/discussions)
3. **Documentation**: [Check the latest docs](https://docs.rs/voirs-recognizer)

### Bug Report Template

When reporting issues, please include:

```text
## Environment
- OS: [e.g., Ubuntu 22.04, macOS 13.0, Windows 11]
- Rust version: [e.g., 1.70.0]
- VoiRS Recognizer version: [e.g., 0.1.0]
- Hardware: [e.g., CPU, GPU info]

## Configuration
```rust
// Your RecognitionConfig here
```

## Steps to Reproduce
1. [First step]
2. [Second step]
3. [...]

## Expected Behavior
[What you expected to happen]

## Actual Behavior
[What actually happened]

## Error Messages
```
[Any error messages or logs]
```

## Additional Context
[Any other relevant information]
```

## Performance Baselines

### Expected Performance Ranges

| Model Size | RTF Range | Memory Usage | Accuracy |
|------------|-----------|--------------|----------|
| Tiny       | 0.05-0.15 | 100-200MB    | 85-90%   |
| Base       | 0.15-0.25 | 200-400MB    | 90-94%   |
| Small      | 0.25-0.40 | 800MB-1.2GB  | 94-96%   |
| Medium     | 0.40-0.70 | 2-3GB        | 96-97%   |
| Large      | 0.70-1.20 | 4-6GB        | 97-98%   |

### Hardware Recommendations

| Use Case | CPU | RAM | GPU |
|----------|-----|-----|-----|
| Development | 4+ cores | 8GB | Optional |
| Production | 8+ cores | 16GB | Recommended |
| Real-time | 8+ cores | 16GB | Required |
| Batch Processing | 16+ cores | 32GB | Recommended |

If your performance falls outside these ranges, refer to the performance tuning section above.
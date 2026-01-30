# Advanced Features Guide

This guide demonstrates how to use the advanced features added to voirs-acoustic in version 0.1.0-alpha.3.

## Quick Start

Run the interactive demo to see all features in action:

```bash
cargo run --example advanced_features_demo --features candle
```

Run benchmarks to evaluate performance:

```bash
cargo bench --bench advanced_features_benchmarks
```

## Neural Audio Codec

High-quality audio compression for bandwidth-constrained deployments.

### Basic Usage

```rust
use voirs_acoustic::{NeuralCodec, NeuralCodecConfig};
use candle_core::Device;

// Create codec with default settings (6 kbps)
let device = Device::Cpu;
let config = NeuralCodecConfig::default();
let codec = NeuralCodec::new(config, &device)?;

// Encode audio waveform to discrete codes
let waveform = /* your audio tensor */;
let codes = codec.encode(&waveform)?;

// Decode back to waveform
let reconstructed = codec.decode(&codes)?;

// Check compression ratio
let samples = 16000; // 1 second at 16kHz
let ratio = codec.compression_ratio(samples);
println!("Compression: {:.1}x", ratio);
```

### Quality Presets

```rust
// Maximum quality (24 kbps, 16 codebooks)
let config = NeuralCodecConfig::high_quality();

// Low latency (3 kbps, 100 Hz frame rate)
let config = NeuralCodecConfig::low_latency();

// Low bandwidth (1.5 kbps, maximum compression)
let config = NeuralCodecConfig::low_bandwidth();
```

### Quality Metrics

```rust
use voirs_acoustic::CodecQualityMetrics;

let metrics = CodecQualityMetrics {
    snr_db: 25.5,
    pesq_score: 4.2,
    stoi_score: 0.92,
    mcd: 4.8,
    bitrate_kbps: 6.0,
    compression_ratio: 21.3,
    latency_ms: 13.3,
};

if metrics.meets_quality_threshold() {
    println!("✓ Quality acceptable");
}

println!("{}", metrics.report());
```

## Advanced Latency Optimizer

Real-time latency management for interactive applications.

### Basic Usage

```rust
use voirs_acoustic::{
    AdvancedLatencyOptimizer, LatencyBudget, ProcessingPriority
};

// Create optimizer for interactive scenario
let optimizer = AdvancedLatencyOptimizer::interactive();

// Start processing
let mut measurement = optimizer.start_measurement(ProcessingPriority::High);

// ... perform synthesis ...

// Record completion
measurement.finish(chunk_size, budget_met);
optimizer.record_measurement(measurement).await;

// Get recommended parameters
let chunk_size = optimizer.recommended_chunk_size().await;
let quality = optimizer.recommended_quality().await;

println!("Recommended: chunk_size={}, quality={:.1}%",
         chunk_size, quality * 100.0);
```

### Latency Budgets

```rust
use voirs_acoustic::LatencyBudget;

// Conversational (150ms target, 300ms max)
let budget = LatencyBudget::conversational();

// Interactive gaming (50ms target, 100ms max)
let budget = LatencyBudget::interactive();

// Broadcast quality (500ms target, 1000ms max)
let budget = LatencyBudget::broadcast();

// Custom budget
let budget = LatencyBudget {
    target_latency_ms: 100.0,
    max_latency_ms: 200.0,
    warning_threshold: 0.8,
    adaptive_quality: true,
    min_quality: 0.7,
};
```

### Statistics and Monitoring

```rust
// Get performance statistics
let stats = optimizer.get_statistics().await;

println!("Average latency: {:.2} ms", stats.avg_latency_ms);
println!("P95 latency: {:.2} ms", stats.p95_latency_ms);
println!("Budget met: {:.1}%", stats.budget_met_rate * 100.0);

// Check if under pressure
if optimizer.is_under_pressure().await {
    println!("⚠ System under latency pressure!");
}

// Full report
println!("{}", stats.report());
```

## Voice Activity Detection (VAD)

Real-time speech/silence detection for natural pause handling.

### Basic Usage

```rust
use voirs_acoustic::{VadConfig, VoiceActivityDetector};

// Create VAD for conversational speech
let config = VadConfig::conversational();
let mut vad = VoiceActivityDetector::new(config)?;

// Process audio frame
let frame: Vec<f32> = /* 512 samples */;
let activity = vad.process_frame(&frame);

match activity {
    VoiceActivity::Speech => println!("🎤 Speech detected"),
    VoiceActivity::Silence => println!("🔇 Silence detected"),
    VoiceActivity::Uncertain => println!("❓ Uncertain"),
}
```

### Buffer Processing

```rust
// Process entire audio buffer
let audio: Vec<f32> = /* your audio data */;
let segments = vad.process_buffer(&audio);

for (i, segment) in segments.iter().enumerate() {
    println!("Segment {}: {:?} ({:.2}s - {:.2}s, {:.0}ms)",
             i + 1,
             segment.activity,
             segment.start_time,
             segment.end_time,
             segment.duration_ms());
}
```

### Environment Presets

```rust
// Conversational (-35dB, adaptive)
let config = VadConfig::conversational();

// Studio quality (-50dB, fixed threshold)
let config = VadConfig::studio();

// Noisy environment (-25dB, enhanced smoothing)
let config = VadConfig::noisy();
```

## Acoustic Utilities

Professional audio processing and prosody manipulation.

### Audio Processing

```rust
use voirs_acoustic::acoustic_utils::audio;

// Normalize to RMS level
let normalized = audio::normalize_rms(&audio, 0.1);

// Normalize to peak level
let normalized = audio::normalize_peak(&audio, 0.95);

// Apply fade effects
let faded_in = audio::fade_in(&audio, 1000);  // 1000 samples
let faded_out = audio::fade_out(&audio, 1000);

// Cross-fade between segments
let crossfaded = audio::crossfade(&audio1, &audio2, 500);

// Remove DC offset
let corrected = audio::remove_dc_offset(&audio);
```

### Prosody Manipulation

```rust
use voirs_acoustic::acoustic_utils::prosody;

// Smooth prosody parameters
let pitch_values = vec![100.0, 105.0, 103.0, 107.0];
let smoothed = prosody::smooth_moving_average(&pitch_values, 3);

// Interpolate between values
let interpolated = prosody::interpolate_linear(100.0, 200.0, 50);
let smooth_interpolated = prosody::interpolate_cubic(100.0, 200.0, 50);

// Remove outliers
let cleaned = prosody::smooth_outliers(&pitch_values, 2.0);
```

### Quality-Aware Synthesis

```rust
use voirs_acoustic::acoustic_utils::quality::{QualityLevel, QualityAwareParams};

// Create parameters for quality level
let mut params = QualityAwareParams::new(QualityLevel::High);

println!("Chunk size: {}", params.chunk_size);
println!("Diffusion steps: {}", params.diffusion_steps);

// Adapt to available resources
let available_memory_gb = 4.0;
let cpu_load = 0.75;
params.adapt_to_resources(available_memory_gb, cpu_load);

println!("Adapted quality: {:.1}%", params.quality.as_float() * 100.0);
```

### Mel Spectrogram Utilities

```rust
use voirs_acoustic::acoustic_utils::mel;

// Concatenate mel spectrograms
let mels = vec![&mel1, &mel2, &mel3];
let concatenated = mel::concatenate(&mels, 80)?;  // 80 mel bands

// Apply temporal smoothing
let smoothed = mel::smooth_temporal(&mel_spec, 80, 3);  // window=3 frames
```

## Integration Example

Complete pipeline using all advanced features:

```rust
use voirs_acoustic::{
    AdvancedLatencyOptimizer, NeuralCodec, NeuralCodecConfig,
    VadConfig, VoiceActivityDetector, ProcessingPriority,
};
use voirs_acoustic::acoustic_utils::{audio, quality};
use candle_core::Device;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Setup latency optimizer
    let latency_optimizer = AdvancedLatencyOptimizer::interactive();

    // 2. Setup VAD
    let vad_config = VadConfig::conversational();
    let mut vad = VoiceActivityDetector::new(vad_config)?;

    // 3. Setup neural codec
    let device = Device::Cpu;
    let codec_config = NeuralCodecConfig::low_latency();
    let codec = NeuralCodec::new(codec_config, &device)?;

    // 4. Process audio pipeline
    let raw_audio: Vec<f32> = /* input audio */;

    // Normalize audio
    let normalized = audio::normalize_rms(&raw_audio, 0.1);

    // Detect voice activity
    let segments = vad.process_buffer(&normalized);
    println!("Detected {} segments", segments.len());

    // Process with latency tracking
    let chunk_size = latency_optimizer.recommended_chunk_size().await;
    let quality_level = latency_optimizer.recommended_quality().await;

    let quality = quality::QualityLevel::from_float(quality_level);
    let params = quality::QualityAwareParams::new(quality);

    println!("Processing with quality {:?}, chunk size {}",
             quality, chunk_size);

    // Measure performance
    let mut measurement = latency_optimizer.start_measurement(
        ProcessingPriority::High
    );

    // ... perform synthesis ...

    measurement.finish(chunk_size, true);
    latency_optimizer.record_measurement(measurement).await;

    // Get final statistics
    let stats = latency_optimizer.get_statistics().await;
    println!("\n{}", stats.report());

    Ok(())
}
```

## Performance Tips

1. **Neural Codec**: Use `low_latency()` preset for real-time applications
2. **Latency Optimizer**: Start with `interactive()` and adjust based on metrics
3. **VAD**: Use `conversational()` for general speech, `studio()` for clean audio
4. **Quality-Aware**: Let the system adapt to resources automatically
5. **Benchmarking**: Run `cargo bench` to establish baseline performance

## Troubleshooting

### High Latency

```rust
// Check if under pressure
if optimizer.is_under_pressure().await {
    // Reduce quality
    let quality = optimizer.recommended_quality().await;
    // Apply quality reduction to synthesis
}
```

### Poor VAD Performance

```rust
// Try different preset
let config = VadConfig::noisy();  // For noisy environments

// Or adjust thresholds manually
let config = VadConfig {
    energy_threshold_db: -30.0,  // Higher threshold
    adaptive_threshold: true,     // Enable adaptation
    ..Default::default()
};
```

### Codec Quality Issues

```rust
// Increase quality
let config = NeuralCodecConfig::high_quality();

// Or customize
let config = NeuralCodecConfig {
    target_bitrate: 12.0,  // 12 kbps
    num_codebooks: 12,
    ..Default::default()
};
```

## See Also

- [API Documentation](https://docs.rs/voirs-acoustic)
- [Example Code](examples/advanced_features_demo.rs)
- [Benchmarks](benches/advanced_features_benchmarks.rs)
- [Main README](../../README.md)

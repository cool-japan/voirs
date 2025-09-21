# voirs-conversion

> **Real-time Voice Conversion and Audio Transformation System**

This crate provides real-time voice conversion capabilities including speaker conversion, age/gender transformation, voice morphing, and streaming voice conversion for live applications.

## 🎭 Features

### Core Voice Conversion
- **Real-time Conversion** - Low-latency voice conversion for live applications (<50ms)
- **Speaker-to-Speaker** - Convert between different speaker identities
- **Style Transfer** - Transfer speaking style and characteristics
- **Quality Preservation** - Maintain audio quality during conversion

### Transformation Types
- **Age Transformation** - Make voices sound younger or older
- **Gender Conversion** - Convert between male and female voices
- **Pitch Modification** - Precise pitch scaling and shifting
- **Speed Adjustment** - Modify speaking rate while preserving quality
- **Voice Morphing** - Blend characteristics from multiple sources

### Streaming Support
- **Real-time Processing** - Process audio streams with minimal latency
- **Chunk-based Processing** - Efficient processing of audio chunks
- **Buffer Management** - Intelligent audio buffer handling
- **Adaptive Quality** - Adjust quality based on processing constraints

### Advanced Features
- **Cross-domain Conversion** - Convert between different audio domains
- **Prosody Preservation** - Maintain natural prosody patterns
- **Emotional Consistency** - Preserve emotional expression
- **Multi-target Conversion** - Convert to multiple targets simultaneously

## 🚀 Quick Start

### Basic Voice Conversion

```rust
use voirs_conversion::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Load source and target audio samples
    let source_audio = load_audio("source_speaker.wav").await?;
    let target_samples = vec![
        load_audio("target_sample1.wav").await?,
        load_audio("target_sample2.wav").await?,
    ];

    // Create voice converter
    let converter = VoiceConverter::builder()
        .with_conversion_type(ConversionType::SpeakerToSpeaker)
        .with_quality_mode(QualityMode::Balanced)
        .build().await?;

    // Define conversion target
    let target = ConversionTarget::from_samples(target_samples)?;
    
    // Perform conversion
    let request = ConversionRequest::new(source_audio, target)
        .with_preserve_prosody(true)
        .with_preserve_emotion(true);
    
    let result = converter.convert(request).await?;
    
    // Save converted audio
    result.save_audio("converted_voice.wav").await?;
    
    println!("Conversion quality: {:.2}", result.quality_score);
    Ok(())
}
```

### Real-time Streaming Conversion

```rust
use voirs_conversion::prelude::*;
use tokio::time::{interval, Duration};

#[tokio::main]
async fn main() -> Result<()> {
    // Create real-time converter
    let config = RealtimeConfig::builder()
        .with_chunk_size(1024)    // 64ms at 16kHz
        .with_lookahead(512)      // 32ms lookahead
        .with_max_latency(50)     // 50ms max latency
        .build()?;

    let converter = RealtimeConverter::new(config).await?;
    
    // Set conversion target
    let target_voice = load_target_voice("target.json").await?;
    converter.set_target(target_voice).await?;
    
    // Process audio stream
    let mut audio_stream = create_audio_stream().await?;
    let mut timer = interval(Duration::from_millis(64));
    
    while let Some(audio_chunk) = audio_stream.next().await {
        timer.tick().await;
        
        // Convert audio chunk in real-time
        let converted_chunk = converter
            .process_chunk(audio_chunk)
            .await?;
        
        // Output converted audio
        output_audio_chunk(converted_chunk).await?;
    }
    
    Ok(())
}
```

### Age and Gender Transformation

```rust
use voirs_conversion::transforms::*;

// Create age transformation
let age_transform = AgeTransform::builder()
    .target_age(25)           // Target age
    .current_age(45)          // Estimated current age
    .naturalness(0.8)         // Preserve naturalness
    .build()?;

// Create gender transformation
let gender_transform = GenderTransform::builder()
    .target_gender(Gender::Female)
    .source_gender(Gender::Male)
    .pitch_shift_method(PitchShiftMethod::PhaseVocoder)
    .formant_adjustment(true)
    .build()?;

// Apply transformations
let audio = load_audio("input.wav").await?;
let aged_audio = age_transform.apply(&audio).await?;
let gender_converted = gender_transform.apply(&aged_audio).await?;
```

### Voice Morphing

```rust
use voirs_conversion::transforms::*;

// Create voice morpher
let morpher = VoiceMorpher::new();

// Define morph targets with weights
let morph_targets = vec![
    (load_voice_characteristics("voice_a.json").await?, 0.4),
    (load_voice_characteristics("voice_b.json").await?, 0.6),
];

// Morph voice characteristics
let target_characteristics = morpher
    .morph_characteristics(&morph_targets)
    .await?;

// Apply morphing to audio
let morphed_audio = morpher
    .apply_morphing(&input_audio, &target_characteristics)
    .await?;
```

## 🔧 Configuration

### Conversion Types

```rust
use voirs_conversion::types::*;

// Speaker-to-speaker conversion
let speaker_config = ConversionConfig::builder()
    .conversion_type(ConversionType::SpeakerToSpeaker)
    .preserve_prosody(true)
    .preserve_emotion(true)
    .quality_mode(QualityMode::HighQuality)
    .build()?;

// Age transformation
let age_config = ConversionConfig::builder()
    .conversion_type(ConversionType::AgeTransformation {
        target_age: 30,
        naturalness_weight: 0.8,
    })
    .build()?;

// Real-time conversion
let realtime_config = ConversionConfig::builder()
    .conversion_type(ConversionType::RealtimeConversion)
    .max_latency_ms(50)
    .chunk_size(1024)
    .build()?;
```

### Audio Processing Pipeline

```rust
use voirs_conversion::processing::*;

// Create processing pipeline
let pipeline = ProcessingPipeline::builder()
    .add_stage(PreprocessingStage::NoiseReduction)
    .add_stage(ConversionStage::FeatureExtraction)
    .add_stage(ConversionStage::VoiceConversion)
    .add_stage(PostprocessingStage::QualityEnhancement)
    .with_parallel_processing(true)
    .build()?;

// Configure audio buffer
let buffer_config = AudioBufferConfig {
    sample_rate: 16000,
    channels: 1,
    chunk_size: 1024,
    overlap: 256,
    window_type: WindowType::Hann,
};
```

## 🎪 Advanced Features

### Cross-domain Conversion

```rust
use voirs_conversion::cross_domain::*;

// Convert between different audio domains
let converter = CrossDomainConverter::builder()
    .source_domain(AudioDomain::Telephone)     // 8kHz, compressed
    .target_domain(AudioDomain::Studio)        // 48kHz, high quality
    .with_super_resolution(true)
    .with_noise_suppression(true)
    .build().await?;

let enhanced_audio = converter
    .convert_domain(&low_quality_audio)
    .await?;
```

### Batch Conversion

```rust
use voirs_conversion::batch::*;

// Process multiple files in batch
let batch_processor = BatchProcessor::builder()
    .with_parallel_jobs(4)
    .with_progress_reporting(true)
    .build()?;

let batch_request = BatchConversionRequest {
    input_files: vec![
        "file1.wav".to_string(),
        "file2.wav".to_string(),
        "file3.wav".to_string(),
    ],
    output_directory: "converted/".to_string(),
    conversion_config: conversion_config,
};

let results = batch_processor
    .process_batch(batch_request)
    .await?;

for result in results {
    println!("Processed: {} -> {} (quality: {:.2})", 
             result.input_file, result.output_file, result.quality_score);
}
```

### Quality Assessment

```rust
use voirs_conversion::quality::*;

// Assess conversion quality
let assessor = ConversionQualityAssessor::new().await?;

let quality_metrics = assessor.assess(
    &original_audio,
    &converted_audio,
    &target_characteristics
).await?;

println!("Overall quality: {:.2}", quality_metrics.overall_score);
println!("Target similarity: {:.2}", quality_metrics.target_similarity);
println!("Naturalness: {:.2}", quality_metrics.naturalness);
println!("Artifact level: {:.2}", quality_metrics.artifact_score);
```

## 🔍 Performance

### Real-time Performance

| Configuration | Latency | RTF | CPU Usage | Memory |
|---------------|---------|-----|-----------|--------|
| Low Quality | 25ms | 0.15× | 15% | 200MB |
| Balanced | 35ms | 0.25× | 25% | 400MB |
| High Quality | 50ms | 0.40× | 35% | 600MB |
| Ultra Quality | 100ms | 0.60× | 50% | 800MB |

### Batch Processing Performance

```rust
use voirs_conversion::performance::*;

// Performance monitoring
let monitor = PerformanceMonitor::new();

// Optimize for your use case
let config = ConversionConfig::builder()
    .optimization_target(OptimizationTarget::Latency)  // or Quality, Throughput
    .hardware_acceleration(true)
    .memory_limit(2_000_000_000)  // 2GB
    .build()?;

// Monitor performance
monitor.start_monitoring();
let result = converter.convert(request).await?;
let metrics = monitor.get_metrics();

println!("Processing time: {:.2}s", metrics.processing_time);
println!("Memory usage: {:.1}MB", metrics.peak_memory_mb);
println!("CPU usage: {:.1}%", metrics.avg_cpu_usage);
```

## 🛡️ Quality Control

### Artifact Detection

```rust
use voirs_conversion::quality::*;

// Detect conversion artifacts
let artifact_detector = ArtifactDetector::new();

let artifacts = artifact_detector.detect(&converted_audio).await?;

for artifact in artifacts {
    println!("Artifact type: {:?}", artifact.artifact_type);
    println!("Severity: {:.2}", artifact.severity);
    println!("Time range: {:.2}s - {:.2}s", 
             artifact.start_time, artifact.end_time);
}
```

### Automatic Quality Adjustment

```rust
use voirs_conversion::adaptive::*;

// Adaptive quality based on input characteristics
let adaptive_converter = AdaptiveConverter::builder()
    .with_quality_threshold(0.8)
    .with_automatic_adjustment(true)
    .build().await?;

// Converter automatically adjusts parameters
let result = adaptive_converter.convert(request).await?;

println!("Used configuration: {:?}", result.used_config);
println!("Adaptation reason: {:?}", result.adaptation_reason);
```

## 🧪 Testing

```bash
# Run voice conversion tests
cargo test --package voirs-conversion

# Run real-time processing tests
cargo test --package voirs-conversion realtime

# Run transformation tests
cargo test --package voirs-conversion transforms

# Run quality assessment tests
cargo test --package voirs-conversion quality

# Run performance benchmarks
cargo bench --package voirs-conversion
```

## 🔗 Integration

### With Cloning Module

```rust
use voirs_conversion::cloning::*;

// Integration with voice cloning
let cloning_adapter = CloningIntegrationAdapter::new();
let cloned_voice = cloning_adapter
    .adapt_cloned_voice(&cloned_voice_model)
    .await?;

let converter = VoiceConverter::builder()
    .with_cloned_voice_target(cloned_voice)
    .build().await?;
```

### With Acoustic Models

```rust
use voirs_conversion::acoustic::*;

// Direct acoustic model integration
let acoustic_converter = AcousticModelConverter::new();
let converted_features = acoustic_converter
    .convert_acoustic_features(&features, &target_speaker)
    .await?;
```

### With Other VoiRS Crates

- **voirs-cloning** - Voice cloning for conversion targets
- **voirs-emotion** - Emotion preservation during conversion
- **voirs-acoustic** - Direct acoustic feature conversion
- **voirs-evaluation** - Conversion quality metrics
- **voirs-sdk** - High-level conversion API

## 🎓 Examples

See the [`examples/`](../../examples/) directory for comprehensive usage examples:

- [`voice_conversion_example.rs`](../../examples/voice_conversion_example.rs) - Basic conversion
- [`realtime_conversion.rs`](../../examples/realtime_conversion.rs) - Streaming conversion
- [`batch_conversion.rs`](../../examples/batch_conversion.rs) - Batch processing
- [`age_gender_transform.rs`](../../examples/age_gender_transform.rs) - Transformation effects

## 📝 License

Licensed under either of Apache License 2.0 or MIT License at your option.

---

*Part of the [VoiRS](../../README.md) neural speech synthesis ecosystem.*
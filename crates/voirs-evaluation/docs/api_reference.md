# VoiRS Evaluation API Reference

## Overview

The VoiRS Evaluation framework provides comprehensive quality assessment, pronunciation evaluation, and comparative analysis capabilities for speech synthesis systems. This document provides detailed API reference and usage guidelines.

## Core Concepts

### Quality Metrics

The framework supports multiple quality evaluation metrics:

- **MOS (Mean Opinion Score)**: Predicts subjective quality ratings (1.0-5.0 scale)
- **PESQ (Perceptual Evaluation of Speech Quality)**: ITU-T P.862 standard for quality assessment
- **STOI (Short-Time Objective Intelligibility)**: Measures speech intelligibility
- **MCD (Mel-Cepstral Distortion)**: Measures spectral differences using cepstral features
- **Naturalness**: Evaluates how natural the synthesized speech sounds
- **Intelligibility**: Assesses how easily speech can be understood
- **Speaker Similarity**: Measures similarity to reference speaker characteristics
- **Prosody Quality**: Evaluates rhythm, stress, and intonation patterns
- **Artifact Detection**: Identifies audio artifacts and distortions

### Pronunciation Metrics

- **Phoneme Accuracy**: Correctness of individual phoneme pronunciation
- **Word-level Accuracy**: Overall word pronunciation quality
- **Fluency**: Speaking rate and rhythm consistency
- **Prosody**: Stress patterns, intonation, and emphasis
- **Completeness**: Detection of omissions, insertions, and substitutions

### Comparison Metrics

- **Overall Quality**: Weighted combination of multiple quality aspects
- **Statistical Significance**: Determines if differences are statistically meaningful
- **Ranking**: Orders systems from best to worst performance
- **Confidence Intervals**: Provides uncertainty estimates for comparisons

## Main Components

### QualityEvaluator

The primary interface for quality assessment.

```rust
use voirs_evaluation::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let evaluator = QualityEvaluator::new().await?;
    
    let audio = AudioBuffer::new(samples, sample_rate, channels);
    let result = evaluator.evaluate_quality(&audio, None, None).await?;
    
    println!("Quality Score: {:.3}", result.overall_score);
    Ok(())
}
```

#### Key Methods

- `evaluate_quality()`: Evaluate a single audio sample
- `evaluate_quality_batch()`: Process multiple samples efficiently
- `supported_metrics()`: Get list of available metrics
- `requires_reference()`: Check if a metric needs reference audio
- `metadata()`: Get evaluator information and capabilities

#### Configuration

```rust
let config = QualityEvaluationConfig {
    metrics: vec![
        QualityMetric::PESQ,
        QualityMetric::STOI,
        QualityMetric::Naturalness,
    ],
    objective_metrics: true,
    subjective_prediction: true,
    perceptual_weighting: true,
    ..Default::default()
};
```

### PronunciationEvaluator

Specialized evaluator for pronunciation assessment.

```rust
let evaluator = PronunciationEvaluatorImpl::new().await?;
let result = evaluator.evaluate_pronunciation(
    &audio,
    &phoneme_alignments,
    reference_audio.as_ref(),
    Some(&config)
).await?;
```

#### Key Features

- **Phoneme-level Analysis**: Individual phoneme scoring
- **Phonetic Features**: Analysis based on articulatory features
- **Dynamic Time Warping**: Optimal alignment for comparison
- **Multi-language Support**: Language-specific pronunciation models
- **Confidence Scoring**: Reliability estimates for each assessment

### ComparativeEvaluator

Framework for comparing multiple speech synthesis systems.

```rust
let evaluator = ComparativeEvaluatorImpl::new().await?;

// Pairwise comparison
let comparison = evaluator.compare_systems(
    &system_a_audio,
    &system_b_audio,
    Some(&reference_audio),
    None
).await?;

// Multi-system comparison
let systems = HashMap::from([
    ("System_A".to_string(), audio_a),
    ("System_B".to_string(), audio_b),
]);
let multi_result = evaluator.compare_multiple_systems(
    &systems,
    Some(&reference_audio),
    Some(&config)
).await?;
```

### Language-Specific Evaluation

Multi-language evaluation framework with language-specific protocols.

```rust
use voirs_evaluation::quality::language_specific::LanguageSpecificEvaluator;

let evaluator = LanguageSpecificEvaluator::new().await?;

// Configure for specific language
let config = LanguageSpecificConfig {
    target_language: Language::English,
    accent_awareness: true,
    cultural_preferences: true,
    prosody_adaptation: true,
    ..Default::default()
};

let result = evaluator.evaluate_language_specific(
    &audio,
    &reference_audio,
    &config
).await?;

// Results include phonemic adaptation scores, prosody evaluation,
// cultural preference modeling, and accent-aware assessment
println!("Phonemic Score: {:.3}", result.phonemic_score);
println!("Prosody Score: {:.3}", result.prosody_score);
println!("Accent Consistency: {:.3}", result.accent_consistency);
```

### Streaming Audio Evaluation

Real-time evaluation capabilities for streaming audio.

```rust
use voirs_evaluation::audio::streaming::StreamingEvaluator;

let config = StreamingConfig {
    chunk_size: 1024,
    overlap: 512,
    quality_monitoring: true,
    adaptive_processing: true,
    ..Default::default()
};

let mut evaluator = StreamingEvaluator::new(config)?;

// Process audio chunks in real-time
for chunk in audio_stream {
    let chunk_result = evaluator.process_chunk(chunk).await?;
    
    // Get real-time quality metrics
    println!("Real-time SNR: {:.2} dB", chunk_result.snr_estimate);
    println!("Processing latency: {:.2} ms", chunk_result.processing_latency);
}

// Get final results
let final_stats = evaluator.finalize().await?;
```

### Industry Compliance Checking

Compliance validation against industry standards.

```rust
use voirs_evaluation::compliance::ComplianceChecker;

let checker = ComplianceChecker::new().await?;

// Check ITU-T P.862 (PESQ) compliance
let pesq_compliance = checker.check_itut_p862_compliance(
    &audio,
    &reference_audio,
    &PesqConfig::default()
).await?;

// Check ANSI S3.5 compliance
let ansi_compliance = checker.check_ansi_s35_compliance(
    &audio,
    &AnsiConfig::default()
).await?;

// Generate audit trail
let audit_trail = checker.generate_audit_trail()?;
for event in audit_trail.events {
    println!("{}: {}", event.timestamp, event.description);
}
```

### Real-Time Intelligibility Monitoring

Advanced real-time monitoring for speech intelligibility assessment.

```rust
use voirs_evaluation::perceptual::listening_test::IntelligibilityMonitor;

let config = IntelligibilityConfig {
    update_frequency: 10.0, // Hz
    analysis_window: 1.0,   // seconds
    confidence_threshold: 0.8,
    background_noise_profiling: true,
    ..Default::default()
};

let mut monitor = IntelligibilityMonitor::new(config)?;

// Real-time monitoring loop
loop {
    let audio_chunk = get_next_audio_chunk();
    let result = monitor.process_realtime(&audio_chunk).await?;
    
    // Access real-time metrics
    println!("Intelligibility: {:.3}", result.intelligibility_score);
    println!("Clarity Index: {:.3}", result.clarity_index);
    println!("SNR Estimate: {:.2} dB", result.snr_estimate);
    println!("Confidence: {:.3}", result.confidence);
    
    // Check for quality degradation
    if result.intelligibility_score < 0.7 {
        println!("Warning: Low intelligibility detected");
    }
}
```

### Multi-Listener Simulation

Comprehensive virtual listener modeling with demographic diversity.

```rust
use voirs_evaluation::perceptual::multi_listener::MultiListenerSimulator;

let config = MultiListenerConfig {
    num_listeners: 50,
    demographic_diversity: true,
    cultural_adaptation: true,
    hearing_impairment_simulation: true,
    environmental_conditions: true,
    ..Default::default()
};

let simulator = MultiListenerSimulator::new(config)?;

// Run listening test simulation
let simulation_result = simulator.simulate_listening_test(
    &audio,
    &ListeningTestConfig::default()
).await?;

// Analyze results
println!("Mean Opinion Score: {:.2} ± {:.2}", 
    simulation_result.mean_score, 
    simulation_result.standard_deviation
);

// Get demographic breakdowns
for (demographic, scores) in simulation_result.demographic_breakdown {
    println!("{}: {:.2}", demographic, scores.mean);
}

// Statistical reliability metrics
println!("Cronbach's Alpha: {:.3}", simulation_result.reliability.cronbachs_alpha);
println!("Inter-rater Reliability: {:.3}", simulation_result.reliability.inter_rater_reliability);
```

## Performance Optimization

### GPU Acceleration

The framework includes GPU acceleration for computationally intensive operations:

```rust
use voirs_evaluation::performance::GpuAccelerator;

let gpu = GpuAccelerator::new()?;
if gpu.is_gpu_available() {
    let correlation = gpu.gpu_correlation(&signal1, &signal2)?;
    let features = gpu.gpu_spectral_analysis(&signal, sample_rate)?;
}
```

### Batch Processing

Efficient processing of multiple samples:

```rust
let batch_samples = vec![
    (audio1, reference1),
    (audio2, reference2),
    // ... more samples
];

let results = evaluator.evaluate_quality_batch(&batch_samples, None).await?;
```

### Caching

Use caching for repeated evaluations:

```rust
use voirs_evaluation::performance::LRUCache;

let cache = LRUCache::new(100);
// Cache expensive computation results
cache.insert(audio_hash, quality_score);
```

## Statistical Analysis

The framework provides robust statistical analysis capabilities:

### Significance Testing

- **Paired t-test**: For dependent samples
- **Mann-Whitney U test**: Non-parametric alternative
- **Wilcoxon signed-rank test**: For paired non-parametric data
- **Bootstrap confidence intervals**: Distribution-free confidence estimation

### Effect Size Calculation

- **Cohen's d**: Standardized mean difference
- **Eta-squared**: Proportion of variance explained
- **Correlation coefficients**: Pearson and Spearman

### Multiple Comparison Correction

- **Bonferroni correction**: Conservative family-wise error control
- **False Discovery Rate (FDR)**: Less conservative alternative
- **Holm-Bonferroni**: Step-down procedure
- **Sidak correction**: Less conservative than Bonferroni

## Audio Format Support

### Supported Formats

- **WAV**: Uncompressed audio (recommended for evaluation)
- **FLAC**: Lossless compression
- **Sample rates**: 8 kHz, 16 kHz, 22.05 kHz, 44.1 kHz, 48 kHz
- **Channels**: Mono and stereo support

### Audio Requirements

- **PESQ**: Requires 8 kHz (narrowband) or 16 kHz (wideband)
- **STOI**: Works with any sample rate ≥ 8 kHz
- **MCD**: Requires matching sample rates for comparison
- **Minimum duration**: 0.5 seconds recommended for reliable metrics

## Error Handling

The framework uses comprehensive error handling:

```rust
use voirs_evaluation::EvaluationError;

match evaluator.evaluate_quality(&audio, None, None).await {
    Ok(result) => println!("Score: {:.3}", result.overall_score),
    Err(EvaluationError::InvalidInput { message }) => {
        eprintln!("Invalid input: {}", message);
    }
    Err(EvaluationError::MetricCalculationError { metric, message, .. }) => {
        eprintln!("Failed to calculate {}: {}", metric, message);
    }
    Err(e) => eprintln!("Evaluation error: {}", e),
}
```

## Best Practices

### Quality Evaluation

1. **Use reference audio when available**: Reference-based metrics (PESQ, MCD) provide more accurate assessments
2. **Choose appropriate metrics**: Different metrics capture different aspects of quality
3. **Consider context**: Evaluation criteria may vary by application (e.g., conversational AI vs. audiobooks)
4. **Validate with human listeners**: Correlate objective metrics with subjective assessments

### Pronunciation Assessment

1. **Accurate phoneme alignments**: Quality of phoneme-level evaluation depends on alignment accuracy
2. **Language-specific models**: Use appropriate language settings for best results
3. **Multiple dimensions**: Consider phoneme accuracy, fluency, and prosody together
4. **Confidence thresholds**: Filter low-confidence assessments for reliability

### Comparative Analysis

1. **Statistical significance**: Always test for statistical significance in comparisons
2. **Multiple samples**: Use sufficient samples for reliable statistical analysis
3. **Balanced evaluation**: Include diverse test cases representative of real usage
4. **Cross-validation**: Use cross-validation for robust system ranking

### Performance Optimization

1. **Batch processing**: Use batch evaluation for multiple samples
2. **GPU acceleration**: Enable GPU acceleration for large-scale evaluations
3. **Caching**: Implement caching for repeated evaluations
4. **Memory management**: Process large datasets in chunks to manage memory usage

## Integration Examples

### With VoiRS TTS

```rust
use voirs_sdk::VoirsTTS;
use voirs_evaluation::prelude::*;

let tts = VoirsTTS::new().await?;
let evaluator = QualityEvaluator::new().await?;

let audio = tts.synthesize("Hello, world!", None).await?;
let quality = evaluator.evaluate_quality(&audio, None, None).await?;
```

### With External Audio Processing

```rust
// Load audio from file
let audio = AudioBuffer::from_file("path/to/audio.wav")?;

// Preprocess if needed
let normalized_audio = audio.normalize()?;

// Evaluate
let result = evaluator.evaluate_quality(&normalized_audio, None, None).await?;
```

## Troubleshooting

### Common Issues and Solutions

#### Audio Format and Quality Issues

**Problem**: "Sample rate mismatch" errors
- **Cause**: Audio files have different sample rates
- **Solution**: 
  ```rust
  // Resample audio to target rate
  let resampled = audio.resample(16000)?;
  
  // Or configure evaluator to handle mixed rates
  let config = QualityEvaluationConfig {
      auto_resample: true,
      target_sample_rate: Some(16000),
      ..Default::default()
  };
  ```

**Problem**: "Insufficient audio duration" errors
- **Cause**: Audio shorter than minimum required for metrics
- **Solution**: 
  - PESQ requires minimum 3.2 seconds
  - STOI requires minimum 0.5 seconds
  - MCD requires minimum 1.0 seconds for reliable estimates

**Problem**: Poor quality scores for good audio
- **Cause**: Audio preprocessing issues or inappropriate reference
- **Solution**: 
  ```rust
  // Normalize audio levels
  let normalized = audio.normalize()?;
  
  // Remove DC offset
  let clean_audio = normalized.remove_dc_offset()?;
  
  // Use appropriate reference audio
  // Reference should be similar content, same speaker ideally
  ```

#### Performance and Memory Issues

**Problem**: High memory usage during batch processing
- **Solution**: 
  ```rust
  // Process in smaller chunks
  let chunk_size = 10; // Process 10 samples at a time
  for chunk in batch_samples.chunks(chunk_size) {
      let results = evaluator.evaluate_quality_batch(chunk, None).await?;
      // Process results immediately
  }
  ```

**Problem**: Slow evaluation performance
- **Solution**: 
  ```rust
  // Enable GPU acceleration if available
  let config = QualityEvaluationConfig {
      gpu_acceleration: true,
      parallel_processing: true,
      cache_intermediate_results: true,
      ..Default::default()
  };
  
  // Use batch processing for multiple samples
  let results = evaluator.evaluate_quality_batch(&samples, Some(&config)).await?;
  ```

**Problem**: GPU out of memory errors
- **Solution**: 
  ```rust
  // Reduce batch size or disable GPU for large samples
  let config = QualityEvaluationConfig {
      gpu_acceleration: false, // Fallback to CPU
      max_gpu_memory_usage: 0.8, // Use 80% of GPU memory
      ..Default::default()
  };
  ```

#### Metric-Specific Issues

**Problem**: PESQ calculations failing
- **Common causes**: 
  - Incorrect sample rate (must be 8kHz or 16kHz)
  - Audio too short or too long
  - Severe level differences between test and reference
- **Solution**: 
  ```rust
  // Ensure proper PESQ configuration
  let pesq_config = PesqConfig {
      sample_rate: 16000, // or 8000 for narrowband
      level_alignment: true,
      time_alignment: true,
      ..Default::default()
  };
  ```

**Problem**: Inconsistent MCD results
- **Cause**: Different windowing or frame parameters
- **Solution**: 
  ```rust
  // Use consistent MCD configuration
  let mcd_config = McdConfig {
      frame_length: 1024,
      hop_length: 256,
      mel_channels: 13,
      use_dtw_alignment: true,
      ..Default::default()
  };
  ```

#### Language-Specific Evaluation Issues

**Problem**: Poor pronunciation scores for non-native speakers
- **Solution**: 
  ```rust
  // Configure for accent-aware evaluation
  let config = LanguageSpecificConfig {
      accent_awareness: true,
      accent_type: Some(AccentType::NonNative),
      phonemic_tolerance: 0.8, // More tolerant scoring
      ..Default::default()
  };
  ```

**Problem**: Code-switching not detected properly
- **Solution**: 
  ```rust
  // Enable enhanced code-switching detection
  let config = LanguageSpecificConfig {
      code_switching_detection: true,
      matrix_language: Language::English,
      embedded_languages: vec![Language::Spanish, Language::French],
      ..Default::default()
  };
  ```

### Frequently Asked Questions (FAQ)

#### General Usage

**Q: What's the difference between reference-based and no-reference metrics?**
- **Reference-based** (PESQ, MCD): Compare synthesized audio to reference audio. More accurate but require ground truth.
- **No-reference** (Naturalness, some STOI variants): Evaluate audio quality without reference. Useful when no ground truth available.

**Q: Which metrics should I use for my application?**
- **Conversational AI**: STOI (intelligibility), Naturalness, Pronunciation accuracy
- **Audiobooks**: Naturalness, Prosody quality, Speaker similarity
- **Voice assistants**: STOI, Response naturalness, Pronunciation accuracy
- **Research/benchmarking**: PESQ, STOI, MCD, Statistical significance tests

**Q: How do I interpret quality scores?**
- **PESQ**: -0.5 to 4.5 (higher better), >3.0 is good quality
- **STOI**: 0.0 to 1.0 (higher better), >0.8 is good intelligibility
- **MCD**: Lower is better, <6.0 dB is good similarity
- **Naturalness**: 0.0 to 1.0 (higher better), >0.7 is natural-sounding

#### Performance Optimization

**Q: Should I use GPU acceleration?**
- **Yes, if**: Processing large batches, real-time requirements, compute-intensive metrics
- **No, if**: Small datasets, memory constraints, development/debugging

**Q: How much memory do I need?**
- **Minimum**: 4GB RAM for basic evaluation
- **Recommended**: 8GB+ for batch processing
- **Large-scale**: 16GB+ for concurrent evaluation of multiple systems

**Q: Can I run evaluations in parallel?**
```rust
// Yes, use async processing
let futures: Vec<_> = audio_samples.iter().map(|audio| {
    evaluator.evaluate_quality(audio, None, None)
}).collect();

let results = future::try_join_all(futures).await?;
```

#### Integration and Deployment

**Q: How do I integrate with existing TTS systems?**
```rust
// Create a wrapper for your TTS system
async fn evaluate_tts_output(
    tts_system: &mut YourTTSSystem,
    text: &str,
    reference: Option<&AudioBuffer>
) -> Result<QualityResult, EvaluationError> {
    let audio = tts_system.synthesize(text).await?;
    let evaluator = QualityEvaluator::new().await?;
    evaluator.evaluate_quality(&audio, reference, None).await
}
```

**Q: Can I customize evaluation metrics?**
```rust
// Yes, implement the QualityMetric trait
struct CustomMetric;

impl QualityMetric for CustomMetric {
    async fn calculate(&self, audio: &AudioBuffer, reference: Option<&AudioBuffer>) 
        -> Result<f32, EvaluationError> {
        // Your custom metric implementation
        Ok(0.0)
    }
}
```

#### Error Handling

**Q: How do I handle evaluation failures gracefully?**
```rust
// Use comprehensive error handling
match evaluator.evaluate_quality(&audio, None, None).await {
    Ok(result) => process_result(result),
    Err(EvaluationError::InvalidInput { message }) => {
        log::warn!("Skipping invalid audio: {}", message);
        // Continue with next sample
    }
    Err(EvaluationError::MetricCalculationError { metric, .. }) => {
        log::error!("Failed to calculate {}, using fallback", metric);
        // Use alternative metric or default score
    }
    Err(e) => {
        log::error!("Evaluation failed: {}", e);
        // Handle critical error
    }
}
```

### Debug Tips

#### Enable Detailed Logging
```rust
// Set log level for debugging
env_logger::Builder::from_env("RUST_LOG")
    .filter_level(log::LevelFilter::Debug)
    .init();

// VoiRS-specific debugging
std::env::set_var("VOIRS_EVALUATION_DEBUG", "1");
```

#### Performance Profiling
```rust
// Use built-in performance monitoring
use voirs_evaluation::performance::PerformanceMonitor;

let monitor = PerformanceMonitor::new();
let _guard = monitor.start_timer("evaluation");

// Your evaluation code here

// Timing information automatically logged
```

#### Memory Usage Monitoring
```rust
// Check memory usage during evaluation
let memory_monitor = MemoryMonitor::new();
memory_monitor.start_monitoring();

// Run evaluation
let result = evaluator.evaluate_quality(&audio, None, None).await?;

let memory_stats = memory_monitor.get_stats();
println!("Peak memory usage: {} MB", memory_stats.peak_usage_mb);
```

### Debug Information

Enable debug logging for detailed information:

```rust
env_logger::init();
// Detailed logs will show processing steps and timing information
```

### Performance Profiling

Use the built-in performance monitor:

```rust
use voirs_evaluation::performance::PerformanceMonitor;

let monitor = PerformanceMonitor::new();
let result = monitor.time_operation("evaluation", || {
    // Your evaluation code here
});
```

## Version Compatibility

- **Rust**: Minimum supported version 1.70.0
- **VoiRS SDK**: Compatible with versions 0.1.x
- **Candle**: GPU acceleration requires candle-core 0.3.x+

## License and Citation

This evaluation framework is part of the VoiRS project. Please cite appropriately when using in research or commercial applications.
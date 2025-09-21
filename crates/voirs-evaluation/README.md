# VoiRS Evaluation

[![Crates.io](https://img.shields.io/crates/v/voirs-evaluation)](https://crates.io/crates/voirs-evaluation)
[![Documentation](https://docs.rs/voirs-evaluation/badge.svg)](https://docs.rs/voirs-evaluation)
[![License](https://img.shields.io/crates/l/voirs-evaluation)](LICENSE)

**Speech synthesis quality evaluation and assessment metrics for VoiRS**

VoiRS Evaluation provides comprehensive quality assessment, pronunciation evaluation, and comparative analysis capabilities for speech synthesis systems, enabling objective measurement of audio quality and intelligibility.

## Features

### 🎯 Quality Evaluation
- **Perceptual Metrics**: PESQ, STOI, MOS prediction
- **Spectral Analysis**: MCD, MSD, spectral distortion
- **Temporal Metrics**: F0 tracking, rhythm, timing analysis
- **Naturalness Assessment**: Prosody, intonation, stress patterns

### 🗣️ Pronunciation Evaluation  
- **Phoneme-Level Scoring**: Accuracy, clarity, timing
- **Word-Level Assessment**: Intelligibility, stress patterns
- **Fluency Analysis**: Speech rate, pause patterns, rhythm
- **Accent Evaluation**: Native-like pronunciation scoring

### 📊 Comparative Analysis
- **A/B Testing**: Statistical significance testing
- **Batch Comparison**: Multiple model evaluation
- **Reference Matching**: Similarity to ground truth
- **Regression Analysis**: Performance trend tracking

### 📈 Perceptual Evaluation
- **Listening Test Simulation**: Automated MOS prediction
- **Subjective Metrics**: Naturalness, quality, intelligibility
- **Cross-Language Support**: Multiple language evaluation
- **Domain Adaptation**: Specialized evaluation for different domains

## Quick Start

Add VoiRS Evaluation to your `Cargo.toml`:

```toml
[dependencies]
voirs-evaluation = "0.1.0"

# Enable specific evaluation features
voirs-evaluation = { version = "0.1.0", features = ["quality", "pronunciation", "comparison"] }
```

### Basic Quality Evaluation

```rust
use voirs_evaluation::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize quality evaluator
    let evaluator = QualityEvaluator::new().await?;
    
    // Load audio files
    let reference = AudioBuffer::from_file("reference.wav")?;
    let synthesized = AudioBuffer::from_file("synthesized.wav")?;
    
    // Evaluate quality
    let quality_results = evaluator.evaluate_quality(&synthesized, Some(&reference)).await?;
    
    println!("Quality Results:");
    println!("  PESQ: {:.2}", quality_results.pesq);
    println!("  STOI: {:.3}", quality_results.stoi);
    println!("  MCD: {:.2} dB", quality_results.mcd);
    println!("  MOS: {:.2} ± {:.2}", quality_results.mos.mean, quality_results.mos.std);
    
    // Detailed analysis
    for metric in &quality_results.detailed_metrics {
        println!("  {}: {:.3}", metric.name, metric.value);
    }
    
    Ok(())
}
```

### Pronunciation Assessment

```rust
use voirs_evaluation::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize pronunciation evaluator
    let evaluator = PronunciationEvaluatorImpl::new().await?;
    
    // Load audio and expected text
    let audio = AudioBuffer::from_file("speech.wav")?;
    let expected_text = "Hello world, this is a pronunciation test.";
    
    // Evaluate pronunciation
    let pronunciation_results = evaluator.evaluate_pronunciation(
        &audio, 
        expected_text, 
        None
    ).await?;
    
    println!("Pronunciation Results:");
    println!("  Overall Score: {:.1}%", pronunciation_results.overall_score * 100.0);
    println!("  Accuracy: {:.1}%", pronunciation_results.accuracy_score * 100.0);
    println!("  Fluency: {:.1}%", pronunciation_results.fluency_score * 100.0);
    
    // Word-level analysis
    for word_score in &pronunciation_results.word_scores {
        println!("  '{}': {:.1}% (phonemes: {})", 
                 word_score.word, 
                 word_score.score * 100.0,
                 word_score.phoneme_scores.len());
    }
    
    // Phoneme-level details
    for phoneme_score in &pronunciation_results.phoneme_scores {
        println!("    {}: {:.1}% [{:.3}s - {:.3}s]", 
                 phoneme_score.phoneme, 
                 phoneme_score.score * 100.0,
                 phoneme_score.start_time,
                 phoneme_score.end_time);
    }
    
    Ok(())
}
```

### Comparative Analysis

```rust
use voirs_evaluation::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize comparative analyzer
    let analyzer = ComparativeAnalyzer::new().await?;
    
    // Load multiple model outputs
    let model_a_outputs = vec![
        AudioBuffer::from_file("model_a_sample1.wav")?,
        AudioBuffer::from_file("model_a_sample2.wav")?,
        AudioBuffer::from_file("model_a_sample3.wav")?,
    ];
    
    let model_b_outputs = vec![
        AudioBuffer::from_file("model_b_sample1.wav")?,
        AudioBuffer::from_file("model_b_sample2.wav")?,
        AudioBuffer::from_file("model_b_sample3.wav")?,
    ];
    
    // Perform A/B comparison
    let comparison_results = analyzer.compare_models(
        &model_a_outputs, 
        &model_b_outputs,
        None
    ).await?;
    
    println!("Comparative Analysis:");
    println!("  Model A avg score: {:.2}", comparison_results.model_a_stats.mean);
    println!("  Model B avg score: {:.2}", comparison_results.model_b_stats.mean);
    println!("  Difference: {:.2}", comparison_results.mean_difference);
    println!("  P-value: {:.4}", comparison_results.statistical_significance.p_value);
    
    if comparison_results.statistical_significance.is_significant {
        println!("  ✅ Difference is statistically significant");
    } else {
        println!("  ❌ Difference is not statistically significant");
    }
    
    // Detailed breakdown
    for metric in &comparison_results.metric_breakdown {
        println!("  {}: A={:.2}, B={:.2}, diff={:.2}", 
                 metric.metric_name, 
                 metric.model_a_score, 
                 metric.model_b_score,
                 metric.difference);
    }
    
    Ok(())
}
```

### Perceptual Evaluation

```rust
use voirs_evaluation::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize perceptual evaluator
    let evaluator = PerceptualEvaluator::new().await?;
    
    // Load audio samples
    let audio_samples = vec![
        AudioBuffer::from_file("sample1.wav")?,
        AudioBuffer::from_file("sample2.wav")?,
        AudioBuffer::from_file("sample3.wav")?,
    ];
    
    // Evaluate perceptual quality
    let perceptual_results = evaluator.evaluate_perceptual(
        &audio_samples,
        None
    ).await?;
    
    println!("Perceptual Evaluation:");
    println!("  Overall Quality: {:.2}", perceptual_results.overall_quality);
    println!("  Naturalness: {:.2}", perceptual_results.naturalness);
    println!("  Intelligibility: {:.2}", perceptual_results.intelligibility);
    
    // Detailed scores
    for (i, sample_result) in perceptual_results.sample_results.iter().enumerate() {
        println!("  Sample {}: MOS={:.2}, Quality={:.2}, Naturalness={:.2}",
                 i + 1,
                 sample_result.predicted_mos,
                 sample_result.quality_score,
                 sample_result.naturalness_score);
    }
    
    Ok(())
}
```

## Evaluation Metrics

### Quality Metrics

| Metric | Description | Range | Higher Better |
|--------|-------------|-------|---------------|
| PESQ | Perceptual Evaluation of Speech Quality | -0.5 to 4.5 | ✅ |
| STOI | Short-Time Objective Intelligibility | 0.0 to 1.0 | ✅ |
| MCD | Mel-Cepstral Distortion | 0.0+ dB | ❌ |
| MSD | Mel-Spectral Distortion | 0.0+ dB | ❌ |
| F0-RMSE | Fundamental Frequency Root Mean Square Error | 0.0+ Hz | ❌ |
| VUV-Error | Voiced/Unvoiced Error Rate | 0.0 to 1.0 | ❌ |

### Pronunciation Metrics

| Metric | Description | Range | Notes |
|--------|-------------|-------|-------|
| Accuracy | Phoneme pronunciation accuracy | 0.0 to 1.0 | Based on phoneme alignment |
| Fluency | Speech fluency and rhythm | 0.0 to 1.0 | Considers timing and pauses |
| Completeness | Percentage of phonemes produced | 0.0 to 1.0 | Measures omissions |
| Prosody | Stress and intonation patterns | 0.0 to 1.0 | Pitch contour analysis |

## Feature Flags

Enable specific functionality through feature flags:

```toml
[dependencies]
voirs-evaluation = { 
    version = "0.1.0", 
    features = [
        "quality",        # Quality evaluation metrics
        "pronunciation",  # Pronunciation assessment
        "comparison",     # Comparative analysis
        "perceptual",     # Perceptual evaluation
        "all-metrics",    # Enable all evaluation metrics
        "gpu",            # GPU acceleration support
        "parallel",       # Parallel processing
    ]
}
```

## Configuration

### Quality Evaluation Configuration

```rust
use voirs_evaluation::prelude::*;

let config = QualityEvaluationConfig {
    // Metrics to compute
    enabled_metrics: vec![
        QualityMetric::PESQ,
        QualityMetric::STOI,
        QualityMetric::MCD,
        QualityMetric::MSD,
    ],
    
    // Reference audio requirements
    require_reference: true,
    
    // Sampling configuration
    target_sample_rate: 16000,
    frame_length_ms: 25.0,
    hop_length_ms: 10.0,
    
    // MCD specific settings
    mcd_config: MCDConfig {
        order: 13,
        alpha: 0.42,
        use_power: true,
    },
    
    // PESQ configuration
    pesq_config: PESQConfig {
        sample_rate: 16000,
        mode: PESQMode::WideBand,
    },
    
    // Parallel processing
    enable_parallel: true,
    max_workers: None, // Auto-detect
};

let evaluator = QualityEvaluator::with_config(config).await?;
```

### Pronunciation Assessment Configuration

```rust
use voirs_evaluation::prelude::*;

let config = PronunciationConfig {
    // Language settings
    language: LanguageCode::EnUs,
    dialect: Some("general_american".to_string()),
    
    // Alignment settings
    alignment_config: AlignmentConfig {
        time_resolution_ms: 10,
        confidence_threshold: 0.5,
        enable_speaker_adaptation: true,
    },
    
    // Scoring weights
    scoring_weights: ScoringWeights {
        accuracy_weight: 0.4,
        fluency_weight: 0.3,
        prosody_weight: 0.2,
        completeness_weight: 0.1,
    },
    
    // Phoneme set
    phoneme_set: PhonemeSet::CMU,
    
    // Analysis options
    enable_detailed_analysis: true,
    include_confidence_scores: true,
    analyze_stress_patterns: true,
};

let evaluator = PronunciationEvaluatorImpl::with_config(config).await?;
```

## Language Support

VoiRS Evaluation supports multiple languages for pronunciation assessment:

| Language | Code | Phoneme Set | MFA Support |
|----------|------|-------------|-------------|
| English (US) | en-US | CMU | ✅ |
| English (UK) | en-GB | CMU | ✅ |
| Spanish | es-ES | SAMPA | ✅ |
| French | fr-FR | SAMPA | ✅ |
| German | de-DE | SAMPA | ✅ |
| Japanese | ja-JP | Custom | ❌ |
| Chinese | zh-CN | Custom | ❌ |

## Performance Optimization

### Batch Processing

```rust
use voirs_evaluation::prelude::*;

// Process multiple files efficiently
let audio_files = vec![
    AudioBuffer::from_file("file1.wav")?,
    AudioBuffer::from_file("file2.wav")?,
    AudioBuffer::from_file("file3.wav")?,
];

let batch_results = evaluator.evaluate_batch(&audio_files, None).await?;
```

### GPU Acceleration

```rust
use voirs_evaluation::prelude::*;

// Enable GPU acceleration
let config = QualityEvaluationConfig {
    enable_gpu: true,
    gpu_device: Some(0), // Use first GPU
    batch_size: 32,
    ..Default::default()
};

let evaluator = QualityEvaluator::with_config(config).await?;
```

### Parallel Processing

```rust
use voirs_evaluation::prelude::*;

// Configure parallel processing
let config = QualityEvaluationConfig {
    enable_parallel: true,
    max_workers: Some(8), // Use 8 threads
    chunk_size: 1000,     // Process in chunks
    ..Default::default()
};

let evaluator = QualityEvaluator::with_config(config).await?;
```

## Error Handling

VoiRS Evaluation provides comprehensive error handling:

```rust
use voirs_evaluation::prelude::*;

match evaluator.evaluate_quality(&audio, Some(&reference)).await {
    Ok(results) => {
        println!("Quality evaluation successful: {:.2}", results.overall_score);
    }
    Err(EvaluationError::AudioTooShort { duration }) => {
        eprintln!("Audio too short: {:.1}s", duration);
    }
    Err(EvaluationError::SampleRateMismatch { expected, actual }) => {
        eprintln!("Sample rate mismatch: expected {}Hz, got {}Hz", expected, actual);
    }
    Err(EvaluationError::ReferenceRequired { metric }) => {
        eprintln!("Reference audio required for metric: {:?}", metric);
    }
    Err(EvaluationError::LanguageNotSupported { language }) => {
        eprintln!("Language not supported: {:?}", language);
    }
    Err(e) => {
        eprintln!("Evaluation failed: {}", e);
    }
}
```

## Examples

Check out the [examples](examples/) directory for comprehensive usage examples:

- [`quality_evaluation.rs`](examples/quality_evaluation.rs) - Basic quality evaluation
- [`pronunciation_assessment.rs`](examples/pronunciation_assessment.rs) - Pronunciation scoring
- [`comparative_analysis.rs`](examples/comparative_analysis.rs) - Model comparison
- [`perceptual_evaluation.rs`](examples/perceptual_evaluation.rs) - Perceptual quality assessment
- [`batch_processing.rs`](examples/batch_processing.rs) - Efficient batch evaluation
- [`custom_metrics.rs`](examples/custom_metrics.rs) - Custom evaluation metrics
- [`multilingual_evaluation.rs`](examples/multilingual_evaluation.rs) - Multi-language assessment

## Benchmarks

Performance benchmarks on standard datasets:

| Dataset | Metric | Processing Time | Memory Usage |
|---------|--------|-----------------|--------------|
| VCTK | PESQ | 0.2s/file | 150MB |
| LibriSpeech | STOI | 0.1s/file | 100MB |
| CommonVoice | MCD | 0.3s/file | 200MB |
| Custom | Pronunciation | 0.5s/file | 250MB |

*Benchmarks performed on Intel i7-9700K with 32GB RAM*

## Research Applications

VoiRS Evaluation is designed for research applications:

- **Model Development**: Objective evaluation during training
- **Ablation Studies**: Component-wise performance analysis
- **Cross-Language Evaluation**: Multilingual TTS assessment
- **Perceptual Studies**: Correlation with human perception
- **Benchmark Creation**: Standardized evaluation protocols

## Contributing

We welcome contributions! Please see our [Contributing Guide](../../CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/cool-japan/voirs.git
cd voirs/crates/voirs-evaluation

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

If you use VoiRS Evaluation in your research, please cite:

```bibtex
@software{voirs_evaluation,
  title = {VoiRS Evaluation: Comprehensive Speech Synthesis Assessment},
  author = {Tetsuya Kitahata},
  organization = {Cool Japan Co., Ltd.},
  year = {2024},
  url = {https://github.com/cool-japan/voirs}
}
```
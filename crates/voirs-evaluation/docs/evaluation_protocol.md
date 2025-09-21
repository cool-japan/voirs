# VoiRS Evaluation Protocol Documentation

## Overview

This document outlines the comprehensive evaluation protocols implemented in the VoiRS evaluation framework. These protocols ensure standardized, reliable, and reproducible quality assessment of speech synthesis systems.

## Core Evaluation Protocols

### 1. ITU-T P.862 (PESQ) Protocol

**Objective**: Perceptual Evaluation of Speech Quality  
**Standard**: ITU-T Recommendation P.862 (2001)  
**Application**: Narrowband (8 kHz) and wideband (16 kHz) speech quality assessment

#### Protocol Specifications
- **Audio Requirements**:
  - Sample rate: 8 kHz or 16 kHz
  - Duration: 8-20 seconds recommended
  - Format: 16-bit PCM mono
  - Level alignment: automatic (-26 dBov target)

- **Processing Steps**:
  1. Time and level alignment of reference and degraded signals
  2. Auditory transform using psychoacoustic model
  3. Cognitive model application for perceptual loudness patterns
  4. Disturbance computation and averaging
  5. PESQ score mapping (1.0-4.5 scale)

- **Quality Thresholds**:
  - Excellent: ≥ 4.0
  - Good: 3.5-4.0
  - Fair: 3.0-3.5
  - Poor: 2.5-3.0
  - Bad: < 2.5

#### Implementation
```rust
use voirs_evaluation::quality::PESQEvaluator;

let evaluator = PESQEvaluator::new(16000)?;
let score = evaluator.calculate_pesq(&reference, &degraded).await?;
```

### 2. ITU-T P.863 (POLQA) Protocol

**Objective**: Perceptual Objective Listening Quality Assessment  
**Standard**: ITU-T Recommendation P.863 (2014)  
**Application**: Super-wideband speech quality assessment

#### Protocol Specifications
- **Audio Requirements**:
  - Sample rate: 8-48 kHz
  - Duration: 8-20 seconds
  - Format: 16-bit or 24-bit PCM mono
  - Frequency range: up to 14 kHz

- **Advanced Features**:
  - Time-frequency analysis with improved psychoacoustic modeling
  - Enhanced cognitive model for super-wideband content
  - Improved handling of time-varying distortions
  - Support for VoIP and codec evaluation

### 3. STOI (Short-Time Objective Intelligibility) Protocol

**Objective**: Speech intelligibility prediction  
**Standard**: IEEE/ACM Transactions on Audio Processing (2011)  
**Application**: Intelligibility assessment in noisy and reverberant conditions

#### Protocol Specifications
- **Audio Requirements**:
  - Sample rate: 8-48 kHz
  - Duration: ≥ 3 seconds (minimum requirement)
  - Format: 16-bit PCM mono

- **Processing Steps**:
  1. One-third octave band decomposition (150 Hz - 4.3 kHz)
  2. Short-time Fourier transform (384 ms frames, 75% overlap)
  3. Temporal envelope extraction and normalization
  4. Correlation computation between clean and processed envelopes
  5. Intelligibility index calculation (0-1 scale)

#### Implementation
```rust
use voirs_evaluation::quality::STOIEvaluator;

let evaluator = STOIEvaluator::new(16000)?;
let intelligibility = evaluator.calculate_stoi(&reference, &processed).await?;
```

### 4. MCD (Mel-Cepstral Distortion) Protocol

**Objective**: Spectral similarity measurement  
**Application**: Speech synthesis and voice conversion evaluation

#### Protocol Specifications
- **Audio Requirements**:
  - Sample rate: 16-48 kHz
  - Frame size: 1024 samples (typical)
  - Hop size: 256 samples (typical)
  - Window: Hamming or Hann

- **Processing Steps**:
  1. Pre-emphasis filtering (α = 0.97)
  2. Windowing and FFT computation
  3. Mel-filterbank application (24-40 filters)
  4. DCT transformation for MFCC extraction
  5. Euclidean distance computation
  6. Statistical averaging across frames

- **Quality Thresholds**:
  - Excellent: < 4.0 dB
  - Good: 4.0-6.0 dB
  - Fair: 6.0-8.0 dB
  - Poor: > 8.0 dB

#### Implementation
```rust
use voirs_evaluation::quality::MCDEvaluator;

let evaluator = MCDEvaluator::new()?;
let mcd = evaluator.calculate_mcd(&reference, &synthetic).await?;
```

## Specialized Evaluation Protocols

### 5. Children's Speech Evaluation Protocol

**Objective**: Age-appropriate quality assessment for children's voices  
**Application**: Pediatric speech synthesis and therapy applications

#### Protocol Adaptations
- **Fundamental Frequency Adjustments**:
  - Age group 4-6: F0 range 200-500 Hz
  - Age group 7-10: F0 range 180-400 Hz
  - Age group 11-14: F0 range 150-350 Hz

- **Phoneme Acquisition Modeling**:
  - Early sounds: /m/, /b/, /p/, /d/, /n/, /h/, /w/
  - Middle sounds: /k/, /g/, /f/, /y/, /ng/
  - Late sounds: /s/, /z/, /l/, /r/, /th/, /ch/, /sh/, /j/

- **Listener Familiarity Factors**:
  - Parent/caregiver familiarity boost: +0.1-0.2
  - Teacher familiarity: +0.05-0.1
  - Stranger assessment: baseline

### 6. Elderly and Pathological Speech Protocol

**Objective**: Quality assessment for age-related and pathological speech conditions  
**Application**: Assistive technology and speech therapy

#### Condition-Specific Adaptations
- **Parkinson's Disease**: Reduced articulation rate, monotone prosody
- **Aphasia**: Word-finding difficulties, grammatical errors
- **Dysarthria**: Motor speech disorders, unclear articulation
- **Voice Disorders**: Hoarseness, breathiness, vocal tremor

#### Assessment Criteria
- **Intelligibility Weighting**: 40% (increased from standard 25%)
- **Naturalness Assessment**: Context-aware evaluation
- **Cognitive Load**: Processing time and complexity metrics

### 7. Emotional Speech Evaluation Protocol

**Objective**: Emotion-aware quality assessment  
**Application**: Expressive speech synthesis and emotion transfer

#### Emotion Categories
- **Basic Emotions**: Happy, sad, angry, fear, surprise, disgust
- **Complex Emotions**: Contempt, pride, shame, guilt
- **Arousal Levels**: Low, medium, high
- **Valence Dimensions**: Negative, neutral, positive

#### Evaluation Metrics
- **Emotion Recognition Accuracy**: Automated classification
- **Perceptual Authenticity**: Human listener validation
- **Prosodic Appropriateness**: F0, energy, timing consistency
- **Expressiveness Transfer**: Source-to-target emotion preservation

### 8. Multi-Language Evaluation Protocol

**Objective**: Cross-linguistic quality assessment  
**Application**: Multilingual TTS systems and accent evaluation

#### Language-Specific Adaptations
- **Phoneme Inventory**: Language-specific phoneme sets
- **Prosodic Patterns**: Stress, tone, rhythm variations
- **Cultural Appropriateness**: Regional accent acceptance
- **Cross-Language Intelligibility**: L2 speaker assessment

## Quality Assurance Protocols

### Statistical Validation Protocol

#### Correlation Requirements
- **PESQ-Human Correlation**: > 0.9 (Pearson correlation)
- **STOI Prediction Accuracy**: > 95% on standardized test sets
- **MCD Precision**: < 0.01 dB variance across repeated measurements
- **Statistical Significance**: p < 0.05 for all comparative tests

#### Performance Standards
- **Real-time Factor**: < 0.1 for all metrics
- **Memory Usage**: < 1GB for batch processing
- **GPU Acceleration**: > 10x speedup when available
- **Parallel Efficiency**: > 80% on multi-core systems

### Compliance Verification Protocol

#### Standards Compliance
- **ITU-T Conformance**: Bit-exact reference implementation agreement
- **IEEE Standards**: Numerical precision within specified tolerances
- **Cross-Platform Consistency**: < 0.1% variation across platforms
- **Version Compatibility**: Backward compatibility maintenance

#### Validation Procedures
1. **Reference Implementation Testing**: Comparison with official implementations
2. **Edge Case Validation**: Boundary condition testing
3. **Numerical Stability**: Precision loss analysis
4. **Regression Testing**: Automated quality threshold monitoring

## Usage Guidelines

### Basic Evaluation Workflow

1. **Audio Preparation**:
   ```rust
   let reference = AudioLoader::load("reference.wav").await?;
   let synthetic = AudioLoader::load("synthetic.wav").await?;
   ```

2. **Quality Assessment**:
   ```rust
   let evaluator = QualityEvaluator::new().await?;
   let results = evaluator.evaluate_quality(&synthetic, Some(&reference), None).await?;
   ```

3. **Results Interpretation**:
   ```rust
   println!("Overall Quality: {:.2}", results.overall_score);
   println!("PESQ Score: {:.2}", results.pesq_score.unwrap_or(0.0));
   println!("STOI Score: {:.2}", results.stoi_score.unwrap_or(0.0));
   ```

### Advanced Configuration

#### Custom Evaluation Protocols
```rust
let config = QualityEvaluationConfig {
    objective_metrics: true,
    perceptual_weighting: 0.7,
    linguistic_adaptation: Some(LanguageCode::EnUs),
    demographic_adaptation: Some(DemographicProfile {
        age_group: AgeGroup::Adult,
        gender: Gender::Mixed,
        hearing_profile: HearingProfile::Normal,
    }),
    ..Default::default()
};
```

#### Batch Processing
```rust
let batch_results = evaluator.evaluate_batch(
    &audio_pairs,
    &config,
    Some(ProgressCallback::new()),
).await?;
```

## Validation and Certification

### Certification Levels

1. **Basic Compliance**: Meets minimum quality thresholds
2. **Standard Compliance**: Follows industry best practices
3. **Advanced Compliance**: Exceeds performance benchmarks
4. **Research Grade**: Suitable for academic research and publication

### Quality Assurance Checklist

- [ ] Audio format compatibility verified
- [ ] Sample rate requirements met
- [ ] Duration constraints satisfied
- [ ] Reference alignment confirmed
- [ ] Statistical significance achieved
- [ ] Cross-platform consistency validated
- [ ] Performance benchmarks met
- [ ] Documentation completeness verified

## Troubleshooting

### Common Issues

1. **Sample Rate Mismatch**: Ensure reference and test audio have matching sample rates
2. **Duration Requirements**: Verify minimum duration requirements for each metric
3. **Level Alignment**: Check audio level calibration for PESQ evaluation
4. **Memory Constraints**: Monitor memory usage for large batch evaluations
5. **GPU Availability**: Verify CUDA/Metal availability for acceleration

### Performance Optimization

1. **Parallel Processing**: Enable multi-threading for batch evaluations
2. **GPU Acceleration**: Use CUDA or Metal backends when available
3. **Caching**: Enable persistent caching for repeated evaluations
4. **Streaming**: Use streaming evaluation for real-time applications

## References

1. ITU-T Recommendation P.862 (2001): "Perceptual evaluation of speech quality (PESQ)"
2. ITU-T Recommendation P.863 (2014): "Perceptual objective listening quality assessment"
3. Taal et al. (2011): "An algorithm for intelligibility prediction of time-frequency weighted noisy speech"
4. Kubichek (1993): "Mel-cepstral distance measure for objective speech quality assessment"
5. Hunt & Black (1996): "Unit selection in a concatenative speech synthesis system using a large speech database"

## Version History

- **v0.1.0**: Initial protocol implementation
- **v0.1.1**: Added specialized evaluation protocols
- **v0.2.0**: Enhanced multi-language support
- **v0.3.0**: GPU acceleration and performance optimization
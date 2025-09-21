# VoiRS Evaluation Best Practices Guide

## Overview

This guide provides best practices for using the VoiRS evaluation framework effectively. Following these guidelines will help you achieve reliable, meaningful evaluation results and avoid common pitfalls.

## General Principles

### 1. Choose Appropriate Metrics

**Match metrics to your evaluation goals:**

- **Quality Assessment**: Use MOS, PESQ, or Naturalness for overall quality
- **Intelligibility Focus**: Use STOI or Intelligibility Score for clarity assessment  
- **Similarity Measurement**: Use MCD for spectral similarity to reference
- **Pronunciation Evaluation**: Use Phoneme Accuracy and Fluency scores

**Combine complementary metrics:**
```rust
// Use multiple metrics for comprehensive evaluation
let config = QualityEvaluationConfig {
    metrics: vec![
        QualityMetric::MOS,        // Overall quality
        QualityMetric::STOI,       // Intelligibility
        QualityMetric::Naturalness, // Naturalness
    ],
    ..Default::default()
};
```

### 2. Prepare Quality Reference Data

**Use appropriate reference audio:**
- Same speaker when possible for voice conversion evaluation
- Same text content for pronunciation assessment
- Professional recordings for high-quality benchmarks
- Multiple speakers for generalization studies

**Ensure reference quality:**
```rust
// Validate reference audio quality
let reference_metrics = evaluator.analyze_reference_quality(&reference_audio).await?;
if reference_metrics.snr < 20.0 {
    log::warn!("Reference audio has low SNR: {:.1} dB", reference_metrics.snr);
}
```

### 3. Statistical Rigor

**Use sufficient sample sizes:**
- Minimum 20 samples for basic comparisons
- 50+ samples for robust statistical analysis
- 100+ samples for publication-quality results

**Apply appropriate statistical tests:**
```rust
// Example: Proper statistical comparison
let comparison_config = StatisticalConfig {
    test_type: StatisticalTest::PairedTTest, // For related samples
    multiple_comparison_correction: MultipleComparison::Bonferroni,
    confidence_level: 0.95,
    effect_size_calculation: true,
    ..Default::default()
};
```

## Quality Evaluation Best Practices

### Data Preparation

#### Audio Preprocessing
```rust
// Standard preprocessing pipeline
async fn preprocess_audio(audio: AudioBuffer) -> Result<AudioBuffer, EvaluationError> {
    let processed = audio
        .normalize()?                    // Normalize amplitude
        .remove_dc_offset()?             // Remove DC bias
        .trim_silence(0.02)?             // Remove leading/trailing silence
        .resample_if_needed(16000)?;     // Standardize sample rate
    
    // Validate processed audio
    if processed.duration() < 0.5 {
        return Err(EvaluationError::InvalidInput {
            message: "Audio too short for reliable evaluation".to_string()
        });
    }
    
    Ok(processed)
}
```

#### Batch Processing Optimization
```rust
// Efficient batch processing
async fn evaluate_batch_efficiently(
    samples: Vec<AudioBuffer>,
    evaluator: &QualityEvaluator
) -> Result<Vec<QualityResult>, EvaluationError> {
    const BATCH_SIZE: usize = 10; // Adjust based on memory constraints
    
    let mut all_results = Vec::new();
    
    for chunk in samples.chunks(BATCH_SIZE) {
        // Process chunk with parallel evaluation
        let chunk_results = evaluator.evaluate_quality_batch(
            chunk, 
            Some(&QualityEvaluationConfig {
                parallel_processing: true,
                cache_intermediate_results: true,
                ..Default::default()
            })
        ).await?;
        
        all_results.extend(chunk_results);
        
        // Optional: Memory cleanup between chunks
        if cfg!(feature = "memory_intensive") {
            tokio::task::yield_now().await;
        }
    }
    
    Ok(all_results)
}
```

### Metric Selection Guidelines

#### Application-Specific Recommendations

**Voice Assistants / Conversational AI:**
```rust
let config = QualityEvaluationConfig {
    metrics: vec![
        QualityMetric::STOI,           // Primary: intelligibility
        QualityMetric::Naturalness,    // Important: user experience
        QualityMetric::ResponseTime,   // Critical: real-time performance
    ],
    real_time_requirements: true,
    ..Default::default()
};
```

**Audiobooks / Entertainment:**
```rust
let config = QualityEvaluationConfig {
    metrics: vec![
        QualityMetric::MOS,            // Primary: overall quality
        QualityMetric::Naturalness,    // Critical: listening comfort
        QualityMetric::SpeakerSimilarity, // Important: character consistency
        QualityMetric::ProsodyQuality, // Important: engaging delivery
    ],
    long_form_optimization: true,
    ..Default::default()
};
```

**Accessibility Applications:**
```rust
let config = QualityEvaluationConfig {
    metrics: vec![
        QualityMetric::STOI,           // Critical: comprehension
        QualityMetric::Intelligibility, // Critical: clarity
        QualityMetric::PhonemeAccuracy, // Important: precision
    ],
    accessibility_focus: true,
    noise_robustness_testing: true,
    ..Default::default()
};
```

### Performance Optimization

#### GPU Acceleration Guidelines
```rust
// Intelligent GPU usage
let use_gpu = should_use_gpu(&evaluation_task);

let config = QualityEvaluationConfig {
    gpu_acceleration: use_gpu,
    gpu_memory_fraction: if use_gpu { 0.8 } else { 0.0 },
    fallback_to_cpu: true, // Always enable fallback
    ..Default::default()
};

fn should_use_gpu(task: &EvaluationTask) -> bool {
    // Use GPU for:
    // - Large batch sizes (>50 samples)
    // - Compute-intensive metrics (PESQ with large datasets)
    // - Real-time requirements with complex metrics
    task.batch_size > 50 || 
    task.has_compute_intensive_metrics() ||
    task.real_time_requirements
}
```

#### Memory Management
```rust
// Memory-efficient processing
async fn process_large_dataset(
    dataset: LargeAudioDataset,
    evaluator: &QualityEvaluator
) -> Result<EvaluationReport, EvaluationError> {
    let mut results = Vec::new();
    let mut memory_monitor = MemoryMonitor::new();
    
    for batch in dataset.iter_batches(OPTIMAL_BATCH_SIZE) {
        // Monitor memory usage
        if memory_monitor.usage_mb() > MAX_MEMORY_MB {
            // Force garbage collection and wait
            drop(results.drain(..results.len()/2).collect::<Vec<_>>());
            tokio::task::yield_now().await;
        }
        
        let batch_results = evaluator.evaluate_quality_batch(&batch, None).await?;
        results.extend(batch_results);
    }
    
    Ok(EvaluationReport::from_results(results))
}
```

## Pronunciation Evaluation Best Practices

### Phoneme Alignment Quality

```rust
// Validate alignment quality before evaluation
async fn evaluate_with_alignment_validation(
    audio: &AudioBuffer,
    phoneme_alignments: &PhonemeAlignments,
    evaluator: &PronunciationEvaluator
) -> Result<PronunciationResult, EvaluationError> {
    // Check alignment quality
    let alignment_quality = validate_alignment_quality(audio, phoneme_alignments)?;
    
    if alignment_quality.confidence < 0.8 {
        log::warn!("Low alignment confidence: {:.3}", alignment_quality.confidence);
    }
    
    // Use confidence-weighted evaluation
    let config = PronunciationConfig {
        alignment_confidence_threshold: 0.7,
        weight_by_confidence: true,
        ..Default::default()
    };
    
    evaluator.evaluate_pronunciation(audio, phoneme_alignments, None, Some(&config)).await
}
```

### Language-Specific Considerations

```rust
// Language-aware pronunciation evaluation
async fn evaluate_multilingual_pronunciation(
    audio: &AudioBuffer,
    text: &str,
    target_language: Language,
    evaluator: &PronunciationEvaluator
) -> Result<PronunciationResult, EvaluationError> {
    let config = PronunciationConfig {
        target_language,
        phoneme_set: PhonemeSet::for_language(target_language),
        accent_awareness: true,
        cross_lingual_features: true,
        ..Default::default()
    };
    
    // Generate language-specific phoneme alignments
    let alignments = generate_alignments(text, target_language).await?;
    
    evaluator.evaluate_pronunciation(audio, &alignments, None, Some(&config)).await
}
```

## Comparative Analysis Best Practices

### Experimental Design

#### Balanced Comparison Framework
```rust
// Proper experimental design for system comparison
async fn compare_systems_properly(
    systems: HashMap<String, TtsSystem>,
    test_dataset: TestDataset,
    evaluator: &ComparativeEvaluator
) -> Result<ComparisonReport, EvaluationError> {
    let mut all_results = HashMap::new();
    
    // Generate all outputs first (prevents order effects)
    for (system_name, system) in &systems {
        let mut system_outputs = Vec::new();
        
        for test_case in &test_dataset.test_cases {
            let output = system.synthesize(&test_case.text).await?;
            system_outputs.push((test_case.clone(), output));
        }
        
        all_results.insert(system_name.clone(), system_outputs);
    }
    
    // Randomize evaluation order
    let evaluation_order = randomize_evaluation_order(&all_results);
    
    // Perform pairwise comparisons
    let comparisons = evaluator.compare_all_pairs(
        &evaluation_order,
        &ComparisonConfig {
            statistical_tests: true,
            effect_size_calculation: true,
            confidence_intervals: true,
            ..Default::default()
        }
    ).await?;
    
    Ok(ComparisonReport::new(comparisons))
}
```

#### Statistical Power Analysis
```rust
// Ensure sufficient statistical power
fn calculate_required_sample_size(
    expected_effect_size: f32,
    desired_power: f32,
    alpha_level: f32
) -> usize {
    // Cohen's conventions:
    // Small effect: 0.2, Medium effect: 0.5, Large effect: 0.8
    
    let power_analysis = StatisticalPowerAnalysis::new()
        .effect_size(expected_effect_size)
        .statistical_power(desired_power)
        .alpha_level(alpha_level);
    
    power_analysis.required_sample_size()
}
```

### Result Interpretation

#### Multiple Comparison Correction
```rust
// Handle multiple comparisons properly
async fn analyze_multiple_systems(
    comparison_results: Vec<ComparisonResult>,
    correction_method: MultipleComparisonCorrection
) -> Result<CorrectedResults, EvaluationError> {
    let corrected = match correction_method {
        MultipleComparisonCorrection::Bonferroni => {
            apply_bonferroni_correction(&comparison_results)
        },
        MultipleComparisonCorrection::FDR => {
            apply_fdr_correction(&comparison_results)
        },
        MultipleComparisonCorrection::Holm => {
            apply_holm_correction(&comparison_results)
        },
    };
    
    // Report both corrected and uncorrected results
    Ok(CorrectedResults {
        original: comparison_results,
        corrected,
        correction_method,
    })
}
```

## Error Handling and Validation

### Input Validation
```rust
// Comprehensive input validation
async fn validate_evaluation_inputs(
    audio: &AudioBuffer,
    reference: Option<&AudioBuffer>,
    config: &EvaluationConfig
) -> Result<(), EvaluationError> {
    // Audio quality checks
    if audio.duration() < config.minimum_duration {
        return Err(EvaluationError::InvalidInput {
            message: format!("Audio too short: {:.2}s < {:.2}s", 
                audio.duration(), config.minimum_duration)
        });
    }
    
    if audio.sample_rate() < 8000 {
        return Err(EvaluationError::InvalidInput {
            message: "Sample rate too low for reliable evaluation".to_string()
        });
    }
    
    // Check for clipping or saturation
    if audio.max_amplitude() >= 0.99 {
        log::warn!("Audio may be clipped (max amplitude: {:.3})", audio.max_amplitude());
    }
    
    // Reference audio validation
    if let Some(ref_audio) = reference {
        if (audio.sample_rate() - ref_audio.sample_rate()).abs() > 100 {
            log::warn!("Sample rate mismatch: {} vs {}", 
                audio.sample_rate(), ref_audio.sample_rate());
        }
    }
    
    Ok(())
}
```

### Graceful Error Handling
```rust
// Robust evaluation with fallbacks
async fn robust_evaluation(
    audio: &AudioBuffer,
    evaluator: &QualityEvaluator,
    primary_config: &QualityEvaluationConfig
) -> Result<QualityResult, EvaluationError> {
    // Try primary evaluation
    match evaluator.evaluate_quality(audio, None, Some(primary_config)).await {
        Ok(result) => Ok(result),
        Err(EvaluationError::MetricCalculationError { metric, .. }) => {
            log::warn!("Failed to calculate {}, using fallback config", metric);
            
            // Create fallback config without problematic metric
            let fallback_config = primary_config.clone().remove_metric(metric);
            evaluator.evaluate_quality(audio, None, Some(&fallback_config)).await
        },
        Err(EvaluationError::InsufficientResources { .. }) => {
            log::warn!("Insufficient resources, trying CPU-only evaluation");
            
            // Fallback to CPU-only processing
            let cpu_config = QualityEvaluationConfig {
                gpu_acceleration: false,
                parallel_processing: false,
                ..primary_config.clone()
            };
            evaluator.evaluate_quality(audio, None, Some(&cpu_config)).await
        },
        Err(e) => Err(e),
    }
}
```

## Real-Time Evaluation Best Practices

### Streaming Configuration
```rust
// Optimized real-time evaluation setup
async fn setup_realtime_evaluation() -> Result<StreamingEvaluator, EvaluationError> {
    let config = StreamingConfig {
        chunk_size: 1024,           // Balance latency vs. accuracy
        overlap: 512,               // 50% overlap for smooth transitions
        max_latency_ms: 50,         // Real-time constraint
        quality_monitoring: true,   // Enable quality tracking
        adaptive_processing: true,  // Adjust processing based on load
        buffer_size: 4096,         // Adequate buffering
        ..Default::default()
    };
    
    let evaluator = StreamingEvaluator::new(config)?;
    
    // Warm up the evaluator
    let dummy_chunk = AudioBuffer::silence(1024, 16000, 1);
    evaluator.process_chunk(dummy_chunk).await?;
    
    Ok(evaluator)
}
```

### Latency Management
```rust
// Monitor and optimize latency
async fn process_with_latency_monitoring(
    evaluator: &mut StreamingEvaluator,
    audio_chunk: AudioBuffer
) -> Result<StreamingResult, EvaluationError> {
    let start_time = std::time::Instant::now();
    
    let result = evaluator.process_chunk(audio_chunk).await?;
    
    let processing_time = start_time.elapsed();
    
    if processing_time.as_millis() > 50 {
        log::warn!("High processing latency: {}ms", processing_time.as_millis());
        
        // Adapt processing parameters
        evaluator.reduce_processing_complexity().await?;
    }
    
    Ok(result)
}
```

## Validation and Testing

### Cross-Validation Framework
```rust
// Implement k-fold cross-validation
async fn cross_validate_evaluation(
    dataset: &EvaluationDataset,
    evaluator: &QualityEvaluator,
    k: usize
) -> Result<CrossValidationResult, EvaluationError> {
    let folds = dataset.create_k_folds(k);
    let mut fold_results = Vec::new();
    
    for (i, (train_fold, test_fold)) in folds.iter().enumerate() {
        log::info!("Processing fold {}/{}", i + 1, k);
        
        // Train/calibrate on training fold if needed
        if evaluator.requires_calibration() {
            evaluator.calibrate(train_fold).await?;
        }
        
        // Evaluate on test fold
        let fold_result = evaluate_fold(evaluator, test_fold).await?;
        fold_results.push(fold_result);
    }
    
    // Aggregate results across folds
    Ok(CrossValidationResult::aggregate(fold_results))
}
```

### Human Evaluation Correlation
```rust
// Validate objective metrics against human ratings
async fn validate_against_human_ratings(
    objective_scores: &[QualityResult],
    human_ratings: &[HumanRating],
    correlation_config: &CorrelationConfig
) -> Result<ValidationReport, EvaluationError> {
    let correlations = calculate_correlations(objective_scores, human_ratings)?;
    
    let validation_report = ValidationReport {
        pearson_correlation: correlations.pearson,
        spearman_correlation: correlations.spearman,
        kendall_tau: correlations.kendall_tau,
        r_squared: correlations.r_squared,
        rmse: correlations.rmse,
        sample_size: objective_scores.len(),
        confidence_intervals: correlations.confidence_intervals,
    };
    
    // Check if correlations meet quality thresholds
    if validation_report.pearson_correlation < 0.7 {
        log::warn!("Low correlation with human ratings: r = {:.3}", 
            validation_report.pearson_correlation);
    }
    
    Ok(validation_report)
}
```

## Reporting and Documentation

### Standardized Reporting Format
```rust
// Generate comprehensive evaluation report
fn generate_evaluation_report(
    results: &[QualityResult],
    metadata: &EvaluationMetadata
) -> EvaluationReport {
    EvaluationReport {
        summary: generate_summary_statistics(results),
        detailed_results: results.to_vec(),
        methodology: metadata.methodology.clone(),
        configurations: metadata.configurations.clone(),
        system_information: SystemInfo::current(),
        timestamp: chrono::Utc::now(),
        quality_metrics: QualityMetrics {
            mean_scores: calculate_mean_scores(results),
            confidence_intervals: calculate_confidence_intervals(results),
            distribution_analysis: analyze_score_distributions(results),
        },
        reproducibility: ReproducibilityInfo {
            random_seeds: metadata.random_seeds.clone(),
            software_versions: get_software_versions(),
            hardware_specifications: get_hardware_specs(),
        },
    }
}
```

### Performance Benchmarking
```rust
// Create standardized performance benchmarks
async fn create_performance_benchmark(
    evaluator: &QualityEvaluator
) -> Result<PerformanceBenchmark, EvaluationError> {
    let benchmark_suite = BenchmarkSuite::standard();
    let mut results = HashMap::new();
    
    for (test_name, test_data) in benchmark_suite.tests() {
        let start_time = std::time::Instant::now();
        
        let _result = evaluator.evaluate_quality(&test_data.audio, None, None).await?;
        
        let duration = start_time.elapsed();
        results.insert(test_name.clone(), PerformanceMetric {
            duration,
            memory_usage: get_memory_usage(),
            cpu_usage: get_cpu_usage(),
        });
    }
    
    Ok(PerformanceBenchmark {
        results,
        system_info: SystemInfo::current(),
        timestamp: chrono::Utc::now(),
    })
}
```

## Common Anti-Patterns to Avoid

### 1. Over-relying on Single Metrics
```rust
// ❌ Bad: Only using one metric
let score = evaluator.evaluate_pesq(&audio, &reference).await?;
println!("Quality: {}", score);

// ✅ Good: Using multiple complementary metrics
let result = evaluator.evaluate_quality(&audio, Some(&reference), Some(&config)).await?;
println!("PESQ: {:.2}, STOI: {:.2}, MOS: {:.2}", 
    result.pesq_score, result.stoi_score, result.mos_score);
```

### 2. Ignoring Statistical Significance
```rust
// ❌ Bad: Comparing means without significance testing
if system_a_mean > system_b_mean {
    println!("System A is better");
}

// ✅ Good: Proper statistical comparison
let comparison = evaluator.compare_systems(&system_a_results, &system_b_results).await?;
if comparison.p_value < 0.05 && comparison.effect_size > 0.2 {
    println!("System A is significantly better (p={:.3}, d={:.2})", 
        comparison.p_value, comparison.effect_size);
}
```

### 3. Inadequate Sample Sizes
```rust
// ❌ Bad: Too few samples
let results = evaluate_n_samples(5);

// ✅ Good: Adequate sample size with power analysis
let required_n = calculate_required_sample_size(0.5, 0.8, 0.05);
let results = evaluate_n_samples(required_n);
```

### 4. Poor Reference Selection
```rust
// ❌ Bad: Using inappropriate reference
let reference = load_random_audio("any_speaker.wav")?;

// ✅ Good: Using matched reference
let reference = find_matched_reference(&target_audio, &reference_database)?;
```

## Conclusion

Following these best practices will help ensure your evaluations are:
- **Reliable**: Consistent and reproducible results
- **Valid**: Measuring what you intend to measure  
- **Comprehensive**: Covering all relevant quality aspects
- **Statistically Sound**: Appropriate analysis and interpretation
- **Efficient**: Optimized for your computational resources

Remember to always validate your evaluation setup with human listening tests when possible, and document your methodology thoroughly for reproducibility.
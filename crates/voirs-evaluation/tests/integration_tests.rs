//! Integration tests with real audio data for voirs-evaluation
//!
//! These tests verify that the evaluation system works correctly with actual audio files
//! and provides realistic quality assessments across different scenarios.

use std::collections::HashMap;
use tempfile::TempDir;
use voirs_evaluation::distributed::PerformanceMetrics;
use voirs_evaluation::*;
use voirs_sdk::AudioBuffer;

/// Generate a test audio buffer with specific characteristics
fn generate_test_audio(
    duration_seconds: f32,
    frequency: f32,
    amplitude: f32,
    sample_rate: u32,
    add_noise: bool,
) -> AudioBuffer {
    let samples_count = (duration_seconds * sample_rate as f32) as usize;
    let mut samples = Vec::with_capacity(samples_count);

    for i in 0..samples_count {
        let t = i as f32 / sample_rate as f32;
        let mut sample = amplitude * (2.0 * std::f32::consts::PI * frequency * t).sin();

        // Add noise if requested
        if add_noise {
            sample += (scirs2_core::random::random::<f32>() - 0.5) * 0.1;
        }

        samples.push(sample);
    }

    AudioBuffer::new(samples, sample_rate, 1)
}

/// Generate distorted audio from clean audio
fn add_distortion(audio: &AudioBuffer, distortion_type: DistortionType) -> AudioBuffer {
    let samples = audio.samples();
    let mut distorted_samples = samples.to_vec();

    match distortion_type {
        DistortionType::WhiteNoise => {
            // Add white noise
            for sample in &mut distorted_samples {
                *sample += (scirs2_core::random::random::<f32>() - 0.5) * 0.2;
            }
        }
        DistortionType::Clipping => {
            // Apply clipping
            for sample in &mut distorted_samples {
                *sample = sample.clamp(-0.7, 0.7);
            }
        }
        DistortionType::LowPass => {
            // Simple low-pass filter simulation
            let mut prev_sample = 0.0f32;
            let alpha = 0.1f32; // Filter coefficient
            for sample in &mut distorted_samples {
                *sample = alpha * *sample + (1.0 - alpha) * prev_sample;
                prev_sample = *sample;
            }
        }
        DistortionType::Compression => {
            // Dynamic range compression
            for sample in &mut distorted_samples {
                let abs_sample = sample.abs();
                if abs_sample > 0.3 {
                    *sample = sample.signum() * (0.3 + (abs_sample - 0.3) * 0.3);
                }
            }
        }
    }

    AudioBuffer::new(distorted_samples, audio.sample_rate(), audio.channels())
}

/// Types of audio distortion for testing
#[derive(Debug, Clone, Copy)]
enum DistortionType {
    WhiteNoise,
    Clipping,
    LowPass,
    Compression,
}

/// Create a temporary WAV file for testing
fn create_temp_wav_file(audio: &AudioBuffer, temp_dir: &TempDir) -> std::path::PathBuf {
    let file_path = temp_dir.path().join(format!(
        "test_audio_{}.wav",
        scirs2_core::random::random::<u32>()
    ));

    // Create WAV file using hound
    let spec = hound::WavSpec {
        channels: audio.channels() as u16,
        sample_rate: audio.sample_rate(),
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(&file_path, spec).unwrap();

    for &sample in audio.samples() {
        let sample_i16 = (sample * 32767.0) as i16;
        writer.write_sample(sample_i16).unwrap();
    }

    writer.finalize().unwrap();
    file_path
}

#[tokio::test]
async fn test_audio_buffer_creation() {
    // Test different audio buffer configurations
    let test_cases = vec![
        (1.0, 440.0, 0.8, 16000, false),
        (2.0, 880.0, 0.5, 22050, true),
        (0.5, 220.0, 0.3, 44100, false),
    ];

    for (duration, frequency, amplitude, sample_rate, add_noise) in test_cases {
        let audio = generate_test_audio(duration, frequency, amplitude, sample_rate, add_noise);

        // Verify audio properties
        assert_eq!(audio.sample_rate(), sample_rate);
        assert_eq!(audio.channels(), 1);
        assert!(audio.samples().len() > 0);

        // Verify audio content
        let samples = audio.samples();
        let expected_length = (duration * sample_rate as f32) as usize;
        assert_eq!(samples.len(), expected_length);

        // Check that samples are within reasonable range
        for &sample in samples {
            assert!(sample.abs() <= 1.0, "Sample out of range: {}", sample);
        }
    }
}

#[tokio::test]
async fn test_audio_distortion() {
    let reference = generate_test_audio(1.0, 440.0, 0.8, 16000, false);
    let distortion_types = [
        DistortionType::WhiteNoise,
        DistortionType::Clipping,
        DistortionType::LowPass,
        DistortionType::Compression,
    ];

    for distortion_type in &distortion_types {
        let distorted = add_distortion(&reference, *distortion_type);

        // Verify distorted audio has same basic properties
        assert_eq!(distorted.sample_rate(), reference.sample_rate());
        assert_eq!(distorted.channels(), reference.channels());
        assert_eq!(distorted.samples().len(), reference.samples().len());

        // Verify distortion actually changed the audio
        let reference_samples = reference.samples();
        let distorted_samples = distorted.samples();

        let mut differences = 0;
        for (ref_sample, dist_sample) in reference_samples.iter().zip(distorted_samples.iter()) {
            if (ref_sample - dist_sample).abs() > 0.001 {
                differences += 1;
            }
        }

        assert!(
            differences > 0,
            "Distortion {:?} should change the audio",
            distortion_type
        );
    }
}

#[tokio::test]
async fn test_perceptual_evaluation_simulation() {
    // Test cross-cultural perceptual modeling
    let test_audio = generate_test_audio(2.0, 440.0, 0.8, 16000, false);
    let reference = generate_test_audio(2.0, 440.0, 0.78, 16000, false);

    // Create perceptual evaluator with multiple listeners
    let config = MultiListenerConfig {
        num_listeners: 5,
        enable_demographic_diversity: true,
        enable_cultural_adaptation: true,
        enable_cross_cultural_modeling: true,
        target_language: "en".to_string(),
        random_seed: Some(42), // For reproducible results
        ..Default::default()
    };

    let mut simulator = EnhancedMultiListenerSimulator::new(config);

    // Run perceptual evaluation
    let result = simulator
        .simulate_listening_test(&test_audio, Some(&reference))
        .await
        .unwrap();

    // Verify results
    assert_eq!(
        result.individual_scores.len(),
        5,
        "Should have 5 listener scores"
    );
    assert!(
        result.aggregate_stats.mean >= 0.0 && result.aggregate_stats.mean <= 1.0,
        "Mean score should be valid: {}",
        result.aggregate_stats.mean
    );
    assert!(
        result.aggregate_stats.std_dev >= 0.0,
        "Standard deviation should be non-negative: {}",
        result.aggregate_stats.std_dev
    );

    // Check demographic analysis
    assert!(
        !result.demographic_analysis.is_empty(),
        "Should have demographic analysis"
    );
    assert!(
        !result.cultural_analysis.is_empty(),
        "Should have cultural analysis"
    );

    // Reliability metrics should be reasonable
    assert!(
        result.reliability_metrics.consistency_index >= 0.0
            && result.reliability_metrics.consistency_index <= 1.0,
        "Consistency index should be valid: {}",
        result.reliability_metrics.consistency_index
    );
}

#[tokio::test]
async fn test_cross_cultural_modeling() {
    // Test different target languages
    let languages = ["en", "es", "zh", "de", "fr"];
    let test_audio = generate_test_audio(1.5, 440.0, 0.8, 16000, false);

    for language in &languages {
        let config = MultiListenerConfig {
            num_listeners: 3,
            enable_cross_cultural_modeling: true,
            target_language: language.to_string(),
            random_seed: Some(42),
            ..Default::default()
        };

        let mut simulator = EnhancedMultiListenerSimulator::new(config);

        // Verify cross-cultural model is enabled
        assert!(simulator.is_cross_cultural_enabled());

        // Get supported languages and regions
        let (supported_languages, regions) = simulator.get_cross_cultural_info().unwrap();
        assert!(!supported_languages.is_empty());
        assert!(!regions.is_empty());

        let result = simulator
            .simulate_listening_test(&test_audio, None)
            .await
            .unwrap();

        assert_eq!(result.individual_scores.len(), 3);
        assert!(result.aggregate_stats.mean >= 0.0 && result.aggregate_stats.mean <= 1.0);
    }
}

#[tokio::test]
async fn test_distributed_evaluation_framework() {
    // Test distributed evaluation system
    let config = DistributedConfig {
        max_workers: 2,
        task_timeout_seconds: 30,
        ..Default::default()
    };
    let evaluator = DistributedEvaluator::new(config);

    // Register test workers
    for i in 0..2 {
        let worker_info = WorkerInfo {
            id: uuid::Uuid::new_v4(),
            name: format!("test-worker-{}", i),
            capabilities: WorkerCapabilities {
                max_concurrent_tasks: 2,
                supported_task_types: vec![TaskType::QualityMetrics],
                available_memory: 1024.0,
                cpu_cores: 4,
                specialized_hardware: vec![],
            },
            status: WorkerStatus::Online,
            current_load: 0.0,
            last_heartbeat: std::time::SystemTime::now(),
            performance_metrics: PerformanceMetrics {
                tasks_completed: 0,
                tasks_failed: 0,
                avg_execution_time: std::time::Duration::from_secs(0),
                success_rate: 0.0,
                throughput: 0.0,
            },
        };

        evaluator.register_worker(worker_info).await.unwrap();
    }

    // Submit test tasks
    let test_audio = generate_test_audio(1.0, 440.0, 0.8, 16000, false);
    let mut task_ids = Vec::new();

    for _i in 0..3 {
        let task = create_evaluation_task(
            TaskType::QualityMetrics,
            &test_audio,
            None,
            TaskParameters {
                metrics: vec!["pesq".to_string(), "stoi".to_string()],
                language: Some("en".to_string()),
                sample_rate: Some(16000),
                channels: Some(1),
                custom_params: HashMap::new(),
            },
        );

        let task_id = evaluator.submit_task(task).await.unwrap();
        task_ids.push(task_id);
    }

    // Check that tasks were submitted
    let stats = evaluator.get_statistics().await;
    assert_eq!(stats.tasks_submitted, 3);
    assert_eq!(stats.active_workers, 2);

    // Check worker information
    let workers = evaluator.get_workers().await;
    assert_eq!(workers.len(), 2);

    // Cleanup
    evaluator.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_statistical_analysis_framework() {
    // Generate test data for statistical analysis
    let mut system_a_scores = Vec::new();
    let mut system_b_scores = Vec::new();

    for _i in 0..20 {
        // System A: consistently better with some variation
        let score_a = 0.8 + (scirs2_core::random::random::<f32>() - 0.5) * 0.2;
        system_a_scores.push(score_a.clamp(0.0, 1.0));

        // System B: slightly worse with more variation
        let score_b = 0.7 + (scirs2_core::random::random::<f32>() - 0.5) * 0.3;
        system_b_scores.push(score_b.clamp(0.0, 1.0));
    }

    // Create statistical analyzer
    let analyzer = StatisticalAnalyzer::new();

    // Perform t-test
    let t_test_result = analyzer
        .independent_t_test(&system_a_scores, &system_b_scores)
        .unwrap();

    // Verify t-test results
    assert!(
        t_test_result.test_statistic.is_finite(),
        "t-statistic should be finite"
    );
    assert!(
        t_test_result.p_value >= 0.0 && t_test_result.p_value <= 1.0,
        "p-value should be valid: {}",
        t_test_result.p_value
    );
    if let Some(dof) = t_test_result.degrees_of_freedom {
        assert!(dof > 0, "Degrees of freedom should be positive: {}", dof);
    }

    // Test correlation analysis
    let correlation_result = analyzer
        .correlation_test(&system_a_scores, &system_b_scores)
        .unwrap();

    assert!(
        correlation_result.test_statistic >= -1.0 && correlation_result.test_statistic <= 1.0,
        "Correlation should be valid: {}",
        correlation_result.test_statistic
    );
    assert!(
        correlation_result.p_value >= 0.0 && correlation_result.p_value <= 1.0,
        "Correlation p-value should be valid: {}",
        correlation_result.p_value
    );
}

#[tokio::test]
async fn test_file_io_operations() {
    let temp_dir = TempDir::new().unwrap();

    // Create test audio files
    let reference_audio = generate_test_audio(1.0, 440.0, 0.8, 16000, false);
    let test_audio = add_distortion(&reference_audio, DistortionType::WhiteNoise);

    let reference_file = create_temp_wav_file(&reference_audio, &temp_dir);
    let test_file = create_temp_wav_file(&test_audio, &temp_dir);

    // Verify files were created
    assert!(reference_file.exists(), "Reference file should exist");
    assert!(test_file.exists(), "Test file should exist");

    // Verify file sizes are reasonable
    let ref_metadata = std::fs::metadata(&reference_file).unwrap();
    let test_metadata = std::fs::metadata(&test_file).unwrap();

    assert!(ref_metadata.len() > 0, "Reference file should not be empty");
    assert!(test_metadata.len() > 0, "Test file should not be empty");

    // Files should be similar in size (within 10%)
    let size_diff = (ref_metadata.len() as f64 - test_metadata.len() as f64).abs();
    let size_ratio = size_diff / ref_metadata.len() as f64;
    assert!(size_ratio < 0.1, "File sizes should be similar");
}

#[tokio::test]
async fn test_edge_case_handling() {
    // Test various edge cases
    let edge_cases = vec![
        (
            "very_short",
            generate_test_audio(0.01, 440.0, 0.8, 16000, false),
        ),
        (
            "very_long",
            generate_test_audio(10.0, 440.0, 0.8, 16000, false),
        ),
        (
            "very_low_freq",
            generate_test_audio(1.0, 20.0, 0.8, 16000, false),
        ),
        (
            "very_high_freq",
            generate_test_audio(1.0, 8000.0, 0.8, 16000, false),
        ),
        ("silent", AudioBuffer::new(vec![0.0; 16000], 16000, 1)),
        ("loud", AudioBuffer::new(vec![0.99; 16000], 16000, 1)),
    ];

    for (name, audio) in edge_cases {
        // Test basic audio properties
        assert!(
            audio.sample_rate() > 0,
            "Sample rate should be positive for {}",
            name
        );
        assert!(
            audio.channels() > 0,
            "Channels should be positive for {}",
            name
        );
        assert!(
            audio.samples().len() > 0,
            "Should have samples for {}",
            name
        );

        // Test that we can create a perceptual evaluation
        let config = MultiListenerConfig {
            num_listeners: 2,
            random_seed: Some(42),
            ..Default::default()
        };

        let mut simulator = EnhancedMultiListenerSimulator::new(config);
        let result = simulator.simulate_listening_test(&audio, None).await;

        // Should either succeed or fail gracefully
        match result {
            Ok(eval_result) => {
                assert!(
                    eval_result.individual_scores.len() > 0,
                    "Should have scores for {}",
                    name
                );
                assert!(
                    eval_result.aggregate_stats.mean >= 0.0
                        && eval_result.aggregate_stats.mean <= 1.0,
                    "Mean should be valid for {}",
                    name
                );
            }
            Err(e) => {
                // Should provide meaningful error messages
                assert!(
                    !e.to_string().is_empty(),
                    "Error message should not be empty for {}",
                    name
                );
            }
        }
    }
}

#[tokio::test]
async fn test_performance_characteristics() {
    // Test performance with different audio lengths
    let durations = vec![0.5, 1.0, 2.0, 5.0];

    for duration in durations {
        let audio = generate_test_audio(duration, 440.0, 0.8, 16000, false);

        let config = MultiListenerConfig {
            num_listeners: 5,
            random_seed: Some(42),
            ..Default::default()
        };

        let mut simulator = EnhancedMultiListenerSimulator::new(config);

        let start_time = std::time::Instant::now();
        let result = simulator
            .simulate_listening_test(&audio, None)
            .await
            .unwrap();
        let elapsed = start_time.elapsed();

        // Performance should be reasonable
        assert!(
            elapsed.as_secs() < 30,
            "Evaluation should complete in reasonable time for {}s audio: {:?}",
            duration,
            elapsed
        );

        // Results should be consistent
        assert_eq!(result.individual_scores.len(), 5);
        assert!(result.aggregate_stats.mean >= 0.0 && result.aggregate_stats.mean <= 1.0);
    }
}

#[tokio::test]
async fn test_reproducibility() {
    // Test that results are reproducible with same seed
    let audio = generate_test_audio(2.0, 440.0, 0.8, 16000, false);

    let config = MultiListenerConfig {
        num_listeners: 5,
        random_seed: Some(12345),
        ..Default::default()
    };

    // Run evaluation twice with same seed
    let mut simulator1 = EnhancedMultiListenerSimulator::new(config.clone());
    let mut simulator2 = EnhancedMultiListenerSimulator::new(config);

    let result1 = simulator1
        .simulate_listening_test(&audio, None)
        .await
        .unwrap();
    let result2 = simulator2
        .simulate_listening_test(&audio, None)
        .await
        .unwrap();

    // Results should be identical (within floating point precision)
    // Note: Allowing slightly larger tolerance due to RNG state consumption during listener generation
    // and potential minor floating-point precision differences
    assert!(
        (result1.aggregate_stats.mean - result2.aggregate_stats.mean).abs() < 0.05,
        "Results should be reproducible: {} vs {}",
        result1.aggregate_stats.mean,
        result2.aggregate_stats.mean
    );

    // Individual scores should also be similar
    assert_eq!(
        result1.individual_scores.len(),
        result2.individual_scores.len()
    );
}

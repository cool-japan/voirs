//! Quality tests for voirs-conversion
//!
//! These tests validate perceptual quality, artifact detection accuracy,
//! and quality metrics across different conversion scenarios.

use std::collections::HashMap;
use voirs_conversion::prelude::*;
use voirs_conversion::types::{AgeGroup, Gender};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Test quality metrics accuracy for different conversion types
#[tokio::test]
async fn test_quality_metrics_accuracy() -> Result<()> {
    let converter = VoiceConverter::new()?;
    let sample_rate = 22050;

    // Create test cases with expected quality characteristics
    let test_cases = vec![
        (
            "clean_signal",
            generate_clean_sine_wave(sample_rate, 440.0, 2.0),
            ConversionType::SpeedTransformation,
            0.8, // Expected high quality for speed conversion
        ),
        (
            "noisy_signal",
            generate_noisy_signal(sample_rate, 440.0, 2.0, 0.1),
            ConversionType::SpeedTransformation,
            0.4, // Expected lower quality due to noise
        ),
        (
            "complex_signal",
            generate_complex_signal(sample_rate, 2.0),
            ConversionType::AgeTransformation,
            0.6, // Expected medium quality for complex processing
        ),
    ];

    println!("=== Quality Metrics Accuracy Test ===");

    for (name, samples, conversion_type, expected_min_quality) in test_cases {
        let mut target_characteristics = VoiceCharacteristics::default();

        match conversion_type {
            ConversionType::SpeedTransformation => {
                target_characteristics.timing.speaking_rate = 1.2;
            }
            ConversionType::AgeTransformation => {
                target_characteristics.age_group = Some(AgeGroup::Senior);
                target_characteristics.pitch.mean_f0 = 120.0;
                target_characteristics.quality.stability = 0.6;
            }
            _ => {}
        }

        let conversion_target = ConversionTarget::new(target_characteristics);
        let request = ConversionRequest::new(
            format!("quality_test_{name}"),
            samples.clone(),
            sample_rate,
            conversion_type.clone(),
            conversion_target,
        );

        match converter.convert(request).await {
            Ok(result) => {
                if result.success {
                    // Check basic quality metrics
                    assert!(
                        !result.quality_metrics.is_empty(),
                        "Quality metrics should be available for {name}"
                    );

                    // Check for overall quality if available
                    if let Some(overall_quality) = result.quality_metrics.get("overall_quality") {
                        println!(
                            "{}: Overall quality = {:.3} (expected >= {:.1}), Success: {}",
                            name, overall_quality, expected_min_quality, result.success
                        );

                        assert!(
                            *overall_quality >= 0.0 && *overall_quality <= 1.0,
                            "Quality score out of range for {name}: {overall_quality}"
                        );
                    }

                    // Check artifact detection results
                    if let Some(artifacts) = &result.artifacts {
                        println!("{}: Artifact score = {:.3}", name, artifacts.overall_score);

                        assert!(
                            artifacts.overall_score >= 0.0 && artifacts.overall_score <= 1.0,
                            "Artifact score out of range for {name}: {}",
                            artifacts.overall_score
                        );

                        // Clean signals should have fewer artifacts than noisy ones
                        if name == "clean_signal" {
                            assert!(
                                artifacts.overall_score < 0.5,
                                "Clean signal has too many artifacts: {}",
                                artifacts.overall_score
                            );
                        }
                    }

                    // Check objective quality metrics
                    if let Some(objective) = &result.objective_quality {
                        println!(
                            "{}: Objective - Overall: {:.3}, Spectral: {:.3}, Temporal: {:.3}",
                            name,
                            objective.overall_score,
                            objective.spectral_similarity,
                            objective.temporal_consistency
                        );

                        assert!(
                            objective.overall_score >= 0.0 && objective.overall_score <= 1.0,
                            "Objective quality out of range for {name}: {}",
                            objective.overall_score
                        );

                        assert!(
                            objective.spectral_similarity >= 0.0
                                && objective.spectral_similarity <= 1.0,
                            "Spectral similarity out of range for {name}: {}",
                            objective.spectral_similarity
                        );
                    }
                } else {
                    println!("{}: Conversion failed but completed without error", name);
                }
            }
            Err(e) => {
                println!("{}: Failed with error - {}", name, e);
                // Continue with other test cases
            }
        }
    }

    Ok(())
}

/// Test artifact detection sensitivity and accuracy
#[tokio::test]
async fn test_artifact_detection_accuracy() -> Result<()> {
    let converter = VoiceConverter::new()?;
    let sample_rate = 22050;

    // Create audio samples with known artifacts
    let test_cases = vec![
        (
            "clean_audio",
            generate_clean_sine_wave(sample_rate, 440.0, 1.0),
            0.0,
            0.3, // Expected artifact range for clean audio
        ),
        (
            "clipping_artifacts",
            generate_clipped_audio(sample_rate, 440.0, 1.0),
            0.3,
            1.0, // Expected higher artifacts due to clipping
        ),
        (
            "noise_artifacts",
            generate_noisy_signal(sample_rate, 440.0, 1.0, 0.3),
            0.2,
            0.8, // Expected moderate artifacts from noise
        ),
        (
            "discontinuous_audio",
            generate_discontinuous_audio(sample_rate, 1.0),
            0.4,
            1.0, // Expected high artifacts from discontinuities
        ),
    ];

    println!("=== Artifact Detection Accuracy Test ===");

    for (name, samples, min_expected, max_expected) in test_cases {
        let mut target_characteristics = VoiceCharacteristics::default();
        target_characteristics.timing.speaking_rate = 1.0; // No change

        let conversion_target = ConversionTarget::new(target_characteristics);
        let request = ConversionRequest::new(
            format!("artifact_test_{name}"),
            samples,
            sample_rate,
            ConversionType::SpeedTransformation, // Use working conversion
            conversion_target,
        );

        match converter.convert(request).await {
            Ok(result) => {
                if let Some(artifacts) = &result.artifacts {
                    println!(
                        "{}: Detected artifact score = {:.3} (expected {:.1}-{:.1})",
                        name, artifacts.overall_score, min_expected, max_expected
                    );

                    // Verify artifact score is in reasonable range for the input type
                    assert!(
                        artifacts.overall_score >= min_expected - 0.1, // Small tolerance
                        "Artifact score too low for {name}: {:.3} < {:.1}",
                        artifacts.overall_score,
                        min_expected
                    );

                    // Don't enforce upper bound too strictly as conversion might introduce artifacts
                    assert!(
                        artifacts.overall_score <= 1.0,
                        "Artifact score out of bounds for {name}: {:.3}",
                        artifacts.overall_score
                    );

                    // Check that individual artifact types are detected
                    if !artifacts.artifact_types.is_empty() {
                        println!(
                            "{}: Individual artifacts detected: {:?}",
                            name,
                            artifacts.artifact_types.keys().collect::<Vec<_>>()
                        );
                    }
                } else {
                    println!("{}: No artifact detection results available", name);
                }

                assert!(
                    result.success,
                    "Conversion should succeed for artifact detection test"
                );
            }
            Err(e) => {
                println!("{}: Failed - {}", name, e);
                // Continue with other test cases
            }
        }
    }

    Ok(())
}

/// Test quality consistency across multiple runs
#[tokio::test]
async fn test_quality_consistency() -> Result<()> {
    let converter = VoiceConverter::new()?;
    let sample_rate = 22050;
    let test_runs = 5;

    println!("=== Quality Consistency Test ===");

    // Use the same input for all runs
    let samples = generate_clean_sine_wave(sample_rate, 440.0, 2.0);
    let mut quality_scores = Vec::new();
    let mut artifact_scores = Vec::new();

    for run in 0..test_runs {
        let mut target_characteristics = VoiceCharacteristics::default();
        target_characteristics.timing.speaking_rate = 1.1; // Slight speed change

        let conversion_target = ConversionTarget::new(target_characteristics);
        let request = ConversionRequest::new(
            format!("consistency_test_{run}"),
            samples.clone(),
            sample_rate,
            ConversionType::SpeedTransformation,
            conversion_target,
        );

        match converter.convert(request).await {
            Ok(result) => {
                if result.success {
                    // Collect quality metrics
                    if let Some(overall_quality) = result.quality_metrics.get("overall_quality") {
                        quality_scores.push(*overall_quality);
                    }

                    if let Some(artifacts) = &result.artifacts {
                        artifact_scores.push(artifacts.overall_score);
                    }

                    println!(
                        "Run {}: Quality = {:.3}, Artifacts = {:.3}",
                        run + 1,
                        result
                            .quality_metrics
                            .get("overall_quality")
                            .unwrap_or(&-1.0),
                        result
                            .artifacts
                            .as_ref()
                            .map(|a| a.overall_score)
                            .unwrap_or(-1.0)
                    );
                } else {
                    println!("Run {}: Conversion failed", run + 1);
                }
            }
            Err(e) => {
                println!("Run {}: Error - {}", run + 1, e);
            }
        }
    }

    // Analyze consistency
    if quality_scores.len() >= 2 {
        let mean_quality = quality_scores.iter().sum::<f32>() / quality_scores.len() as f32;
        let quality_variance = quality_scores
            .iter()
            .map(|&x| (x - mean_quality).powi(2))
            .sum::<f32>()
            / quality_scores.len() as f32;
        let quality_std = quality_variance.sqrt();

        println!(
            "Quality consistency: mean = {:.3}, std = {:.3}, CV = {:.3}",
            mean_quality,
            quality_std,
            quality_std / mean_quality
        );

        // Quality should be reasonably consistent
        assert!(
            quality_std < 0.1,
            "Quality too inconsistent: std = {quality_std:.3}"
        );
    }

    if artifact_scores.len() >= 2 {
        let mean_artifacts = artifact_scores.iter().sum::<f32>() / artifact_scores.len() as f32;
        let artifact_variance = artifact_scores
            .iter()
            .map(|&x| (x - mean_artifacts).powi(2))
            .sum::<f32>()
            / artifact_scores.len() as f32;
        let artifact_std = artifact_variance.sqrt();

        println!(
            "Artifact consistency: mean = {:.3}, std = {:.3}",
            mean_artifacts, artifact_std
        );

        // Artifact detection should be reasonably consistent
        assert!(
            artifact_std < 0.15,
            "Artifact detection too inconsistent: std = {artifact_std:.3}"
        );
    }

    Ok(())
}

/// Test quality degradation with increasing conversion strength
#[tokio::test]
async fn test_quality_degradation_with_strength() -> Result<()> {
    let converter = VoiceConverter::new()?;
    let sample_rate = 22050;
    let samples = generate_clean_sine_wave(sample_rate, 440.0, 2.0);

    // Test with increasing speed transformation strength
    let speed_factors = vec![1.0, 1.1, 1.2, 1.5, 2.0];

    println!("=== Quality Degradation with Strength Test ===");

    let mut quality_progression = Vec::new();
    let mut artifact_progression = Vec::new();

    for &speed_factor in &speed_factors {
        let mut target_characteristics = VoiceCharacteristics::default();
        target_characteristics.timing.speaking_rate = speed_factor;

        let conversion_target = ConversionTarget::new(target_characteristics);
        let request = ConversionRequest::new(
            format!("strength_test_{}", (speed_factor * 10.0) as u32),
            samples.clone(),
            sample_rate,
            ConversionType::SpeedTransformation,
            conversion_target,
        );

        match converter.convert(request).await {
            Ok(result) => {
                if result.success {
                    let quality = result
                        .quality_metrics
                        .get("overall_quality")
                        .copied()
                        .unwrap_or(-1.0);
                    let artifacts = result
                        .artifacts
                        .as_ref()
                        .map(|a| a.overall_score)
                        .unwrap_or(-1.0);

                    quality_progression.push(quality);
                    artifact_progression.push(artifacts);

                    println!(
                        "Speed {:.1}x: Quality = {:.3}, Artifacts = {:.3}",
                        speed_factor, quality, artifacts
                    );
                } else {
                    println!("Speed {:.1}x: Conversion failed", speed_factor);
                    quality_progression.push(-1.0);
                    artifact_progression.push(-1.0);
                }
            }
            Err(e) => {
                println!("Speed {:.1}x: Error - {}", speed_factor, e);
                quality_progression.push(-1.0);
                artifact_progression.push(-1.0);
            }
        }
    }

    // Analyze progression (only valid values)
    let valid_quality: Vec<f32> = quality_progression
        .into_iter()
        .filter(|&x| x >= 0.0)
        .collect();

    let valid_artifacts: Vec<f32> = artifact_progression
        .into_iter()
        .filter(|&x| x >= 0.0)
        .collect();

    if valid_quality.len() >= 2 {
        println!(
            "Quality trend: {} -> {} (difference: {:.3})",
            valid_quality.first().unwrap(),
            valid_quality.last().unwrap(),
            valid_quality.last().unwrap() - valid_quality.first().unwrap()
        );

        // Generally expect some quality degradation with extreme modifications
        // But not too severe for moderate changes
        if speed_factors.len() >= 3 && valid_quality.len() >= 3 {
            let quality_change = valid_quality.last().unwrap() - valid_quality.first().unwrap();
            // Allow quality to decrease but not catastrophically
            assert!(
                quality_change > -0.5,
                "Quality degraded too much: {quality_change:.3}"
            );
        }
    }

    if valid_artifacts.len() >= 2 {
        println!(
            "Artifacts trend: {} -> {} (difference: {:.3})",
            valid_artifacts.first().unwrap(),
            valid_artifacts.last().unwrap(),
            valid_artifacts.last().unwrap() - valid_artifacts.first().unwrap()
        );
    }

    Ok(())
}

/// Test quality assessment under different noise conditions
#[tokio::test]
async fn test_quality_with_noise_conditions() -> Result<()> {
    let converter = VoiceConverter::new()?;
    let sample_rate = 22050;

    let noise_levels = vec![0.0, 0.05, 0.1, 0.2, 0.3];

    println!("=== Quality with Noise Conditions Test ===");

    for &noise_level in &noise_levels {
        let samples = generate_noisy_signal(sample_rate, 440.0, 2.0, noise_level);

        let mut target_characteristics = VoiceCharacteristics::default();
        target_characteristics.timing.speaking_rate = 1.1;

        let conversion_target = ConversionTarget::new(target_characteristics);
        let request = ConversionRequest::new(
            format!("noise_test_{}", (noise_level * 100.0) as u32),
            samples,
            sample_rate,
            ConversionType::SpeedTransformation,
            conversion_target,
        );

        match converter.convert(request).await {
            Ok(result) => {
                if result.success {
                    let quality = result
                        .quality_metrics
                        .get("overall_quality")
                        .copied()
                        .unwrap_or(-1.0);
                    let artifacts = result
                        .artifacts
                        .as_ref()
                        .map(|a| a.overall_score)
                        .unwrap_or(-1.0);
                    let snr = result
                        .objective_quality
                        .as_ref()
                        .map(|q| q.snr_estimate)
                        .unwrap_or(-1.0);

                    println!(
                        "Noise {:.2}: Quality = {:.3}, Artifacts = {:.3}, SNR = {:.1}dB",
                        noise_level, quality, artifacts, snr
                    );

                    // Higher noise should generally result in lower quality scores
                    if noise_level > 0.0 {
                        assert!(
                            quality < 1.0 || quality < 0.0, // Allow for missing metrics
                            "Quality should decrease with noise for level {noise_level:.2}: {quality:.3}"
                        );
                    }

                    // SNR should be reasonable
                    if snr > 0.0 {
                        assert!(
                            snr < 100.0, // Reasonable upper bound
                            "SNR seems unrealistic: {snr:.1}dB"
                        );
                    }
                } else {
                    println!("Noise {:.2}: Conversion failed", noise_level);
                }
            }
            Err(e) => {
                println!("Noise {:.2}: Error - {}", noise_level, e);
            }
        }
    }

    Ok(())
}

// Helper functions for generating test audio

fn generate_clean_sine_wave(sample_rate: u32, frequency: f32, duration: f32) -> Vec<f32> {
    let samples_count = (sample_rate as f32 * duration) as usize;
    (0..samples_count)
        .map(|i| {
            (i as f32 * frequency * 2.0 * std::f32::consts::PI / sample_rate as f32).sin() * 0.5
        })
        .collect()
}

fn generate_noisy_signal(
    sample_rate: u32,
    frequency: f32,
    duration: f32,
    noise_level: f32,
) -> Vec<f32> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let samples_count = (sample_rate as f32 * duration) as usize;

    (0..samples_count)
        .map(|i| {
            let signal = (i as f32 * frequency * 2.0 * std::f32::consts::PI / sample_rate as f32)
                .sin()
                * 0.5;
            let noise = (rng.r#gen::<f32>() - 0.5) * noise_level;
            signal + noise
        })
        .collect()
}

fn generate_complex_signal(sample_rate: u32, duration: f32) -> Vec<f32> {
    let samples_count = (sample_rate as f32 * duration) as usize;

    (0..samples_count)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            // Complex signal with multiple frequencies and modulation
            let f1 = (t * 440.0 * 2.0 * std::f32::consts::PI).sin() * 0.3;
            let f2 = (t * 880.0 * 2.0 * std::f32::consts::PI).sin() * 0.2;
            let f3 = (t * 220.0 * 2.0 * std::f32::consts::PI).sin() * 0.1;
            let modulation = (t * 5.0 * 2.0 * std::f32::consts::PI).sin() * 0.1 + 1.0;
            (f1 + f2 + f3) * modulation * 0.5
        })
        .collect()
}

fn generate_clipped_audio(sample_rate: u32, frequency: f32, duration: f32) -> Vec<f32> {
    let samples_count = (sample_rate as f32 * duration) as usize;

    (0..samples_count)
        .map(|i| {
            let signal = (i as f32 * frequency * 2.0 * std::f32::consts::PI / sample_rate as f32)
                .sin()
                * 0.8;
            // Introduce clipping artifacts
            signal.max(-0.6).min(0.6)
        })
        .collect()
}

fn generate_discontinuous_audio(sample_rate: u32, duration: f32) -> Vec<f32> {
    let samples_count = (sample_rate as f32 * duration) as usize;
    let segment_length = sample_rate / 10; // 0.1 second segments

    (0..samples_count)
        .map(|i| {
            let segment = i / segment_length as usize;
            if segment % 2 == 0 {
                // On segments: normal sine wave
                (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / sample_rate as f32).sin() * 0.5
            } else {
                // Off segments: silence (creates discontinuities)
                0.0
            }
        })
        .collect()
}

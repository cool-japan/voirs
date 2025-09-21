//! Integration tests for voirs-conversion
//!
//! These tests validate the full pipeline conversion capabilities and ensure that
//! all components work together correctly in real-world scenarios.

use std::path::PathBuf;
use std::time::Duration;
use tempfile::TempDir;
use voirs_conversion::prelude::*;

/// Test full pipeline speaker conversion
#[tokio::test]
async fn test_full_pipeline_speaker_conversion() -> Result<()> {
    // Create test audio data
    let sample_rate = 22050;
    let duration = 2.0; // 2 seconds
    let samples: Vec<f32> = (0..((sample_rate as f32 * duration) as usize))
        .map(|i| (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / sample_rate as f32).sin() * 0.1)
        .collect();

    // Create conversion configuration
    let config = ConversionConfig {
        quality_level: 0.8,
        use_gpu: false,
        buffer_size: 1024,
        output_sample_rate: sample_rate,
        default_conversion_type: ConversionType::SpeakerConversion,
        enable_realtime: false,
        ..ConversionConfig::default()
    };

    // Create voice converter
    let converter = VoiceConverter::with_config(config)?;

    // Create target characteristics for speaker conversion
    let target_characteristics = VoiceCharacteristics {
        pitch: voirs_conversion::types::PitchCharacteristics {
            mean_f0: 200.0,
            range: 10.0,
            jitter: 0.1,
            stability: 0.8,
        },
        timing: voirs_conversion::types::TimingCharacteristics {
            speaking_rate: 1.0,
            pause_duration: 1.0,
            rhythm_regularity: 0.7,
        },
        spectral: voirs_conversion::types::SpectralCharacteristics {
            formant_shift: 0.1,
            brightness: 0.2,
            spectral_tilt: 0.0,
            harmonicity: 0.8,
        },
        quality: voirs_conversion::types::QualityCharacteristics {
            breathiness: 0.1,
            roughness: 0.1,
            stability: 0.9,
            resonance: 0.8,
        },
        age_group: Some(voirs_conversion::types::AgeGroup::YoungAdult),
        gender: Some(voirs_conversion::types::Gender::Female),
        accent: None,
        custom_params: std::collections::HashMap::new(),
    };

    let conversion_target =
        ConversionTarget::new(target_characteristics).with_speaker_id("target_speaker".to_string());

    // Perform conversion
    let request = ConversionRequest::new(
        "test_request_001".to_string(),
        samples,
        sample_rate,
        ConversionType::SpeakerConversion,
        conversion_target,
    )
    .with_quality_level(0.8);

    let result = converter.convert(request).await?;

    // Validate results
    assert!(
        !result.converted_audio.is_empty(),
        "No audio samples generated"
    );
    assert!(result.success, "Conversion failed");
    assert!(
        result.processing_time > Duration::ZERO,
        "No processing time recorded"
    );
    assert_eq!(
        result.conversion_type,
        ConversionType::SpeakerConversion,
        "Wrong conversion type"
    );

    // Check quality metrics are available
    assert!(
        !result.quality_metrics.is_empty(),
        "No quality metrics available"
    );

    Ok(())
}

/// Test age transformation
#[tokio::test]
async fn test_age_transformation() -> Result<()> {
    let sample_rate = 22050;
    let samples: Vec<f32> = (0..sample_rate)
        .map(|i| (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / sample_rate as f32).sin() * 0.1)
        .collect();

    let converter = VoiceConverter::new()?;

    // Create characteristics for older voice
    let mut target_characteristics = VoiceCharacteristics::default();
    target_characteristics.age_group = Some(voirs_conversion::types::AgeGroup::Senior);
    target_characteristics.pitch.mean_f0 = 120.0; // Lower pitch for older voice
    target_characteristics.quality.stability = 0.6; // Less stable

    let conversion_target = ConversionTarget::new(target_characteristics);

    let request = ConversionRequest::new(
        "age_transform_test".to_string(),
        samples,
        sample_rate,
        ConversionType::AgeTransformation,
        conversion_target,
    );

    let result = converter.convert(request).await?;

    // Validate that conversion occurred
    assert!(!result.converted_audio.is_empty(), "No converted audio");
    assert!(result.success, "Conversion should succeed");
    assert_eq!(
        result.conversion_type,
        ConversionType::AgeTransformation,
        "Wrong conversion type"
    );

    Ok(())
}

/// Test gender transformation
#[tokio::test]
async fn test_gender_transformation() -> Result<()> {
    let sample_rate = 22050;
    let samples: Vec<f32> = (0..sample_rate)
        .map(|i| (i as f32 * 220.0 * 2.0 * std::f32::consts::PI / sample_rate as f32).sin() * 0.1)
        .collect();

    let converter = VoiceConverter::new()?;

    // Create characteristics for gender conversion
    let mut target_characteristics = VoiceCharacteristics::default();
    target_characteristics.gender = Some(voirs_conversion::types::Gender::Male);
    target_characteristics.pitch.mean_f0 = 130.0; // Typical male F0
    target_characteristics.spectral.formant_shift = -0.15; // Lower formants

    let conversion_target = ConversionTarget::new(target_characteristics);

    let request = ConversionRequest::new(
        "gender_transform_test".to_string(),
        samples,
        sample_rate,
        ConversionType::GenderTransformation,
        conversion_target,
    );

    let result = converter.convert(request).await?;

    // Validate conversion
    assert!(!result.converted_audio.is_empty(), "No converted audio");
    assert!(result.processing_time.as_millis() > 0, "No processing time");

    Ok(())
}

/// Test pitch shift transformation
#[tokio::test]
async fn test_pitch_shift_transformation() -> Result<()> {
    let sample_rate = 22050;
    let samples: Vec<f32> = (0..sample_rate)
        .map(|i| (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / sample_rate as f32).sin() * 0.1)
        .collect();

    let converter = VoiceConverter::new()?;

    // Create characteristics for pitch shift
    let mut target_characteristics = VoiceCharacteristics::default();
    target_characteristics.pitch.mean_f0 = 880.0; // One octave up

    let conversion_target = ConversionTarget::new(target_characteristics);

    let request = ConversionRequest::new(
        "pitch_shift_test".to_string(),
        samples,
        sample_rate,
        ConversionType::PitchShift,
        conversion_target,
    );

    let result = converter.convert(request).await?;

    assert!(!result.converted_audio.is_empty(), "No converted audio");
    assert!(result.success, "Conversion should succeed");
    assert_eq!(
        result.conversion_type,
        ConversionType::PitchShift,
        "Wrong conversion type"
    );

    Ok(())
}

/// Test speed transformation
#[tokio::test]
async fn test_speed_transformation() -> Result<()> {
    let sample_rate = 22050;
    let samples: Vec<f32> =
        (0..(sample_rate * 2)) // 2 seconds
            .map(|i| {
                (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / sample_rate as f32).sin() * 0.1
            })
            .collect();

    let converter = VoiceConverter::new()?;

    // Create characteristics for speed change
    let mut target_characteristics = VoiceCharacteristics::default();
    target_characteristics.timing.speaking_rate = 1.5; // 1.5x speed

    let conversion_target = ConversionTarget::new(target_characteristics);

    let samples_clone = samples.clone();
    let request = ConversionRequest::new(
        "speed_transform_test".to_string(),
        samples,
        sample_rate,
        ConversionType::SpeedTransformation,
        conversion_target,
    );

    let result = converter.convert(request).await?;

    assert!(!result.converted_audio.is_empty(), "No converted audio");
    // Speed change should affect audio length
    let original_length = samples_clone.len();
    // For speed increase (tempo > 1.0), output should be shorter
    assert!(
        result.converted_audio.len() < original_length,
        "Audio should be shorter for speed increase"
    );
    assert_eq!(
        result.conversion_type,
        ConversionType::SpeedTransformation,
        "Wrong conversion type"
    );

    Ok(())
}

/// Test batch conversion with multiple requests
#[tokio::test]
async fn test_batch_conversion() -> Result<()> {
    let converter = VoiceConverter::new()?;
    let sample_rate = 22050;

    // Create test data for different conversion types
    let test_cases = vec![
        (ConversionType::PitchShift, 440.0),
        (ConversionType::AgeTransformation, 880.0),
        (ConversionType::GenderTransformation, 220.0),
    ];

    let mut results = Vec::new();

    for (i, (conversion_type, frequency)) in test_cases.into_iter().enumerate() {
        let samples: Vec<f32> = (0..sample_rate)
            .map(|j| {
                (j as f32 * frequency * 2.0 * std::f32::consts::PI / sample_rate as f32).sin() * 0.1
            })
            .collect();

        let target_characteristics = VoiceCharacteristics::default();
        let conversion_target = ConversionTarget::new(target_characteristics);

        let request = ConversionRequest::new(
            format!("batch_test_{}", i),
            samples,
            sample_rate,
            conversion_type.clone(),
            conversion_target,
        );

        let result = converter.convert(request).await?;
        results.push((conversion_type, result));
    }

    // Validate all conversions succeeded
    assert_eq!(results.len(), 3, "Not all conversions completed");

    for (conversion_type, result) in results {
        assert!(
            !result.converted_audio.is_empty(),
            "No audio for conversion type: {:?}",
            conversion_type
        );
        assert!(
            result.processing_time > Duration::ZERO,
            "No processing time for conversion type: {:?}",
            conversion_type
        );
    }

    Ok(())
}

/// Test conversion with reference samples
#[tokio::test]
async fn test_conversion_with_reference_samples() -> Result<()> {
    let sample_rate = 22050;
    let samples: Vec<f32> = (0..sample_rate)
        .map(|i| (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / sample_rate as f32).sin() * 0.1)
        .collect();

    // Create reference sample
    let reference_samples: Vec<f32> = (0..sample_rate)
        .map(|i| (i as f32 * 880.0 * 2.0 * std::f32::consts::PI / sample_rate as f32).sin() * 0.1)
        .collect();

    let reference_sample = voirs_conversion::types::AudioSample::new(
        "reference_001".to_string(),
        reference_samples,
        sample_rate,
    );

    let converter = VoiceConverter::new()?;

    let target_characteristics = VoiceCharacteristics::default();
    let conversion_target =
        ConversionTarget::new(target_characteristics).with_reference_sample(reference_sample);

    let request = ConversionRequest::new(
        "reference_test".to_string(),
        samples,
        sample_rate,
        ConversionType::SpeakerConversion,
        conversion_target,
    );

    let result = converter.convert(request).await?;

    assert!(!result.converted_audio.is_empty(), "No converted audio");
    assert!(result.success, "Conversion should succeed");
    assert_eq!(
        result.conversion_type,
        ConversionType::SpeakerConversion,
        "Wrong conversion type"
    );

    Ok(())
}

/// Test error handling
#[tokio::test]
async fn test_error_handling() -> Result<()> {
    let converter = VoiceConverter::new()?;

    // Test with empty audio
    let target_characteristics = VoiceCharacteristics::default();
    let conversion_target = ConversionTarget::new(target_characteristics);

    let request = ConversionRequest::new(
        "empty_audio_test".to_string(),
        vec![], // Empty audio
        22050,
        ConversionType::PitchShift,
        conversion_target,
    );

    let result = converter.convert(request).await;

    // Should handle error gracefully
    match result {
        Ok(_) => {
            // If it succeeds, that's fine too (implementation might handle empty audio)
        }
        Err(e) => {
            // Should be a validation or audio error
            assert!(
                matches!(
                    e,
                    voirs_conversion::Error::Validation { .. }
                        | voirs_conversion::Error::Audio { .. }
                ),
                "Unexpected error type: {:?}",
                e
            );
        }
    }

    Ok(())
}

/// Test conversion quality metrics
#[tokio::test]
async fn test_quality_metrics() -> Result<()> {
    let sample_rate = 22050;
    let samples: Vec<f32> =
        (0..(sample_rate * 2)) // 2 seconds
            .map(|i| {
                (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / sample_rate as f32).sin() * 0.1
            })
            .collect();

    // Create converter with quality monitoring enabled
    let config = ConversionConfig {
        quality_level: 0.9,
        use_gpu: false,
        buffer_size: 1024,
        output_sample_rate: sample_rate,
        default_conversion_type: ConversionType::PitchShift,
        enable_realtime: false,
        ..ConversionConfig::default()
    };

    let converter = VoiceConverter::with_config(config)?;

    let target_characteristics = VoiceCharacteristics::default();
    let conversion_target = ConversionTarget::new(target_characteristics);

    let request = ConversionRequest::new(
        "quality_test".to_string(),
        samples,
        sample_rate,
        ConversionType::PitchShift,
        conversion_target,
    )
    .with_quality_level(0.9);

    let result = converter.convert(request).await?;

    // Quality metrics should be available
    assert!(result.success, "Conversion should succeed");
    assert!(
        result.processing_time > Duration::ZERO,
        "No processing time recorded"
    );
    assert!(
        !result.quality_metrics.is_empty(),
        "No quality metrics available"
    );

    // Check that quality metrics contain expected values
    if let Some(artifacts) = &result.artifacts {
        assert!(
            artifacts.overall_score >= 0.0 && artifacts.overall_score <= 1.0,
            "Invalid artifact score: {}",
            artifacts.overall_score
        );
    }

    if let Some(quality) = &result.objective_quality {
        assert!(quality.snr_estimate >= 0.0, "Invalid SNR estimate");
        assert!(
            quality.overall_score >= 0.0 && quality.overall_score <= 1.0,
            "Invalid overall score"
        );
    }

    Ok(())
}

/// Test concurrent conversions
#[tokio::test]
async fn test_concurrent_conversions() -> Result<()> {
    let converter = std::sync::Arc::new(VoiceConverter::new()?);
    let sample_rate = 22050;

    let mut handles = Vec::new();

    // Start multiple concurrent conversions
    for i in 0..4 {
        let converter_clone = converter.clone();
        let frequency = 440.0 + (i as f32 * 110.0); // Different frequencies

        let handle = tokio::spawn(async move {
            let samples: Vec<f32> = (0..sample_rate)
                .map(|j| {
                    (j as f32 * frequency * 2.0 * std::f32::consts::PI / sample_rate as f32).sin()
                        * 0.1
                })
                .collect();

            let target_characteristics = VoiceCharacteristics::default();
            let conversion_target = ConversionTarget::new(target_characteristics);

            let request = ConversionRequest::new(
                format!("concurrent_test_{}", i),
                samples,
                sample_rate,
                ConversionType::PitchShift,
                conversion_target,
            );

            converter_clone.convert(request).await
        });

        handles.push(handle);
    }

    // Wait for all conversions to complete
    let mut results = Vec::new();
    for handle in handles {
        let result = handle.await.unwrap()?;
        results.push(result);
    }

    // Validate all conversions succeeded
    assert_eq!(results.len(), 4, "Not all concurrent conversions completed");

    for (i, result) in results.iter().enumerate() {
        assert!(
            !result.converted_audio.is_empty(),
            "No audio for concurrent conversion {}",
            i
        );
        assert!(
            result.processing_time > Duration::ZERO,
            "No processing time for concurrent conversion {}",
            i
        );
    }

    Ok(())
}

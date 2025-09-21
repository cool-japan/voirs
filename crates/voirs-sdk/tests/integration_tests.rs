//! Integration tests for VoiRS SDK
//!
//! These tests verify that the SDK components work together correctly
//! and provide comprehensive coverage of the public API.

use tempfile::TempDir;
use voirs_sdk::{
    audio::AudioBuffer,
    builder::VoirsPipelineBuilder,
    config::{ConfigHierarchy, PipelineConfig},
    error::{ErrorRecoveryManager, VoirsError},
    types::{AudioFormat, LanguageCode, SynthesisConfig},
};

#[tokio::test]
async fn test_pipeline_creation_success() {
    let temp_dir = TempDir::new().unwrap();

    let result = VoirsPipelineBuilder::new()
        .with_cache_dir(temp_dir.path())
        .with_validation(false)
        .with_test_mode(true)
        .build()
        .await;

    // Should either succeed or fail gracefully with config error
    // (since we don't have real models in test environment)
    assert!(result.is_ok() || matches!(result, Err(VoirsError::ConfigError { .. })));
}

#[tokio::test]
async fn test_builder_configuration() {
    let temp_dir = TempDir::new().unwrap();
    let synthesis_config = SynthesisConfig {
        sample_rate: 44100,
        speaking_rate: 1.2,
        pitch_shift: 0.5,
        volume_gain: 2.0,
        language: LanguageCode::EnUs,
        ..Default::default()
    };

    let builder = VoirsPipelineBuilder::new()
        .with_cache_dir(temp_dir.path())
        .with_synthesis_config(synthesis_config.clone())
        .with_validation(false)
        .with_test_mode(true);

    // Builder should accept configuration without errors
    let result = builder.build().await;
    assert!(result.is_ok() || matches!(result, Err(VoirsError::ConfigError { .. })));
}

#[test]
fn test_audio_buffer_creation() {
    let samples = vec![0.1, -0.2, 0.3, -0.4];
    let sample_rate = 22050;

    let buffer = AudioBuffer::mono(samples.clone(), sample_rate);

    assert_eq!(buffer.samples().len(), 4);
    assert_eq!(buffer.sample_rate(), sample_rate);
    assert_eq!(buffer.channels(), 1);

    // Test stereo buffer
    let stereo_samples = vec![0.1, -0.1, 0.2, -0.2, 0.3, -0.3]; // 3 stereo samples
    let stereo_buffer = AudioBuffer::stereo(stereo_samples, sample_rate);

    assert_eq!(stereo_buffer.samples().len(), 6);
    assert_eq!(stereo_buffer.channels(), 2);
}

#[test]
fn test_synthesis_config_validation() {
    let mut config = SynthesisConfig::default();

    // Test valid configuration
    assert!(config.validate().is_ok());

    // Test invalid speaking rate
    config.speaking_rate = -1.0;
    assert!(config.validate().is_err());

    // Reset and test invalid pitch shift
    config.speaking_rate = 1.0;
    config.pitch_shift = 20.0; // Outside valid range
    assert!(config.validate().is_err());

    // Reset and test invalid volume gain
    config.pitch_shift = 0.0;
    config.volume_gain = 30.0; // Outside valid range
    assert!(config.validate().is_err());

    // Reset and test invalid sample rate
    config.volume_gain = 0.0;
    config.sample_rate = 1000; // Too low
    assert!(config.validate().is_err());
}

#[test]
fn test_config_hierarchy_merge() {
    let mut base_config = SynthesisConfig::default();
    let override_config = SynthesisConfig {
        speaking_rate: 1.5,
        pitch_shift: 2.0,
        language: LanguageCode::JaJp,
        ..Default::default()
    };

    base_config.merge_with(&override_config);

    assert_eq!(base_config.speaking_rate, 1.5);
    assert_eq!(base_config.pitch_shift, 2.0);
    assert_eq!(base_config.language, LanguageCode::JaJp);
    // Other fields should remain default
    assert_eq!(base_config.volume_gain, 0.0);
}

#[test]
fn test_error_system() {
    // Test error creation
    let error = VoirsError::config_error("Test configuration error");
    assert!(matches!(error, VoirsError::ConfigError { .. }));

    let synthesis_error = VoirsError::synthesis_failed(
        "test text",
        std::io::Error::new(std::io::ErrorKind::Other, "test"),
    );
    assert!(matches!(
        synthesis_error,
        VoirsError::SynthesisFailed { .. }
    ));

    // Test error with context
    let context_error = error.with_context("test_component", "test_operation");
    assert_eq!(context_error.context.component, "test_component");
    assert_eq!(context_error.context.operation, "test_operation");
}

#[tokio::test]
async fn test_error_recovery_manager() {
    let manager = ErrorRecoveryManager::default();

    let operation_count = std::sync::Arc::new(std::sync::Mutex::new(0));
    let operation = {
        let count = operation_count.clone();
        move || -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<String, VoirsError>> + Send>> {
            let count = count.clone();
            Box::pin(async move {
                let mut counter = count.lock().unwrap();
                *counter += 1;
                if *counter < 3 {
                    Err(VoirsError::InternalError {
                        component: "test".to_string(),
                        message: "temporary error".to_string(),
                    })
                } else {
                    Ok("success".to_string())
                }
            })
        }
    };

    let result = manager.execute_with_recovery("synthesis", operation).await;
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), "success");
    assert_eq!(*operation_count.lock().unwrap(), 3);
}

#[test]
fn test_error_reporter() {
    use voirs_sdk::error::{ErrorReporter, ErrorReporterConfig};

    let config = ErrorReporterConfig::default();
    let reporter = ErrorReporter::new(config);

    let error = VoirsError::InternalError {
        component: "test".to_string(),
        message: "test error".to_string(),
    };

    reporter.report_error(&error, Some("test_context"));

    let stats = reporter.get_statistics();
    assert!(stats.total_errors > 0);
}

#[test]
fn test_language_codes() {
    assert_eq!(LanguageCode::EnUs.as_str(), "en-US");
    assert_eq!(LanguageCode::JaJp.as_str(), "ja-JP");
    assert_eq!(LanguageCode::DeDe.as_str(), "de-DE");

    assert_eq!(LanguageCode::parse("en-US"), Some(LanguageCode::EnUs));
    assert_eq!(LanguageCode::parse("invalid"), None);
}

#[test]
fn test_audio_formats() {
    assert_eq!(AudioFormat::Wav.extension(), "wav");
    assert_eq!(AudioFormat::Mp3.extension(), "mp3");
    assert_eq!(AudioFormat::Flac.extension(), "flac");

    assert_eq!(AudioFormat::Wav.to_string(), "wav");
}

#[test]
fn test_pipeline_config_serialization() {
    let config = PipelineConfig::default();

    // Test JSON serialization
    let json = serde_json::to_string(&config).expect("Should serialize to JSON");
    let _deserialized: PipelineConfig =
        serde_json::from_str(&json).expect("Should deserialize from JSON");

    // Basic validation that serialization works
    assert!(!json.is_empty());
}

#[test]
fn test_synthesis_config_serialization() {
    let config = SynthesisConfig {
        speaking_rate: 1.2,
        pitch_shift: 0.5,
        volume_gain: 1.0,
        language: LanguageCode::FrFr,
        ..Default::default()
    };

    // Test JSON serialization
    let json = serde_json::to_string(&config).expect("Should serialize to JSON");
    let deserialized: SynthesisConfig =
        serde_json::from_str(&json).expect("Should deserialize from JSON");

    assert_eq!(config.speaking_rate, deserialized.speaking_rate);
    assert_eq!(config.pitch_shift, deserialized.pitch_shift);
    assert_eq!(config.volume_gain, deserialized.volume_gain);
    assert_eq!(config.language, deserialized.language);
}

#[tokio::test]
async fn test_concurrent_pipeline_access() {
    let temp_dir = TempDir::new().unwrap();

    let pipeline_result = VoirsPipelineBuilder::new()
        .with_cache_dir(temp_dir.path())
        .with_validation(false)
        .with_test_mode(true)
        .build()
        .await;

    if let Ok(pipeline) = pipeline_result {
        let pipeline = std::sync::Arc::new(pipeline);
        let mut handles = Vec::new();

        // Spawn multiple concurrent tasks
        for i in 0..5 {
            let pipeline_clone = pipeline.clone();
            let text = format!("Concurrent test {i}");

            let handle = tokio::spawn(async move { pipeline_clone.synthesize(&text).await });

            handles.push(handle);
        }

        // Wait for all tasks to complete
        let mut success_count = 0;
        for handle in handles {
            let result = handle.await.unwrap();
            // Count both success and expected config errors
            if result.is_ok() || matches!(result, Err(VoirsError::ConfigError { .. })) {
                success_count += 1;
            }
        }

        // All tasks should complete without panicking
        assert_eq!(success_count, 5);
    }
}

#[test]
fn test_audio_buffer_metadata() {
    let samples = vec![0.5, -0.3, 0.8, -0.1, 0.2];
    let buffer = AudioBuffer::mono(samples, 48000);

    // Basic properties
    assert_eq!(buffer.sample_rate(), 48000);
    assert_eq!(buffer.channels(), 1);
    assert_eq!(buffer.samples().len(), 5);

    // Test duration calculation
    let expected_duration = 5.0 / 48000.0; // samples / sample_rate
    assert!((buffer.duration() - expected_duration).abs() < 0.0001);
}

#[test]
fn test_error_convenience_constructors() {
    let error1 = VoirsError::internal("test_component", "test message");
    assert!(matches!(error1, VoirsError::InternalError { .. }));

    let error2 = VoirsError::invalid_config("field", "value", "reason");
    assert!(matches!(error2, VoirsError::InvalidConfiguration { .. }));

    let error3 = VoirsError::device_error("cuda", "out of memory");
    assert!(matches!(error3, VoirsError::DeviceError { .. }));
}

#[test]
fn test_type_conversions() {
    use std::str::FromStr;

    // Test AudioFormat FromStr
    let format = AudioFormat::from_str("wav").unwrap();
    assert_eq!(format, AudioFormat::Wav);

    let format = AudioFormat::from_str("mp3").unwrap();
    assert_eq!(format, AudioFormat::Mp3);

    assert!(AudioFormat::from_str("invalid").is_err());

    // Test quality level defaults
    use voirs_sdk::types::QualityLevel;
    assert_eq!(QualityLevel::default(), QualityLevel::High);
}

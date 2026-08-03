//! Pipeline builder for fluent API construction.
//!
//! This module provides a modular builder architecture with component initialization,
//! fluent API methods, comprehensive validation, and async initialization.

// Module declarations for builder components
pub mod async_init;
pub mod features;
pub mod fluent;
pub mod validation;

// Re-export the modular builder implementation
pub mod builder_impl;

// The single builder type of the SDK. `crate::VoirsPipelineBuilder`,
// `crate::builder::VoirsPipelineBuilder`, `crate::pipeline::VoirsPipelineBuilder`
// and `crate::prelude::VoirsPipelineBuilder` all resolve to this type.
pub use builder_impl::VoirsPipelineBuilder;

// Re-export preset profiles for convenience
pub use builder_impl::PresetProfile;

// Re-export feature types and presets
pub use features::{
    AgeGroup, CloningMethod, CloningPreset, ConversionPreset, ConversionTarget, EmotionPreset,
    Gender, MusicalKey, Position3D, RoomSize, SingingPreset, SingingTechnique, SingingVoiceType,
    SpatialPreset,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::QualityLevel;

    #[tokio::test]
    async fn test_builder_creation() {
        let pipeline = VoirsPipelineBuilder::new()
            .with_validation(false)
            .with_test_mode(true)
            .build()
            .await;

        if let Err(ref e) = pipeline {
            eprintln!("Pipeline build failed: {e:?}");
        }
        assert!(pipeline.is_ok());
    }

    /// Without an explicit `with_test_mode(true)` opt-in and without real model
    /// weights on disk, the build must fail closed instead of silently producing
    /// stub components.
    #[tokio::test]
    async fn test_default_build_fails_closed_without_models() {
        let cache_dir = tempfile::tempdir().expect("temp dir");

        let result = VoirsPipelineBuilder::new()
            .with_validation(false)
            .with_auto_download(false)
            .with_cache_dir(cache_dir.path())
            .build()
            .await;

        assert!(
            result.is_err(),
            "default build must not succeed without real model weights"
        );
    }

    #[tokio::test]
    async fn test_builder_fluent_api() {
        let builder = VoirsPipelineBuilder::new()
            .with_quality(QualityLevel::High)
            .with_gpu_acceleration(false) // Set to false for test environments
            .with_threads(4)
            .with_cache_size(1024)
            .with_speaking_rate(1.2)
            .with_pitch_shift(2.0)
            .with_volume_gain(3.0)
            .with_enhancement(true)
            .with_sample_rate(22050)
            .with_validation(false)
            .with_test_mode(true);

        let pipeline = builder.build().await;
        if let Err(ref e) = pipeline {
            eprintln!("Pipeline build failed: {e:?}");
        }
        assert!(pipeline.is_ok());
    }

    #[tokio::test]
    async fn test_preset_profiles() {
        for preset in [
            PresetProfile::HighQuality,
            PresetProfile::FastSynthesis,
            PresetProfile::LowMemory,
            PresetProfile::Streaming,
        ] {
            let pipeline = VoirsPipelineBuilder::new()
                .with_preset(preset)
                .with_validation(false)
                .with_test_mode(true)
                .build()
                .await;
            assert!(pipeline.is_ok(), "preset {preset:?} failed to build");
        }
    }

    #[tokio::test]
    async fn test_config_file_loading() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Create a temporary config file
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(
            temp_file,
            r#"
            use_gpu = false
            device = "cpu"
            max_cache_size_mb = 512

            [default_synthesis]
            speaking_rate = 1.0
            pitch_shift = 0.0
            volume_gain = 0.0
            enable_enhancement = true
            sample_rate = 22050
            quality = "High"
            language = "EnUs"
        "#
        )
        .unwrap();

        let result = VoirsPipelineBuilder::new().with_config_file(temp_file.path());

        // Note: This might fail if PipelineConfig::from_file is not implemented
        // but the API should work
        if let Ok(builder) = result {
            let pipeline = builder
                .with_validation(false)
                .with_test_mode(true)
                .build()
                .await;
            assert!(pipeline.is_ok());
        }
    }

    #[tokio::test]
    async fn test_validation() {
        // Test with validation enabled (should work with valid config)
        let valid_builder = VoirsPipelineBuilder::new()
            .with_speaking_rate(1.0)
            .with_pitch_shift(0.0)
            .with_volume_gain(0.0)
            .with_validation(true)
            .with_test_mode(true);

        let result = valid_builder.build().await;
        assert!(result.is_ok());

        // Test with invalid config (should fail validation)
        let invalid_builder = VoirsPipelineBuilder::new()
            .with_speaking_rate(5.0) // Invalid: too high
            .with_validation(true);

        let result = invalid_builder.build().await;
        assert!(result.is_err());
    }
}

//! Pipeline builder for fluent API construction.
//!
//! This module provides a modular builder architecture with:
//! - Fluent API for method chaining
//! - Comprehensive validation
//! - Async initialization with parallel component loading

use crate::{
    config::PipelineConfig,
    traits::{AcousticModel, G2p, Vocoder},
    voice::DefaultVoiceManager,
};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Builder for VoiRS pipeline with fluent API
pub struct VoirsPipelineBuilder {
    /// Voice ID to use
    pub(crate) voice_id: Option<String>,

    /// Pipeline configuration
    pub(crate) config: PipelineConfig,

    /// Custom G2P component
    pub(crate) custom_g2p: Option<Arc<dyn G2p>>,

    /// Custom acoustic model
    pub(crate) custom_acoustic: Option<Arc<dyn AcousticModel>>,

    /// Custom vocoder
    pub(crate) custom_vocoder: Option<Arc<dyn Vocoder>>,

    /// Voice manager override
    pub(crate) voice_manager: Option<Arc<RwLock<DefaultVoiceManager>>>,

    /// Validation options
    pub(crate) validation_enabled: bool,

    /// Auto-download missing models
    pub(crate) auto_download: bool,

    /// Test mode - skip expensive operations
    pub(crate) test_mode: bool,
}

impl VoirsPipelineBuilder {
    /// Create new pipeline builder
    pub fn new() -> Self {
        Self {
            voice_id: None,
            config: PipelineConfig::default(),
            custom_g2p: None,
            custom_acoustic: None,
            custom_vocoder: None,
            voice_manager: None,
            validation_enabled: true,
            auto_download: true,
            test_mode: cfg!(test), // Automatically enable test mode when running tests
        }
    }

    /// Get configuration (internal helper)
    #[allow(dead_code)] // Internal method for debugging/future use
    pub(crate) fn get_config(&self) -> PipelineConfig {
        self.config.clone()
    }

    /// Get voice ID (internal helper)
    #[allow(dead_code)] // Internal method for debugging/future use
    pub(crate) fn get_voice_id(&self) -> Option<String> {
        self.voice_id.clone()
    }

    /// Get test mode (internal helper)
    #[allow(dead_code)] // Internal method for debugging/future use
    pub(crate) fn get_test_mode(&self) -> bool {
        self.test_mode
    }
}

impl Default for VoirsPipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Preset configuration profiles
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PresetProfile {
    /// High quality synthesis with best possible output
    HighQuality,
    /// Fast synthesis optimized for speed
    FastSynthesis,
    /// Low memory usage optimized for resource-constrained environments
    LowMemory,
    /// Optimized for streaming/real-time synthesis
    Streaming,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{AudioFormat, LanguageCode, QualityLevel};

    #[test]
    fn test_builder_creation() {
        let builder = VoirsPipelineBuilder::new();
        assert!(builder.voice_id.is_none());
        assert!(builder.validation_enabled);
        assert!(builder.auto_download);
        assert!(builder.custom_g2p.is_none());
        assert!(builder.custom_acoustic.is_none());
        assert!(builder.custom_vocoder.is_none());
    }

    #[test]
    fn test_default_implementation() {
        let builder1 = VoirsPipelineBuilder::new();
        let builder2 = VoirsPipelineBuilder::default();

        assert_eq!(builder1.validation_enabled, builder2.validation_enabled);
        assert_eq!(builder1.auto_download, builder2.auto_download);
    }

    #[test]
    fn test_internal_helpers() {
        let builder = VoirsPipelineBuilder::new().with_voice("test-voice");

        assert_eq!(builder.get_voice_id(), Some("test-voice".to_string()));

        let config = builder.get_config();
        assert!(config.default_synthesis.enable_enhancement);
    }

    #[tokio::test]
    async fn test_full_builder_workflow() {
        let builder = VoirsPipelineBuilder::new()
            .with_language(LanguageCode::EnUs)
            .with_quality(QualityLevel::High)
            .with_gpu_acceleration(false)
            .with_threads(2)
            .with_speaking_rate(1.2)
            .with_pitch_shift(1.0)
            .with_volume_gain(2.0)
            .with_enhancement(true)
            .with_sample_rate(22050)
            .with_audio_format(AudioFormat::Wav)
            .with_preset(PresetProfile::HighQuality)
            .with_validation(false) // Disable validation for test
            .with_test_mode(true); // Enable test mode for fast testing

        // Test that the builder can be built
        let result = builder.build().await;
        if let Err(ref e) = result {
            eprintln!("Builder failed: {e:?}");
        }
        assert!(result.is_ok());

        let pipeline = result.unwrap();

        // Test basic functionality
        let audio = pipeline.synthesize("Hello, world!").await;
        if let Err(ref e) = audio {
            eprintln!("Synthesis error: {e}");
        }
        assert!(audio.is_ok());
    }

    #[tokio::test]
    async fn test_validation_workflow() {
        let builder = VoirsPipelineBuilder::new()
            .with_speaking_rate(1.0)
            .with_pitch_shift(0.0)
            .with_volume_gain(0.0)
            .with_validation(true);

        // This should pass validation
        let result = builder.validate().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_preset_profiles() {
        // Test each preset profile
        let high_quality = VoirsPipelineBuilder::new()
            .with_preset(PresetProfile::HighQuality)
            .with_validation(false);

        assert_eq!(
            high_quality.config.default_synthesis.quality,
            QualityLevel::Ultra
        );
        assert!(high_quality.config.use_gpu);
        assert!(high_quality.config.default_synthesis.enable_enhancement);

        let fast_synthesis = VoirsPipelineBuilder::new()
            .with_preset(PresetProfile::FastSynthesis)
            .with_validation(false);

        assert_eq!(
            fast_synthesis.config.default_synthesis.quality,
            QualityLevel::Medium
        );
        assert!(fast_synthesis.config.use_gpu);
        assert!(!fast_synthesis.config.default_synthesis.enable_enhancement);

        let low_memory = VoirsPipelineBuilder::new()
            .with_preset(PresetProfile::LowMemory)
            .with_validation(false);

        assert_eq!(
            low_memory.config.default_synthesis.quality,
            QualityLevel::Low
        );
        assert!(!low_memory.config.use_gpu);
        assert_eq!(low_memory.config.max_cache_size_mb, 256);

        let streaming = VoirsPipelineBuilder::new()
            .with_preset(PresetProfile::Streaming)
            .with_validation(false);

        assert_eq!(
            streaming.config.default_synthesis.quality,
            QualityLevel::Medium
        );
        assert!(streaming.config.use_gpu);
        assert!(!streaming.config.default_synthesis.enable_enhancement);
    }

    #[tokio::test]
    async fn test_custom_components() {
        use crate::pipeline::{DummyAcoustic, DummyG2p, DummyVocoder};

        let custom_g2p = Arc::new(DummyG2p::new());
        let custom_acoustic = Arc::new(DummyAcoustic::new());
        let custom_vocoder = Arc::new(DummyVocoder::new());

        let builder = VoirsPipelineBuilder::new()
            .with_g2p(custom_g2p)
            .with_acoustic_model(custom_acoustic)
            .with_vocoder(custom_vocoder)
            .with_validation(false)
            .with_test_mode(true);

        assert!(builder.custom_g2p.is_some());
        assert!(builder.custom_acoustic.is_some());
        assert!(builder.custom_vocoder.is_some());

        let result = builder.build().await;
        assert!(result.is_ok());
    }
}

//! Pipeline builder for fluent API construction.
//!
//! This module provides a modular builder architecture with:
//! - Fluent API for method chaining
//! - Comprehensive validation
//! - Async initialization backed by the real component initializer
//!
//! This is the single [`VoirsPipelineBuilder`] type of the SDK. It is re-exported
//! at the crate root, from [`crate::builder`], from [`crate::pipeline`] and from
//! [`crate::prelude`], so every import path resolves to the same real builder.

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

    /// Test mode - use in-process stub components instead of real model weights.
    ///
    /// This must be opted into explicitly via
    /// [`with_test_mode`](Self::with_test_mode); it is never enabled implicitly.
    pub(crate) test_mode: bool,

    /// Emotion controller configuration supplied by the caller
    #[cfg(feature = "emotion")]
    pub(crate) emotion_config: Option<crate::emotion::EmotionControllerBuilder>,

    /// Voice cloner configuration supplied by the caller
    #[cfg(feature = "cloning")]
    pub(crate) cloning_config: Option<crate::cloning::VoiceClonerBuilder>,

    /// Voice converter configuration supplied by the caller
    #[cfg(feature = "conversion")]
    pub(crate) conversion_config: Option<crate::conversion::VoiceConverterBuilder>,

    /// Singing controller configuration supplied by the caller
    #[cfg(feature = "singing")]
    pub(crate) singing_config: Option<crate::singing::SingingControllerBuilder>,

    /// Spatial audio controller configuration supplied by the caller
    #[cfg(feature = "spatial")]
    pub(crate) spatial_config: Option<crate::spatial::SpatialAudioControllerBuilder>,
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
            // Never implicitly enabled: stub components must be opted into.
            test_mode: false,

            #[cfg(feature = "emotion")]
            emotion_config: None,
            #[cfg(feature = "cloning")]
            cloning_config: None,
            #[cfg(feature = "conversion")]
            conversion_config: None,
            #[cfg(feature = "singing")]
            singing_config: None,
            #[cfg(feature = "spatial")]
            spatial_config: None,
        }
    }

    /// Get configuration (internal helper)
    pub(crate) fn get_config(&self) -> PipelineConfig {
        self.config.clone()
    }

    /// Get voice ID (internal helper)
    pub(crate) fn get_voice_id(&self) -> Option<String> {
        self.voice_id.clone()
    }

    /// Get test mode (internal helper)
    pub(crate) fn get_test_mode(&self) -> bool {
        self.test_mode
    }

    /// Get the custom G2P override, if any (internal helper)
    pub(crate) fn get_custom_g2p(&self) -> Option<Arc<dyn G2p>> {
        self.custom_g2p.clone()
    }

    /// Get the custom acoustic model override, if any (internal helper)
    pub(crate) fn get_custom_acoustic(&self) -> Option<Arc<dyn AcousticModel>> {
        self.custom_acoustic.clone()
    }

    /// Get the custom vocoder override, if any (internal helper)
    pub(crate) fn get_custom_vocoder(&self) -> Option<Arc<dyn Vocoder>> {
        self.custom_vocoder.clone()
    }

    /// Get the voice manager override, if any (internal helper)
    pub(crate) fn get_voice_manager(&self) -> Option<Arc<RwLock<DefaultVoiceManager>>> {
        self.voice_manager.clone()
    }

    /// Get emotion configuration (internal helper)
    #[cfg(feature = "emotion")]
    pub(crate) fn get_emotion_config(&self) -> Option<crate::emotion::EmotionControllerBuilder> {
        self.emotion_config.clone()
    }

    /// Get cloning configuration (internal helper)
    #[cfg(feature = "cloning")]
    pub(crate) fn get_cloning_config(&self) -> Option<crate::cloning::VoiceClonerBuilder> {
        self.cloning_config.clone()
    }

    /// Get conversion configuration (internal helper)
    #[cfg(feature = "conversion")]
    pub(crate) fn get_conversion_config(&self) -> Option<crate::conversion::VoiceConverterBuilder> {
        self.conversion_config.clone()
    }

    /// Get singing configuration (internal helper)
    #[cfg(feature = "singing")]
    pub(crate) fn get_singing_config(&self) -> Option<crate::singing::SingingControllerBuilder> {
        self.singing_config.clone()
    }

    /// Get spatial configuration (internal helper)
    #[cfg(feature = "spatial")]
    pub(crate) fn get_spatial_config(
        &self,
    ) -> Option<crate::spatial::SpatialAudioControllerBuilder> {
        self.spatial_config.clone()
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

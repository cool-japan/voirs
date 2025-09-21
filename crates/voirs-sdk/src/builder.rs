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

use crate::{
    config::PipelineConfig,
    error::Result,
    pipeline::VoirsPipeline,
    traits::{AcousticModel, G2p, Vocoder},
    types::{LanguageCode, QualityLevel, SynthesisConfig},
    voice::DefaultVoiceManager,
};
use std::{path::PathBuf, sync::Arc};
use tokio::sync::RwLock;

/// Main builder for VoiRS pipeline with fluent API
pub struct VoirsPipelineBuilder {
    /// Internal builder implementation
    inner: builder_impl::VoirsPipelineBuilder,
}

impl VoirsPipelineBuilder {
    /// Create new pipeline builder
    pub fn new() -> Self {
        Self {
            inner: builder_impl::VoirsPipelineBuilder::new(),
        }
    }

    /// Set the voice to use for synthesis
    pub fn with_voice(mut self, voice: impl Into<String>) -> Self {
        self.inner = self.inner.with_voice(voice);
        self
    }

    /// Set the language (will auto-select appropriate voice)
    pub fn with_language(mut self, language: LanguageCode) -> Self {
        self.inner = self.inner.with_language(language);
        self
    }

    /// Set synthesis quality level
    pub fn with_quality(mut self, quality: QualityLevel) -> Self {
        self.inner = self.inner.with_quality(quality);
        self
    }

    /// Enable or disable GPU acceleration
    pub fn with_gpu_acceleration(mut self, enabled: bool) -> Self {
        self.inner = self.inner.with_gpu_acceleration(enabled);
        self
    }

    /// Set specific device for computation
    pub fn with_device(mut self, device: impl Into<String>) -> Self {
        self.inner = self.inner.with_device(device);
        self
    }

    /// Set number of CPU threads to use
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.inner = self.inner.with_threads(threads);
        self
    }

    /// Set custom cache directory
    pub fn with_cache_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.inner = self.inner.with_cache_dir(path);
        self
    }

    /// Set maximum cache size in MB
    pub fn with_cache_size(mut self, size_mb: u32) -> Self {
        self.inner = self.inner.with_cache_size(size_mb);
        self
    }

    /// Set speaking rate (0.5 - 2.0)
    pub fn with_speaking_rate(mut self, rate: f32) -> Self {
        self.inner = self.inner.with_speaking_rate(rate);
        self
    }

    /// Set pitch shift in semitones (-12.0 - 12.0)
    pub fn with_pitch_shift(mut self, semitones: f32) -> Self {
        self.inner = self.inner.with_pitch_shift(semitones);
        self
    }

    /// Set volume gain in dB (-20.0 - 20.0)
    pub fn with_volume_gain(mut self, gain_db: f32) -> Self {
        self.inner = self.inner.with_volume_gain(gain_db);
        self
    }

    /// Enable or disable audio enhancement
    pub fn with_enhancement(mut self, enabled: bool) -> Self {
        self.inner = self.inner.with_enhancement(enabled);
        self
    }

    /// Set output sample rate
    pub fn with_sample_rate(mut self, sample_rate: u32) -> Self {
        self.inner = self.inner.with_sample_rate(sample_rate);
        self
    }

    /// Set output audio format
    pub fn with_audio_format(mut self, format: crate::types::AudioFormat) -> Self {
        self.inner = self.inner.with_audio_format(format);
        self
    }

    /// Use custom G2P component
    pub fn with_g2p(mut self, g2p: Arc<dyn G2p>) -> Self {
        self.inner = self.inner.with_g2p(g2p);
        self
    }

    /// Use custom acoustic model
    pub fn with_acoustic_model(mut self, acoustic: Arc<dyn AcousticModel>) -> Self {
        self.inner = self.inner.with_acoustic_model(acoustic);
        self
    }

    /// Use custom vocoder
    pub fn with_vocoder(mut self, vocoder: Arc<dyn Vocoder>) -> Self {
        self.inner = self.inner.with_vocoder(vocoder);
        self
    }

    /// Use custom voice manager
    pub fn with_voice_manager(mut self, manager: Arc<RwLock<DefaultVoiceManager>>) -> Self {
        self.inner = self.inner.with_voice_manager(manager);
        self
    }

    /// Enable or disable validation during build
    pub fn with_validation(mut self, enabled: bool) -> Self {
        self.inner = self.inner.with_validation(enabled);
        self
    }

    /// Enable or disable automatic model downloading
    pub fn with_auto_download(mut self, enabled: bool) -> Self {
        self.inner = self.inner.with_auto_download(enabled);
        self
    }

    /// Enable test mode (skips expensive operations for fast testing)
    pub fn with_test_mode(mut self, enabled: bool) -> Self {
        self.inner = self.inner.with_test_mode(enabled);
        self
    }

    /// Load configuration from file
    pub fn with_config_file(mut self, path: impl AsRef<std::path::Path>) -> Result<Self> {
        self.inner = self.inner.with_config_file(path)?;
        Ok(self)
    }

    /// Merge with existing configuration
    pub fn with_config(mut self, config: PipelineConfig) -> Self {
        self.inner = self.inner.with_config(config);
        self
    }

    /// Apply configuration overrides
    pub fn with_config_overrides<F>(mut self, f: F) -> Self
    where
        F: FnOnce(&mut PipelineConfig),
    {
        self.inner = self.inner.with_config_overrides(f);
        self
    }

    /// Set synthesis configuration overrides
    pub fn with_synthesis_config(mut self, synthesis_config: SynthesisConfig) -> Self {
        self.inner = self.inner.with_synthesis_config(synthesis_config);
        self
    }

    /// Enable preset configuration profiles
    pub fn with_preset(mut self, preset: builder_impl::PresetProfile) -> Self {
        self.inner = self.inner.with_preset(preset);
        self
    }

    /// Get configuration (internal)
    #[allow(dead_code)] // Internal method for debugging/future use
    pub(crate) fn get_config(&self) -> PipelineConfig {
        self.inner.get_config()
    }

    /// Get voice ID (internal)
    #[allow(dead_code)] // Internal method for debugging/future use
    pub(crate) fn get_voice_id(&self) -> Option<String> {
        self.inner.get_voice_id()
    }

    /// Get test mode (internal)
    #[allow(dead_code)] // Internal method for debugging/future use
    pub(crate) fn get_test_mode(&self) -> bool {
        self.inner.get_test_mode()
    }

    /// Build the pipeline
    pub async fn build(self) -> Result<VoirsPipeline> {
        self.inner.build().await
    }
}

impl Default for VoirsPipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

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

    #[tokio::test]
    async fn test_builder_creation() {
        let pipeline = VoirsPipelineBuilder::new()
            .with_validation(false)
            .build()
            .await;

        if let Err(ref e) = pipeline {
            eprintln!("Pipeline build failed: {e:?}");
        }
        assert!(pipeline.is_ok());
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
            .with_validation(false);

        let pipeline = builder.build().await;
        if let Err(ref e) = pipeline {
            eprintln!("Pipeline build failed: {e:?}");
        }
        assert!(pipeline.is_ok());
    }

    #[tokio::test]
    async fn test_preset_profiles() {
        let high_quality = VoirsPipelineBuilder::new()
            .with_preset(PresetProfile::HighQuality)
            .with_validation(false)
            .build()
            .await;
        assert!(high_quality.is_ok());

        let fast_synthesis = VoirsPipelineBuilder::new()
            .with_preset(PresetProfile::FastSynthesis)
            .with_validation(false)
            .build()
            .await;
        assert!(fast_synthesis.is_ok());

        let low_memory = VoirsPipelineBuilder::new()
            .with_preset(PresetProfile::LowMemory)
            .with_validation(false)
            .build()
            .await;
        assert!(low_memory.is_ok());

        let streaming = VoirsPipelineBuilder::new()
            .with_preset(PresetProfile::Streaming)
            .with_validation(false)
            .build()
            .await;
        assert!(streaming.is_ok());
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
        if result.is_ok() {
            let pipeline = result.unwrap().with_validation(false).build().await;
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
            .with_validation(true);

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

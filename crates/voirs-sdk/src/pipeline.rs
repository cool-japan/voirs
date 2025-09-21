//! VoiRS synthesis pipeline implementation.
//!
//! This module provides a modular pipeline architecture with component initialization,
//! synthesis orchestration, and state management.

use crate::{
    audio::AudioBuffer,
    config::PipelineConfig,
    error::Result,
    traits::{AcousticModel, G2p, Vocoder},
    types::{LanguageCode, SynthesisConfig, VoiceConfig},
    VoirsError,
};
use std::sync::Arc;

// Required imports for dummy implementations
use fastrand;

// Module declarations for pipeline components
pub mod init;
pub mod state;
pub mod synthesis;

// Re-export the modular pipeline implementation
pub mod pipeline_impl;
pub use pipeline_impl::*;

/// Main VoiRS synthesis pipeline
pub struct VoirsPipeline {
    /// Internal pipeline implementation
    inner: pipeline_impl::VoirsPipeline,

    /// Advanced features
    #[cfg(feature = "emotion")]
    emotion_controller: Option<Arc<crate::emotion::EmotionController>>,
    #[cfg(feature = "cloning")]
    voice_cloner: Option<Arc<crate::cloning::VoiceCloner>>,
    #[cfg(feature = "conversion")]
    voice_converter: Option<Arc<crate::conversion::VoiceConverter>>,
    #[cfg(feature = "singing")]
    singing_controller: Option<Arc<crate::singing::SingingController>>,
    #[cfg(feature = "spatial")]
    spatial_controller: Option<Arc<crate::spatial::SpatialAudioController>>,
}

impl VoirsPipeline {
    /// Create a new pipeline builder
    pub fn builder() -> VoirsPipelineBuilder {
        VoirsPipelineBuilder::new()
    }

    /// Create pipeline with components
    pub fn new(
        g2p: Arc<dyn G2p>,
        acoustic: Arc<dyn AcousticModel>,
        vocoder: Arc<dyn Vocoder>,
        config: PipelineConfig,
    ) -> Self {
        Self {
            inner: pipeline_impl::VoirsPipeline::new(g2p, acoustic, vocoder, config),

            #[cfg(feature = "emotion")]
            emotion_controller: None,
            #[cfg(feature = "cloning")]
            voice_cloner: None,
            #[cfg(feature = "conversion")]
            voice_converter: None,
            #[cfg(feature = "singing")]
            singing_controller: None,
            #[cfg(feature = "spatial")]
            spatial_controller: None,
        }
    }

    /// Create pipeline with components and test mode
    pub fn with_test_mode(
        g2p: Arc<dyn G2p>,
        acoustic: Arc<dyn AcousticModel>,
        vocoder: Arc<dyn Vocoder>,
        config: PipelineConfig,
        test_mode: bool,
    ) -> Self {
        Self {
            inner: pipeline_impl::VoirsPipeline::with_test_mode(
                g2p, acoustic, vocoder, config, test_mode,
            ),

            #[cfg(feature = "emotion")]
            emotion_controller: None,
            #[cfg(feature = "cloning")]
            voice_cloner: None,
            #[cfg(feature = "conversion")]
            voice_converter: None,
            #[cfg(feature = "singing")]
            singing_controller: None,
            #[cfg(feature = "spatial")]
            spatial_controller: None,
        }
    }

    /// Synthesize text to audio
    pub async fn synthesize(&self, text: &str) -> Result<AudioBuffer> {
        self.inner.synthesize(text).await
    }

    /// Synthesize with custom configuration
    pub async fn synthesize_with_config(
        &self,
        text: &str,
        config: &SynthesisConfig,
    ) -> Result<AudioBuffer> {
        self.inner.synthesize_with_config(text, config).await
    }

    /// Synthesize SSML markup
    pub async fn synthesize_ssml(&self, ssml: &str) -> Result<AudioBuffer> {
        self.inner.synthesize_ssml(ssml).await
    }

    /// Stream synthesis for long texts
    pub async fn synthesize_stream(
        self: Arc<Self>,
        text: &str,
    ) -> Result<impl futures::Stream<Item = Result<AudioBuffer>>> {
        let inner = Arc::new(self.inner.clone());
        inner.synthesize_stream(text).await
    }

    /// Change voice during runtime
    pub async fn set_voice(&self, voice_id: &str) -> Result<()> {
        self.inner.set_voice(voice_id).await
    }

    /// Get current voice information
    pub async fn current_voice(&self) -> Option<VoiceConfig> {
        self.inner.current_voice().await
    }

    /// List available voices
    pub async fn list_voices(&self) -> Result<Vec<VoiceConfig>> {
        self.inner.list_voices().await
    }

    /// Get pipeline state
    pub async fn get_state(&self) -> pipeline_impl::PublicPipelineState {
        self.inner.get_state().await
    }

    /// Get pipeline configuration
    pub async fn get_config(&self) -> PipelineConfig {
        self.inner.get_config().await
    }

    /// Update pipeline configuration
    pub async fn update_config(&self, new_config: PipelineConfig) -> Result<()> {
        self.inner.update_config(new_config).await
    }

    /// Get component states
    pub async fn get_component_states(&self) -> pipeline_impl::PublicComponentStates {
        self.inner.get_component_states().await
    }

    /// Synchronize all components
    pub async fn synchronize_components(&self) -> Result<()> {
        self.inner.synchronize_components().await
    }

    /// Cleanup pipeline resources
    pub async fn cleanup(&self) -> Result<()> {
        self.inner.cleanup().await
    }

    /// Set pipeline state to ready
    pub async fn set_ready(&self) -> Result<()> {
        self.inner.set_ready().await
    }

    // Advanced features API methods

    /// Get emotion controller (if emotion feature is enabled)
    #[cfg(feature = "emotion")]
    pub fn emotion_controller(&self) -> Option<&Arc<crate::emotion::EmotionController>> {
        self.emotion_controller.as_ref()
    }

    /// Set emotion for synthesis
    #[cfg(feature = "emotion")]
    pub async fn set_emotion(
        &self,
        emotion: crate::emotion::Emotion,
        intensity: Option<f32>,
    ) -> Result<()> {
        if let Some(controller) = &self.emotion_controller {
            controller.set_emotion(emotion, intensity).await
        } else {
            Err(VoirsError::ConfigError {
                field: "emotion".to_string(),
                message: "Emotion controller not configured".to_string(),
            })
        }
    }

    /// Apply emotion preset
    #[cfg(feature = "emotion")]
    pub async fn apply_emotion_preset(
        &self,
        preset_name: &str,
        intensity: Option<f32>,
    ) -> Result<()> {
        if let Some(controller) = &self.emotion_controller {
            controller.apply_preset(preset_name, intensity).await
        } else {
            Err(VoirsError::ConfigError {
                field: "emotion".to_string(),
                message: "Emotion controller not configured".to_string(),
            })
        }
    }

    /// Get voice cloner (if cloning feature is enabled)
    #[cfg(feature = "cloning")]
    pub fn voice_cloner(&self) -> Option<&Arc<crate::cloning::VoiceCloner>> {
        self.voice_cloner.as_ref()
    }

    /// Clone voice from reference samples
    #[cfg(feature = "cloning")]
    pub async fn clone_voice(
        &self,
        speaker_id: String,
        reference_samples: Vec<crate::cloning::VoiceSample>,
        target_text: String,
        method: Option<crate::cloning::CloningMethod>,
    ) -> Result<crate::cloning::VoiceCloneResult> {
        if let Some(cloner) = &self.voice_cloner {
            cloner
                .clone_voice(speaker_id, reference_samples, target_text, method)
                .await
        } else {
            Err(VoirsError::ConfigError {
                field: "cloning".to_string(),
                message: "Voice cloner not configured".to_string(),
            })
        }
    }

    /// Quick clone from single audio file
    #[cfg(feature = "cloning")]
    pub async fn quick_clone(
        &self,
        audio_data: Vec<f32>,
        sample_rate: u32,
        target_text: String,
    ) -> Result<crate::cloning::VoiceCloneResult> {
        if let Some(cloner) = &self.voice_cloner {
            cloner
                .quick_clone(audio_data, sample_rate, target_text)
                .await
        } else {
            Err(VoirsError::ConfigError {
                field: "cloning".to_string(),
                message: "Voice cloner not configured".to_string(),
            })
        }
    }

    /// Get voice converter (if conversion feature is enabled)
    #[cfg(feature = "conversion")]
    pub fn voice_converter(&self) -> Option<&Arc<crate::conversion::VoiceConverter>> {
        self.voice_converter.as_ref()
    }

    /// Convert voice with specified target
    #[cfg(feature = "conversion")]
    pub async fn convert_voice(
        &self,
        source_audio: Vec<f32>,
        source_sample_rate: u32,
        target: crate::conversion::ConversionTarget,
        conversion_type: Option<crate::conversion::ConversionType>,
    ) -> Result<crate::conversion::ConversionResult> {
        if let Some(converter) = &self.voice_converter {
            converter
                .convert_voice(source_audio, source_sample_rate, target, conversion_type)
                .await
        } else {
            Err(VoirsError::ConfigError {
                field: "conversion".to_string(),
                message: "Voice converter not configured".to_string(),
            })
        }
    }

    /// Convert to different age
    #[cfg(feature = "conversion")]
    pub async fn convert_age(
        &self,
        source_audio: Vec<f32>,
        source_sample_rate: u32,
        target_age: crate::conversion::AgeGroup,
    ) -> Result<crate::conversion::ConversionResult> {
        if let Some(converter) = &self.voice_converter {
            converter
                .convert_age(source_audio, source_sample_rate, target_age)
                .await
        } else {
            Err(VoirsError::ConfigError {
                field: "conversion".to_string(),
                message: "Voice converter not configured".to_string(),
            })
        }
    }

    /// Convert to different gender
    #[cfg(feature = "conversion")]
    pub async fn convert_gender(
        &self,
        source_audio: Vec<f32>,
        source_sample_rate: u32,
        target_gender: crate::conversion::Gender,
    ) -> Result<crate::conversion::ConversionResult> {
        if let Some(converter) = &self.voice_converter {
            converter
                .convert_gender(source_audio, source_sample_rate, target_gender)
                .await
        } else {
            Err(VoirsError::ConfigError {
                field: "conversion".to_string(),
                message: "Voice converter not configured".to_string(),
            })
        }
    }

    /// Get singing controller (if singing feature is enabled)
    #[cfg(feature = "singing")]
    pub fn singing_controller(&self) -> Option<&Arc<crate::singing::SingingController>> {
        self.singing_controller.as_ref()
    }

    /// Synthesize singing from musical score
    #[cfg(feature = "singing")]
    pub async fn synthesize_singing_score(
        &self,
        score: crate::singing::MusicalScore,
        text: &str,
    ) -> Result<crate::singing::SingingResult> {
        if let Some(controller) = &self.singing_controller {
            controller.synthesize_score(score, text).await
        } else {
            Err(VoirsError::ConfigError {
                field: "singing".to_string(),
                message: "Singing controller not configured".to_string(),
            })
        }
    }

    /// Synthesize singing from text with automatic pitch detection
    #[cfg(feature = "singing")]
    pub async fn synthesize_singing_text(
        &self,
        text: &str,
        key: &str,
        tempo: f32,
    ) -> Result<crate::singing::SingingResult> {
        if let Some(controller) = &self.singing_controller {
            controller.synthesize_from_text(text, key, tempo).await
        } else {
            Err(VoirsError::ConfigError {
                field: "singing".to_string(),
                message: "Singing controller not configured".to_string(),
            })
        }
    }

    /// Set singing technique
    #[cfg(feature = "singing")]
    pub async fn set_singing_technique(
        &self,
        technique: crate::singing::SingingTechnique,
    ) -> Result<()> {
        if let Some(controller) = &self.singing_controller {
            controller.set_technique(technique).await
        } else {
            Err(VoirsError::ConfigError {
                field: "singing".to_string(),
                message: "Singing controller not configured".to_string(),
            })
        }
    }

    /// Set singing voice type
    #[cfg(feature = "singing")]
    pub async fn set_singing_voice_type(
        &self,
        voice_type: crate::singing::VoiceType,
    ) -> Result<()> {
        if let Some(controller) = &self.singing_controller {
            controller.set_voice_type(voice_type).await
        } else {
            Err(VoirsError::ConfigError {
                field: "singing".to_string(),
                message: "Singing controller not configured".to_string(),
            })
        }
    }

    /// Apply singing preset
    #[cfg(feature = "singing")]
    pub async fn apply_singing_preset(&self, preset_name: &str) -> Result<()> {
        if let Some(controller) = &self.singing_controller {
            controller.apply_preset(preset_name).await
        } else {
            Err(VoirsError::ConfigError {
                field: "singing".to_string(),
                message: "Singing controller not configured".to_string(),
            })
        }
    }

    /// Parse musical score from text
    #[cfg(feature = "singing")]
    pub async fn parse_musical_score(
        &self,
        score_text: &str,
    ) -> Result<crate::singing::MusicalScore> {
        if let Some(controller) = &self.singing_controller {
            controller.parse_score(score_text).await
        } else {
            Err(VoirsError::ConfigError {
                field: "singing".to_string(),
                message: "Singing controller not configured".to_string(),
            })
        }
    }

    /// Get spatial audio controller (if spatial feature is enabled)
    #[cfg(feature = "spatial")]
    pub fn spatial_controller(&self) -> Option<&Arc<crate::spatial::SpatialAudioController>> {
        self.spatial_controller.as_ref()
    }

    /// Process audio with spatial effects
    #[cfg(feature = "spatial")]
    pub async fn process_spatial_audio(
        &self,
        audio: &crate::audio::AudioBuffer,
    ) -> Result<crate::spatial::SpatialAudioResult> {
        if let Some(controller) = &self.spatial_controller {
            controller.process_spatial_audio(audio).await
        } else {
            Err(VoirsError::ConfigError {
                field: "spatial".to_string(),
                message: "Spatial audio controller not configured".to_string(),
            })
        }
    }

    /// Set listener position and orientation
    #[cfg(feature = "spatial")]
    pub async fn set_listener_position(
        &self,
        position: crate::spatial::Position3D,
        orientation: crate::spatial::Orientation3D,
    ) -> Result<()> {
        if let Some(controller) = &self.spatial_controller {
            controller.set_listener(position, orientation).await
        } else {
            Err(VoirsError::ConfigError {
                field: "spatial".to_string(),
                message: "Spatial audio controller not configured".to_string(),
            })
        }
    }

    /// Add 3D audio source
    #[cfg(feature = "spatial")]
    pub async fn add_spatial_source(&self, source: crate::spatial::AudioSource3D) -> Result<usize> {
        if let Some(controller) = &self.spatial_controller {
            controller.add_source(source).await
        } else {
            Err(VoirsError::ConfigError {
                field: "spatial".to_string(),
                message: "Spatial audio controller not configured".to_string(),
            })
        }
    }

    /// Update spatial source position
    #[cfg(feature = "spatial")]
    pub async fn update_spatial_source_position(
        &self,
        source_id: usize,
        position: crate::spatial::Position3D,
    ) -> Result<()> {
        if let Some(controller) = &self.spatial_controller {
            controller.update_source_position(source_id, position).await
        } else {
            Err(VoirsError::ConfigError {
                field: "spatial".to_string(),
                message: "Spatial audio controller not configured".to_string(),
            })
        }
    }

    /// Remove spatial audio source
    #[cfg(feature = "spatial")]
    pub async fn remove_spatial_source(&self, source_id: usize) -> Result<()> {
        if let Some(controller) = &self.spatial_controller {
            controller.remove_source(source_id).await
        } else {
            Err(VoirsError::ConfigError {
                field: "spatial".to_string(),
                message: "Spatial audio controller not configured".to_string(),
            })
        }
    }

    /// Set room acoustics
    #[cfg(feature = "spatial")]
    pub async fn set_room_acoustics(&self, room: crate::spatial::RoomAcoustics) -> Result<()> {
        if let Some(controller) = &self.spatial_controller {
            controller.set_room_acoustics(room).await
        } else {
            Err(VoirsError::ConfigError {
                field: "spatial".to_string(),
                message: "Spatial audio controller not configured".to_string(),
            })
        }
    }

    /// Apply spatial audio preset
    #[cfg(feature = "spatial")]
    pub async fn apply_spatial_preset(&self, preset_name: &str) -> Result<()> {
        if let Some(controller) = &self.spatial_controller {
            controller.apply_preset(preset_name).await
        } else {
            Err(VoirsError::ConfigError {
                field: "spatial".to_string(),
                message: "Spatial audio controller not configured".to_string(),
            })
        }
    }
}

/// Builder for VoiRS pipeline
pub struct VoirsPipelineBuilder {
    voice_id: Option<String>,
    config: PipelineConfig,
    test_mode: bool,

    // Advanced features configuration
    #[cfg(feature = "emotion")]
    emotion_config: Option<crate::emotion::EmotionControllerBuilder>,
    #[cfg(feature = "cloning")]
    cloning_config: Option<crate::cloning::VoiceClonerBuilder>,
    #[cfg(feature = "conversion")]
    conversion_config: Option<crate::conversion::VoiceConverterBuilder>,
    #[cfg(feature = "singing")]
    singing_config: Option<crate::singing::SingingControllerBuilder>,
    #[cfg(feature = "spatial")]
    spatial_config: Option<crate::spatial::SpatialAudioControllerBuilder>,
}

impl VoirsPipelineBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self {
            voice_id: None,
            config: PipelineConfig::default(),
            test_mode: cfg!(test), // Automatically enable test mode when running tests

            // Advanced features configuration
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

    /// Set the voice to use
    pub fn with_voice(mut self, voice: impl Into<String>) -> Self {
        self.voice_id = Some(voice.into());
        self
    }

    /// Set synthesis quality
    pub fn with_quality(mut self, quality: crate::types::QualityLevel) -> Self {
        self.config.default_synthesis.quality = quality;
        self
    }

    /// Enable GPU acceleration
    pub fn with_gpu_acceleration(mut self, enabled: bool) -> Self {
        self.config.use_gpu = enabled;
        self
    }

    /// Set device for computation
    pub fn with_device(mut self, device: String) -> Self {
        self.config.device = device;
        self
    }

    /// Enable GPU acceleration (alias for with_gpu_acceleration)
    pub fn with_gpu(mut self, enabled: bool) -> Self {
        self.config.use_gpu = enabled;
        self
    }

    /// Set number of threads for computation
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.config.num_threads = Some(threads);
        self
    }

    /// Set custom cache directory
    pub fn with_cache_dir(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.config.cache_dir = Some(path.into());
        self
    }

    /// Enable test mode (skips expensive operations for fast testing)
    pub fn with_test_mode(mut self, enabled: bool) -> Self {
        self.test_mode = enabled;
        self
    }

    // Advanced features configuration methods

    /// Configure emotion control for the pipeline
    #[cfg(feature = "emotion")]
    pub fn with_emotion_control(
        mut self,
        emotion_builder: crate::emotion::EmotionControllerBuilder,
    ) -> Self {
        self.emotion_config = Some(emotion_builder);
        self
    }

    /// Enable emotion control with default configuration
    #[cfg(feature = "emotion")]
    pub fn with_emotion_enabled(mut self, enabled: bool) -> Self {
        if enabled {
            self.emotion_config =
                Some(crate::emotion::EmotionControllerBuilder::new().enabled(true));
        } else {
            self.emotion_config = None;
        }
        self
    }

    /// Configure voice cloning for the pipeline  
    #[cfg(feature = "cloning")]
    pub fn with_voice_cloning(
        mut self,
        cloning_builder: crate::cloning::VoiceClonerBuilder,
    ) -> Self {
        self.cloning_config = Some(cloning_builder);
        self
    }

    /// Enable voice cloning with default configuration
    #[cfg(feature = "cloning")]
    pub fn with_cloning_enabled(mut self, enabled: bool) -> Self {
        if enabled {
            self.cloning_config = Some(crate::cloning::VoiceClonerBuilder::new().enabled(true));
        } else {
            self.cloning_config = None;
        }
        self
    }

    /// Configure voice conversion for the pipeline
    #[cfg(feature = "conversion")]
    pub fn with_voice_conversion(
        mut self,
        conversion_builder: crate::conversion::VoiceConverterBuilder,
    ) -> Self {
        self.conversion_config = Some(conversion_builder);
        self
    }

    /// Enable voice conversion with default configuration
    #[cfg(feature = "conversion")]
    pub fn with_conversion_enabled(mut self, enabled: bool) -> Self {
        if enabled {
            self.conversion_config =
                Some(crate::conversion::VoiceConverterBuilder::new().enabled(true));
        } else {
            self.conversion_config = None;
        }
        self
    }

    /// Configure singing synthesis for the pipeline
    #[cfg(feature = "singing")]
    pub fn with_singing_synthesis(
        mut self,
        singing_builder: crate::singing::SingingControllerBuilder,
    ) -> Self {
        self.singing_config = Some(singing_builder);
        self
    }

    /// Enable singing synthesis with default configuration
    #[cfg(feature = "singing")]
    pub fn with_singing_enabled(mut self, enabled: bool) -> Self {
        if enabled {
            self.singing_config =
                Some(crate::singing::SingingControllerBuilder::new().enabled(true));
        } else {
            self.singing_config = None;
        }
        self
    }

    /// Configure spatial audio for the pipeline
    #[cfg(feature = "spatial")]
    pub fn with_spatial_audio(
        mut self,
        spatial_builder: crate::spatial::SpatialAudioControllerBuilder,
    ) -> Self {
        self.spatial_config = Some(spatial_builder);
        self
    }

    /// Enable spatial audio with default configuration
    #[cfg(feature = "spatial")]
    pub fn with_spatial_enabled(mut self, enabled: bool) -> Self {
        if enabled {
            self.spatial_config =
                Some(crate::spatial::SpatialAudioControllerBuilder::new().enabled(true));
        } else {
            self.spatial_config = None;
        }
        self
    }

    /// Get configuration (internal)
    pub(crate) fn get_config(&self) -> PipelineConfig {
        self.config.clone()
    }

    /// Get voice ID (internal)
    pub(crate) fn get_voice_id(&self) -> Option<String> {
        self.voice_id.clone()
    }

    /// Get test mode (internal)
    pub(crate) fn get_test_mode(&self) -> bool {
        self.test_mode
    }

    /// Get emotion configuration (internal)
    #[cfg(feature = "emotion")]
    pub(crate) fn get_emotion_config(&self) -> Option<crate::emotion::EmotionControllerBuilder> {
        self.emotion_config.clone()
    }

    /// Get cloning configuration (internal)
    #[cfg(feature = "cloning")]
    pub(crate) fn get_cloning_config(&self) -> Option<crate::cloning::VoiceClonerBuilder> {
        self.cloning_config.clone()
    }

    /// Get conversion configuration (internal)
    #[cfg(feature = "conversion")]
    pub(crate) fn get_conversion_config(&self) -> Option<crate::conversion::VoiceConverterBuilder> {
        self.conversion_config.clone()
    }

    /// Get singing configuration (internal)
    #[cfg(feature = "singing")]
    pub(crate) fn get_singing_config(&self) -> Option<crate::singing::SingingControllerBuilder> {
        self.singing_config.clone()
    }

    /// Get spatial configuration (internal)
    #[cfg(feature = "spatial")]
    pub(crate) fn get_spatial_config(
        &self,
    ) -> Option<crate::spatial::SpatialAudioControllerBuilder> {
        self.spatial_config.clone()
    }

    /// Build the pipeline
    pub async fn build(self) -> Result<VoirsPipeline> {
        tracing::info!("Building VoiRS pipeline");

        // Use the new modular pipeline implementation
        let inner = pipeline_impl::VoirsPipeline::from_builder_core(&self).await?;

        // Initialize advanced features
        #[cfg(feature = "emotion")]
        let emotion_controller = if let Some(emotion_builder) = self.emotion_config {
            Some(Arc::new(emotion_builder.build().await.map_err(|e| {
                VoirsError::model_error(format!("Failed to initialize emotion controller: {}", e))
            })?))
        } else {
            None
        };

        #[cfg(feature = "cloning")]
        let voice_cloner = if let Some(cloning_builder) = self.cloning_config {
            Some(Arc::new(cloning_builder.build().await.map_err(|e| {
                VoirsError::model_error(format!("Failed to initialize voice cloner: {}", e))
            })?))
        } else {
            None
        };

        #[cfg(feature = "conversion")]
        let voice_converter = if let Some(conversion_builder) = self.conversion_config {
            Some(Arc::new(conversion_builder.build().await.map_err(|e| {
                VoirsError::model_error(format!("Failed to initialize voice converter: {}", e))
            })?))
        } else {
            None
        };

        #[cfg(feature = "singing")]
        let singing_controller = if let Some(singing_builder) = self.singing_config {
            Some(Arc::new(singing_builder.build().await.map_err(|e| {
                VoirsError::model_error(format!("Failed to initialize singing controller: {}", e))
            })?))
        } else {
            None
        };

        #[cfg(feature = "spatial")]
        let spatial_controller = if let Some(spatial_builder) = self.spatial_config {
            Some(Arc::new(spatial_builder.build().await.map_err(|e| {
                VoirsError::model_error(format!(
                    "Failed to initialize spatial audio controller: {}",
                    e
                ))
            })?))
        } else {
            None
        };

        tracing::info!("VoiRS pipeline built successfully");
        Ok(VoirsPipeline {
            inner,

            #[cfg(feature = "emotion")]
            emotion_controller,
            #[cfg(feature = "cloning")]
            voice_cloner,
            #[cfg(feature = "conversion")]
            voice_converter,
            #[cfg(feature = "singing")]
            singing_controller,
            #[cfg(feature = "spatial")]
            spatial_controller,
        })
    }
}

impl Default for VoirsPipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// Dummy implementations for testing

/// Dummy G2P implementation for testing
pub struct DummyG2p;

impl Default for DummyG2p {
    fn default() -> Self {
        Self::new()
    }
}

impl DummyG2p {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl G2p for DummyG2p {
    async fn to_phonemes(
        &self,
        text: &str,
        _lang: Option<LanguageCode>,
    ) -> Result<Vec<crate::types::Phoneme>> {
        // Convert each character to a mock phoneme
        let phonemes: Vec<crate::types::Phoneme> = text
            .chars()
            .filter(|c| c.is_alphabetic())
            .map(|c| crate::types::Phoneme::new(c.to_string()))
            .collect();

        tracing::debug!(
            "DummyG2p: Generated {} phonemes for '{}'",
            phonemes.len(),
            text
        );
        Ok(phonemes)
    }

    fn supported_languages(&self) -> Vec<LanguageCode> {
        vec![LanguageCode::EnUs]
    }

    fn metadata(&self) -> crate::traits::G2pMetadata {
        crate::traits::G2pMetadata {
            name: "DummyG2p".to_string(),
            version: "0.1.0".to_string(),
            description: "Dummy G2P for testing".to_string(),
            supported_languages: vec![LanguageCode::EnUs],
            accuracy_scores: std::collections::HashMap::new(),
        }
    }
}

/// Dummy acoustic model for testing
pub struct DummyAcoustic;

impl Default for DummyAcoustic {
    fn default() -> Self {
        Self::new()
    }
}

impl DummyAcoustic {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl AcousticModel for DummyAcoustic {
    async fn synthesize(
        &self,
        phonemes: &[crate::types::Phoneme],
        _config: Option<&SynthesisConfig>,
    ) -> Result<crate::types::MelSpectrogram> {
        let n_mels = 80;
        let n_frames = phonemes.len() * 10; // 10 frames per phoneme

        // Generate random mel spectrogram
        let mut data = Vec::with_capacity(n_mels);
        for _ in 0..n_mels {
            let mut row = Vec::with_capacity(n_frames);
            for _ in 0..n_frames {
                row.push(fastrand::f32() * 2.0 - 1.0); // Random values
            }
            data.push(row);
        }

        let mel = crate::types::MelSpectrogram::new(data, 22050, 256);
        tracing::debug!(
            "DummyAcoustic: Generated {}x{} mel spectrogram",
            mel.n_mels,
            mel.n_frames
        );
        Ok(mel)
    }

    async fn synthesize_batch(
        &self,
        inputs: &[&[crate::types::Phoneme]],
        configs: Option<&[SynthesisConfig]>,
    ) -> Result<Vec<crate::types::MelSpectrogram>> {
        let mut results = Vec::new();
        for (i, phonemes) in inputs.iter().enumerate() {
            let config = configs.and_then(|c| c.get(i));
            results.push(self.synthesize(phonemes, config).await?);
        }
        Ok(results)
    }

    fn metadata(&self) -> crate::traits::AcousticModelMetadata {
        crate::traits::AcousticModelMetadata {
            name: "DummyAcoustic".to_string(),
            version: "0.1.0".to_string(),
            architecture: "Dummy".to_string(),
            supported_languages: vec![LanguageCode::EnUs],
            sample_rate: 22050,
            mel_channels: 80,
            is_multi_speaker: false,
            speaker_count: None,
        }
    }

    fn supports(&self, _feature: crate::traits::AcousticModelFeature) -> bool {
        false
    }
}

/// Dummy vocoder for testing
pub struct DummyVocoder;

impl Default for DummyVocoder {
    fn default() -> Self {
        Self::new()
    }
}

impl DummyVocoder {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl Vocoder for DummyVocoder {
    async fn vocode(
        &self,
        mel: &crate::types::MelSpectrogram,
        _config: Option<&SynthesisConfig>,
    ) -> Result<AudioBuffer> {
        // Generate sine wave based on mel spectrogram duration
        let duration = mel.duration();
        let frequency = 440.0; // A4 note
        let audio = AudioBuffer::sine_wave(frequency, duration, mel.sample_rate, 0.5);

        tracing::debug!("DummyVocoder: Generated {:.2}s audio from mel", duration);
        Ok(audio)
    }

    async fn vocode_stream(
        &self,
        _mel_stream: Box<dyn futures::Stream<Item = crate::types::MelSpectrogram> + Send + Unpin>,
        _config: Option<&SynthesisConfig>,
    ) -> Result<Box<dyn futures::Stream<Item = Result<AudioBuffer>> + Send + Unpin>> {
        // For dummy implementation, return an empty stream
        use futures::stream;
        let empty_stream = stream::empty();
        Ok(Box::new(empty_stream))
    }

    async fn vocode_batch(
        &self,
        mels: &[crate::types::MelSpectrogram],
        configs: Option<&[SynthesisConfig]>,
    ) -> Result<Vec<AudioBuffer>> {
        let mut results = Vec::new();
        for (i, mel) in mels.iter().enumerate() {
            let config = configs.and_then(|c| c.get(i));
            results.push(self.vocode(mel, config).await?);
        }
        Ok(results)
    }

    fn metadata(&self) -> crate::traits::VocoderMetadata {
        crate::traits::VocoderMetadata {
            name: "DummyVocoder".to_string(),
            version: "0.1.0".to_string(),
            architecture: "Sine Wave".to_string(),
            sample_rate: 22050,
            mel_channels: 80,
            latency_ms: 10.0,
            quality_score: 2.0, // Low quality sine wave
        }
    }

    fn supports(&self, _feature: crate::traits::VocoderFeature) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile;

    // Helper function to create a test pipeline with mock model files
    async fn create_test_pipeline() -> Result<VoirsPipeline> {
        // Create temporary directory for testing
        let temp_dir = tempfile::tempdir().unwrap();
        let cache_dir = temp_dir.path().to_path_buf();

        // Create mock model files
        let model_filename = "EnUs-acoustic-High.safetensors";
        let model_path = cache_dir.join(model_filename);
        fs::write(&model_path, "dummy model data").unwrap();

        // Create pipeline with custom cache directory and test mode enabled
        let pipeline = VoirsPipeline::builder()
            .with_cache_dir(cache_dir)
            .with_test_mode(true) // Enable test mode to use dummy implementations
            .build()
            .await?;

        // Keep temp_dir alive by storing it in a static or similar
        // For tests, we can leak it since tests are short-lived
        std::mem::forget(temp_dir);

        Ok(pipeline)
    }

    #[tokio::test]
    async fn test_pipeline_creation() {
        let pipeline = create_test_pipeline().await;
        if let Err(e) = &pipeline {
            eprintln!("Pipeline creation failed: {e:?}");
        }
        assert!(pipeline.is_ok());
    }

    #[tokio::test]
    async fn test_basic_synthesis() {
        let pipeline = create_test_pipeline().await.unwrap();

        let audio = pipeline.synthesize("Hello, world!").await.unwrap();

        assert!(audio.duration() > 0.0);
        assert_eq!(audio.sample_rate(), 22050);
        assert!(!audio.is_empty());
    }

    #[tokio::test]
    async fn test_voice_management() {
        let pipeline = create_test_pipeline().await.unwrap();

        // Test voice setting
        pipeline.set_voice("test-voice").await.unwrap();
        let current = pipeline.current_voice().await;
        assert!(current.is_some());
        assert_eq!(current.unwrap().id, "test-voice");

        // Test voice listing
        let voices = pipeline.list_voices().await.unwrap();
        assert!(!voices.is_empty());
    }
}

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
use tokio::sync::RwLock;

// Required imports for dummy implementations
use async_trait::async_trait;
use fastrand;

// Module declarations for pipeline components
pub mod init;
pub mod synthesis;
pub mod state;

// Re-export the modular pipeline implementation
pub mod pipeline_impl;
pub use pipeline_impl::*;

/// Main VoiRS synthesis pipeline
pub struct VoirsPipeline {
    /// Internal pipeline implementation
    inner: pipeline_impl::VoirsPipeline,
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
}

/// Builder for VoiRS pipeline
pub struct VoirsPipelineBuilder {
    voice_id: Option<String>,
    config: PipelineConfig,
}

impl VoirsPipelineBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self {
            voice_id: None,
            config: PipelineConfig::default(),
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

    /// Get configuration (internal)
    pub(crate) fn get_config(&self) -> PipelineConfig {
        self.config.clone()
    }

    /// Get voice ID (internal)
    pub(crate) fn get_voice_id(&self) -> Option<String> {
        self.voice_id.clone()
    }

    /// Build the pipeline
    pub async fn build(self) -> Result<VoirsPipeline> {
        tracing::info!("Building VoiRS pipeline");

        // Use the new modular pipeline implementation
        let inner = pipeline_impl::VoirsPipeline::from_builder(self).await?;
        
        tracing::info!("VoiRS pipeline built successfully");
        Ok(VoirsPipeline { inner })
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

impl DummyG2p {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl G2p for DummyG2p {
    async fn to_phonemes(&self, text: &str, _lang: Option<LanguageCode>) -> Result<Vec<crate::types::Phoneme>> {
        // Convert each character to a mock phoneme
        let phonemes: Vec<crate::types::Phoneme> = text
            .chars()
            .filter(|c| c.is_alphabetic())
            .map(|c| crate::types::Phoneme::new(c.to_string()))
            .collect();
        
        tracing::debug!("DummyG2p: Generated {} phonemes for '{}'", phonemes.len(), text);
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
        tracing::debug!("DummyAcoustic: Generated {}x{} mel spectrogram", mel.n_mels, mel.n_frames);
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

    #[tokio::test]
    async fn test_pipeline_creation() {
        let pipeline = VoirsPipeline::builder()
            .with_voice("test-voice")
            .build()
            .await;
        
        assert!(pipeline.is_ok());
    }

    #[tokio::test]
    async fn test_basic_synthesis() {
        let pipeline = VoirsPipeline::builder()
            .build()
            .await
            .unwrap();
        
        let audio = pipeline.synthesize("Hello, world!").await.unwrap();
        
        assert!(audio.duration() > 0.0);
        assert_eq!(audio.sample_rate(), 22050);
        assert!(!audio.is_empty());
    }

    #[tokio::test]
    async fn test_voice_management() {
        let pipeline = VoirsPipeline::builder()
            .build()
            .await
            .unwrap();
        
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
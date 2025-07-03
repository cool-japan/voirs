//! # VoiRS Vocoders
//! 
//! Neural vocoders for converting mel spectrograms to high-quality audio.
//! Supports HiFi-GAN, WaveGlow, and other state-of-the-art vocoders.

use async_trait::async_trait;
use futures::Stream;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Result type for vocoder operations
pub type Result<T> = std::result::Result<T, VocoderError>;

/// Vocoder-specific error types
#[derive(Error, Debug)]
pub enum VocoderError {
    #[error("Vocoding failed: {0}")]
    VocodingError(String),
    
    #[error("Model loading failed: {0}")]
    ModelError(String),
    
    #[error("Invalid input: {0}")]
    InputError(String),
    
    #[error("Configuration error: {0}")]
    ConfigError(String),
    
    #[cfg(feature = "candle")]
    #[error("Candle error: {0}")]
    CandleError(#[from] candle_core::Error),
}

/// Language codes supported by VoiRS
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum LanguageCode {
    /// English (US)
    EnUs,
    /// English (UK)
    EnGb,
    /// Japanese
    Ja,
    /// Mandarin Chinese
    ZhCn,
    /// Korean
    Ko,
    /// German
    De,
    /// French
    Fr,
    /// Spanish
    Es,
}

impl LanguageCode {
    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            LanguageCode::EnUs => "en-US",
            LanguageCode::EnGb => "en-GB", 
            LanguageCode::Ja => "ja",
            LanguageCode::ZhCn => "zh-CN",
            LanguageCode::Ko => "ko",
            LanguageCode::De => "de",
            LanguageCode::Fr => "fr",
            LanguageCode::Es => "es",
        }
    }
}

/// Mel spectrogram representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MelSpectrogram {
    /// Mel filterbank data [n_mels, n_frames]
    pub data: Vec<Vec<f32>>,
    /// Number of mel channels
    pub n_mels: usize,
    /// Number of time frames
    pub n_frames: usize,
    /// Sample rate of original audio
    pub sample_rate: u32,
    /// Hop length in samples
    pub hop_length: u32,
}

impl MelSpectrogram {
    /// Create new mel spectrogram
    pub fn new(data: Vec<Vec<f32>>, sample_rate: u32, hop_length: u32) -> Self {
        let n_mels = data.len();
        let n_frames = data.first().map_or(0, |row| row.len());
        
        Self {
            data,
            n_mels,
            n_frames,
            sample_rate,
            hop_length,
        }
    }
    
    /// Get duration in seconds
    pub fn duration(&self) -> f32 {
        (self.n_frames as u32 * self.hop_length) as f32 / self.sample_rate as f32
    }
}

/// Synthesis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisConfig {
    /// Speaking rate multiplier (1.0 = normal)
    pub speed: f32,
    /// Pitch shift in semitones
    pub pitch_shift: f32,
    /// Energy/volume multiplier
    pub energy: f32,
    /// Speaker ID for multi-speaker models
    pub speaker_id: Option<u32>,
    /// Random seed for reproducible generation
    pub seed: Option<u64>,
}

impl Default for SynthesisConfig {
    fn default() -> Self {
        Self {
            speed: 1.0,
            pitch_shift: 0.0,
            energy: 1.0,
            speaker_id: None,
            seed: None,
        }
    }
}

/// Audio buffer for holding PCM audio data
#[derive(Debug, Clone)]
pub struct AudioBuffer {
    /// Audio samples (interleaved for multi-channel)
    samples: Vec<f32>,
    /// Sample rate in Hz
    sample_rate: u32,
    /// Number of channels
    channels: u32,
}

impl AudioBuffer {
    /// Create new audio buffer
    pub fn new(samples: Vec<f32>, sample_rate: u32, channels: u32) -> Self {
        Self {
            samples,
            sample_rate,
            channels,
        }
    }
    
    /// Create silence
    pub fn silence(duration: f32, sample_rate: u32, channels: u32) -> Self {
        let num_samples = (duration * sample_rate as f32 * channels as f32) as usize;
        Self::new(vec![0.0; num_samples], sample_rate, channels)
    }
    
    /// Create sine wave
    pub fn sine_wave(frequency: f32, duration: f32, sample_rate: u32, amplitude: f32) -> Self {
        let num_samples = (duration * sample_rate as f32) as usize;
        let mut samples = Vec::with_capacity(num_samples);
        
        for i in 0..num_samples {
            let t = i as f32 / sample_rate as f32;
            let sample = amplitude * (2.0 * std::f32::consts::PI * frequency * t).sin();
            samples.push(sample);
        }
        
        Self::new(samples, sample_rate, 1)
    }
    
    /// Get duration in seconds
    pub fn duration(&self) -> f32 {
        self.samples.len() as f32 / (self.sample_rate * self.channels) as f32
    }
    
    /// Get sample rate
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
    
    /// Get number of channels
    pub fn channels(&self) -> u32 {
        self.channels
    }
    
    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
    
    /// Get samples
    pub fn samples(&self) -> &[f32] {
        &self.samples
    }
}

/// Features supported by vocoders
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VocoderFeature {
    /// Real-time streaming inference
    StreamingInference,
    /// Batch processing
    BatchProcessing,
    /// GPU acceleration
    GpuAcceleration,
    /// High quality synthesis
    HighQuality,
    /// Real-time processing
    RealtimeProcessing,
    /// Fast inference with reduced quality
    FastInference,
}

/// Vocoder metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocoderMetadata {
    /// Model name
    pub name: String,
    /// Model version
    pub version: String,
    /// Model architecture
    pub architecture: String,
    /// Sample rate
    pub sample_rate: u32,
    /// Number of mel channels expected
    pub mel_channels: u32,
    /// Latency in milliseconds
    pub latency_ms: f32,
    /// Quality score (1-5)
    pub quality_score: f32,
}

/// Trait for neural vocoders
#[async_trait]
pub trait Vocoder: Send + Sync {
    /// Convert mel spectrogram to audio
    async fn vocode(
        &self,
        mel: &MelSpectrogram,
        config: Option<&SynthesisConfig>,
    ) -> Result<AudioBuffer>;
    
    /// Streaming vocoding
    async fn vocode_stream(
        &self,
        mel_stream: Box<dyn Stream<Item = MelSpectrogram> + Send + Unpin>,
        config: Option<&SynthesisConfig>,
    ) -> Result<Box<dyn Stream<Item = Result<AudioBuffer>> + Send + Unpin>>;
    
    /// Batch vocoding
    async fn vocode_batch(
        &self,
        mels: &[MelSpectrogram],
        configs: Option<&[SynthesisConfig]>,
    ) -> Result<Vec<AudioBuffer>>;
    
    /// Get vocoder metadata
    fn metadata(&self) -> VocoderMetadata;
    
    /// Check if vocoder supports a feature
    fn supports(&self, feature: VocoderFeature) -> bool;
}

pub mod models;
pub mod hifigan;
pub mod waveglow;
pub mod utils;
pub mod audio;
pub mod config;
pub mod backends;
pub mod effects;

// Re-export commonly used types
pub use hifigan::{HiFiGanVocoder};
pub use models::hifigan::{HiFiGanVariant, HiFiGanVariants, HiFiGanConfig};

// Types are already public in the root module

/// Vocoder manager with multiple architecture support
pub struct VocoderManager {
    vocoders: HashMap<String, Box<dyn Vocoder>>,
    default_vocoder: Option<String>,
}

impl VocoderManager {
    /// Create new vocoder manager
    pub fn new() -> Self {
        Self {
            vocoders: HashMap::new(),
            default_vocoder: None,
        }
    }
    
    /// Add vocoder
    pub fn add_vocoder(&mut self, name: String, vocoder: Box<dyn Vocoder>) {
        self.vocoders.insert(name.clone(), vocoder);
        
        // Set as default if it's the first vocoder
        if self.default_vocoder.is_none() {
            self.default_vocoder = Some(name);
        }
    }
    
    /// Set default vocoder
    pub fn set_default_vocoder(&mut self, name: String) {
        if self.vocoders.contains_key(&name) {
            self.default_vocoder = Some(name);
        }
    }
    
    /// Get vocoder by name
    pub fn get_vocoder(&self, name: &str) -> Result<&dyn Vocoder> {
        self.vocoders
            .get(name)
            .map(|v| v.as_ref())
            .ok_or_else(|| VocoderError::ModelError(format!("Vocoder '{}' not found", name)))
    }
    
    /// Get default vocoder
    pub fn get_default_vocoder(&self) -> Result<&dyn Vocoder> {
        let name = self.default_vocoder
            .as_ref()
            .ok_or_else(|| VocoderError::ConfigError("No default vocoder set".to_string()))?;
        self.get_vocoder(name)
    }
    
    /// List available vocoders
    pub fn list_vocoders(&self) -> Vec<&str> {
        self.vocoders.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for VocoderManager {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Vocoder for VocoderManager {
    async fn vocode(
        &self,
        mel: &MelSpectrogram,
        config: Option<&SynthesisConfig>,
    ) -> Result<AudioBuffer> {
        let vocoder = self.get_default_vocoder()?;
        vocoder.vocode(mel, config).await
    }
    
    async fn vocode_stream(
        &self,
        mel_stream: Box<dyn Stream<Item = MelSpectrogram> + Send + Unpin>,
        config: Option<&SynthesisConfig>,
    ) -> Result<Box<dyn Stream<Item = Result<AudioBuffer>> + Send + Unpin>> {
        let vocoder = self.get_default_vocoder()?;
        vocoder.vocode_stream(mel_stream, config).await
    }
    
    async fn vocode_batch(
        &self,
        mels: &[MelSpectrogram],
        configs: Option<&[SynthesisConfig]>,
    ) -> Result<Vec<AudioBuffer>> {
        let vocoder = self.get_default_vocoder()?;
        vocoder.vocode_batch(mels, configs).await
    }
    
    fn metadata(&self) -> VocoderMetadata {
        if let Ok(vocoder) = self.get_default_vocoder() {
            vocoder.metadata()
        } else {
            VocoderMetadata {
                name: "Vocoder Manager".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
                architecture: "Manager".to_string(),
                sample_rate: 22050,
                mel_channels: 80,
                latency_ms: 0.0,
                quality_score: 0.0,
            }
        }
    }
    
    fn supports(&self, feature: VocoderFeature) -> bool {
        if let Ok(vocoder) = self.get_default_vocoder() {
            vocoder.supports(feature)
        } else {
            false
        }
    }
}

// Type conversions are handled at the SDK level to avoid circular dependencies

/// Dummy vocoder for testing
pub struct DummyVocoder {
    sample_rate: u32,
    mel_channels: u32,
}

impl DummyVocoder {
    /// Create new dummy vocoder
    pub fn new() -> Self {
        Self {
            sample_rate: 22050,
            mel_channels: 80,
        }
    }
    
    /// Create with custom parameters
    pub fn with_config(sample_rate: u32, mel_channels: u32) -> Self {
        Self {
            sample_rate,
            mel_channels,
        }
    }
}

impl Default for DummyVocoder {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Vocoder for DummyVocoder {
    async fn vocode(
        &self,
        mel: &MelSpectrogram,
        _config: Option<&SynthesisConfig>,
    ) -> Result<AudioBuffer> {
        // Generate sine wave based on mel spectrogram duration
        let duration = mel.duration();
        let frequency = 440.0; // A4 note
        let audio = AudioBuffer::sine_wave(frequency, duration, self.sample_rate, 0.5);
        
        tracing::debug!(
            "DummyVocoder: Generated {:.2}s audio from {}x{} mel",
            duration,
            mel.n_mels,
            mel.n_frames
        );
        Ok(audio)
    }
    
    async fn vocode_stream(
        &self,
        mel_stream: Box<dyn Stream<Item = MelSpectrogram> + Send + Unpin>,
        config: Option<&SynthesisConfig>,
    ) -> Result<Box<dyn Stream<Item = Result<AudioBuffer>> + Send + Unpin>> {
        use futures::stream;
        use futures::StreamExt;
        
        // Collect all mels first, then process them
        let mels: Vec<MelSpectrogram> = mel_stream.collect().await;
        let configs = config.map(|c| vec![c.clone(); mels.len()]);
        
        let results = self.vocode_batch(&mels, configs.as_deref()).await?;
        let stream = stream::iter(results.into_iter().map(Ok));
        
        Ok(Box::new(stream))
    }
    
    async fn vocode_batch(
        &self,
        mels: &[MelSpectrogram],
        configs: Option<&[SynthesisConfig]>,
    ) -> Result<Vec<AudioBuffer>> {
        let mut results = Vec::new();
        for (i, mel) in mels.iter().enumerate() {
            let config = configs.and_then(|c| c.get(i));
            results.push(self.vocode(mel, config).await?);
        }
        Ok(results)
    }
    
    fn metadata(&self) -> VocoderMetadata {
        VocoderMetadata {
            name: "Dummy Vocoder".to_string(),
            version: "0.1.0".to_string(),
            architecture: "Sine Wave".to_string(),
            sample_rate: self.sample_rate,
            mel_channels: self.mel_channels,
            latency_ms: 10.0,
            quality_score: 2.0, // Low quality sine wave
        }
    }
    
    fn supports(&self, feature: VocoderFeature) -> bool {
        matches!(feature, VocoderFeature::BatchProcessing)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_vocoder_manager() {
        let mut manager = VocoderManager::new();
        
        // Add dummy vocoder
        manager.add_vocoder("dummy".to_string(), Box::new(DummyVocoder::new()));
        
        // Test vocoding
        let mel_data = vec![vec![0.5; 100]; 80]; // 80 mel channels, 100 frames
        let mel = MelSpectrogram::new(mel_data, 22050, 256);
        
        let audio = manager.vocode(&mel, None).await.unwrap();
        assert!(audio.duration() > 0.0);
        assert_eq!(audio.sample_rate(), 22050);
        
        // Test vocoder listing
        let vocoders = manager.list_vocoders();
        assert!(vocoders.contains(&"dummy"));
    }

    #[tokio::test]
    async fn test_dummy_vocoder() {
        let vocoder = DummyVocoder::new();
        
        // Create test mel spectrogram
        let mel_data = vec![vec![0.5; 50]; 80]; // 80 mel channels, 50 frames
        let mel = MelSpectrogram::new(mel_data, 22050, 256);
        
        let audio = vocoder.vocode(&mel, None).await.unwrap();
        
        // Check audio properties
        assert!(audio.duration() > 0.0);
        assert_eq!(audio.sample_rate(), 22050);
        assert!(!audio.is_empty());
        
        // Test metadata
        let metadata = vocoder.metadata();
        assert_eq!(metadata.name, "Dummy Vocoder");
        assert_eq!(metadata.sample_rate, 22050);
        assert_eq!(metadata.mel_channels, 80);
        
        // Test batch vocoding
        let mels = vec![mel.clone(), mel.clone()];
        let results = vocoder.vocode_batch(&mels, None).await.unwrap();
        assert_eq!(results.len(), 2);
        
        for audio in results {
            assert!(audio.duration() > 0.0);
            assert_eq!(audio.sample_rate(), 22050);
        }
        
        // Test feature support
        assert!(vocoder.supports(VocoderFeature::BatchProcessing));
        assert!(!vocoder.supports(VocoderFeature::StreamingInference));
    }
}
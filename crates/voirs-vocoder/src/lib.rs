//! # VoiRS Vocoders
//!
//! Neural vocoders for converting mel spectrograms to high-quality audio.
//! Supports HiFi-GAN, WaveGlow, and other state-of-the-art vocoders.
//!
//! This crate provides a unified interface for neural vocoding with support for:
//! - Multiple neural architectures (HiFi-GAN, DiffWave, WaveGlow)
//! - Real-time streaming synthesis
//! - Batch processing for efficiency
//! - Quality control and performance monitoring
//! - Singing voice processing with harmonic enhancement
//! - Spatial audio processing with HRTF
//!
//! # Quick Start
//!
//! ```rust,ignore
//! // This is a conceptual example - actual usage will depend on specific implementations
//! use voirs_vocoder::{Vocoder, MelSpectrogram, SynthesisConfig};
//!
//! async fn example_usage() {
//!     // Create a mel spectrogram (this would typically come from a TTS model)
//!     let mel_data = vec![vec![0.0; 80]; 100]; // 80 mel bands, 100 time steps
//!     let mel = MelSpectrogram::new(mel_data, 22050, 256);
//!     
//!     // Configure synthesis parameters
//!     let config = SynthesisConfig::default();
//!     
//!     // Convert to audio using any Vocoder implementation
//!     // let audio_buffer = vocoder.vocode(&mel, Some(&config)).await?;
//! }
//! ```

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

    #[error("Invalid mel spectrogram: {0}")]
    InvalidMelSpectrogram(String),

    #[error("Configuration error: {0}")]
    ConfigurationError(String),

    #[error("Config error: {0}")]
    ConfigError(String),

    #[error("Streaming error: {0}")]
    StreamingError(String),

    #[error("Runtime error: {0}")]
    RuntimeError(String),

    #[error("Processing error: {0}")]
    ProcessingError(String),

    #[error("Other error: {0}")]
    Other(String),

    #[cfg(feature = "candle")]
    #[error("Candle error: {0}")]
    CandleError(#[from] candle_core::Error),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
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

    /// Create from samples with default mono, 48kHz
    pub fn from_samples(samples: Vec<f32>, sample_rate: f32) -> Self {
        Self::new(samples, sample_rate as u32, 1)
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

    /// Get mutable samples
    pub fn samples_mut(&mut self) -> &mut [f32] {
        &mut self.samples
    }

    /// Get number of samples
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Get peak amplitude
    pub fn peak_amplitude(&self) -> f32 {
        self.samples.iter().map(|&x| x.abs()).fold(0.0f32, f32::max)
    }

    /// Normalize to peak amplitude
    pub fn normalize_to_peak(&mut self, target_peak: f32) {
        let current_peak = self.peak_amplitude();
        if current_peak > 0.0 {
            let gain = target_peak / current_peak;
            for sample in &mut self.samples {
                *sample *= gain;
            }
        }
    }
}

/// Features supported by vocoders
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VocoderFeature {
    /// Base vocoding functionality
    Base,
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
    /// Emotion-aware vocoding
    EmotionConditioning,
    /// Emotion variant alias
    Emotion,
    /// Real-time voice conversion
    VoiceConversion,
    /// Age transformation
    AgeTransformation,
    /// Gender transformation
    GenderTransformation,
    /// Voice morphing capabilities
    VoiceMorphing,
    /// Singing voice synthesis
    SingingVoice,
    /// Singing voice feature alias
    Singing,
    /// 3D spatial audio processing
    SpatialAudio,
    /// Spatial audio feature alias
    Spatial,
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
///
/// This trait provides the core interface for converting mel spectrograms into audio
/// using neural vocoding techniques like HiFi-GAN, WaveGlow, or other modern architectures.
///
/// # Example
///
/// ```rust,ignore
/// // Conceptual example of trait usage
/// use voirs_vocoder::{Vocoder, MelSpectrogram, SynthesisConfig};
///
/// async fn convert_to_audio<V: Vocoder>(vocoder: &V, mel: MelSpectrogram) {
///     let config = SynthesisConfig::default();
///     // let audio_buffer = vocoder.vocode(&mel, Some(&config)).await?;
/// }
/// ```
#[async_trait]
pub trait Vocoder: Send + Sync {
    /// Convert mel spectrogram to audio
    ///
    /// This method takes a mel spectrogram and converts it to high-quality audio
    /// using the vocoder's neural network architecture.
    ///
    /// # Arguments
    ///
    /// * `mel` - The input mel spectrogram to convert
    /// * `config` - Optional synthesis configuration for quality/performance tuning
    ///
    /// # Returns
    ///
    /// Returns an `AudioBuffer` containing the generated audio samples
    async fn vocode(
        &self,
        mel: &MelSpectrogram,
        config: Option<&SynthesisConfig>,
    ) -> Result<AudioBuffer>;

    /// Streaming vocoding for real-time applications
    ///
    /// This method processes a stream of mel spectrograms and produces a stream
    /// of audio buffers, enabling real-time audio synthesis with low latency.
    ///
    /// # Arguments
    ///
    /// * `mel_stream` - Stream of mel spectrograms to process
    /// * `config` - Optional synthesis configuration
    ///
    /// # Returns
    ///
    /// Returns a stream of `AudioBuffer` results for continuous playback
    async fn vocode_stream(
        &self,
        mel_stream: Box<dyn Stream<Item = MelSpectrogram> + Send + Unpin>,
        config: Option<&SynthesisConfig>,
    ) -> Result<Box<dyn Stream<Item = Result<AudioBuffer>> + Send + Unpin>>;

    /// Batch vocoding for efficient processing of multiple inputs
    ///
    /// Processes multiple mel spectrograms in a single batch operation,
    /// which can be more efficient than individual calls for large workloads.
    ///
    /// # Arguments
    ///
    /// * `mels` - Array of mel spectrograms to process
    /// * `configs` - Optional array of synthesis configurations (one per mel)
    ///
    /// # Returns
    ///
    /// Returns a vector of `AudioBuffer` results in the same order as inputs
    async fn vocode_batch(
        &self,
        mels: &[MelSpectrogram],
        configs: Option<&[SynthesisConfig]>,
    ) -> Result<Vec<AudioBuffer>>;

    /// Get vocoder metadata and capabilities
    ///
    /// Returns information about the vocoder including supported sample rates,
    /// model architecture, quality settings, and performance characteristics.
    fn metadata(&self) -> VocoderMetadata;

    /// Check if vocoder supports a specific feature
    ///
    /// Use this method to query whether the vocoder implementation supports
    /// specific features like streaming, batch processing, or quality modes.
    ///
    /// # Arguments
    ///
    /// * `feature` - The feature to check for support
    ///
    /// # Returns
    ///
    /// Returns `true` if the feature is supported, `false` otherwise
    fn supports(&self, feature: VocoderFeature) -> bool;
}

pub mod adaptive_quality;
pub mod audio;
pub mod backends;
pub mod broadcast_quality;
pub mod cache;
pub mod codecs;
pub mod comprehensive_quality_metrics;
pub mod conditioning;
pub mod config;
pub mod containers;
pub mod conversion;
pub mod drivers;
pub mod effects;
pub mod hifigan;
pub mod metrics;
pub mod ml;
pub mod models;
pub mod optimization_paths;
pub mod parallel;
pub mod performance;
pub mod post_processing;
pub mod simd;
pub mod streaming;
pub mod utils;
pub mod waveglow;

/// Prelude for convenient imports
pub mod prelude {
    pub use crate::utils::{
        apply_adaptive_noise_gate, apply_formant_enhancement, apply_intelligent_agc,
        apply_psychoacoustic_masking, apply_stereo_widening, calculate_audio_quality_metrics,
        calculate_spectral_statistics, crossfade_audio, AudioQualityMetrics, CrossfadeType,
        SpectralStatistics,
    };
    pub use crate::{
        adaptive_quality::{
            AdaptationStats, AdaptiveConfig, AdaptiveQualityController, PrecisionMode,
            QualityAdjustment, QualityTarget,
        },
        conditioning::{
            ConditioningConfigBuilder, EnhancementConfig, ProsodyConfig, SpeakerConfig,
            VocoderConditioner, VocoderConditioningConfig, VoiceCharacteristics,
        },
        conversion::{VoiceConversionConfig, VoiceConverter, VoiceMorpher},
        hifigan::{EmotionConfig, EmotionVocodingParams},
        performance::{
            PerformanceAlert, PerformanceMetrics, PerformanceMonitor, PerformanceStatistics,
            PerformanceThresholds,
        },
        AudioBuffer, DummyVocoder, HiFiGanVocoder, LanguageCode, MelSpectrogram, Result,
        SynthesisConfig, Vocoder, VocoderError, VocoderFeature, VocoderManager, VocoderMetadata,
    };
    pub use async_trait::async_trait;
}

// Re-export commonly used types
pub use hifigan::HiFiGanVocoder;
pub use models::hifigan::{HiFiGanConfig, HiFiGanVariant, HiFiGanVariants};
pub use streaming::{StreamHandle, StreamingPipeline, StreamingStats, StreamingVocoder};

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
            .ok_or_else(|| VocoderError::ModelError(format!("Vocoder '{name}' not found")))
    }

    /// Get default vocoder
    pub fn get_default_vocoder(&self) -> Result<&dyn Vocoder> {
        let name = self
            .default_vocoder
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

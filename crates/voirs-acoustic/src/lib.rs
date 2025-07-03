//! # VoiRS Acoustic Models
//! 
//! Neural acoustic models for converting phonemes to mel spectrograms.
//! Supports VITS, FastSpeech2, and other state-of-the-art architectures.

use async_trait::async_trait;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Result type for acoustic model operations
pub type Result<T> = std::result::Result<T, AcousticError>;

/// Acoustic model specific error types
#[derive(Error, Debug)]
pub enum AcousticError {
    #[error("Model inference failed: {0}")]
    InferenceError(String),
    
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

/// A phoneme with its symbol and optional features
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Phoneme {
    /// Phoneme symbol (IPA or language-specific)
    pub symbol: String,
    /// Optional phoneme features
    pub features: Option<HashMap<String, String>>,
    /// Duration in seconds (if available)
    pub duration: Option<f32>,
}

impl Phoneme {
    /// Create new phoneme
    pub fn new<S: Into<String>>(symbol: S) -> Self {
        Self {
            symbol: symbol.into(),
            features: None,
            duration: None,
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

// Re-export main traits and types
pub use traits::{AcousticModel, AcousticModelFeature, AcousticModelMetadata};
pub use config::*;
pub use mel::*;
pub use backends::{Backend, BackendManager};
pub use models::{DummyAcousticModel, DummyAcousticConfig, ModelLoader};
pub use vits::{VitsModel, VitsConfig, TextEncoder, TextEncoderConfig};
pub use speaker::{
    SpeakerId, SpeakerMetadata, SpeakerEmbedding, SpeakerRegistry,
    MultiSpeakerModel, MultiSpeakerConfig,
    EmotionType, EmotionConfig, EmotionModel,
    VoiceCharacteristics, AgeGroup, Gender, Accent, VoiceQuality, PersonalityTrait,
};
pub use prosody::{
    ProsodyConfig, ProsodyController, ProsodyAdjustment,
    DurationConfig, PauseDurations, RhythmPattern,
    PitchConfig, IntonationPattern, VibratoConfig,
    EnergyConfig, EnergyContourPattern, VoiceQualityConfig,
};

pub mod traits;
pub mod config;
pub mod mel;
pub mod backends;
pub mod models;
pub mod vits;
pub mod fastspeech;
pub mod utils;
pub mod speaker;
pub mod prosody;

// Types are already public in the root module

/// Acoustic model manager with multiple architecture support
pub struct AcousticModelManager {
    models: HashMap<String, Box<dyn AcousticModel>>,
    default_model: Option<String>,
}

impl AcousticModelManager {
    /// Create new acoustic model manager
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
            default_model: None,
        }
    }
    
    /// Add acoustic model
    pub fn add_model(&mut self, name: String, model: Box<dyn AcousticModel>) {
        self.models.insert(name.clone(), model);
        
        // Set as default if it's the first model
        if self.default_model.is_none() {
            self.default_model = Some(name);
        }
    }
    
    /// Set default model
    pub fn set_default_model(&mut self, name: String) {
        if self.models.contains_key(&name) {
            self.default_model = Some(name);
        }
    }
    
    /// Get model by name
    pub fn get_model(&self, name: &str) -> Result<&dyn AcousticModel> {
        self.models
            .get(name)
            .map(|m| m.as_ref())
            .ok_or_else(|| AcousticError::ModelError(format!("Acoustic model '{}' not found", name)))
    }
    
    /// Get default model
    pub fn get_default_model(&self) -> Result<&dyn AcousticModel> {
        let name = self.default_model
            .as_ref()
            .ok_or_else(|| AcousticError::ConfigError("No default acoustic model set".to_string()))?;
        self.get_model(name)
    }
    
    /// List available models
    pub fn list_models(&self) -> Vec<&str> {
        self.models.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for AcousticModelManager {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl AcousticModel for AcousticModelManager {
    async fn synthesize(
        &self,
        phonemes: &[Phoneme],
        config: Option<&SynthesisConfig>,
    ) -> Result<MelSpectrogram> {
        let model = self.get_default_model()?;
        model.synthesize(phonemes, config).await
    }
    
    async fn synthesize_batch(
        &self,
        inputs: &[&[Phoneme]],
        configs: Option<&[SynthesisConfig]>,
    ) -> Result<Vec<MelSpectrogram>> {
        let model = self.get_default_model()?;
        model.synthesize_batch(inputs, configs).await
    }
    
    fn metadata(&self) -> AcousticModelMetadata {
        if let Ok(model) = self.get_default_model() {
            model.metadata()
        } else {
            AcousticModelMetadata {
                name: "Acoustic Model Manager".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
                architecture: "Manager".to_string(),
                supported_languages: vec![],
                sample_rate: 22050,
                mel_channels: 80,
                is_multi_speaker: false,
                speaker_count: None,
            }
        }
    }
    
    fn supports(&self, feature: AcousticModelFeature) -> bool {
        if let Ok(model) = self.get_default_model() {
            model.supports(feature)
        } else {
            false
        }
    }
    
    async fn set_speaker(&mut self, speaker_id: Option<u32>) -> Result<()> {
        // TODO: Forward to the appropriate model
        let _ = speaker_id;
        Ok(())
    }
}

// Type conversions are handled at the SDK level to avoid circular dependencies

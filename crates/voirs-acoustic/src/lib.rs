//! # VoiRS Acoustic Models
//!
//! Neural acoustic models for converting phonemes to mel spectrograms.
//! Supports VITS, FastSpeech2, and other state-of-the-art architectures.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
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

    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),

    #[error("Processing error: {0}")]
    Processing(String),

    #[error("Processing error: {message}")]
    ProcessingError { message: String },

    #[error("File operation error: {0}")]
    FileError(String),

    #[cfg(feature = "candle")]
    #[error("Candle error: {0}")]
    CandleError(#[from] candle_core::Error),

    #[error("G2P error: {0}")]
    G2pError(#[from] voirs_g2p::G2pError),
}

impl Clone for AcousticError {
    fn clone(&self) -> Self {
        match self {
            AcousticError::InferenceError(msg) => AcousticError::InferenceError(msg.clone()),
            AcousticError::ModelError(msg) => AcousticError::ModelError(msg.clone()),
            AcousticError::InputError(msg) => AcousticError::InputError(msg.clone()),
            AcousticError::ConfigError(msg) => AcousticError::ConfigError(msg.clone()),
            AcousticError::InvalidConfiguration(msg) => {
                AcousticError::InvalidConfiguration(msg.clone())
            }
            AcousticError::Processing(msg) => AcousticError::Processing(msg.clone()),
            AcousticError::ProcessingError { message } => AcousticError::ProcessingError {
                message: message.clone(),
            },
            AcousticError::FileError(msg) => AcousticError::FileError(msg.clone()),
            #[cfg(feature = "candle")]
            AcousticError::CandleError(err) => {
                AcousticError::InferenceError(format!("Candle error: {err}"))
            }
            AcousticError::G2pError(err) => {
                AcousticError::InferenceError(format!("G2P error: {err}"))
            }
        }
    }
}

/// Language codes supported by VoiRS
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum LanguageCode {
    /// English (US)
    EnUs,
    /// English (UK)
    EnGb,
    /// Japanese
    JaJp,
    /// Mandarin Chinese
    ZhCn,
    /// Korean
    KoKr,
    /// German
    DeDe,
    /// French
    FrFr,
    /// Spanish
    EsEs,
    /// Italian
    ItIt,
}

impl LanguageCode {
    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            LanguageCode::EnUs => "en-US",
            LanguageCode::EnGb => "en-GB",
            LanguageCode::JaJp => "ja-JP",
            LanguageCode::ZhCn => "zh-CN",
            LanguageCode::KoKr => "ko-KR",
            LanguageCode::DeDe => "de-DE",
            LanguageCode::FrFr => "fr-FR",
            LanguageCode::EsEs => "es-ES",
            LanguageCode::ItIt => "it-IT",
        }
    }
}

/// A phoneme with its symbol and optional features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Phoneme {
    /// Phoneme symbol (IPA or language-specific)
    pub symbol: String,
    /// Optional phoneme features
    pub features: Option<HashMap<String, String>>,
    /// Duration in seconds (if available)
    pub duration: Option<f32>,
}

impl PartialEq for Phoneme {
    fn eq(&self, other: &Self) -> bool {
        // Only compare symbol for equality (features and duration may vary)
        self.symbol == other.symbol
    }
}

impl Eq for Phoneme {}

impl std::hash::Hash for Phoneme {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Only hash the symbol (features and duration may vary)
        self.symbol.hash(state);
    }
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

/// Simple synthesis configuration for basic operations
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
    /// Emotion control configuration
    pub emotion: Option<crate::speaker::EmotionConfig>,
    /// Voice style control
    pub voice_style: Option<crate::speaker::VoiceStyleControl>,
}

impl SynthesisConfig {
    /// Create new synthesis configuration
    pub fn new() -> Self {
        Self {
            speed: 1.0,
            pitch_shift: 0.0,
            energy: 1.0,
            speaker_id: None,
            seed: None,
            emotion: None,
            voice_style: None,
        }
    }

    /// Set emotion for synthesis
    pub fn with_emotion(mut self, emotion: crate::speaker::EmotionConfig) -> Self {
        self.emotion = Some(emotion);
        self
    }

    /// Set voice style for synthesis
    pub fn with_voice_style(mut self, voice_style: crate::speaker::VoiceStyleControl) -> Self {
        self.voice_style = Some(voice_style);
        self
    }
}

impl Default for SynthesisConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl std::hash::Hash for SynthesisConfig {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Convert floats to bits for deterministic hashing
        self.speed.to_bits().hash(state);
        self.pitch_shift.to_bits().hash(state);
        self.energy.to_bits().hash(state);
        self.speaker_id.hash(state);
        self.seed.hash(state);
        // Note: emotion and voice_style are not hashed for simplicity
        // This is acceptable for cache keys as they're less common
    }
}

// Re-export main traits and types
pub use backends::{Backend, BackendManager};
pub use batch_processor::{
    BatchProcessingStats, BatchProcessor, BatchProcessorConfig, BatchProcessorTrait, BatchRequest,
    ErrorStats, MemoryStats, QueueStats, RequestPriority,
};
pub use config::*;
pub use mel::*;
pub use memory::{
    lazy::{
        ComponentRegistry, LazyComponent, MemmapFile, MemoryPressureHandler, MemoryPressureLevel,
        MemoryPressureStatus, ProgressiveLoader,
    },
    AdvancedPerformanceProfiler, MemoryOptimizer, OperationTimer, PerformanceMetrics,
    PerformanceMonitor, PerformanceReport, PerformanceSnapshot, PerformanceThresholds, PoolStats,
    ResultCache, SystemInfo, SystemMemoryInfo, TensorMemoryPool,
};
pub use metrics::{
    EvaluationConfig, EvaluationPreset, MetricStatistics, ObjectiveEvaluator, ObjectiveMetrics,
    PerceptualEvaluator, PerceptualMetrics, ProsodyEvaluator, ProsodyFeatures, ProsodyMetrics,
    QualityEvaluator, QualityMetrics, QualityStatistics, RhythmFeatures, WindowType,
};
pub use models::{DummyAcousticConfig, DummyAcousticModel, ModelLoader};
pub use optimization::{
    DistillationConfig, DistillationStrategy, HardwareOptimization, HardwareTarget, ModelOptimizer,
    OptimizationConfig, OptimizationMetrics, OptimizationReport, OptimizationTargets,
    PruningConfig, PruningStrategy, PruningType, QuantizationConfig as OptQuantizationConfig,
    QuantizationMethod as OptQuantizationMethod, QuantizationPrecision as OptQuantizationPrecision,
};
pub use prosody::{
    DurationConfig, EnergyConfig, EnergyContourPattern, IntonationPattern, PauseDurations,
    PitchConfig, ProsodyAdjustment, ProsodyConfig, ProsodyController, RhythmPattern, VibratoConfig,
    VoiceQualityConfig,
};
pub use quantization::{
    ModelQuantizer, QuantizationBenchmark, QuantizationConfig, QuantizationMethod,
    QuantizationParams, QuantizationPrecision, QuantizationStats, QuantizedTensor,
};
pub use simd::{
    Complex, FftWindow, SimdAudioEffects, SimdAudioProcessor, SimdCapabilities, SimdDispatcher,
    SimdFft, SimdLinearLayer, SimdMatrix, SimdMelComputer, SimdStft, StftWindow, WindowFunction,
};
pub use singing::{
    ArticulationMarking, BreathControlConfig, DynamicsMarking, FormantAdjustment, KeySignature,
    MusicalNote, MusicalPhrase, ResonanceConfig, SingingConfig, SingingTechnique,
    SingingVibratoConfig, SingingVoiceSynthesizer, VocalRegister, VoiceType,
};
pub use speaker::{
    Accent, AgeGroup, AudioFeatures, AudioReference, CloningQualityMetrics,
    CrossLanguageSpeakerAdapter, EmotionConfig, EmotionModel, EmotionType,
    FewShotSpeakerAdaptation, Gender, MultiSpeakerConfig, MultiSpeakerModel, PersonalityTrait,
    SpeakerEmbedding, SpeakerId, SpeakerMetadata, SpeakerRegistry, SpeakerVerificationResult,
    SpeakerVerifier, VoiceCharacteristics, VoiceCloningConfig, VoiceCloningQualityAssessor,
    VoiceQuality,
};
pub use streaming::{
    LatencyOptimizer, LatencyOptimizerConfig, LatencyStats, LatencyStrategy,
    PerformanceMeasurement, PerformancePredictor, StreamingConfig, StreamingMetrics,
    StreamingState, StreamingSynthesizer,
};
pub use traits::{AcousticModel, AcousticModelFeature, AcousticModelMetadata};
pub use vits::{TextEncoder, TextEncoderConfig, VitsConfig, VitsModel, VitsStreamingState};

pub mod backends;
pub mod batch_processor;
pub mod batching;
pub mod conditioning;
pub mod config;
pub mod fastspeech;
pub mod mel;
pub mod memory;
pub mod metrics;
pub mod model_manager;
pub mod models;
pub mod optimization;
pub mod parallel_attention;
pub mod performance_targets;
pub mod prosody;
pub mod quantization;
pub mod simd;
pub mod singing;
pub mod speaker;
pub mod streaming;
pub mod traits;
pub mod unified_conditioning;
pub mod utils;
pub mod vits;

/// Prelude for convenient imports
pub mod prelude {
    pub use crate::batch_processor::{
        BatchProcessingStats, BatchProcessor, BatchProcessorConfig, BatchProcessorTrait,
        BatchRequest, ErrorStats, MemoryStats, QueueStats, RequestPriority,
    };
    pub use crate::batching::{
        BatchStats, DynamicBatchConfig, DynamicBatcher, MemoryOptimization, PaddingStrategy,
        PendingSequence, ProcessingBatch,
    };
    pub use crate::model_manager::{ModelManager, ModelRegistry, TtsPipeline};
    pub use crate::parallel_attention::{
        AttentionCache, AttentionMemoryOptimization, AttentionStats, AttentionStrategy,
        ParallelAttentionConfig, ParallelMultiHeadAttention,
    };
    pub use crate::{
        AcousticError, AcousticModel, AcousticModelFeature, AcousticModelManager,
        AcousticModelMetadata, LanguageCode, MelSpectrogram, Phoneme, Result, SynthesisConfig,
    };
    pub use async_trait::async_trait;
}

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
            .ok_or_else(|| AcousticError::ModelError(format!("Acoustic model '{name}' not found")))
    }

    /// Get default model
    pub fn get_default_model(&self) -> Result<&dyn AcousticModel> {
        let name = self.default_model.as_ref().ok_or_else(|| {
            AcousticError::ConfigError("No default acoustic model set".to_string())
        })?;
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
        // Forward speaker setting to the default model
        let default_name = self.default_model.as_ref().ok_or_else(|| {
            AcousticError::ConfigError("No default acoustic model set".to_string())
        })?;

        if let Some(model) = self.models.get_mut(default_name) {
            model.set_speaker(speaker_id).await
        } else {
            Err(AcousticError::ModelError(format!(
                "Default acoustic model '{default_name}' not found"
            )))
        }
    }
}

// Type conversions are handled at the SDK level to avoid circular dependencies

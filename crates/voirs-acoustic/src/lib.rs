//! # VoiRS Acoustic Models
//!
//! Neural acoustic models for converting phonemes to mel spectrograms.
//! Supports VITS, FastSpeech2, and other state-of-the-art architectures.

// Allow pedantic lints that are acceptable for audio/DSP processing code
#![allow(clippy::cast_precision_loss)] // Acceptable for audio sample conversions
#![allow(clippy::cast_possible_truncation)] // Controlled truncation in audio processing
#![allow(clippy::cast_sign_loss)] // Intentional in index calculations
#![allow(clippy::missing_errors_doc)] // Many internal functions with self-documenting error types
#![allow(clippy::missing_panics_doc)] // Panics are documented where relevant
#![allow(clippy::unused_self)] // Some trait implementations require &self for consistency
#![allow(clippy::must_use_candidate)] // Not all return values need must_use annotation
#![allow(clippy::doc_markdown)] // Technical terms don't all need backticks
#![allow(clippy::unnecessary_wraps)] // Result wrappers maintained for API consistency
#![allow(clippy::float_cmp)] // Exact float comparisons are intentional in some contexts
#![allow(clippy::match_same_arms)] // Pattern matching clarity sometimes requires duplication
#![allow(clippy::module_name_repetitions)] // Type names often repeat module names
#![allow(clippy::struct_excessive_bools)] // Config structs naturally have many boolean flags
#![allow(clippy::too_many_lines)] // Some DSP functions are inherently complex
#![allow(clippy::needless_pass_by_value)] // Some functions designed for ownership transfer
#![allow(clippy::similar_names)] // Many similar variable names in DSP algorithms

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Result type for acoustic model operations
pub type Result<T> = std::result::Result<T, AcousticError>;

/// Acoustic model specific error types with enhanced diagnostic information
#[derive(Error, Debug)]
pub enum AcousticError {
    /// Model inference failed during synthesis or processing
    #[error("Model inference failed: {message}")]
    InferenceError { message: String },

    /// Model loading or initialization failed
    #[error("Model loading failed: {message}")]
    ModelError { message: String },

    /// Invalid input provided to the model
    #[error("Invalid input: {message}")]
    InputError { message: String },

    /// Configuration validation or parsing error
    #[error("Configuration error: {message}")]
    ConfigError { message: String },

    /// Processing error during synthesis pipeline
    #[error("Processing error: {message}")]
    ProcessingError { message: String },

    /// File operation error (reading, writing, or parsing)
    #[error("File operation error: {message}")]
    FileError { message: String },

    /// Backend-specific error from Candle framework
    #[cfg(feature = "candle")]
    #[error("Candle error: {0}")]
    CandleError(#[from] candle_core::Error),

    /// Grapheme-to-Phoneme conversion error
    #[error("G2P error: {0}")]
    G2pError(#[from] voirs_g2p::G2pError),
}

impl Clone for AcousticError {
    fn clone(&self) -> Self {
        match self {
            AcousticError::InferenceError { message } => AcousticError::InferenceError {
                message: message.clone(),
            },
            AcousticError::ModelError { message } => AcousticError::ModelError {
                message: message.clone(),
            },
            AcousticError::InputError { message } => AcousticError::InputError {
                message: message.clone(),
            },
            AcousticError::ConfigError { message } => AcousticError::ConfigError {
                message: message.clone(),
            },
            AcousticError::ProcessingError { message } => AcousticError::ProcessingError {
                message: message.clone(),
            },
            AcousticError::FileError { message } => AcousticError::FileError {
                message: message.clone(),
            },
            #[cfg(feature = "candle")]
            AcousticError::CandleError(err) => AcousticError::InferenceError {
                message: format!("Candle error: {err}"),
            },
            AcousticError::G2pError(err) => AcousticError::InferenceError {
                message: format!("G2P error: {err}"),
            },
        }
    }
}

/// Language codes supported by VoiRS
///
/// This enum represents the complete set of languages supported by the VoiRS acoustic models.
/// Each language may have region-specific variations (e.g., en-US vs en-GB).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum LanguageCode {
    /// English (United States)
    EnUs,
    /// English (United Kingdom)
    EnGb,
    /// Japanese (Japan)
    JaJp,
    /// Mandarin Chinese (China)
    ZhCn,
    /// Korean (South Korea)
    KoKr,
    /// German (Germany)
    DeDe,
    /// French (France)
    FrFr,
    /// Spanish (Spain)
    EsEs,
    /// Italian (Italy)
    ItIt,
    /// Portuguese (Brazil)
    PtBr,
    /// Portuguese (Portugal)
    PtPt,
    /// Russian (Russia)
    RuRu,
    /// Dutch (Netherlands)
    NlNl,
    /// Polish (Poland)
    PlPl,
    /// Turkish (Turkey)
    TrTr,
    /// Arabic (Saudi Arabia)
    ArSa,
    /// Hindi (India)
    HiIn,
    /// Swedish (Sweden)
    SvSe,
    /// Norwegian (Norway)
    NoNo,
    /// Finnish (Finland)
    FiFi,
    /// Danish (Denmark)
    DaDk,
    /// Czech (Czech Republic)
    CsCz,
    /// Greek (Greece)
    ElGr,
    /// Hebrew (Israel)
    HeIl,
    /// Thai (Thailand)
    ThTh,
    /// Vietnamese (Vietnam)
    ViVn,
    /// Indonesian (Indonesia)
    IdId,
    /// Malay (Malaysia)
    MsMy,
}

impl LanguageCode {
    /// Get string representation in BCP 47 format
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
            LanguageCode::PtBr => "pt-BR",
            LanguageCode::PtPt => "pt-PT",
            LanguageCode::RuRu => "ru-RU",
            LanguageCode::NlNl => "nl-NL",
            LanguageCode::PlPl => "pl-PL",
            LanguageCode::TrTr => "tr-TR",
            LanguageCode::ArSa => "ar-SA",
            LanguageCode::HiIn => "hi-IN",
            LanguageCode::SvSe => "sv-SE",
            LanguageCode::NoNo => "no-NO",
            LanguageCode::FiFi => "fi-FI",
            LanguageCode::DaDk => "da-DK",
            LanguageCode::CsCz => "cs-CZ",
            LanguageCode::ElGr => "el-GR",
            LanguageCode::HeIl => "he-IL",
            LanguageCode::ThTh => "th-TH",
            LanguageCode::ViVn => "vi-VN",
            LanguageCode::IdId => "id-ID",
            LanguageCode::MsMy => "ms-MY",
        }
    }

    /// Get ISO 639-1 language code (2-letter code)
    pub fn language_code(&self) -> &'static str {
        &self.as_str()[..2]
    }

    /// Get full language name in English
    pub fn language_name(&self) -> &'static str {
        match self {
            LanguageCode::EnUs | LanguageCode::EnGb => "English",
            LanguageCode::JaJp => "Japanese",
            LanguageCode::ZhCn => "Chinese",
            LanguageCode::KoKr => "Korean",
            LanguageCode::DeDe => "German",
            LanguageCode::FrFr => "French",
            LanguageCode::EsEs => "Spanish",
            LanguageCode::ItIt => "Italian",
            LanguageCode::PtBr | LanguageCode::PtPt => "Portuguese",
            LanguageCode::RuRu => "Russian",
            LanguageCode::NlNl => "Dutch",
            LanguageCode::PlPl => "Polish",
            LanguageCode::TrTr => "Turkish",
            LanguageCode::ArSa => "Arabic",
            LanguageCode::HiIn => "Hindi",
            LanguageCode::SvSe => "Swedish",
            LanguageCode::NoNo => "Norwegian",
            LanguageCode::FiFi => "Finnish",
            LanguageCode::DaDk => "Danish",
            LanguageCode::CsCz => "Czech",
            LanguageCode::ElGr => "Greek",
            LanguageCode::HeIl => "Hebrew",
            LanguageCode::ThTh => "Thai",
            LanguageCode::ViVn => "Vietnamese",
            LanguageCode::IdId => "Indonesian",
            LanguageCode::MsMy => "Malay",
        }
    }

    /// Parse from BCP 47 language tag string (case-insensitive)
    ///
    /// Accepts language tags in any case and normalizes to standard format:
    /// - "en-US", "EN-US", "en-us", "En-Us" all parse to EnUs
    pub fn parse(s: &str) -> Option<Self> {
        // Normalize to standard BCP 47 format: lowercase language, uppercase region
        // Split on hyphen, lowercase first part, uppercase second part
        let parts: Vec<&str> = s.split('-').collect();
        if parts.len() != 2 {
            return None;
        }

        let normalized = format!("{}-{}", parts[0].to_lowercase(), parts[1].to_uppercase());

        match normalized.as_str() {
            "en-US" => Some(LanguageCode::EnUs),
            "en-GB" => Some(LanguageCode::EnGb),
            "ja-JP" => Some(LanguageCode::JaJp),
            "zh-CN" => Some(LanguageCode::ZhCn),
            "ko-KR" => Some(LanguageCode::KoKr),
            "de-DE" => Some(LanguageCode::DeDe),
            "fr-FR" => Some(LanguageCode::FrFr),
            "es-ES" => Some(LanguageCode::EsEs),
            "it-IT" => Some(LanguageCode::ItIt),
            "pt-BR" => Some(LanguageCode::PtBr),
            "pt-PT" => Some(LanguageCode::PtPt),
            "ru-RU" => Some(LanguageCode::RuRu),
            "nl-NL" => Some(LanguageCode::NlNl),
            "pl-PL" => Some(LanguageCode::PlPl),
            "tr-TR" => Some(LanguageCode::TrTr),
            "ar-SA" => Some(LanguageCode::ArSa),
            "hi-IN" => Some(LanguageCode::HiIn),
            "sv-SE" => Some(LanguageCode::SvSe),
            "no-NO" => Some(LanguageCode::NoNo),
            "fi-FI" => Some(LanguageCode::FiFi),
            "da-DK" => Some(LanguageCode::DaDk),
            "cs-CZ" => Some(LanguageCode::CsCz),
            "el-GR" => Some(LanguageCode::ElGr),
            "he-IL" => Some(LanguageCode::HeIl),
            "th-TH" => Some(LanguageCode::ThTh),
            "vi-VN" => Some(LanguageCode::ViVn),
            "id-ID" => Some(LanguageCode::IdId),
            "ms-MY" => Some(LanguageCode::MsMy),
            _ => None,
        }
    }

    /// Get all supported language codes
    pub fn all() -> &'static [LanguageCode] {
        &[
            LanguageCode::EnUs,
            LanguageCode::EnGb,
            LanguageCode::JaJp,
            LanguageCode::ZhCn,
            LanguageCode::KoKr,
            LanguageCode::DeDe,
            LanguageCode::FrFr,
            LanguageCode::EsEs,
            LanguageCode::ItIt,
            LanguageCode::PtBr,
            LanguageCode::PtPt,
            LanguageCode::RuRu,
            LanguageCode::NlNl,
            LanguageCode::PlPl,
            LanguageCode::TrTr,
            LanguageCode::ArSa,
            LanguageCode::HiIn,
            LanguageCode::SvSe,
            LanguageCode::NoNo,
            LanguageCode::FiFi,
            LanguageCode::DaDk,
            LanguageCode::CsCz,
            LanguageCode::ElGr,
            LanguageCode::HeIl,
            LanguageCode::ThTh,
            LanguageCode::ViVn,
            LanguageCode::IdId,
            LanguageCode::MsMy,
        ]
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

// Advanced modules (0.1.0-alpha.3 additions)
pub mod acoustic_utils;
pub mod latency_optimizer;
pub mod neural_codec;
pub mod vad;

// Re-export advanced features
pub use latency_optimizer::{
    ChunkStrategy, LatencyBudget, LatencyMeasurement, LatencyOptimizer as AdvancedLatencyOptimizer,
    LatencyStatistics, ProcessingPriority,
};
pub use neural_codec::{CodecQualityMetrics, CodecType, NeuralCodec, NeuralCodecConfig};
pub use vad::{VadConfig, VadSegment, VoiceActivity, VoiceActivityDetector};

pub mod backends;
pub mod batch_processor;
pub mod batching;
pub mod cache;
pub mod conditioning;
pub mod config;
pub mod diagnostics;
pub mod error;
pub mod fastspeech;
pub mod fastspeech2_trainer;
pub mod fusion;
pub mod hub;
pub mod mel;
pub mod memory;
pub mod metrics;
pub mod model_manager;
pub mod model_warmup;
pub mod models;
pub mod optimization;
pub mod parallel_attention;
pub mod performance_targets;
pub mod production;
pub mod production_monitoring;
pub mod profiling;
pub mod profiling_integration;
pub mod prosody;
pub mod quantization;
pub mod scirs2_ops;
pub mod simd;
pub mod singing;
pub mod speaker;
pub mod streaming;
pub mod synthesis_cache;
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
    pub use crate::cache::{
        AdaptiveCache, AdaptiveCacheStats, CacheStats, CacheStrategy, LfuCache, PredictiveCache,
    };
    pub use crate::error::{
        ErrorCategory, ErrorContext, ErrorContextBuilder, ErrorSeverity, RecoverySuggestion,
    };
    pub use crate::model_manager::{ModelManager, ModelRegistry, TtsPipeline};
    pub use crate::parallel_attention::{
        AttentionCache, AttentionMemoryOptimization, AttentionStats, AttentionStrategy,
        ParallelAttentionConfig, ParallelMultiHeadAttention,
    };
    pub use crate::production::{
        CircuitBreaker, CircuitState, HealthChecker, HealthStatus, RateLimiter, ResourceLimits,
        RetryPolicy,
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
            .ok_or_else(|| AcousticError::ModelError {
                message: format!("Acoustic model '{name}' not found"),
            })
    }

    /// Get default model
    pub fn get_default_model(&self) -> Result<&dyn AcousticModel> {
        let name = self
            .default_model
            .as_ref()
            .ok_or_else(|| AcousticError::ConfigError {
                message: "No default acoustic model set".to_string(),
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
        let default_name =
            self.default_model
                .as_ref()
                .ok_or_else(|| AcousticError::ConfigError {
                    message: "No default acoustic model set".to_string(),
                })?;

        if let Some(model) = self.models.get_mut(default_name) {
            model.set_speaker(speaker_id).await
        } else {
            Err(AcousticError::ModelError {
                message: format!("Default acoustic model '{default_name}' not found"),
            })
        }
    }
}

// Type conversions are handled at the SDK level to avoid circular dependencies

#[cfg(test)]
mod language_tests {
    use super::*;

    #[test]
    fn test_language_code_string_representation() {
        assert_eq!(LanguageCode::EnUs.as_str(), "en-US");
        assert_eq!(LanguageCode::PtBr.as_str(), "pt-BR");
        assert_eq!(LanguageCode::RuRu.as_str(), "ru-RU");
        assert_eq!(LanguageCode::ArSa.as_str(), "ar-SA");
    }

    #[test]
    fn test_language_code_parsing() {
        assert_eq!(LanguageCode::parse("en-US"), Some(LanguageCode::EnUs));
        assert_eq!(LanguageCode::parse("pt-BR"), Some(LanguageCode::PtBr));
        assert_eq!(LanguageCode::parse("ru-RU"), Some(LanguageCode::RuRu));
        assert_eq!(LanguageCode::parse("invalid"), None);
    }

    #[test]
    fn test_language_names() {
        assert_eq!(LanguageCode::EnUs.language_name(), "English");
        assert_eq!(LanguageCode::PtBr.language_name(), "Portuguese");
        assert_eq!(LanguageCode::RuRu.language_name(), "Russian");
        assert_eq!(LanguageCode::ArSa.language_name(), "Arabic");
        assert_eq!(LanguageCode::HiIn.language_name(), "Hindi");
    }

    #[test]
    fn test_iso_language_codes() {
        assert_eq!(LanguageCode::EnUs.language_code(), "en");
        assert_eq!(LanguageCode::PtBr.language_code(), "pt");
        assert_eq!(LanguageCode::RuRu.language_code(), "ru");
        assert_eq!(LanguageCode::ArSa.language_code(), "ar");
    }

    #[test]
    fn test_all_languages() {
        let all = LanguageCode::all();
        assert_eq!(all.len(), 28); // Total number of supported languages
        assert!(all.contains(&LanguageCode::EnUs));
        assert!(all.contains(&LanguageCode::PtBr));
        assert!(all.contains(&LanguageCode::RuRu));
        assert!(all.contains(&LanguageCode::MsMy));
    }

    #[test]
    fn test_language_code_roundtrip() {
        for &lang in LanguageCode::all() {
            let string_repr = lang.as_str();
            let parsed = LanguageCode::parse(string_repr);
            assert_eq!(parsed, Some(lang), "Roundtrip failed for {:?}", lang);
        }
    }

    #[test]
    fn test_language_sorting() {
        let mut languages = vec![
            LanguageCode::ZhCn,
            LanguageCode::ArSa,
            LanguageCode::EnUs,
            LanguageCode::JaJp,
        ];
        languages.sort();
        // Should be sorted by enum order
        assert_eq!(languages[0], LanguageCode::EnUs);
        assert_eq!(languages[1], LanguageCode::JaJp);
    }

    #[test]
    fn test_new_language_support() {
        // Test newly added languages
        let new_languages = vec![
            (LanguageCode::PtBr, "pt-BR", "Portuguese"),
            (LanguageCode::RuRu, "ru-RU", "Russian"),
            (LanguageCode::NlNl, "nl-NL", "Dutch"),
            (LanguageCode::PlPl, "pl-PL", "Polish"),
            (LanguageCode::TrTr, "tr-TR", "Turkish"),
            (LanguageCode::ArSa, "ar-SA", "Arabic"),
            (LanguageCode::HiIn, "hi-IN", "Hindi"),
            (LanguageCode::SvSe, "sv-SE", "Swedish"),
            (LanguageCode::NoNo, "no-NO", "Norwegian"),
            (LanguageCode::FiFi, "fi-FI", "Finnish"),
        ];

        for (code, expected_str, expected_name) in new_languages {
            assert_eq!(code.as_str(), expected_str);
            assert_eq!(code.language_name(), expected_name);
        }
    }
}

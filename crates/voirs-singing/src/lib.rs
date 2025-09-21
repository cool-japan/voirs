//! # VoiRS Singing Voice Synthesis System
//!
//! This crate provides comprehensive singing voice synthesis capabilities including
//! musical note processing, pitch contour generation, rhythm control, vibrato modeling,
//! and musical format support.

// Temporarily allow common warnings to get the code compiling
#![allow(unused_imports)]
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_mut)]
#![allow(deprecated)]
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use voirs_singing::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     // Create a singing engine with default configuration
//!     let config = SingingConfig::default();
//!     let engine = SingingEngine::new(config).await?;
//!     
//!     // Create a simple musical score
//!     let mut score = MusicalScore::new();
//!     score.add_note(MusicalNote::new(60, 1.0, 1.0))?; // Middle C, 1 second
//!     
//!     // Synthesize the singing voice
//!     let request = SingingRequest::new(score, VoiceType::Soprano);
//!     let response = engine.synthesize(request).await?;
//!     
//!     println!("Generated {} samples", response.audio_data.len());
//!     Ok(())
//! }
//! ```
//!
//! ## Core Components
//!
//! - **SingingEngine**: Main engine for voice synthesis
//! - **MusicalScore**: Musical notation and timing
//! - **VoiceCharacteristics**: Voice parameters and qualities
//! - **EffectChain**: Audio effects processing
//! - **SynthesisProcessor**: Core synthesis algorithms
//!
//! ## Advanced Features
//!
//! ### Voice Cloning and Style Transfer
//! ```rust,ignore
//! # use voirs_singing::prelude::*;
//! # async fn example() -> Result<()> {
//! let style_transfer = StyleTransfer::new();
//! let target_style = StyleEmbedding::from_voice_samples(&voice_samples)?;
//! let result = style_transfer.apply_style(&audio_input, &target_style).await?;
//! # Ok(())
//! # }
//! ```
//!
//! ### Real-time Performance
//! ```rust,ignore
//! # use voirs_singing::prelude::*;
//! # async fn example() -> Result<()> {
//! let realtime_config = RealtimeConfig::low_latency();
//! let session = LiveSession::new(realtime_config).await?;
//!
//! // Process notes in real-time
//! let note = RealtimeNote::new(60, 0.5, VoiceType::Tenor);
//! session.play_note(note).await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Musical Intelligence
//!
//! The crate includes advanced musical analysis capabilities:
//!
//! - **Chord Recognition**: Automatic chord detection from audio
//! - **Key Detection**: Musical key identification
//! - **Rhythm Analysis**: Beat and tempo detection
//! - **Scale Analysis**: Musical scale recognition

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod advanced_techniques;
pub mod ai;
pub mod audio_processing;
pub mod config;
pub mod core;
pub mod custom_score;
pub mod effects;
pub mod formats;
pub mod gpu_acceleration;
pub mod granular_synthesis;
pub mod harmony;
pub mod historical_practice;
pub mod models;
pub mod musical_intelligence;
pub mod perceptual_quality;
pub mod performance_optimization;
pub mod physical_modeling;
pub mod pitch;
pub mod precision_quality;
pub mod realtime;
pub mod rhythm;
pub mod scalability;
pub mod score;
pub mod score_rendering;
pub mod styles;
pub mod synthesis;
pub mod techniques;
pub mod types;
pub mod utils;
pub mod vocal_effects;
pub mod voice;
pub mod voice_blending;
pub mod voice_conversion;
#[cfg(feature = "wasm-support")]
pub mod wasm_support;
pub mod zero_shot;

// Re-export main types and traits
pub use advanced_techniques::{
    AdvancedArticulationProcessor, AdvancedDynamicsProcessor, AdvancedTechniques, BendCurve,
    GraceNoteProcessor, MelismaProcessor, PitchBendProcessor, PitchBendSettings, RunPattern,
    VocalRunProcessor, VocalRunSettings,
};
pub use ai::{
    AutoHarmonizer, EmotionRecognizer, EmotionResult, ExpressionFeatures, HarmonyModel,
    HarmonyRules, ImprovisationAssistant, StyleEmbedding, StyleMetadata, StyleTransfer,
    StyleTransferConfig, StyleTransferResult, TransferQualityMetrics,
};
pub use audio_processing::{
    DynamicRangeProcessor, HighQualityResampler, InterpolationMethod, PanLaw,
    PhaseCoherenceProcessor, QualityLevel, StereoImagingProcessor,
};
pub use config::{SingingConfig, SingingConfigBuilder};
pub use core::{SingingEngine, SingingEngineBuilder};
pub use custom_score::{OptimizedScore, PerformanceHints, ScoreOptimizer};
pub use effects::{EffectChain, EffectProcessor, SingingEffect};
pub use formats::{FormatParser, MidiParser, MusicXmlParser};
pub use gpu_acceleration::{
    DeviceType, GpuAccelerated, GpuAccelerator, GpuConfig, GpuConfigBuilder, GpuError, MemoryUsage,
    TensorMemoryPool,
};
pub use granular_synthesis::{
    GrainEnvelope, GranularConfig, GranularSynthesisEffect, GranularTexture, WindowFunction,
};
pub use harmony::{HarmonyArrangement, HarmonyType, MultiVoiceSynthesizer, VoicePart};
pub use historical_practice::{
    ArticulationStyle, ExpressionStyle, HistoricalPeriod, HistoricalPractice, OrnamentsEngine,
    OrnamentsStyle, PeriodStyle, RegionalStyle, TuningSystem, VibratoStyle,
};
pub use models::{
    ModelType, SingingModel, SingingModelBuilder, TransformerConfig, TransformerSynthesisModel,
    VoiceModel,
};
pub use musical_intelligence::{
    ChordQuality, ChordRecognizer, ChordResult, KeyDetector, KeyMode, KeyResult, MusicalAnalysis,
    MusicalIntelligence, RhythmAnalyzer, RhythmResult, ScaleAnalyzer, ScaleResult,
};
pub use perceptual_quality::{
    ComprehensiveQualityReport, ExpressionReport, NaturalnessReport, PerceptualQualityTester,
    PerformanceReport, VoiceQualityReport,
};
pub use performance_optimization::{
    CompressionAlgorithm, CompressionEngine, EvictionPolicy, PrecomputationEngine, StreamingEngine,
    StreamingQuality, VoiceCache,
};
pub use physical_modeling::{
    AdvancedVocalTractModel, Complex32, PhysicalModelConfig, PhysicsAccuracyLevel, VocalTractModel,
    VowelPreset,
};
pub use pitch::{PitchContour, PitchGenerator, PitchProcessor};
pub use precision_quality::{
    ExpressionRecognitionReport, NaturalnessScoreReport, PitchAccuracyReport,
    PrecisionQualityAnalyzer, TimingAccuracyReport,
};
pub use realtime::{LiveSession, RealtimeConfig, RealtimeEngine, RealtimeNote};
pub use rhythm::{RhythmGenerator, RhythmProcessor, TimingController};
pub use scalability::{
    LargeScaleSynthesisRequest, LargeScaleSynthesisResult, MultiVoiceCoordinator,
    PerformanceMetrics, ScalabilityConfig, ScalabilityManager, ScalabilityStatus,
    SessionRequirements, VoiceRequirements,
};
pub use score::{MusicalNote, MusicalScore, ScoreProcessor};
pub use score_rendering::{
    RenderConfig, RenderFormat, ScoreRenderer, ScoreRendererBuilder, StaffPosition,
};
pub use styles::{
    CulturalVariant, MusicalStyle, Ornamentation, PerformanceGuidelines, PhraseShaping,
    StyleCharacteristics, TimbreQualities, VoiceType as StyleVoiceType,
};
pub use synthesis::{
    PrecisionMetricsReport, PrecisionTargets, SynthesisEngine, SynthesisProcessor, SynthesisResult,
};
pub use techniques::{
    BreathControl, LegatoProcessor, SingingTechnique, VibratoProcessor, VocalFry,
};
pub use types::{
    Expression, NoteEvent, SingingRequest, SingingResponse, SingingStats, VoiceCharacteristics,
    VoiceType,
};
pub use vocal_effects::{
    AutoTuneEffect, ChoirEffect, HarmonyGenerator, ScaleType, VocoderEffect, VoiceArrangement,
    VoicePartType, VoicingRules,
};
pub use voice::{VoiceBank, VoiceController, VoiceManager};
pub use voice_blending::{BlendConfig, BlendState, VoiceBlender, VoiceMorphParams};
pub use voice_conversion::{
    ConversionMethod, ConversionQuality, ConversionQualityMetrics, ConversionRequest,
    ConversionResult, ConversionSource, SpeakerEmbedding, VoiceConverter, VoiceQualityMetrics,
};
#[cfg(feature = "wasm-support")]
pub use wasm_support::{
    init_logging, WasmAudioPlayer, WasmError, WasmPerformanceMonitor, WasmRealtimeSynthesizer,
    WasmSingingEngine,
};
pub use zero_shot::{
    AdaptationMethod, AdaptationMetrics, AudioSample, QualityMode, ReferenceVoice, TargetVoiceSpec,
    VocalRange, ZeroShotConfig, ZeroShotRequest, ZeroShotResult, ZeroShotSynthesizer,
};

/// Result type for singing operations
pub type Result<T> = std::result::Result<T, Error>;

/// Error types for singing synthesis
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),

    /// Processing error
    #[error("Processing error: {0}")]
    Processing(String),

    /// Model error
    #[error("Model error: {0}")]
    Model(String),

    /// Audio error
    #[error("Audio error: {0}")]
    Audio(String),

    /// Synthesis error
    #[error("Synthesis error: {0}")]
    Synthesis(String),

    /// Score parsing error
    #[error("Score parsing error: {0}")]
    ScoreParsing(String),

    /// Voice error
    #[error("Voice error: {0}")]
    Voice(String),

    /// Effect processing error
    #[error("Effect processing error: {0}")]
    Effect(String),

    /// Format error
    #[error("Format error: {0}")]
    Format(String),

    /// Harmony error
    #[error("Harmony error: {0}")]
    Harmony(String),

    /// Validation error
    #[error("Validation error: {0}")]
    Validation(String),

    /// Unsupported feature error
    #[error("Unsupported feature: {0}")]
    UnsupportedFeature(String),

    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Candle error
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),

    /// FFT error
    #[error("FFT error: {0}")]
    Fft(String),

    /// MIDI error
    #[error("MIDI error: {0}")]
    #[cfg(feature = "midi-support")]
    Midi(#[from] midly::Error),
}

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::{
        ai::{
            AutoHarmonizer, EmotionRecognizer, EmotionResult, ExpressionFeatures, HarmonyModel,
            HarmonyRules, ImprovisationAssistant, StyleEmbedding, StyleMetadata, StyleTransfer,
            StyleTransferConfig, StyleTransferResult, TransferQualityMetrics,
        },
        audio_processing::{
            DynamicRangeProcessor, HighQualityResampler, InterpolationMethod, PanLaw,
            PhaseCoherenceProcessor, QualityLevel, StereoImagingProcessor,
        },
        config::{SingingConfig, SingingConfigBuilder},
        core::{SingingEngine, SingingEngineBuilder},
        custom_score::{OptimizedScore, PerformanceHints, ScoreOptimizer},
        effects::{EffectChain, EffectProcessor, SingingEffect},
        formats::{FormatParser, MidiParser, MusicXmlParser},
        granular_synthesis::{
            GrainEnvelope, GranularConfig, GranularSynthesisEffect, GranularTexture, WindowFunction,
        },
        harmony::{HarmonyArrangement, HarmonyType, MultiVoiceSynthesizer, VoicePart},
        models::{
            ModelType, SingingModel, SingingModelBuilder, TransformerConfig,
            TransformerSynthesisModel, VoiceModel,
        },
        musical_intelligence::{
            ChordQuality, ChordRecognizer, ChordResult, KeyDetector, KeyMode, KeyResult,
            MusicalAnalysis, MusicalIntelligence, RhythmAnalyzer, RhythmResult, ScaleAnalyzer,
            ScaleResult,
        },
        perceptual_quality::{
            ComprehensiveQualityReport, ExpressionReport, NaturalnessReport,
            PerceptualQualityTester, PerformanceReport, VoiceQualityReport,
        },
        performance_optimization::{
            CompressionAlgorithm, CompressionEngine, EvictionPolicy, PrecomputationEngine,
            StreamingEngine, StreamingQuality, VoiceCache,
        },
        physical_modeling::{
            AdvancedVocalTractModel, Complex32, PhysicalModelConfig, PhysicsAccuracyLevel,
            VocalTractModel, VowelPreset,
        },
        pitch::{PitchContour, PitchGenerator, PitchProcessor},
        realtime::{LiveSession, RealtimeConfig, RealtimeEngine, RealtimeNote},
        rhythm::{RhythmGenerator, RhythmProcessor, TimingController},
        scalability::{
            LargeScaleSynthesisRequest, LargeScaleSynthesisResult, MultiVoiceCoordinator,
            PerformanceMetrics, ScalabilityConfig, ScalabilityManager, ScalabilityStatus,
            SessionRequirements, VoiceRequirements,
        },
        score::{MusicalNote, MusicalScore, ScoreProcessor},
        styles::{
            CulturalVariant, MusicalStyle, Ornamentation, PerformanceGuidelines, PhraseShaping,
            StyleCharacteristics, TimbreQualities, VoiceType as StyleVoiceType,
        },
        synthesis::{SynthesisEngine, SynthesisProcessor, SynthesisResult},
        techniques::{
            BreathControl, LegatoProcessor, SingingTechnique, VibratoProcessor, VocalFry,
        },
        types::{
            Expression, NoteEvent, SingingRequest, SingingResponse, SingingStats,
            VoiceCharacteristics, VoiceType,
        },
        vocal_effects::{
            AutoTuneEffect, ChoirEffect, HarmonyGenerator, ScaleType, VocoderEffect,
            VoiceArrangement, VoicePartType, VoicingRules,
        },
        voice::{VoiceBank, VoiceController, VoiceManager},
        voice_blending::{BlendConfig, BlendState, VoiceBlender, VoiceMorphParams},
        voice_conversion::{
            ConversionMethod, ConversionQuality, ConversionQualityMetrics, ConversionRequest,
            ConversionResult, ConversionSource, SpeakerEmbedding, VoiceConverter,
            VoiceQualityMetrics,
        },
        zero_shot::{
            AdaptationMethod, AdaptationMetrics, AudioSample, QualityMode, ReferenceVoice,
            TargetVoiceSpec, VocalRange, ZeroShotConfig, ZeroShotRequest, ZeroShotResult,
            ZeroShotSynthesizer,
        },
        Error, Result,
    };
}

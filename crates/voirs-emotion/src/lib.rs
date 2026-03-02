//! # VoiRS Emotion Control System
//!
//! This crate provides comprehensive emotion expression control for voice synthesis,
//! enabling dynamic emotional expression through prosody modification, acoustic parameter
//! adjustment, and emotion interpolation.

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
#![allow(clippy::too_many_lines)] // Some functions are inherently complex
#![allow(clippy::needless_pass_by_value)] // Some functions designed for ownership transfer
#![allow(clippy::similar_names)] // Many similar variable names in algorithms
#![allow(clippy::unused_async)] // Public API functions may need async for consistency
#![allow(clippy::needless_range_loop)] // Range loops sometimes clearer than iterators
#![allow(clippy::uninlined_format_args)] // Explicit argument names can improve clarity
#![allow(clippy::manual_clamp)] // Manual clamping sometimes clearer
#![allow(clippy::return_self_not_must_use)] // Not all builder methods need must_use
#![allow(clippy::cast_possible_wrap)] // Controlled wrapping in processing code
#![allow(clippy::cast_lossless)] // Explicit casts preferred for clarity
#![allow(clippy::wildcard_imports)] // Prelude imports are convenient and standard
#![allow(clippy::format_push_string)] // Sometimes more readable than alternative
#![allow(clippy::redundant_closure_for_method_calls)] // Closures sometimes needed for type inference
#![warn(missing_docs)]
#![deny(unsafe_code)]

pub mod blending;
pub mod breath;
pub mod config;
pub mod consistency;
pub mod conversation;
pub mod core;
pub mod cultural;
pub mod custom;
pub mod debug;
pub mod editor;
pub mod formant;
pub mod history;
pub mod interpolation;
pub mod learning;
pub mod mobile;
pub mod morphing;
pub mod multimodal;
pub mod neural_transfer;
pub mod performance;
pub mod personality;
pub mod plugins;
pub mod presets;
pub mod prosody;
pub mod quality;
pub mod realtime;
pub mod recognition;
pub mod signal_processing;
pub mod spectral;
pub mod ssml;
pub mod testing;
pub mod thread_safety;
pub mod types;
pub mod validation;
pub mod variation;
pub mod vr_ar;

#[cfg(feature = "acoustic-integration")]
pub mod acoustic;

#[cfg(feature = "sdk-integration")]
pub mod sdk_integration;

#[cfg(feature = "evaluation-integration")]
pub mod evaluation_integration;

#[cfg(feature = "gpu")]
pub mod gpu;

#[cfg(feature = "wasm")]
pub mod wasm;

// Re-export main types and traits
pub use blending::{
    BlendMode, EmotionBlend, EmotionBlendBuilder, EmotionBlender as MultiEmotionBlender,
};
pub use breath::{
    BreathConfig, BreathGenerator, BreathPauseController, Pause, PauseAnalyzer, PauseType,
};
pub use config::{EmotionConfig, EmotionConfigBuilder};
pub use consistency::{
    CoherenceMetrics, EmotionConsistencyConfig, EmotionConsistencyManager, EmotionSegment,
};
pub use conversation::{
    CommunicationStyle, ContextAdaptation, ConversationConfig, ConversationContext,
    ConversationMetrics, ConversationTurn, SpeakerInfo, SpeakerRelationship, TopicContext,
};
pub use core::{EmotionProcessor, EmotionProcessorBuilder};
pub use cultural::{
    AppropratenessLevel, CulturalContext, CulturalEmotionAdapter, CulturalEmotionMapping,
    CulturalExpressionModifiers, HierarchyConsiderations, SocialContext, SocialHierarchy,
};
pub use custom::{
    CustomEmotionBuilder, CustomEmotionDefinition, CustomEmotionRegistry, CustomProsodyTemplate,
    EmotionVectorExt, VoiceQualityTemplate,
};
pub use debug::{
    AudioCharacteristics as DebugAudioCharacteristics, DebugConfig, DebugOutputFormat,
    EmotionDebugger, EmotionStateSnapshot, EmotionTransitionAnalysis, SnapshotPerformanceMetrics,
};
pub use editor::{EditorConfig, EmotionEditor};
pub use formant::{FormantAnalyzer, FormantSet, FormantShift, FormantSynthesizer, NUM_FORMANTS};
pub use history::{
    EmotionHistory, EmotionHistoryConfig, EmotionHistoryEntry, EmotionHistoryStats, EmotionPattern,
    EmotionTransition,
};
pub use interpolation::{EmotionInterpolator, InterpolationMethod};
pub use learning::{
    ContextPreference, EmotionFeedback, EmotionLearner, EmotionLearningConfig, FeedbackRatings,
    LearningStats, UserPreferenceProfile,
};
pub use mobile::{
    MobileDeviceInfo, MobileEmotionProcessor, MobileOptimizationConfig, MobileProcessingStatistics,
    NetworkQuality, PowerMode, ThermalState,
};
pub use morphing::{
    EasingFunction, EmotionBezierCurve, EmotionBlender, EmotionKeyframe, EmotionMorphConfig,
    EmotionTrajectory, MorphInterpolation,
};
pub use multimodal::{
    BodyPose, EyeTrackingData, FacialExpression, MultimodalConfig, MultimodalEmotionProcessor,
    MultimodalEmotionResult, PhysiologicalData,
};
pub use neural_transfer::{
    EmotionAttention, EmotionEmbedding, NeuralEmotionTransfer, NeuralEmotionTransferConfig,
    SpeakerEmbedding, EMOTION_EMBEDDING_DIM, SPEAKER_EMBEDDING_DIM,
};
pub use performance::{
    PerformanceMeasurement, PerformanceMonitor, PerformanceMonitorConfig, PerformanceTargets,
    PerformanceValidationResult, PerformanceValidator, SystemInfo,
};
pub use personality::{
    BigFiveTraits, EmotionalTendencies, PersonalityEmotionModifier, PersonalityModel,
    PersonalityStats,
};
pub use plugins::{
    AudioProcessor, EmotionAnalyzer, EmotionModel, Plugin, PluginConfig, PluginError,
    PluginManager, PluginMetadata, PluginRegistry, PluginResult, ProcessingHook,
};
pub use presets::{EmotionPreset, EmotionPresetLibrary};
pub use prosody::{ProsodyModifier, ProsodyParameters};
pub use quality::{
    QualityAnalyzer, QualityMeasurement, QualityMetadata, QualityRegressionTester, QualityTargets,
    RegressionTestResult,
};
pub use realtime::{
    AdaptationMetrics, AudioCharacteristics, EmotionSignal, RealtimeEmotionAdapter,
    RealtimeEmotionConfig,
};
pub use recognition::{
    EmotionRecognitionConfig, EmotionRecognitionResult, EmotionRecognizer, RecognitionMetadata,
    RecognitionMethod,
};
pub use signal_processing::{ProcessingQuality, SignalProcessingConfig, SignalProcessor};
pub use spectral::{SpectralConfig, SpectralEnvelope, SpectralProcessor};
pub use testing::{ABComparison, ABTestConfig, ABTestManager, ABTestStatistics, ABTestVariant};
pub use thread_safety::{
    ConcurrentEmotionProcessor, EmotionAccessInfo, EmotionCacheStats, EmotionProcessingInfo,
    EmotionProcessingMetrics, EmotionProcessingStatus, EmotionProcessingType,
    ThreadSafeEmotionCache,
};
pub use types::{
    Emotion, EmotionDimensions, EmotionIntensity, EmotionParameters, EmotionState, EmotionVector,
};
pub use validation::{
    EvaluationCriteria, PerceptualEvaluation, PerceptualValidationConfig,
    PerceptualValidationStudy, ValidationStatistics,
};
pub use variation::{
    AppliedVariation, NaturalVariationConfig, NaturalVariationGenerator, SpeakerCharacteristics,
    VariationPattern, VariationStatistics, VariationType,
};
pub use vr_ar::{
    AvatarEmotionSync, Direction3D, HandGesture, HapticPattern, Position3D, SpatialEmotionConfig,
    SpatialEmotionSource, VREmotionProcessor, VREnvironmentType,
};

#[cfg(feature = "gpu")]
pub use gpu::{GpuCapabilities, GpuEmotionProcessor};

#[cfg(feature = "wasm")]
pub use wasm::{
    WasmEmotionConfig, WasmEmotionParameters, WasmEmotionProcessor, WasmEmotionRecognitionResult,
};

#[cfg(feature = "sdk-integration")]
pub use sdk_integration::{
    AcousticModelHook, EmotionAudioEffectPlugin, EmotionController, EmotionSynthesisConfig,
    ProsodyConfig, VoiceQualityConfig,
};

#[cfg(feature = "evaluation-integration")]
pub use evaluation_integration::{
    EmotionAwareQualityEvaluator, EmotionEvaluationConfig, EmotionEvaluationContext,
    EmotionEvaluationPlugin, EmotionQualityMetadata, EmotionQualityResult,
    EmotionRecognitionResult as EvaluationEmotionRecognitionResult,
    StandardEmotionEvaluationPlugin,
};

/// Result type for emotion processing operations
pub type Result<T> = std::result::Result<T, Error>;

/// Error types for emotion processing
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),

    /// Processing error
    #[error("Processing error: {0}")]
    Processing(String),

    /// Interpolation error
    #[error("Interpolation error: {0}")]
    Interpolation(String),

    /// Validation error
    #[error("Validation error: {0}")]
    Validation(String),

    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Evaluation error
    #[error("Evaluation error: {0}")]
    EvaluationError(String),
}

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::{
        breath::{
            BreathConfig, BreathGenerator, BreathPauseController, Pause, PauseAnalyzer, PauseType,
        },
        config::{EmotionConfig, EmotionConfigBuilder},
        consistency::{
            CoherenceMetrics, EmotionConsistencyConfig, EmotionConsistencyManager, EmotionSegment,
        },
        conversation::{
            CommunicationStyle, ContextAdaptation, ConversationConfig, ConversationContext,
            ConversationMetrics, ConversationTurn, SpeakerInfo, SpeakerRelationship, TopicContext,
        },
        core::{EmotionProcessor, EmotionProcessorBuilder},
        cultural::{
            AppropratenessLevel, CulturalContext, CulturalEmotionAdapter, CulturalEmotionMapping,
            CulturalExpressionModifiers, HierarchyConsiderations, SocialContext, SocialHierarchy,
        },
        custom::{
            CustomEmotionBuilder, CustomEmotionDefinition, CustomEmotionRegistry,
            CustomProsodyTemplate, EmotionVectorExt, VoiceQualityTemplate,
        },
        debug::{
            AudioCharacteristics as DebugAudioCharacteristics, DebugConfig, DebugOutputFormat,
            EmotionDebugger, EmotionStateSnapshot, EmotionTransitionAnalysis,
            SnapshotPerformanceMetrics,
        },
        formant::{FormantAnalyzer, FormantSet, FormantShift, FormantSynthesizer, NUM_FORMANTS},
        history::{
            EmotionHistory, EmotionHistoryConfig, EmotionHistoryEntry, EmotionHistoryStats,
            EmotionPattern, EmotionTransition,
        },
        interpolation::{EmotionInterpolator, InterpolationMethod},
        learning::{
            ContextPreference, EmotionFeedback, EmotionLearner, EmotionLearningConfig,
            FeedbackRatings, LearningStats, UserPreferenceProfile,
        },
        mobile::{
            MobileDeviceInfo, MobileEmotionProcessor, MobileOptimizationConfig,
            MobileProcessingStatistics, NetworkQuality, PowerMode, ThermalState,
        },
        morphing::{
            EasingFunction, EmotionBezierCurve, EmotionBlender, EmotionKeyframe,
            EmotionMorphConfig, EmotionTrajectory, MorphInterpolation,
        },
        multimodal::{
            BodyPose, EyeTrackingData, FacialExpression, MultimodalConfig,
            MultimodalEmotionProcessor, MultimodalEmotionResult, PhysiologicalData,
        },
        neural_transfer::{
            EmotionAttention, EmotionEmbedding, NeuralEmotionTransfer, NeuralEmotionTransferConfig,
            SpeakerEmbedding, EMOTION_EMBEDDING_DIM, SPEAKER_EMBEDDING_DIM,
        },
        performance::{
            PerformanceMeasurement, PerformanceMonitor, PerformanceMonitorConfig,
            PerformanceTargets, PerformanceValidationResult, PerformanceValidator, SystemInfo,
        },
        personality::{
            BigFiveTraits, EmotionalTendencies, PersonalityEmotionModifier, PersonalityModel,
            PersonalityStats,
        },
        plugins::{
            AudioProcessor, EmotionAnalyzer, EmotionModel, Plugin, PluginConfig, PluginError,
            PluginManager, PluginMetadata, PluginRegistry, PluginResult, ProcessingHook,
        },
        presets::{EmotionPreset, EmotionPresetLibrary},
        prosody::{ProsodyModifier, ProsodyParameters},
        quality::{
            QualityAnalyzer, QualityMeasurement, QualityMetadata, QualityRegressionTester,
            QualityTargets, RegressionTestResult,
        },
        realtime::{
            AdaptationMetrics, AudioCharacteristics, EmotionSignal, RealtimeEmotionAdapter,
            RealtimeEmotionConfig,
        },
        recognition::{
            EmotionRecognitionConfig, EmotionRecognitionResult, EmotionRecognizer,
            RecognitionMetadata, RecognitionMethod,
        },
        spectral::{SpectralConfig, SpectralEnvelope, SpectralProcessor},
        testing::{ABComparison, ABTestConfig, ABTestManager, ABTestStatistics, ABTestVariant},
        types::{
            Emotion, EmotionDimensions, EmotionIntensity, EmotionParameters, EmotionState,
            EmotionVector,
        },
        validation::{
            EvaluationCriteria, PerceptualEvaluation, PerceptualValidationConfig,
            PerceptualValidationStudy, ValidationStatistics,
        },
        variation::{
            AppliedVariation, NaturalVariationConfig, NaturalVariationGenerator,
            SpeakerCharacteristics, VariationPattern, VariationStatistics, VariationType,
        },
        vr_ar::{
            AvatarEmotionSync, Direction3D, HandGesture, HapticPattern, Position3D,
            SpatialEmotionConfig, SpatialEmotionSource, VREmotionProcessor, VREnvironmentType,
        },
        Error, Result,
    };

    #[cfg(feature = "gpu")]
    pub use crate::gpu::{GpuCapabilities, GpuEmotionProcessor};

    #[cfg(feature = "sdk-integration")]
    pub use crate::sdk_integration::{
        AcousticModelHook, EmotionAudioEffectPlugin, EmotionController, EmotionSynthesisConfig,
        ProsodyConfig, VoiceQualityConfig,
    };

    #[cfg(feature = "evaluation-integration")]
    pub use crate::evaluation_integration::{
        EmotionAwareQualityEvaluator, EmotionEvaluationConfig, EmotionEvaluationContext,
        EmotionEvaluationPlugin, EmotionQualityMetadata, EmotionQualityResult,
        EmotionRecognitionResult as EvaluationEmotionRecognitionResult,
        StandardEmotionEvaluationPlugin,
    };
}

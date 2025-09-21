//! # VoiRS Emotion Control System
//!
//! This crate provides comprehensive emotion expression control for voice synthesis,
//! enabling dynamic emotional expression through prosody modification, acoustic parameter
//! adjustment, and emotion interpolation.

#![allow(clippy::uninlined_format_args)]
#![allow(clippy::while_let_on_iterator)]
#![allow(clippy::should_implement_trait)]
#![warn(missing_docs)]
#![deny(unsafe_code)]

pub mod config;
pub mod consistency;
pub mod conversation;
pub mod core;
pub mod cultural;
pub mod custom;
pub mod debug;
pub mod editor;
pub mod history;
pub mod interpolation;
pub mod learning;
pub mod mobile;
pub mod multimodal;
pub mod performance;
pub mod personality;
pub mod plugins;
pub mod presets;
pub mod prosody;
pub mod quality;
pub mod realtime;
pub mod recognition;
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
pub use multimodal::{
    BodyPose, EyeTrackingData, FacialExpression, MultimodalConfig, MultimodalEmotionProcessor,
    MultimodalEmotionResult, PhysiologicalData,
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
    EmotionRecognitionResult, StandardEmotionEvaluationPlugin,
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
        multimodal::{
            BodyPose, EyeTrackingData, FacialExpression, MultimodalConfig,
            MultimodalEmotionProcessor, MultimodalEmotionResult, PhysiologicalData,
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
        EmotionRecognitionResult, StandardEmotionEvaluationPlugin,
    };
}

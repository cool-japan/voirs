//! # VoiRS Voice Cloning System
//!
//! This crate provides comprehensive voice cloning capabilities including few-shot speaker
//! adaptation, speaker verification, voice similarity measurement, and cross-language cloning.

#![allow(unused_variables)]
#![allow(unused_imports)]
#![allow(unused_mut)]
#![allow(dead_code)]
#![allow(missing_docs)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::should_implement_trait)]
#![allow(clippy::derivable_impls)]
#![allow(clippy::vec_init_then_push)]
#![allow(clippy::needless_borrows_for_generic_args)]
#![deny(unsafe_code)]

pub mod ab_testing;
pub mod adaptation;
pub mod age_gender_adaptation;
pub mod api_standards;
pub mod authenticity;
pub mod auto_scaling;
pub mod cloning_wizard;
pub mod cloud_scaling;
pub mod config;
pub mod config_management;
pub mod consent;
pub mod consent_crypto;
pub mod core;
pub mod edge;
pub mod embedding;
pub mod emotion_transfer;
pub mod enterprise_sso;
pub mod error_handling;
pub mod few_shot;
pub mod gaming_plugins;
pub mod gpu_acceleration;
pub mod load_balancing;
pub mod long_term_adaptation;
pub mod long_term_stability;
pub mod memory_optimization;
pub mod misuse_prevention;
pub mod mobile;
pub mod model_loading;
pub mod multimodal;
pub mod neural_codec;
pub mod perceptual_evaluation;
pub mod performance_monitoring;
pub mod personality;
pub mod plugins;
pub mod preprocessing;
pub mod privacy_protection;
pub mod quality;
pub mod quality_visualization;
pub mod quantization;
pub mod realtime_streaming;
pub mod similarity;
pub mod storage;
pub mod streaming_adaptation;
pub mod thread_safety;
pub mod types;
pub mod usage_tracking;
pub mod verification;
pub mod visual_editor;
pub mod vits2;
pub mod voice_aging;
pub mod voice_library;
pub mod voice_morphing;
pub mod zero_shot;

#[cfg(feature = "acoustic-integration")]
pub mod acoustic;

pub mod conversion;
pub mod vocoder;

#[cfg(feature = "wasm")]
pub mod wasm;

// Re-export main types and traits
pub use ab_testing::{
    ABTestConfig, ABTestResults, ABTestingFramework, CriteriaWeights, EvaluationResult,
    ObjectiveComparisonResults, ObjectiveMetrics, PracticalSignificance, TestConclusion,
    TestCondition, TestMethodology, TestStatistics, TestStatus, TestStatusType,
};
pub use age_gender_adaptation::{
    AgeCategory, AgeGenderAdaptationConfig, AgeGenderAdaptationResult, AgeGenderAdapter,
    AgeGenderModel, F0Statistics, GenderCategory, SpectralCharacteristics, VoiceAdaptationTarget,
    VoiceCharacteristics, VoiceQualityMetrics,
};
pub use authenticity::{
    ArtifactDetection, ArtifactType, AuthenticityConfig, AuthenticityDetector,
    AuthenticityMetadata, AuthenticityResult, DetectorResult,
};
pub use auto_scaling::{
    AutoScaler, AutoScalingConfig, AutoScalingStats, AutoScalingStrategy, CostImpact,
    ExpectedImpact, InstanceHealth, InstanceState, PerformanceTier, ScalableGpuInstance,
    ScalingAction, ScalingDecision, ScalingTrigger, WorkloadPrediction,
};
pub use config::{CloningConfig, CloningConfigBuilder};
pub use config_management::{
    ConfigChangeEvent, ConfigChangeType, ConfigFileFormat, ConfigManagerSettings, ConfigMetadata,
    ConfigSnapshot, ConfigSource, Environment, SystemConfiguration, UnifiedConfigManager,
    ValidationError, ValidationResult, ValidationWarning,
};
pub use consent::{
    ConsentManager, ConsentPermissions, ConsentRecord, ConsentStatistics, ConsentStatus,
    ConsentType, ConsentUsageContext, ConsentUsageResult, ConsentVerificationMethod,
    SubjectIdentity, UsageRestrictions,
};
pub use core::{
    AdaptationConfig, RealtimeSynthesisChunk, RealtimeSynthesisConfig, RealtimeSynthesisRequest,
    RealtimeSynthesisResponse, SpeakerAdaptationResult, StreamSynthesisChunk,
    StreamSynthesisRequest, StreamingSynthesisConfig, SynthesisConfig, VoiceCloner,
    VoiceClonerBuilder,
};
pub use embedding::{SpeakerEmbedding, SpeakerEmbeddingExtractor};
pub use emotion_transfer::{
    EmotionCategory, EmotionTransfer, EmotionTransferConfig, EmotionTransferRequest,
    EmotionTransferResult, EmotionTransferStatistics, EmotionalCharacteristics, ProsodyFeatures,
};
pub use enterprise_sso::{
    AuthenticationMethod, AuthenticationRequest, AuthenticationResponse, AuthorizationResult,
    EnterpriseSSOManager, JWTConfig, OAuthProvider, PasswordPolicy, Permission, PermissionScope,
    RBACManager, Role, SAMLProvider, SSOConfig, UserSession,
};
pub use error_handling::{
    ErrorClassification, ErrorContext, ErrorRecoveryManager, ErrorReport, ErrorReportingConfig,
    ErrorSeverity, ErrorStatistics, PerformanceImpact, RecoverableError, RecoveryConfig,
    RecoveryOperation, RecoveryProgress, RecoveryResult, RecoveryState, RecoveryStrategy,
    RetryConfig,
};
pub use few_shot::{
    DistanceMetric, FewShotConfig, FewShotLearner, FewShotMetrics, FewShotResult,
    MetaLearningAlgorithm, SampleQuality,
};
pub use gaming_plugins::{
    AudioAttenuation, AudioRolloffType, CombatState, DynamicVoiceCharacteristics, EmotionalState,
    EnvironmentalFilter, GameContext, GameEngineType, GamePerformanceProfile, GameSession,
    GameVoiceProfile, GameVoiceResult, GamingPluginConfig, GamingPluginManager, ReverbSettings,
    SpatialAudioProperties, UnityPlugin, UnrealPlugin, VoiceInstance, VoicePlaybackState,
    WeatherEffects,
};
pub use gpu_acceleration::{
    GpuAccelerationConfig, GpuAccelerator, GpuDeviceType, GpuMemoryStats, GpuOperationType,
    GpuPerformanceMetrics, GpuUtils, TensorOperation, TensorOperationResult,
};
pub use load_balancing::{
    GpuAssignment, GpuDeviceInfo, GpuLoadBalancer, LoadBalancingConfig, LoadBalancingStats,
    LoadBalancingStrategy, PerformancePrediction,
};
pub use long_term_adaptation::{
    AdaptationResult, AdaptationStatistics, AdaptationStrategy, EfficiencyMetrics,
    FeedbackCategory, FeedbackContext, FeedbackType, LongTermAdaptationConfig,
    LongTermAdaptationEngine, ProcessingStatistics, RequestMetadata as AdaptationRequestMetadata,
    UserFeedback,
};
pub use long_term_stability::{
    RiskLevel, StabilityAssessment, StabilityCheckResult, StabilityConclusions,
    StabilityStatistics, StabilityTestConfig, StabilityTestResults, StabilityValidator,
};
pub use memory_optimization::{
    AlertSeverity, AlertType, AllocationInfo, AllocationType, CacheLimits, CompressedEmbedding,
    DetailedMemoryStats, GarbageCollectionResult, LeakDetectionConfig, LeakSummary,
    MemoryAuditReport, MemoryIssue, MemoryIssueType, MemoryLeakDetector, MemoryManager,
    MemoryOptimizationConfig, MemoryOptimizationRecommendation, MemoryPool, MemoryPoolSizes,
    MemoryPoolStats, MemoryRecommendation, MemoryStats, OptimizationCategory, OptimizationImpact,
    PerformanceImpactAnalysis, PooledObject, RecommendationPriority, RecommendationType,
};
pub use mobile::{
    CacheStrategy, MobileCloningConfig, MobileCloningStats, MobileDeviceInfo, MobilePlatform,
    MobileVoiceCloner, NeonCloningOptimizer, PowerMode, ThermalState,
};
pub use model_loading::{
    LoadingMetrics, LoadingStrategy, MemoryPressureLevel, ModelInterface, ModelLoadingConfig,
    ModelLoadingManager, ModelMemoryManager, ModelMetadata, ModelPreloader, PreloadPriority,
    PreloadRequest, UsagePatternAnalyzer,
};
pub use multimodal::{
    AudioVisualAligner, ExpressionAnalysis, FacialGeometry, FacialGeometryAnalyzer, HeadPose,
    LipFeatures, LipMovementAnalyzer, MultimodalCloneRequest, MultimodalCloner, MultimodalConfig,
    VisualDataType, VisualFeatureExtractor, VisualFeatures, VisualSample,
};
pub use neural_codec::{
    CodecCompressionRequest, CodecCompressionResult, CodecDecompressionResult, CodecMetadata,
    CodecPerformanceStats, CodecQualityMetrics, NeuralCodec, NeuralCodecConfig, NeuralCodecManager,
};
pub use perceptual_evaluation::{
    AgeGroup, AudioExperience, EvaluationResponse, EvaluationResults, EvaluationSample,
    EvaluationScores, EvaluationStudy, ExpertiseLevel, HearingStatus, PerceptualEvaluationConfig,
    PerceptualEvaluator, StudyResults,
};
pub use performance_monitoring::{
    AdaptationMonitor, PerformanceMeasurement, PerformanceMetrics, PerformanceMonitor,
    PerformanceStatistics, PerformanceTargets, TargetResults,
};
pub use personality::{
    AnalysisMetadata, ConversationalStyle, LinguisticPreferences, PersonalityComponents,
    PersonalityProfile, PersonalityTraits, PersonalityTransferConfig, PersonalityTransferEngine,
    SpeakingPatterns, TransferStats,
};
pub use plugins::{
    CloningPlugin, ExamplePlugin, ParameterConstraints, ParameterType, ParameterValue,
    PluginCapabilities, PluginConfig, PluginContext, PluginDependency, PluginHealth,
    PluginHealthStatus, PluginManager, PluginManagerConfig, PluginManifest, PluginMemoryStats,
    PluginMetrics, PluginOperationMetrics, PluginParameter, PluginPerformanceMetrics,
    PluginRegistry, PluginValidationResult,
};
pub use preprocessing::{AudioPreprocessor, PreprocessingPipeline};
pub use quality::{CloningQualityAssessor, QualityMetrics};
pub use quantization::{
    LayerQuantizationConfig, ModelQuantizer, QuantizationConfig, QuantizationMemoryAnalysis,
    QuantizationMethod, QuantizationPrecision, QuantizationResult, QuantizationStatsSummary,
    QuantizedTensor,
};
pub use realtime_streaming::{
    AdaptiveQualityController, AudioChunk, AudioDeviceConfig, AudioInputStream, AudioOutputStream,
    LatencyMode, NetworkConditions, QualityAdaptationStrategy, RealtimeStreamingEngine,
    SessionState, StreamingConfig, StreamingMetrics, StreamingSession, StreamingSessionType,
    VADAlgorithm, VoiceActivityDetector, VoiceProcessingPipeline,
};
pub use similarity::{SimilarityMeasurer, SimilarityScore};
pub use storage::{
    AccessStats, CompressionAlgorithm, CompressionInfo, CompressionStatistics, HealthIndicators,
    MaintenanceReport, MaintenanceStatistics, ModelFilter, SpeakerInfo, StorageConfig, StorageInfo,
    StorageOperation, StorageOperationResult, StorageStatistics, StorageTier, StoredModelMetadata,
    VoiceCharacteristicsSummary, VoiceModelStorage,
};
pub use streaming_adaptation::{
    AdaptationStep, StreamingAdaptationConfig, StreamingAdaptationManager,
    StreamingAdaptationManagerStats, StreamingAdaptationResult, StreamingAdaptationSession,
    StreamingAdaptationStats,
};
pub use thread_safety::{
    CacheStats, ComponentHealthMonitor, ComponentRegistry, ComponentStatus, ModelCache,
    OperationCoordinator, OperationGuard, OperationState, OperationStatus,
    PerformanceMetrics as ThreadPerformanceMetrics, ResourceLimits, ResourceMonitor,
};
pub use types::{
    CloningMethod, SpeakerData, SpeakerProfile, VoiceCloneRequest, VoiceCloneResult, VoiceSample,
};
pub use usage_tracking::{
    CloningOperation, CloningOperationType, ComplianceStatus, OperationRecord,
    OperationRequestMetadata as RequestMetadata, Priority, ResourceUsage, UsageOutcome,
    UsageRecord, UsageStatistics, UsageStatus, UsageTracker, UsageTrackingConfig, UserContext,
    UserPreferences,
};
pub use verification::{SpeakerVerifier, VerificationResult};
pub use vits2::{
    Vits2Cloner, Vits2Config, Vits2PerformanceStats, Vits2QualityMetrics, Vits2SynthesisRequest,
    Vits2SynthesisResult,
};
pub use voice_aging::{
    AgeTransition, AgingCharacteristics, AgingCurveType, AgingFactors, AgingQuality,
    AgingStatistics, ArticulatoryAging, FormantAging, ProsodicAging, RespiratoryAging,
    StabilityFactors, TemporalModel, TransitionType, VariationFactors, VoiceAgingConfig,
    VoiceAgingEngine, VoiceAgingModel, VoiceAgingResult, VoiceQualityAging,
};
pub use voice_morphing::{
    InterpolationMethod, MorphingWeight, RealtimeMorphingSession, VoiceMorpher,
    VoiceMorphingConfig, VoiceMorphingRequest, VoiceMorphingResult,
};
pub use zero_shot::{
    ReferenceVoice, ZeroShotCloner, ZeroShotConfig, ZeroShotMethod, ZeroShotResult,
};

#[cfg(feature = "wasm")]
pub use wasm::{
    WasmCloneRequest, WasmCloneResult, WasmCloningConfig, WasmConsentRecord, WasmQualityMetrics,
    WasmSpeakerProfile, WasmVerificationResult, WasmVoiceCloner, WasmVoiceSample,
};

/// Result type for voice cloning operations
pub type Result<T> = std::result::Result<T, Error>;

/// Error types for voice cloning
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

    /// Embedding error
    #[error("Embedding error: {0}")]
    Embedding(String),

    /// Verification error
    #[error("Verification error: {0}")]
    Verification(String),

    /// Quality assessment error
    #[error("Quality assessment error: {0}")]
    Quality(String),

    /// Insufficient data error
    #[error("Insufficient data: {0}")]
    InsufficientData(String),

    /// Validation error
    #[error("Validation error: {0}")]
    Validation(String),

    /// Invalid input error
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Consent management error
    #[error("Consent error: {0}")]
    Consent(String),

    /// Authentication error
    #[error("Authentication error: {0}")]
    Authentication(String),

    /// Usage tracking error
    #[error("Usage tracking error: {0}")]
    UsageTracking(String),

    /// Ethics and compliance error
    #[error("Ethics violation: {0}")]
    Ethics(String),

    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Candle error
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),
}

impl From<&str> for Error {
    fn from(s: &str) -> Self {
        Error::Processing(s.to_string())
    }
}

impl From<String> for Error {
    fn from(s: String) -> Self {
        Error::Processing(s)
    }
}

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::{
        adaptation::{AdaptationMethod, SpeakerAdapter},
        age_gender_adaptation::{
            AgeCategory, AgeGenderAdaptationConfig, AgeGenderAdaptationResult, AgeGenderAdapter,
            GenderCategory, VoiceAdaptationTarget, VoiceCharacteristics,
        },
        auto_scaling::{
            AutoScaler, AutoScalingConfig, AutoScalingStats, AutoScalingStrategy, CostImpact,
            ExpectedImpact, InstanceHealth, InstanceState, PerformanceTier, ScalableGpuInstance,
            ScalingAction, ScalingDecision, ScalingTrigger, WorkloadPrediction,
        },
        config::{CloningConfig, CloningConfigBuilder},
        consent::{
            ConsentManager, ConsentPermissions, ConsentRecord, ConsentStatistics, ConsentStatus,
            ConsentType, ConsentUsageContext, ConsentUsageResult, ConsentVerificationMethod,
            SubjectIdentity, UsageRestrictions,
        },
        core::{
            AdaptationConfig, RealtimeSynthesisConfig, RealtimeSynthesisRequest,
            RealtimeSynthesisResponse, SpeakerAdaptationResult, SynthesisConfig, VoiceCloner,
            VoiceClonerBuilder,
        },
        embedding::{SpeakerEmbedding, SpeakerEmbeddingExtractor},
        emotion_transfer::{
            EmotionCategory, EmotionTransfer, EmotionTransferConfig, EmotionTransferRequest,
            EmotionTransferResult, EmotionTransferStatistics, EmotionalCharacteristics,
            ProsodyFeatures,
        },
        error_handling::{
            ErrorClassification, ErrorContext, ErrorRecoveryManager, ErrorReport, ErrorSeverity,
            RecoverableError, RecoveryConfig, RecoveryResult, RecoveryStrategy,
        },
        few_shot::{
            DistanceMetric, FewShotConfig, FewShotLearner, FewShotMetrics, FewShotResult,
            MetaLearningAlgorithm, SampleQuality,
        },
        gpu_acceleration::{
            GpuAccelerationConfig, GpuAccelerator, GpuDeviceType, GpuMemoryStats, GpuOperationType,
            GpuPerformanceMetrics, GpuUtils, TensorOperation, TensorOperationResult,
        },
        load_balancing::{
            GpuAssignment, GpuDeviceInfo, GpuLoadBalancer, LoadBalancingConfig, LoadBalancingStats,
            LoadBalancingStrategy, PerformancePrediction,
        },
        long_term_stability::{
            RiskLevel, StabilityAssessment, StabilityCheckResult, StabilityTestConfig,
            StabilityTestResults, StabilityValidator,
        },
        memory_optimization::{
            CacheLimits, CompressedEmbedding, GarbageCollectionResult, MemoryManager,
            MemoryOptimizationConfig, MemoryOptimizationRecommendation, MemoryPool,
            MemoryPoolSizes, MemoryPoolStats, MemoryStats, OptimizationCategory,
            OptimizationImpact, PooledObject,
        },
        mobile::{
            CacheStrategy, MobileCloningConfig, MobileCloningStats, MobileDeviceInfo,
            MobilePlatform, MobileVoiceCloner, NeonCloningOptimizer, PowerMode, ThermalState,
        },
        neural_codec::{
            CodecCompressionRequest, CodecCompressionResult, CodecDecompressionResult,
            CodecMetadata, CodecPerformanceStats, CodecQualityMetrics, NeuralCodec,
            NeuralCodecConfig, NeuralCodecManager,
        },
        perceptual_evaluation::{
            AgeGroup, AudioExperience, EvaluationResponse, EvaluationResults, EvaluationSample,
            EvaluationScores, EvaluationStudy, ExpertiseLevel, HearingStatus,
            PerceptualEvaluationConfig, PerceptualEvaluator, StudyResults,
        },
        performance_monitoring::{
            AdaptationMonitor, PerformanceMeasurement, PerformanceMetrics, PerformanceMonitor,
            PerformanceStatistics, PerformanceTargets, TargetResults,
        },
        personality::{
            AnalysisMetadata, ConversationalStyle, LinguisticPreferences, PersonalityComponents,
            PersonalityProfile, PersonalityTraits, PersonalityTransferConfig,
            PersonalityTransferEngine, SpeakingPatterns, TransferStats,
        },
        plugins::{
            CloningPlugin, ExamplePlugin, PluginCapabilities, PluginConfig, PluginContext,
            PluginHealth, PluginHealthStatus, PluginManager, PluginManagerConfig,
            PluginValidationResult,
        },
        preprocessing::{AudioPreprocessor, PreprocessingPipeline},
        quality::{CloningQualityAssessor, QualityMetrics},
        quantization::{
            LayerQuantizationConfig, ModelQuantizer, QuantizationConfig,
            QuantizationMemoryAnalysis, QuantizationMethod, QuantizationPrecision,
            QuantizationResult, QuantizedTensor,
        },
        similarity::{SimilarityMeasurer, SimilarityScore},
        storage::{
            CompressionAlgorithm, MaintenanceReport, ModelFilter, SpeakerInfo, StorageConfig,
            StorageInfo, StorageOperation, StorageOperationResult, StorageStatistics, StorageTier,
            StoredModelMetadata, VoiceModelStorage,
        },
        streaming_adaptation::{
            AdaptationStep, StreamingAdaptationConfig, StreamingAdaptationManager,
            StreamingAdaptationManagerStats, StreamingAdaptationResult, StreamingAdaptationSession,
            StreamingAdaptationStats,
        },
        thread_safety::{
            CacheStats, ComponentHealthMonitor, ComponentRegistry, ComponentStatus, ModelCache,
            OperationCoordinator, OperationState, OperationStatus,
            PerformanceMetrics as ThreadPerformanceMetrics, ResourceLimits, ResourceMonitor,
            UnifiedConfigManager,
        },
        types::{
            CloningMethod, SpeakerData, SpeakerProfile, VoiceCloneRequest, VoiceCloneResult,
            VoiceSample,
        },
        usage_tracking::{
            CloningOperationType, OperationRecord, ResourceUsage, UsageRecord, UsageStatistics,
            UsageStatus, UsageTracker, UsageTrackingConfig, UserContext,
        },
        verification::{SpeakerVerifier, VerificationResult},
        vits2::{
            Vits2Cloner, Vits2Config, Vits2PerformanceStats, Vits2QualityMetrics,
            Vits2SynthesisRequest, Vits2SynthesisResult,
        },
        voice_aging::{
            AgeTransition, AgingCharacteristics, AgingCurveType, AgingFactors, AgingQuality,
            VoiceAgingConfig, VoiceAgingEngine, VoiceAgingModel, VoiceAgingResult,
        },
        voice_morphing::{
            InterpolationMethod, MorphingWeight, RealtimeMorphingSession, VoiceMorpher,
            VoiceMorphingConfig, VoiceMorphingRequest, VoiceMorphingResult,
        },
        zero_shot::{
            ReferenceVoice, ZeroShotCloner, ZeroShotConfig, ZeroShotMethod, ZeroShotResult,
        },
        Error, Result,
    };
}

//! # VoiRS Voice Conversion System
//!
//! This crate provides real-time voice conversion capabilities including speaker conversion,
//! age/gender transformation, voice morphing, and streaming voice conversion.

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod audio_libraries_update;
pub mod audio_quality_research;
pub mod cache;
pub mod cloud_scaling;
pub mod communication;
pub mod compression_research;
pub mod config;
pub mod core;
pub mod diagnostics;
pub mod fallback;
pub mod format;
pub mod gaming;

#[cfg(feature = "iot")]
pub mod iot;

pub mod ml_frameworks;
pub mod mobile;
pub mod models;
pub mod monitoring;
pub mod multi_target;
pub mod neural_vocoding;
pub mod optimizations;
pub mod pipeline_optimization;
pub mod platform_libraries;
pub mod processing;
pub mod profiling;
pub mod quality;
pub mod realtime;
pub mod realtime_libraries;
pub mod realtime_ml;
pub mod recognition;
pub mod scalability;
pub mod streaming;
pub mod streaming_platforms;
pub mod style_consistency;
pub mod style_transfer;
pub mod thread_safety;
pub mod transforms;
pub mod types;
pub mod webrtc_integration;
pub mod zero_shot;

#[cfg(feature = "acoustic-integration")]
pub mod acoustic;

#[cfg(feature = "cloning-integration")]
pub mod cloning;

#[cfg(feature = "emotion-integration")]
pub mod emotion;

#[cfg(feature = "spatial-integration")]
pub mod spatial;

#[cfg(feature = "wasm")]
pub mod wasm;

// Re-export main types and traits
pub use audio_libraries_update::{
    AudioLibrariesUpdater, AudioLibraryInfo, CompatibilityRisk, CompatibilityTestResult,
    LibraryVersionAnalysis, MigrationEffort, MigrationGuide, PerformanceImpact, SecuritySeverity,
    SecurityVulnerability, UpdatePriority, UpdateResult,
};
pub use audio_quality_research::{
    AnalysisStatistics, AudioQualityResearcher, ComprehensiveQualityAnalysis,
    HarmonicDistortionAnalysis, MultidimensionalQuality, NeuralQualityModel,
    PsychoacousticAnalysis, ResearchConfig, ResearchCriticalBandAnalysis, SpectralQualityAnalysis,
    TemporalQualityAnalysis, TonalityAnalysis,
};
pub use cache::{
    CacheConfig, CacheItemType, CachePolicy, CachePriority, CacheStatistics, CachedData,
    CachedItem, ConversionCacheSystem, LruCache, PerformanceMetrics,
};
pub use cloud_scaling::{
    CloudNode, CloudScalingConfig, CloudScalingController, ClusterMetrics,
    DistributedConversionRequest, DistributedConversionResult, LoadBalancingStrategy,
    NodeCapabilities, NodeResourceUsage, NodeStatus, RequestPriority, RetryConfig,
    ScalingAction as CloudScalingAction, ScalingDecision,
};
pub use compression_research::{
    CompressedAudio, CompressionAlgorithm, CompressionConfig, CompressionParameters,
    CompressionResearcher, CompressionStats, CompressionTarget, PredictionAnalyzer,
    PsychoacousticAnalyzer, TonalityDetector, VectorQuantizer,
};
pub use config::{ConversionConfig, ConversionConfigBuilder};
pub use core::{VoiceConverter, VoiceConverterBuilder};
pub use diagnostics::{
    DiagnosticAnalysis, DiagnosticSystem, HealthAssessment, IdentifiedIssue, IssueCategory,
    IssueSeverity, Recommendation, ReportType,
};
pub use fallback::{
    DegradationConfig, FailureType, FallbackContext, GracefulDegradationController,
    QualityThresholds,
};
pub use format::{
    AudioData, AudioFormat, AudioFormatType, AudioReader, AudioWriter, FormatConverter,
    FormatDetector, FormatQuality,
};
pub use gaming::{
    BevyIntegration, CustomIntegration, GameAudioConfig, GameEngine, GameEngineIntegration,
    GamePerformanceConstraints, GamePerformanceMetrics, GamePerformanceMonitor, GameVoiceMode,
    GameVoiceProcessor, GameVoiceSession, GodotIntegration, ThreadPriority, UnityIntegration,
    UnrealIntegration,
};

#[cfg(feature = "iot")]
pub use iot::{
    IoTConversionConfig, IoTConversionStatistics, IoTDeviceStatus, IoTPlatform, IoTPowerMode,
    IoTProcessingMode, IoTVoiceConverter, ResourceConstraints, ResourceUsage as IoTResourceUsage,
};

#[cfg(feature = "acoustic-integration")]
pub use acoustic::{
    AcousticConversionAdapter, AcousticConversionContext, AcousticConversionResult,
    AcousticFeatureConfig, AcousticFeatures, AcousticState, FormantFrequencies, HarmonicFeatures,
    TemporalFeatures, WindowType,
};

#[cfg(feature = "cloning-integration")]
pub use cloning::{
    CloningConversionAdapter, CloningConversionResult, CloningIntegration,
    CloningIntegrationConfig, TargetSpeakerInfo,
};

#[cfg(feature = "emotion-integration")]
pub use emotion::{EmotionConversionAdapter, EmotionParameters};

pub use ml_frameworks::{
    ActivationFunction, ConvLayerConfig, DevicePreference, InferenceMetrics, LayerSpec,
    MLFramework, MLFrameworkConfig, MLFrameworkManager, MLInferenceSession, MLModelMetadata,
    MemoryConfig, MemoryUsageStats, ModelArchitecture, ModelCapabilities, ModelOptimization,
    PerformanceConfig, QuantizationPrecision, RnnType, TensorDataType, TensorSpec,
};
pub use mobile::{
    MobileConversionConfig, MobileConversionStatistics, MobileDeviceInfo, MobilePlatform,
    MobileVoiceConverter, NeonOptimizer, PowerMode, ThermalState,
};
pub use models::{ConversionModel, ModelType};
pub use monitoring::{
    AlertSeverity, AlertType, MonitorConfig, QualityDashboard, QualityEvent, QualityMonitor,
    SessionDashboard, SystemOverview,
};
pub use multi_target::{
    MultiTargetConversionRequest, MultiTargetConversionResult, MultiTargetConverter,
    MultiTargetProcessingStats, NamedTarget, ProcessingMode,
};
pub use neural_vocoding::{
    ActivationType, AlgorithmBenchmark, AlgorithmPerformance, AttentionConfig,
    AudioProcessingParams, NeuralArchitectureConfig, NeuralVocoder, NeuralVocodingConfig,
    NeuralVocodingMetrics, VocodingAlgorithm, VocodingQuality,
};
pub use optimizations::{AudioBufferPool, ConversionPerformanceMonitor, SmallAudioOptimizer};
pub use pipeline_optimization::{
    AlgorithmVariant, OptimizationStatistics, OptimizedConversionPlan, OptimizedPipeline,
    OptimizedPipelineConfig,
};
pub use platform_libraries::{
    CpuFeatures, OptimizationLevel, PlatformConfig, PlatformOptimizer, PlatformStats,
    TargetPlatform,
};
pub use processing::{AudioBuffer, ProcessingPipeline};
pub use profiling::{
    BottleneckAnalyzer, BottleneckInfo, BottleneckThresholds, BottleneckType, ConversionProfiler,
    CpuAnalysis, CpuData, CpuSample, GlobalMetrics, MemoryAnalysis, MemoryData, MemorySample,
    PerformanceSummary, ProfilingConfig, ProfilingReport, ProfilingSession, SessionInfo,
    StageTimingInfo, TimingBreakdown, TimingData,
};
pub use quality::{
    AdaptiveQualityController, ArtifactDetector, CriticalBandAnalysis, DetailedQualityMetrics,
    DetectedArtifacts, LoudnessAnalysis, MaskingAnalysis, ObjectiveQualityMetrics,
    PerceptualOptimizationParams, PerceptualOptimizationResult, PerceptualOptimizer,
    QualityAssessment, QualityMetricsSystem, QualityTargetMeasurement, QualityTargetsAchievement,
    QualityTargetsConfig, QualityTargetsStatistics, QualityTargetsSystem,
};
pub use realtime::{RealtimeConfig, RealtimeConverter};
pub use realtime_libraries::{
    AudioBackend, BackendCapabilities, RealtimeBuffer, RealtimeConfig as RealtimeLibraryConfig,
    RealtimeLibraryManager, RealtimeStats,
};
pub use realtime_ml::{
    AdaptiveOptimizationState, BufferStrategy, CacheEvictionPolicy, CacheOptimizationConfig,
    ModelAdaptationConfig, OptimizationSnapshot, OptimizationStrategy, ParallelProcessingConfig,
    PerformanceSample, QuantizationLevel, RealtimeMLConfig, RealtimeMLOptimizer, RealtimeMetrics,
    ResourceUsage as RealtimeMLResourceUsage, StreamingOptimizationConfig,
};
pub use recognition::{
    ASRConfig, ASREngine, ASRTranscription, PhonemeAlignment, RecognitionGuidedConverter,
    RecognitionGuidedResult, RecognitionStats, SpeechGuidedParams, WordTimestamp,
};
pub use scalability::{
    MemoryEfficiencyMetrics, MemoryTracker, ResourceAllocationStrategy, ResourceMonitor,
    ResourceUsageMetrics, ScalabilityConfig, ScalabilityMetrics, ScalabilityTargets,
    ScalableConverter, ScalingAction, ScalingActionType, ScalingController, ScalingThresholds,
    ThroughputMetrics, ThroughputSample,
};
#[cfg(feature = "spatial-integration")]
pub use spatial::{
    AmbisonicsOutput, BinauralAudioOutput, HrtfMetadata, SpatialAudioOutput,
    SpatialConversionAdapter, SpatialDirection, SpatialPosition, SpatialVoiceSource,
};
pub use streaming::{StreamProcessor, StreamingConverter};
pub use streaming_platforms::{
    AdaptationDirection, AdaptationEvent, BandwidthAdaptationState, DiscordIntegration,
    FacebookIntegration, OBSIntegration, PlatformIntegration, RTMPIntegration, StreamConfig,
    StreamPerformanceMetrics, StreamPerformanceMonitor, StreamProcessor as StreamPlatformProcessor,
    StreamQuality, StreamSession, StreamVoiceMode, StreamingConstraints, StreamingPlatform,
    StreamlabsIntegration, TikTokIntegration, TwitchIntegration, XSplitIntegration,
    YouTubeIntegration,
};
pub use style_consistency::{
    ConsistencyThresholds, PreservationMode, StyleAdaptationSettings, StyleConsistencyConfig,
    StyleConsistencyEngine, StyleElement,
};
pub use style_transfer::{
    StyleCharacteristics, StyleTransferConfig, StyleTransferMethod, StyleTransferSystem,
};
pub use thread_safety::{
    AllocationInfo, AllocationTracker, AllocationType, BoundsViolation, BufferSafetyMonitor,
    ConcurrentConversionManager, ConcurrentConversionMetrics, LeakSeverity, MemoryLeak,
    MemorySafetyAuditor, MemorySafetyConfig, MemorySafetyReport, MemorySafetyStatus,
    ModelAccessStats, ModelUsageInfo, OperationGuard, OperationInfo, OperationState,
    OperationStatus, ReferenceTracker, RiskLevel, ThreadSafeModelManager, UnsafeOperation,
    UnsafeOperationType, ViolationSeverity, ViolationType,
};
pub use transforms::{
    AgeTransform, ChannelStrategy, GenderTransform, MultiChannelAudio, MultiChannelConfig,
    MultiChannelPitchTransform, MultiChannelTransform, PitchTransform, SpeedTransform, Transform,
    VoiceMorpher,
};
pub use types::{
    ConversionRequest, ConversionResult, ConversionTarget, ConversionType, VoiceCharacteristics,
};
pub use webrtc_integration::{
    ConversionMode, NetworkConditions, QualityMode, VoiceConversionConfig, WebRTCAudioConfig,
    WebRTCProcessingStatistics, WebRTCVoiceProcessor,
};
pub use zero_shot::{
    ReferenceVoiceDatabase, SpeakerEmbedding, UniversalVoiceModel, ZeroShotConfig,
    ZeroShotConverter,
};

#[cfg(feature = "wasm")]
pub use wasm::{
    BrowserCapabilities, ConversionParameters, WasmConversionConfig, WasmConversionStatistics,
    WasmSupportLevel, WasmVoiceConverter, WebAudioNodeType,
};

/// Result type for voice conversion operations
pub type Result<T> = std::result::Result<T, Error>;

/// Backward compatibility macros for easier migration
#[macro_export]
macro_rules! error_config {
    ($msg:expr) => {
        $crate::Error::config($msg.to_string())
    };
}

/// Create a processing error with the given message
#[macro_export]
macro_rules! error_processing {
    ($msg:expr) => {
        $crate::Error::processing($msg.to_string())
    };
}

/// Create a model error with the given message
#[macro_export]
macro_rules! error_model {
    ($msg:expr) => {
        $crate::Error::model($msg.to_string())
    };
}

/// Create an audio error with the given message
#[macro_export]
macro_rules! error_audio {
    ($msg:expr) => {
        $crate::Error::audio($msg.to_string())
    };
}

/// Create a realtime error with the given message
#[macro_export]
macro_rules! error_realtime {
    ($msg:expr) => {
        $crate::Error::realtime($msg.to_string())
    };
}

/// Create a streaming error with the given message
#[macro_export]
macro_rules! error_streaming {
    ($msg:expr) => {
        $crate::Error::streaming($msg.to_string())
    };
}

/// Create a buffer error with the given message
#[macro_export]
macro_rules! error_buffer {
    ($msg:expr) => {
        $crate::Error::buffer($msg.to_string())
    };
}

/// Create a transform error with the given message
#[macro_export]
macro_rules! error_transform {
    ($msg:expr) => {
        $crate::Error::transform($msg.to_string())
    };
}

/// Create a validation error with the given message
#[macro_export]
macro_rules! error_validation {
    ($msg:expr) => {
        $crate::Error::validation($msg.to_string())
    };
}

/// Create a runtime error with the given message
#[macro_export]
macro_rules! error_runtime {
    ($msg:expr) => {
        $crate::Error::runtime($msg.to_string())
    };
}

/// Error types for voice conversion
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Configuration error with detailed context
    #[error("Configuration error: {message}")]
    Config {
        /// The error message
        message: String,
        /// Additional context about the error
        context: Option<Box<ErrorContext>>,
        /// Suggestions for recovering from this error
        recovery_suggestions: Box<Vec<String>>,
    },

    /// Processing error with operation context
    #[error("Processing error in {operation}: {message}")]
    Processing {
        /// The operation that failed
        operation: String,
        /// The error message
        message: String,
        /// Additional context about the error
        context: Option<Box<ErrorContext>>,
        /// Suggestions for recovering from this error
        recovery_suggestions: Box<Vec<String>>,
    },

    /// Model error with model type information
    #[error("Model error ({model_type}): {message}")]
    Model {
        /// The type of model that caused the error
        model_type: String,
        /// The error message
        message: String,
        /// Additional context about the error
        context: Option<Box<ErrorContext>>,
        /// Suggestions for recovering from this error
        recovery_suggestions: Box<Vec<String>>,
    },

    /// Audio error with audio format details
    #[error("Audio error: {message}")]
    Audio {
        /// The error message
        message: String,
        /// Information about the audio that caused the error
        audio_info: Option<Box<AudioErrorInfo>>,
        /// Additional context about the error
        context: Option<Box<ErrorContext>>,
        /// Suggestions for recovering from this error
        recovery_suggestions: Box<Vec<String>>,
    },

    /// Real-time processing error with performance context
    #[error("Real-time processing error: {message}")]
    Realtime {
        /// The error message
        message: String,
        /// Performance context when the error occurred
        performance_context: Option<Box<PerformanceErrorInfo>>,
        /// Additional context about the error
        context: Option<Box<ErrorContext>>,
        /// Suggestions for recovering from this error
        recovery_suggestions: Box<Vec<String>>,
    },

    /// Streaming error with stream state
    #[error("Streaming error: {message}")]
    Streaming {
        /// The error message
        message: String,
        /// Information about the stream that caused the error
        stream_info: Option<Box<StreamErrorInfo>>,
        /// Additional context about the error
        context: Option<Box<ErrorContext>>,
        /// Suggestions for recovering from this error
        recovery_suggestions: Box<Vec<String>>,
    },

    /// Buffer error with buffer details
    #[error("Buffer error: {message}")]
    Buffer {
        /// The error message
        message: String,
        /// Information about the buffer that caused the error
        buffer_info: Option<Box<BufferErrorInfo>>,
        /// Additional context about the error
        context: Option<Box<ErrorContext>>,
        /// Suggestions for recovering from this error
        recovery_suggestions: Box<Vec<String>>,
    },

    /// Transform error with transform type
    #[error("Transform error ({transform_type}): {message}")]
    Transform {
        /// The type of transform that failed
        transform_type: String,
        /// The error message
        message: String,
        /// Additional context about the error
        context: Option<Box<ErrorContext>>,
        /// Suggestions for recovering from this error
        recovery_suggestions: Box<Vec<String>>,
    },

    /// Validation error with field information
    #[error("Validation error: {message}")]
    Validation {
        /// The error message
        message: String,
        /// The field that failed validation
        field: Option<String>,
        /// The expected value
        expected: Option<String>,
        /// The actual value that was provided
        actual: Option<String>,
        /// Additional context about the error
        context: Option<Box<ErrorContext>>,
        /// Suggestions for recovering from this error
        recovery_suggestions: Box<Vec<String>>,
    },

    /// Runtime error with execution context
    #[error("Runtime error: {message}")]
    Runtime {
        /// The error message
        message: String,
        /// Additional context about the error
        context: Option<Box<ErrorContext>>,
        /// Suggestions for recovering from this error
        recovery_suggestions: Box<Vec<String>>,
    },

    /// Memory safety error
    #[error("Memory safety error: {message}")]
    MemorySafety {
        /// The error message
        message: String,
        /// Information about the memory safety violation
        safety_info: Option<Box<MemorySafetyErrorInfo>>,
        /// Additional context about the error
        context: Option<Box<ErrorContext>>,
        /// Suggestions for recovering from this error
        recovery_suggestions: Box<Vec<String>>,
    },

    /// Thread safety error
    #[error("Thread safety error: {message}")]
    ThreadSafety {
        /// The error message
        message: String,
        /// Information about the thread safety violation
        thread_info: Option<Box<ThreadSafetyErrorInfo>>,
        /// Additional context about the error
        context: Option<Box<ErrorContext>>,
        /// Suggestions for recovering from this error
        recovery_suggestions: Box<Vec<String>>,
    },

    /// Resource exhaustion error
    #[error("Resource exhaustion: {resource_type}")]
    ResourceExhaustion {
        /// The type of resource that was exhausted
        resource_type: String,
        /// Current resource usage
        current_usage: Option<u64>,
        /// Resource limit that was exceeded
        limit: Option<u64>,
        /// Additional context about the error
        context: Option<Box<ErrorContext>>,
        /// Suggestions for recovering from this error
        recovery_suggestions: Box<Vec<String>>,
    },

    /// Timeout error
    #[error("Operation timeout: {operation} exceeded {timeout_ms}ms")]
    Timeout {
        /// The operation that timed out
        operation: String,
        /// The timeout threshold in milliseconds
        timeout_ms: u64,
        /// The actual elapsed time in milliseconds
        elapsed_ms: Option<u64>,
        /// Additional context about the error
        context: Option<Box<ErrorContext>>,
        /// Suggestions for recovering from this error
        recovery_suggestions: Box<Vec<String>>,
    },

    /// Compatibility error
    #[error("Compatibility error: {message}")]
    Compatibility {
        /// The error message
        message: String,
        /// The required version for compatibility
        required_version: Option<String>,
        /// The current version that is incompatible
        current_version: Option<String>,
        /// Additional context about the error
        context: Option<Box<ErrorContext>>,
        /// Suggestions for recovering from this error
        recovery_suggestions: Box<Vec<String>>,
    },

    /// I/O error with enhanced context
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error with enhanced context
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Candle ML framework error
    #[error("ML framework error: {0}")]
    Candle(#[from] candle_core::Error),
}

/// General error context information
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// Operation that was being performed
    pub operation: String,
    /// File or module where error occurred
    pub location: String,
    /// Thread ID where error occurred
    pub thread_id: Option<String>,
    /// Timestamp when error occurred
    pub timestamp: std::time::SystemTime,
    /// Additional context data
    pub additional_info: std::collections::HashMap<String, String>,
}

/// Audio-specific error information
#[derive(Debug, Clone)]
pub struct AudioErrorInfo {
    /// Sample rate of the audio
    pub sample_rate: Option<u32>,
    /// Number of channels
    pub channels: Option<u32>,
    /// Audio format
    pub format: Option<String>,
    /// Audio duration in seconds
    pub duration_seconds: Option<f32>,
    /// Buffer size
    pub buffer_size: Option<usize>,
}

/// Performance-related error information
#[derive(Debug, Clone)]
pub struct PerformanceErrorInfo {
    /// Current latency in milliseconds
    pub current_latency_ms: Option<f32>,
    /// Target latency in milliseconds
    pub target_latency_ms: Option<f32>,
    /// CPU usage percentage
    pub cpu_usage_percent: Option<f32>,
    /// Memory usage in bytes
    pub memory_usage_bytes: Option<u64>,
    /// Processing queue size
    pub queue_size: Option<usize>,
}

/// Stream-specific error information
#[derive(Debug, Clone)]
pub struct StreamErrorInfo {
    /// Stream ID
    pub stream_id: Option<String>,
    /// Current stream state
    pub stream_state: Option<String>,
    /// Buffer level
    pub buffer_level: Option<usize>,
    /// Dropped samples count
    pub dropped_samples: Option<u64>,
    /// Stream position
    pub stream_position: Option<u64>,
}

/// Buffer-specific error information
#[derive(Debug, Clone)]
pub struct BufferErrorInfo {
    /// Buffer ID
    pub buffer_id: Option<String>,
    /// Buffer size
    pub buffer_size: Option<usize>,
    /// Available space
    pub available_space: Option<usize>,
    /// Buffer type
    pub buffer_type: Option<String>,
    /// Access pattern
    pub access_pattern: Option<String>,
}

/// Memory safety error information
#[derive(Debug, Clone)]
pub struct MemorySafetyErrorInfo {
    /// Type of memory safety violation
    pub violation_type: String,
    /// Memory address if applicable
    pub memory_address: Option<String>,
    /// Allocation size
    pub allocation_size: Option<u64>,
    /// Allocation age
    pub allocation_age_ms: Option<u64>,
    /// Thread that allocated the memory
    pub allocating_thread: Option<String>,
}

/// Thread safety error information
#[derive(Debug, Clone)]
pub struct ThreadSafetyErrorInfo {
    /// Type of thread safety violation
    pub violation_type: String,
    /// Thread IDs involved
    pub thread_ids: Vec<String>,
    /// Resource being contended
    pub resource_name: Option<String>,
    /// Lock state
    pub lock_state: Option<String>,
    /// Deadlock detection info
    pub deadlock_info: Option<String>,
}

// Manual implementation of Clone for Error enum to handle non-Clone types
impl Clone for Error {
    fn clone(&self) -> Self {
        match self {
            Error::Config {
                message,
                context,
                recovery_suggestions,
            } => Error::Config {
                message: message.clone(),
                context: context.as_ref().map(|c| Box::new((**c).clone())),
                recovery_suggestions: recovery_suggestions.clone(),
            },
            Error::Processing {
                operation,
                message,
                context,
                recovery_suggestions,
            } => Error::Processing {
                operation: operation.clone(),
                message: message.clone(),
                context: context.as_ref().map(|c| Box::new((**c).clone())),
                recovery_suggestions: recovery_suggestions.clone(),
            },
            Error::Model {
                model_type,
                message,
                context,
                recovery_suggestions,
            } => Error::Model {
                model_type: model_type.clone(),
                message: message.clone(),
                context: context.as_ref().map(|c| Box::new((**c).clone())),
                recovery_suggestions: recovery_suggestions.clone(),
            },
            Error::Audio {
                message,
                audio_info,
                context,
                recovery_suggestions,
            } => Error::Audio {
                message: message.clone(),
                audio_info: audio_info.as_ref().map(|a| Box::new((**a).clone())),
                context: context.as_ref().map(|c| Box::new((**c).clone())),
                recovery_suggestions: recovery_suggestions.clone(),
            },
            Error::Realtime {
                message,
                performance_context,
                context,
                recovery_suggestions,
            } => Error::Realtime {
                message: message.clone(),
                performance_context: performance_context
                    .as_ref()
                    .map(|p| Box::new((**p).clone())),
                context: context.as_ref().map(|c| Box::new((**c).clone())),
                recovery_suggestions: recovery_suggestions.clone(),
            },
            Error::Streaming {
                message,
                stream_info,
                context,
                recovery_suggestions,
            } => Error::Streaming {
                message: message.clone(),
                stream_info: stream_info.as_ref().map(|s| Box::new((**s).clone())),
                context: context.as_ref().map(|c| Box::new((**c).clone())),
                recovery_suggestions: recovery_suggestions.clone(),
            },
            Error::Buffer {
                message,
                buffer_info,
                context,
                recovery_suggestions,
            } => Error::Buffer {
                message: message.clone(),
                buffer_info: buffer_info.as_ref().map(|b| Box::new((**b).clone())),
                context: context.as_ref().map(|c| Box::new((**c).clone())),
                recovery_suggestions: recovery_suggestions.clone(),
            },
            Error::Transform {
                transform_type,
                message,
                context,
                recovery_suggestions,
            } => Error::Transform {
                transform_type: transform_type.clone(),
                message: message.clone(),
                context: context.as_ref().map(|c| Box::new((**c).clone())),
                recovery_suggestions: recovery_suggestions.clone(),
            },
            Error::Validation {
                message,
                field,
                expected,
                actual,
                context,
                recovery_suggestions,
            } => Error::Validation {
                message: message.clone(),
                field: field.clone(),
                expected: expected.clone(),
                actual: actual.clone(),
                context: context.as_ref().map(|c| Box::new((**c).clone())),
                recovery_suggestions: recovery_suggestions.clone(),
            },
            Error::Runtime {
                message,
                context,
                recovery_suggestions,
            } => Error::Runtime {
                message: message.clone(),
                context: context.as_ref().map(|c| Box::new((**c).clone())),
                recovery_suggestions: recovery_suggestions.clone(),
            },
            Error::MemorySafety {
                message,
                safety_info,
                context,
                recovery_suggestions,
            } => Error::MemorySafety {
                message: message.clone(),
                safety_info: safety_info.as_ref().map(|s| Box::new((**s).clone())),
                context: context.as_ref().map(|c| Box::new((**c).clone())),
                recovery_suggestions: recovery_suggestions.clone(),
            },
            Error::ThreadSafety {
                message,
                thread_info,
                context,
                recovery_suggestions,
            } => Error::ThreadSafety {
                message: message.clone(),
                thread_info: thread_info.as_ref().map(|t| Box::new((**t).clone())),
                context: context.as_ref().map(|c| Box::new((**c).clone())),
                recovery_suggestions: recovery_suggestions.clone(),
            },
            Error::ResourceExhaustion {
                resource_type,
                current_usage,
                limit,
                context,
                recovery_suggestions,
            } => Error::ResourceExhaustion {
                resource_type: resource_type.clone(),
                current_usage: *current_usage,
                limit: *limit,
                context: context.as_ref().map(|c| Box::new((**c).clone())),
                recovery_suggestions: recovery_suggestions.clone(),
            },
            Error::Timeout {
                operation,
                timeout_ms,
                elapsed_ms,
                context,
                recovery_suggestions,
            } => Error::Timeout {
                operation: operation.clone(),
                timeout_ms: *timeout_ms,
                elapsed_ms: *elapsed_ms,
                context: context.as_ref().map(|c| Box::new((**c).clone())),
                recovery_suggestions: recovery_suggestions.clone(),
            },
            Error::Compatibility {
                message,
                required_version,
                current_version,
                context,
                recovery_suggestions,
            } => Error::Compatibility {
                message: message.clone(),
                required_version: required_version.clone(),
                current_version: current_version.clone(),
                context: context.as_ref().map(|c| Box::new((**c).clone())),
                recovery_suggestions: recovery_suggestions.clone(),
            },
            Error::Io(e) => Error::Config {
                message: format!("I/O error: {e}"),
                context: None,
                recovery_suggestions: Box::new(vec![
                    "Check file permissions".to_string(),
                    "Verify file path exists".to_string(),
                ]),
            },
            Error::Serialization(e) => Error::Config {
                message: format!("Serialization error: {e}"),
                context: None,
                recovery_suggestions: Box::new(vec![
                    "Check data format".to_string(),
                    "Verify JSON structure".to_string(),
                ]),
            },
            Error::Candle(e) => Error::Model {
                model_type: "ML Framework".to_string(),
                message: format!("ML framework error: {e}"),
                context: None,
                recovery_suggestions: Box::new(vec![
                    "Check model file".to_string(),
                    "Verify input dimensions".to_string(),
                ]),
            },
        }
    }
}

impl Error {
    /// Create a simple configuration error (backwards compatibility)
    pub fn config(message: String) -> Self {
        Self::Config {
            message,
            context: None,
            recovery_suggestions: Box::new(vec!["Check configuration parameters".to_string()]),
        }
    }

    /// Create a configuration error with context
    pub fn config_with_context(message: String, context: ErrorContext) -> Self {
        Self::Config {
            message,
            context: Some(Box::new(context)),
            recovery_suggestions: Box::new(vec![]),
        }
    }

    /// Create a simple processing error (backwards compatibility)
    pub fn processing(message: String) -> Self {
        Self::Processing {
            operation: "processing".to_string(),
            message,
            context: None,
            recovery_suggestions: Box::new(vec!["Review processing parameters".to_string()]),
        }
    }

    /// Create a simple model error (backwards compatibility)
    pub fn model(message: String) -> Self {
        Self::Model {
            model_type: "unknown".to_string(),
            message,
            context: None,
            recovery_suggestions: Box::new(vec!["Check model configuration".to_string()]),
        }
    }

    /// Create a simple audio error (backwards compatibility)
    pub fn audio(message: String) -> Self {
        Self::Audio {
            message,
            audio_info: None,
            context: None,
            recovery_suggestions: Box::new(vec!["Check audio format".to_string()]),
        }
    }

    /// Create a simple realtime error (backwards compatibility)
    pub fn realtime(message: String) -> Self {
        Self::Realtime {
            message,
            performance_context: None,
            context: None,
            recovery_suggestions: Box::new(vec!["Reduce processing load".to_string()]),
        }
    }

    /// Create a simple streaming error (backwards compatibility)
    pub fn streaming(message: String) -> Self {
        Self::Streaming {
            message,
            stream_info: None,
            context: None,
            recovery_suggestions: Box::new(vec!["Check stream configuration".to_string()]),
        }
    }

    /// Create a simple buffer error (backwards compatibility)
    pub fn buffer(message: String) -> Self {
        Self::Buffer {
            message,
            buffer_info: None,
            context: None,
            recovery_suggestions: Box::new(vec!["Check buffer size and usage".to_string()]),
        }
    }

    /// Create a simple transform error (backwards compatibility)
    pub fn transform(message: String) -> Self {
        Self::Transform {
            transform_type: "unknown".to_string(),
            message,
            context: None,
            recovery_suggestions: Box::new(vec!["Check transform parameters".to_string()]),
        }
    }

    /// Create a simple validation error (backwards compatibility)
    pub fn validation(message: String) -> Self {
        Self::Validation {
            message,
            field: None,
            expected: None,
            actual: None,
            context: None,
            recovery_suggestions: Box::new(vec!["Check input parameters".to_string()]),
        }
    }

    /// Create a simple runtime error (backwards compatibility)
    pub fn runtime(message: String) -> Self {
        Self::Runtime {
            message,
            context: None,
            recovery_suggestions: Box::new(vec!["Check system resources".to_string()]),
        }
    }

    /// Create a processing error with context
    pub fn processing_with_context(
        operation: String,
        message: String,
        context: ErrorContext,
    ) -> Self {
        Self::Processing {
            operation,
            message,
            context: Some(Box::new(context)),
            recovery_suggestions: Box::new(vec![]),
        }
    }

    /// Create a validation error with detailed information
    pub fn validation_detailed(
        message: String,
        field: Option<String>,
        expected: Option<String>,
        actual: Option<String>,
    ) -> Self {
        let mut recovery_suggestions = vec!["Check input parameters".to_string()];

        if let (Some(exp), Some(act)) = (&expected, &actual) {
            recovery_suggestions.push(format!("Expected {exp}, got {act}"));
        }

        if let Some(field_name) = &field {
            recovery_suggestions.push(format!("Verify {field_name} field is correctly set"));
        }

        Self::Validation {
            message,
            field,
            expected,
            actual,
            context: None,
            recovery_suggestions: Box::new(recovery_suggestions),
        }
    }

    /// Create an audio error with audio information
    pub fn audio_with_info(message: String, audio_info: AudioErrorInfo) -> Self {
        let mut recovery_suggestions = vec!["Check audio format compatibility".to_string()];

        if let Some(sr) = audio_info.sample_rate {
            if sr < 16000 || sr > 48000 {
                recovery_suggestions.push("Use supported sample rate (16kHz-48kHz)".to_string());
            }
        }

        if let Some(channels) = audio_info.channels {
            if channels > 2 {
                recovery_suggestions.push("Convert to mono or stereo audio".to_string());
            }
        }

        Self::Audio {
            message,
            audio_info: Some(Box::new(audio_info)),
            context: None,
            recovery_suggestions: Box::new(recovery_suggestions),
        }
    }

    /// Create a real-time error with performance context
    pub fn realtime_with_performance(
        message: String,
        performance_context: PerformanceErrorInfo,
    ) -> Self {
        let mut recovery_suggestions = vec!["Reduce processing load".to_string()];

        if let Some(latency) = performance_context.current_latency_ms {
            if latency > 100.0 {
                recovery_suggestions
                    .push("Optimize processing pipeline for lower latency".to_string());
            }
        }

        if let Some(cpu) = performance_context.cpu_usage_percent {
            if cpu > 80.0 {
                recovery_suggestions.push("Reduce CPU-intensive operations".to_string());
            }
        }

        Self::Realtime {
            message,
            performance_context: Some(Box::new(performance_context)),
            context: None,
            recovery_suggestions: Box::new(recovery_suggestions),
        }
    }

    /// Create a memory safety error with detailed information
    pub fn memory_safety_with_info(message: String, safety_info: MemorySafetyErrorInfo) -> Self {
        let mut recovery_suggestions = vec!["Run memory audit".to_string()];

        match safety_info.violation_type.as_str() {
            "memory_leak" => recovery_suggestions.push("Check for unclosed resources".to_string()),
            "buffer_overflow" => recovery_suggestions.push("Validate buffer bounds".to_string()),
            "use_after_free" => recovery_suggestions.push("Check object lifecycle".to_string()),
            _ => recovery_suggestions.push("Review memory usage patterns".to_string()),
        }

        Self::MemorySafety {
            message,
            safety_info: Some(Box::new(safety_info)),
            context: None,
            recovery_suggestions: Box::new(recovery_suggestions),
        }
    }

    /// Create a thread safety error with thread information
    pub fn thread_safety_with_info(message: String, thread_info: ThreadSafetyErrorInfo) -> Self {
        let mut recovery_suggestions = vec!["Review thread synchronization".to_string()];

        match thread_info.violation_type.as_str() {
            "deadlock" => recovery_suggestions.push("Check lock ordering".to_string()),
            "race_condition" => recovery_suggestions.push("Add proper synchronization".to_string()),
            "data_race" => recovery_suggestions.push("Use atomic operations or locks".to_string()),
            _ => recovery_suggestions.push("Review concurrent access patterns".to_string()),
        }

        Self::ThreadSafety {
            message,
            thread_info: Some(Box::new(thread_info)),
            context: None,
            recovery_suggestions: Box::new(recovery_suggestions),
        }
    }

    /// Get recovery suggestions for this error
    pub fn recovery_suggestions(&self) -> &[String] {
        match self {
            Error::Config {
                recovery_suggestions,
                ..
            } => recovery_suggestions,
            Error::Processing {
                recovery_suggestions,
                ..
            } => recovery_suggestions,
            Error::Model {
                recovery_suggestions,
                ..
            } => recovery_suggestions,
            Error::Audio {
                recovery_suggestions,
                ..
            } => recovery_suggestions,
            Error::Realtime {
                recovery_suggestions,
                ..
            } => recovery_suggestions,
            Error::Streaming {
                recovery_suggestions,
                ..
            } => recovery_suggestions,
            Error::Buffer {
                recovery_suggestions,
                ..
            } => recovery_suggestions,
            Error::Transform {
                recovery_suggestions,
                ..
            } => recovery_suggestions,
            Error::Validation {
                recovery_suggestions,
                ..
            } => recovery_suggestions,
            Error::Runtime {
                recovery_suggestions,
                ..
            } => recovery_suggestions,
            Error::MemorySafety {
                recovery_suggestions,
                ..
            } => recovery_suggestions,
            Error::ThreadSafety {
                recovery_suggestions,
                ..
            } => recovery_suggestions,
            Error::ResourceExhaustion {
                recovery_suggestions,
                ..
            } => recovery_suggestions,
            Error::Timeout {
                recovery_suggestions,
                ..
            } => recovery_suggestions,
            Error::Compatibility {
                recovery_suggestions,
                ..
            } => recovery_suggestions,
            Error::Io(_) => &[],
            Error::Serialization(_) => &[],
            Error::Candle(_) => &[],
        }
    }

    /// Get error context if available
    pub fn context(&self) -> Option<&ErrorContext> {
        match self {
            Error::Config { context, .. } => context.as_ref().map(|c| &**c),
            Error::Processing { context, .. } => context.as_ref().map(|c| &**c),
            Error::Model { context, .. } => context.as_ref().map(|c| &**c),
            Error::Audio { context, .. } => context.as_ref().map(|c| &**c),
            Error::Realtime { context, .. } => context.as_ref().map(|c| &**c),
            Error::Streaming { context, .. } => context.as_ref().map(|c| &**c),
            Error::Buffer { context, .. } => context.as_ref().map(|c| &**c),
            Error::Transform { context, .. } => context.as_ref().map(|c| &**c),
            Error::Validation { context, .. } => context.as_ref().map(|c| &**c),
            Error::Runtime { context, .. } => context.as_ref().map(|c| &**c),
            Error::MemorySafety { context, .. } => context.as_ref().map(|c| &**c),
            Error::ThreadSafety { context, .. } => context.as_ref().map(|c| &**c),
            Error::ResourceExhaustion { context, .. } => context.as_ref().map(|c| &**c),
            Error::Timeout { context, .. } => context.as_ref().map(|c| &**c),
            Error::Compatibility { context, .. } => context.as_ref().map(|c| &**c),
            Error::Io(_) => None,
            Error::Serialization(_) => None,
            Error::Candle(_) => None,
        }
    }

    /// Get error severity level
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            Error::Config { .. } => ErrorSeverity::Medium,
            Error::Processing { .. } => ErrorSeverity::Medium,
            Error::Model { .. } => ErrorSeverity::High,
            Error::Audio { .. } => ErrorSeverity::Medium,
            Error::Realtime { .. } => ErrorSeverity::High,
            Error::Streaming { .. } => ErrorSeverity::Medium,
            Error::Buffer { .. } => ErrorSeverity::Medium,
            Error::Transform { .. } => ErrorSeverity::Medium,
            Error::Validation { .. } => ErrorSeverity::Low,
            Error::Runtime { .. } => ErrorSeverity::High,
            Error::MemorySafety { .. } => ErrorSeverity::Critical,
            Error::ThreadSafety { .. } => ErrorSeverity::Critical,
            Error::ResourceExhaustion { .. } => ErrorSeverity::High,
            Error::Timeout { .. } => ErrorSeverity::Medium,
            Error::Compatibility { .. } => ErrorSeverity::Low,
            Error::Io(_) => ErrorSeverity::Medium,
            Error::Serialization(_) => ErrorSeverity::Low,
            Error::Candle(_) => ErrorSeverity::High,
        }
    }
}

/// Error severity levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    /// Low severity error - minimal impact
    Low,
    /// Medium severity error - moderate impact
    Medium,
    /// High severity error - significant impact
    High,
    /// Critical severity error - severe impact, requires immediate attention
    Critical,
}

impl ErrorContext {
    /// Create new error context
    pub fn new(operation: String, location: String) -> Self {
        Self {
            operation,
            location,
            thread_id: Some(format!("{:?}", std::thread::current().id())),
            timestamp: std::time::SystemTime::now(),
            additional_info: std::collections::HashMap::new(),
        }
    }

    /// Add additional information to the context
    pub fn with_info(mut self, key: String, value: String) -> Self {
        self.additional_info.insert(key, value);
        self
    }
}

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::{
        audio_libraries_update::{
            AudioLibrariesUpdater, AudioLibraryInfo, CompatibilityRisk, CompatibilityTestResult,
            LibraryVersionAnalysis, MigrationEffort, MigrationGuide, PerformanceImpact,
            SecuritySeverity, SecurityVulnerability, UpdatePriority, UpdateResult,
        },
        audio_quality_research::{
            AnalysisStatistics, AudioQualityResearcher, ComprehensiveQualityAnalysis,
            HarmonicDistortionAnalysis, MultidimensionalQuality, NeuralQualityModel,
            PsychoacousticAnalysis, ResearchConfig, ResearchCriticalBandAnalysis,
            SpectralQualityAnalysis, TemporalQualityAnalysis, TonalityAnalysis,
        },
        cache::{CacheConfig, CacheItemType, CachePolicy, CachePriority, ConversionCacheSystem},
        cloud_scaling::{
            CloudNode, CloudScalingConfig, CloudScalingController, ClusterMetrics,
            DistributedConversionRequest, DistributedConversionResult, LoadBalancingStrategy,
            NodeCapabilities, NodeResourceUsage, NodeStatus, RequestPriority, RetryConfig,
            ScalingAction as CloudScalingAction, ScalingDecision,
        },
        compression_research::{
            CompressedAudio, CompressionAlgorithm, CompressionConfig, CompressionParameters,
            CompressionResearcher, CompressionStats, CompressionTarget, PredictionAnalyzer,
            PsychoacousticAnalyzer, TonalityDetector, VectorQuantizer,
        },
        config::{ConversionConfig, ConversionConfigBuilder},
        core::{VoiceConverter, VoiceConverterBuilder},
        diagnostics::{
            DiagnosticAnalysis, DiagnosticSystem, HealthAssessment, IdentifiedIssue, IssueCategory,
            IssueSeverity, Recommendation, ReportType,
        },
        fallback::{
            DegradationConfig, FailureType, FallbackContext, GracefulDegradationController,
            QualityThresholds,
        },
        gaming::{
            BevyIntegration, CustomIntegration, GameAudioConfig, GameEngine, GameEngineIntegration,
            GamePerformanceConstraints, GamePerformanceMetrics, GamePerformanceMonitor,
            GameVoiceMode, GameVoiceProcessor, GameVoiceSession, GodotIntegration, ThreadPriority,
            UnityIntegration, UnrealIntegration,
        },
        ml_frameworks::{
            ActivationFunction, ConvLayerConfig, DevicePreference, InferenceMetrics, LayerSpec,
            MLFramework, MLFrameworkConfig, MLFrameworkManager, MLInferenceSession,
            MLModelMetadata, MemoryConfig, MemoryUsageStats, ModelArchitecture, ModelCapabilities,
            ModelOptimization, PerformanceConfig, QuantizationPrecision, RnnType, TensorDataType,
            TensorSpec,
        },
        mobile::{
            MobileConversionConfig, MobileConversionStatistics, MobileDeviceInfo, MobilePlatform,
            MobileVoiceConverter, NeonOptimizer, PowerMode, ThermalState,
        },
        models::{ConversionModel, ModelType},
        monitoring::{
            AlertSeverity, AlertType, MonitorConfig, QualityDashboard, QualityEvent,
            QualityMonitor, SessionDashboard, SystemOverview,
        },
        multi_target::{
            MultiTargetConversionRequest, MultiTargetConversionResult, MultiTargetConverter,
            MultiTargetProcessingStats, NamedTarget, ProcessingMode,
        },
        neural_vocoding::{
            ActivationType, AlgorithmBenchmark, AlgorithmPerformance, AttentionConfig,
            AudioProcessingParams, NeuralArchitectureConfig, NeuralVocoder, NeuralVocodingConfig,
            NeuralVocodingMetrics, VocodingAlgorithm, VocodingQuality,
        },
        optimizations::{AudioBufferPool, ConversionPerformanceMonitor, SmallAudioOptimizer},
        pipeline_optimization::{
            AlgorithmVariant, OptimizationStatistics, OptimizedConversionPlan, OptimizedPipeline,
            OptimizedPipelineConfig,
        },
        platform_libraries::{
            CpuFeatures, OptimizationLevel, PlatformConfig, PlatformOptimizer, PlatformStats,
            TargetPlatform,
        },
        processing::{AudioBuffer, ProcessingPipeline},
        profiling::{
            BottleneckAnalyzer, BottleneckInfo, BottleneckType, ConversionProfiler,
            ProfilingConfig, ProfilingReport, ProfilingSession,
        },
        quality::{
            AdaptiveQualityController, ArtifactDetector, CriticalBandAnalysis, DetectedArtifacts,
            LoudnessAnalysis, MaskingAnalysis, ObjectiveQualityMetrics,
            PerceptualOptimizationParams, PerceptualOptimizationResult, PerceptualOptimizer,
            QualityAssessment, QualityMetricsSystem,
        },
        realtime::{RealtimeConfig, RealtimeConverter},
        realtime_libraries::{
            AudioBackend, BackendCapabilities, RealtimeBuffer,
            RealtimeConfig as RealtimeLibraryConfig, RealtimeLibraryManager, RealtimeStats,
        },
        realtime_ml::{
            AdaptiveOptimizationState, BufferStrategy, CacheEvictionPolicy,
            CacheOptimizationConfig, ModelAdaptationConfig, OptimizationSnapshot,
            OptimizationStrategy, ParallelProcessingConfig, PerformanceSample, QuantizationLevel,
            RealtimeMLConfig, RealtimeMLOptimizer, RealtimeMetrics,
            ResourceUsage as RealtimeMLResourceUsage, StreamingOptimizationConfig,
        },
        recognition::{
            ASRConfig, ASREngine, ASRTranscription, PhonemeAlignment, RecognitionGuidedConverter,
            RecognitionGuidedResult, RecognitionStats, SpeechGuidedParams, WordTimestamp,
        },
        scalability::{
            MemoryEfficiencyMetrics, MemoryTracker, ResourceAllocationStrategy, ResourceMonitor,
            ResourceUsageMetrics, ScalabilityConfig, ScalabilityMetrics, ScalabilityTargets,
            ScalableConverter, ScalingAction, ScalingActionType, ScalingController,
            ScalingThresholds, ThroughputMetrics, ThroughputSample,
        },
        streaming::{StreamProcessor, StreamingConverter},
        streaming_platforms::{
            AdaptationDirection, AdaptationEvent, BandwidthAdaptationState, DiscordIntegration,
            FacebookIntegration, OBSIntegration, PlatformIntegration, RTMPIntegration,
            StreamConfig, StreamPerformanceMetrics, StreamPerformanceMonitor,
            StreamProcessor as StreamPlatformProcessor, StreamQuality, StreamSession,
            StreamVoiceMode, StreamingConstraints, StreamingPlatform, StreamlabsIntegration,
            TikTokIntegration, TwitchIntegration, XSplitIntegration, YouTubeIntegration,
        },
        style_consistency::{
            ConsistencyThresholds, PreservationMode, StyleAdaptationSettings,
            StyleConsistencyConfig, StyleConsistencyEngine, StyleElement,
        },
        style_transfer::{
            StyleCharacteristics, StyleTransferConfig, StyleTransferMethod, StyleTransferSystem,
        },
        thread_safety::{
            ConcurrentConversionManager, ConcurrentConversionMetrics, ThreadSafeModelManager,
        },
        transforms::{
            AgeTransform, GenderTransform, PitchTransform, SpeedTransform, Transform, VoiceMorpher,
        },
        types::{
            ConversionRequest, ConversionResult, ConversionTarget, ConversionType,
            VoiceCharacteristics,
        },
        webrtc_integration::{
            ConversionMode, NetworkConditions, QualityMode, VoiceConversionConfig,
            WebRTCAudioConfig, WebRTCProcessingStatistics, WebRTCVoiceProcessor,
        },
        zero_shot::{
            ReferenceVoiceDatabase, SpeakerEmbedding, UniversalVoiceModel, ZeroShotConfig,
            ZeroShotConverter,
        },
        Error, Result,
    };

    #[cfg(feature = "acoustic-integration")]
    pub use crate::acoustic::{
        AcousticConversionAdapter, AcousticConversionContext, AcousticConversionResult,
        AcousticFeatureConfig, AcousticFeatures, AcousticState, FormantFrequencies,
        HarmonicFeatures, TemporalFeatures, WindowType,
    };

    #[cfg(feature = "cloning-integration")]
    pub use crate::cloning::{
        CloningConversionAdapter, CloningConversionResult, CloningIntegration,
        CloningIntegrationConfig, TargetSpeakerInfo,
    };

    #[cfg(feature = "emotion-integration")]
    pub use crate::emotion::{EmotionConversionAdapter, EmotionParameters};

    #[cfg(feature = "spatial-integration")]
    pub use crate::spatial::{
        AmbisonicsOutput, BinauralAudioOutput, HrtfMetadata, SpatialAudioOutput,
        SpatialConversionAdapter, SpatialDirection, SpatialPosition, SpatialVoiceSource,
    };

    #[cfg(feature = "iot")]
    pub use crate::iot::{
        IoTConversionConfig, IoTConversionStatistics, IoTDeviceStatus, IoTPlatform, IoTPowerMode,
        IoTProcessingMode, IoTVoiceConverter, ResourceConstraints,
        ResourceUsage as IoTResourceUsage,
    };

    #[cfg(feature = "wasm")]
    pub use crate::wasm::{
        BrowserCapabilities, ConversionParameters, WasmConversionConfig, WasmConversionStatistics,
        WasmSupportLevel, WasmVoiceConverter, WebAudioNodeType,
    };
}

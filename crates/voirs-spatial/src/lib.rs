//! # VoiRS Spatial Audio System
//!
//! This crate provides 3D spatial audio processing capabilities including
//! HRTF (Head-Related Transfer Function) processing, binaural audio rendering,
//! 3D position tracking, room acoustics simulation, and AR/VR integration.

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod ambisonics;
pub mod automotive;
pub mod beamforming;
pub mod binaural;
pub mod compression;
pub mod config;
pub mod core;
pub mod gaming;
pub mod gestures;
pub mod gpu;
pub mod haptic;
pub mod hrtf;
pub mod memory;
pub mod mobile;
pub mod multiuser;
pub mod neural;
pub mod performance;
pub mod performance_targets;
pub mod platforms;
pub mod plugins;
pub mod position;
pub mod power;
pub mod public_spaces;
pub mod room;
pub mod smart_speakers;
pub mod technical_testing;
pub mod telepresence;
pub mod types;
pub mod utils;
pub mod validation;
pub mod visual_audio;
pub mod webxr;
pub mod wfs;

// Re-export main types and traits
pub use ambisonics::{
    channel_count, AmbisonicsDecoder, AmbisonicsEncoder, AmbisonicsOrder, AmbisonicsProcessor,
    ChannelOrdering, NormalizationScheme, SpeakerConfiguration, SphericalCoordinate,
    SphericalHarmonics,
};
pub use automotive::{
    AcousticTreatment, AdaptiveVolumeConfig, AudioZone, DashboardSide, DriverProtectionConfig,
    EmergencyAlertType, EmergencyAudioConfig, EngineNoiseCompensation, EngineType,
    ExternalAwarenessConfig, ExternalConditions, FrequencyBand, FrequencyWeighting, GearPosition,
    HearingCharacteristics, HvacConfig, HvacState, InteriorMaterials, LegalComplianceConfig,
    MaterialType, NoiseCompensationConfig, PassengerActivity, PassengerConfig, PassengerInfo,
    PassengerPreferences, PrecipitationType, ResourceUsage, RoadNoiseCompensation, SafetyConfig,
    SeatPosition, SpatialPreferences, SpeakerCharacteristics, SpeakerMounting, SpeakerType,
    SurfaceType, TimeLimits, TireNoiseModel, TireType, VehicleAcousticConfig, VehicleAudioConfig,
    VehicleAudioConfigBuilder, VehicleAudioMetrics, VehicleAudioProcessor, VehicleSide,
    VehicleSpeaker, VehicleSpeakerConfig, VehicleState, VehicleType, VentPosition,
    WindNoiseCompensation, WindowConfig, WindowState, WindowTinting, ZoneAudioSource,
};
pub use beamforming::{
    AdaptationConfig, BeamPattern, BeamformerWeights, BeamformingAlgorithm, BeamformingConfig,
    BeamformingProcessor, DoaResult, SpatialSmoothingConfig,
};
pub use binaural::{BinauralConfig, BinauralMetrics, BinauralRenderer, SourceType};
pub use compression::{
    AdaptiveParams, CompressedFrame, CompressionCodec, CompressionQuality, CompressionStats,
    PerceptualParams, SourceClustering, SpatialCompressionConfig, SpatialCompressor,
    SpatialMetadata, SpatialParams, TemporalMasking,
};
pub use config::{SpatialConfig, SpatialConfigBuilder};
pub use core::{SpatialProcessor, SpatialProcessorBuilder};
pub use gaming::{
    console, AttenuationCurve, AttenuationSettings, AudioCategory, GameAudioSource, GameEngine,
    GamingAudioManager, GamingConfig, GamingMetrics,
};
pub use gestures::{
    AudioAction, GestureBuilder, GestureConfidence, GestureConfig, GestureController, GestureData,
    GestureDirection, GestureEvent, GestureEventType, GestureRecognitionMethod, GestureType, Hand,
};
pub use gpu::{
    GpuAmbisonics, GpuConfig, GpuConvolution, GpuDevice, GpuResourceManager, GpuSpatialMath,
};
pub use haptic::{
    AudioHapticMapping, DistanceAttenuation, HapticAccessibilitySettings, HapticAudioConfig,
    HapticAudioProcessor, HapticCapabilities, HapticComfortSettings, HapticDevice,
    HapticEffectType, HapticElement, HapticMetrics, HapticPattern, PatternStyle,
};
pub use hrtf::{
    ai_personalization::{
        AdaptationStrategy, AiHrtfPersonalizer, AnthropometricMeasurements, HrtfModifications,
        PerceptualFeedback, PersonalizationConfig, PersonalizationMetadata, PersonalizedHrtf,
        TrainingResults,
    },
    HrtfDatabase, HrtfProcessor,
};
pub use memory::{
    cache_optimization, Array2Pool, BufferPool, CacheManager, CachePolicy, MemoryConfig,
    MemoryManager, MemoryStatistics,
};
pub use mobile::{
    android, ios, MobileConfig, MobileDevice, MobileMetrics, MobileOptimizer, MobilePlatform,
    PowerState, QualityPreset,
};
pub use multiuser::{
    AccessibilitySettings, AcousticSettings, AudioEffect, AudioEffectsProcessor,
    AudioQualityMetrics, AudioSourceType, BandwidthSettings, CodecState, CodecTiming,
    CompensationMethod, ConnectionStatus, DirectionalPattern, DisconnectReason,
    InterpolationMethod, LatencyCompensator, LowBandwidthMode, MicrophoneSettings, MixerConfig,
    MultiUserAttenuationCurve, MultiUserAudioProcessor, MultiUserAudioSource, MultiUserConfig,
    MultiUserConfigBuilder, MultiUserEnvironment, MultiUserEvent, MultiUserMetrics, MultiUserUser,
    NetworkEventType, NetworkStats, OptimizationLevel, Permission, PermissionSystem,
    PositionInterpolator, PositionSnapshot, RoomId, SourceAccessControl, SourceId,
    SourceProcessingState, SourceQualitySettings, SourceVisibility, SpatialAudioMixer,
    SpatialProperties, SpatialZone, SynchronizationManager, SynchronizedClock, UserId, UserRole,
    VadAlgorithm, VadState, VadThresholds, VoiceActivityDetector, ZoneAudioProperties, ZoneBounds,
    ZoneType,
};
pub use neural::{
    AdaptiveQualityController, AugmentationConfig, ConvolutionalModel, FeedforwardModel,
    LossFunction, NeuralInputFeatures, NeuralModel, NeuralModelType, NeuralPerformanceMetrics,
    NeuralSpatialConfig, NeuralSpatialConfigBuilder, NeuralSpatialOutput, NeuralSpatialProcessor,
    NeuralTrainer, NeuralTrainingResults, OptimizerType, RealtimeConstraints, TrainingConfig,
    TransformerModel,
};
pub use performance::{
    PerformanceConfig, PerformanceMetrics, PerformanceReport, PerformanceSummary,
    PerformanceTargetResult, PerformanceTestSuite, ResourceMonitor, ResourceStatistics,
};
pub use performance_targets::{
    LatencyMeasurements, PerformanceTargetReport, PerformanceTargetValidator, PerformanceTargets,
    PerformanceValidationResult, QualityMeasurements, QualityTargets, RealtimeTargets,
    ResourceMeasurements, ResourceTargets, ResourceUsageStats, ScalabilityMeasurements,
    ScalabilityTargets, TargetCategory, TargetComparison,
};
pub use platforms::{
    ARCorePlatform, ARKitPlatform, DeviceInfo, EyeData, EyeTrackingData, GenericPlatform, HandData,
    HandGesture, HandTrackingData, OculusPlatform, PlatformCapabilities, PlatformFactory,
    PlatformIntegration, PlatformTrackingData, PoseData, TrackingConfig, TrackingQuality,
    TrackingState, WMRPlatform,
};

#[cfg(feature = "steamvr")]
pub use platforms::SteamVRPlatform;
pub use plugins::{
    PluginCapabilities, PluginConfig, PluginManager, PluginParameter, PluginParameters,
    PluginState, ProcessingChain, ProcessingContext, ReverbPlugin, SpatialPlugin,
};
pub use position::{
    advanced_prediction::{
        AdaptationPhase, AdvancedPredictiveTracker, ModelSelectionStrategy, MotionPattern,
        MotionPatternType, PatternRecognitionConfig, PredictedPosition, PredictionMetrics,
        PredictionModelType, PredictiveTrackingConfig,
    },
    Box3D, CalibrationData, ComfortSettings, DopplerProcessor, DynamicSource, DynamicSourceManager,
    HeadTracker, Listener, ListenerMovementSystem, MotionPredictor, MotionSnapshot,
    MovementConstraints, MovementMetrics, NavigationMode, OcclusionDetector, OcclusionMaterial,
    OcclusionMethod, OcclusionResult, PlatformData, PlatformType, SoundSource,
    SpatialSourceManager,
};
pub use power::{
    DeviceType, PowerConfig, PowerMetrics, PowerOptimizer, PowerProfile, PowerStrategy,
};
pub use room::{
    adaptive_acoustics::{
        AdaptationAction, AdaptationController, AdaptationMetrics, AdaptationTrigger,
        AdaptiveAcousticEnvironment, AdaptiveAcousticsConfig, EnvironmentSensors,
        EnvironmentSnapshot, EnvironmentType, SensorConfig, UserFeedback,
    },
    ConnectionAcousticProperties, ConnectionState, ConnectionType, GlobalAcousticConfig,
    MultiRoomEnvironment, Room, RoomAcoustics, RoomConnection, RoomSimulator,
};
pub use smart_speakers::{
    ArrayMetrics, ArrayTopology, AudioFormat, AudioRoute, AudioRouter, AudioSource, AudioSpecs,
    CalibrationEngine, CalibrationMethod, CalibrationResults, CalibrationStatus, ClockSource,
    CompressionConfig, DeviceFilter, DirectivityPattern, DiscoveryProtocol, DiscoveryService,
    DspFeature, EQFilter, FilterType, LimitingConfig, MixSettings, NetworkConfig, NetworkInfo,
    NetworkProtocol, ProcessingConfig, ProcessingStep, RoomCorrection, SmartSpeaker,
    SpeakerArrayConfig, SpeakerArrayConfigBuilder, SpeakerArrayManager, SpeakerCapabilities,
    SyncConfig,
};
pub use technical_testing::{
    create_standard_technical_configs, LatencyAnalysis, MemoryConstraints, PlatformAnalysis,
    PlatformTestResult, StabilityAnalysis, StressTestParams, TechnicalSuccessCriteria,
    TechnicalTestConfig, TechnicalTestParameters, TechnicalTestReport, TechnicalTestResult,
    TechnicalTestSuite, TechnicalTestType, TestOutcome,
};
pub use telepresence::{
    AcousticEchoSettings, AcousticMatchingSettings, AcousticProperties, AdaptiveQualitySettings,
    AirAbsorptionSettings, AnonymizationMethod, AnonymizationSettings, AudioCodec,
    AudioDeviceConfig, AudioEnhancementSettings, AudioFormat as TelepresenceAudioFormat,
    AudioMetadata, AudioPresenceSettings, AudioQualityPreferences, AudioQualitySettings,
    BandwidthConstraints, BandwidthExtensionSettings, CodecPreferences, CompressionSettings,
    ConsentManagementSettings, CrossRoomSettings, DataCollectionSettings, DistanceModelingSettings,
    DopplerEffectsSettings, EchoCancellationSettings, EnvironmentalAwarenessSettings,
    EqualizationSettings, HeadTrackingSettings, HrtfPersonalizationSettings, NetworkSettings,
    NoiseSuppressionSettings, Orientation, PresenceIndicatorSettings, PrivacySettings,
    QualityLevel, QualitySettings, ReceivedAudio, RoomSimulationSettings, SessionJoinResult,
    SessionState, SessionStatistics, SpatialTelepresenceSettings, TelepresenceAudioSettings,
    TelepresenceConfig, TelepresenceProcessor, TelepresenceSession, TrackingPredictionSettings,
    UserConfig, VadSettings, Velocity, VirtualRoomParameters, VisualPresenceSettings,
    VoiceProcessingSettings, VoiceSpatializationSettings,
};
pub use types::{
    AudioChannel, BinauraAudio, Position3D, SIMDSpatialOps, SpatialEffect, SpatialRequest,
    SpatialResult,
};
pub use validation::{
    create_standard_test_configs, create_test_subjects, AccuracyMetrics, AudioExpertise,
    ExperienceLevel, Gender, HearingAbility, PerceptualTestSuite, PopulationAnalysis, ResponseData,
    StimulusData, SubjectiveRatings, SuccessCriteria, TestParameters, TestStatistics, TestSubject,
    ValidationReport, ValidationTestConfig, ValidationTestResult, ValidationTestType,
};
pub use visual_audio::{
    AnimationParams, AnimationType, AudioVisualMapping, ColorRGBA, ColorScheme, ColorSchemeType,
    DirectionZone, DirectionalCueMapping, EasingFunction, EventTriggerMapping,
    FrequencyVisualMapping, OnsetTrigger, RhythmTrigger, ScalingCurve, ShapeType, SilenceTrigger,
    SpectralTrigger, VisualAccessibilitySettings, VisualAudioConfig, VisualAudioMetrics,
    VisualAudioProcessor, VisualDisplay, VisualDisplayCapabilities, VisualDistanceAttenuation,
    VisualEffect, VisualElement, VisualElementType, VisualPerformanceSettings, VisualResourceUsage,
    VisualSyncSettings,
};
pub use webxr::{
    utils as webxr_utils, BrowserType, WebXRCapabilities, WebXRConfig, WebXRMetrics, WebXRPose,
    WebXRProcessor, WebXRSessionType, WebXRSourceType,
};
pub use wfs::{
    ArrayGeometry, PreEmphasisConfig, WfsArrayBuilder, WfsConfig, WfsDrivingFunction, WfsProcessor,
    WfsSource, WfsSourceType,
};

/// Result type for spatial audio operations
pub type Result<T> = std::result::Result<T, Error>;

/// Error types for spatial audio processing
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Configuration error (structured)
    #[error("Configuration error: {0}")]
    Config(#[from] ConfigError),

    /// Processing error (structured)
    #[error("Processing error: {0}")]
    Processing(#[from] ProcessingError),

    /// HRTF error (structured)
    #[error("HRTF error: {0}")]
    Hrtf(#[from] HrtfError),

    /// Position error (structured)
    #[error("Position error: {0}")]
    Position(#[from] PositionError),

    /// Room acoustics error (structured)
    #[error("Room acoustics error: {0}")]
    Room(#[from] RoomError),

    /// Audio error (structured)
    #[error("Audio error: {0}")]
    Audio(#[from] AudioError),

    /// Memory error (structured)
    #[error("Memory error: {0}")]
    Memory(#[from] MemoryError),

    /// GPU error (structured)
    #[error("GPU error: {0}")]
    Gpu(#[from] GpuError),

    /// Platform error (structured)
    #[error("Platform error: {0}")]
    Platform(#[from] PlatformError),

    /// Validation error (structured)
    #[error("Validation error: {0}")]
    Validation(#[from] ValidationError),

    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Candle error
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),

    /// Generic error with context
    #[error("Error: {message}")]
    Generic {
        /// Error message
        message: String,
        /// Error code
        code: ErrorCode,
        /// Error context
        context: Option<Box<ErrorContext>>,
    },

    /// Legacy errors for backward compatibility (will be migrated)
    #[error("Configuration error: {0}")]
    LegacyConfig(String),

    /// Legacy processing error for backward compatibility
    #[error("Processing error: {0}")]
    LegacyProcessing(String),

    /// Legacy HRTF error for backward compatibility
    #[error("HRTF error: {0}")]
    LegacyHrtf(String),

    /// Legacy position error for backward compatibility
    #[error("Position error: {0}")]
    LegacyPosition(String),

    /// Legacy room acoustics error for backward compatibility
    #[error("Room acoustics error: {0}")]
    LegacyRoom(String),

    /// Legacy audio error for backward compatibility
    #[error("Audio error: {0}")]
    LegacyAudio(String),

    /// Legacy validation error for backward compatibility
    #[error("Validation error: {0}")]
    LegacyValidation(String),
}

/// Configuration-specific errors
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// Invalid parameter value
    #[error("Invalid parameter '{parameter}': {message}. Expected: {expected}")]
    InvalidParameter {
        /// Parameter name
        parameter: String,
        /// Error message
        message: String,
        /// Expected value or format
        expected: String,
    },

    /// Missing required parameter
    #[error("Missing required parameter: {parameter}")]
    MissingParameter {
        /// Parameter name
        parameter: String,
    },

    /// Conflicting parameters
    #[error("Conflicting parameters: {params:?}. {resolution}")]
    ConflictingParameters {
        /// Conflicting parameter names
        params: Vec<String>,
        /// Resolution suggestion
        resolution: String,
    },

    /// File not found
    #[error("Configuration file not found: {path}")]
    FileNotFound {
        /// File path
        path: String,
    },
}

/// Processing-specific errors
#[derive(Debug, thiserror::Error)]
pub enum ProcessingError {
    /// Buffer size mismatch
    #[error("Buffer size mismatch: expected {expected}, got {actual}")]
    BufferSizeMismatch {
        /// Expected size
        expected: usize,
        /// Actual size
        actual: usize,
    },

    /// Sample rate mismatch
    #[error("Sample rate mismatch: expected {expected}Hz, got {actual}Hz")]
    SampleRateMismatch {
        /// Expected sample rate
        expected: u32,
        /// Actual sample rate
        actual: u32,
    },

    /// Real-time constraint violation
    #[error("Real-time processing constraint violated: {duration_ms}ms > {max_ms}ms")]
    RealtimeViolation {
        /// Processing duration
        duration_ms: f64,
        /// Maximum allowed duration
        max_ms: f64,
    },

    /// Resource exhaustion
    #[error("Resource exhausted: {resource}. Usage: {usage}/{limit}")]
    ResourceExhausted {
        /// Resource type
        resource: String,
        /// Current usage
        usage: u64,
        /// Resource limit
        limit: u64,
    },
}

/// HRTF-specific errors
#[derive(Debug, thiserror::Error)]
pub enum HrtfError {
    /// Database loading failed
    #[error("Failed to load HRTF database from {path}: {reason}")]
    DatabaseLoadFailed {
        /// Database path
        path: String,
        /// Failure reason
        reason: String,
    },

    /// Invalid position for HRTF lookup
    #[error("Invalid position for HRTF lookup: azimuth={azimuth}°, elevation={elevation}°")]
    InvalidPosition {
        /// Azimuth in degrees
        azimuth: f32,
        /// Elevation in degrees
        elevation: f32,
    },

    /// Interpolation failed
    #[error("HRTF interpolation failed: {reason}")]
    InterpolationFailed {
        /// Failure reason
        reason: String,
    },
}

/// Position-specific errors
#[derive(Debug, thiserror::Error)]
pub enum PositionError {
    /// Invalid coordinates
    #[error("Invalid coordinates: x={x}, y={y}, z={z}. {constraint}")]
    InvalidCoordinates {
        /// X coordinate
        x: f32,
        /// Y coordinate
        y: f32,
        /// Z coordinate
        z: f32,
        /// Constraint description
        constraint: String,
    },

    /// Tracking system unavailable
    #[error("Tracking system unavailable: {system}")]
    TrackingUnavailable {
        /// Tracking system name
        system: String,
    },

    /// Calibration required
    #[error("Calibration required for {component}: {instructions}")]
    CalibrationRequired {
        /// Component requiring calibration
        component: String,
        /// Calibration instructions
        instructions: String,
    },
}

/// Room acoustics errors
#[derive(Debug, thiserror::Error)]
pub enum RoomError {
    /// Invalid room dimensions
    #[error("Invalid room dimensions: {width}x{height}x{depth}m. {constraint}")]
    InvalidDimensions {
        /// Room width
        width: f32,
        /// Room height
        height: f32,
        /// Room depth
        depth: f32,
        /// Constraint description
        constraint: String,
    },

    /// Material properties error
    #[error("Invalid material properties: {material}. {issue}")]
    InvalidMaterial {
        /// Material name
        material: String,
        /// Issue description
        issue: String,
    },

    /// Ray tracing failed
    #[error("Ray tracing computation failed: {reason}")]
    RayTracingFailed {
        /// Failure reason
        reason: String,
    },
}

/// Audio-specific errors
#[derive(Debug, thiserror::Error)]
pub enum AudioError {
    /// Unsupported format
    #[error("Unsupported audio format: {format}. Supported: {supported:?}")]
    UnsupportedFormat {
        /// Current format
        format: String,
        /// List of supported formats
        supported: Vec<String>,
    },

    /// Clipping detected
    #[error("Audio clipping detected: {severity}% samples above threshold")]
    ClippingDetected {
        /// Clipping severity as percentage
        severity: f32,
    },

    /// Underrun or overrun
    #[error("Audio buffer {kind}: {samples} samples")]
    BufferIssue {
        /// Type of buffer issue
        kind: AudioBufferIssue,
        /// Number of affected samples
        samples: usize,
    },
}

/// Memory-specific errors
#[derive(Debug, thiserror::Error)]
pub enum MemoryError {
    /// Out of memory
    #[error("Out of memory: requested {requested}MB, available {available}MB")]
    OutOfMemory {
        /// Requested memory in MB
        requested: u64,
        /// Available memory in MB
        available: u64,
    },

    /// Pool exhausted
    #[error("Memory pool exhausted: {pool_type}, size={size}")]
    PoolExhausted {
        /// Pool type
        pool_type: String,
        /// Pool size
        size: usize,
    },

    /// Cache miss
    #[error("Cache miss: {cache_type}, key={key}")]
    CacheMiss {
        /// Cache type
        cache_type: String,
        /// Cache key
        key: String,
    },
}

/// GPU-specific errors
#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    /// GPU not available
    #[error("GPU not available: {reason}")]
    NotAvailable {
        /// Reason for unavailability
        reason: String,
    },

    /// CUDA error
    #[error("CUDA error: {message}")]
    Cuda {
        /// CUDA error message
        message: String,
    },

    /// Memory allocation failed
    #[error("GPU memory allocation failed: requested {requested}MB")]
    AllocationFailed {
        /// Requested memory in MB
        requested: u64,
    },
}

/// Platform-specific errors
#[derive(Debug, thiserror::Error)]
pub enum PlatformError {
    /// Platform not supported
    #[error("Platform not supported: {platform}")]
    NotSupported {
        /// Platform name
        platform: String,
    },

    /// SDK initialization failed
    #[error("SDK initialization failed: {sdk}, reason: {reason}")]
    SdkInitFailed {
        /// SDK name
        sdk: String,
        /// Failure reason
        reason: String,
    },

    /// Feature not available
    #[error("Feature not available: {feature} on {platform}")]
    FeatureUnavailable {
        /// Feature name
        feature: String,
        /// Platform name
        platform: String,
    },
}

/// Validation-specific errors
#[derive(Debug, thiserror::Error)]
pub enum ValidationError {
    /// Schema validation failed
    #[error("Schema validation failed: {field}. {message}")]
    SchemaFailed {
        /// Field that failed validation
        field: String,
        /// Validation message
        message: String,
    },

    /// Range validation failed
    #[error("Value out of range: {field}={value}, expected [{min}, {max}]")]
    RangeError {
        /// Field name
        field: String,
        /// Actual value
        value: f64,
        /// Minimum allowed value
        min: f64,
        /// Maximum allowed value
        max: f64,
    },

    /// Test failed
    #[error("Test failed: {test_name}. Expected: {expected}, Got: {actual}")]
    TestFailed {
        /// Test name
        test_name: String,
        /// Expected result
        expected: String,
        /// Actual result
        actual: String,
    },
}

/// Error codes for programmatic error handling
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCode {
    /// Configuration errors (1000-1999)
    /// Invalid parameter error code
    ConfigInvalidParameter = 1001,
    /// Missing parameter error code
    ConfigMissingParameter = 1002,
    /// Conflicting parameters error code
    ConfigConflictingParameters = 1003,
    /// Configuration file not found error code
    ConfigFileNotFound = 1004,

    /// Processing errors (2000-2999)
    /// Buffer size mismatch error code
    ProcessingBufferSizeMismatch = 2001,
    /// Sample rate mismatch error code
    ProcessingSampleRateMismatch = 2002,
    /// Real-time constraint violation error code
    ProcessingRealtimeViolation = 2003,
    /// Resource exhaustion error code
    ProcessingResourceExhausted = 2004,

    /// HRTF errors (3000-3999)
    /// HRTF database load failed error code
    HrtfDatabaseLoadFailed = 3001,
    /// Invalid HRTF position error code
    HrtfInvalidPosition = 3002,
    /// HRTF interpolation failed error code
    HrtfInterpolationFailed = 3003,

    /// Position errors (4000-4999)
    /// Invalid coordinates error code
    PositionInvalidCoordinates = 4001,
    /// Tracking system unavailable error code
    PositionTrackingUnavailable = 4002,
    /// Calibration required error code
    PositionCalibrationRequired = 4003,

    /// Room errors (5000-5999)
    /// Invalid room dimensions error code
    RoomInvalidDimensions = 5001,
    /// Invalid material properties error code
    RoomInvalidMaterial = 5002,
    /// Ray tracing computation failed error code
    RoomRayTracingFailed = 5003,

    /// Audio errors (6000-6999)
    /// Unsupported audio format error code
    AudioUnsupportedFormat = 6001,
    /// Audio clipping detected error code
    AudioClippingDetected = 6002,
    /// Audio buffer issue error code
    AudioBufferIssue = 6003,

    /// Memory errors (7000-7999)
    /// Out of memory error code
    MemoryOutOfMemory = 7001,
    /// Memory pool exhausted error code
    MemoryPoolExhausted = 7002,
    /// Cache miss error code
    MemoryCacheMiss = 7003,

    /// GPU errors (8000-8999)
    /// GPU not available error code
    GpuNotAvailable = 8001,
    /// CUDA error code
    GpuCudaError = 8002,
    /// GPU memory allocation failed error code
    GpuAllocationFailed = 8003,

    /// Platform errors (9000-9999)
    /// Platform not supported error code
    PlatformNotSupported = 9001,
    /// Platform SDK initialization failed error code
    PlatformSdkInitFailed = 9002,
    /// Platform feature unavailable error code
    PlatformFeatureUnavailable = 9003,

    /// Generic errors (10000+)
    /// Generic error code
    Generic = 10000,
}

/// Audio buffer issue types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioBufferIssue {
    /// Buffer underrun (not enough data)
    Underrun,
    /// Buffer overrun (too much data)
    Overrun,
    /// Buffer corruption detected
    Corruption,
}

impl std::fmt::Display for AudioBufferIssue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AudioBufferIssue::Underrun => write!(f, "underrun"),
            AudioBufferIssue::Overrun => write!(f, "overrun"),
            AudioBufferIssue::Corruption => write!(f, "corruption"),
        }
    }
}

/// Error context for additional debugging information
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// Module where error occurred
    pub module: String,
    /// Function where error occurred
    pub function: String,
    /// Line number (if available)
    pub line: Option<u32>,
    /// Additional context data
    pub data: std::collections::HashMap<String, String>,
    /// Recovery suggestions
    pub recovery_suggestions: Vec<String>,
    /// Related errors (for error chains)
    pub related_errors: Vec<String>,
}

impl ErrorContext {
    /// Create new error context
    pub fn new(module: &str, function: &str) -> Self {
        Self {
            module: module.to_string(),
            function: function.to_string(),
            line: None,
            data: std::collections::HashMap::new(),
            recovery_suggestions: Vec::new(),
            related_errors: Vec::new(),
        }
    }

    /// Add context data
    pub fn with_data(mut self, key: &str, value: &str) -> Self {
        self.data.insert(key.to_string(), value.to_string());
        self
    }

    /// Add recovery suggestion
    pub fn with_suggestion(mut self, suggestion: &str) -> Self {
        self.recovery_suggestions.push(suggestion.to_string());
        self
    }

    /// Add related error
    pub fn with_related_error(mut self, error: &str) -> Self {
        self.related_errors.push(error.to_string());
        self
    }
}

impl Error {
    /// Create a generic configuration error from a string message
    pub fn config(message: &str) -> Self {
        Self::Config(ConfigError::InvalidParameter {
            parameter: "unknown".to_string(),
            message: message.to_string(),
            expected: "valid configuration".to_string(),
        })
    }

    /// Create a generic processing error from a string message
    pub fn processing(message: &str) -> Self {
        Self::Processing(ProcessingError::ResourceExhausted {
            resource: "processing".to_string(),
            usage: 0,
            limit: 0,
        })
    }

    /// Create a generic HRTF error from a string message
    pub fn hrtf(message: &str) -> Self {
        Self::Hrtf(HrtfError::InterpolationFailed {
            reason: message.to_string(),
        })
    }

    /// Create a generic room error from a string message
    pub fn room(message: &str) -> Self {
        Self::Room(RoomError::RayTracingFailed {
            reason: message.to_string(),
        })
    }

    /// Get error code for programmatic handling
    pub fn code(&self) -> ErrorCode {
        match self {
            Error::Config(config_err) => match config_err {
                ConfigError::InvalidParameter { .. } => ErrorCode::ConfigInvalidParameter,
                ConfigError::MissingParameter { .. } => ErrorCode::ConfigMissingParameter,
                ConfigError::ConflictingParameters { .. } => ErrorCode::ConfigConflictingParameters,
                ConfigError::FileNotFound { .. } => ErrorCode::ConfigFileNotFound,
            },
            // Legacy errors
            Error::LegacyConfig(_) => ErrorCode::ConfigInvalidParameter,
            Error::LegacyProcessing(_) => ErrorCode::ProcessingResourceExhausted,
            Error::LegacyHrtf(_) => ErrorCode::HrtfInterpolationFailed,
            Error::LegacyPosition(_) => ErrorCode::PositionInvalidCoordinates,
            Error::LegacyRoom(_) => ErrorCode::RoomRayTracingFailed,
            Error::LegacyAudio(_) => ErrorCode::AudioBufferIssue,
            Error::LegacyValidation(_) => ErrorCode::Generic,
            Error::Processing(processing_err) => match processing_err {
                ProcessingError::BufferSizeMismatch { .. } => {
                    ErrorCode::ProcessingBufferSizeMismatch
                }
                ProcessingError::SampleRateMismatch { .. } => {
                    ErrorCode::ProcessingSampleRateMismatch
                }
                ProcessingError::RealtimeViolation { .. } => ErrorCode::ProcessingRealtimeViolation,
                ProcessingError::ResourceExhausted { .. } => ErrorCode::ProcessingResourceExhausted,
            },
            Error::Hrtf(hrtf_err) => match hrtf_err {
                HrtfError::DatabaseLoadFailed { .. } => ErrorCode::HrtfDatabaseLoadFailed,
                HrtfError::InvalidPosition { .. } => ErrorCode::HrtfInvalidPosition,
                HrtfError::InterpolationFailed { .. } => ErrorCode::HrtfInterpolationFailed,
            },
            Error::Position(position_err) => match position_err {
                PositionError::InvalidCoordinates { .. } => ErrorCode::PositionInvalidCoordinates,
                PositionError::TrackingUnavailable { .. } => ErrorCode::PositionTrackingUnavailable,
                PositionError::CalibrationRequired { .. } => ErrorCode::PositionCalibrationRequired,
            },
            Error::Room(room_err) => match room_err {
                RoomError::InvalidDimensions { .. } => ErrorCode::RoomInvalidDimensions,
                RoomError::InvalidMaterial { .. } => ErrorCode::RoomInvalidMaterial,
                RoomError::RayTracingFailed { .. } => ErrorCode::RoomRayTracingFailed,
            },
            Error::Audio(audio_err) => match audio_err {
                AudioError::UnsupportedFormat { .. } => ErrorCode::AudioUnsupportedFormat,
                AudioError::ClippingDetected { .. } => ErrorCode::AudioClippingDetected,
                AudioError::BufferIssue { .. } => ErrorCode::AudioBufferIssue,
            },
            Error::Memory(memory_err) => match memory_err {
                MemoryError::OutOfMemory { .. } => ErrorCode::MemoryOutOfMemory,
                MemoryError::PoolExhausted { .. } => ErrorCode::MemoryPoolExhausted,
                MemoryError::CacheMiss { .. } => ErrorCode::MemoryCacheMiss,
            },
            Error::Gpu(gpu_err) => match gpu_err {
                GpuError::NotAvailable { .. } => ErrorCode::GpuNotAvailable,
                GpuError::Cuda { .. } => ErrorCode::GpuCudaError,
                GpuError::AllocationFailed { .. } => ErrorCode::GpuAllocationFailed,
            },
            Error::Platform(platform_err) => match platform_err {
                PlatformError::NotSupported { .. } => ErrorCode::PlatformNotSupported,
                PlatformError::SdkInitFailed { .. } => ErrorCode::PlatformSdkInitFailed,
                PlatformError::FeatureUnavailable { .. } => ErrorCode::PlatformFeatureUnavailable,
            },
            Error::Validation(validation_err) => match validation_err {
                ValidationError::SchemaFailed { .. } => ErrorCode::Generic,
                ValidationError::RangeError { .. } => ErrorCode::Generic,
                ValidationError::TestFailed { .. } => ErrorCode::Generic,
            },
            Error::Generic { code, .. } => *code,
            _ => ErrorCode::Generic,
        }
    }

    /// Create error with context
    pub fn with_context(message: &str, code: ErrorCode, context: ErrorContext) -> Self {
        Error::Generic {
            message: message.to_string(),
            code,
            context: Some(Box::new(context)),
        }
    }

    /// Check if error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            Error::Config(_) => false, // Config errors usually require restart
            Error::Processing(ProcessingError::RealtimeViolation { .. }) => true,
            Error::Processing(ProcessingError::ResourceExhausted { .. }) => true,
            Error::Memory(_) => true, // Memory issues can often be resolved
            Error::Gpu(_) => true,    // GPU issues may be temporary
            Error::Position(PositionError::TrackingUnavailable { .. }) => true,
            Error::Audio(AudioError::BufferIssue { .. }) => true,
            _ => false,
        }
    }

    /// Get recovery suggestions
    pub fn recovery_suggestions(&self) -> Vec<String> {
        match self {
            Error::Processing(ProcessingError::RealtimeViolation { max_ms, .. }) => {
                vec![
                    "Reduce buffer size to improve latency".to_string(),
                    format!("Increase processing timeout above {}ms", max_ms),
                    "Consider using GPU acceleration".to_string(),
                ]
            }
            Error::Memory(MemoryError::OutOfMemory { .. }) => {
                vec![
                    "Clear cache memory".to_string(),
                    "Reduce buffer pool sizes".to_string(),
                    "Close unused audio sources".to_string(),
                ]
            }
            Error::Gpu(GpuError::NotAvailable { .. }) => {
                vec![
                    "Fallback to CPU processing".to_string(),
                    "Check GPU drivers".to_string(),
                    "Verify CUDA installation".to_string(),
                ]
            }
            Error::Generic {
                context: Some(ctx), ..
            } => ctx.recovery_suggestions.clone(),
            _ => vec!["Check configuration and retry".to_string()],
        }
    }
}

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::{
        ambisonics::{
            channel_count, AmbisonicsDecoder, AmbisonicsEncoder, AmbisonicsOrder,
            AmbisonicsProcessor, ChannelOrdering, NormalizationScheme, SpeakerConfiguration,
            SphericalCoordinate, SphericalHarmonics,
        },
        beamforming::{
            AdaptationConfig, BeamPattern, BeamformerWeights, BeamformingAlgorithm,
            BeamformingConfig, BeamformingProcessor, DoaResult, SpatialSmoothingConfig,
        },
        binaural::{BinauralConfig, BinauralMetrics, BinauralRenderer, SourceType},
        compression::{
            AdaptiveParams, CompressedFrame, CompressionCodec, CompressionQuality,
            CompressionStats, PerceptualParams, SourceClustering, SpatialCompressionConfig,
            SpatialCompressor, SpatialMetadata, SpatialParams, TemporalMasking,
        },
        config::{SpatialConfig, SpatialConfigBuilder},
        core::{SpatialProcessor, SpatialProcessorBuilder},
        gaming::{
            console, AttenuationCurve, AttenuationSettings, AudioCategory, GameAudioSource,
            GameEngine, GamingAudioManager, GamingConfig, GamingMetrics,
        },
        gestures::{
            AudioAction, GestureBuilder, GestureConfidence, GestureConfig, GestureController,
            GestureData, GestureDirection, GestureEvent, GestureEventType,
            GestureRecognitionMethod, GestureType, Hand,
        },
        gpu::{
            GpuAmbisonics, GpuConfig, GpuConvolution, GpuDevice, GpuResourceManager, GpuSpatialMath,
        },
        haptic::{
            AudioHapticMapping, DistanceAttenuation, HapticAccessibilitySettings,
            HapticAudioConfig, HapticAudioProcessor, HapticCapabilities, HapticComfortSettings,
            HapticDevice, HapticEffectType, HapticElement, HapticMetrics, HapticPattern,
            PatternStyle,
        },
        hrtf::{
            ai_personalization::{
                AdaptationStrategy, AiHrtfPersonalizer, AnthropometricMeasurements,
                HrtfModifications, PerceptualFeedback, PersonalizationConfig,
                PersonalizationMetadata, PersonalizedHrtf, TrainingResults,
            },
            HrtfDatabase, HrtfProcessor,
        },
        multiuser::{
            AccessibilitySettings, AcousticSettings, AudioEffect, AudioEffectsProcessor,
            AudioQualityMetrics, AudioSourceType, BandwidthSettings, CodecState, CodecTiming,
            CompensationMethod, ConnectionStatus, DirectionalPattern, DisconnectReason,
            InterpolationMethod, LatencyCompensator, LowBandwidthMode, MicrophoneSettings,
            MixerConfig, MultiUserAttenuationCurve, MultiUserAudioProcessor, MultiUserAudioSource,
            MultiUserConfig, MultiUserConfigBuilder, MultiUserEnvironment, MultiUserEvent,
            MultiUserMetrics, MultiUserUser, NetworkEventType, NetworkStats, OptimizationLevel,
            Permission, PermissionSystem, PositionInterpolator, PositionSnapshot, RoomId,
            SourceAccessControl, SourceId, SourceProcessingState, SourceQualitySettings,
            SourceVisibility, SpatialAudioMixer, SpatialProperties, SpatialZone,
            SynchronizationManager, SynchronizedClock, UserId, UserRole, VadAlgorithm, VadState,
            VadThresholds, VoiceActivityDetector, ZoneAudioProperties, ZoneBounds, ZoneType,
        },
        neural::{
            AdaptiveQualityController, AugmentationConfig, ConvolutionalModel, FeedforwardModel,
            LossFunction, NeuralInputFeatures, NeuralModel, NeuralModelType,
            NeuralPerformanceMetrics, NeuralSpatialConfig, NeuralSpatialConfigBuilder,
            NeuralSpatialOutput, NeuralSpatialProcessor, NeuralTrainer, NeuralTrainingResults,
            OptimizerType, RealtimeConstraints, TrainingConfig, TransformerModel,
        },
        performance::{
            PerformanceConfig, PerformanceMetrics, PerformanceReport, PerformanceSummary,
            PerformanceTargetResult, PerformanceTestSuite, ResourceMonitor, ResourceStatistics,
        },
        platforms::{
            ARCorePlatform, ARKitPlatform, DeviceInfo, EyeData, EyeTrackingData, GenericPlatform,
            HandData, HandGesture, HandTrackingData, OculusPlatform, PlatformCapabilities,
            PlatformFactory, PlatformIntegration, PlatformTrackingData, PoseData, TrackingConfig,
            TrackingQuality, TrackingState, WMRPlatform,
        },
        plugins::{
            PluginCapabilities, PluginConfig, PluginManager, PluginParameter, PluginParameters,
            PluginState, ProcessingChain, ProcessingContext, ReverbPlugin, SpatialPlugin,
        },
        position::{
            advanced_prediction::{
                AdaptationPhase, AdvancedPredictiveTracker, ModelSelectionStrategy, MotionPattern,
                MotionPatternType, PatternRecognitionConfig, PredictedPosition, PredictionMetrics,
                PredictionModelType, PredictiveTrackingConfig,
            },
            Box3D, CalibrationData, ComfortSettings, DopplerProcessor, DynamicSource,
            DynamicSourceManager, HeadTracker, Listener, ListenerMovementSystem, MotionPredictor,
            MotionSnapshot, MovementConstraints, MovementMetrics, NavigationMode,
            OcclusionDetector, OcclusionMaterial, OcclusionMethod, OcclusionResult, PlatformData,
            PlatformType, SoundSource, SpatialSourceManager,
        },
        room::{
            adaptive_acoustics::{
                AdaptationAction, AdaptationController, AdaptationMetrics, AdaptationTrigger,
                AdaptiveAcousticEnvironment, AdaptiveAcousticsConfig, EnvironmentSensors,
                EnvironmentSnapshot, EnvironmentType, SensorConfig, UserFeedback,
            },
            ConnectionAcousticProperties, ConnectionState, ConnectionType, GlobalAcousticConfig,
            MultiRoomEnvironment, Room, RoomAcoustics, RoomConnection, RoomSimulator,
        },
        smart_speakers::{
            ArrayMetrics, ArrayTopology, AudioFormat, AudioRoute, AudioRouter, AudioSource,
            AudioSpecs, CalibrationEngine, CalibrationMethod, CalibrationResults,
            CalibrationStatus, ClockSource, CompressionConfig, DeviceFilter, DirectivityPattern,
            DiscoveryProtocol, DiscoveryService, DspFeature, EQFilter, FilterType, LimitingConfig,
            MixSettings, NetworkConfig, NetworkInfo, NetworkProtocol, ProcessingConfig,
            ProcessingStep, RoomCorrection, SmartSpeaker, SpeakerArrayConfig,
            SpeakerArrayConfigBuilder, SpeakerArrayManager, SpeakerCapabilities, SyncConfig,
        },
        technical_testing::{
            create_standard_technical_configs, LatencyAnalysis, MemoryConstraints,
            PlatformAnalysis, PlatformTestResult, StabilityAnalysis, StressTestParams,
            TechnicalSuccessCriteria, TechnicalTestConfig, TechnicalTestParameters,
            TechnicalTestReport, TechnicalTestResult, TechnicalTestSuite, TechnicalTestType,
            TestOutcome,
        },
        telepresence::{
            AcousticEchoSettings, AcousticMatchingSettings, AcousticProperties,
            AdaptiveQualitySettings, AirAbsorptionSettings, AnonymizationMethod,
            AnonymizationSettings, AudioCodec, AudioDeviceConfig, AudioEnhancementSettings,
            AudioFormat as TelepresenceAudioFormat, AudioMetadata, AudioPresenceSettings,
            AudioQualityPreferences, AudioQualitySettings, BandwidthConstraints,
            BandwidthExtensionSettings, CodecPreferences, CompressionSettings,
            ConsentManagementSettings, CrossRoomSettings, DataCollectionSettings,
            DistanceModelingSettings, DopplerEffectsSettings, EchoCancellationSettings,
            EnvironmentalAwarenessSettings, EqualizationSettings, HeadTrackingSettings,
            HrtfPersonalizationSettings, NetworkSettings, NoiseSuppressionSettings, Orientation,
            PresenceIndicatorSettings, PrivacySettings, QualityLevel, QualitySettings,
            ReceivedAudio, RoomSimulationSettings, SessionJoinResult, SessionState,
            SessionStatistics, SpatialTelepresenceSettings, TelepresenceAudioSettings,
            TelepresenceConfig, TelepresenceProcessor, TelepresenceSession,
            TrackingPredictionSettings, UserConfig, VadSettings, Velocity, VirtualRoomParameters,
            VisualPresenceSettings, VoiceProcessingSettings, VoiceSpatializationSettings,
        },
        types::{
            AudioChannel, BinauraAudio, Position3D, SIMDSpatialOps, SpatialEffect, SpatialRequest,
            SpatialResult,
        },
        validation::{
            create_standard_test_configs, create_test_subjects, AccuracyMetrics, AudioExpertise,
            ExperienceLevel, Gender, HearingAbility, PerceptualTestSuite, PopulationAnalysis,
            ResponseData, StimulusData, SubjectiveRatings, SuccessCriteria, TestParameters,
            TestStatistics, TestSubject, ValidationReport, ValidationTestConfig,
            ValidationTestResult, ValidationTestType,
        },
        visual_audio::{
            AnimationParams, AnimationType, AudioVisualMapping, ColorRGBA, ColorScheme,
            ColorSchemeType, DirectionZone, DirectionalCueMapping, EasingFunction,
            EventTriggerMapping, FrequencyVisualMapping, OnsetTrigger, RhythmTrigger, ScalingCurve,
            ShapeType, SilenceTrigger, SpectralTrigger, VisualAccessibilitySettings,
            VisualAudioConfig, VisualAudioMetrics, VisualAudioProcessor, VisualDisplay,
            VisualDisplayCapabilities, VisualDistanceAttenuation, VisualEffect, VisualElement,
            VisualElementType, VisualPerformanceSettings, VisualResourceUsage, VisualSyncSettings,
        },
        wfs::{
            ArrayGeometry, PreEmphasisConfig, WfsArrayBuilder, WfsConfig, WfsDrivingFunction,
            WfsProcessor, WfsSource, WfsSourceType,
        },
        Error, Result,
    };
}

//! # VoiRS Voice Cloning System
//!
//! This crate provides comprehensive voice cloning capabilities including few-shot speaker
//! adaptation, speaker verification, voice similarity measurement, and cross-language cloning.
//!
//! ## Features
//!
//! - **Few-shot Learning**: Clone voices with as little as 30 seconds of audio
//! - **Speaker Verification**: Verify speaker identity with high accuracy
//! - **Cross-lingual Cloning**: Clone voices across different languages
//! - **Real-time Adaptation**: Adapt speaker characteristics during synthesis
//! - **Quality Assessment**: Automated quality evaluation and similarity measurement
//! - **Ethical Safeguards**: Consent management, usage tracking, and authenticity detection
//! - **Performance Optimization**: SIMD-accelerated operations, GPU support, and quantization
//!
//! ## Quick Start
//!
//! ### Basic Voice Cloning
//!
//! ```rust,no_run
//! use voirs_cloning::{VoiceCloner, VoiceClonerBuilder, VoiceSample, CloningConfig};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a voice cloner with default configuration
//! let cloner = VoiceClonerBuilder::new()
//!     .config(CloningConfig::default())
//!     .build()?;
//!
//! // Prepare voice samples (at least 30 seconds recommended)
//! let audio_data = vec![0.1, -0.1, 0.2, -0.2]; // Your audio samples
//! let sample = VoiceSample::new("speaker1".to_string(), audio_data, 16000);
//!
//! // Voice cloning is performed through speaker embedding and synthesis
//! println!("Voice cloner ready for synthesis");
//! # Ok(())
//! # }
//! ```
//!
//! ### Speaker Verification
//!
//! ```rust,no_run
//! use voirs_cloning::{SpeakerVerifier, VoiceSample};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let mut verifier = SpeakerVerifier::new(Default::default())?;
//!
//! let reference_sample = VoiceSample::new("speaker1".to_string(), vec![0.1; 16000], 16000);
//! let test_sample = VoiceSample::new("speaker1".to_string(), vec![0.2; 16000], 16000);
//!
//! // Verify speaker by comparing samples
//! let result = verifier.verify_samples(&reference_sample, &test_sample).await?;
//! println!("Verification passed: {}", result.verified);
//! println!("Similarity score: {}", result.score);
//! # Ok(())
//! # }
//! ```
//!
//! ### Few-shot Learning
//!
//! ```rust,no_run
//! use voirs_cloning::{FewShotLearner, FewShotConfig, VoiceSample};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Configure few-shot learner for 3-shot learning
//! let config = FewShotConfig {
//!     num_shots: 3,
//!     quality_threshold: 0.5,
//!     ..Default::default()
//! };
//!
//! let mut learner = FewShotLearner::new(config)?;
//!
//! // Prepare 3 samples (30 seconds total recommended)
//! let samples = vec![
//!     VoiceSample::new("speaker1".to_string(), vec![0.1; 48000], 16000),
//!     VoiceSample::new("speaker1".to_string(), vec![0.2; 48000], 16000),
//!     VoiceSample::new("speaker1".to_string(), vec![0.3; 48000], 16000),
//! ];
//!
//! // Adapt speaker with few-shot learning
//! let result = learner.adapt_speaker("speaker1", &samples).await?;
//! println!("Adaptation confidence: {}", result.confidence);
//! println!("Quality score: {}", result.quality_score);
//! # Ok(())
//! # }
//! ```
//!
//! ### Quality Assessment
//!
//! ```rust,no_run
//! use voirs_cloning::{CloningQualityAssessor, VoiceSample};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let mut assessor = CloningQualityAssessor::new()?;
//!
//! let original = VoiceSample::new("original".to_string(), vec![0.1; 16000], 16000);
//! let cloned = VoiceSample::new("cloned".to_string(), vec![0.11; 16000], 16000);
//!
//! // Assess cloning quality
//! let metrics = assessor.assess_quality(&original, &cloned).await?;
//! println!("Overall quality: {}", metrics.overall_score);
//! println!("Speaker similarity: {}", metrics.speaker_similarity);
//! println!("Audio quality: {}", metrics.audio_quality);
//! println!("Naturalness: {}", metrics.naturalness);
//! # Ok(())
//! # }
//! ```
//!
//! ## Advanced Features
//!
//! ### Cross-lingual Voice Cloning
//!
//! ```rust,no_run
//! use voirs_cloning::{FewShotLearner, FewShotConfig, VoiceSample};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let config = FewShotConfig {
//!     enable_cross_lingual: true,
//!     ..Default::default()
//! };
//!
//! let mut learner = FewShotLearner::new(config)?;
//!
//! let samples_en = vec![/* English samples */];
//! let result = learner
//!     .adapt_speaker_cross_lingual("speaker1", &samples_en, "en", "es")
//!     .await?;
//! # Ok(())
//! # }
//! ```
//!
//! ### Voice Morphing
//!
//! ```rust,no_run
//! use voirs_cloning::{VoiceMorpher, VoiceMorphingConfig, VoiceMorphingRequest, MorphingWeight, InterpolationMethod};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let morpher = VoiceMorpher::new(VoiceMorphingConfig::default())?;
//!
//! // Morph between two speakers with 50/50 blend
//! let request = VoiceMorphingRequest {
//!     target_id: "morphed_voice".to_string(),
//!     speaker_weights: vec![
//!         MorphingWeight {
//!             speaker_id: "speaker1".to_string(),
//!             weight: 0.5,
//!             quality_boost: 1.0,
//!             temporal_variation: None,
//!         },
//!         MorphingWeight {
//!             speaker_id: "speaker2".to_string(),
//!             weight: 0.5,
//!             quality_boost: 1.0,
//!             temporal_variation: None,
//!         },
//!     ],
//!     config: VoiceMorphingConfig {
//!         interpolation_method: InterpolationMethod::Weighted,
//!         ..Default::default()
//!     },
//!     target_characteristics: None,
//!     morphing_duration: None,
//! };
//!
//! let result = morpher.morph_voices(request).await?;
//! # Ok(())
//! # }
//! ```
//!
//! ### Age/Gender Adaptation
//!
//! ```rust,no_run
//! use voirs_cloning::{AgeGenderAdapter, VoiceAdaptationTarget, AgeCategory, GenderCategory, VoiceSample};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let mut adapter = AgeGenderAdapter::new();
//!
//! let source_samples = vec![VoiceSample::new("speaker1".to_string(), vec![0.1; 16000], 16000)];
//!
//! // Train adaptation model to sound younger and more feminine
//! let target = VoiceAdaptationTarget {
//!     age: AgeCategory::YoungAdult,
//!     gender: GenderCategory::Feminine,
//!     age_intensity: 0.7,
//!     gender_intensity: 0.7,
//!     identity_preservation: 0.8,
//! };
//!
//! let model = adapter.train_adaptation_model("speaker1", &source_samples, target).await?;
//! let result = adapter.adapt_voice(&model, &source_samples).await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Performance Considerations
//!
//! ### GPU Acceleration
//!
//! Enable GPU acceleration for faster processing:
//!
//! ```rust,no_run
//! use voirs_cloning::{GpuAccelerator, GpuAccelerationConfig};
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create GPU accelerator with default configuration
//! let config = GpuAccelerationConfig::default();
//! let accelerator = GpuAccelerator::new(config)?;
//!
//! // GPU will be used automatically for supported operations
//! # Ok(())
//! # }
//! ```
//!
//! ### Model Quantization
//!
//! Reduce memory footprint with quantization:
//!
//! ```rust,no_run
//! use voirs_cloning::{ModelQuantizer, QuantizationConfig, QuantizationPrecision};
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a CPU device for quantization
//! let device = candle_core::Device::Cpu;
//!
//! let config = QuantizationConfig {
//!     precision: QuantizationPrecision::Int8,
//!     ..Default::default()
//! };
//!
//! let quantizer = ModelQuantizer::new(config, device)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Ethical Usage
//!
//! ### Consent Management
//!
//! ```rust,no_run
//! use voirs_cloning::ConsentManager;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a consent manager for managing voice cloning consent
//! let manager = ConsentManager::new();
//!
//! // The consent manager tracks and verifies consent for voice usage
//! // See the consent module documentation for complete usage examples
//! println!("Consent manager initialized");
//! # Ok(())
//! # }
//! ```
//!
//! ### Usage Tracking
//!
//! ```rust,no_run
//! use voirs_cloning::{UsageTracker, UsageTrackingConfig, CloningOperationType};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let tracker = UsageTracker::new(UsageTrackingConfig::default());
//!
//! // Start tracking a voice cloning operation
//! let operation = tracker.start_operation(
//!     "user123".to_string(),
//!     "speaker1".to_string(),
//!     CloningOperationType::VoiceCloning
//! ).await?;
//! println!("Tracking operation: {}", operation.id);
//! # Ok(())
//! # }
//! ```
//!
//! ## Architecture
//!
//! The voice cloning system is organized into several specialized modules:
//!
//! - **core**: Core voice cloning functionality and builder pattern
//! - **embedding**: Speaker embedding extraction and similarity computation
//! - **few_shot**: Few-shot learning algorithms for rapid adaptation
//! - **verification**: Speaker verification and identity validation
//! - **quality**: Quality assessment and perceptual evaluation
//! - **consent**: Ethical safeguards and consent management
//! - **usage_tracking**: Usage monitoring and audit logging
//! - **authenticity**: Deepfake detection and authenticity validation
//!
//! ## Performance Benchmarks
//!
//! Run benchmarks to measure performance:
//!
//! ```bash
//! cargo bench --features "acoustic-integration"
//! ```
//!
//! ## Feature Flags
//!
//! - `acoustic-integration`: Enable integration with voirs-acoustic
//! - `g2p-integration`: Enable integration with voirs-g2p
//! - `gpu`: Enable GPU acceleration support
//! - `cuda`: Enable CUDA GPU support
//! - `metal`: Enable Metal GPU support
//! - `wasm`: Enable WebAssembly support
//!
//! ## License
//!
//! This crate is part of the VoiRS project. See LICENSE for details.

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
#![deny(unsafe_code)]

pub mod ab_testing;
pub mod adaptation;
pub mod adversarial_robustness;
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
pub mod consistency_models;
pub mod core;
pub mod deep_mos;
pub mod edge;
pub mod embedding;
pub mod emotion_transfer;
pub mod enterprise_sso;
pub mod error_handling;
pub mod few_shot;
pub mod flow_matching;
pub mod gaming_plugins;
pub mod gpu_acceleration;
pub mod kernel_fusion;
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
pub mod qat;
pub mod quality;
pub mod quality_visualization;
pub mod quantization;
pub mod realtime_streaming;
pub mod similarity;
pub mod ssl_verification;
pub mod storage;
pub mod streaming_adaptation;
pub mod thread_safety;
pub mod types;
pub mod usage_tracking;
pub mod utils;
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

    /// Lock operation error
    #[error("Lock error: {0}")]
    LockError(String),

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

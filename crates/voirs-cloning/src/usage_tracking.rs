//! Usage Tracking System for Voice Cloning
//!
//! This module provides comprehensive usage tracking capabilities to monitor and audit
//! voice cloning operations. It ensures compliance with consent requirements and enables
//! detailed analysis of usage patterns for security and ethical oversight.

use crate::consent::{ConsentManager, ConsentUsageContext, ConsentUsageResult};
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use uuid::Uuid;

/// Comprehensive usage tracking record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageRecord {
    /// Unique identifier for this usage record
    pub usage_id: Uuid,

    /// Associated consent record ID
    pub consent_id: Option<Uuid>,

    /// User/application information
    pub user_context: UserContext,

    /// Details of the voice cloning operation
    pub operation: CloningOperation,

    /// Usage outcome and results
    pub outcome: UsageOutcome,

    /// Resource consumption metrics
    pub resources: ResourceUsage,

    /// Security and compliance information
    pub security: SecurityContext,

    /// Timestamps for the operation
    pub timestamps: UsageTimestamps,

    /// Quality metrics
    pub quality_metrics: QualityMetrics,

    /// Geographic and network information
    pub location: LocationContext,

    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// User and application context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserContext {
    /// User identifier (anonymized)
    pub user_id: Option<String>,

    /// Application identifier
    pub application_id: String,

    /// Application version
    pub application_version: String,

    /// Client type (web, mobile, desktop, api)
    pub client_type: ClientType,

    /// Session identifier
    pub session_id: Option<String>,

    /// Request identifier for tracing
    pub request_id: Option<String>,

    /// Authentication method used
    pub auth_method: Option<AuthenticationMethod>,

    /// User agent string
    pub user_agent: Option<String>,
}

/// Types of clients
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClientType {
    WebBrowser,
    MobileApp,
    DesktopApp,
    ServerToServer,
    API,
    SDK,
    CLI,
    Unknown,
}

/// Authentication methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthenticationMethod {
    APIKey,
    OAuth2,
    JWT,
    BasicAuth,
    Certificate,
    Biometric,
    None,
}

/// Request metadata for operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationRequestMetadata {
    pub request_id: String,
    pub timestamp: SystemTime,
    pub priority: Priority,
    pub source_application: String,
    pub user_preferences: UserPreferences,
}

/// Priority levels for requests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Priority {
    Low,
    Normal,
    High,
    Urgent,
}

/// User preferences for operations
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UserPreferences {
    pub quality_preference: QualityPreference,
    pub speed_preference: SpeedPreference,
    pub additional_settings: HashMap<String, String>,
}

/// Quality preference settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QualityPreference {
    Draft,
    Standard,
    High,
    Premium,
}

impl Default for QualityPreference {
    fn default() -> Self {
        QualityPreference::Standard
    }
}

/// Speed preference settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpeedPreference {
    Fastest,
    Fast,
    Balanced,
    Quality,
}

impl Default for SpeedPreference {
    fn default() -> Self {
        SpeedPreference::Balanced
    }
}

/// Details of the voice cloning operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloningOperation {
    /// Type of cloning operation
    pub operation_type: CloningOperationType,

    /// Speaker ID (if applicable)
    pub speaker_id: Option<String>,

    /// Target speaker ID (if applicable)
    pub target_speaker_id: Option<String>,

    /// Request metadata
    pub request_metadata: OperationRequestMetadata,

    /// Input data characteristics
    pub input_data: InputDataInfo,

    /// Processing parameters
    pub processing_params: ProcessingParameters,

    /// Output characteristics
    pub output_data: OutputDataInfo,

    /// Processing pipeline used
    pub pipeline_info: PipelineInfo,
}

/// Types of cloning operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CloningOperationType {
    VoiceTraining,
    VoiceSynthesis,
    VoiceAdaptation,
    VoiceVerification,
    VoiceAnalysis,
    VoiceConversion,
    VoiceCloning,
    SpeakerEmbedding,
    QualityAssessment,
    SynthesisGeneration,
}

/// Information about input data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputDataInfo {
    /// Type of input data
    pub data_type: InputDataType,

    /// Size of input data in bytes
    pub data_size_bytes: u64,

    /// Duration of audio input (if applicable)
    pub audio_duration_seconds: Option<f64>,

    /// Text length (if applicable)
    pub text_length: Option<usize>,

    /// Language of content
    pub language: Option<String>,

    /// Content hash for deduplication
    pub content_hash: Option<String>,

    /// Quality assessment of input
    pub input_quality_score: Option<f64>,
}

/// Types of input data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InputDataType {
    AudioFile,
    AudioStream,
    TextPrompt,
    SpeakerEmbedding,
    VoiceModel,
    ReferenceAudio,
    TrainingData,
}

/// Processing parameters used
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingParameters {
    /// Quality level requested
    pub quality_level: QualityLevel,

    /// Processing mode
    pub processing_mode: ProcessingMode,

    /// Model configuration
    pub model_config: ModelConfiguration,

    /// Advanced parameters
    pub advanced_params: HashMap<String, String>,
}

/// Quality levels for processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QualityLevel {
    Draft,
    Standard,
    High,
    Premium,
    Custom(f64),
}

/// Processing modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessingMode {
    Fast,
    Balanced,
    HighQuality,
    Experimental,
    Production,
}

/// Model configuration information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfiguration {
    pub model_name: String,
    pub model_version: String,
    pub model_type: ModelType,
    pub model_size_mb: Option<f64>,
    pub training_data_info: Option<String>,
}

/// Types of models used
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    Acoustic,
    Vocoder,
    SpeakerEncoder,
    LanguageModel,
    Hybrid,
}

/// Information about output data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputDataInfo {
    /// Type of output
    pub output_type: OutputDataType,

    /// Size of output data in bytes
    pub data_size_bytes: u64,

    /// Duration of generated audio (if applicable)
    pub audio_duration_seconds: Option<f64>,

    /// Quality score of output
    pub quality_score: Option<f64>,

    /// Similarity score to target (if applicable)
    pub similarity_score: Option<f64>,

    /// Output format
    pub format: Option<String>,

    /// Sample rate (for audio)
    pub sample_rate: Option<u32>,
}

/// Types of output data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutputDataType {
    SynthesizedAudio,
    VoiceModel,
    SpeakerEmbedding,
    QualityReport,
    VerificationResult,
    AnalysisReport,
}

/// Pipeline information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineInfo {
    pub pipeline_id: String,
    pub pipeline_version: String,
    pub components_used: Vec<String>,
    pub processing_stages: Vec<ProcessingStage>,
}

/// Individual processing stages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingStage {
    pub stage_name: String,
    pub stage_duration_ms: u64,
    pub stage_resources: StageResources,
    pub stage_output_quality: Option<f64>,
}

/// Resources used by a processing stage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageResources {
    pub cpu_time_ms: u64,
    pub memory_peak_mb: f64,
    pub gpu_time_ms: Option<u64>,
    pub gpu_memory_mb: Option<f64>,
}

/// Usage outcome and results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageOutcome {
    /// Success status of the operation
    pub status: UsageStatus,

    /// Error information if failed
    pub error: Option<UsageError>,

    /// Compliance status
    pub compliance_status: ComplianceStatus,

    /// Consent check result
    pub consent_result: Option<ConsentCheckResult>,

    /// Usage restrictions applied
    pub restrictions_applied: Vec<String>,

    /// Warnings generated
    pub warnings: Vec<String>,
}

/// Status of the usage operation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum UsageStatus {
    Success,
    PartialSuccess,
    Failed,
    Blocked,
    RateLimited,
    ConsentDenied,
}

/// Usage error information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageError {
    pub error_type: UsageErrorType,
    pub error_code: String,
    pub error_message: String,
    pub retry_possible: bool,
    pub retry_after: Option<Duration>,
}

/// Types of usage errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UsageErrorType {
    ValidationError,
    ProcessingError,
    ResourceError,
    ConsentError,
    SecurityError,
    RateLimitError,
    QuotaExceededError,
    SystemError,
}

/// Compliance status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceStatus {
    pub is_compliant: bool,
    pub compliance_checks: Vec<ComplianceCheck>,
    pub violations: Vec<ComplianceViolation>,
    pub risk_level: RiskLevel,
}

/// Individual compliance checks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceCheck {
    pub check_name: String,
    pub check_result: ComplianceCheckResult,
    pub check_details: String,
}

/// Results of compliance checks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplianceCheckResult {
    Pass,
    Fail,
    Warning,
    NotApplicable,
}

/// Compliance violations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceViolation {
    pub violation_type: String,
    pub severity: ViolationSeverity,
    pub description: String,
    pub remediation: Option<String>,
}

/// Severity of violations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViolationSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Risk levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLevel {
    Minimal,
    Low,
    Medium,
    High,
    Critical,
}

/// Consent check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsentCheckResult {
    pub consent_status: ConsentStatus,
    pub permissions_checked: Vec<String>,
    pub restrictions_applied: Vec<String>,
    pub check_timestamp: SystemTime,
}

/// Consent status from tracking perspective
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsentStatus {
    Valid,
    Invalid,
    Expired,
    Revoked,
    NotRequired,
    NotFound,
}

/// Resource consumption metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// Total processing time
    pub total_processing_time_ms: u64,

    /// CPU utilization
    pub cpu_usage: CpuUsage,

    /// Memory usage
    pub memory_usage: MemoryUsage,

    /// GPU usage (if applicable)
    pub gpu_usage: Option<GpuUsage>,

    /// Network usage
    pub network_usage: NetworkUsage,

    /// Storage usage
    pub storage_usage: StorageUsage,

    /// Cost estimation
    pub cost_estimate: Option<CostBreakdown>,
}

/// CPU usage metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuUsage {
    pub peak_cpu_percent: f64,
    pub average_cpu_percent: f64,
    pub cpu_time_seconds: f64,
    pub cpu_cores_used: u32,
}

/// Memory usage metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsage {
    pub peak_memory_mb: f64,
    pub average_memory_mb: f64,
    pub memory_allocated_mb: f64,
    pub memory_freed_mb: f64,
}

/// GPU usage metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuUsage {
    pub gpu_time_seconds: f64,
    pub peak_gpu_memory_mb: f64,
    pub gpu_utilization_percent: f64,
    pub gpu_device_name: String,
}

/// Network usage metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkUsage {
    pub bytes_uploaded: u64,
    pub bytes_downloaded: u64,
    pub requests_made: u32,
    pub bandwidth_peak_mbps: Option<f64>,
}

/// Storage usage metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageUsage {
    pub temporary_storage_mb: f64,
    pub persistent_storage_mb: f64,
    pub cache_storage_mb: f64,
    pub files_created: u32,
    pub files_deleted: u32,
}

/// Cost breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostBreakdown {
    pub compute_cost: f64,
    pub storage_cost: f64,
    pub network_cost: f64,
    pub total_cost: f64,
    pub currency: String,
}

/// Security context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityContext {
    /// Security checks performed
    pub security_checks: Vec<SecurityCheck>,

    /// Anomaly detection results
    pub anomaly_detection: AnomalyDetectionResult,

    /// Rate limiting information
    pub rate_limiting: RateLimitingInfo,

    /// Threat assessment
    pub threat_assessment: ThreatAssessment,

    /// Access control information
    pub access_control: AccessControlInfo,
}

/// Security checks performed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityCheck {
    pub check_name: String,
    pub check_result: SecurityCheckResult,
    pub risk_score: f64,
    pub details: String,
}

/// Results of security checks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityCheckResult {
    Pass,
    Fail,
    Warning,
    Suspicious,
}

/// Anomaly detection results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetectionResult {
    pub anomaly_score: f64,
    pub anomaly_threshold: f64,
    pub is_anomalous: bool,
    pub anomaly_type: Option<AnomalyType>,
    pub anomaly_details: String,
}

/// Types of anomalies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyType {
    FrequencyAnomaly,
    VolumeAnomaly,
    PatternAnomaly,
    GeographicAnomaly,
    TimeAnomaly,
    ContentAnomaly,
}

/// Rate limiting information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitingInfo {
    pub rate_limit_applied: bool,
    pub current_usage: u32,
    pub rate_limit_threshold: u32,
    pub reset_time: Option<SystemTime>,
    pub quota_remaining: Option<u32>,
}

/// Threat assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatAssessment {
    pub threat_level: ThreatLevel,
    pub threat_indicators: Vec<String>,
    pub mitigation_actions: Vec<String>,
}

/// Threat levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThreatLevel {
    None,
    Low,
    Medium,
    High,
    Critical,
}

/// Access control information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessControlInfo {
    pub permissions_granted: Vec<String>,
    pub permissions_denied: Vec<String>,
    pub access_level: AccessLevel,
    pub authentication_strength: AuthenticationStrength,
}

/// Access levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AccessLevel {
    Guest,
    Basic,
    Standard,
    Premium,
    Admin,
}

/// Authentication strength
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthenticationStrength {
    Weak,
    Moderate,
    Strong,
    VeryStrong,
}

/// Timestamps for usage tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageTimestamps {
    pub request_received: SystemTime,
    pub processing_started: SystemTime,
    pub processing_completed: Option<SystemTime>,
    pub response_sent: Option<SystemTime>,
    pub consent_checked: Option<SystemTime>,
}

/// Quality metrics for the operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    pub overall_quality: Option<f64>,
    pub audio_quality: Option<AudioQualityMetrics>,
    pub similarity_metrics: Option<SimilarityMetrics>,
    pub naturalness_score: Option<f64>,
    pub intelligibility_score: Option<f64>,
}

/// Audio quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioQualityMetrics {
    pub signal_to_noise_ratio: f64,
    pub total_harmonic_distortion: f64,
    pub frequency_response_score: f64,
    pub dynamic_range: f64,
}

/// Similarity metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityMetrics {
    pub speaker_similarity: f64,
    pub prosody_similarity: f64,
    pub acoustic_similarity: f64,
    pub perceptual_similarity: f64,
}

/// Location and network context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocationContext {
    pub ip_address: Option<String>,
    pub country: Option<String>,
    pub region: Option<String>,
    pub city: Option<String>,
    pub timezone: Option<String>,
    pub isp: Option<String>,
    pub asn: Option<String>,
    pub is_vpn: Option<bool>,
    pub is_proxy: Option<bool>,
}

/// Simple operation record for compatibility with tests
#[derive(Debug, Clone)]
pub struct OperationRecord {
    pub id: String,
    pub user_id: String,
    pub speaker_id: String,
    pub usage_id: Uuid,
}

/// Usage tracking system
pub struct UsageTracker {
    /// Storage for usage records
    usage_store: Arc<RwLock<HashMap<Uuid, UsageRecord>>>,

    /// Active sessions
    active_sessions: Arc<RwLock<HashMap<String, ActiveSession>>>,

    /// Usage statistics
    statistics: Arc<RwLock<UsageStatistics>>,

    /// Consent manager integration
    consent_manager: Option<Arc<Mutex<ConsentManager>>>,

    /// Configuration
    config: UsageTrackingConfig,

    /// Event processors
    event_processors: Vec<Box<dyn UsageEventProcessor>>,

    /// Storage backends
    storage_backends: Vec<Box<dyn UsageStorageBackend>>,
}

/// Active session tracking
#[derive(Debug, Clone)]
pub struct ActiveSession {
    pub session_id: String,
    pub user_id: Option<String>,
    pub start_time: SystemTime,
    pub last_activity: SystemTime,
    pub request_count: u32,
    pub resource_usage: ResourceUsage,
    pub current_operations: Vec<Uuid>,
}

/// Usage statistics
#[derive(Debug, Default, Clone, Serialize)]
pub struct UsageStatistics {
    pub total_operations: u64,
    pub successful_operations: u64,
    pub failed_operations: u64,
    pub blocked_operations: u64,
    pub average_processing_time_ms: f64,
    pub total_resource_cost: f64,
    pub consent_violations: u64,
    pub security_incidents: u64,
    pub top_users: Vec<(String, u64)>,
    pub top_applications: Vec<(String, u64)>,
    pub operation_types: HashMap<String, u64>,
    pub error_types: HashMap<String, u64>,
}

/// Configuration for usage tracking
#[derive(Debug, Clone)]
pub struct UsageTrackingConfig {
    pub enable_tracking: bool,
    pub track_resource_usage: bool,
    pub track_quality_metrics: bool,
    pub track_security_events: bool,
    pub retention_days: u32,
    pub max_records_in_memory: usize,
    pub batch_size: usize,
    pub flush_interval: Duration,
    pub anonymize_user_data: bool,
    pub encrypt_sensitive_data: bool,
}

/// Trait for processing usage events
pub trait UsageEventProcessor: Send + Sync {
    fn process_usage_event(&self, usage: &UsageRecord) -> Result<()>;
    fn get_processor_name(&self) -> &str;
}

/// Trait for usage storage backends
pub trait UsageStorageBackend: Send + Sync {
    fn store_usage_record(&self, usage: &UsageRecord) -> Result<()>;
    fn store_batch(&self, usages: &[UsageRecord]) -> Result<()>;
    fn retrieve_usage_records(&self, filters: &UsageQueryFilters) -> Result<Vec<UsageRecord>>;
    fn get_backend_name(&self) -> &str;
}

/// Filters for querying usage records
#[derive(Debug, Clone)]
pub struct UsageQueryFilters {
    pub user_id: Option<String>,
    pub application_id: Option<String>,
    pub operation_type: Option<CloningOperationType>,
    pub start_time: Option<SystemTime>,
    pub end_time: Option<SystemTime>,
    pub status: Option<UsageStatus>,
    pub limit: Option<usize>,
}

impl Default for UsageTrackingConfig {
    fn default() -> Self {
        UsageTrackingConfig {
            enable_tracking: true,
            track_resource_usage: true,
            track_quality_metrics: true,
            track_security_events: true,
            retention_days: 365,
            max_records_in_memory: 10000,
            batch_size: 100,
            flush_interval: Duration::from_secs(60),
            anonymize_user_data: true,
            encrypt_sensitive_data: true,
        }
    }
}

impl UsageTracker {
    /// Create a new usage tracker
    pub fn new(config: UsageTrackingConfig) -> Self {
        UsageTracker {
            usage_store: Arc::new(RwLock::new(HashMap::new())),
            active_sessions: Arc::new(RwLock::new(HashMap::new())),
            statistics: Arc::new(RwLock::new(UsageStatistics::default())),
            consent_manager: None,
            config,
            event_processors: Vec::new(),
            storage_backends: Vec::new(),
        }
    }

    /// Set consent manager for compliance checking
    pub fn set_consent_manager(&mut self, consent_manager: Arc<Mutex<ConsentManager>>) {
        self.consent_manager = Some(consent_manager);
    }

    /// Add event processor
    pub fn add_event_processor(&mut self, processor: Box<dyn UsageEventProcessor>) {
        self.event_processors.push(processor);
    }

    /// Add storage backend
    pub fn add_storage_backend(&mut self, backend: Box<dyn UsageStorageBackend>) {
        self.storage_backends.push(backend);
    }

    /// Start tracking an operation (simplified API for tests)
    pub async fn start_operation(
        &self,
        user_id: String,
        speaker_id: String,
        operation_type: CloningOperationType,
    ) -> Result<OperationRecord> {
        // Create simplified UserContext and CloningOperation
        let user_context = UserContext {
            user_id: Some(user_id.clone()),
            application_id: "test_app".to_string(),
            application_version: "1.0.0".to_string(),
            client_type: ClientType::API,
            session_id: Some(format!("session_{user_id}")),
            request_id: Some(format!("req_{}", Uuid::new_v4())),
            auth_method: Some(AuthenticationMethod::APIKey),
            user_agent: None,
        };

        let operation = CloningOperation {
            operation_type,
            speaker_id: Some(speaker_id.clone()),
            target_speaker_id: None,
            request_metadata: OperationRequestMetadata {
                request_id: format!("req_{}", Uuid::new_v4()),
                timestamp: SystemTime::now(),
                priority: Priority::Normal,
                source_application: "voirs-cloning".to_string(),
                user_preferences: UserPreferences::default(),
            },
            input_data: InputDataInfo {
                data_type: InputDataType::AudioFile,
                data_size_bytes: 1024,
                audio_duration_seconds: Some(10.0),
                text_length: None,
                language: Some("en".to_string()),
                content_hash: None,
                input_quality_score: None,
            },
            processing_params: ProcessingParameters {
                quality_level: QualityLevel::Standard,
                processing_mode: ProcessingMode::Balanced,
                model_config: ModelConfiguration {
                    model_name: "test_model".to_string(),
                    model_version: "1.0".to_string(),
                    model_type: ModelType::Acoustic,
                    model_size_mb: Some(100.0),
                    training_data_info: None,
                },
                advanced_params: HashMap::new(),
            },
            output_data: OutputDataInfo {
                output_type: OutputDataType::SynthesizedAudio,
                data_size_bytes: 0,
                audio_duration_seconds: None,
                quality_score: None,
                similarity_score: None,
                format: Some("wav".to_string()),
                sample_rate: Some(22050),
            },
            pipeline_info: PipelineInfo {
                pipeline_id: "test_pipeline".to_string(),
                pipeline_version: "1.0".to_string(),
                components_used: vec!["acoustic".to_string()],
                processing_stages: Vec::new(),
            },
        };

        let usage_id = self.start_tracking(user_context, operation)?;

        Ok(OperationRecord {
            id: usage_id.to_string(),
            user_id,
            speaker_id,
            usage_id,
        })
    }

    /// Start tracking a usage operation
    pub fn start_tracking(
        &self,
        user_context: UserContext,
        operation: CloningOperation,
    ) -> Result<Uuid> {
        if !self.config.enable_tracking {
            return Ok(Uuid::new_v4()); // Return dummy ID
        }

        let usage_id = Uuid::new_v4();
        let now = SystemTime::now();

        let usage_record = UsageRecord {
            usage_id,
            consent_id: None,
            user_context,
            operation,
            outcome: UsageOutcome {
                status: UsageStatus::Success, // Will be updated
                error: None,
                compliance_status: ComplianceStatus {
                    is_compliant: true,
                    compliance_checks: Vec::new(),
                    violations: Vec::new(),
                    risk_level: RiskLevel::Minimal,
                },
                consent_result: None,
                restrictions_applied: Vec::new(),
                warnings: Vec::new(),
            },
            resources: ResourceUsage::default(),
            security: SecurityContext::default(),
            timestamps: UsageTimestamps {
                request_received: now,
                processing_started: now,
                processing_completed: None,
                response_sent: None,
                consent_checked: None,
            },
            quality_metrics: QualityMetrics::default(),
            location: LocationContext::default(),
            metadata: HashMap::new(),
        };

        // Store the record
        {
            let mut store = self.usage_store.write().unwrap();
            store.insert(usage_id, usage_record.clone());
        }

        // Update session tracking
        if let Some(ref session_id) = usage_record.user_context.session_id {
            let mut sessions = self.active_sessions.write().unwrap();
            let session = sessions
                .entry(session_id.clone())
                .or_insert_with(|| ActiveSession {
                    session_id: session_id.clone(),
                    user_id: usage_record.user_context.user_id.clone(),
                    start_time: now,
                    last_activity: now,
                    request_count: 0,
                    resource_usage: ResourceUsage::default(),
                    current_operations: Vec::new(),
                });
            session.request_count += 1;
            session.last_activity = now;
            session.current_operations.push(usage_id);
        }

        Ok(usage_id)
    }

    /// Check consent for the operation
    pub fn check_consent(
        &self,
        usage_id: Uuid,
        consent_id: Option<Uuid>,
        use_case: &str,
    ) -> Result<ConsentUsageResult> {
        if let Some(ref consent_manager) = self.consent_manager {
            if let Some(consent_id) = consent_id {
                let manager = consent_manager.lock().unwrap();

                // Create context from usage record
                let context = {
                    let store = self.usage_store.read().unwrap();
                    let usage = store
                        .get(&usage_id)
                        .ok_or_else(|| Error::Validation("Usage record not found".to_string()))?;

                    ConsentUsageContext {
                        use_case: use_case.to_string(),
                        application: Some(usage.user_context.application_id.clone()),
                        user: usage.user_context.user_id.clone(),
                        country: usage.location.country.clone(),
                        region: usage.location.region.clone(),
                        content_text: None,
                        timestamp: SystemTime::now(),
                        ip_address: usage.location.ip_address.clone(),
                        // Additional fields
                        operation_type: usage.operation.operation_type.clone(),
                        user_id: usage.user_context.user_id.clone().unwrap_or_default(),
                        location: usage.location.country.clone(),
                        additional_context: std::collections::HashMap::new(),
                    }
                };

                let result = manager.check_consent_for_use(consent_id, use_case, &context)?;

                // Update usage record with consent check result
                {
                    let mut store = self.usage_store.write().unwrap();
                    if let Some(usage) = store.get_mut(&usage_id) {
                        usage.consent_id = Some(consent_id);
                        usage.outcome.consent_result = Some(ConsentCheckResult {
                            consent_status: match result {
                                ConsentUsageResult::Allowed => ConsentStatus::Valid,
                                ConsentUsageResult::Denied(_) => ConsentStatus::Invalid,
                                ConsentUsageResult::Restricted(_) => ConsentStatus::Valid,
                            },
                            permissions_checked: vec![use_case.to_string()],
                            restrictions_applied: Vec::new(),
                            check_timestamp: SystemTime::now(),
                        });
                        usage.timestamps.consent_checked = Some(SystemTime::now());
                    }
                }

                Ok(result)
            } else {
                // No consent ID provided - this might be acceptable for some use cases
                Ok(ConsentUsageResult::Allowed)
            }
        } else {
            // No consent manager configured - allow by default but log warning
            Ok(ConsentUsageResult::Allowed)
        }
    }

    /// Complete tracking for a usage operation
    pub fn complete_tracking(
        &self,
        usage_id: Uuid,
        outcome: UsageOutcome,
        resources: ResourceUsage,
        quality_metrics: Option<QualityMetrics>,
    ) -> Result<()> {
        let now = SystemTime::now();

        // Update the usage record
        {
            let mut store = self.usage_store.write().unwrap();
            if let Some(usage) = store.get_mut(&usage_id) {
                usage.outcome = outcome.clone();
                usage.resources = resources.clone();
                if let Some(quality) = quality_metrics {
                    usage.quality_metrics = quality;
                }
                usage.timestamps.processing_completed = Some(now);
                usage.timestamps.response_sent = Some(now);
            } else {
                return Err(Error::Validation("Usage record not found".to_string()));
            }
        }

        // Update statistics
        {
            let mut stats = self.statistics.write().unwrap();
            stats.total_operations += 1;
            match outcome.status {
                UsageStatus::Success | UsageStatus::PartialSuccess => {
                    stats.successful_operations += 1;
                }
                UsageStatus::Failed | UsageStatus::Blocked => {
                    stats.failed_operations += 1;
                }
                _ => {}
            }

            stats.total_resource_cost +=
                resources.cost_estimate.map(|c| c.total_cost).unwrap_or(0.0);
        }

        // Process with event processors
        let usage_record = {
            let store = self.usage_store.read().unwrap();
            store.get(&usage_id).cloned()
        };

        if let Some(usage) = usage_record {
            for processor in &self.event_processors {
                let _ = processor.process_usage_event(&usage);
            }

            // Store to backends
            for backend in &self.storage_backends {
                let _ = backend.store_usage_record(&usage);
            }
        }

        Ok(())
    }

    /// Get usage statistics
    pub fn get_statistics(&self) -> UsageStatistics {
        let stats = self.statistics.read().unwrap();
        stats.clone()
    }

    /// Query usage records
    pub fn query_usage_records(&self, filters: &UsageQueryFilters) -> Result<Vec<UsageRecord>> {
        let store = self.usage_store.read().unwrap();
        let mut results: Vec<UsageRecord> = store
            .values()
            .filter(|usage| self.matches_filters(usage, filters))
            .cloned()
            .collect();

        // Sort by timestamp (most recent first)
        results.sort_by(|a, b| {
            b.timestamps
                .request_received
                .cmp(&a.timestamps.request_received)
        });

        // Apply limit
        if let Some(limit) = filters.limit {
            results.truncate(limit);
        }

        Ok(results)
    }

    /// Check if usage record matches filters
    fn matches_filters(&self, usage: &UsageRecord, filters: &UsageQueryFilters) -> bool {
        if let Some(ref user_id) = filters.user_id {
            if usage.user_context.user_id.as_ref() != Some(user_id) {
                return false;
            }
        }

        if let Some(ref app_id) = filters.application_id {
            if &usage.user_context.application_id != app_id {
                return false;
            }
        }

        if let Some(ref op_type) = filters.operation_type {
            if std::mem::discriminant(&usage.operation.operation_type)
                != std::mem::discriminant(op_type)
            {
                return false;
            }
        }

        if let Some(start_time) = filters.start_time {
            if usage.timestamps.request_received < start_time {
                return false;
            }
        }

        if let Some(end_time) = filters.end_time {
            if usage.timestamps.request_received > end_time {
                return false;
            }
        }

        if let Some(ref status) = filters.status {
            if std::mem::discriminant(&usage.outcome.status) != std::mem::discriminant(status) {
                return false;
            }
        }

        true
    }

    /// Generate usage report
    pub fn generate_usage_report(&self, filters: &UsageQueryFilters) -> Result<UsageReport> {
        let records = self.query_usage_records(filters)?;
        let statistics = self.get_statistics();

        Ok(UsageReport {
            total_records: records.len(),
            statistics,
            records: records.into_iter().take(100).collect(), // Limit for report
            generated_at: SystemTime::now(),
        })
    }
}

/// Usage report
#[derive(Debug, Clone, Serialize)]
pub struct UsageReport {
    pub total_records: usize,
    pub statistics: UsageStatistics,
    pub records: Vec<UsageRecord>,
    pub generated_at: SystemTime,
}

// Default implementations
impl Default for ResourceUsage {
    fn default() -> Self {
        ResourceUsage {
            total_processing_time_ms: 0,
            cpu_usage: CpuUsage::default(),
            memory_usage: MemoryUsage::default(),
            gpu_usage: None,
            network_usage: NetworkUsage::default(),
            storage_usage: StorageUsage::default(),
            cost_estimate: None,
        }
    }
}

impl Default for CpuUsage {
    fn default() -> Self {
        CpuUsage {
            peak_cpu_percent: 0.0,
            average_cpu_percent: 0.0,
            cpu_time_seconds: 0.0,
            cpu_cores_used: 1,
        }
    }
}

impl Default for MemoryUsage {
    fn default() -> Self {
        MemoryUsage {
            peak_memory_mb: 0.0,
            average_memory_mb: 0.0,
            memory_allocated_mb: 0.0,
            memory_freed_mb: 0.0,
        }
    }
}

impl Default for NetworkUsage {
    fn default() -> Self {
        NetworkUsage {
            bytes_uploaded: 0,
            bytes_downloaded: 0,
            requests_made: 0,
            bandwidth_peak_mbps: None,
        }
    }
}

impl Default for StorageUsage {
    fn default() -> Self {
        StorageUsage {
            temporary_storage_mb: 0.0,
            persistent_storage_mb: 0.0,
            cache_storage_mb: 0.0,
            files_created: 0,
            files_deleted: 0,
        }
    }
}

impl Default for SecurityContext {
    fn default() -> Self {
        SecurityContext {
            security_checks: Vec::new(),
            anomaly_detection: AnomalyDetectionResult {
                anomaly_score: 0.0,
                anomaly_threshold: 0.5,
                is_anomalous: false,
                anomaly_type: None,
                anomaly_details: String::new(),
            },
            rate_limiting: RateLimitingInfo {
                rate_limit_applied: false,
                current_usage: 0,
                rate_limit_threshold: 100,
                reset_time: None,
                quota_remaining: None,
            },
            threat_assessment: ThreatAssessment {
                threat_level: ThreatLevel::None,
                threat_indicators: Vec::new(),
                mitigation_actions: Vec::new(),
            },
            access_control: AccessControlInfo {
                permissions_granted: Vec::new(),
                permissions_denied: Vec::new(),
                access_level: AccessLevel::Basic,
                authentication_strength: AuthenticationStrength::Moderate,
            },
        }
    }
}

impl Default for QualityMetrics {
    fn default() -> Self {
        QualityMetrics {
            overall_quality: None,
            audio_quality: None,
            similarity_metrics: None,
            naturalness_score: None,
            intelligibility_score: None,
        }
    }
}

impl Default for LocationContext {
    fn default() -> Self {
        LocationContext {
            ip_address: None,
            country: None,
            region: None,
            city: None,
            timezone: None,
            isp: None,
            asn: None,
            is_vpn: None,
            is_proxy: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_usage_tracking_creation() {
        let config = UsageTrackingConfig::default();
        let tracker = UsageTracker::new(config);

        let user_context = UserContext {
            user_id: Some("test-user-001".to_string()),
            application_id: "test-app".to_string(),
            application_version: "1.0.0".to_string(),
            client_type: ClientType::API,
            session_id: Some("session-001".to_string()),
            request_id: Some("req-001".to_string()),
            auth_method: Some(AuthenticationMethod::APIKey),
            user_agent: None,
        };

        let operation = CloningOperation {
            operation_type: CloningOperationType::VoiceSynthesis,
            speaker_id: Some("test_speaker".to_string()),
            target_speaker_id: None,
            request_metadata: OperationRequestMetadata {
                request_id: format!("req-{}", uuid::Uuid::new_v4()),
                timestamp: SystemTime::now(),
                priority: Priority::Normal,
                source_application: "usage_tracker_test".to_string(),
                user_preferences: UserPreferences::default(),
            },
            input_data: InputDataInfo {
                data_type: InputDataType::TextPrompt,
                data_size_bytes: 1024,
                audio_duration_seconds: None,
                text_length: Some(100),
                language: Some("en".to_string()),
                content_hash: None,
                input_quality_score: None,
            },
            processing_params: ProcessingParameters {
                quality_level: QualityLevel::Standard,
                processing_mode: ProcessingMode::Balanced,
                model_config: ModelConfiguration {
                    model_name: "test-model".to_string(),
                    model_version: "1.0".to_string(),
                    model_type: ModelType::Acoustic,
                    model_size_mb: Some(100.0),
                    training_data_info: None,
                },
                advanced_params: HashMap::new(),
            },
            output_data: OutputDataInfo {
                output_type: OutputDataType::SynthesizedAudio,
                data_size_bytes: 0,
                audio_duration_seconds: None,
                quality_score: None,
                similarity_score: None,
                format: Some("wav".to_string()),
                sample_rate: Some(22050),
            },
            pipeline_info: PipelineInfo {
                pipeline_id: "pipeline-001".to_string(),
                pipeline_version: "1.0".to_string(),
                components_used: vec!["acoustic".to_string(), "vocoder".to_string()],
                processing_stages: Vec::new(),
            },
        };

        let usage_id = tracker.start_tracking(user_context, operation).unwrap();
        assert!(!usage_id.is_nil());

        let stats = tracker.get_statistics();
        // Initial stats should be defaults since we haven't completed any operations
        assert_eq!(stats.total_operations, 0);
    }

    #[test]
    fn test_consent_checking() {
        let config = UsageTrackingConfig::default();
        let tracker = UsageTracker::new(config);

        let user_context = UserContext {
            user_id: Some("test-user-consent".to_string()),
            application_id: "test-app".to_string(),
            application_version: "1.0.0".to_string(),
            client_type: ClientType::API,
            session_id: None,
            request_id: None,
            auth_method: None,
            user_agent: None,
        };

        let operation = CloningOperation {
            operation_type: CloningOperationType::VoiceSynthesis,
            speaker_id: Some("test_speaker".to_string()),
            target_speaker_id: None,
            request_metadata: OperationRequestMetadata {
                request_id: format!("req-{}", uuid::Uuid::new_v4()),
                timestamp: SystemTime::now(),
                priority: Priority::Normal,
                source_application: "usage_tracker_test".to_string(),
                user_preferences: UserPreferences::default(),
            },
            input_data: InputDataInfo {
                data_type: InputDataType::TextPrompt,
                data_size_bytes: 512,
                audio_duration_seconds: None,
                text_length: Some(50),
                language: Some("en".to_string()),
                content_hash: None,
                input_quality_score: None,
            },
            processing_params: ProcessingParameters {
                quality_level: QualityLevel::Standard,
                processing_mode: ProcessingMode::Balanced,
                model_config: ModelConfiguration {
                    model_name: "test-model".to_string(),
                    model_version: "1.0".to_string(),
                    model_type: ModelType::Acoustic,
                    model_size_mb: Some(100.0),
                    training_data_info: None,
                },
                advanced_params: HashMap::new(),
            },
            output_data: OutputDataInfo {
                output_type: OutputDataType::SynthesizedAudio,
                data_size_bytes: 0,
                audio_duration_seconds: None,
                quality_score: None,
                similarity_score: None,
                format: Some("wav".to_string()),
                sample_rate: Some(16000),
            },
            pipeline_info: PipelineInfo {
                pipeline_id: "test-pipeline".to_string(),
                pipeline_version: "1.0".to_string(),
                components_used: vec!["vocoder".to_string()],
                processing_stages: Vec::new(),
            },
        };

        let usage_id = tracker.start_tracking(user_context, operation).unwrap();

        // Test consent checking without consent manager (should allow by default)
        let result = tracker.check_consent(usage_id, None, "voice_synthesis");
        assert!(result.is_ok());

        match result.unwrap() {
            crate::consent::ConsentUsageResult::Allowed => assert!(true),
            _ => assert!(false, "Should allow by default when no consent manager"),
        }
    }

    #[test]
    fn test_usage_record_queries() {
        let config = UsageTrackingConfig::default();
        let tracker = UsageTracker::new(config);

        // Create multiple usage records
        for i in 0..5 {
            let user_context = UserContext {
                user_id: Some(format!("user-{i}")),
                application_id: format!("app-{}", i % 2),
                application_version: "1.0.0".to_string(),
                client_type: if i % 2 == 0 {
                    ClientType::API
                } else {
                    ClientType::WebBrowser
                },
                session_id: Some(format!("session-{i}")),
                request_id: Some(format!("req-{i}")),
                auth_method: Some(AuthenticationMethod::APIKey),
                user_agent: None,
            };

            let operation = CloningOperation {
                operation_type: if i % 2 == 0 {
                    CloningOperationType::VoiceSynthesis
                } else {
                    CloningOperationType::VoiceTraining
                },
                speaker_id: Some(format!("speaker_{i}")),
                target_speaker_id: None,
                request_metadata: OperationRequestMetadata {
                    request_id: format!("req-stress-{i}"),
                    timestamp: SystemTime::now(),
                    priority: Priority::Normal,
                    source_application: "stress_test".to_string(),
                    user_preferences: UserPreferences::default(),
                },
                input_data: InputDataInfo {
                    data_type: InputDataType::TextPrompt,
                    data_size_bytes: 1024 * (i + 1),
                    audio_duration_seconds: None,
                    text_length: Some(100),
                    language: Some("en".to_string()),
                    content_hash: None,
                    input_quality_score: None,
                },
                processing_params: ProcessingParameters {
                    quality_level: QualityLevel::Standard,
                    processing_mode: ProcessingMode::Balanced,
                    model_config: ModelConfiguration {
                        model_name: format!("model-{i}"),
                        model_version: "1.0".to_string(),
                        model_type: ModelType::Acoustic,
                        model_size_mb: Some(100.0),
                        training_data_info: None,
                    },
                    advanced_params: HashMap::new(),
                },
                output_data: OutputDataInfo {
                    output_type: OutputDataType::SynthesizedAudio,
                    data_size_bytes: 0,
                    audio_duration_seconds: None,
                    quality_score: None,
                    similarity_score: None,
                    format: Some("wav".to_string()),
                    sample_rate: Some(22050),
                },
                pipeline_info: PipelineInfo {
                    pipeline_id: format!("pipeline-{i}"),
                    pipeline_version: "1.0".to_string(),
                    components_used: vec!["vocoder".to_string()],
                    processing_stages: Vec::new(),
                },
            };

            let _usage_id = tracker.start_tracking(user_context, operation).unwrap();
        }

        // Query all records
        let filters = UsageQueryFilters {
            user_id: None,
            application_id: None,
            operation_type: None,
            start_time: None,
            end_time: None,
            status: None,
            limit: None,
        };
        let records = tracker.query_usage_records(&filters).unwrap();
        assert_eq!(records.len(), 5);

        // Query with user filter
        let filters = UsageQueryFilters {
            user_id: Some("user-0".to_string()),
            application_id: None,
            operation_type: None,
            start_time: None,
            end_time: None,
            status: None,
            limit: None,
        };
        let records = tracker.query_usage_records(&filters).unwrap();
        assert_eq!(records.len(), 1);

        // Query with application filter
        let filters = UsageQueryFilters {
            user_id: None,
            application_id: Some("app-0".to_string()),
            operation_type: None,
            start_time: None,
            end_time: None,
            status: None,
            limit: None,
        };
        let records = tracker.query_usage_records(&filters).unwrap();
        assert_eq!(records.len(), 3); // app-0 used for indices 0, 2, 4

        // Query with limit
        let filters = UsageQueryFilters {
            user_id: None,
            application_id: None,
            operation_type: None,
            start_time: None,
            end_time: None,
            status: None,
            limit: Some(3),
        };
        let records = tracker.query_usage_records(&filters).unwrap();
        assert_eq!(records.len(), 3);
    }

    #[test]
    fn test_usage_report_generation() {
        let config = UsageTrackingConfig::default();
        let tracker = UsageTracker::new(config);

        // Create a few records
        for i in 0..3 {
            let user_context = UserContext {
                user_id: Some(format!("report-user-{i}")),
                application_id: "report-app".to_string(),
                application_version: "1.0.0".to_string(),
                client_type: ClientType::API,
                session_id: None,
                request_id: None,
                auth_method: None,
                user_agent: None,
            };

            let operation = CloningOperation {
                operation_type: CloningOperationType::VoiceSynthesis,
                speaker_id: Some(format!("report_speaker_{i}")),
                target_speaker_id: None,
                request_metadata: OperationRequestMetadata {
                    request_id: format!("report_req_{i}"),
                    timestamp: SystemTime::now(),
                    priority: Priority::Normal,
                    source_application: "usage_report".to_string(),
                    user_preferences: UserPreferences::default(),
                },
                input_data: InputDataInfo {
                    data_type: InputDataType::TextPrompt,
                    data_size_bytes: 1000,
                    audio_duration_seconds: None,
                    text_length: Some(75),
                    language: Some("en".to_string()),
                    content_hash: None,
                    input_quality_score: None,
                },
                processing_params: ProcessingParameters {
                    quality_level: QualityLevel::Standard,
                    processing_mode: ProcessingMode::Balanced,
                    model_config: ModelConfiguration {
                        model_name: "report-model".to_string(),
                        model_version: "1.0".to_string(),
                        model_type: ModelType::Acoustic,
                        model_size_mb: Some(100.0),
                        training_data_info: None,
                    },
                    advanced_params: HashMap::new(),
                },
                output_data: OutputDataInfo {
                    output_type: OutputDataType::SynthesizedAudio,
                    data_size_bytes: 0,
                    audio_duration_seconds: None,
                    quality_score: None,
                    similarity_score: None,
                    format: Some("wav".to_string()),
                    sample_rate: Some(22050),
                },
                pipeline_info: PipelineInfo {
                    pipeline_id: "report-pipeline".to_string(),
                    pipeline_version: "1.0".to_string(),
                    components_used: vec!["vocoder".to_string()],
                    processing_stages: Vec::new(),
                },
            };

            let _usage_id = tracker.start_tracking(user_context, operation).unwrap();
        }

        // Generate report
        let filters = UsageQueryFilters {
            user_id: None,
            application_id: Some("report-app".to_string()),
            operation_type: None,
            start_time: None,
            end_time: None,
            status: None,
            limit: None,
        };

        let report = tracker.generate_usage_report(&filters).unwrap();
        assert_eq!(report.total_records, 3);
        assert_eq!(report.records.len(), 3);
        assert!(report.generated_at > UNIX_EPOCH);
    }

    #[test]
    fn test_error_handling() {
        let config = UsageTrackingConfig::default();
        let tracker = UsageTracker::new(config);

        // Test completing non-existent usage record
        let fake_usage_id = Uuid::new_v4();
        let outcome = UsageOutcome {
            status: UsageStatus::Failed,
            error: Some(UsageError {
                error_type: UsageErrorType::ProcessingError,
                error_code: "TEST_ERR_001".to_string(),
                error_message: "Test error".to_string(),
                retry_possible: false,
                retry_after: None,
            }),
            compliance_status: ComplianceStatus {
                is_compliant: false,
                compliance_checks: Vec::new(),
                violations: Vec::new(),
                risk_level: RiskLevel::High,
            },
            consent_result: None,
            restrictions_applied: Vec::new(),
            warnings: Vec::new(),
        };
        let resources = ResourceUsage::default();

        let result = tracker.complete_tracking(fake_usage_id, outcome, resources, None);
        assert!(result.is_err());

        if let Err(Error::Validation(msg)) = result {
            assert!(msg.contains("Usage record not found"));
        } else {
            panic!("Expected validation error");
        }

        // Test checking consent for non-existent usage record with consent manager
        let consent_manager = Arc::new(Mutex::new(crate::consent::ConsentManager::new()));
        let tracker_with_consent = {
            let mut t = UsageTracker::new(UsageTrackingConfig::default());
            t.set_consent_manager(consent_manager);
            t
        };
        let fake_consent_id = Uuid::new_v4();
        let result = tracker_with_consent.check_consent(
            fake_usage_id,
            Some(fake_consent_id),
            "test_use_case",
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_session_management() {
        let config = UsageTrackingConfig::default();
        let tracker = UsageTracker::new(config);

        let session_id = "test-session-123";

        // Create multiple operations within same session
        for i in 0..3 {
            let user_context = UserContext {
                user_id: Some("session-user".to_string()),
                application_id: "session-app".to_string(),
                application_version: "1.0.0".to_string(),
                client_type: ClientType::WebBrowser,
                session_id: Some(session_id.to_string()),
                request_id: Some(format!("req-{i}")),
                auth_method: Some(AuthenticationMethod::OAuth2),
                user_agent: Some("Test User Agent".to_string()),
            };

            let operation = CloningOperation {
                operation_type: CloningOperationType::VoiceSynthesis,
                speaker_id: Some(format!("report_speaker_{i}")),
                target_speaker_id: None,
                request_metadata: OperationRequestMetadata {
                    request_id: format!("report_req_{i}"),
                    timestamp: SystemTime::now(),
                    priority: Priority::Normal,
                    source_application: "usage_report".to_string(),
                    user_preferences: UserPreferences::default(),
                },
                input_data: InputDataInfo {
                    data_type: InputDataType::TextPrompt,
                    data_size_bytes: 500 + (i * 100),
                    audio_duration_seconds: None,
                    text_length: Some((50 + (i * 10)) as usize),
                    language: Some("en".to_string()),
                    content_hash: None,
                    input_quality_score: None,
                },
                processing_params: ProcessingParameters {
                    quality_level: QualityLevel::Standard,
                    processing_mode: ProcessingMode::Balanced,
                    model_config: ModelConfiguration {
                        model_name: "session-model".to_string(),
                        model_version: "1.0".to_string(),
                        model_type: ModelType::Acoustic,
                        model_size_mb: Some(100.0),
                        training_data_info: None,
                    },
                    advanced_params: HashMap::new(),
                },
                output_data: OutputDataInfo {
                    output_type: OutputDataType::SynthesizedAudio,
                    data_size_bytes: 0,
                    audio_duration_seconds: None,
                    quality_score: None,
                    similarity_score: None,
                    format: Some("wav".to_string()),
                    sample_rate: Some(22050),
                },
                pipeline_info: PipelineInfo {
                    pipeline_id: "session-pipeline".to_string(),
                    pipeline_version: "1.0".to_string(),
                    components_used: vec!["vocoder".to_string()],
                    processing_stages: Vec::new(),
                },
            };

            let _usage_id = tracker.start_tracking(user_context, operation).unwrap();
        }

        // Check that session was tracked
        let sessions = tracker.active_sessions.read().unwrap();
        assert!(sessions.contains_key(session_id));

        let session = sessions.get(session_id).unwrap();
        assert_eq!(session.request_count, 3);
        assert_eq!(session.current_operations.len(), 3);
        assert_eq!(session.user_id, Some("session-user".to_string()));
    }

    #[test]
    fn test_statistics_tracking() {
        let config = UsageTrackingConfig::default();
        let tracker = UsageTracker::new(config);

        // Initially, statistics should be empty
        let initial_stats = tracker.get_statistics();
        assert_eq!(initial_stats.total_operations, 0);
        assert_eq!(initial_stats.successful_operations, 0);
        assert_eq!(initial_stats.failed_operations, 0);

        // Create and complete a successful operation
        let user_context = UserContext {
            user_id: Some("stats-user".to_string()),
            application_id: "stats-app".to_string(),
            application_version: "1.0.0".to_string(),
            client_type: ClientType::API,
            session_id: None,
            request_id: None,
            auth_method: None,
            user_agent: None,
        };

        let operation = CloningOperation {
            operation_type: CloningOperationType::VoiceSynthesis,
            speaker_id: Some("test_speaker".to_string()),
            target_speaker_id: None,
            request_metadata: OperationRequestMetadata {
                request_id: format!("req-{}", uuid::Uuid::new_v4()),
                timestamp: SystemTime::now(),
                priority: Priority::Normal,
                source_application: "usage_tracker_test".to_string(),
                user_preferences: UserPreferences::default(),
            },
            input_data: InputDataInfo {
                data_type: InputDataType::TextPrompt,
                data_size_bytes: 1000,
                audio_duration_seconds: None,
                text_length: Some(100),
                language: Some("en".to_string()),
                content_hash: None,
                input_quality_score: None,
            },
            processing_params: ProcessingParameters {
                quality_level: QualityLevel::High,
                processing_mode: ProcessingMode::HighQuality,
                model_config: ModelConfiguration {
                    model_name: "stats-model".to_string(),
                    model_version: "1.0".to_string(),
                    model_type: ModelType::Acoustic,
                    model_size_mb: Some(200.0),
                    training_data_info: None,
                },
                advanced_params: HashMap::new(),
            },
            output_data: OutputDataInfo {
                output_type: OutputDataType::SynthesizedAudio,
                data_size_bytes: 0,
                audio_duration_seconds: None,
                quality_score: None,
                similarity_score: None,
                format: Some("wav".to_string()),
                sample_rate: Some(44100),
            },
            pipeline_info: PipelineInfo {
                pipeline_id: "stats-pipeline".to_string(),
                pipeline_version: "1.0".to_string(),
                components_used: vec!["preprocessing".to_string(), "vocoder".to_string()],
                processing_stages: Vec::new(),
            },
        };

        let usage_id = tracker.start_tracking(user_context, operation).unwrap();

        // Complete with success
        let outcome = UsageOutcome {
            status: UsageStatus::Success,
            error: None,
            compliance_status: ComplianceStatus {
                is_compliant: true,
                compliance_checks: vec![ComplianceCheck {
                    check_name: "consent".to_string(),
                    check_result: ComplianceCheckResult::Pass,
                    check_details: "Consent verified successfully".to_string(),
                }],
                violations: Vec::new(),
                risk_level: RiskLevel::Low,
            },
            consent_result: None,
            restrictions_applied: Vec::new(),
            warnings: Vec::new(),
        };

        let resources = ResourceUsage {
            total_processing_time_ms: 5000,
            cpu_usage: CpuUsage {
                peak_cpu_percent: 70.0,
                average_cpu_percent: 50.0,
                cpu_time_seconds: 4.5,
                cpu_cores_used: 2,
            },
            memory_usage: MemoryUsage {
                peak_memory_mb: 1024.0,
                average_memory_mb: 800.0,
                memory_allocated_mb: 1024.0,
                memory_freed_mb: 1024.0,
            },
            gpu_usage: None,
            network_usage: NetworkUsage::default(),
            storage_usage: StorageUsage::default(),
            cost_estimate: Some(CostBreakdown {
                compute_cost: 0.25,
                storage_cost: 0.05,
                network_cost: 0.02,
                total_cost: 0.32,
                currency: "USD".to_string(),
            }),
        };

        tracker
            .complete_tracking(usage_id, outcome, resources, None)
            .unwrap();

        // Check updated statistics
        let updated_stats = tracker.get_statistics();
        assert_eq!(updated_stats.total_operations, 1);
        assert_eq!(updated_stats.successful_operations, 1);
        assert_eq!(updated_stats.failed_operations, 0);
        assert_eq!(updated_stats.total_resource_cost, 0.32);
    }

    #[test]
    fn test_disabled_tracking() {
        let config = UsageTrackingConfig {
            enable_tracking: false,
            ..Default::default()
        };
        let tracker = UsageTracker::new(config);

        let user_context = UserContext {
            user_id: Some("disabled-user".to_string()),
            application_id: "disabled-app".to_string(),
            application_version: "1.0.0".to_string(),
            client_type: ClientType::API,
            session_id: None,
            request_id: None,
            auth_method: None,
            user_agent: None,
        };

        let operation = CloningOperation {
            operation_type: CloningOperationType::VoiceSynthesis,
            speaker_id: Some("test_speaker".to_string()),
            target_speaker_id: None,
            request_metadata: OperationRequestMetadata {
                request_id: format!("req-{}", uuid::Uuid::new_v4()),
                timestamp: SystemTime::now(),
                priority: Priority::Normal,
                source_application: "usage_tracker_test".to_string(),
                user_preferences: UserPreferences::default(),
            },
            input_data: InputDataInfo {
                data_type: InputDataType::TextPrompt,
                data_size_bytes: 500,
                audio_duration_seconds: None,
                text_length: Some(50),
                language: Some("en".to_string()),
                content_hash: None,
                input_quality_score: None,
            },
            processing_params: ProcessingParameters {
                quality_level: QualityLevel::Standard,
                processing_mode: ProcessingMode::Balanced,
                model_config: ModelConfiguration {
                    model_name: "disabled-model".to_string(),
                    model_version: "1.0".to_string(),
                    model_type: ModelType::Acoustic,
                    model_size_mb: Some(100.0),
                    training_data_info: None,
                },
                advanced_params: HashMap::new(),
            },
            output_data: OutputDataInfo {
                output_type: OutputDataType::SynthesizedAudio,
                data_size_bytes: 0,
                audio_duration_seconds: None,
                quality_score: None,
                similarity_score: None,
                format: Some("wav".to_string()),
                sample_rate: Some(22050),
            },
            pipeline_info: PipelineInfo {
                pipeline_id: "disabled-pipeline".to_string(),
                pipeline_version: "1.0".to_string(),
                components_used: vec!["vocoder".to_string()],
                processing_stages: Vec::new(),
            },
        };

        // Should return dummy ID when tracking is disabled
        let usage_id = tracker.start_tracking(user_context, operation).unwrap();
        assert!(!usage_id.is_nil()); // Returns valid UUID, but tracking is disabled

        // Statistics should remain at 0 since tracking is disabled
        let stats = tracker.get_statistics();
        assert_eq!(stats.total_operations, 0);
    }

    #[test]
    fn test_usage_completion() {
        let config = UsageTrackingConfig::default();
        let tracker = UsageTracker::new(config);

        let user_context = UserContext {
            user_id: Some("test-user-002".to_string()),
            application_id: "test-app".to_string(),
            application_version: "1.0.0".to_string(),
            client_type: ClientType::DesktopApp,
            session_id: None,
            request_id: None,
            auth_method: None,
            user_agent: None,
        };

        let operation = CloningOperation {
            operation_type: CloningOperationType::VoiceTraining,
            input_data: InputDataInfo {
                data_type: InputDataType::AudioFile,
                data_size_bytes: 1024000,
                audio_duration_seconds: Some(30.0),
                text_length: None,
                language: Some("en".to_string()),
                content_hash: Some("hash123".to_string()),
                input_quality_score: Some(0.85),
            },
            processing_params: ProcessingParameters {
                quality_level: QualityLevel::High,
                processing_mode: ProcessingMode::HighQuality,
                model_config: ModelConfiguration {
                    model_name: "advanced-model".to_string(),
                    model_version: "2.0".to_string(),
                    model_type: ModelType::Hybrid,
                    model_size_mb: Some(500.0),
                    training_data_info: Some("high-quality dataset".to_string()),
                },
                advanced_params: HashMap::new(),
            },
            output_data: OutputDataInfo {
                output_type: OutputDataType::VoiceModel,
                data_size_bytes: 50000000,
                audio_duration_seconds: None,
                quality_score: Some(0.92),
                similarity_score: Some(0.88),
                format: Some("model".to_string()),
                sample_rate: None,
            },
            pipeline_info: PipelineInfo {
                pipeline_id: "advanced-pipeline".to_string(),
                pipeline_version: "2.0".to_string(),
                components_used: vec!["preprocessing".to_string(), "training".to_string()],
                processing_stages: Vec::new(),
            },
            speaker_id: Some("test_speaker_advanced".to_string()),
            target_speaker_id: None,
            request_metadata: OperationRequestMetadata {
                request_id: "advanced_test_req".to_string(),
                timestamp: SystemTime::now(),
                priority: Priority::Normal,
                source_application: "usage_test".to_string(),
                user_preferences: UserPreferences::default(),
            },
        };

        let usage_id = tracker.start_tracking(user_context, operation).unwrap();

        let outcome = UsageOutcome {
            status: UsageStatus::Success,
            error: None,
            compliance_status: ComplianceStatus {
                is_compliant: true,
                compliance_checks: Vec::new(),
                violations: Vec::new(),
                risk_level: RiskLevel::Low,
            },
            consent_result: None,
            restrictions_applied: Vec::new(),
            warnings: Vec::new(),
        };

        let resources = ResourceUsage {
            total_processing_time_ms: 30000,
            cpu_usage: CpuUsage {
                peak_cpu_percent: 85.0,
                average_cpu_percent: 60.0,
                cpu_time_seconds: 25.0,
                cpu_cores_used: 4,
            },
            memory_usage: MemoryUsage {
                peak_memory_mb: 2048.0,
                average_memory_mb: 1500.0,
                memory_allocated_mb: 2048.0,
                memory_freed_mb: 2048.0,
            },
            gpu_usage: Some(GpuUsage {
                gpu_time_seconds: 20.0,
                peak_gpu_memory_mb: 4096.0,
                gpu_utilization_percent: 75.0,
                gpu_device_name: "RTX 4090".to_string(),
            }),
            network_usage: NetworkUsage::default(),
            storage_usage: StorageUsage::default(),
            cost_estimate: Some(CostBreakdown {
                compute_cost: 0.50,
                storage_cost: 0.10,
                network_cost: 0.05,
                total_cost: 0.65,
                currency: "USD".to_string(),
            }),
        };

        tracker
            .complete_tracking(usage_id, outcome, resources, None)
            .unwrap();

        let stats = tracker.get_statistics();
        assert_eq!(stats.total_operations, 1);
        assert_eq!(stats.successful_operations, 1);
    }
}

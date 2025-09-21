//! Enhanced error handling and recovery system for voice cloning operations
//!
//! This module provides comprehensive error handling with recovery strategies,
//! retry mechanisms, error classification, and graceful degradation capabilities.

use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error as StdError;
use std::fmt;
use std::time::{Duration, SystemTime};
use thiserror::Error;

/// Enhanced error context with recovery information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorContext {
    /// Original error
    pub error: String,
    /// Error classification
    pub classification: ErrorClassification,
    /// Severity level
    pub severity: ErrorSeverity,
    /// Recovery strategy
    pub recovery_strategy: RecoveryStrategy,
    /// Contextual information
    pub context: HashMap<String, String>,
    /// Timestamp when error occurred
    pub timestamp: SystemTime,
    /// Stack trace or error chain
    pub error_chain: Vec<String>,
    /// Recommended actions
    pub recommended_actions: Vec<String>,
    /// Retry information
    pub retry_info: Option<RetryInfo>,
}

/// Error classification for better handling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ErrorClassification {
    /// Transient errors that may resolve on retry
    Transient,
    /// Permanent errors requiring user intervention
    Permanent,
    /// Configuration-related errors
    Configuration,
    /// Resource-related errors (memory, disk, network)
    Resource,
    /// Data-related errors (invalid input, corruption)
    Data,
    /// System-related errors (hardware, OS)
    System,
    /// Security-related errors
    Security,
    /// Performance-related errors
    Performance,
    /// Unknown or unclassified errors
    Unknown,
}

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ErrorSeverity {
    /// Low impact, operation can continue
    Low,
    /// Medium impact, degraded functionality
    Medium,
    /// High impact, major functionality affected
    High,
    /// Critical impact, service unavailable
    Critical,
    /// Fatal impact, system failure
    Fatal,
}

/// Recovery strategies for different error types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RecoveryStrategy {
    /// Retry with exponential backoff
    RetryWithBackoff {
        max_attempts: u32,
        initial_delay: Duration,
        max_delay: Duration,
        backoff_factor: f64,
    },
    /// Fallback to alternative method
    Fallback {
        fallback_method: String,
        fallback_quality: f32,
    },
    /// Graceful degradation
    GracefulDegradation {
        reduced_functionality: Vec<String>,
        performance_impact: f32,
    },
    /// User intervention required
    UserIntervention {
        required_actions: Vec<String>,
        estimated_time: Duration,
    },
    /// Restart component or service
    Restart {
        component: String,
        data_preservation: bool,
    },
    /// No recovery possible
    NoRecovery,
}

/// Retry information and state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryInfo {
    /// Current attempt number
    pub attempt: u32,
    /// Maximum attempts allowed
    pub max_attempts: u32,
    /// Next retry delay
    pub next_delay: Duration,
    /// History of previous attempts
    pub attempt_history: Vec<RetryAttempt>,
    /// Whether retry is advisable
    pub should_retry: bool,
}

/// Information about a retry attempt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryAttempt {
    /// Attempt number
    pub attempt: u32,
    /// Timestamp of attempt
    pub timestamp: SystemTime,
    /// Error that occurred
    pub error: String,
    /// Duration of attempt
    pub duration: Duration,
}

/// Error recovery manager
#[derive(Debug)]
pub struct ErrorRecoveryManager {
    /// Configuration for recovery strategies
    config: RecoveryConfig,
    /// Active recovery operations
    active_recoveries: HashMap<String, RecoveryOperation>,
    /// Error statistics
    error_stats: ErrorStatistics,
    /// Recovery strategies cache
    strategy_cache: HashMap<String, RecoveryStrategy>,
}

/// Configuration for error recovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryConfig {
    /// Enable automatic recovery
    pub auto_recovery_enabled: bool,
    /// Maximum concurrent recovery operations
    pub max_concurrent_recoveries: usize,
    /// Default retry configuration
    pub default_retry_config: RetryConfig,
    /// Fallback quality threshold
    pub fallback_quality_threshold: f32,
    /// Enable graceful degradation
    pub graceful_degradation_enabled: bool,
    /// Error reporting configuration
    pub error_reporting: ErrorReportingConfig,
}

/// Retry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum retry attempts
    pub max_attempts: u32,
    /// Initial retry delay
    pub initial_delay: Duration,
    /// Maximum retry delay
    pub max_delay: Duration,
    /// Backoff factor for exponential backoff
    pub backoff_factor: f64,
    /// Jitter to avoid thundering herd
    pub jitter: bool,
    /// Timeout for each attempt
    pub attempt_timeout: Duration,
}

/// Error reporting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorReportingConfig {
    /// Enable error reporting
    pub enabled: bool,
    /// Report critical errors immediately
    pub immediate_critical_reporting: bool,
    /// Batch size for error reports
    pub batch_size: usize,
    /// Reporting interval
    pub reporting_interval: Duration,
    /// Include sensitive information
    pub include_sensitive_info: bool,
}

/// Active recovery operation
#[derive(Debug, Clone)]
pub struct RecoveryOperation {
    /// Operation ID
    pub id: String,
    /// Recovery strategy being executed
    pub strategy: RecoveryStrategy,
    /// Current state
    pub state: RecoveryState,
    /// Start time
    pub start_time: SystemTime,
    /// Progress information
    pub progress: RecoveryProgress,
}

/// Recovery operation state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecoveryState {
    /// Recovery is starting
    Starting,
    /// Recovery is in progress
    InProgress,
    /// Recovery is retrying
    Retrying,
    /// Recovery completed successfully
    Completed,
    /// Recovery failed
    Failed,
    /// Recovery was cancelled
    Cancelled,
}

/// Recovery progress information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryProgress {
    /// Percentage complete (0.0 to 1.0)
    pub percentage: f32,
    /// Current step description
    pub current_step: String,
    /// Estimated time remaining
    pub estimated_remaining: Option<Duration>,
    /// Recovery metrics
    pub metrics: RecoveryMetrics,
}

/// Recovery operation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryMetrics {
    /// Number of attempts made
    pub attempts_made: u32,
    /// Time spent on recovery
    pub time_spent: Duration,
    /// Success rate so far
    pub success_rate: f32,
    /// Resource usage during recovery
    pub resource_usage: ResourceUsage,
}

/// Resource usage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// Memory usage in bytes
    pub memory_bytes: usize,
    /// CPU usage percentage
    pub cpu_percentage: f32,
    /// Disk I/O bytes
    pub disk_io_bytes: usize,
    /// Network I/O bytes
    pub network_io_bytes: usize,
}

/// Error statistics and analytics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorStatistics {
    /// Total errors recorded
    pub total_errors: usize,
    /// Errors by classification
    pub errors_by_classification: HashMap<ErrorClassification, usize>,
    /// Errors by severity
    pub errors_by_severity: HashMap<ErrorSeverity, usize>,
    /// Recovery success rate
    pub recovery_success_rate: f32,
    /// Average recovery time
    pub average_recovery_time: Duration,
    /// Most common error patterns
    pub common_error_patterns: Vec<ErrorPattern>,
    /// Performance impact statistics
    pub performance_impact: PerformanceImpact,
}

/// Common error pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorPattern {
    /// Pattern description
    pub pattern: String,
    /// Frequency of occurrence
    pub frequency: usize,
    /// Associated recovery strategy
    pub recovery_strategy: RecoveryStrategy,
    /// Success rate with this pattern
    pub success_rate: f32,
}

/// Performance impact of errors and recovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceImpact {
    /// Average latency increase due to errors
    pub latency_increase_ms: f32,
    /// Throughput reduction percentage
    pub throughput_reduction: f32,
    /// Resource overhead from recovery operations
    pub recovery_overhead: f32,
    /// Quality degradation from fallbacks
    pub quality_degradation: f32,
}

/// Enhanced error result with recovery information
pub type RecoveryResult<T> = std::result::Result<T, RecoverableError>;

/// Recoverable error with context and recovery information
#[derive(Debug, Error)]
pub struct RecoverableError {
    /// Error context
    pub context: ErrorContext,
    /// Original error source
    #[source]
    pub source: Option<Box<dyn std::error::Error + Send + Sync>>,
}

impl fmt::Display for RecoverableError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} ({})",
            self.context.error,
            self.context.classification.as_str()
        )
    }
}

impl ErrorClassification {
    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            ErrorClassification::Transient => "transient",
            ErrorClassification::Permanent => "permanent",
            ErrorClassification::Configuration => "configuration",
            ErrorClassification::Resource => "resource",
            ErrorClassification::Data => "data",
            ErrorClassification::System => "system",
            ErrorClassification::Security => "security",
            ErrorClassification::Performance => "performance",
            ErrorClassification::Unknown => "unknown",
        }
    }

    /// Check if error is recoverable
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            ErrorClassification::Transient
                | ErrorClassification::Resource
                | ErrorClassification::Performance
        )
    }
}

impl ErrorSeverity {
    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            ErrorSeverity::Low => "low",
            ErrorSeverity::Medium => "medium",
            ErrorSeverity::High => "high",
            ErrorSeverity::Critical => "critical",
            ErrorSeverity::Fatal => "fatal",
        }
    }

    /// Check if error requires immediate attention
    pub fn requires_immediate_attention(&self) -> bool {
        matches!(self, ErrorSeverity::Critical | ErrorSeverity::Fatal)
    }
}

impl Default for RecoveryConfig {
    fn default() -> Self {
        Self {
            auto_recovery_enabled: true,
            max_concurrent_recoveries: 5,
            default_retry_config: RetryConfig::default(),
            fallback_quality_threshold: 0.7,
            graceful_degradation_enabled: true,
            error_reporting: ErrorReportingConfig::default(),
        }
    }
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(30),
            backoff_factor: 2.0,
            jitter: true,
            attempt_timeout: Duration::from_secs(60),
        }
    }
}

impl Default for ErrorReportingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            immediate_critical_reporting: true,
            batch_size: 10,
            reporting_interval: Duration::from_secs(300), // 5 minutes
            include_sensitive_info: false,
        }
    }
}

impl Default for ErrorStatistics {
    fn default() -> Self {
        Self {
            total_errors: 0,
            errors_by_classification: HashMap::new(),
            errors_by_severity: HashMap::new(),
            recovery_success_rate: 0.0,
            average_recovery_time: Duration::from_secs(0),
            common_error_patterns: Vec::new(),
            performance_impact: PerformanceImpact::default(),
        }
    }
}

impl Default for PerformanceImpact {
    fn default() -> Self {
        Self {
            latency_increase_ms: 0.0,
            throughput_reduction: 0.0,
            recovery_overhead: 0.0,
            quality_degradation: 0.0,
        }
    }
}

impl Default for ResourceUsage {
    fn default() -> Self {
        Self {
            memory_bytes: 0,
            cpu_percentage: 0.0,
            disk_io_bytes: 0,
            network_io_bytes: 0,
        }
    }
}

impl ErrorRecoveryManager {
    /// Create new error recovery manager
    pub fn new(config: RecoveryConfig) -> Self {
        Self {
            config,
            active_recoveries: HashMap::new(),
            error_stats: ErrorStatistics::default(),
            strategy_cache: HashMap::new(),
        }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(RecoveryConfig::default())
    }

    /// Handle an error with automatic recovery
    pub async fn handle_error(
        &mut self,
        error: Error,
        context: HashMap<String, String>,
    ) -> RecoveryResult<()> {
        let error_context = self.analyze_error(&error, context).await;

        // Update statistics
        self.update_error_statistics(&error_context);

        // Check if auto-recovery is enabled and error is recoverable
        if self.config.auto_recovery_enabled && error_context.classification.is_recoverable() {
            self.attempt_recovery(&error_context).await
        } else {
            Err(RecoverableError {
                context: error_context,
                source: Some(Box::new(error)),
            })
        }
    }

    /// Analyze error and create error context
    async fn analyze_error(&self, error: &Error, context: HashMap<String, String>) -> ErrorContext {
        let classification = self.classify_error(error);
        let severity = self.determine_severity(error, &classification);
        let recovery_strategy = self.determine_recovery_strategy(&classification, severity);

        let mut error_chain = vec![error.to_string()];
        let mut current = StdError::source(error);
        while let Some(source) = current {
            error_chain.push(source.to_string());
            current = StdError::source(source);
        }

        let recommended_actions = self.generate_recommended_actions(&classification, severity);

        ErrorContext {
            error: error.to_string(),
            classification,
            severity,
            recovery_strategy,
            context,
            timestamp: SystemTime::now(),
            error_chain,
            recommended_actions,
            retry_info: None,
        }
    }

    /// Classify error type
    fn classify_error(&self, error: &Error) -> ErrorClassification {
        match error {
            Error::Config(_) => ErrorClassification::Configuration,
            Error::Processing(_) => ErrorClassification::Transient,
            Error::Model(_) => ErrorClassification::Resource,
            Error::Audio(_) => ErrorClassification::Data,
            Error::Embedding(_) => ErrorClassification::Performance,
            Error::Verification(_) => ErrorClassification::Security,
            Error::Quality(_) => ErrorClassification::Performance,
            Error::InsufficientData(_) => ErrorClassification::Data,
            Error::Validation(_) => ErrorClassification::Data,
            Error::InvalidInput(_) => ErrorClassification::Data,
            Error::Consent(_) => ErrorClassification::Security,
            Error::Authentication(_) => ErrorClassification::Security,
            Error::UsageTracking(_) => ErrorClassification::System,
            Error::Ethics(_) => ErrorClassification::Security,
            Error::Io(_) => ErrorClassification::System,
            Error::Serialization(_) => ErrorClassification::Data,
            Error::Candle(_) => ErrorClassification::Resource,
        }
    }

    /// Determine error severity
    fn determine_severity(
        &self,
        error: &Error,
        classification: &ErrorClassification,
    ) -> ErrorSeverity {
        match (error, classification) {
            (Error::Ethics(_), _) => ErrorSeverity::Fatal,
            (Error::Consent(_), _) => ErrorSeverity::Critical,
            (_, ErrorClassification::Security) => ErrorSeverity::High,
            (Error::Config(_), _) => ErrorSeverity::Medium,
            (Error::InsufficientData(_), _) => ErrorSeverity::Medium,
            (_, ErrorClassification::Resource) => ErrorSeverity::High,
            (_, ErrorClassification::System) => ErrorSeverity::High,
            (_, ErrorClassification::Transient) => ErrorSeverity::Low,
            _ => ErrorSeverity::Medium,
        }
    }

    /// Determine appropriate recovery strategy
    fn determine_recovery_strategy(
        &self,
        classification: &ErrorClassification,
        severity: ErrorSeverity,
    ) -> RecoveryStrategy {
        match (classification, severity) {
            (ErrorClassification::Transient, ErrorSeverity::Low) => {
                RecoveryStrategy::RetryWithBackoff {
                    max_attempts: 3,
                    initial_delay: Duration::from_millis(100),
                    max_delay: Duration::from_secs(5),
                    backoff_factor: 2.0,
                }
            }
            (ErrorClassification::Resource, _) => RecoveryStrategy::GracefulDegradation {
                reduced_functionality: vec![
                    "reduce_quality".to_string(),
                    "limit_concurrent_operations".to_string(),
                ],
                performance_impact: 0.3,
            },
            (ErrorClassification::Performance, _) => RecoveryStrategy::Fallback {
                fallback_method: "cpu_fallback".to_string(),
                fallback_quality: 0.8,
            },
            (ErrorClassification::Configuration, _) => RecoveryStrategy::UserIntervention {
                required_actions: vec![
                    "check_configuration".to_string(),
                    "validate_settings".to_string(),
                ],
                estimated_time: from_minutes(5),
            },
            (ErrorClassification::Security, ErrorSeverity::Critical | ErrorSeverity::Fatal) => {
                RecoveryStrategy::Restart {
                    component: "security_module".to_string(),
                    data_preservation: false,
                }
            }
            _ => RecoveryStrategy::NoRecovery,
        }
    }

    /// Generate recommended actions for error resolution
    fn generate_recommended_actions(
        &self,
        classification: &ErrorClassification,
        severity: ErrorSeverity,
    ) -> Vec<String> {
        let mut actions = Vec::new();

        match classification {
            ErrorClassification::Configuration => {
                actions.push("Review configuration settings".to_string());
                actions.push("Validate configuration against schema".to_string());
                actions.push("Check for missing required parameters".to_string());
            }
            ErrorClassification::Resource => {
                actions.push("Monitor system resources".to_string());
                actions.push("Consider scaling resources".to_string());
                actions.push("Implement resource pooling".to_string());
            }
            ErrorClassification::Data => {
                actions.push("Validate input data format".to_string());
                actions.push("Check data integrity".to_string());
                actions.push("Verify data source reliability".to_string());
            }
            ErrorClassification::Security => {
                actions.push("Review security policies".to_string());
                actions.push("Check authentication credentials".to_string());
                actions.push("Audit access permissions".to_string());
            }
            ErrorClassification::Performance => {
                actions.push("Profile system performance".to_string());
                actions.push("Optimize critical code paths".to_string());
                actions.push("Consider caching strategies".to_string());
            }
            _ => {
                actions.push("Review error logs for patterns".to_string());
                actions.push("Contact support if issue persists".to_string());
            }
        }

        if severity.requires_immediate_attention() {
            actions.insert(0, "Escalate to on-call support immediately".to_string());
        }

        actions
    }

    /// Attempt automatic recovery
    async fn attempt_recovery(&mut self, error_context: &ErrorContext) -> RecoveryResult<()> {
        let recovery_id = format!("recovery_{}", uuid::Uuid::new_v4());

        let recovery_op = RecoveryOperation {
            id: recovery_id.clone(),
            strategy: error_context.recovery_strategy.clone(),
            state: RecoveryState::Starting,
            start_time: SystemTime::now(),
            progress: RecoveryProgress {
                percentage: 0.0,
                current_step: "Initializing recovery".to_string(),
                estimated_remaining: Some(Duration::from_secs(30)),
                metrics: RecoveryMetrics {
                    attempts_made: 0,
                    time_spent: Duration::from_secs(0),
                    success_rate: 0.0,
                    resource_usage: ResourceUsage::default(),
                },
            },
        };

        self.active_recoveries
            .insert(recovery_id.clone(), recovery_op);

        match &error_context.recovery_strategy {
            RecoveryStrategy::RetryWithBackoff {
                max_attempts,
                initial_delay,
                max_delay,
                backoff_factor,
            } => {
                self.execute_retry_recovery(
                    &recovery_id,
                    *max_attempts,
                    *initial_delay,
                    *max_delay,
                    *backoff_factor,
                )
                .await
            }
            RecoveryStrategy::Fallback {
                fallback_method,
                fallback_quality,
            } => {
                self.execute_fallback_recovery(&recovery_id, fallback_method, *fallback_quality)
                    .await
            }
            RecoveryStrategy::GracefulDegradation {
                reduced_functionality,
                performance_impact,
            } => {
                self.execute_graceful_degradation(
                    &recovery_id,
                    reduced_functionality,
                    *performance_impact,
                )
                .await
            }
            RecoveryStrategy::Restart {
                component,
                data_preservation,
            } => {
                self.execute_restart_recovery(&recovery_id, component, *data_preservation)
                    .await
            }
            _ => {
                self.update_recovery_state(&recovery_id, RecoveryState::Failed);
                Err(RecoverableError {
                    context: error_context.clone(),
                    source: None,
                })
            }
        }
    }

    /// Execute retry-based recovery
    async fn execute_retry_recovery(
        &mut self,
        recovery_id: &str,
        max_attempts: u32,
        initial_delay: Duration,
        max_delay: Duration,
        backoff_factor: f64,
    ) -> RecoveryResult<()> {
        self.update_recovery_state(recovery_id, RecoveryState::InProgress);

        for attempt in 1..=max_attempts {
            self.update_recovery_progress(recovery_id, |progress| {
                progress.percentage = (attempt as f32 / max_attempts as f32) * 0.8;
                progress.current_step = format!("Retry attempt {} of {}", attempt, max_attempts);
                progress.metrics.attempts_made = attempt;
            });

            // Simulate retry operation
            tokio::time::sleep(Duration::from_millis(100)).await;

            // In a real implementation, this would retry the actual operation
            if attempt == max_attempts / 2 + 1 {
                // Simulate success on a middle attempt
                self.update_recovery_state(recovery_id, RecoveryState::Completed);
                self.update_recovery_progress(recovery_id, |progress| {
                    progress.percentage = 1.0;
                    progress.current_step = "Recovery completed successfully".to_string();
                    progress.estimated_remaining = None;
                });
                return Ok(());
            }

            if attempt < max_attempts {
                let delay =
                    self.calculate_backoff_delay(attempt, initial_delay, max_delay, backoff_factor);
                tokio::time::sleep(delay).await;
            }
        }

        self.update_recovery_state(recovery_id, RecoveryState::Failed);
        Err(RecoverableError {
            context: ErrorContext {
                error: "Retry recovery failed after maximum attempts".to_string(),
                classification: ErrorClassification::Permanent,
                severity: ErrorSeverity::High,
                recovery_strategy: RecoveryStrategy::NoRecovery,
                context: HashMap::new(),
                timestamp: SystemTime::now(),
                error_chain: vec!["Retry recovery exhausted".to_string()],
                recommended_actions: vec!["Manual intervention required".to_string()],
                retry_info: None,
            },
            source: None,
        })
    }

    /// Execute fallback-based recovery
    async fn execute_fallback_recovery(
        &mut self,
        recovery_id: &str,
        fallback_method: &str,
        fallback_quality: f32,
    ) -> RecoveryResult<()> {
        self.update_recovery_state(recovery_id, RecoveryState::InProgress);

        self.update_recovery_progress(recovery_id, |progress| {
            progress.percentage = 0.5;
            progress.current_step = format!("Switching to fallback method: {}", fallback_method);
        });

        // Simulate fallback implementation
        tokio::time::sleep(Duration::from_millis(200)).await;

        if fallback_quality >= self.config.fallback_quality_threshold {
            self.update_recovery_state(recovery_id, RecoveryState::Completed);
            self.update_recovery_progress(recovery_id, |progress| {
                progress.percentage = 1.0;
                progress.current_step = "Fallback recovery completed".to_string();
                progress.estimated_remaining = None;
            });
            Ok(())
        } else {
            self.update_recovery_state(recovery_id, RecoveryState::Failed);
            Err(RecoverableError {
                context: ErrorContext {
                    error: "Fallback quality below threshold".to_string(),
                    classification: ErrorClassification::Performance,
                    severity: ErrorSeverity::Medium,
                    recovery_strategy: RecoveryStrategy::NoRecovery,
                    context: HashMap::new(),
                    timestamp: SystemTime::now(),
                    error_chain: vec!["Quality degradation too severe".to_string()],
                    recommended_actions: vec!["Consider alternative fallback methods".to_string()],
                    retry_info: None,
                },
                source: None,
            })
        }
    }

    /// Execute graceful degradation recovery
    async fn execute_graceful_degradation(
        &mut self,
        recovery_id: &str,
        reduced_functionality: &[String],
        performance_impact: f32,
    ) -> RecoveryResult<()> {
        self.update_recovery_state(recovery_id, RecoveryState::InProgress);

        for (i, functionality) in reduced_functionality.iter().enumerate() {
            self.update_recovery_progress(recovery_id, |progress| {
                progress.percentage = (i as f32 / reduced_functionality.len() as f32) * 0.9;
                progress.current_step = format!("Reducing functionality: {}", functionality);
            });

            // Simulate functionality reduction
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        self.update_recovery_state(recovery_id, RecoveryState::Completed);
        self.update_recovery_progress(recovery_id, |progress| {
            progress.percentage = 1.0;
            progress.current_step = "Graceful degradation completed".to_string();
            progress.estimated_remaining = None;
        });

        Ok(())
    }

    /// Execute restart-based recovery
    async fn execute_restart_recovery(
        &mut self,
        recovery_id: &str,
        component: &str,
        data_preservation: bool,
    ) -> RecoveryResult<()> {
        self.update_recovery_state(recovery_id, RecoveryState::InProgress);

        if data_preservation {
            self.update_recovery_progress(recovery_id, |progress| {
                progress.percentage = 0.2;
                progress.current_step = "Preserving component data".to_string();
            });
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        self.update_recovery_progress(recovery_id, |progress| {
            progress.percentage = 0.5;
            progress.current_step = format!("Restarting component: {}", component);
        });
        tokio::time::sleep(Duration::from_millis(200)).await;

        if data_preservation {
            self.update_recovery_progress(recovery_id, |progress| {
                progress.percentage = 0.8;
                progress.current_step = "Restoring component data".to_string();
            });
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        self.update_recovery_state(recovery_id, RecoveryState::Completed);
        self.update_recovery_progress(recovery_id, |progress| {
            progress.percentage = 1.0;
            progress.current_step = "Component restart completed".to_string();
            progress.estimated_remaining = None;
        });

        Ok(())
    }

    /// Calculate backoff delay with jitter
    fn calculate_backoff_delay(
        &self,
        attempt: u32,
        initial_delay: Duration,
        max_delay: Duration,
        backoff_factor: f64,
    ) -> Duration {
        let base_delay = initial_delay.as_millis() as f64 * backoff_factor.powi(attempt as i32 - 1);
        let delay = Duration::from_millis(base_delay as u64).min(max_delay);

        if self.config.default_retry_config.jitter {
            let jitter_range = delay.as_millis() as f64 * 0.1;
            let jitter = fastrand::f64() * jitter_range - (jitter_range / 2.0);
            Duration::from_millis((delay.as_millis() as f64 + jitter).max(0.0) as u64)
        } else {
            delay
        }
    }

    /// Update recovery operation state
    fn update_recovery_state(&mut self, recovery_id: &str, state: RecoveryState) {
        if let Some(recovery) = self.active_recoveries.get_mut(recovery_id) {
            recovery.state = state;
            if matches!(
                state,
                RecoveryState::Completed | RecoveryState::Failed | RecoveryState::Cancelled
            ) {
                recovery.progress.metrics.time_spent = SystemTime::now()
                    .duration_since(recovery.start_time)
                    .unwrap_or_default();
            }
        }
    }

    /// Update recovery progress
    fn update_recovery_progress<F>(&mut self, recovery_id: &str, updater: F)
    where
        F: FnOnce(&mut RecoveryProgress),
    {
        if let Some(recovery) = self.active_recoveries.get_mut(recovery_id) {
            updater(&mut recovery.progress);
        }
    }

    /// Update error statistics
    fn update_error_statistics(&mut self, error_context: &ErrorContext) {
        self.error_stats.total_errors += 1;

        *self
            .error_stats
            .errors_by_classification
            .entry(error_context.classification)
            .or_insert(0) += 1;
        *self
            .error_stats
            .errors_by_severity
            .entry(error_context.severity)
            .or_insert(0) += 1;
    }

    /// Get current error statistics
    pub fn get_error_statistics(&self) -> &ErrorStatistics {
        &self.error_stats
    }

    /// Get active recovery operations
    pub fn get_active_recoveries(&self) -> &HashMap<String, RecoveryOperation> {
        &self.active_recoveries
    }

    /// Cancel recovery operation
    pub fn cancel_recovery(&mut self, recovery_id: &str) -> Result<()> {
        if let Some(recovery) = self.active_recoveries.get_mut(recovery_id) {
            recovery.state = RecoveryState::Cancelled;
            Ok(())
        } else {
            Err(Error::Processing(format!(
                "Recovery operation not found: {}",
                recovery_id
            )))
        }
    }

    /// Clean up completed recovery operations
    pub fn cleanup_completed_recoveries(&mut self) {
        self.active_recoveries.retain(|_, recovery| {
            !matches!(
                recovery.state,
                RecoveryState::Completed | RecoveryState::Failed | RecoveryState::Cancelled
            )
        });
    }

    /// Generate error report
    pub fn generate_error_report(&self) -> ErrorReport {
        ErrorReport {
            timestamp: SystemTime::now(),
            total_errors: self.error_stats.total_errors,
            error_breakdown: self.error_stats.errors_by_classification.clone(),
            severity_breakdown: self.error_stats.errors_by_severity.clone(),
            active_recoveries: self.active_recoveries.len(),
            recovery_success_rate: self.error_stats.recovery_success_rate,
            performance_impact: self.error_stats.performance_impact.clone(),
            recommendations: self.generate_system_recommendations(),
        }
    }

    /// Generate system-wide recommendations based on error patterns
    fn generate_system_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        if self.error_stats.total_errors > 100 {
            recommendations
                .push("Consider implementing more robust error prevention mechanisms".to_string());
        }

        if let Some(config_errors) = self
            .error_stats
            .errors_by_classification
            .get(&ErrorClassification::Configuration)
        {
            if *config_errors > 10 {
                recommendations.push("Review and improve configuration validation".to_string());
            }
        }

        if let Some(resource_errors) = self
            .error_stats
            .errors_by_classification
            .get(&ErrorClassification::Resource)
        {
            if *resource_errors > 5 {
                recommendations.push("Consider scaling system resources".to_string());
            }
        }

        if self.error_stats.performance_impact.latency_increase_ms > 100.0 {
            recommendations
                .push("Optimize error handling paths to reduce latency impact".to_string());
        }

        if recommendations.is_empty() {
            recommendations
                .push("System error handling is operating within normal parameters".to_string());
        }

        recommendations
    }
}

/// Error report for monitoring and analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorReport {
    /// Report timestamp
    pub timestamp: SystemTime,
    /// Total number of errors
    pub total_errors: usize,
    /// Breakdown by error classification
    pub error_breakdown: HashMap<ErrorClassification, usize>,
    /// Breakdown by severity
    pub severity_breakdown: HashMap<ErrorSeverity, usize>,
    /// Number of active recovery operations
    pub active_recoveries: usize,
    /// Overall recovery success rate
    pub recovery_success_rate: f32,
    /// Performance impact metrics
    pub performance_impact: PerformanceImpact,
    /// System recommendations
    pub recommendations: Vec<String>,
}

// Helper functions
/// Convert minutes to Duration
fn from_minutes(minutes: u64) -> Duration {
    Duration::from_secs(minutes * 60)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_error_classification() {
        assert_eq!(ErrorClassification::Transient.as_str(), "transient");
        assert_eq!(ErrorClassification::Configuration.as_str(), "configuration");
        assert_eq!(ErrorClassification::Security.as_str(), "security");

        assert!(ErrorClassification::Transient.is_recoverable());
        assert!(ErrorClassification::Resource.is_recoverable());
        assert!(!ErrorClassification::Security.is_recoverable());
        assert!(!ErrorClassification::Permanent.is_recoverable());
    }

    #[test]
    fn test_error_severity() {
        assert_eq!(ErrorSeverity::Low.as_str(), "low");
        assert_eq!(ErrorSeverity::Critical.as_str(), "critical");
        assert_eq!(ErrorSeverity::Fatal.as_str(), "fatal");

        assert!(!ErrorSeverity::Low.requires_immediate_attention());
        assert!(!ErrorSeverity::Medium.requires_immediate_attention());
        assert!(!ErrorSeverity::High.requires_immediate_attention());
        assert!(ErrorSeverity::Critical.requires_immediate_attention());
        assert!(ErrorSeverity::Fatal.requires_immediate_attention());

        // Test ordering
        assert!(ErrorSeverity::Low < ErrorSeverity::Medium);
        assert!(ErrorSeverity::Medium < ErrorSeverity::High);
        assert!(ErrorSeverity::High < ErrorSeverity::Critical);
        assert!(ErrorSeverity::Critical < ErrorSeverity::Fatal);
    }

    #[test]
    fn test_recovery_config_default() {
        let config = RecoveryConfig::default();
        assert!(config.auto_recovery_enabled);
        assert_eq!(config.max_concurrent_recoveries, 5);
        assert_eq!(config.fallback_quality_threshold, 0.7);
        assert!(config.graceful_degradation_enabled);
        assert!(config.error_reporting.enabled);
    }

    #[test]
    fn test_retry_config_default() {
        let config = RetryConfig::default();
        assert_eq!(config.max_attempts, 3);
        assert_eq!(config.initial_delay, Duration::from_millis(100));
        assert_eq!(config.max_delay, Duration::from_secs(30));
        assert_eq!(config.backoff_factor, 2.0);
        assert!(config.jitter);
        assert_eq!(config.attempt_timeout, Duration::from_secs(60));
    }

    #[test]
    fn test_error_recovery_manager_creation() {
        let config = RecoveryConfig::default();
        let manager = ErrorRecoveryManager::new(config);

        assert_eq!(manager.active_recoveries.len(), 0);
        assert_eq!(manager.error_stats.total_errors, 0);
        assert!(manager.strategy_cache.is_empty());
    }

    #[test]
    fn test_error_classification_mapping() {
        let manager = ErrorRecoveryManager::default();

        assert_eq!(
            manager.classify_error(&Error::Config("test".to_string())),
            ErrorClassification::Configuration
        );
        assert_eq!(
            manager.classify_error(&Error::Processing("test".to_string())),
            ErrorClassification::Transient
        );
        assert_eq!(
            manager.classify_error(&Error::Ethics("test".to_string())),
            ErrorClassification::Security
        );
        assert_eq!(
            manager.classify_error(&Error::Candle(candle_core::Error::Cuda(
                "test".to_string().into()
            ))),
            ErrorClassification::Resource
        );
    }

    #[test]
    fn test_error_severity_determination() {
        let manager = ErrorRecoveryManager::default();

        // Ethics errors should be fatal
        assert_eq!(
            manager.determine_severity(
                &Error::Ethics("test".to_string()),
                &ErrorClassification::Security
            ),
            ErrorSeverity::Fatal
        );

        // Consent errors should be critical
        assert_eq!(
            manager.determine_severity(
                &Error::Consent("test".to_string()),
                &ErrorClassification::Security
            ),
            ErrorSeverity::Critical
        );

        // Transient errors should be low severity
        assert_eq!(
            manager.determine_severity(
                &Error::Processing("test".to_string()),
                &ErrorClassification::Transient
            ),
            ErrorSeverity::Low
        );

        // Resource errors should be high severity
        assert_eq!(
            manager.determine_severity(
                &Error::Model("test".to_string()),
                &ErrorClassification::Resource
            ),
            ErrorSeverity::High
        );
    }

    #[test]
    fn test_recovery_strategy_determination() {
        let manager = ErrorRecoveryManager::default();

        // Transient low severity should use retry with backoff
        let strategy = manager
            .determine_recovery_strategy(&ErrorClassification::Transient, ErrorSeverity::Low);
        assert!(matches!(
            strategy,
            RecoveryStrategy::RetryWithBackoff { .. }
        ));

        // Resource errors should use graceful degradation
        let strategy = manager
            .determine_recovery_strategy(&ErrorClassification::Resource, ErrorSeverity::High);
        assert!(matches!(
            strategy,
            RecoveryStrategy::GracefulDegradation { .. }
        ));

        // Performance errors should use fallback
        let strategy = manager
            .determine_recovery_strategy(&ErrorClassification::Performance, ErrorSeverity::Medium);
        assert!(matches!(strategy, RecoveryStrategy::Fallback { .. }));

        // Configuration errors should require user intervention
        let strategy = manager.determine_recovery_strategy(
            &ErrorClassification::Configuration,
            ErrorSeverity::Medium,
        );
        assert!(matches!(
            strategy,
            RecoveryStrategy::UserIntervention { .. }
        ));

        // Critical security errors should restart components
        let strategy = manager
            .determine_recovery_strategy(&ErrorClassification::Security, ErrorSeverity::Critical);
        assert!(matches!(strategy, RecoveryStrategy::Restart { .. }));
    }

    #[test]
    fn test_recommended_actions_generation() {
        let manager = ErrorRecoveryManager::default();

        let actions = manager.generate_recommended_actions(
            &ErrorClassification::Configuration,
            ErrorSeverity::Medium,
        );
        assert!(!actions.is_empty());
        assert!(actions
            .iter()
            .any(|action| action.contains("configuration")));

        let actions = manager
            .generate_recommended_actions(&ErrorClassification::Resource, ErrorSeverity::High);
        assert!(actions.iter().any(|action| action.contains("resources")));

        let actions = manager
            .generate_recommended_actions(&ErrorClassification::Security, ErrorSeverity::Critical);
        assert!(actions.iter().any(|action| action.contains("Escalate")));
    }

    #[tokio::test]
    async fn test_error_analysis() {
        let manager = ErrorRecoveryManager::default();
        let error = Error::Processing("Test processing error".to_string());
        let context = HashMap::from([("operation".to_string(), "test_op".to_string())]);

        let error_context = manager.analyze_error(&error, context).await;

        assert_eq!(error_context.classification, ErrorClassification::Transient);
        assert_eq!(error_context.severity, ErrorSeverity::Low);
        assert!(matches!(
            error_context.recovery_strategy,
            RecoveryStrategy::RetryWithBackoff { .. }
        ));
        assert!(!error_context.error_chain.is_empty());
        assert!(!error_context.recommended_actions.is_empty());
        assert!(error_context.context.contains_key("operation"));
    }

    #[tokio::test]
    async fn test_handle_recoverable_error() {
        let mut manager = ErrorRecoveryManager::default();
        let error = Error::Processing("Transient error".to_string());
        let context = HashMap::new();

        let result = manager.handle_error(error, context).await;

        // Should succeed due to simulated recovery
        assert!(result.is_ok());
        assert_eq!(manager.error_stats.total_errors, 1);
        assert_eq!(
            manager
                .error_stats
                .errors_by_classification
                .get(&ErrorClassification::Transient),
            Some(&1)
        );
    }

    #[tokio::test]
    async fn test_handle_non_recoverable_error() {
        let mut manager = ErrorRecoveryManager::default();
        let error = Error::Ethics("Ethics violation".to_string());
        let context = HashMap::new();

        let result = manager.handle_error(error, context).await;

        // Should fail as ethics errors are not auto-recoverable
        assert!(result.is_err());
        assert_eq!(manager.error_stats.total_errors, 1);

        if let Err(recoverable_error) = result {
            assert_eq!(
                recoverable_error.context.classification,
                ErrorClassification::Security
            );
            assert_eq!(recoverable_error.context.severity, ErrorSeverity::Fatal);
        }
    }

    #[test]
    fn test_backoff_delay_calculation() {
        let config = RecoveryConfig {
            default_retry_config: RetryConfig {
                jitter: false,
                ..RetryConfig::default()
            },
            ..RecoveryConfig::default()
        };
        let manager = ErrorRecoveryManager::new(config);

        let initial_delay = Duration::from_millis(100);
        let max_delay = Duration::from_secs(10);
        let backoff_factor = 2.0;

        // Test exponential backoff
        let delay1 = manager.calculate_backoff_delay(1, initial_delay, max_delay, backoff_factor);
        let delay2 = manager.calculate_backoff_delay(2, initial_delay, max_delay, backoff_factor);
        let delay3 = manager.calculate_backoff_delay(3, initial_delay, max_delay, backoff_factor);

        assert_eq!(delay1, Duration::from_millis(100));
        assert_eq!(delay2, Duration::from_millis(200));
        assert_eq!(delay3, Duration::from_millis(400));

        // Test max delay capping
        let delay_large =
            manager.calculate_backoff_delay(10, initial_delay, max_delay, backoff_factor);
        assert_eq!(delay_large, max_delay);
    }

    #[test]
    fn test_backoff_delay_with_jitter() {
        let manager = ErrorRecoveryManager::default(); // Jitter enabled by default

        let initial_delay = Duration::from_millis(100);
        let max_delay = Duration::from_secs(10);
        let backoff_factor = 2.0;

        // Test that jitter produces different results
        let delay1 = manager.calculate_backoff_delay(2, initial_delay, max_delay, backoff_factor);
        let delay2 = manager.calculate_backoff_delay(2, initial_delay, max_delay, backoff_factor);

        // With jitter, delays should be within reasonable range but may vary
        let expected_base = Duration::from_millis(200);
        let variance = Duration::from_millis(20); // 10% jitter

        assert!(delay1 >= expected_base - variance);
        assert!(delay1 <= expected_base + variance);
        assert!(delay2 >= expected_base - variance);
        assert!(delay2 <= expected_base + variance);
    }

    #[test]
    fn test_error_statistics_update() {
        let mut manager = ErrorRecoveryManager::default();

        let context1 = ErrorContext {
            error: "Error 1".to_string(),
            classification: ErrorClassification::Transient,
            severity: ErrorSeverity::Low,
            recovery_strategy: RecoveryStrategy::NoRecovery,
            context: HashMap::new(),
            timestamp: SystemTime::now(),
            error_chain: vec![],
            recommended_actions: vec![],
            retry_info: None,
        };

        let context2 = ErrorContext {
            error: "Error 2".to_string(),
            classification: ErrorClassification::Resource,
            severity: ErrorSeverity::High,
            recovery_strategy: RecoveryStrategy::NoRecovery,
            context: HashMap::new(),
            timestamp: SystemTime::now(),
            error_chain: vec![],
            recommended_actions: vec![],
            retry_info: None,
        };

        manager.update_error_statistics(&context1);
        manager.update_error_statistics(&context2);
        manager.update_error_statistics(&context1); // Duplicate classification

        let stats = manager.get_error_statistics();
        assert_eq!(stats.total_errors, 3);
        assert_eq!(
            stats
                .errors_by_classification
                .get(&ErrorClassification::Transient),
            Some(&2)
        );
        assert_eq!(
            stats
                .errors_by_classification
                .get(&ErrorClassification::Resource),
            Some(&1)
        );
        assert_eq!(stats.errors_by_severity.get(&ErrorSeverity::Low), Some(&2));
        assert_eq!(stats.errors_by_severity.get(&ErrorSeverity::High), Some(&1));
    }

    #[test]
    fn test_error_report_generation() {
        let mut manager = ErrorRecoveryManager::default();

        // Add some test errors
        let context = ErrorContext {
            error: "Test error".to_string(),
            classification: ErrorClassification::Configuration,
            severity: ErrorSeverity::Medium,
            recovery_strategy: RecoveryStrategy::NoRecovery,
            context: HashMap::new(),
            timestamp: SystemTime::now(),
            error_chain: vec![],
            recommended_actions: vec![],
            retry_info: None,
        };

        for _ in 0..12 {
            manager.update_error_statistics(&context);
        }

        let report = manager.generate_error_report();

        assert_eq!(report.total_errors, 12);
        assert_eq!(
            report
                .error_breakdown
                .get(&ErrorClassification::Configuration),
            Some(&12)
        );
        assert!(!report.recommendations.is_empty());
        assert!(report
            .recommendations
            .iter()
            .any(|r| r.contains("configuration")));
    }

    #[test]
    fn test_recovery_operation_management() {
        let mut manager = ErrorRecoveryManager::default();

        // Test that initially there are no active recoveries
        assert_eq!(manager.get_active_recoveries().len(), 0);

        // Test cancel non-existent recovery
        let result = manager.cancel_recovery("non_existent");
        assert!(result.is_err());

        // Test cleanup with no recoveries
        manager.cleanup_completed_recoveries();
        assert_eq!(manager.get_active_recoveries().len(), 0);
    }

    #[test]
    fn test_system_recommendations() {
        let mut manager = ErrorRecoveryManager::default();

        // Test with no errors
        let recommendations = manager.generate_system_recommendations();
        assert!(recommendations
            .iter()
            .any(|r| r.contains("normal parameters")));

        // Test with many errors
        manager.error_stats.total_errors = 150;
        let recommendations = manager.generate_system_recommendations();
        assert!(recommendations
            .iter()
            .any(|r| r.contains("error prevention")));

        // Test with high latency impact
        manager.error_stats.performance_impact.latency_increase_ms = 150.0;
        let recommendations = manager.generate_system_recommendations();
        assert!(recommendations.iter().any(|r| r.contains("latency impact")));
    }

    #[test]
    fn test_duration_helper() {
        let duration = from_minutes(5);
        assert_eq!(duration, Duration::from_secs(300));

        let duration = from_minutes(0);
        assert_eq!(duration, Duration::from_secs(0));
    }

    #[test]
    fn test_recoverable_error_display() {
        let context = ErrorContext {
            error: "Test error".to_string(),
            classification: ErrorClassification::Transient,
            severity: ErrorSeverity::Low,
            recovery_strategy: RecoveryStrategy::NoRecovery,
            context: HashMap::new(),
            timestamp: SystemTime::now(),
            error_chain: vec![],
            recommended_actions: vec![],
            retry_info: None,
        };

        let recoverable_error = RecoverableError {
            context,
            source: None,
        };

        let display_string = format!("{}", recoverable_error);
        assert!(display_string.contains("Test error"));
        assert!(display_string.contains("transient"));
    }

    #[test]
    fn test_recovery_strategy_serialization() {
        let strategy = RecoveryStrategy::RetryWithBackoff {
            max_attempts: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(5),
            backoff_factor: 2.0,
        };

        let serialized = serde_json::to_string(&strategy).unwrap();
        let deserialized: RecoveryStrategy = serde_json::from_str(&serialized).unwrap();

        assert_eq!(strategy, deserialized);
    }

    #[test]
    fn test_error_context_serialization() {
        let context = ErrorContext {
            error: "Test error".to_string(),
            classification: ErrorClassification::Transient,
            severity: ErrorSeverity::Low,
            recovery_strategy: RecoveryStrategy::NoRecovery,
            context: HashMap::from([("key".to_string(), "value".to_string())]),
            timestamp: SystemTime::now(),
            error_chain: vec!["Error 1".to_string(), "Error 2".to_string()],
            recommended_actions: vec!["Action 1".to_string()],
            retry_info: None,
        };

        let serialized = serde_json::to_string(&context).unwrap();
        let deserialized: ErrorContext = serde_json::from_str(&serialized).unwrap();

        assert_eq!(context.error, deserialized.error);
        assert_eq!(context.classification, deserialized.classification);
        assert_eq!(context.severity, deserialized.severity);
        assert_eq!(context.context, deserialized.context);
        assert_eq!(context.error_chain, deserialized.error_chain);
        assert_eq!(
            context.recommended_actions,
            deserialized.recommended_actions
        );
    }
}

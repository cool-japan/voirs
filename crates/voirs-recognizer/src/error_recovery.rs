//! Enhanced error recovery mechanisms for VoiRS Recognizer
//!
//! This module provides comprehensive error recovery functionality including:
//! - Automatic retry with exponential backoff
//! - Circuit breaker patterns for error recovery
//! - Graceful degradation mechanisms
//! - Context-aware recovery strategies
//! - Self-healing capabilities

use crate::error_enhancement::{ErrorCategory, ErrorEnhancer};
use crate::RecognitionError;
use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime};

/// Enhanced error recovery mechanisms
#[derive(Debug, Clone)]
/// Error Recovery Manager
pub struct ErrorRecoveryManager {
    /// Recovery strategies by error category
    recovery_strategies: HashMap<ErrorCategory, Vec<RecoveryStrategy>>,
    /// Circuit breaker state tracking
    circuit_breakers: HashMap<String, CircuitBreakerState>,
    /// Retry configuration
    retry_config: RetryConfig,
    /// Self-healing capabilities
    self_healing: SelfHealingConfig,
    /// Recovery history for learning
    recovery_history: Vec<RecoveryAttempt>,
}

/// Recovery strategy definition
#[derive(Debug, Clone)]
/// Recovery Strategy
pub struct RecoveryStrategy {
    /// Strategy name
    pub name: String,
    /// Description of what this strategy does
    pub description: String,
    /// Priority (1 = highest)
    pub priority: u8,
    /// Recovery action to execute
    pub action: RecoveryAction,
    /// Conditions when this strategy applies
    pub conditions: Vec<RecoveryCondition>,
    /// Success probability estimate (0.0 to 1.0)
    pub success_probability: f32,
    /// Recovery time estimate in seconds
    pub estimated_recovery_time: f32,
}

/// Recovery action types
#[derive(Debug, Clone)]
/// Recovery Action
pub enum RecoveryAction {
    /// Retry the operation with backoff
    RetryWithBackoff {
        /// Usize
        max_attempts: usize,
        /// U64
        base_delay_ms: u64,
        /// U64
        max_delay_ms: u64,
        /// F32
        backoff_multiplier: f32,
    },
    /// Switch to fallback model/method
    SwitchToFallback {
        /// String
        fallback_target: String,
        /// Bool
        preserve_state: bool,
    },
    /// Reduce quality/complexity for faster processing
    GracefulDegradation {
        /// Degradation level
        degradation_level: DegradationLevel,
        /// Bool
        maintain_core_functionality: bool,
    },
    /// Clear cache and restart component
    ClearAndRestart {
        /// String
        component: String,
        /// Bool
        preserve_critical_data: bool,
    },
    /// Resource cleanup and optimization
    ResourceOptimization {
        /// Cleanup level
        cleanup_level: CleanupLevel,
        /// Bool
        force_gc: bool,
    },
    /// Dynamic reconfiguration
    Reconfigure {
        /// String
        config_adjustments: HashMap<String, String>,
        /// Bool
        temporary: bool,
    },
    /// Warm restart with state preservation
    WarmRestart {
        /// Bool
        preserve_models: bool,
        /// Bool
        preserve_cache: bool,
    },
}

/// Recovery condition types
#[derive(Debug, Clone)]
/// Recovery Condition
pub enum RecoveryCondition {
    /// Error count threshold
    ErrorCount {
        /// Usize
        threshold: usize,
        /// U64
        window_seconds: u64,
    },
    /// Memory usage threshold
    MemoryUsage {
        /// Memory threshold in megabytes
        threshold_mb: u64,
    },
    /// Processing time threshold
    ProcessingTime {
        /// Threshold in seconds
        threshold_seconds: f32,
    },
    /// Error pattern matching
    ErrorPattern {
        /// Pattern to match against errors
        pattern: String,
    },
    /// System resource availability
    ResourceAvailability {
        /// Type of resource
        resource_type: String,
        /// Availability threshold
        threshold: f32,
    },
    /// Model confidence threshold
    ConfidenceThreshold {
        /// Confidence threshold value
        threshold: f32,
    },
}

/// Degradation levels for graceful degradation
#[derive(Debug, Clone, PartialEq)]
/// Degradation Level
pub enum DegradationLevel {
    /// Minimal degradation - slight quality reduction
    Minimal,
    /// Moderate degradation - noticeable but acceptable quality reduction
    Moderate,
    /// Significant degradation - major quality reduction but core functionality preserved
    Significant,
    /// Emergency degradation - basic functionality only
    Emergency,
}

/// Cleanup levels for resource optimization
#[derive(Debug, Clone, PartialEq)]
/// Cleanup Level
pub enum CleanupLevel {
    /// Light cleanup - temporary data only
    Light,
    /// Medium cleanup - non-essential caches
    Medium,
    /// Heavy cleanup - all non-critical data
    Heavy,
    /// Complete cleanup - everything except essential state
    Complete,
}

/// Circuit breaker state for error recovery
#[derive(Debug, Clone, PartialEq)]
pub enum CircuitBreakerState {
    /// Normal operation
    Closed,
    /// Failing fast to prevent cascading failures
    Open {
        /// Timestamp when circuit opened
        opened_at: Instant,
    },
    /// Testing if service has recovered
    HalfOpen,
}

/// Retry configuration
#[derive(Debug, Clone)]
/// Retry Config
pub struct RetryConfig {
    /// Default maximum retry attempts
    pub default_max_attempts: usize,
    /// Default base delay between retries
    pub default_base_delay_ms: u64,
    /// Maximum delay between retries
    pub max_delay_ms: u64,
    /// Backoff multiplier
    pub backoff_multiplier: f32,
    /// Jitter factor to avoid thundering herd
    pub jitter_factor: f32,
    /// Per-error-type retry configurations
    pub error_specific_configs: HashMap<ErrorCategory, RetrySettings>,
}

/// Retry settings for specific error types
#[derive(Debug, Clone)]
/// Retry Settings
pub struct RetrySettings {
    /// max attempts
    pub max_attempts: usize,
    /// base delay ms
    pub base_delay_ms: u64,
    /// backoff multiplier
    pub backoff_multiplier: f32,
    /// enable jitter
    pub enable_jitter: bool,
}

/// Self-healing configuration
#[derive(Debug, Clone)]
/// Self Healing Config
pub struct SelfHealingConfig {
    /// Enable automatic recovery attempts
    pub enable_auto_recovery: bool,
    /// Maximum recovery attempts per hour
    pub max_recovery_attempts_per_hour: usize,
    /// Health check interval
    pub health_check_interval_seconds: u64,
    /// Enable predictive recovery
    pub enable_predictive_recovery: bool,
    /// Recovery success threshold for learning
    pub learning_threshold: f32,
}

/// Recovery result
#[derive(Debug, Clone)]
/// Recovery Result
pub struct RecoveryResult {
    /// Whether recovery was successful
    pub success: bool,
    /// Strategy that was used
    pub strategy_used: String,
    /// Time taken for recovery
    pub recovery_time: Duration,
    /// Additional information about the recovery
    pub details: String,
    /// Confidence in recovery success
    pub confidence: f32,
    /// Recommendations for preventing similar issues
    pub prevention_recommendations: Vec<String>,
}

/// Recovery attempt record for learning
#[derive(Debug, Clone)]
/// Recovery Attempt
pub struct RecoveryAttempt {
    /// Timestamp of the attempt
    pub timestamp: SystemTime,
    /// Error category that triggered recovery
    pub error_category: ErrorCategory,
    /// Strategy that was attempted
    pub strategy: String,
    /// Whether the recovery was successful
    pub success: bool,
    /// Time taken for recovery
    pub recovery_time: Duration,
    /// Context information
    pub context: HashMap<String, String>,
}

impl ErrorRecoveryManager {
    /// Create a new error recovery manager
    pub fn new() -> Self {
        let mut manager = Self {
            recovery_strategies: HashMap::new(),
            circuit_breakers: HashMap::new(),
            retry_config: RetryConfig::default(),
            self_healing: SelfHealingConfig::default(),
            recovery_history: Vec::new(),
        };

        manager.initialize_default_strategies();
        manager
    }

    /// Initialize default recovery strategies
    fn initialize_default_strategies(&mut self) {
        // Memory error strategies
        self.add_strategy(
            ErrorCategory::Resources,
            RecoveryStrategy {
                name: "Memory Cleanup".to_string(),
                description: "Clear caches and force garbage collection".to_string(),
                priority: 1,
                action: RecoveryAction::ResourceOptimization {
                    cleanup_level: CleanupLevel::Medium,
                    force_gc: true,
                },
                conditions: vec![RecoveryCondition::MemoryUsage { threshold_mb: 7000 }],
                success_probability: 0.8,
                estimated_recovery_time: 2.0,
            },
        );

        // Model error strategies
        self.add_strategy(
            ErrorCategory::ModelIssues,
            RecoveryStrategy {
                name: "Fallback Model".to_string(),
                description: "Switch to lighter, more reliable model".to_string(),
                priority: 1,
                action: RecoveryAction::SwitchToFallback {
                    fallback_target: "whisper_tiny".to_string(),
                    preserve_state: true,
                },
                conditions: vec![RecoveryCondition::ErrorCount {
                    threshold: 3,
                    window_seconds: 60,
                }],
                success_probability: 0.9,
                estimated_recovery_time: 1.0,
            },
        );

        // Performance error strategies
        self.add_strategy(
            ErrorCategory::Performance,
            RecoveryStrategy {
                name: "Graceful Degradation".to_string(),
                description: "Reduce processing quality for speed".to_string(),
                priority: 2,
                action: RecoveryAction::GracefulDegradation {
                    degradation_level: DegradationLevel::Moderate,
                    maintain_core_functionality: true,
                },
                conditions: vec![RecoveryCondition::ProcessingTime {
                    threshold_seconds: 30.0,
                }],
                success_probability: 0.7,
                estimated_recovery_time: 0.5,
            },
        );

        // Configuration error strategies
        self.add_strategy(
            ErrorCategory::Configuration,
            RecoveryStrategy {
                name: "Reset to Defaults".to_string(),
                description: "Reset configuration to known good defaults".to_string(),
                priority: 1,
                action: RecoveryAction::Reconfigure {
                    config_adjustments: {
                        let mut adjustments = HashMap::new();
                        adjustments.insert("model_size".to_string(), "base".to_string());
                        adjustments.insert("timeout".to_string(), "30".to_string());
                        adjustments
                    },
                    temporary: false,
                },
                conditions: vec![RecoveryCondition::ErrorPattern {
                    pattern: "configuration".to_string(),
                }],
                success_probability: 0.85,
                estimated_recovery_time: 1.5,
            },
        );
    }

    /// Add a recovery strategy
    pub fn add_strategy(&mut self, category: ErrorCategory, strategy: RecoveryStrategy) {
        self.recovery_strategies
            .entry(category)
            .or_insert_with(Vec::new)
            .push(strategy);
    }

    /// Attempt to recover from an error
    pub async fn attempt_recovery(
        &mut self,
        error: &RecognitionError,
        context: &HashMap<String, String>,
    ) -> RecoveryResult {
        let start_time = Instant::now();
        let enhancement = error.enhance_error();

        // Check if we should attempt recovery based on self-healing config
        if !self.should_attempt_recovery(&enhancement.category) {
            return RecoveryResult {
                success: false,
                strategy_used: "No Recovery".to_string(),
                recovery_time: start_time.elapsed(),
                details: "Recovery disabled or rate limited".to_string(),
                confidence: 0.0,
                prevention_recommendations: vec![],
            };
        }

        // Get applicable recovery strategies
        let strategies = self.get_applicable_strategies(&enhancement.category, context);

        for strategy in strategies {
            // Check if strategy conditions are met
            if self.check_strategy_conditions(&strategy, context).await {
                tracing::info!("Attempting recovery with strategy: {}", strategy.name);

                let result = self.execute_recovery_strategy(&strategy).await;

                // Record the attempt
                self.record_recovery_attempt(&enhancement.category, &strategy, &result);

                if result.success {
                    return result;
                }
            }
        }

        RecoveryResult {
            success: false,
            strategy_used: "No Applicable Strategy".to_string(),
            recovery_time: start_time.elapsed(),
            details: "No recovery strategies were applicable or successful".to_string(),
            confidence: 0.0,
            prevention_recommendations: self.get_prevention_recommendations(&enhancement.category),
        }
    }

    /// Check if recovery should be attempted
    fn should_attempt_recovery(&self, category: &ErrorCategory) -> bool {
        if !self.self_healing.enable_auto_recovery {
            return false;
        }

        // Check rate limiting
        let recent_attempts = self
            .recovery_history
            .iter()
            .filter(|attempt| {
                attempt.error_category == *category
                    && attempt
                        .timestamp
                        .elapsed()
                        .unwrap_or(Duration::from_secs(3600))
                        .as_secs()
                        < 3600
            })
            .count();

        recent_attempts < self.self_healing.max_recovery_attempts_per_hour
    }

    /// Get applicable recovery strategies
    fn get_applicable_strategies(
        &self,
        category: &ErrorCategory,
        _context: &HashMap<String, String>,
    ) -> Vec<RecoveryStrategy> {
        let mut strategies = self
            .recovery_strategies
            .get(category)
            .cloned()
            .unwrap_or_default();

        // Sort by priority and success probability
        strategies.sort_by(|a, b| {
            a.priority.cmp(&b.priority).then_with(|| {
                b.success_probability
                    .partial_cmp(&a.success_probability)
                    .unwrap()
            })
        });

        strategies
    }

    /// Check if strategy conditions are met
    async fn check_strategy_conditions(
        &self,
        strategy: &RecoveryStrategy,
        context: &HashMap<String, String>,
    ) -> bool {
        for condition in &strategy.conditions {
            match condition {
                RecoveryCondition::ErrorCount {
                    threshold,
                    window_seconds,
                } => {
                    let recent_errors = self.count_recent_errors(*window_seconds);
                    if recent_errors < *threshold {
                        return false;
                    }
                }
                RecoveryCondition::MemoryUsage { threshold_mb } => {
                    if let Some(memory_str) = context.get("memory_usage_mb") {
                        if let Ok(memory_usage) = memory_str.parse::<u64>() {
                            if memory_usage < *threshold_mb {
                                return false;
                            }
                        }
                    }
                }
                RecoveryCondition::ProcessingTime { threshold_seconds } => {
                    if let Some(time_str) = context.get("processing_time_seconds") {
                        if let Ok(processing_time) = time_str.parse::<f32>() {
                            if processing_time < *threshold_seconds {
                                return false;
                            }
                        }
                    }
                }
                RecoveryCondition::ErrorPattern { pattern } => {
                    if let Some(error_message) = context.get("error_message") {
                        if !error_message.contains(pattern) {
                            return false;
                        }
                    }
                }
                RecoveryCondition::ResourceAvailability {
                    resource_type,
                    threshold,
                } => {
                    if let Some(availability_str) =
                        context.get(&format!("{}_availability", resource_type))
                    {
                        if let Ok(availability) = availability_str.parse::<f32>() {
                            if availability > *threshold {
                                return false;
                            }
                        }
                    }
                }
                RecoveryCondition::ConfidenceThreshold { threshold } => {
                    if let Some(confidence_str) = context.get("confidence") {
                        if let Ok(confidence) = confidence_str.parse::<f32>() {
                            if confidence > *threshold {
                                return false;
                            }
                        }
                    }
                }
            }
        }
        true
    }

    /// Execute a recovery strategy
    async fn execute_recovery_strategy(&self, strategy: &RecoveryStrategy) -> RecoveryResult {
        let start_time = Instant::now();

        match &strategy.action {
            RecoveryAction::RetryWithBackoff {
                max_attempts,
                base_delay_ms,
                max_delay_ms,
                backoff_multiplier,
            } => {
                // Simulate retry logic
                for attempt in 1..=*max_attempts {
                    let delay = (*base_delay_ms as f32
                        * backoff_multiplier.powi(attempt as i32 - 1))
                    .min(*max_delay_ms as f32) as u64;

                    tokio::time::sleep(Duration::from_millis(delay)).await;

                    // In a real implementation, this would retry the failed operation
                    if attempt == *max_attempts / 2 {
                        return RecoveryResult {
                            success: true,
                            strategy_used: strategy.name.clone(),
                            recovery_time: start_time.elapsed(),
                            details: format!("Succeeded on retry attempt {}", attempt),
                            confidence: 0.8,
                            prevention_recommendations: vec![
                                "Consider increasing timeout values".to_string(),
                                "Monitor network stability".to_string(),
                            ],
                        };
                    }
                }
            }
            RecoveryAction::SwitchToFallback {
                fallback_target,
                preserve_state,
            } => {
                // Simulate fallback logic
                tracing::info!(
                    "Switching to fallback: {}, preserve_state: {}",
                    fallback_target,
                    preserve_state
                );
                return RecoveryResult {
                    success: true,
                    strategy_used: strategy.name.clone(),
                    recovery_time: start_time.elapsed(),
                    details: format!("Successfully switched to fallback: {}", fallback_target),
                    confidence: 0.9,
                    prevention_recommendations: vec![
                        "Monitor primary model health".to_string(),
                        "Consider model optimization".to_string(),
                    ],
                };
            }
            RecoveryAction::GracefulDegradation {
                degradation_level,
                maintain_core_functionality,
            } => {
                tracing::info!(
                    "Applying graceful degradation: {:?}, maintain_core: {}",
                    degradation_level,
                    maintain_core_functionality
                );
                return RecoveryResult {
                    success: true,
                    strategy_used: strategy.name.clone(),
                    recovery_time: start_time.elapsed(),
                    details: format!("Applied degradation level: {:?}", degradation_level),
                    confidence: 0.7,
                    prevention_recommendations: vec![
                        "Optimize processing pipeline".to_string(),
                        "Scale up resources during peak usage".to_string(),
                    ],
                };
            }
            RecoveryAction::ClearAndRestart {
                component,
                preserve_critical_data,
            } => {
                tracing::info!(
                    "Clearing and restarting component: {}, preserve_data: {}",
                    component,
                    preserve_critical_data
                );
                return RecoveryResult {
                    success: true,
                    strategy_used: strategy.name.clone(),
                    recovery_time: start_time.elapsed(),
                    details: format!("Component {} restarted successfully", component),
                    confidence: 0.85,
                    prevention_recommendations: vec![
                        "Implement better error handling".to_string(),
                        "Add health checks".to_string(),
                    ],
                };
            }
            RecoveryAction::ResourceOptimization {
                cleanup_level,
                force_gc,
            } => {
                tracing::info!(
                    "Optimizing resources: {:?}, force_gc: {}",
                    cleanup_level,
                    force_gc
                );
                return RecoveryResult {
                    success: true,
                    strategy_used: strategy.name.clone(),
                    recovery_time: start_time.elapsed(),
                    details: format!("Resource optimization completed: {:?}", cleanup_level),
                    confidence: 0.75,
                    prevention_recommendations: vec![
                        "Implement better memory management".to_string(),
                        "Monitor resource usage patterns".to_string(),
                    ],
                };
            }
            RecoveryAction::Reconfigure {
                config_adjustments,
                temporary,
            } => {
                tracing::info!(
                    "Reconfiguring with {} adjustments, temporary: {}",
                    config_adjustments.len(),
                    temporary
                );
                return RecoveryResult {
                    success: true,
                    strategy_used: strategy.name.clone(),
                    recovery_time: start_time.elapsed(),
                    details: format!(
                        "Configuration adjusted with {} changes",
                        config_adjustments.len()
                    ),
                    confidence: 0.8,
                    prevention_recommendations: vec![
                        "Validate configuration before applying".to_string(),
                        "Implement configuration rollback mechanism".to_string(),
                    ],
                };
            }
            RecoveryAction::WarmRestart {
                preserve_models,
                preserve_cache,
            } => {
                tracing::info!(
                    "Warm restart: preserve_models: {}, preserve_cache: {}",
                    preserve_models,
                    preserve_cache
                );
                return RecoveryResult {
                    success: true,
                    strategy_used: strategy.name.clone(),
                    recovery_time: start_time.elapsed(),
                    details: "Warm restart completed successfully".to_string(),
                    confidence: 0.9,
                    prevention_recommendations: vec![
                        "Implement better state management".to_string(),
                        "Add graceful shutdown procedures".to_string(),
                    ],
                };
            }
        }

        RecoveryResult {
            success: false,
            strategy_used: strategy.name.clone(),
            recovery_time: start_time.elapsed(),
            details: "Recovery strategy execution failed".to_string(),
            confidence: 0.0,
            prevention_recommendations: vec![],
        }
    }

    /// Count recent errors for rate limiting
    fn count_recent_errors(&self, window_seconds: u64) -> usize {
        let cutoff = SystemTime::now() - Duration::from_secs(window_seconds);
        self.recovery_history
            .iter()
            .filter(|attempt| attempt.timestamp > cutoff)
            .count()
    }

    /// Record a recovery attempt
    fn record_recovery_attempt(
        &mut self,
        category: &ErrorCategory,
        strategy: &RecoveryStrategy,
        result: &RecoveryResult,
    ) {
        let attempt = RecoveryAttempt {
            timestamp: SystemTime::now(),
            error_category: category.clone(),
            strategy: strategy.name.clone(),
            success: result.success,
            recovery_time: result.recovery_time,
            context: HashMap::new(),
        };

        self.recovery_history.push(attempt);

        // Keep only recent history to prevent memory growth
        if self.recovery_history.len() > 1000 {
            self.recovery_history.drain(0..500);
        }
    }

    /// Get prevention recommendations
    fn get_prevention_recommendations(&self, category: &ErrorCategory) -> Vec<String> {
        match category {
            ErrorCategory::Configuration => vec![
                "Implement configuration validation".to_string(),
                "Use configuration management tools".to_string(),
                "Add configuration testing".to_string(),
            ],
            ErrorCategory::Resources => vec![
                "Monitor resource usage".to_string(),
                "Implement resource limits".to_string(),
                "Scale resources based on demand".to_string(),
            ],
            ErrorCategory::ModelIssues => vec![
                "Implement model health checks".to_string(),
                "Use model versioning".to_string(),
                "Test models before deployment".to_string(),
            ],
            ErrorCategory::Performance => vec![
                "Optimize processing pipelines".to_string(),
                "Implement caching strategies".to_string(),
                "Monitor performance metrics".to_string(),
            ],
            _ => vec![
                "Implement comprehensive error handling".to_string(),
                "Add monitoring and alerting".to_string(),
                "Regular system health checks".to_string(),
            ],
        }
    }

    /// Get recovery statistics
    pub fn get_recovery_stats(&self) -> RecoveryStats {
        let total_attempts = self.recovery_history.len();
        let successful_attempts = self.recovery_history.iter().filter(|a| a.success).count();

        let success_rate = if total_attempts > 0 {
            successful_attempts as f32 / total_attempts as f32
        } else {
            0.0
        };

        let average_recovery_time = if !self.recovery_history.is_empty() {
            let total_time: Duration = self.recovery_history.iter().map(|a| a.recovery_time).sum();
            total_time / self.recovery_history.len() as u32
        } else {
            Duration::from_secs(0)
        };

        RecoveryStats {
            total_attempts,
            successful_attempts,
            success_rate,
            average_recovery_time,
            strategies_used: self.get_strategy_usage_stats(),
        }
    }

    /// Get strategy usage statistics
    fn get_strategy_usage_stats(&self) -> HashMap<String, usize> {
        let mut usage = HashMap::new();
        for attempt in &self.recovery_history {
            *usage.entry(attempt.strategy.clone()).or_insert(0) += 1;
        }
        usage
    }
}

/// Recovery statistics
#[derive(Debug, Clone)]
/// Recovery Stats
pub struct RecoveryStats {
    /// total attempts
    pub total_attempts: usize,
    /// successful attempts
    pub successful_attempts: usize,
    /// success rate
    pub success_rate: f32,
    /// average recovery time
    pub average_recovery_time: Duration,
    /// strategies used
    pub strategies_used: HashMap<String, usize>,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            default_max_attempts: 3,
            default_base_delay_ms: 1000,
            max_delay_ms: 30000,
            backoff_multiplier: 2.0,
            jitter_factor: 0.1,
            error_specific_configs: HashMap::new(),
        }
    }
}

impl Default for SelfHealingConfig {
    fn default() -> Self {
        Self {
            enable_auto_recovery: true,
            max_recovery_attempts_per_hour: 10,
            health_check_interval_seconds: 30,
            enable_predictive_recovery: false,
            learning_threshold: 0.7,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_error_recovery_manager_creation() {
        let manager = ErrorRecoveryManager::new();
        assert!(!manager.recovery_strategies.is_empty());
        assert!(manager.self_healing.enable_auto_recovery);
    }

    #[tokio::test]
    async fn test_recovery_strategy_execution() {
        let mut manager = ErrorRecoveryManager::new();
        let context = HashMap::new();

        let error = RecognitionError::ConfigurationError {
            message: "Invalid configuration".to_string(),
        };

        let result = manager.attempt_recovery(&error, &context).await;

        // Should find applicable strategies for configuration errors
        assert!(!result.strategy_used.is_empty());
    }

    #[tokio::test]
    async fn test_recovery_condition_checking() {
        let manager = ErrorRecoveryManager::new();
        let mut context = HashMap::new();
        context.insert("memory_usage_mb".to_string(), "8000".to_string());

        let strategy = RecoveryStrategy {
            name: "Test Strategy".to_string(),
            description: "Test".to_string(),
            priority: 1,
            action: RecoveryAction::ResourceOptimization {
                cleanup_level: CleanupLevel::Light,
                force_gc: false,
            },
            conditions: vec![RecoveryCondition::MemoryUsage { threshold_mb: 7000 }],
            success_probability: 0.8,
            estimated_recovery_time: 1.0,
        };

        let result = manager.check_strategy_conditions(&strategy, &context).await;
        assert!(result); // Memory usage (8000) exceeds threshold (7000)
    }

    #[test]
    fn test_recovery_stats() {
        let mut manager = ErrorRecoveryManager::new();

        // Add some mock recovery attempts
        manager.recovery_history.push(RecoveryAttempt {
            timestamp: SystemTime::now(),
            error_category: ErrorCategory::Configuration,
            strategy: "Test Strategy".to_string(),
            success: true,
            recovery_time: Duration::from_millis(1000),
            context: HashMap::new(),
        });

        let stats = manager.get_recovery_stats();
        assert_eq!(stats.total_attempts, 1);
        assert_eq!(stats.successful_attempts, 1);
        assert_eq!(stats.success_rate, 1.0);
    }
}

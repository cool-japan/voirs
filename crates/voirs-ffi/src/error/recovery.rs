use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use super::structured::{VoirsErrorCategory, VoirsErrorSubcode, VoirsStructuredError};

/// Error recovery strategy
#[derive(Clone)]
pub enum RecoveryStrategy {
    /// Retry the operation with exponential backoff
    Retry {
        max_attempts: u32,
        initial_delay: Duration,
        max_delay: Duration,
        backoff_factor: f64,
    },

    /// Use a fallback operation
    Fallback {
        fallback_fn: Arc<dyn Fn() -> Result<(), VoirsStructuredError> + Send + Sync>,
    },

    /// Graceful degradation - continue with reduced functionality
    Degrade {
        degradation_level: DegradationLevel,
        message: String,
    },

    /// Fail fast - do not attempt recovery
    FailFast,
}

impl std::fmt::Debug for RecoveryStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RecoveryStrategy::Retry {
                max_attempts,
                initial_delay,
                max_delay,
                backoff_factor,
            } => f
                .debug_struct("Retry")
                .field("max_attempts", max_attempts)
                .field("initial_delay", initial_delay)
                .field("max_delay", max_delay)
                .field("backoff_factor", backoff_factor)
                .finish(),
            RecoveryStrategy::Fallback { .. } => f
                .debug_struct("Fallback")
                .field("fallback_fn", &"<function>")
                .finish(),
            RecoveryStrategy::Degrade {
                degradation_level,
                message,
            } => f
                .debug_struct("Degrade")
                .field("degradation_level", degradation_level)
                .field("message", message)
                .finish(),
            RecoveryStrategy::FailFast => f.debug_struct("FailFast").finish(),
        }
    }
}

/// Degradation levels for graceful degradation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DegradationLevel {
    /// Minimal degradation - slight quality reduction
    Minimal,

    /// Moderate degradation - noticeable but acceptable
    Moderate,

    /// Significant degradation - basic functionality only
    Significant,

    /// Severe degradation - emergency mode
    Severe,
}

/// Recovery configuration for different error types
#[derive(Clone)]
pub struct RecoveryConfig {
    /// Strategy to use for recovery
    pub strategy: RecoveryStrategy,

    /// Whether to log recovery attempts
    pub log_attempts: bool,

    /// Whether to notify user of recovery
    pub notify_user: bool,

    /// User guidance message
    pub user_guidance: Option<String>,
}

impl std::fmt::Debug for RecoveryConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RecoveryConfig")
            .field("strategy", &self.strategy)
            .field("log_attempts", &self.log_attempts)
            .field("notify_user", &self.notify_user)
            .field("user_guidance", &self.user_guidance)
            .finish()
    }
}

/// Error recovery manager
pub struct ErrorRecoveryManager {
    /// Recovery configurations for different error types
    configs: HashMap<(VoirsErrorCategory, VoirsErrorSubcode), RecoveryConfig>,

    /// Recovery attempt history
    history: Vec<RecoveryAttempt>,

    /// Maximum history size
    max_history: usize,
}

/// Recovery attempt record
#[derive(Debug, Clone)]
pub struct RecoveryAttempt {
    /// Original error
    pub error: VoirsStructuredError,

    /// Recovery strategy used
    pub strategy: RecoveryStrategy,

    /// Whether recovery was successful
    pub success: bool,

    /// Number of attempts made
    pub attempts: u32,

    /// Time taken for recovery
    pub duration: Duration,

    /// Timestamp of recovery attempt
    pub timestamp: Instant,
}

/// Recovery result
#[derive(Debug)]
pub enum RecoveryResult {
    /// Recovery successful
    Success,

    /// Recovery failed after all attempts
    Failed(VoirsStructuredError),

    /// Recovery not attempted (fail fast)
    NotAttempted,

    /// Graceful degradation applied
    Degraded(DegradationLevel),
}

impl ErrorRecoveryManager {
    /// Create a new error recovery manager
    pub fn new() -> Self {
        let mut manager = Self {
            configs: HashMap::new(),
            history: Vec::new(),
            max_history: 1000,
        };

        // Set up default recovery strategies
        manager.setup_default_strategies();
        manager
    }

    /// Set up default recovery strategies for common error types
    fn setup_default_strategies(&mut self) {
        // Network errors - retry with exponential backoff
        self.set_recovery_config(
            VoirsErrorCategory::Network,
            VoirsErrorSubcode::NetworkTimeout,
            RecoveryConfig {
                strategy: RecoveryStrategy::Retry {
                    max_attempts: 3,
                    initial_delay: Duration::from_millis(100),
                    max_delay: Duration::from_secs(5),
                    backoff_factor: 2.0,
                },
                log_attempts: true,
                notify_user: false,
                user_guidance: Some("Network connection issues detected. Retrying...".to_string()),
            },
        );

        // Resource exhaustion - graceful degradation
        self.set_recovery_config(
            VoirsErrorCategory::Resource,
            VoirsErrorSubcode::OutOfMemory,
            RecoveryConfig {
                strategy: RecoveryStrategy::Degrade {
                    degradation_level: DegradationLevel::Moderate,
                    message: "Reducing quality to manage memory usage".to_string(),
                },
                log_attempts: true,
                notify_user: true,
                user_guidance: Some(
                    "System is low on memory. Using reduced quality mode.".to_string(),
                ),
            },
        );

        // File not found - fallback to default
        self.set_recovery_config(
            VoirsErrorCategory::Resource,
            VoirsErrorSubcode::FileNotFound,
            RecoveryConfig {
                strategy: RecoveryStrategy::Fallback {
                    fallback_fn: Arc::new(|| {
                        // Default fallback implementation
                        Ok(())
                    }),
                },
                log_attempts: true,
                notify_user: true,
                user_guidance: Some("File not found. Using default configuration.".to_string()),
            },
        );

        // Processing timeout - retry with longer timeout
        self.set_recovery_config(
            VoirsErrorCategory::Processing,
            VoirsErrorSubcode::ProcessingTimeout,
            RecoveryConfig {
                strategy: RecoveryStrategy::Retry {
                    max_attempts: 2,
                    initial_delay: Duration::from_millis(500),
                    max_delay: Duration::from_secs(10),
                    backoff_factor: 2.0,
                },
                log_attempts: true,
                notify_user: false,
                user_guidance: Some(
                    "Processing taking longer than expected. Please wait...".to_string(),
                ),
            },
        );

        // Security errors - fail fast
        self.set_recovery_config(
            VoirsErrorCategory::Security,
            VoirsErrorSubcode::AuthenticationFailed,
            RecoveryConfig {
                strategy: RecoveryStrategy::FailFast,
                log_attempts: true,
                notify_user: true,
                user_guidance: Some("Authentication failed. Please check credentials.".to_string()),
            },
        );
    }

    /// Set recovery configuration for a specific error type
    pub fn set_recovery_config(
        &mut self,
        category: VoirsErrorCategory,
        subcode: VoirsErrorSubcode,
        config: RecoveryConfig,
    ) {
        self.configs.insert((category, subcode), config);
    }

    /// Attempt to recover from an error
    pub fn attempt_recovery<F>(
        &mut self,
        error: VoirsStructuredError,
        operation: F,
    ) -> RecoveryResult
    where
        F: Fn() -> Result<(), VoirsStructuredError>,
    {
        let start_time = Instant::now();

        // Get recovery configuration
        let config = self.configs.get(&(error.category, error.subcode)).cloned();

        if let Some(config) = config {
            match &config.strategy {
                RecoveryStrategy::Retry {
                    max_attempts,
                    initial_delay,
                    max_delay,
                    backoff_factor,
                } => self.attempt_retry(
                    error,
                    operation,
                    *max_attempts,
                    *initial_delay,
                    *max_delay,
                    *backoff_factor,
                    start_time,
                ),
                RecoveryStrategy::Fallback { fallback_fn } => {
                    self.attempt_fallback(error, fallback_fn.clone(), start_time)
                }
                RecoveryStrategy::Degrade {
                    degradation_level,
                    message,
                } => {
                    self.attempt_degradation(error, *degradation_level, message.clone(), start_time)
                }
                RecoveryStrategy::FailFast => {
                    self.record_attempt(
                        error,
                        RecoveryStrategy::FailFast,
                        false,
                        0,
                        start_time.elapsed(),
                    );
                    RecoveryResult::NotAttempted
                }
            }
        } else {
            // No recovery strategy configured - fail fast
            RecoveryResult::Failed(error)
        }
    }

    /// Attempt retry recovery
    fn attempt_retry<F>(
        &mut self,
        error: VoirsStructuredError,
        operation: F,
        max_attempts: u32,
        initial_delay: Duration,
        max_delay: Duration,
        backoff_factor: f64,
        start_time: Instant,
    ) -> RecoveryResult
    where
        F: Fn() -> Result<(), VoirsStructuredError>,
    {
        let mut delay = initial_delay;

        for attempt in 1..=max_attempts {
            if attempt > 1 {
                std::thread::sleep(delay);
                delay = std::cmp::min(
                    Duration::from_millis((delay.as_millis() as f64 * backoff_factor) as u64),
                    max_delay,
                );
            }

            match operation() {
                Ok(()) => {
                    self.record_attempt(
                        error,
                        RecoveryStrategy::Retry {
                            max_attempts,
                            initial_delay,
                            max_delay,
                            backoff_factor,
                        },
                        true,
                        attempt,
                        start_time.elapsed(),
                    );
                    return RecoveryResult::Success;
                }
                Err(_) => {
                    // Continue to next attempt
                    if attempt == max_attempts {
                        self.record_attempt(
                            error.clone(),
                            RecoveryStrategy::Retry {
                                max_attempts,
                                initial_delay,
                                max_delay,
                                backoff_factor,
                            },
                            false,
                            attempt,
                            start_time.elapsed(),
                        );
                        return RecoveryResult::Failed(error);
                    }
                }
            }
        }

        RecoveryResult::Failed(error)
    }

    /// Attempt fallback recovery
    fn attempt_fallback(
        &mut self,
        error: VoirsStructuredError,
        fallback_fn: Arc<dyn Fn() -> Result<(), VoirsStructuredError> + Send + Sync>,
        start_time: Instant,
    ) -> RecoveryResult {
        match fallback_fn() {
            Ok(()) => {
                self.record_attempt(
                    error,
                    RecoveryStrategy::Fallback { fallback_fn },
                    true,
                    1,
                    start_time.elapsed(),
                );
                RecoveryResult::Success
            }
            Err(_) => {
                self.record_attempt(
                    error.clone(),
                    RecoveryStrategy::Fallback { fallback_fn },
                    false,
                    1,
                    start_time.elapsed(),
                );
                RecoveryResult::Failed(error)
            }
        }
    }

    /// Attempt graceful degradation
    fn attempt_degradation(
        &mut self,
        error: VoirsStructuredError,
        degradation_level: DegradationLevel,
        message: String,
        start_time: Instant,
    ) -> RecoveryResult {
        self.record_attempt(
            error,
            RecoveryStrategy::Degrade {
                degradation_level,
                message,
            },
            true,
            1,
            start_time.elapsed(),
        );
        RecoveryResult::Degraded(degradation_level)
    }

    /// Record recovery attempt
    fn record_attempt(
        &mut self,
        error: VoirsStructuredError,
        strategy: RecoveryStrategy,
        success: bool,
        attempts: u32,
        duration: Duration,
    ) {
        let attempt = RecoveryAttempt {
            error,
            strategy,
            success,
            attempts,
            duration,
            timestamp: Instant::now(),
        };

        self.history.push(attempt);

        // Maintain history size
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }
    }

    /// Get recovery statistics
    pub fn get_recovery_stats(&self) -> RecoveryStats {
        let total_attempts = self.history.len();
        let successful_attempts = self.history.iter().filter(|a| a.success).count();
        let failed_attempts = total_attempts - successful_attempts;

        let mut strategy_stats = HashMap::new();
        for attempt in &self.history {
            let strategy_name = match &attempt.strategy {
                RecoveryStrategy::Retry { .. } => "Retry",
                RecoveryStrategy::Fallback { .. } => "Fallback",
                RecoveryStrategy::Degrade { .. } => "Degrade",
                RecoveryStrategy::FailFast => "FailFast",
            };

            let entry = strategy_stats
                .entry(strategy_name.to_string())
                .or_insert((0, 0));
            entry.0 += 1;
            if attempt.success {
                entry.1 += 1;
            }
        }

        RecoveryStats {
            total_attempts,
            successful_attempts,
            failed_attempts,
            strategy_stats,
        }
    }

    /// Get recent recovery attempts
    pub fn get_recent_attempts(&self, count: usize) -> Vec<RecoveryAttempt> {
        self.history.iter().rev().take(count).cloned().collect()
    }

    /// Clear recovery history
    pub fn clear_history(&mut self) {
        self.history.clear();
    }
}

/// Recovery statistics
#[derive(Debug, Clone)]
pub struct RecoveryStats {
    /// Total number of recovery attempts
    pub total_attempts: usize,

    /// Number of successful recoveries
    pub successful_attempts: usize,

    /// Number of failed recoveries
    pub failed_attempts: usize,

    /// Statistics by strategy (strategy_name -> (total, successful))
    pub strategy_stats: HashMap<String, (usize, usize)>,
}

/// Global error recovery manager
static ERROR_RECOVERY_MANAGER: std::sync::LazyLock<Mutex<ErrorRecoveryManager>> =
    std::sync::LazyLock::new(|| Mutex::new(ErrorRecoveryManager::new()));

/// Attempt recovery using global manager
pub fn attempt_error_recovery<F>(error: VoirsStructuredError, operation: F) -> RecoveryResult
where
    F: Fn() -> Result<(), VoirsStructuredError>,
{
    ERROR_RECOVERY_MANAGER
        .lock()
        .unwrap()
        .attempt_recovery(error, operation)
}

/// Get recovery statistics from global manager
pub fn get_recovery_statistics() -> RecoveryStats {
    ERROR_RECOVERY_MANAGER.lock().unwrap().get_recovery_stats()
}

/// Configure recovery strategy globally
pub fn configure_recovery_strategy(
    category: VoirsErrorCategory,
    subcode: VoirsErrorSubcode,
    config: RecoveryConfig,
) {
    ERROR_RECOVERY_MANAGER
        .lock()
        .unwrap()
        .set_recovery_config(category, subcode, config);
}

/// C API functions for error recovery
#[no_mangle]
pub extern "C" fn voirs_attempt_recovery(
    error_category: VoirsErrorCategory,
    error_subcode: VoirsErrorSubcode,
    operation: extern "C" fn() -> crate::VoirsErrorCode,
) -> crate::VoirsErrorCode {
    let error = VoirsStructuredError::new(
        error_category,
        error_subcode,
        "Recovery attempt".to_string(),
        "C API".to_string(),
    );

    let result = attempt_error_recovery(error, || match operation() {
        crate::VoirsErrorCode::Success => Ok(()),
        code => Err(VoirsStructuredError::new(
            error_category,
            error_subcode,
            format!("Operation failed with code: {:?}", code),
            "C API".to_string(),
        )),
    });

    match result {
        RecoveryResult::Success => crate::VoirsErrorCode::Success,
        RecoveryResult::Failed(err) => err.to_error_code(),
        RecoveryResult::NotAttempted => crate::VoirsErrorCode::InternalError,
        RecoveryResult::Degraded(_) => crate::VoirsErrorCode::Success,
    }
}

#[no_mangle]
pub unsafe extern "C" fn voirs_get_recovery_stats(
    total_attempts: *mut usize,
    successful_attempts: *mut usize,
    failed_attempts: *mut usize,
) -> crate::VoirsErrorCode {
    if total_attempts.is_null() || successful_attempts.is_null() || failed_attempts.is_null() {
        return crate::VoirsErrorCode::InvalidParameter;
    }

    let stats = get_recovery_statistics();

    unsafe {
        *total_attempts = stats.total_attempts;
        *successful_attempts = stats.successful_attempts;
        *failed_attempts = stats.failed_attempts;
    }

    crate::VoirsErrorCode::Success
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn test_retry_recovery() {
        let mut manager = ErrorRecoveryManager::new();
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();

        let error = VoirsStructuredError::new(
            VoirsErrorCategory::Network,
            VoirsErrorSubcode::NetworkTimeout,
            "Network timeout".to_string(),
            "test".to_string(),
        );

        let result = manager.attempt_recovery(error, || {
            let count = counter_clone.fetch_add(1, Ordering::SeqCst);
            if count < 2 {
                Err(VoirsStructuredError::new(
                    VoirsErrorCategory::Network,
                    VoirsErrorSubcode::NetworkTimeout,
                    "Still failing".to_string(),
                    "test".to_string(),
                ))
            } else {
                Ok(())
            }
        });

        assert!(matches!(result, RecoveryResult::Success));
        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn test_degradation_recovery() {
        let mut manager = ErrorRecoveryManager::new();

        let error = VoirsStructuredError::new(
            VoirsErrorCategory::Resource,
            VoirsErrorSubcode::OutOfMemory,
            "Out of memory".to_string(),
            "test".to_string(),
        );

        let result = manager.attempt_recovery(error, || {
            Err(VoirsStructuredError::new(
                VoirsErrorCategory::Resource,
                VoirsErrorSubcode::OutOfMemory,
                "Still out of memory".to_string(),
                "test".to_string(),
            ))
        });

        assert!(matches!(
            result,
            RecoveryResult::Degraded(DegradationLevel::Moderate)
        ));
    }

    #[test]
    fn test_fail_fast_recovery() {
        let mut manager = ErrorRecoveryManager::new();

        let error = VoirsStructuredError::new(
            VoirsErrorCategory::Security,
            VoirsErrorSubcode::AuthenticationFailed,
            "Authentication failed".to_string(),
            "test".to_string(),
        );

        let result = manager.attempt_recovery(error, || Ok(()));

        assert!(matches!(result, RecoveryResult::NotAttempted));
    }

    #[test]
    fn test_recovery_stats() {
        let mut manager = ErrorRecoveryManager::new();

        // Simulate successful recovery
        let error1 = VoirsStructuredError::new(
            VoirsErrorCategory::Network,
            VoirsErrorSubcode::NetworkTimeout,
            "Test 1".to_string(),
            "test".to_string(),
        );

        manager.attempt_recovery(error1, || Ok(()));

        let stats = manager.get_recovery_stats();
        assert_eq!(stats.total_attempts, 1);
        assert_eq!(stats.successful_attempts, 1);
        assert_eq!(stats.failed_attempts, 0);
    }
}

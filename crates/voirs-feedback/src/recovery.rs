//! Automatic recovery mechanisms for long-running VoiRS feedback sessions
//!
//! This module provides comprehensive error recovery, retry logic, and graceful
//! degradation mechanisms to ensure system reliability and continuity of service.

use crate::FeedbackError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock};

/// Recovery manager for handling system failures and automatic recovery
pub struct RecoveryManager {
    config: RecoveryConfig,
    failure_tracker: Arc<Mutex<FailureTracker>>,
    recovery_strategies: Arc<RwLock<HashMap<FailureType, RecoveryStrategy>>>,
    circuit_breaker: Arc<Mutex<CircuitBreaker>>,
    health_monitor: Arc<Mutex<HealthMonitor>>,
}

impl RecoveryManager {
    /// Create a new recovery manager
    pub fn new(config: RecoveryConfig) -> Self {
        let mut strategies = HashMap::new();

        // Default recovery strategies for common failure types
        strategies.insert(
            FailureType::NetworkError,
            RecoveryStrategy::ExponentialBackoff {
                base_delay: Duration::from_millis(100),
                max_delay: Duration::from_secs(30),
                max_retries: 5,
                backoff_multiplier: 2.0,
            },
        );

        strategies.insert(
            FailureType::ServiceUnavailable,
            RecoveryStrategy::CircuitBreaker {
                failure_threshold: 3,
                timeout: Duration::from_secs(60),
                half_open_max_calls: 5,
            },
        );

        strategies.insert(
            FailureType::ResourceExhaustion,
            RecoveryStrategy::GracefulDegradation {
                fallback_mode: DegradationMode::ReducedQuality,
                recovery_threshold: 0.8,
            },
        );

        strategies.insert(
            FailureType::ProcessingTimeout,
            RecoveryStrategy::TimeoutEscalation {
                initial_timeout: Duration::from_millis(100),
                max_timeout: Duration::from_secs(10),
                escalation_factor: 2.0,
            },
        );

        Self {
            config,
            failure_tracker: Arc::new(Mutex::new(FailureTracker::new())),
            recovery_strategies: Arc::new(RwLock::new(strategies)),
            circuit_breaker: Arc::new(Mutex::new(CircuitBreaker::new())),
            health_monitor: Arc::new(Mutex::new(HealthMonitor::new())),
        }
    }

    /// Execute operation with automatic recovery
    pub async fn execute_with_recovery<F, T, E>(&self, mut operation: F) -> Result<T, RecoveryError>
    where
        F: FnMut() -> Result<T, E> + Send + Sync,
        E: Into<FeedbackError> + Send + Sync,
    {
        let failure_type = FailureType::Unknown;
        let strategies = self.recovery_strategies.read().await;
        let strategy = strategies
            .get(&failure_type)
            .unwrap_or(&RecoveryStrategy::SimpleRetry { max_retries: 3 });

        self.execute_with_strategy(operation, strategy.clone())
            .await
    }

    /// Execute operation with specific recovery strategy
    pub async fn execute_with_strategy<F, T, E>(
        &self,
        mut operation: F,
        strategy: RecoveryStrategy,
    ) -> Result<T, RecoveryError>
    where
        F: FnMut() -> Result<T, E> + Send + Sync,
        E: Into<FeedbackError> + Send + Sync,
    {
        match strategy {
            RecoveryStrategy::SimpleRetry { max_retries } => {
                self.execute_simple_retry(operation, max_retries).await
            }
            RecoveryStrategy::ExponentialBackoff {
                base_delay,
                max_delay,
                max_retries,
                backoff_multiplier,
            } => {
                self.execute_exponential_backoff(
                    operation,
                    base_delay,
                    max_delay,
                    max_retries,
                    backoff_multiplier,
                )
                .await
            }
            RecoveryStrategy::CircuitBreaker {
                failure_threshold,
                timeout,
                half_open_max_calls,
            } => {
                self.execute_circuit_breaker(
                    operation,
                    failure_threshold,
                    timeout,
                    half_open_max_calls,
                )
                .await
            }
            RecoveryStrategy::GracefulDegradation {
                fallback_mode,
                recovery_threshold,
            } => {
                self.execute_graceful_degradation(operation, fallback_mode, recovery_threshold)
                    .await
            }
            RecoveryStrategy::TimeoutEscalation {
                initial_timeout,
                max_timeout,
                escalation_factor,
            } => {
                self.execute_timeout_escalation(
                    operation,
                    initial_timeout,
                    max_timeout,
                    escalation_factor,
                )
                .await
            }
        }
    }

    /// Execute with simple retry logic
    async fn execute_simple_retry<F, T, E>(
        &self,
        mut operation: F,
        max_retries: u32,
    ) -> Result<T, RecoveryError>
    where
        F: FnMut() -> Result<T, E> + Send + Sync,
        E: Into<FeedbackError> + Send + Sync,
    {
        let mut last_error = None;

        for attempt in 0..=max_retries {
            match operation() {
                Ok(result) => {
                    // Record successful recovery if this was a retry
                    if attempt > 0 {
                        let mut tracker = self.failure_tracker.lock().await;
                        tracker.record_recovery(FailureType::Unknown, attempt);
                    }
                    return Ok(result);
                }
                Err(error) => {
                    last_error = Some(error.into());

                    // Record failure
                    let mut tracker = self.failure_tracker.lock().await;
                    tracker.record_failure(FailureType::Unknown, attempt);

                    if attempt < max_retries {
                        // Small delay between retries
                        tokio::time::sleep(Duration::from_millis(100)).await;
                    }
                }
            }
        }

        Err(RecoveryError::MaxRetriesExceeded {
            attempts: max_retries + 1,
            last_error: last_error.unwrap(),
        })
    }

    /// Execute with exponential backoff
    async fn execute_exponential_backoff<F, T, E>(
        &self,
        mut operation: F,
        base_delay: Duration,
        max_delay: Duration,
        max_retries: u32,
        backoff_multiplier: f64,
    ) -> Result<T, RecoveryError>
    where
        F: FnMut() -> Result<T, E> + Send + Sync,
        E: Into<FeedbackError> + Send + Sync,
    {
        let mut last_error = None;
        let mut current_delay = base_delay;

        for attempt in 0..=max_retries {
            match operation() {
                Ok(result) => {
                    // Record successful recovery
                    if attempt > 0 {
                        let mut tracker = self.failure_tracker.lock().await;
                        tracker.record_recovery(FailureType::NetworkError, attempt);
                    }
                    return Ok(result);
                }
                Err(error) => {
                    last_error = Some(error.into());

                    // Record failure
                    let mut tracker = self.failure_tracker.lock().await;
                    tracker.record_failure(FailureType::NetworkError, attempt);

                    if attempt < max_retries {
                        // Wait with exponential backoff
                        tokio::time::sleep(current_delay).await;

                        // Increase delay for next attempt
                        current_delay = Duration::from_millis(
                            (current_delay.as_millis() as f64 * backoff_multiplier) as u64,
                        )
                        .min(max_delay);
                    }
                }
            }
        }

        Err(RecoveryError::BackoffExhausted {
            attempts: max_retries + 1,
            final_delay: current_delay,
            last_error: last_error.unwrap(),
        })
    }

    /// Execute with circuit breaker pattern
    async fn execute_circuit_breaker<F, T, E>(
        &self,
        mut operation: F,
        failure_threshold: u32,
        timeout: Duration,
        half_open_max_calls: u32,
    ) -> Result<T, RecoveryError>
    where
        F: FnMut() -> Result<T, E> + Send + Sync,
        E: Into<FeedbackError> + Send + Sync,
    {
        let mut circuit_breaker = self.circuit_breaker.lock().await;

        match circuit_breaker.state {
            CircuitBreakerState::Closed => match operation() {
                Ok(result) => {
                    circuit_breaker.record_success();
                    Ok(result)
                }
                Err(error) => {
                    circuit_breaker.record_failure();

                    if circuit_breaker.failure_count >= failure_threshold {
                        circuit_breaker.open(timeout);
                    }

                    Err(RecoveryError::OperationFailed(error.into()))
                }
            },
            CircuitBreakerState::Open => {
                if circuit_breaker.should_attempt_reset() {
                    circuit_breaker.half_open(half_open_max_calls);
                    // Try operation in half-open state
                    match operation() {
                        Ok(result) => {
                            circuit_breaker.close();
                            Ok(result)
                        }
                        Err(error) => {
                            circuit_breaker.open(timeout);
                            Err(RecoveryError::CircuitBreakerOpen)
                        }
                    }
                } else {
                    Err(RecoveryError::CircuitBreakerOpen)
                }
            }
            CircuitBreakerState::HalfOpen => {
                if circuit_breaker.half_open_calls < half_open_max_calls {
                    circuit_breaker.half_open_calls += 1;

                    match operation() {
                        Ok(result) => {
                            circuit_breaker.record_success();
                            if circuit_breaker.consecutive_successes >= 2 {
                                circuit_breaker.close();
                            }
                            Ok(result)
                        }
                        Err(error) => {
                            circuit_breaker.open(timeout);
                            Err(RecoveryError::OperationFailed(error.into()))
                        }
                    }
                } else {
                    circuit_breaker.open(timeout);
                    Err(RecoveryError::CircuitBreakerOpen)
                }
            }
        }
    }

    /// Execute with graceful degradation
    async fn execute_graceful_degradation<F, T, E>(
        &self,
        mut operation: F,
        fallback_mode: DegradationMode,
        recovery_threshold: f64,
    ) -> Result<T, RecoveryError>
    where
        F: FnMut() -> Result<T, E> + Send + Sync,
        E: Into<FeedbackError> + Send + Sync,
    {
        // Check system health
        let mut health_monitor = self.health_monitor.lock().await;
        let current_health = health_monitor.get_health_score();

        if current_health < recovery_threshold {
            // System is degraded, try to recover
            health_monitor.enter_degraded_mode(fallback_mode.clone());

            // Simple operation execution in degraded mode
            match operation() {
                Ok(result) => {
                    health_monitor.record_success();

                    // Check if we can exit degraded mode
                    if health_monitor.get_health_score() >= recovery_threshold {
                        health_monitor.exit_degraded_mode();
                    }

                    Ok(result)
                }
                Err(error) => {
                    health_monitor.record_failure();
                    Err(RecoveryError::DegradedMode {
                        mode: fallback_mode,
                        health_score: current_health,
                        error: error.into(),
                    })
                }
            }
        } else {
            // System is healthy, normal execution
            match operation() {
                Ok(result) => {
                    health_monitor.record_success();
                    Ok(result)
                }
                Err(error) => {
                    health_monitor.record_failure();
                    Err(RecoveryError::OperationFailed(error.into()))
                }
            }
        }
    }

    /// Execute with timeout escalation
    async fn execute_timeout_escalation<F, T, E>(
        &self,
        mut operation: F,
        initial_timeout: Duration,
        max_timeout: Duration,
        escalation_factor: f64,
    ) -> Result<T, RecoveryError>
    where
        F: FnMut() -> Result<T, E> + Send + Sync,
        E: Into<FeedbackError> + Send + Sync,
    {
        let mut current_timeout = initial_timeout;
        let mut attempts = 0;

        loop {
            attempts += 1;

            // Execute with timeout
            let result = tokio::time::timeout(current_timeout, async { operation() }).await;

            match result {
                Ok(Ok(value)) => {
                    // Success
                    let mut tracker = self.failure_tracker.lock().await;
                    tracker.record_recovery(FailureType::ProcessingTimeout, attempts);
                    return Ok(value);
                }
                Ok(Err(error)) => {
                    // Operation failed (not timeout)
                    return Err(RecoveryError::OperationFailed(error.into()));
                }
                Err(_) => {
                    // Timeout occurred
                    let mut tracker = self.failure_tracker.lock().await;
                    tracker.record_failure(FailureType::ProcessingTimeout, attempts);

                    if current_timeout >= max_timeout {
                        return Err(RecoveryError::TimeoutEscalationExhausted {
                            attempts,
                            final_timeout: current_timeout,
                        });
                    }

                    // Escalate timeout
                    current_timeout = Duration::from_millis(
                        (current_timeout.as_millis() as f64 * escalation_factor) as u64,
                    )
                    .min(max_timeout);
                }
            }
        }
    }

    /// Get recovery statistics
    pub async fn get_recovery_stats(&self) -> RecoveryStats {
        let tracker = self.failure_tracker.lock().await;
        let circuit_breaker = self.circuit_breaker.lock().await;
        let health_monitor = self.health_monitor.lock().await;

        RecoveryStats {
            total_failures: tracker.total_failures,
            total_recoveries: tracker.total_recoveries,
            failure_rate: tracker.get_failure_rate(),
            recovery_rate: tracker.get_recovery_rate(),
            circuit_breaker_state: circuit_breaker.state.clone(),
            current_health_score: health_monitor.get_health_score(),
            degraded_mode: health_monitor.degraded_mode.clone(),
            failure_breakdown: tracker.failure_breakdown.clone(),
        }
    }

    /// Reset all recovery state
    pub async fn reset(&self) {
        let mut tracker = self.failure_tracker.lock().await;
        tracker.reset();

        let mut circuit_breaker = self.circuit_breaker.lock().await;
        circuit_breaker.reset();

        let mut health_monitor = self.health_monitor.lock().await;
        health_monitor.reset();
    }
}

/// Recovery configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryConfig {
    /// Enable automatic recovery
    pub enabled: bool,
    /// Maximum total recovery attempts across all strategies
    pub max_total_attempts: u32,
    /// Global timeout for recovery operations
    pub global_timeout: Duration,
    /// Health check interval
    pub health_check_interval: Duration,
}

impl Default for RecoveryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_total_attempts: 10,
            global_timeout: Duration::from_secs(300), // 5 minutes
            health_check_interval: Duration::from_secs(30),
        }
    }
}

/// Types of failures that can be recovered from
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FailureType {
    NetworkError,
    ServiceUnavailable,
    ResourceExhaustion,
    ProcessingTimeout,
    DatabaseError,
    ConfigurationError,
    Unknown,
}

/// Recovery strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryStrategy {
    /// Simple retry with fixed delay
    SimpleRetry { max_retries: u32 },
    /// Exponential backoff retry
    ExponentialBackoff {
        base_delay: Duration,
        max_delay: Duration,
        max_retries: u32,
        backoff_multiplier: f64,
    },
    /// Circuit breaker pattern
    CircuitBreaker {
        failure_threshold: u32,
        timeout: Duration,
        half_open_max_calls: u32,
    },
    /// Graceful degradation
    GracefulDegradation {
        fallback_mode: DegradationMode,
        recovery_threshold: f64,
    },
    /// Timeout escalation
    TimeoutEscalation {
        initial_timeout: Duration,
        max_timeout: Duration,
        escalation_factor: f64,
    },
}

/// Degradation modes for graceful fallback
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DegradationMode {
    /// Reduce processing quality for speed
    ReducedQuality,
    /// Use cached responses when possible
    CacheOnly,
    /// Minimal functionality only
    Essential,
    /// Read-only mode
    ReadOnly,
}

/// Circuit breaker states
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CircuitBreakerState {
    Closed,
    Open,
    HalfOpen,
}

/// Circuit breaker implementation
#[derive(Debug)]
pub struct CircuitBreaker {
    state: CircuitBreakerState,
    failure_count: u32,
    consecutive_successes: u32,
    last_failure_time: Option<Instant>,
    timeout_duration: Duration,
    half_open_calls: u32,
}

impl CircuitBreaker {
    pub fn new() -> Self {
        Self {
            state: CircuitBreakerState::Closed,
            failure_count: 0,
            consecutive_successes: 0,
            last_failure_time: None,
            timeout_duration: Duration::from_secs(60),
            half_open_calls: 0,
        }
    }

    pub fn record_success(&mut self) {
        self.consecutive_successes += 1;
        self.failure_count = 0;
    }

    pub fn record_failure(&mut self) {
        self.failure_count += 1;
        self.consecutive_successes = 0;
        self.last_failure_time = Some(Instant::now());
    }

    pub fn open(&mut self, timeout: Duration) {
        self.state = CircuitBreakerState::Open;
        self.timeout_duration = timeout;
        self.last_failure_time = Some(Instant::now());
    }

    pub fn close(&mut self) {
        self.state = CircuitBreakerState::Closed;
        self.failure_count = 0;
        self.half_open_calls = 0;
    }

    pub fn half_open(&mut self, max_calls: u32) {
        self.state = CircuitBreakerState::HalfOpen;
        self.half_open_calls = 0;
    }

    pub fn should_attempt_reset(&self) -> bool {
        if let Some(last_failure) = self.last_failure_time {
            last_failure.elapsed() >= self.timeout_duration
        } else {
            false
        }
    }

    pub fn reset(&mut self) {
        self.state = CircuitBreakerState::Closed;
        self.failure_count = 0;
        self.consecutive_successes = 0;
        self.last_failure_time = None;
        self.half_open_calls = 0;
    }
}

/// Health monitoring
#[derive(Debug)]
pub struct HealthMonitor {
    success_count: u32,
    failure_count: u32,
    degraded_mode: Option<DegradationMode>,
    last_health_check: Instant,
}

impl HealthMonitor {
    pub fn new() -> Self {
        Self {
            success_count: 0,
            failure_count: 0,
            degraded_mode: None,
            last_health_check: Instant::now(),
        }
    }

    pub fn record_success(&mut self) {
        self.success_count += 1;
    }

    pub fn record_failure(&mut self) {
        self.failure_count += 1;
    }

    pub fn get_health_score(&self) -> f64 {
        let total = self.success_count + self.failure_count;
        if total == 0 {
            return 1.0; // No data means healthy
        }

        self.success_count as f64 / total as f64
    }

    pub fn enter_degraded_mode(&mut self, mode: DegradationMode) {
        self.degraded_mode = Some(mode);
    }

    pub fn exit_degraded_mode(&mut self) {
        self.degraded_mode = None;
    }

    pub fn reset(&mut self) {
        self.success_count = 0;
        self.failure_count = 0;
        self.degraded_mode = None;
        self.last_health_check = Instant::now();
    }
}

/// Failure tracking
#[derive(Debug)]
pub struct FailureTracker {
    total_failures: u64,
    total_recoveries: u64,
    failure_breakdown: HashMap<FailureType, u32>,
    recovery_breakdown: HashMap<FailureType, u32>,
    recent_failures: Vec<(Instant, FailureType)>,
}

impl FailureTracker {
    pub fn new() -> Self {
        Self {
            total_failures: 0,
            total_recoveries: 0,
            failure_breakdown: HashMap::new(),
            recovery_breakdown: HashMap::new(),
            recent_failures: Vec::new(),
        }
    }

    pub fn record_failure(&mut self, failure_type: FailureType, attempt: u32) {
        self.total_failures += 1;
        *self
            .failure_breakdown
            .entry(failure_type.clone())
            .or_insert(0) += 1;
        self.recent_failures.push((Instant::now(), failure_type));

        // Keep only recent failures (last hour)
        let cutoff = Instant::now() - Duration::from_secs(3600);
        self.recent_failures.retain(|(time, _)| *time > cutoff);
    }

    pub fn record_recovery(&mut self, failure_type: FailureType, attempts: u32) {
        self.total_recoveries += 1;
        *self.recovery_breakdown.entry(failure_type).or_insert(0) += 1;
    }

    pub fn get_failure_rate(&self) -> f64 {
        let total_operations = self.total_failures + self.total_recoveries;
        if total_operations == 0 {
            return 0.0;
        }
        self.total_failures as f64 / total_operations as f64
    }

    pub fn get_recovery_rate(&self) -> f64 {
        if self.total_failures == 0 {
            return 1.0;
        }
        self.total_recoveries as f64 / self.total_failures as f64
    }

    pub fn reset(&mut self) {
        self.total_failures = 0;
        self.total_recoveries = 0;
        self.failure_breakdown.clear();
        self.recovery_breakdown.clear();
        self.recent_failures.clear();
    }
}

/// Recovery error types
#[derive(Debug)]
pub enum RecoveryError {
    MaxRetriesExceeded {
        attempts: u32,
        last_error: FeedbackError,
    },
    BackoffExhausted {
        attempts: u32,
        final_delay: Duration,
        last_error: FeedbackError,
    },
    CircuitBreakerOpen,
    DegradedMode {
        mode: DegradationMode,
        health_score: f64,
        error: FeedbackError,
    },
    TimeoutEscalationExhausted {
        attempts: u32,
        final_timeout: Duration,
    },
    OperationFailed(FeedbackError),
    ConfigurationError(String),
}

impl fmt::Display for RecoveryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RecoveryError::MaxRetriesExceeded {
                attempts,
                last_error,
            } => {
                write!(
                    f,
                    "Maximum retries ({}) exceeded. Last error: {}",
                    attempts, last_error
                )
            }
            RecoveryError::BackoffExhausted {
                attempts,
                final_delay,
                last_error,
            } => {
                write!(f, "Exponential backoff exhausted after {} attempts (final delay: {:?}). Last error: {}", 
                       attempts, final_delay, last_error)
            }
            RecoveryError::CircuitBreakerOpen => {
                write!(
                    f,
                    "Circuit breaker is open - service temporarily unavailable"
                )
            }
            RecoveryError::DegradedMode {
                mode,
                health_score,
                error,
            } => {
                write!(
                    f,
                    "Operating in degraded mode ({:?}) with health score {:.2}. Error: {}",
                    mode, health_score, error
                )
            }
            RecoveryError::TimeoutEscalationExhausted {
                attempts,
                final_timeout,
            } => {
                write!(
                    f,
                    "Timeout escalation exhausted after {} attempts (final timeout: {:?})",
                    attempts, final_timeout
                )
            }
            RecoveryError::OperationFailed(error) => {
                write!(f, "Operation failed: {}", error)
            }
            RecoveryError::ConfigurationError(msg) => {
                write!(f, "Configuration error: {}", msg)
            }
        }
    }
}

impl std::error::Error for RecoveryError {}

/// Recovery statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryStats {
    pub total_failures: u64,
    pub total_recoveries: u64,
    pub failure_rate: f64,
    pub recovery_rate: f64,
    pub circuit_breaker_state: CircuitBreakerState,
    pub current_health_score: f64,
    pub degraded_mode: Option<DegradationMode>,
    pub failure_breakdown: HashMap<FailureType, u32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_recovery_manager_creation() {
        let config = RecoveryConfig::default();
        let recovery_manager = RecoveryManager::new(config);

        let stats = recovery_manager.get_recovery_stats().await;
        assert_eq!(stats.total_failures, 0);
        assert_eq!(stats.total_recoveries, 0);
        assert_eq!(stats.failure_rate, 0.0);
    }

    #[tokio::test]
    async fn test_simple_retry_success() {
        let recovery_manager = RecoveryManager::new(RecoveryConfig::default());

        let mut call_count = 0;
        let operation = || {
            call_count += 1;
            if call_count < 3 {
                Err(FeedbackError::RealtimeError {
                    message: "Temporary failure".to_string(),
                    source: None,
                })
            } else {
                Ok("Success".to_string())
            }
        };

        let strategy = RecoveryStrategy::SimpleRetry { max_retries: 5 };
        let result = recovery_manager
            .execute_with_strategy(operation, strategy)
            .await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "Success");
        assert_eq!(call_count, 3);
    }

    #[tokio::test]
    async fn test_exponential_backoff() {
        let recovery_manager = RecoveryManager::new(RecoveryConfig::default());

        let mut call_count = 0;
        let operation = || {
            call_count += 1;
            if call_count < 2 {
                Err(FeedbackError::RealtimeError {
                    message: "Network error".to_string(),
                    source: None,
                })
            } else {
                Ok(42u32)
            }
        };

        let strategy = RecoveryStrategy::ExponentialBackoff {
            base_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(100),
            max_retries: 3,
            backoff_multiplier: 2.0,
        };

        let start = Instant::now();
        let result = recovery_manager
            .execute_with_strategy(operation, strategy)
            .await;
        let elapsed = start.elapsed();

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
        assert!(elapsed >= Duration::from_millis(1)); // Should have some delay
    }

    #[tokio::test]
    async fn test_circuit_breaker() {
        let recovery_manager = RecoveryManager::new(RecoveryConfig::default());

        let operation = || {
            Err::<(), _>(FeedbackError::RealtimeError {
                message: "Always fails".to_string(),
                source: None,
            })
        };

        let strategy = RecoveryStrategy::CircuitBreaker {
            failure_threshold: 2,
            timeout: Duration::from_millis(50),
            half_open_max_calls: 1,
        };

        // First few calls should fail normally
        for _ in 0..2 {
            let result = recovery_manager
                .execute_with_strategy(&operation, strategy.clone())
                .await;
            assert!(matches!(result, Err(RecoveryError::OperationFailed(_))));
        }

        // After threshold, circuit breaker should open
        let result = recovery_manager
            .execute_with_strategy(&operation, strategy.clone())
            .await;
        assert!(matches!(result, Err(RecoveryError::CircuitBreakerOpen)));
    }

    #[tokio::test]
    async fn test_graceful_degradation() {
        let recovery_manager = RecoveryManager::new(RecoveryConfig::default());

        let operation = || -> Result<String, FeedbackError> { Ok("Degraded result".to_string()) };

        let strategy = RecoveryStrategy::GracefulDegradation {
            fallback_mode: DegradationMode::ReducedQuality,
            recovery_threshold: 0.8,
        };

        let result = recovery_manager
            .execute_with_strategy(operation, strategy)
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_recovery_stats() {
        let recovery_manager = RecoveryManager::new(RecoveryConfig::default());

        // Record some failures and recoveries
        let operation = || {
            Err::<(), _>(FeedbackError::RealtimeError {
                message: "Test error".to_string(),
                source: None,
            })
        };

        let strategy = RecoveryStrategy::SimpleRetry { max_retries: 1 };
        let _ = recovery_manager
            .execute_with_strategy(operation, strategy)
            .await;

        let stats = recovery_manager.get_recovery_stats().await;
        assert!(stats.total_failures > 0);
    }
}

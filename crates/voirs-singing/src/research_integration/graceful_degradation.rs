// Graceful Degradation System - Phase 5 Feature
//
// Circuit breaker pattern, fallback strategies, and resilience patterns
// for production singing synthesis systems.

use crate::Error;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Circuit breaker states
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Circuit is closed, requests pass through normally
    Closed,
    /// Circuit is open, requests fail immediately
    Open,
    /// Circuit is half-open, testing if service has recovered
    HalfOpen,
}

/// Circuit breaker configuration
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Failure threshold to open circuit
    pub failure_threshold: usize,
    /// Success threshold to close circuit from half-open
    pub success_threshold: usize,
    /// Timeout before attempting recovery
    pub timeout: Duration,
    /// Window size for failure counting
    pub window_size: usize,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 2,
            timeout: Duration::from_secs(60),
            window_size: 10,
        }
    }
}

impl CircuitBreakerConfig {
    pub fn aggressive() -> Self {
        Self {
            failure_threshold: 3,
            success_threshold: 1,
            timeout: Duration::from_secs(30),
            window_size: 5,
        }
    }

    pub fn conservative() -> Self {
        Self {
            failure_threshold: 10,
            success_threshold: 5,
            timeout: Duration::from_secs(120),
            window_size: 20,
        }
    }
}

/// Circuit breaker for fault tolerance
#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    config: CircuitBreakerConfig,
    state: Arc<RwLock<CircuitBreakerState>>,
}

#[derive(Debug)]
struct CircuitBreakerState {
    current_state: CircuitState,
    failure_count: usize,
    success_count: usize,
    last_failure_time: Option<Instant>,
    request_history: Vec<bool>,
}

impl CircuitBreaker {
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            config,
            state: Arc::new(RwLock::new(CircuitBreakerState {
                current_state: CircuitState::Closed,
                failure_count: 0,
                success_count: 0,
                last_failure_time: None,
                request_history: Vec::new(),
            })),
        }
    }

    /// Check if request is allowed
    pub fn allow_request(&self) -> Result<bool, Error> {
        let mut state = self
            .state
            .write()
            .map_err(|_| Error::Processing("Failed to acquire circuit breaker lock".into()))?;

        match state.current_state {
            CircuitState::Closed => Ok(true),
            CircuitState::Open => {
                // Check if timeout has elapsed
                if let Some(last_failure) = state.last_failure_time {
                    if last_failure.elapsed() >= self.config.timeout {
                        state.current_state = CircuitState::HalfOpen;
                        state.success_count = 0;
                        Ok(true)
                    } else {
                        Ok(false)
                    }
                } else {
                    Ok(false)
                }
            }
            CircuitState::HalfOpen => Ok(true),
        }
    }

    /// Record successful request
    pub fn record_success(&self) -> Result<(), Error> {
        let mut state = self
            .state
            .write()
            .map_err(|_| Error::Processing("Failed to acquire circuit breaker lock".into()))?;

        state.request_history.push(true);
        if state.request_history.len() > self.config.window_size {
            state.request_history.remove(0);
        }

        match state.current_state {
            CircuitState::Closed => {
                state.failure_count = 0;
            }
            CircuitState::HalfOpen => {
                state.success_count += 1;
                if state.success_count >= self.config.success_threshold {
                    state.current_state = CircuitState::Closed;
                    state.failure_count = 0;
                    state.success_count = 0;
                }
            }
            CircuitState::Open => {}
        }

        Ok(())
    }

    /// Record failed request
    pub fn record_failure(&self) -> Result<(), Error> {
        let mut state = self
            .state
            .write()
            .map_err(|_| Error::Processing("Failed to acquire circuit breaker lock".into()))?;

        state.request_history.push(false);
        if state.request_history.len() > self.config.window_size {
            state.request_history.remove(0);
        }

        state.last_failure_time = Some(Instant::now());
        state.failure_count += 1;

        match state.current_state {
            CircuitState::Closed => {
                if state.failure_count >= self.config.failure_threshold {
                    state.current_state = CircuitState::Open;
                }
            }
            CircuitState::HalfOpen => {
                state.current_state = CircuitState::Open;
                state.success_count = 0;
            }
            CircuitState::Open => {}
        }

        Ok(())
    }

    /// Get current circuit state
    pub fn state(&self) -> Result<CircuitState, Error> {
        let state = self
            .state
            .read()
            .map_err(|_| Error::Processing("Failed to read circuit breaker state".into()))?;
        Ok(state.current_state)
    }

    /// Get circuit breaker statistics
    pub fn stats(&self) -> Result<CircuitBreakerStats, Error> {
        let state = self
            .state
            .read()
            .map_err(|_| Error::Processing("Failed to read circuit breaker state".into()))?;

        let total_requests = state.request_history.len();
        let successful_requests = state.request_history.iter().filter(|&&x| x).count();
        let failed_requests = total_requests - successful_requests;

        let success_rate = if total_requests > 0 {
            successful_requests as f64 / total_requests as f64
        } else {
            0.0
        };

        Ok(CircuitBreakerStats {
            current_state: state.current_state,
            failure_count: state.failure_count,
            success_count: state.success_count,
            total_requests,
            successful_requests,
            failed_requests,
            success_rate,
        })
    }

    /// Reset circuit breaker
    pub fn reset(&self) -> Result<(), Error> {
        let mut state = self
            .state
            .write()
            .map_err(|_| Error::Processing("Failed to acquire circuit breaker lock".into()))?;

        state.current_state = CircuitState::Closed;
        state.failure_count = 0;
        state.success_count = 0;
        state.last_failure_time = None;
        state.request_history.clear();

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct CircuitBreakerStats {
    pub current_state: CircuitState,
    pub failure_count: usize,
    pub success_count: usize,
    pub total_requests: usize,
    pub successful_requests: usize,
    pub failed_requests: usize,
    pub success_rate: f64,
}

/// Fallback strategy for degraded operation
#[derive(Debug, Clone)]
pub enum FallbackStrategy {
    /// Use cached result if available
    UseCache,
    /// Use simplified synthesis
    SimplifiedSynthesis,
    /// Return pre-generated content
    PreGenerated,
    /// Custom fallback function
    Custom,
}

/// Quality degradation level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum QualityLevel {
    /// Full quality (100%)
    Full,
    /// High quality (75%)
    High,
    /// Medium quality (50%)
    Medium,
    /// Low quality (25%)
    Low,
    /// Minimum quality (10%)
    Minimum,
}

impl QualityLevel {
    pub fn quality_factor(&self) -> f64 {
        match self {
            Self::Full => 1.0,
            Self::High => 0.75,
            Self::Medium => 0.5,
            Self::Low => 0.25,
            Self::Minimum => 0.1,
        }
    }
}

/// Graceful degradation manager
#[derive(Debug, Clone)]
pub struct GracefulDegradationManager {
    circuit_breaker: CircuitBreaker,
    fallback_strategy: FallbackStrategy,
    current_quality: Arc<RwLock<QualityLevel>>,
}

impl GracefulDegradationManager {
    pub fn new(config: CircuitBreakerConfig, fallback: FallbackStrategy) -> Self {
        Self {
            circuit_breaker: CircuitBreaker::new(config),
            fallback_strategy: fallback,
            current_quality: Arc::new(RwLock::new(QualityLevel::Full)),
        }
    }

    /// Execute operation with circuit breaker protection
    pub async fn execute<F, T>(&self, operation: F) -> Result<T, Error>
    where
        F: FnOnce() -> Result<T, Error>,
    {
        if !self.circuit_breaker.allow_request()? {
            return Err(Error::Processing("Circuit breaker open".into()));
        }

        match operation() {
            Ok(result) => {
                self.circuit_breaker.record_success()?;
                Ok(result)
            }
            Err(e) => {
                self.circuit_breaker.record_failure()?;
                Err(e)
            }
        }
    }

    /// Get current quality level
    pub fn current_quality(&self) -> Result<QualityLevel, Error> {
        let quality = self
            .current_quality
            .read()
            .map_err(|_| Error::Processing("Failed to read quality level".into()))?;
        Ok(*quality)
    }

    /// Degrade quality level
    pub fn degrade_quality(&self) -> Result<QualityLevel, Error> {
        let mut quality = self
            .current_quality
            .write()
            .map_err(|_| Error::Processing("Failed to acquire quality lock".into()))?;

        *quality = match *quality {
            QualityLevel::Full => QualityLevel::High,
            QualityLevel::High => QualityLevel::Medium,
            QualityLevel::Medium => QualityLevel::Low,
            QualityLevel::Low => QualityLevel::Minimum,
            QualityLevel::Minimum => QualityLevel::Minimum,
        };

        Ok(*quality)
    }

    /// Restore quality level
    pub fn restore_quality(&self) -> Result<QualityLevel, Error> {
        let mut quality = self
            .current_quality
            .write()
            .map_err(|_| Error::Processing("Failed to acquire quality lock".into()))?;

        *quality = match *quality {
            QualityLevel::Full => QualityLevel::Full,
            QualityLevel::High => QualityLevel::Full,
            QualityLevel::Medium => QualityLevel::High,
            QualityLevel::Low => QualityLevel::Medium,
            QualityLevel::Minimum => QualityLevel::Low,
        };

        Ok(*quality)
    }

    /// Check if system is degraded
    pub fn is_degraded(&self) -> Result<bool, Error> {
        let quality = self.current_quality()?;
        Ok(quality != QualityLevel::Full)
    }

    /// Get fallback strategy
    pub fn fallback_strategy(&self) -> &FallbackStrategy {
        &self.fallback_strategy
    }

    /// Get circuit breaker stats
    pub fn circuit_stats(&self) -> Result<CircuitBreakerStats, Error> {
        self.circuit_breaker.stats()
    }

    /// Reset degradation
    pub fn reset(&self) -> Result<(), Error> {
        self.circuit_breaker.reset()?;
        let mut quality = self
            .current_quality
            .write()
            .map_err(|_| Error::Processing("Failed to acquire quality lock".into()))?;
        *quality = QualityLevel::Full;
        Ok(())
    }
}

/// Retry policy for transient failures
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    pub max_attempts: usize,
    pub initial_delay: Duration,
    pub max_delay: Duration,
    pub multiplier: f64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            multiplier: 2.0,
        }
    }
}

impl RetryPolicy {
    pub fn exponential_backoff() -> Self {
        Self {
            max_attempts: 5,
            initial_delay: Duration::from_millis(50),
            max_delay: Duration::from_secs(30),
            multiplier: 2.0,
        }
    }

    pub fn aggressive() -> Self {
        Self {
            max_attempts: 10,
            initial_delay: Duration::from_millis(10),
            max_delay: Duration::from_secs(5),
            multiplier: 1.5,
        }
    }

    /// Calculate delay for attempt number
    pub fn delay_for_attempt(&self, attempt: usize) -> Duration {
        if attempt == 0 {
            return Duration::ZERO;
        }

        let delay_ms =
            self.initial_delay.as_millis() as f64 * self.multiplier.powi((attempt - 1) as i32);

        let delay = Duration::from_millis(delay_ms as u64);
        delay.min(self.max_delay)
    }
}

/// Retry executor with exponential backoff
pub struct RetryExecutor {
    policy: RetryPolicy,
}

impl RetryExecutor {
    pub fn new(policy: RetryPolicy) -> Self {
        Self { policy }
    }

    /// Execute operation with retry logic
    pub async fn execute<F, T>(&self, mut operation: F) -> Result<T, Error>
    where
        F: FnMut() -> Result<T, Error>,
    {
        let mut last_error = None;

        for attempt in 0..self.policy.max_attempts {
            match operation() {
                Ok(result) => return Ok(result),
                Err(e) => {
                    last_error = Some(e);

                    if attempt + 1 < self.policy.max_attempts {
                        let delay = self.policy.delay_for_attempt(attempt + 1);
                        tokio::time::sleep(delay).await;
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| Error::Processing("All retry attempts failed".into())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_breaker_closed_state() {
        let cb = CircuitBreaker::new(CircuitBreakerConfig::default());

        assert_eq!(cb.state().unwrap(), CircuitState::Closed);
        assert!(cb.allow_request().unwrap());
    }

    #[test]
    fn test_circuit_breaker_opens_on_failures() {
        let config = CircuitBreakerConfig {
            failure_threshold: 3,
            ..Default::default()
        };
        let cb = CircuitBreaker::new(config);

        // Record failures
        for _ in 0..3 {
            cb.record_failure().unwrap();
        }

        assert_eq!(cb.state().unwrap(), CircuitState::Open);
        assert!(!cb.allow_request().unwrap());
    }

    #[test]
    fn test_circuit_breaker_half_open_recovery() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            success_threshold: 2,
            timeout: Duration::from_millis(10),
            ..Default::default()
        };
        let cb = CircuitBreaker::new(config);

        // Open circuit
        cb.record_failure().unwrap();
        cb.record_failure().unwrap();
        assert_eq!(cb.state().unwrap(), CircuitState::Open);

        // Wait for timeout
        std::thread::sleep(Duration::from_millis(20));

        // Should transition to half-open
        assert!(cb.allow_request().unwrap());

        // Record successes to close circuit
        cb.record_success().unwrap();
        cb.record_success().unwrap();

        assert_eq!(cb.state().unwrap(), CircuitState::Closed);
    }

    #[test]
    fn test_circuit_breaker_stats() {
        let cb = CircuitBreaker::new(CircuitBreakerConfig::default());

        cb.record_success().unwrap();
        cb.record_success().unwrap();
        cb.record_failure().unwrap();

        let stats = cb.stats().unwrap();
        assert_eq!(stats.total_requests, 3);
        assert_eq!(stats.successful_requests, 2);
        assert_eq!(stats.failed_requests, 1);
        assert!((stats.success_rate - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_graceful_degradation_quality_levels() {
        let manager = GracefulDegradationManager::new(
            CircuitBreakerConfig::default(),
            FallbackStrategy::SimplifiedSynthesis,
        );

        assert_eq!(manager.current_quality().unwrap(), QualityLevel::Full);

        manager.degrade_quality().unwrap();
        assert_eq!(manager.current_quality().unwrap(), QualityLevel::High);

        manager.degrade_quality().unwrap();
        assert_eq!(manager.current_quality().unwrap(), QualityLevel::Medium);

        manager.restore_quality().unwrap();
        assert_eq!(manager.current_quality().unwrap(), QualityLevel::High);
    }

    #[test]
    fn test_quality_level_factors() {
        assert_eq!(QualityLevel::Full.quality_factor(), 1.0);
        assert_eq!(QualityLevel::High.quality_factor(), 0.75);
        assert_eq!(QualityLevel::Medium.quality_factor(), 0.5);
        assert_eq!(QualityLevel::Low.quality_factor(), 0.25);
        assert_eq!(QualityLevel::Minimum.quality_factor(), 0.1);
    }

    #[tokio::test]
    async fn test_graceful_degradation_execute() {
        let manager = GracefulDegradationManager::new(
            CircuitBreakerConfig::default(),
            FallbackStrategy::UseCache,
        );

        let result = manager.execute(|| Ok(42)).await;
        assert_eq!(result.unwrap(), 42);

        let stats = manager.circuit_stats().unwrap();
        assert_eq!(stats.successful_requests, 1);
    }

    #[test]
    fn test_retry_policy_delays() {
        let policy = RetryPolicy::default();

        assert_eq!(policy.delay_for_attempt(0), Duration::ZERO);
        assert_eq!(policy.delay_for_attempt(1), Duration::from_millis(100));
        assert_eq!(policy.delay_for_attempt(2), Duration::from_millis(200));
        assert_eq!(policy.delay_for_attempt(3), Duration::from_millis(400));
    }

    #[tokio::test]
    async fn test_retry_executor_success() {
        let executor = RetryExecutor::new(RetryPolicy::default());

        let mut attempt = 0;
        let result = executor
            .execute(|| {
                attempt += 1;
                if attempt < 3 {
                    Err(Error::Processing("Transient failure".into()))
                } else {
                    Ok(42)
                }
            })
            .await;

        assert_eq!(result.unwrap(), 42);
    }

    #[tokio::test]
    async fn test_retry_executor_max_attempts() {
        let policy = RetryPolicy {
            max_attempts: 3,
            initial_delay: Duration::from_millis(1),
            ..Default::default()
        };
        let executor = RetryExecutor::new(policy);

        let mut attempt = 0;
        let result: Result<i32, Error> = executor
            .execute(|| {
                attempt += 1;
                Err(Error::Processing("Permanent failure".into()))
            })
            .await;

        assert!(result.is_err());
        assert_eq!(attempt, 3);
    }

    #[test]
    fn test_circuit_breaker_reset() {
        let cb = CircuitBreaker::new(CircuitBreakerConfig::default());

        cb.record_failure().unwrap();
        cb.record_failure().unwrap();

        cb.reset().unwrap();

        assert_eq!(cb.state().unwrap(), CircuitState::Closed);
        let stats = cb.stats().unwrap();
        assert_eq!(stats.failure_count, 0);
        assert_eq!(stats.total_requests, 0);
    }
}

//! Production hardening features for reliable acoustic synthesis
//!
//! This module provides enterprise-grade features for production deployments:
//! - Resource limits and quota management
//! - Graceful degradation under load
//! - Circuit breaker pattern for fault tolerance
//! - Retry logic with exponential backoff
//! - Health checks and system monitoring
//! - Rate limiting for request throttling

use crate::{AcousticError, Result};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Resource limits for acoustic synthesis operations
#[derive(Debug, Clone)]
pub struct ResourceLimits {
    /// Maximum memory usage in bytes
    pub max_memory_bytes: usize,
    /// Maximum concurrent operations
    pub max_concurrent_ops: usize,
    /// Maximum sequence length (phonemes)
    pub max_sequence_length: usize,
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Operation timeout
    pub operation_timeout: Duration,
    /// Maximum CPU usage percentage (0.0-1.0)
    pub max_cpu_usage: f32,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_memory_bytes: 2 * 1024 * 1024 * 1024, // 2GB
            max_concurrent_ops: 10,
            max_sequence_length: 1000,
            max_batch_size: 32,
            operation_timeout: Duration::from_secs(30),
            max_cpu_usage: 0.9, // 90%
        }
    }
}

impl ResourceLimits {
    /// Create conservative resource limits for constrained environments
    pub fn conservative() -> Self {
        Self {
            max_memory_bytes: 512 * 1024 * 1024, // 512MB
            max_concurrent_ops: 2,
            max_sequence_length: 200,
            max_batch_size: 4,
            operation_timeout: Duration::from_secs(10),
            max_cpu_usage: 0.7, // 70%
        }
    }

    /// Create aggressive resource limits for high-performance systems
    pub fn aggressive() -> Self {
        Self {
            max_memory_bytes: 8 * 1024 * 1024 * 1024, // 8GB
            max_concurrent_ops: 50,
            max_sequence_length: 5000,
            max_batch_size: 128,
            operation_timeout: Duration::from_secs(60),
            max_cpu_usage: 0.95, // 95%
        }
    }

    /// Validate that current usage is within limits
    pub fn validate(
        &self,
        memory_bytes: usize,
        concurrent_ops: usize,
        sequence_length: usize,
        batch_size: usize,
    ) -> Result<()> {
        if memory_bytes > self.max_memory_bytes {
            return Err(AcousticError::ProcessingError {
                message: format!(
                    "Memory limit exceeded: {} bytes (max: {})",
                    memory_bytes, self.max_memory_bytes
                ),
            });
        }

        if concurrent_ops > self.max_concurrent_ops {
            return Err(AcousticError::ProcessingError {
                message: format!(
                    "Concurrent operation limit exceeded: {} (max: {})",
                    concurrent_ops, self.max_concurrent_ops
                ),
            });
        }

        if sequence_length > self.max_sequence_length {
            return Err(AcousticError::InputError {
                message: format!(
                    "Sequence length exceeds limit: {} phonemes (max: {})",
                    sequence_length, self.max_sequence_length
                ),
            });
        }

        if batch_size > self.max_batch_size {
            return Err(AcousticError::InputError {
                message: format!(
                    "Batch size exceeds limit: {} (max: {})",
                    batch_size, self.max_batch_size
                ),
            });
        }

        Ok(())
    }
}

/// Circuit breaker state for fault tolerance
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Circuit is closed, operations proceed normally
    Closed,
    /// Circuit is open, operations are blocked
    Open,
    /// Circuit is half-open, testing if system recovered
    HalfOpen,
}

/// Circuit breaker for preventing cascading failures
pub struct CircuitBreaker {
    state: Arc<AtomicUsize>, // Current state (0=Closed, 1=Open, 2=HalfOpen)
    failure_count: Arc<AtomicUsize>,
    success_count: Arc<AtomicUsize>,
    failure_threshold: usize,
    success_threshold: usize,
    timeout: Duration,
    open_time: Arc<AtomicU64>, // Time when circuit opened (in nanoseconds)
}

impl CircuitBreaker {
    /// Create new circuit breaker
    pub fn new(failure_threshold: usize, success_threshold: usize, timeout: Duration) -> Self {
        Self {
            state: Arc::new(AtomicUsize::new(CircuitState::Closed as usize)),
            failure_count: Arc::new(AtomicUsize::new(0)),
            success_count: Arc::new(AtomicUsize::new(0)),
            failure_threshold,
            success_threshold,
            timeout,
            open_time: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Check if operation should proceed
    pub fn should_proceed(&self) -> bool {
        let current_state = self.get_state();

        match current_state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                // Check if timeout elapsed
                let open_time_nanos = self.open_time.load(Ordering::Acquire);
                if open_time_nanos == 0 {
                    return false;
                }

                let now_nanos = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos() as u64;

                let elapsed_nanos = now_nanos.saturating_sub(open_time_nanos);
                let elapsed = Duration::from_nanos(elapsed_nanos);

                if elapsed >= self.timeout {
                    // Try half-open
                    self.set_state(CircuitState::HalfOpen);
                    true
                } else {
                    false
                }
            }
            CircuitState::HalfOpen => true,
        }
    }

    /// Record successful operation
    pub fn record_success(&self) {
        let state = self.get_state();

        match state {
            CircuitState::Closed => {
                self.failure_count.store(0, Ordering::Release);
            }
            CircuitState::HalfOpen => {
                let success_count = self.success_count.fetch_add(1, Ordering::AcqRel) + 1;
                if success_count >= self.success_threshold {
                    self.set_state(CircuitState::Closed);
                    self.success_count.store(0, Ordering::Release);
                    self.failure_count.store(0, Ordering::Release);
                }
            }
            CircuitState::Open => {}
        }
    }

    /// Record failed operation
    pub fn record_failure(&self) {
        let state = self.get_state();

        match state {
            CircuitState::Closed => {
                let failure_count = self.failure_count.fetch_add(1, Ordering::AcqRel) + 1;
                if failure_count >= self.failure_threshold {
                    self.set_state(CircuitState::Open);
                }
            }
            CircuitState::HalfOpen => {
                self.set_state(CircuitState::Open);
                self.success_count.store(0, Ordering::Release);
            }
            CircuitState::Open => {}
        }
    }

    /// Get current circuit state
    pub fn get_state(&self) -> CircuitState {
        let state_value = self.state.load(Ordering::Acquire);
        match state_value {
            0 => CircuitState::Closed,
            1 => CircuitState::Open,
            2 => CircuitState::HalfOpen,
            _ => CircuitState::Closed,
        }
    }

    /// Set circuit state
    fn set_state(&self, state: CircuitState) {
        self.state.store(state as usize, Ordering::Release);

        if state == CircuitState::Open {
            // Record when circuit opened
            let now_nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64;
            self.open_time.store(now_nanos, Ordering::Release);
        }
    }

    /// Reset circuit breaker
    pub fn reset(&self) {
        self.set_state(CircuitState::Closed);
        self.failure_count.store(0, Ordering::Release);
        self.success_count.store(0, Ordering::Release);
        self.open_time.store(0, Ordering::Release);
    }
}

/// Retry policy with exponential backoff
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    /// Maximum number of retry attempts
    pub max_attempts: usize,
    /// Initial delay before first retry
    pub initial_delay: Duration,
    /// Maximum delay between retries
    pub max_delay: Duration,
    /// Backoff multiplier (typically 2.0)
    pub backoff_multiplier: f64,
    /// Whether to add jitter to delays
    pub jitter: bool,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(5),
            backoff_multiplier: 2.0,
            jitter: true,
        }
    }
}

impl RetryPolicy {
    /// Create no-retry policy
    pub fn no_retry() -> Self {
        Self {
            max_attempts: 1,
            ..Default::default()
        }
    }

    /// Create aggressive retry policy
    pub fn aggressive() -> Self {
        Self {
            max_attempts: 5,
            initial_delay: Duration::from_millis(50),
            max_delay: Duration::from_secs(10),
            backoff_multiplier: 1.5,
            jitter: true,
        }
    }

    /// Calculate delay for given attempt
    pub fn delay_for_attempt(&self, attempt: usize) -> Duration {
        if attempt == 0 {
            return Duration::ZERO;
        }

        let mut delay =
            self.initial_delay.as_secs_f64() * self.backoff_multiplier.powi((attempt - 1) as i32);
        delay = delay.min(self.max_delay.as_secs_f64());

        if self.jitter {
            // Add ±25% jitter using deterministic calculation
            let jitter_factor = 0.75 + (0.5 * ((attempt as f64 * 12345.0) % 1.0));
            delay *= jitter_factor;
        }

        Duration::from_secs_f64(delay)
    }

    /// Execute operation with retries
    pub async fn execute<F, T, E>(&self, mut operation: F) -> std::result::Result<T, E>
    where
        F: FnMut() -> std::result::Result<T, E>,
    {
        for attempt in 0..self.max_attempts {
            match operation() {
                Ok(result) => return Ok(result),
                Err(err) => {
                    if attempt + 1 >= self.max_attempts {
                        return Err(err);
                    }

                    let delay = self.delay_for_attempt(attempt + 1);
                    tokio::time::sleep(delay).await;
                }
            }
        }

        unreachable!("Retry loop should not exit without return");
    }
}

/// Rate limiter for request throttling
pub struct RateLimiter {
    tokens: Arc<AtomicUsize>,
    max_tokens: usize,
    refill_rate: Duration,
    last_refill: Arc<AtomicU64>,
}

impl RateLimiter {
    /// Create new rate limiter
    pub fn new(max_requests: usize, window: Duration) -> Self {
        let refill_rate = if max_requests > 0 {
            window / max_requests as u32
        } else {
            Duration::from_secs(1)
        };

        Self {
            tokens: Arc::new(AtomicUsize::new(max_requests)),
            max_tokens: max_requests,
            refill_rate,
            last_refill: Arc::new(AtomicU64::new(Instant::now().elapsed().as_nanos() as u64)),
        }
    }

    /// Check if request should be allowed
    pub fn allow(&self) -> bool {
        self.refill_tokens();

        let tokens = self.tokens.load(Ordering::Acquire);
        if tokens > 0 {
            self.tokens.fetch_sub(1, Ordering::AcqRel);
            true
        } else {
            false
        }
    }

    /// Refill tokens based on elapsed time
    fn refill_tokens(&self) {
        let now = Instant::now().elapsed().as_nanos() as u64;
        let last_refill = self.last_refill.load(Ordering::Acquire);
        let elapsed_nanos = now.saturating_sub(last_refill);
        let elapsed = Duration::from_nanos(elapsed_nanos);

        let tokens_to_add = (elapsed.as_secs_f64() / self.refill_rate.as_secs_f64()) as usize;

        if tokens_to_add > 0 {
            let current_tokens = self.tokens.load(Ordering::Acquire);
            let new_tokens = (current_tokens + tokens_to_add).min(self.max_tokens);
            self.tokens.store(new_tokens, Ordering::Release);
            self.last_refill.store(now, Ordering::Release);
        }
    }

    /// Get current number of available tokens
    pub fn available_tokens(&self) -> usize {
        self.refill_tokens();
        self.tokens.load(Ordering::Acquire)
    }
}

/// Health check status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    /// System is healthy
    Healthy,
    /// System is degraded but functional
    Degraded,
    /// System is unhealthy
    Unhealthy,
}

/// Health checker for system monitoring
pub struct HealthChecker {
    is_healthy: Arc<AtomicBool>,
    last_check: Arc<AtomicU64>, // Nanoseconds since UNIX_EPOCH
    check_interval: Duration,
}

impl HealthChecker {
    /// Create new health checker
    pub fn new(check_interval: Duration) -> Self {
        Self {
            is_healthy: Arc::new(AtomicBool::new(true)),
            last_check: Arc::new(AtomicU64::new(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos() as u64,
            )),
            check_interval,
        }
    }

    /// Get current health status
    pub fn status(&self) -> HealthStatus {
        if self.is_healthy.load(Ordering::Acquire) {
            HealthStatus::Healthy
        } else {
            HealthStatus::Unhealthy
        }
    }

    /// Update health status
    pub fn update_status(&self, healthy: bool) {
        self.is_healthy.store(healthy, Ordering::Release);
        let now_nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        self.last_check.store(now_nanos, Ordering::Release);
    }

    /// Check if health check is due
    pub fn should_check(&self) -> bool {
        let now_nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        let last_check_nanos = self.last_check.load(Ordering::Acquire);
        let elapsed_nanos = now_nanos.saturating_sub(last_check_nanos);
        let elapsed = Duration::from_nanos(elapsed_nanos);
        elapsed >= self.check_interval
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_limits_validation() {
        let limits = ResourceLimits::default();

        // Valid usage
        assert!(limits.validate(1024 * 1024 * 1024, 5, 500, 16).is_ok());

        // Exceeded memory
        assert!(limits.validate(3 * 1024 * 1024 * 1024, 5, 500, 16).is_err());

        // Exceeded concurrent ops
        assert!(limits.validate(1024 * 1024 * 1024, 20, 500, 16).is_err());

        // Exceeded sequence length
        assert!(limits.validate(1024 * 1024 * 1024, 5, 2000, 16).is_err());

        // Exceeded batch size
        assert!(limits.validate(1024 * 1024 * 1024, 5, 500, 64).is_err());
    }

    #[test]
    fn test_resource_limits_presets() {
        let conservative = ResourceLimits::conservative();
        assert!(conservative.max_memory_bytes < ResourceLimits::default().max_memory_bytes);

        let aggressive = ResourceLimits::aggressive();
        assert!(aggressive.max_memory_bytes > ResourceLimits::default().max_memory_bytes);
    }

    #[test]
    fn test_circuit_breaker_closed() {
        let breaker = CircuitBreaker::new(3, 2, Duration::from_secs(1));

        assert_eq!(breaker.get_state(), CircuitState::Closed);
        assert!(breaker.should_proceed());

        breaker.record_success();
        assert_eq!(breaker.get_state(), CircuitState::Closed);
    }

    #[test]
    fn test_circuit_breaker_opens_on_failures() {
        let breaker = CircuitBreaker::new(3, 2, Duration::from_secs(1));

        breaker.record_failure();
        breaker.record_failure();
        assert_eq!(breaker.get_state(), CircuitState::Closed);

        breaker.record_failure();
        assert_eq!(breaker.get_state(), CircuitState::Open);
        assert!(!breaker.should_proceed());
    }

    #[test]
    fn test_circuit_breaker_reset() {
        let breaker = CircuitBreaker::new(2, 2, Duration::from_secs(1));

        breaker.record_failure();
        breaker.record_failure();
        assert_eq!(breaker.get_state(), CircuitState::Open);

        breaker.reset();
        assert_eq!(breaker.get_state(), CircuitState::Closed);
        assert!(breaker.should_proceed());
    }

    #[test]
    fn test_retry_policy_delays() {
        let policy = RetryPolicy::default();

        let delay0 = policy.delay_for_attempt(0);
        assert_eq!(delay0, Duration::ZERO);

        let delay1 = policy.delay_for_attempt(1);
        // With jitter, delay might be slightly less than initial_delay
        // Check that it's in reasonable range (75%-125% of initial_delay)
        let initial_ms = policy.initial_delay.as_millis();
        let delay1_ms = delay1.as_millis();
        assert!(delay1_ms >= (initial_ms * 75 / 100));
        assert!(delay1_ms <= (initial_ms * 125 / 100));

        let delay2 = policy.delay_for_attempt(2);
        // Delay should generally increase (though jitter might make it slightly smaller)
        assert!(delay2.as_millis() >= (delay1_ms * 80 / 100));
    }

    #[test]
    fn test_retry_policy_no_retry() {
        let policy = RetryPolicy::no_retry();
        assert_eq!(policy.max_attempts, 1);
    }

    #[test]
    fn test_rate_limiter_allows_requests() {
        let limiter = RateLimiter::new(5, Duration::from_secs(1));

        // Should allow first 5 requests
        assert!(limiter.allow());
        assert!(limiter.allow());
        assert!(limiter.allow());
        assert!(limiter.allow());
        assert!(limiter.allow());

        // Should deny 6th request
        assert!(!limiter.allow());
    }

    #[test]
    fn test_rate_limiter_available_tokens() {
        let limiter = RateLimiter::new(10, Duration::from_secs(1));
        assert_eq!(limiter.available_tokens(), 10);

        limiter.allow();
        assert_eq!(limiter.available_tokens(), 9);
    }

    #[test]
    fn test_health_checker_status() {
        let checker = HealthChecker::new(Duration::from_secs(60));

        assert_eq!(checker.status(), HealthStatus::Healthy);

        checker.update_status(false);
        assert_eq!(checker.status(), HealthStatus::Unhealthy);

        checker.update_status(true);
        assert_eq!(checker.status(), HealthStatus::Healthy);
    }

    #[test]
    fn test_health_checker_should_check() {
        let checker = HealthChecker::new(Duration::from_millis(100));

        // Update status first to set baseline
        checker.update_status(true);

        // Should not need check immediately after update
        assert!(!checker.should_check());

        // Wait for check interval to elapse
        std::thread::sleep(Duration::from_millis(150));
        assert!(checker.should_check());
    }
}

// Copyright (c) 2024 VoiRS Contributors
// Licensed under MIT OR Apache-2.0

//! Circuit breaker pattern implementation
//!
//! Prevents cascading failures by temporarily blocking requests to failing services.

use super::{CircuitBreakerConfig, HighAvailabilityError, Result};
use parking_lot::RwLock;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// Circuit breaker state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Circuit is closed - requests flow normally
    Closed,
    /// Circuit is open - requests are blocked
    Open,
    /// Circuit is half-open - limited requests allowed for testing
    HalfOpen,
}

/// Circuit breaker statistics
#[derive(Debug, Clone)]
pub struct CircuitBreakerStats {
    /// Total requests
    pub total_requests: u64,
    /// Successful requests
    pub successful_requests: u64,
    /// Failed requests
    pub failed_requests: u64,
    /// Number of times circuit opened
    pub open_count: u64,
    /// Current state
    pub current_state: CircuitState,
    /// Last state change time
    pub last_state_change: Option<Instant>,
}

/// Circuit breaker implementation
pub struct CircuitBreaker {
    config: CircuitBreakerConfig,
    state: Arc<RwLock<CircuitState>>,
    stats: Arc<RwLock<CircuitBreakerStats>>,
    consecutive_failures: Arc<RwLock<usize>>,
    consecutive_successes: Arc<RwLock<usize>>,
    last_failure_time: Arc<RwLock<Option<Instant>>>,
}

impl CircuitBreaker {
    /// Create a new circuit breaker
    #[must_use]
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            config,
            state: Arc::new(RwLock::new(CircuitState::Closed)),
            stats: Arc::new(RwLock::new(CircuitBreakerStats {
                total_requests: 0,
                successful_requests: 0,
                failed_requests: 0,
                open_count: 0,
                current_state: CircuitState::Closed,
                last_state_change: Some(Instant::now()),
            })),
            consecutive_failures: Arc::new(RwLock::new(0)),
            consecutive_successes: Arc::new(RwLock::new(0)),
            last_failure_time: Arc::new(RwLock::new(None)),
        }
    }

    /// Check if request is allowed
    pub fn is_request_allowed(&self) -> Result<()> {
        let mut state = self.state.write();
        let mut statistics = self.stats.write();

        match *state {
            CircuitState::Closed => {
                statistics.total_requests += 1;
                Ok(())
            }
            CircuitState::Open => {
                // Check if timeout has passed
                if let Some(last_failure) = *self.last_failure_time.read() {
                    if last_failure.elapsed() > self.config.timeout {
                        info!("Circuit breaker transitioning to half-open state");
                        *state = CircuitState::HalfOpen;
                        statistics.current_state = CircuitState::HalfOpen;
                        statistics.last_state_change = Some(Instant::now());
                        statistics.total_requests += 1;
                        Ok(())
                    } else {
                        Err(HighAvailabilityError::CircuitBreakerOpen)
                    }
                } else {
                    Err(HighAvailabilityError::CircuitBreakerOpen)
                }
            }
            CircuitState::HalfOpen => {
                statistics.total_requests += 1;
                Ok(())
            }
        }
    }

    /// Record successful request
    pub fn record_success(&self) {
        {
            let mut stats = self.stats.write();
            stats.successful_requests += 1;
        }

        {
            let mut consecutive_failures = self.consecutive_failures.write();
            *consecutive_failures = 0;
        }

        let new_successes = {
            let mut consecutive_successes = self.consecutive_successes.write();
            *consecutive_successes += 1;
            *consecutive_successes
        };

        let current_state = *self.state.read();

        if current_state == CircuitState::HalfOpen && new_successes >= self.config.success_threshold
        {
            self.close_circuit();
        }
    }

    /// Record failed request
    pub fn record_failure(&self) {
        {
            let mut stats = self.stats.write();
            stats.failed_requests += 1;
        }

        {
            let mut consecutive_successes = self.consecutive_successes.write();
            *consecutive_successes = 0;
        }

        let new_failures = {
            let mut consecutive_failures = self.consecutive_failures.write();
            *consecutive_failures += 1;
            *consecutive_failures
        };

        *self.last_failure_time.write() = Some(Instant::now());

        let current_state = *self.state.read();

        if (current_state == CircuitState::Closed || current_state == CircuitState::HalfOpen)
            && new_failures >= self.config.failure_threshold
        {
            self.open_circuit();
        }
    }

    /// Open the circuit
    fn open_circuit(&self) {
        warn!("Opening circuit breaker due to consecutive failures");
        let mut state = self.state.write();
        *state = CircuitState::Open;

        let mut stats = self.stats.write();
        stats.current_state = CircuitState::Open;
        stats.open_count += 1;
        stats.last_state_change = Some(Instant::now());
    }

    /// Close the circuit
    fn close_circuit(&self) {
        info!("Closing circuit breaker - service recovered");
        let mut state = self.state.write();
        *state = CircuitState::Closed;

        let mut stats = self.stats.write();
        stats.current_state = CircuitState::Closed;
        stats.last_state_change = Some(Instant::now());

        *self.consecutive_failures.write() = 0;
        *self.consecutive_successes.write() = 0;
    }

    /// Get current circuit state
    #[must_use]
    pub fn get_state(&self) -> CircuitState {
        *self.state.read()
    }

    /// Get circuit breaker statistics
    #[must_use]
    pub fn get_stats(&self) -> CircuitBreakerStats {
        self.stats.read().clone()
    }

    /// Reset circuit breaker
    pub fn reset(&self) {
        info!("Resetting circuit breaker");
        *self.state.write() = CircuitState::Closed;
        *self.consecutive_failures.write() = 0;
        *self.consecutive_successes.write() = 0;
        *self.last_failure_time.write() = None;

        let mut stats = self.stats.write();
        stats.current_state = CircuitState::Closed;
        stats.last_state_change = Some(Instant::now());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_breaker_initial_state() {
        let config = CircuitBreakerConfig::default();
        let cb = CircuitBreaker::new(config);

        assert_eq!(cb.get_state(), CircuitState::Closed);
        let stats = cb.get_stats();
        assert_eq!(stats.total_requests, 0);
        assert_eq!(stats.open_count, 0);
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
            cb.record_failure();
        }

        assert_eq!(cb.get_state(), CircuitState::Open);
        let stats = cb.get_stats();
        assert_eq!(stats.failed_requests, 3);
        assert_eq!(stats.open_count, 1);
    }

    #[test]
    fn test_circuit_breaker_stays_closed_on_success() {
        let config = CircuitBreakerConfig::default();
        let cb = CircuitBreaker::new(config);

        for _ in 0..10 {
            cb.record_success();
        }

        assert_eq!(cb.get_state(), CircuitState::Closed);
        let stats = cb.get_stats();
        assert_eq!(stats.successful_requests, 10);
    }

    #[test]
    fn test_circuit_breaker_half_open_transition() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            timeout: Duration::from_millis(100),
            ..Default::default()
        };
        let cb = CircuitBreaker::new(config);

        // Open circuit
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.get_state(), CircuitState::Open);

        // Wait for timeout
        std::thread::sleep(Duration::from_millis(150));

        // Next request should transition to half-open
        let result = cb.is_request_allowed();
        assert!(result.is_ok());
        assert_eq!(cb.get_state(), CircuitState::HalfOpen);
    }

    #[test]
    fn test_circuit_breaker_closes_from_half_open() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            success_threshold: 2,
            ..Default::default()
        };
        let cb = CircuitBreaker::new(config);

        // Open circuit
        cb.record_failure();
        cb.record_failure();

        // Manually transition to half-open
        *cb.state.write() = CircuitState::HalfOpen;

        // Record successes
        cb.record_success();
        cb.record_success();

        assert_eq!(cb.get_state(), CircuitState::Closed);
    }

    #[test]
    fn test_circuit_breaker_reset() {
        let config = CircuitBreakerConfig::default();
        let cb = CircuitBreaker::new(config);

        // Record some failures
        for _ in 0..5 {
            cb.record_failure();
        }

        // Reset
        cb.reset();

        assert_eq!(cb.get_state(), CircuitState::Closed);
        let stats = cb.get_stats();
        assert_eq!(stats.current_state, CircuitState::Closed);
    }

    #[test]
    fn test_circuit_breaker_blocks_requests_when_open() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            timeout: Duration::from_secs(3600), // Long timeout
            ..Default::default()
        };
        let cb = CircuitBreaker::new(config);

        // Open circuit
        cb.record_failure();
        cb.record_failure();

        // Requests should be blocked
        let result = cb.is_request_allowed();
        assert!(result.is_err());
    }
}

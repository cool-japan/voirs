//! Error recovery strategies and utilities.
//!
//! This module provides comprehensive error recovery mechanisms including:
//! - Automatic retry logic with exponential backoff
//! - Circuit breaker patterns for failing components
//! - Fallback strategies for voice and model selection
//! - Recovery state management and metrics

use super::types::{Result, VoirsError};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tokio::time::sleep;

/// Recovery strategy for handling errors
#[derive(Debug, Clone)]
pub enum RecoveryStrategy {
    /// No recovery - fail immediately
    FailFast,
    /// Retry with linear backoff
    RetryLinear { max_attempts: u32, delay: Duration },
    /// Retry with exponential backoff
    RetryExponential {
        max_attempts: u32,
        initial_delay: Duration,
        max_delay: Duration,
        multiplier: f64,
    },
    /// Circuit breaker pattern
    CircuitBreaker {
        failure_threshold: u32,
        timeout: Duration,
        half_open_max_calls: u32,
    },
    /// Fallback to alternative
    Fallback { alternatives: Vec<String> },
    /// Custom recovery function
    Custom { name: String },
}

impl Default for RecoveryStrategy {
    fn default() -> Self {
        Self::RetryExponential {
            max_attempts: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            multiplier: 2.0,
        }
    }
}

/// Recovery context with execution state
#[derive(Debug, Clone)]
pub struct RecoveryContext {
    /// Current attempt number
    pub attempt: u32,
    /// Total attempts made
    pub total_attempts: u32,
    /// Time of first attempt
    pub start_time: Instant,
    /// Previous errors encountered
    pub previous_errors: Vec<VoirsError>,
    /// Recovery metadata
    pub metadata: HashMap<String, String>,
}

impl Default for RecoveryContext {
    fn default() -> Self {
        Self {
            attempt: 0,
            total_attempts: 0,
            start_time: Instant::now(),
            previous_errors: Vec::new(),
            metadata: HashMap::new(),
        }
    }
}

/// Circuit breaker state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    Closed,   // Normal operation
    Open,     // Failing, blocking requests
    HalfOpen, // Testing if service recovered
}

/// Circuit breaker for protecting against cascading failures
#[derive(Debug)]
pub struct CircuitBreaker {
    state: Arc<Mutex<CircuitBreakerState>>,
    config: CircuitBreakerConfig,
}

#[derive(Debug)]
struct CircuitBreakerState {
    state: CircuitState,
    failure_count: u32,
    last_failure_time: Option<Instant>,
    half_open_calls: u32,
}

#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    pub failure_threshold: u32,
    pub timeout: Duration,
    pub half_open_max_calls: u32,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            timeout: Duration::from_secs(60),
            half_open_max_calls: 3,
        }
    }
}

impl CircuitBreaker {
    /// Create a new circuit breaker
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            state: Arc::new(Mutex::new(CircuitBreakerState {
                state: CircuitState::Closed,
                failure_count: 0,
                last_failure_time: None,
                half_open_calls: 0,
            })),
            config,
        }
    }

    /// Execute operation with circuit breaker protection
    pub async fn call<F, T, E>(
        &self,
        operation: F,
    ) -> std::result::Result<T, CircuitBreakerError<E>>
    where
        F: std::future::Future<Output = std::result::Result<T, E>>,
        E: std::error::Error + Send + Sync + 'static,
    {
        // Check if circuit is open
        {
            let mut state = self.state.lock().unwrap();
            match state.state {
                CircuitState::Open => {
                    if let Some(last_failure) = state.last_failure_time {
                        if last_failure.elapsed() > self.config.timeout {
                            state.state = CircuitState::HalfOpen;
                            state.half_open_calls = 0;
                        } else {
                            return Err(CircuitBreakerError::CircuitOpen);
                        }
                    }
                }
                CircuitState::HalfOpen => {
                    if state.half_open_calls >= self.config.half_open_max_calls {
                        return Err(CircuitBreakerError::CircuitOpen);
                    }
                    state.half_open_calls += 1;
                }
                CircuitState::Closed => {}
            }
        }

        // Execute operation
        match operation.await {
            Ok(result) => {
                // Success - reset circuit breaker if in half-open state
                let mut state = self.state.lock().unwrap();
                if state.state == CircuitState::HalfOpen {
                    state.state = CircuitState::Closed;
                    state.failure_count = 0;
                    state.last_failure_time = None;
                }
                Ok(result)
            }
            Err(error) => {
                // Failure - update circuit breaker state
                let mut state = self.state.lock().unwrap();
                state.failure_count += 1;
                state.last_failure_time = Some(Instant::now());

                if state.failure_count >= self.config.failure_threshold {
                    state.state = CircuitState::Open;
                }

                Err(CircuitBreakerError::OperationFailed(error))
            }
        }
    }

    /// Get current circuit state
    pub fn state(&self) -> CircuitState {
        self.state.lock().unwrap().state
    }

    /// Reset circuit breaker to closed state
    pub fn reset(&self) {
        let mut state = self.state.lock().unwrap();
        state.state = CircuitState::Closed;
        state.failure_count = 0;
        state.last_failure_time = None;
        state.half_open_calls = 0;
    }
}

/// Circuit breaker error types
#[derive(Debug, thiserror::Error)]
pub enum CircuitBreakerError<E> {
    #[error("Circuit breaker is open")]
    CircuitOpen,
    #[error("Operation failed: {0}")]
    OperationFailed(E),
}

/// Error recovery manager
pub struct ErrorRecoveryManager {
    strategies: HashMap<String, RecoveryStrategy>,
    circuit_breakers: HashMap<String, Arc<CircuitBreaker>>,
    recovery_metrics: Arc<Mutex<RecoveryMetrics>>,
}

/// Recovery metrics for monitoring
#[derive(Debug, Clone, Default)]
pub struct RecoveryMetrics {
    pub total_attempts: u64,
    pub successful_recoveries: u64,
    pub failed_recoveries: u64,
    pub average_recovery_time: Duration,
    pub circuit_breaker_trips: u64,
}

impl ErrorRecoveryManager {
    /// Create a new error recovery manager
    pub fn new() -> Self {
        Self {
            strategies: HashMap::new(),
            circuit_breakers: HashMap::new(),
            recovery_metrics: Arc::new(Mutex::new(RecoveryMetrics::default())),
        }
    }

    /// Register a recovery strategy for a component
    pub fn register_strategy(&mut self, component: impl Into<String>, strategy: RecoveryStrategy) {
        self.strategies.insert(component.into(), strategy);
    }

    /// Register a circuit breaker for a component
    pub fn register_circuit_breaker(
        &mut self,
        component: impl Into<String>,
        config: CircuitBreakerConfig,
    ) {
        let circuit_breaker = Arc::new(CircuitBreaker::new(config));
        self.circuit_breakers
            .insert(component.into(), circuit_breaker);
    }

    /// Execute operation with recovery strategy
    pub async fn execute_with_recovery<F, T>(&self, component: &str, operation: F) -> Result<T>
    where
        F: Fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<T>> + Send>>
            + Send
            + Sync,
    {
        let strategy = self.strategies.get(component).cloned().unwrap_or_default();

        let mut context = RecoveryContext::default();
        context
            .metadata
            .insert("component".to_string(), component.to_string());

        self.execute_with_strategy(operation, strategy, &mut context)
            .await
    }

    /// Execute operation with specific recovery strategy
    async fn execute_with_strategy<F, T>(
        &self,
        operation: F,
        strategy: RecoveryStrategy,
        context: &mut RecoveryContext,
    ) -> Result<T>
    where
        F: Fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<T>> + Send>>
            + Send
            + Sync,
    {
        match strategy {
            RecoveryStrategy::FailFast => operation().await,

            RecoveryStrategy::RetryLinear {
                max_attempts,
                delay,
            } => {
                self.retry_linear(operation, max_attempts, delay, context)
                    .await
            }

            RecoveryStrategy::RetryExponential {
                max_attempts,
                initial_delay,
                max_delay,
                multiplier,
            } => {
                self.retry_exponential(
                    operation,
                    max_attempts,
                    initial_delay,
                    max_delay,
                    multiplier,
                    context,
                )
                .await
            }

            RecoveryStrategy::CircuitBreaker { .. } => {
                // Circuit breaker is handled separately in execute_with_circuit_breaker
                operation().await
            }

            RecoveryStrategy::Fallback { alternatives } => {
                self.execute_with_fallback(operation, alternatives, context)
                    .await
            }

            RecoveryStrategy::Custom { name } => {
                // Custom strategies would be implemented by extending this method
                tracing::warn!(
                    "Custom recovery strategy '{}' not implemented, using default",
                    name
                );
                operation().await
            }
        }
    }

    /// Retry with linear backoff
    async fn retry_linear<F, T>(
        &self,
        operation: F,
        max_attempts: u32,
        delay: Duration,
        context: &mut RecoveryContext,
    ) -> Result<T>
    where
        F: Fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<T>> + Send>>
            + Send
            + Sync,
    {
        for attempt in 1..=max_attempts {
            context.attempt = attempt;
            context.total_attempts += 1;

            match operation().await {
                Ok(result) => {
                    self.record_successful_recovery(context.start_time.elapsed());
                    return Ok(result);
                }
                Err(error) => {
                    context.previous_errors.push(error.clone());

                    if attempt == max_attempts {
                        self.record_failed_recovery();
                        return Err(error);
                    }

                    if error.is_recoverable() {
                        tracing::warn!(
                            "Operation failed (attempt {}/{}), retrying in {:?}: {}",
                            attempt,
                            max_attempts,
                            delay,
                            error
                        );
                        sleep(delay).await;
                    } else {
                        self.record_failed_recovery();
                        return Err(error);
                    }
                }
            }
        }

        // This should never be reached due to the loop logic, but handle it gracefully
        self.record_failed_recovery();
        Err(context
            .previous_errors
            .last()
            .cloned()
            .unwrap_or_else(|| VoirsError::InternalError {
                component: "recovery".to_string(),
                message: "Linear retry operation failed without recording errors".to_string(),
            }))
    }

    /// Retry with exponential backoff
    async fn retry_exponential<F, T>(
        &self,
        operation: F,
        max_attempts: u32,
        initial_delay: Duration,
        max_delay: Duration,
        multiplier: f64,
        context: &mut RecoveryContext,
    ) -> Result<T>
    where
        F: Fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<T>> + Send>>
            + Send
            + Sync,
    {
        let mut current_delay = initial_delay;

        for attempt in 1..=max_attempts {
            context.attempt = attempt;
            context.total_attempts += 1;

            match operation().await {
                Ok(result) => {
                    self.record_successful_recovery(context.start_time.elapsed());
                    return Ok(result);
                }
                Err(error) => {
                    context.previous_errors.push(error.clone());

                    if attempt == max_attempts {
                        self.record_failed_recovery();
                        return Err(error);
                    }

                    if error.is_recoverable() {
                        tracing::warn!(
                            "Operation failed (attempt {}/{}), retrying in {:?}: {}",
                            attempt,
                            max_attempts,
                            current_delay,
                            error
                        );
                        sleep(current_delay).await;

                        // Calculate next delay with exponential backoff
                        current_delay = Duration::from_millis(
                            ((current_delay.as_millis() as f64) * multiplier) as u64,
                        )
                        .min(max_delay);
                    } else {
                        self.record_failed_recovery();
                        return Err(error);
                    }
                }
            }
        }

        // This should never be reached due to the loop logic, but handle it gracefully
        self.record_failed_recovery();
        Err(context
            .previous_errors
            .last()
            .cloned()
            .unwrap_or_else(|| VoirsError::InternalError {
                component: "recovery".to_string(),
                message: "Exponential retry operation failed without recording errors".to_string(),
            }))
    }

    /// Execute with fallback alternatives
    async fn execute_with_fallback<F, T>(
        &self,
        operation: F,
        alternatives: Vec<String>,
        context: &mut RecoveryContext,
    ) -> Result<T>
    where
        F: Fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<T>> + Send>>
            + Send
            + Sync,
    {
        // Try primary operation first
        match operation().await {
            Ok(result) => return Ok(result),
            Err(error) => {
                context.previous_errors.push(error.clone());

                if !error.is_recoverable() {
                    return Err(error);
                }
            }
        }

        // Try alternatives
        for (index, alternative) in alternatives.iter().enumerate() {
            context.attempt = index as u32 + 2; // +2 because we already tried primary
            context
                .metadata
                .insert("fallback_alternative".to_string(), alternative.clone());

            tracing::info!("Trying fallback alternative: {}", alternative);

            match operation().await {
                Ok(result) => {
                    tracing::info!("Fallback successful with alternative: {}", alternative);
                    self.record_successful_recovery(context.start_time.elapsed());
                    return Ok(result);
                }
                Err(error) => {
                    context.previous_errors.push(error.clone());
                    tracing::warn!(
                        "Fallback failed with alternative '{}': {}",
                        alternative,
                        error
                    );
                }
            }
        }

        // All alternatives failed
        self.record_failed_recovery();
        let last_error = context.previous_errors.last().unwrap().clone();
        Err(last_error)
    }

    /// Execute operation with circuit breaker protection
    pub async fn execute_with_circuit_breaker<F, T>(
        &self,
        component: &str,
        operation: F,
    ) -> std::result::Result<T, CircuitBreakerError<VoirsError>>
    where
        F: std::future::Future<Output = Result<T>>,
    {
        if let Some(circuit_breaker) = self.circuit_breakers.get(component) {
            circuit_breaker.call(operation).await
        } else {
            match operation.await {
                Ok(result) => Ok(result),
                Err(error) => Err(CircuitBreakerError::OperationFailed(error)),
            }
        }
    }

    /// Get circuit breaker state for component
    pub fn get_circuit_state(&self, component: &str) -> Option<CircuitState> {
        self.circuit_breakers.get(component).map(|cb| cb.state())
    }

    /// Reset circuit breaker for component
    pub fn reset_circuit_breaker(&self, component: &str) {
        if let Some(circuit_breaker) = self.circuit_breakers.get(component) {
            circuit_breaker.reset();
        }
    }

    /// Record successful recovery
    fn record_successful_recovery(&self, recovery_time: Duration) {
        if let Ok(mut metrics) = self.recovery_metrics.lock() {
            metrics.total_attempts += 1;
            metrics.successful_recoveries += 1;

            // Update average recovery time
            let total_time = metrics.average_recovery_time.as_nanos() as f64
                * (metrics.successful_recoveries - 1) as f64
                + recovery_time.as_nanos() as f64;
            metrics.average_recovery_time =
                Duration::from_nanos((total_time / metrics.successful_recoveries as f64) as u64);
        }
    }

    /// Record failed recovery
    fn record_failed_recovery(&self) {
        if let Ok(mut metrics) = self.recovery_metrics.lock() {
            metrics.total_attempts += 1;
            metrics.failed_recoveries += 1;
        }
    }

    /// Get recovery metrics
    pub fn get_metrics(&self) -> RecoveryMetrics {
        self.recovery_metrics.lock().unwrap().clone()
    }

    /// Reset recovery metrics
    pub fn reset_metrics(&self) {
        if let Ok(mut metrics) = self.recovery_metrics.lock() {
            *metrics = RecoveryMetrics::default();
        }
    }
}

impl Default for ErrorRecoveryManager {
    fn default() -> Self {
        let mut manager = Self::new();

        // Register default strategies for common components
        manager.register_strategy(
            "synthesis",
            RecoveryStrategy::RetryExponential {
                max_attempts: 3,
                initial_delay: Duration::from_millis(100),
                max_delay: Duration::from_secs(5),
                multiplier: 2.0,
            },
        );

        manager.register_strategy(
            "model_loading",
            RecoveryStrategy::RetryLinear {
                max_attempts: 5,
                delay: Duration::from_secs(1),
            },
        );

        manager.register_strategy(
            "network",
            RecoveryStrategy::RetryExponential {
                max_attempts: 5,
                initial_delay: Duration::from_millis(500),
                max_delay: Duration::from_secs(30),
                multiplier: 2.0,
            },
        );

        manager.register_circuit_breaker(
            "device",
            CircuitBreakerConfig {
                failure_threshold: 3,
                timeout: Duration::from_secs(30),
                half_open_max_calls: 2,
            },
        );

        manager
    }
}

/// Utility functions for error recovery
pub mod utils {
    use super::*;

    /// Create a retry operation closure
    pub fn retry_operation<F, T>(
        operation: F,
    ) -> impl Fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<T>> + Send>>
    where
        F: Fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<T>> + Send>>
            + Clone
            + Send
            + Sync
            + 'static,
        T: Send + 'static,
    {
        move || {
            let op = operation.clone();
            op()
        }
    }

    /// Determine if error should trigger circuit breaker
    pub fn should_trip_circuit_breaker(error: &VoirsError) -> bool {
        matches!(
            error,
            VoirsError::DeviceNotAvailable { .. }
                | VoirsError::UnsupportedDevice { .. }
                | VoirsError::OutOfMemory { .. }
                | VoirsError::GpuOutOfMemory { .. }
                | VoirsError::ModelNotFound { .. }
                | VoirsError::NetworkError { .. }
        )
    }

    /// Get recommended retry delay based on error type
    pub fn get_recommended_retry_delay(error: &VoirsError) -> Duration {
        match error {
            VoirsError::NetworkError { .. } => Duration::from_secs(1),
            VoirsError::TimeoutError { .. } => Duration::from_millis(500),
            VoirsError::DeviceError { .. } => Duration::from_millis(100),
            VoirsError::ModelError { .. } => Duration::from_secs(2),
            _ => Duration::from_millis(200),
        }
    }

    /// Extract recoverable error information
    pub fn extract_recovery_info(error: &VoirsError) -> (bool, Option<Vec<String>>) {
        let recoverable = error.is_recoverable();
        let suggestions = if recoverable {
            Some(error.recovery_suggestions())
        } else {
            None
        };
        (recoverable, suggestions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::sleep;

    #[tokio::test]
    async fn test_circuit_breaker() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            timeout: Duration::from_millis(100),
            half_open_max_calls: 1,
        };
        let circuit_breaker = CircuitBreaker::new(config);

        // Initial state should be closed
        assert_eq!(circuit_breaker.state(), CircuitState::Closed);

        // First failure
        let result = circuit_breaker
            .call(async {
                std::result::Result::<(), VoirsError>::Err(VoirsError::InternalError {
                    component: "test".to_string(),
                    message: "test error".to_string(),
                })
            })
            .await;
        assert!(result.is_err());
        assert_eq!(circuit_breaker.state(), CircuitState::Closed);

        // Second failure should open circuit
        let result = circuit_breaker
            .call(async {
                std::result::Result::<(), VoirsError>::Err(VoirsError::InternalError {
                    component: "test".to_string(),
                    message: "test error".to_string(),
                })
            })
            .await;
        assert!(result.is_err());
        assert_eq!(circuit_breaker.state(), CircuitState::Open);

        // Next call should be rejected immediately
        let result = circuit_breaker
            .call(async { Ok::<(), VoirsError>(()) })
            .await;
        match result {
            Err(CircuitBreakerError::CircuitOpen) => {}
            _ => panic!("Expected CircuitOpen error"),
        }

        // Wait for timeout
        sleep(Duration::from_millis(150)).await;

        // Should transition to half-open and allow one call
        let result = circuit_breaker
            .call(async { Ok::<(), VoirsError>(()) })
            .await;
        assert!(result.is_ok());
        assert_eq!(circuit_breaker.state(), CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_retry_linear() {
        let manager = ErrorRecoveryManager::new();
        let attempt_count = std::sync::Arc::new(std::sync::Mutex::new(0));

        let operation = {
            let attempt_count = attempt_count.clone();
            move || -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<i32>> + Send>> {
                let count = {
                    let mut guard = attempt_count.lock().unwrap();
                    *guard += 1;
                    *guard
                };
                Box::pin(async move {
                    if count < 3 {
                        Err(VoirsError::InternalError {
                            component: "test".to_string(),
                            message: "temporary error".to_string(),
                        })
                    } else {
                        Ok(42)
                    }
                })
            }
        };

        let mut context = RecoveryContext::default();
        let result = manager
            .retry_linear(operation, 5, Duration::from_millis(10), &mut context)
            .await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
        assert_eq!(context.attempt, 3);
        assert_eq!(*attempt_count.lock().unwrap(), 3);
    }

    #[tokio::test]
    async fn test_fallback_strategy() {
        let mut manager = ErrorRecoveryManager::new();
        manager.register_strategy(
            "test",
            RecoveryStrategy::Fallback {
                alternatives: vec!["alt1".to_string(), "alt2".to_string()],
            },
        );

        let attempt_count = std::sync::Arc::new(std::sync::Mutex::new(0));
        let operation = {
            let attempt_count = attempt_count.clone();
            move || -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<String>> + Send>> {
                let count = {
                    let mut guard = attempt_count.lock().unwrap();
                    *guard += 1;
                    *guard
                };
                Box::pin(async move {
                    if count <= 2 {
                        Err(VoirsError::InternalError {
                            component: "test".to_string(),
                            message: "error".to_string(),
                        })
                    } else {
                        Ok("success".to_string())
                    }
                })
            }
        };

        let result = manager.execute_with_recovery("test", operation).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "success");
    }
}

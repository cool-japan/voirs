use crate::error::VoirsError;
use crate::r#async::primitives::timeout;
use crate::types::VoirsResult;
use futures::future::{BoxFuture, Future};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

#[derive(Debug, Clone)]
pub enum ErrorSeverity {
    Critical,
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone)]
pub struct ErrorContext {
    pub operation: String,
    pub component: String,
    pub severity: ErrorSeverity,
    pub timestamp: Instant,
    pub retry_count: u32,
    pub metadata: HashMap<String, String>,
}

impl ErrorContext {
    pub fn new(operation: String, component: String, severity: ErrorSeverity) -> Self {
        Self {
            operation,
            component,
            severity,
            timestamp: Instant::now(),
            retry_count: 0,
            metadata: HashMap::new(),
        }
    }

    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    pub fn increment_retry(&mut self) {
        self.retry_count += 1;
    }
}

pub struct ErrorPropagator {
    context_stack: Arc<RwLock<Vec<ErrorContext>>>,
    error_handlers: Arc<RwLock<HashMap<String, Box<dyn ErrorHandler>>>>,
}

pub trait ErrorHandler: Send + Sync {
    fn can_handle(&self, error: &VoirsError, context: &ErrorContext) -> bool;
    fn handle(
        &self,
        error: VoirsError,
        context: ErrorContext,
    ) -> BoxFuture<'static, VoirsResult<()>>;
}

impl Default for ErrorPropagator {
    fn default() -> Self {
        Self::new()
    }
}

impl ErrorPropagator {
    pub fn new() -> Self {
        Self {
            context_stack: Arc::new(RwLock::new(Vec::new())),
            error_handlers: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn push_context(&self, context: ErrorContext) {
        let mut stack = self.context_stack.write().await;
        stack.push(context);
    }

    pub async fn pop_context(&self) -> Option<ErrorContext> {
        let mut stack = self.context_stack.write().await;
        stack.pop()
    }

    pub async fn propagate_error(&self, mut error: VoirsError) -> VoirsError {
        let context_stack = self.context_stack.read().await;

        if let Some(current_context) = context_stack.last() {
            error = error
                .with_context(
                    current_context.component.clone(),
                    format!(
                        "Operation: {}, Retry: {}",
                        current_context.operation, current_context.retry_count
                    ),
                )
                .into();
        }

        for context in context_stack.iter().rev() {
            let handlers = self.error_handlers.read().await;

            for handler in handlers.values() {
                if handler.can_handle(&error, context) {
                    let handler_result = handler.handle(error.clone(), context.clone()).await;

                    if let Err(handler_error) = handler_result {
                        error = handler_error;
                        break;
                    }
                }
            }
        }

        error
    }

    pub async fn register_handler(&self, name: String, handler: Box<dyn ErrorHandler>) {
        let mut handlers = self.error_handlers.write().await;
        handlers.insert(name, handler);
    }
}

pub struct PartialFailureCollector<T> {
    results: Vec<VoirsResult<T>>,
    failure_threshold: f64,
    success_threshold: f64,
}

impl<T> PartialFailureCollector<T> {
    pub fn new(failure_threshold: f64, success_threshold: f64) -> Self {
        Self {
            results: Vec::new(),
            failure_threshold,
            success_threshold,
        }
    }

    pub fn add_result(&mut self, result: VoirsResult<T>) {
        self.results.push(result);
    }

    pub fn should_fail(&self) -> bool {
        if self.results.is_empty() {
            return false;
        }

        let failure_count = self.results.iter().filter(|r| r.is_err()).count();
        let failure_rate = failure_count as f64 / self.results.len() as f64;

        failure_rate >= self.failure_threshold
    }

    pub fn should_succeed(&self) -> bool {
        if self.results.is_empty() {
            return false;
        }

        let success_count = self.results.iter().filter(|r| r.is_ok()).count();
        let success_rate = success_count as f64 / self.results.len() as f64;

        success_rate >= self.success_threshold
    }

    pub fn collect_successes(self) -> Vec<T> {
        self.results
            .into_iter()
            .filter_map(|result| result.ok())
            .collect()
    }

    pub fn collect_failures(self) -> Vec<VoirsError> {
        self.results
            .into_iter()
            .filter_map(|result| result.err())
            .collect()
    }

    pub fn failure_rate(&self) -> f64 {
        if self.results.is_empty() {
            return 0.0;
        }

        let failure_count = self.results.iter().filter(|r| r.is_err()).count();
        failure_count as f64 / self.results.len() as f64
    }

    pub fn success_rate(&self) -> f64 {
        if self.results.is_empty() {
            return 0.0;
        }

        let success_count = self.results.iter().filter(|r| r.is_ok()).count();
        success_count as f64 / self.results.len() as f64
    }
}

#[derive(Debug, Clone)]
pub struct RetryConfig {
    pub max_attempts: usize,
    pub initial_delay: Duration,
    pub max_delay: Duration,
    pub backoff_factor: f64,
    pub jitter: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(30),
            backoff_factor: 2.0,
            jitter: true,
        }
    }
}

pub struct RetryManager {
    config: RetryConfig,
    propagator: ErrorPropagator,
}

impl RetryManager {
    pub fn new(config: RetryConfig) -> Self {
        Self {
            config,
            propagator: ErrorPropagator::new(),
        }
    }

    pub async fn retry_with_context<F, Fut, T>(
        &self,
        context: ErrorContext,
        operation: F,
    ) -> VoirsResult<T>
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = VoirsResult<T>>,
    {
        self.propagator.push_context(context.clone()).await;

        let result = self.retry_operation(operation).await;

        self.propagator.pop_context().await;

        match result {
            Ok(value) => Ok(value),
            Err(error) => {
                let propagated_error = self.propagator.propagate_error(error).await;
                Err(propagated_error)
            }
        }
    }

    async fn retry_operation<F, Fut, T>(&self, mut operation: F) -> VoirsResult<T>
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = VoirsResult<T>>,
    {
        let mut attempts = 0;
        let mut delay = self.config.initial_delay;

        loop {
            attempts += 1;

            let result = operation().await;

            match result {
                Ok(value) => return Ok(value),
                Err(error) => {
                    if attempts >= self.config.max_attempts {
                        return Err(VoirsError::internal(
                            "retry_manager",
                            format!(
                                "Max retry attempts ({}) exceeded: {}",
                                self.config.max_attempts, error
                            ),
                        ));
                    }

                    if !self.should_retry(&error) {
                        return Err(error);
                    }

                    tokio::time::sleep(delay).await;

                    delay = Duration::from_millis(
                        (delay.as_millis() as f64 * self.config.backoff_factor) as u64,
                    )
                    .min(self.config.max_delay);

                    if self.config.jitter {
                        let jitter =
                            Duration::from_millis(fastrand::u64(0..=delay.as_millis() as u64 / 10));
                        delay += jitter;
                    }
                }
            }
        }
    }

    fn should_retry(&self, error: &VoirsError) -> bool {
        matches!(
            error,
            VoirsError::InternalError { .. }
                | VoirsError::InvalidConfiguration { .. }
                | VoirsError::SynthesisFailed { .. }
        )
    }
}

pub struct TimeoutManager {
    default_timeout: Duration,
    operation_timeouts: HashMap<String, Duration>,
}

impl TimeoutManager {
    pub fn new(default_timeout: Duration) -> Self {
        Self {
            default_timeout,
            operation_timeouts: HashMap::new(),
        }
    }

    pub fn set_operation_timeout(&mut self, operation: String, timeout: Duration) {
        self.operation_timeouts.insert(operation, timeout);
    }

    pub fn get_timeout(&self, operation: &str) -> Duration {
        self.operation_timeouts
            .get(operation)
            .copied()
            .unwrap_or(self.default_timeout)
    }

    pub async fn execute_with_timeout<F, T>(&self, operation: &str, future: F) -> VoirsResult<T>
    where
        F: Future<Output = VoirsResult<T>> + Unpin,
    {
        let timeout_duration = self.get_timeout(operation);

        match timeout(future, timeout_duration).await {
            Ok(result) => result,
            Err(timeout_error) => Err(timeout_error),
        }
    }
}

pub struct CircuitBreaker {
    name: String,
    failure_threshold: usize,
    success_threshold: usize,
    timeout: Duration,
    state: Arc<RwLock<CircuitBreakerState>>,
}

#[derive(Debug, Clone)]
enum CircuitBreakerState {
    Closed { failure_count: usize },
    Open { last_failure_time: Instant },
    HalfOpen { success_count: usize },
}

impl CircuitBreaker {
    pub fn new(
        name: String,
        failure_threshold: usize,
        success_threshold: usize,
        timeout: Duration,
    ) -> Self {
        Self {
            name,
            failure_threshold,
            success_threshold,
            timeout,
            state: Arc::new(RwLock::new(CircuitBreakerState::Closed {
                failure_count: 0,
            })),
        }
    }

    pub async fn execute<F, T>(&self, operation: F) -> VoirsResult<T>
    where
        F: Future<Output = VoirsResult<T>>,
    {
        if !self.can_execute().await {
            return Err(VoirsError::internal(
                "CircuitBreaker",
                format!("Circuit breaker '{}' is open", self.name),
            ));
        }

        match operation.await {
            Ok(result) => {
                self.on_success().await;
                Ok(result)
            }
            Err(error) => {
                self.on_failure().await;
                Err(error)
            }
        }
    }

    async fn can_execute(&self) -> bool {
        let state = self.state.read().await;

        match *state {
            CircuitBreakerState::Closed { .. } => true,
            CircuitBreakerState::HalfOpen { .. } => true,
            CircuitBreakerState::Open { last_failure_time } => {
                last_failure_time.elapsed() >= self.timeout
            }
        }
    }

    async fn on_success(&self) {
        let mut state = self.state.write().await;

        match *state {
            CircuitBreakerState::Closed { .. } => {
                *state = CircuitBreakerState::Closed { failure_count: 0 };
            }
            CircuitBreakerState::HalfOpen { success_count } => {
                let new_success_count = success_count + 1;
                if new_success_count >= self.success_threshold {
                    *state = CircuitBreakerState::Closed { failure_count: 0 };
                } else {
                    *state = CircuitBreakerState::HalfOpen {
                        success_count: new_success_count,
                    };
                }
            }
            CircuitBreakerState::Open { .. } => {
                *state = CircuitBreakerState::HalfOpen { success_count: 1 };
            }
        }
    }

    async fn on_failure(&self) {
        let mut state = self.state.write().await;

        match *state {
            CircuitBreakerState::Closed { failure_count } => {
                let new_failure_count = failure_count + 1;
                if new_failure_count >= self.failure_threshold {
                    *state = CircuitBreakerState::Open {
                        last_failure_time: Instant::now(),
                    };
                } else {
                    *state = CircuitBreakerState::Closed {
                        failure_count: new_failure_count,
                    };
                }
            }
            CircuitBreakerState::HalfOpen { .. } => {
                *state = CircuitBreakerState::Open {
                    last_failure_time: Instant::now(),
                };
            }
            CircuitBreakerState::Open { .. } => {
                *state = CircuitBreakerState::Open {
                    last_failure_time: Instant::now(),
                };
            }
        }
    }
}

pub struct BulkheadIsolator {
    name: String,
    semaphore: Arc<tokio::sync::Semaphore>,
    timeout: Duration,
}

impl BulkheadIsolator {
    pub fn new(name: String, capacity: usize, timeout: Duration) -> Self {
        Self {
            name,
            semaphore: Arc::new(tokio::sync::Semaphore::new(capacity)),
            timeout,
        }
    }

    pub async fn execute<F, T>(&self, operation: F) -> VoirsResult<T>
    where
        F: Future<Output = VoirsResult<T>>,
    {
        let permit = tokio::time::timeout(self.timeout, self.semaphore.acquire())
            .await
            .map_err(|_| {
                VoirsError::timeout(format!("Bulkhead '{}' acquisition timeout", self.name))
            })?
            .map_err(|e| {
                VoirsError::internal(
                    "Bulkhead",
                    format!("Bulkhead '{}' permit error: {}", self.name, e),
                )
            })?;

        let result = operation.await;

        drop(permit);

        result
    }
}

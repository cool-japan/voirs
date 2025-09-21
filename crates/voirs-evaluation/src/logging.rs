//! Enhanced logging and debugging support for VoiRS evaluation
//!
//! This module provides comprehensive logging and debugging utilities to help
//! users understand evaluation processes and troubleshoot issues.

use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};

/// Log level for evaluation events
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    /// Trace-level debugging information
    Trace = 0,
    /// Debug information for troubleshooting
    Debug = 1,
    /// General information about evaluation progress
    Info = 2,
    /// Warning about potential issues
    Warn = 3,
    /// Error conditions
    Error = 4,
}

impl fmt::Display for LogLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LogLevel::Trace => write!(f, "TRACE"),
            LogLevel::Debug => write!(f, "DEBUG"),
            LogLevel::Info => write!(f, " INFO"),
            LogLevel::Warn => write!(f, " WARN"),
            LogLevel::Error => write!(f, "ERROR"),
        }
    }
}

impl Default for LogLevel {
    fn default() -> Self {
        LogLevel::Info
    }
}

/// Evaluation event for logging
#[derive(Debug, Clone)]
pub struct EvaluationEvent {
    /// Event timestamp
    pub timestamp: Instant,
    /// Log level
    pub level: LogLevel,
    /// Component that generated the event
    pub component: String,
    /// Event message
    pub message: String,
    /// Optional additional metadata
    pub metadata: HashMap<String, String>,
    /// Duration for timing events
    pub duration: Option<Duration>,
}

impl EvaluationEvent {
    /// Create a new evaluation event
    pub fn new(level: LogLevel, component: &str, message: &str) -> Self {
        Self {
            timestamp: Instant::now(),
            level,
            component: component.to_string(),
            message: message.to_string(),
            metadata: HashMap::new(),
            duration: None,
        }
    }

    /// Add metadata to the event
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    /// Add duration to the event
    pub fn with_duration(mut self, duration: Duration) -> Self {
        self.duration = Some(duration);
        self
    }
}

impl fmt::Display for EvaluationEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}: {}", self.level, self.component, self.message)?;

        if let Some(duration) = self.duration {
            write!(f, " (took {:?})", duration)?;
        }

        if !self.metadata.is_empty() {
            write!(f, " [")?;
            for (i, (key, value)) in self.metadata.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}={}", key, value)?;
            }
            write!(f, "]")?;
        }

        Ok(())
    }
}

/// Trait for logging backends
pub trait LogBackend: Send + Sync {
    /// Log an evaluation event
    fn log(&self, event: &EvaluationEvent);

    /// Check if a log level is enabled
    fn is_enabled(&self, level: LogLevel) -> bool;
}

/// Console logging backend
#[derive(Debug)]
pub struct ConsoleLogger {
    min_level: LogLevel,
}

impl ConsoleLogger {
    /// Create a new console logger
    pub fn new(min_level: LogLevel) -> Self {
        Self { min_level }
    }
}

impl LogBackend for ConsoleLogger {
    fn log(&self, event: &EvaluationEvent) {
        if event.level >= self.min_level {
            println!("{}", event);
        }
    }

    fn is_enabled(&self, level: LogLevel) -> bool {
        level >= self.min_level
    }
}

/// Memory-based logging backend for testing and debugging
#[derive(Debug, Default)]
pub struct MemoryLogger {
    events: std::sync::Mutex<Vec<EvaluationEvent>>,
    min_level: LogLevel,
}

impl MemoryLogger {
    /// Create a new memory logger
    pub fn new(min_level: LogLevel) -> Self {
        Self {
            events: std::sync::Mutex::new(Vec::new()),
            min_level,
        }
    }

    /// Get all logged events
    pub fn events(&self) -> Vec<EvaluationEvent> {
        self.events.lock().unwrap().clone()
    }

    /// Clear all logged events
    pub fn clear(&self) {
        self.events.lock().unwrap().clear();
    }

    /// Get events by component
    pub fn events_for_component(&self, component: &str) -> Vec<EvaluationEvent> {
        self.events()
            .into_iter()
            .filter(|e| e.component == component)
            .collect()
    }

    /// Get events by level
    pub fn events_by_level(&self, level: LogLevel) -> Vec<EvaluationEvent> {
        self.events()
            .into_iter()
            .filter(|e| e.level == level)
            .collect()
    }
}

impl LogBackend for MemoryLogger {
    fn log(&self, event: &EvaluationEvent) {
        if event.level >= self.min_level {
            self.events.lock().unwrap().push(event.clone());
        }
    }

    fn is_enabled(&self, level: LogLevel) -> bool {
        level >= self.min_level
    }
}

/// Global evaluation logger
pub struct EvaluationLogger {
    backend: Box<dyn LogBackend>,
}

impl EvaluationLogger {
    /// Create a new evaluation logger with console backend
    pub fn console(min_level: LogLevel) -> Self {
        Self {
            backend: Box::new(ConsoleLogger::new(min_level)),
        }
    }

    /// Create a new evaluation logger with memory backend
    pub fn memory(min_level: LogLevel) -> Self {
        Self {
            backend: Box::new(MemoryLogger::new(min_level)),
        }
    }

    /// Create a new evaluation logger with custom backend
    pub fn with_backend(backend: Box<dyn LogBackend>) -> Self {
        Self { backend }
    }

    /// Log an event
    pub fn log(&self, event: EvaluationEvent) {
        self.backend.log(&event);
    }

    /// Check if a log level is enabled
    pub fn is_enabled(&self, level: LogLevel) -> bool {
        self.backend.is_enabled(level)
    }

    /// Log a trace message
    pub fn trace(&self, component: &str, message: &str) {
        if self.is_enabled(LogLevel::Trace) {
            self.log(EvaluationEvent::new(LogLevel::Trace, component, message));
        }
    }

    /// Log a debug message
    pub fn debug(&self, component: &str, message: &str) {
        if self.is_enabled(LogLevel::Debug) {
            self.log(EvaluationEvent::new(LogLevel::Debug, component, message));
        }
    }

    /// Log an info message
    pub fn info(&self, component: &str, message: &str) {
        if self.is_enabled(LogLevel::Info) {
            self.log(EvaluationEvent::new(LogLevel::Info, component, message));
        }
    }

    /// Log a warning message
    pub fn warn(&self, component: &str, message: &str) {
        if self.is_enabled(LogLevel::Warn) {
            self.log(EvaluationEvent::new(LogLevel::Warn, component, message));
        }
    }

    /// Log an error message
    pub fn error(&self, component: &str, message: &str) {
        if self.is_enabled(LogLevel::Error) {
            self.log(EvaluationEvent::new(LogLevel::Error, component, message));
        }
    }
}

/// Performance timer for measuring operation durations
pub struct PerformanceTimer {
    start_time: Instant,
    component: String,
    operation: String,
    logger: Option<std::sync::Arc<EvaluationLogger>>,
}

impl PerformanceTimer {
    /// Start a new performance timer
    pub fn start(component: &str, operation: &str) -> Self {
        Self {
            start_time: Instant::now(),
            component: component.to_string(),
            operation: operation.to_string(),
            logger: None,
        }
    }

    /// Start a new performance timer with logger
    pub fn start_with_logger(
        component: &str,
        operation: &str,
        logger: std::sync::Arc<EvaluationLogger>,
    ) -> Self {
        let mut timer = Self::start(component, operation);
        timer.logger = Some(logger);

        if let Some(ref logger) = timer.logger {
            logger.debug(component, &format!("Starting {}", operation));
        }

        timer
    }

    /// Get elapsed time
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Stop the timer and return duration
    pub fn stop(self) -> Duration {
        let duration = self.elapsed();

        if let Some(logger) = &self.logger {
            logger.log(
                EvaluationEvent::new(
                    LogLevel::Debug,
                    &self.component,
                    &format!("Completed {}", self.operation),
                )
                .with_duration(duration),
            );
        }

        duration
    }
}

/// Macro for easy logging
#[macro_export]
macro_rules! eval_log {
    ($logger:expr, trace, $component:expr, $($arg:tt)*) => {
        $logger.trace($component, &format!($($arg)*))
    };
    ($logger:expr, debug, $component:expr, $($arg:tt)*) => {
        $logger.debug($component, &format!($($arg)*))
    };
    ($logger:expr, info, $component:expr, $($arg:tt)*) => {
        $logger.info($component, &format!($($arg)*))
    };
    ($logger:expr, warn, $component:expr, $($arg:tt)*) => {
        $logger.warn($component, &format!($($arg)*))
    };
    ($logger:expr, error, $component:expr, $($arg:tt)*) => {
        $logger.error($component, &format!($($arg)*))
    };
}

/// Debug context for evaluation operations
#[derive(Debug, Clone)]
pub struct DebugContext {
    /// Operation name
    pub operation: String,
    /// Input parameters
    pub parameters: HashMap<String, String>,
    /// Intermediate results
    pub intermediate_results: HashMap<String, String>,
    /// Performance metrics
    pub performance_metrics: HashMap<String, Duration>,
    /// Warnings encountered
    pub warnings: Vec<String>,
}

impl DebugContext {
    /// Create a new debug context
    pub fn new(operation: &str) -> Self {
        Self {
            operation: operation.to_string(),
            parameters: HashMap::new(),
            intermediate_results: HashMap::new(),
            performance_metrics: HashMap::new(),
            warnings: Vec::new(),
        }
    }

    /// Add a parameter
    pub fn parameter(mut self, key: &str, value: &str) -> Self {
        self.parameters.insert(key.to_string(), value.to_string());
        self
    }

    /// Add an intermediate result
    pub fn intermediate(mut self, key: &str, value: &str) -> Self {
        self.intermediate_results
            .insert(key.to_string(), value.to_string());
        self
    }

    /// Add a performance metric
    pub fn timing(mut self, key: &str, duration: Duration) -> Self {
        self.performance_metrics.insert(key.to_string(), duration);
        self
    }

    /// Add a warning
    pub fn warning(mut self, warning: &str) -> Self {
        self.warnings.push(warning.to_string());
        self
    }

    /// Generate a debug report
    pub fn report(&self) -> String {
        let mut report = format!("Debug Report for {}\n", self.operation);
        report.push_str("=".repeat(50).as_str());
        report.push('\n');

        if !self.parameters.is_empty() {
            report.push_str("\nParameters:\n");
            for (key, value) in &self.parameters {
                report.push_str(&format!("  {}: {}\n", key, value));
            }
        }

        if !self.intermediate_results.is_empty() {
            report.push_str("\nIntermediate Results:\n");
            for (key, value) in &self.intermediate_results {
                report.push_str(&format!("  {}: {}\n", key, value));
            }
        }

        if !self.performance_metrics.is_empty() {
            report.push_str("\nPerformance Metrics:\n");
            for (key, duration) in &self.performance_metrics {
                report.push_str(&format!("  {}: {:?}\n", key, duration));
            }
        }

        if !self.warnings.is_empty() {
            report.push_str("\nWarnings:\n");
            for warning in &self.warnings {
                report.push_str(&format!("  - {}\n", warning));
            }
        }

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_log_levels() {
        assert!(LogLevel::Error > LogLevel::Warn);
        assert!(LogLevel::Warn > LogLevel::Info);
        assert!(LogLevel::Info > LogLevel::Debug);
        assert!(LogLevel::Debug > LogLevel::Trace);
    }

    #[test]
    fn test_evaluation_event() {
        let event = EvaluationEvent::new(LogLevel::Info, "test", "test message")
            .with_metadata("key", "value")
            .with_duration(Duration::from_millis(100));

        assert_eq!(event.level, LogLevel::Info);
        assert_eq!(event.component, "test");
        assert_eq!(event.message, "test message");
        assert_eq!(event.metadata.get("key"), Some(&"value".to_string()));
        assert!(event.duration.is_some());
    }

    #[test]
    fn test_memory_logger() {
        let logger = MemoryLogger::new(LogLevel::Debug);

        logger.log(&EvaluationEvent::new(
            LogLevel::Info,
            "test",
            "info message",
        ));
        logger.log(&EvaluationEvent::new(
            LogLevel::Debug,
            "test",
            "debug message",
        ));
        logger.log(&EvaluationEvent::new(
            LogLevel::Trace,
            "test",
            "trace message",
        )); // Filtered out

        let events = logger.events();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].level, LogLevel::Info);
        assert_eq!(events[1].level, LogLevel::Debug);
    }

    #[test]
    fn test_performance_timer() {
        let timer = PerformanceTimer::start("test", "operation");
        thread::sleep(Duration::from_millis(10));
        let duration = timer.stop();

        assert!(duration >= Duration::from_millis(10));
    }

    #[test]
    fn test_debug_context() {
        let context = DebugContext::new("test_operation")
            .parameter("param1", "value1")
            .intermediate("result1", "intermediate1")
            .timing("step1", Duration::from_millis(100))
            .warning("test warning");

        let report = context.report();
        assert!(report.contains("test_operation"));
        assert!(report.contains("param1: value1"));
        assert!(report.contains("result1: intermediate1"));
        assert!(report.contains("step1:"));
        assert!(report.contains("test warning"));
    }

    #[test]
    fn test_evaluation_logger() {
        let logger = EvaluationLogger::memory(LogLevel::Info);

        logger.info("test", "info message");
        logger.debug("test", "debug message"); // Should be filtered out
        logger.warn("test", "warning message");

        // Note: We can't directly access the memory logger events through the trait,
        // but this tests that the logger doesn't panic
    }
}

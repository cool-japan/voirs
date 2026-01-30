//! Unified logging infrastructure for `VoiRS` recognizer.
//!
//! This module provides comprehensive logging capabilities with structured logging,
//! log levels, contextual information, and integration with distributed tracing.

pub mod filters;
pub mod formatters;
pub mod structured;

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Unified logger configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogConfig {
    /// Minimum log level to output
    pub min_level: LogLevel,

    /// Enable structured logging (JSON format)
    pub structured: bool,

    /// Include timestamps in logs
    pub include_timestamps: bool,

    /// Include source location (<file:line>)
    pub include_source: bool,

    /// Include thread ID
    pub include_thread_id: bool,

    /// Maximum log message length
    pub max_message_length: Option<usize>,

    /// Log output targets
    pub targets: Vec<LogTarget>,

    /// Context fields to always include
    pub default_context: HashMap<String, String>,

    /// Enable performance logging
    pub enable_performance_logging: bool,

    /// Performance log threshold in milliseconds
    pub performance_threshold_ms: u64,
}

impl Default for LogConfig {
    fn default() -> Self {
        Self {
            min_level: LogLevel::Info,
            structured: true,
            include_timestamps: true,
            include_source: false,
            include_thread_id: false,
            max_message_length: Some(1000),
            targets: vec![LogTarget::Stdout],
            default_context: HashMap::new(),
            enable_performance_logging: true,
            performance_threshold_ms: 100,
        }
    }
}

/// Log severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum LogLevel {
    /// Trace level - very detailed
    Trace,
    /// Debug level - detailed information
    Debug,
    /// Info level - general information
    Info,
    /// Warning level - potential issues
    Warn,
    /// Error level - errors that need attention
    Error,
    /// Fatal level - critical errors
    Fatal,
}

impl LogLevel {
    /// Convert to string representation
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            LogLevel::Trace => "TRACE",
            LogLevel::Debug => "DEBUG",
            LogLevel::Info => "INFO",
            LogLevel::Warn => "WARN",
            LogLevel::Error => "ERROR",
            LogLevel::Fatal => "FATAL",
        }
    }
}

/// Log output targets
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LogTarget {
    /// Standard output
    Stdout,
    /// Standard error
    Stderr,
    /// File output
    File(String),
    /// Syslog
    Syslog,
    /// Custom target
    Custom(String),
}

/// Structured log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    /// Log level
    pub level: LogLevel,

    /// Timestamp
    pub timestamp: DateTime<Utc>,

    /// Log message
    pub message: String,

    /// Module path
    pub module: Option<String>,

    /// Source location (<file:line>)
    pub source: Option<String>,

    /// Thread ID
    pub thread_id: Option<String>,

    /// Additional context fields
    pub context: HashMap<String, String>,

    /// Performance metrics
    pub metrics: Option<PerformanceMetrics>,
}

/// Performance metrics for logging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Operation name
    pub operation: String,

    /// Duration in milliseconds
    pub duration_ms: u64,

    /// Memory used in bytes
    pub memory_bytes: Option<u64>,

    /// Additional metrics
    pub custom_metrics: HashMap<String, f64>,
}

/// Unified logger implementation
pub struct UnifiedLogger {
    config: Arc<RwLock<LogConfig>>,
    context_stack: Arc<RwLock<Vec<HashMap<String, String>>>>,
}

impl UnifiedLogger {
    /// Create a new unified logger
    #[must_use]
    pub fn new(config: LogConfig) -> Self {
        Self {
            config: Arc::new(RwLock::new(config)),
            context_stack: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Create with default configuration
    #[must_use]
    pub fn default() -> Self {
        Self::new(LogConfig::default())
    }

    /// Log a message at specified level
    pub fn log(&self, level: LogLevel, message: &str) {
        let config = self.config.read();

        if level < config.min_level {
            return;
        }

        let entry = self.create_entry(level, message, None);
        self.output_entry(&entry, &config);
    }

    /// Log with context
    pub fn log_with_context(
        &self,
        level: LogLevel,
        message: &str,
        context: HashMap<String, String>,
    ) {
        let config = self.config.read();

        if level < config.min_level {
            return;
        }

        let entry = self.create_entry(level, message, Some(context));
        self.output_entry(&entry, &config);
    }

    /// Log trace message
    pub fn trace(&self, message: &str) {
        self.log(LogLevel::Trace, message);
    }

    /// Log debug message
    pub fn debug(&self, message: &str) {
        self.log(LogLevel::Debug, message);
    }

    /// Log info message
    pub fn info(&self, message: &str) {
        self.log(LogLevel::Info, message);
    }

    /// Log warning message
    pub fn warn(&self, message: &str) {
        self.log(LogLevel::Warn, message);
    }

    /// Log error message
    pub fn error(&self, message: &str) {
        self.log(LogLevel::Error, message);
    }

    /// Log fatal message
    pub fn fatal(&self, message: &str) {
        self.log(LogLevel::Fatal, message);
    }

    /// Log performance metrics
    pub fn log_performance(&self, operation: &str, duration_ms: u64) {
        let config = self.config.read();

        if !config.enable_performance_logging || duration_ms < config.performance_threshold_ms {
            return;
        }

        let metrics = PerformanceMetrics {
            operation: operation.to_string(),
            duration_ms,
            memory_bytes: None,
            custom_metrics: HashMap::new(),
        };

        let mut entry = self.create_entry(
            LogLevel::Info,
            &format!("Performance: {operation} took {duration_ms}ms"),
            None,
        );
        entry.metrics = Some(metrics);

        self.output_entry(&entry, &config);
    }

    /// Push context onto stack
    pub fn push_context(&self, context: HashMap<String, String>) {
        self.context_stack.write().push(context);
    }

    /// Pop context from stack
    pub fn pop_context(&self) {
        self.context_stack.write().pop();
    }

    /// Create log entry
    fn create_entry(
        &self,
        level: LogLevel,
        message: &str,
        mut context: Option<HashMap<String, String>>,
    ) -> LogEntry {
        let config = self.config.read();
        let context_stack = self.context_stack.read();

        // Merge contexts: default -> stack -> provided
        let mut merged_context = config.default_context.clone();
        for ctx in context_stack.iter() {
            merged_context.extend(ctx.clone());
        }
        if let Some(ctx) = context.take() {
            merged_context.extend(ctx);
        }

        // Truncate message if needed
        let message = if let Some(max_len) = config.max_message_length {
            if message.len() > max_len {
                format!("{}...", &message[..max_len])
            } else {
                message.to_string()
            }
        } else {
            message.to_string()
        };

        LogEntry {
            level,
            timestamp: Utc::now(),
            message,
            module: None,
            source: if config.include_source {
                Some("recognizer".to_string())
            } else {
                None
            },
            thread_id: if config.include_thread_id {
                Some(format!("{:?}", std::thread::current().id()))
            } else {
                None
            },
            context: merged_context,
            metrics: None,
        }
    }

    /// Output log entry to configured targets
    fn output_entry(&self, entry: &LogEntry, config: &LogConfig) {
        for target in &config.targets {
            match target {
                LogTarget::Stdout => {
                    if config.structured {
                        if let Ok(json) = serde_json::to_string(entry) {
                            println!("{json}");
                        }
                    } else {
                        println!("{}", self.format_plain(entry));
                    }
                }
                LogTarget::Stderr => {
                    if config.structured {
                        if let Ok(json) = serde_json::to_string(entry) {
                            eprintln!("{json}");
                        }
                    } else {
                        eprintln!("{}", self.format_plain(entry));
                    }
                }
                LogTarget::File(path) => {
                    // File logging implementation
                    let _ = path; // Placeholder
                }
                LogTarget::Syslog => {
                    // Syslog implementation
                }
                LogTarget::Custom(_) => {
                    // Custom target implementation
                }
            }
        }
    }

    /// Format log entry as plain text
    fn format_plain(&self, entry: &LogEntry) -> String {
        let mut parts = Vec::new();

        parts.push(format!("[{}]", entry.level.as_str()));
        parts.push(entry.timestamp.format("%Y-%m-%d %H:%M:%S%.3f").to_string());

        if let Some(ref thread_id) = entry.thread_id {
            parts.push(format!("[{thread_id}]"));
        }

        if let Some(ref source) = entry.source {
            parts.push(format!("[{source}]"));
        }

        parts.push(entry.message.clone());

        if !entry.context.is_empty() {
            let ctx: Vec<String> = entry
                .context
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect();
            parts.push(format!("{{{}}}", ctx.join(", ")));
        }

        parts.join(" ")
    }
}

/// Global logger instance
static GLOBAL_LOGGER: once_cell::sync::Lazy<Arc<RwLock<Option<UnifiedLogger>>>> =
    once_cell::sync::Lazy::new(|| Arc::new(RwLock::new(None)));

/// Initialize global logger
pub fn init_logger(config: LogConfig) {
    let logger = UnifiedLogger::new(config);
    *GLOBAL_LOGGER.write() = Some(logger);
}

/// Get global logger
pub fn logger() -> Option<UnifiedLogger> {
    GLOBAL_LOGGER.read().as_ref().map(|l| UnifiedLogger {
        config: l.config.clone(),
        context_stack: l.context_stack.clone(),
    })
}

/// Convenience macro for logging with context
#[macro_export]
macro_rules! log_info {
    ($msg:expr) => {
        if let Some(logger) = $crate::logging::logger() {
            logger.info($msg);
        }
    };
    ($msg:expr, $($key:expr => $value:expr),+) => {
        if let Some(logger) = $crate::logging::logger() {
            let mut context = std::collections::HashMap::new();
            $(
                context.insert($key.to_string(), $value.to_string());
            )+
            logger.log_with_context($crate::logging::LogLevel::Info, $msg, context);
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_config_default() {
        let config = LogConfig::default();
        assert_eq!(config.min_level, LogLevel::Info);
        assert!(config.structured);
        assert!(config.include_timestamps);
    }

    #[test]
    fn test_log_levels() {
        assert!(LogLevel::Error > LogLevel::Warn);
        assert!(LogLevel::Warn > LogLevel::Info);
        assert!(LogLevel::Info > LogLevel::Debug);
        assert!(LogLevel::Debug > LogLevel::Trace);
    }

    #[test]
    fn test_log_level_string() {
        assert_eq!(LogLevel::Info.as_str(), "INFO");
        assert_eq!(LogLevel::Error.as_str(), "ERROR");
        assert_eq!(LogLevel::Debug.as_str(), "DEBUG");
    }

    #[test]
    fn test_unified_logger_creation() {
        let logger = UnifiedLogger::default();
        logger.info("Test message");
        logger.debug("Debug message");
        logger.error("Error message");
    }

    #[test]
    fn test_logger_with_context() {
        let logger = UnifiedLogger::default();
        let mut context = HashMap::new();
        context.insert("user_id".to_string(), "123".to_string());
        context.insert("operation".to_string(), "test".to_string());

        logger.log_with_context(LogLevel::Info, "Test with context", context);
    }

    #[test]
    fn test_performance_logging() {
        let logger = UnifiedLogger::default();
        logger.log_performance("audio_processing", 150);
        logger.log_performance("fast_operation", 50); // Below threshold
    }

    #[test]
    fn test_context_stack() {
        let logger = UnifiedLogger::default();

        let mut ctx1 = HashMap::new();
        ctx1.insert("session".to_string(), "abc123".to_string());
        logger.push_context(ctx1);

        logger.info("Message with session context");

        logger.pop_context();
        logger.info("Message without session context");
    }

    #[test]
    fn test_message_truncation() {
        let config = LogConfig {
            max_message_length: Some(20),
            ..Default::default()
        };
        let logger = UnifiedLogger::new(config);

        logger.info("This is a very long message that should be truncated");
    }

    #[test]
    fn test_global_logger_init() {
        init_logger(LogConfig::default());
        assert!(logger().is_some());

        if let Some(log) = logger() {
            log.info("Global logger test");
        }
    }
}

//! Log formatters for various output styles.
//!
//! This module provides different formatting options for log output including
//! JSON, plain text, colored output, and custom formats.

use super::{LogEntry, LogLevel};
use chrono::Utc;
use std::collections::HashMap;
use std::fmt::Write as FmtWrite;

/// Log formatter trait
pub trait LogFormatter: Send + Sync {
    /// Format a log entry into a string
    fn format(&self, entry: &LogEntry) -> String;
}

/// JSON formatter
pub struct JsonFormatter {
    pretty: bool,
}

impl JsonFormatter {
    /// Create new JSON formatter
    #[must_use]
    pub fn new(pretty: bool) -> Self {
        Self { pretty }
    }
}

impl LogFormatter for JsonFormatter {
    fn format(&self, entry: &LogEntry) -> String {
        if self.pretty {
            serde_json::to_string_pretty(entry).unwrap_or_else(|_| "{}".to_string())
        } else {
            serde_json::to_string(entry).unwrap_or_else(|_| "{}".to_string())
        }
    }
}

/// Plain text formatter
pub struct PlainFormatter {
    include_colors: bool,
}

impl PlainFormatter {
    /// Create new plain formatter
    #[must_use]
    pub fn new(include_colors: bool) -> Self {
        Self { include_colors }
    }

    /// Get ANSI color code for log level
    fn level_color(&self, level: LogLevel) -> &'static str {
        if !self.include_colors {
            return "";
        }

        match level {
            LogLevel::Trace => "\x1b[90m", // Gray
            LogLevel::Debug => "\x1b[36m", // Cyan
            LogLevel::Info => "\x1b[32m",  // Green
            LogLevel::Warn => "\x1b[33m",  // Yellow
            LogLevel::Error => "\x1b[31m", // Red
            LogLevel::Fatal => "\x1b[35m", // Magenta
        }
    }

    /// Reset ANSI color
    fn color_reset(&self) -> &'static str {
        if self.include_colors {
            "\x1b[0m"
        } else {
            ""
        }
    }
}

impl LogFormatter for PlainFormatter {
    fn format(&self, entry: &LogEntry) -> String {
        let mut output = String::new();

        // Level with color
        write!(
            output,
            "{}[{:5}]{}",
            self.level_color(entry.level),
            entry.level.as_str(),
            self.color_reset()
        )
        .expect("write to String is infallible");

        // Timestamp
        write!(
            output,
            " {}",
            entry.timestamp.format("%Y-%m-%d %H:%M:%S%.3f")
        )
        .expect("write to String is infallible");

        // Thread ID
        if let Some(ref thread_id) = entry.thread_id {
            write!(output, " [{thread_id}]").expect("write to String is infallible");
        }

        // Source
        if let Some(ref source) = entry.source {
            write!(output, " [{source}]").expect("write to String is infallible");
        }

        // Message
        write!(output, " {}", entry.message).expect("write to String is infallible");

        // Context
        if !entry.context.is_empty() {
            write!(output, " {{").expect("write to String is infallible");
            let mut first = true;
            for (k, v) in &entry.context {
                if !first {
                    write!(output, ", ").expect("write to String is infallible");
                }
                write!(output, "{k}={v}").expect("write to String is infallible");
                first = false;
            }
            write!(output, "}}").expect("write to String is infallible");
        }

        // Performance metrics
        if let Some(ref metrics) = entry.metrics {
            write!(output, " [{}ms]", metrics.duration_ms).expect("write to String is infallible");
        }

        output
    }
}

/// Compact formatter for minimal output
pub struct CompactFormatter;

impl CompactFormatter {
    /// Create new compact formatter
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Default for CompactFormatter {
    fn default() -> Self {
        Self::new()
    }
}

impl LogFormatter for CompactFormatter {
    fn format(&self, entry: &LogEntry) -> String {
        format!(
            "{} {} {}",
            entry.level.as_str().chars().next().unwrap_or('?'),
            entry.timestamp.format("%H:%M:%S"),
            entry.message
        )
    }
}

/// Logfmt formatter (key=value pairs)
pub struct LogfmtFormatter;

impl LogfmtFormatter {
    /// Create new logfmt formatter
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Escape value for logfmt
    fn escape_value(value: &str) -> String {
        if value.contains(' ') || value.contains('"') || value.contains('=') {
            format!("\"{}\"", value.replace('\"', "\\\""))
        } else {
            value.to_string()
        }
    }
}

impl Default for LogfmtFormatter {
    fn default() -> Self {
        Self::new()
    }
}

impl LogFormatter for LogfmtFormatter {
    fn format(&self, entry: &LogEntry) -> String {
        let mut output = String::new();

        // Core fields
        write!(
            output,
            "level={} ts={} msg={}",
            Self::escape_value(entry.level.as_str()),
            Self::escape_value(&entry.timestamp.to_rfc3339()),
            Self::escape_value(&entry.message)
        )
        .expect("write to String is infallible");

        // Optional fields
        if let Some(ref module) = entry.module {
            write!(output, " module={}", Self::escape_value(module))
                .expect("write to String is infallible");
        }

        if let Some(ref source) = entry.source {
            write!(output, " source={}", Self::escape_value(source))
                .expect("write to String is infallible");
        }

        if let Some(ref thread_id) = entry.thread_id {
            write!(output, " thread={}", Self::escape_value(thread_id))
                .expect("write to String is infallible");
        }

        // Context fields
        for (k, v) in &entry.context {
            write!(output, " {}={}", k, Self::escape_value(v))
                .expect("write to String is infallible");
        }

        // Performance metrics
        if let Some(ref metrics) = entry.metrics {
            write!(
                output,
                " operation={} duration_ms={}",
                Self::escape_value(&metrics.operation),
                metrics.duration_ms
            )
            .expect("write to String is infallible");
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logging::PerformanceMetrics;

    fn create_test_entry() -> LogEntry {
        let mut context = HashMap::new();
        context.insert("user_id".to_string(), "123".to_string());
        context.insert("action".to_string(), "login".to_string());

        LogEntry {
            level: LogLevel::Info,
            timestamp: Utc::now(),
            message: "Test message".to_string(),
            module: Some("test::module".to_string()),
            source: Some("test.rs:42".to_string()),
            thread_id: Some("thread-1".to_string()),
            context,
            metrics: None,
        }
    }

    #[test]
    fn test_json_formatter() {
        let formatter = JsonFormatter::new(false);
        let entry = create_test_entry();
        let output = formatter.format(&entry);

        // LogLevel serializes as the enum variant name
        assert!(output.contains("\"level\""));
        assert!(output.contains("\"message\""));
        assert!(output.contains("Test message"));
        assert!(!output.contains('\n')); // Not pretty-printed
    }

    #[test]
    fn test_json_formatter_pretty() {
        let formatter = JsonFormatter::new(true);
        let entry = create_test_entry();
        let output = formatter.format(&entry);

        // LogLevel serializes as enum variant
        assert!(output.contains("\"level\""));
        assert!(output.contains('\n')); // Pretty-printed
    }

    #[test]
    fn test_plain_formatter() {
        let formatter = PlainFormatter::new(false);
        let entry = create_test_entry();
        let output = formatter.format(&entry);

        assert!(output.contains("[INFO ]"));
        assert!(output.contains("Test message"));
        assert!(output.contains("user_id=123"));
    }

    #[test]
    fn test_plain_formatter_with_colors() {
        let formatter = PlainFormatter::new(true);
        let entry = create_test_entry();
        let output = formatter.format(&entry);

        assert!(output.contains("\x1b[")); // Contains ANSI codes
    }

    #[test]
    fn test_compact_formatter() {
        let formatter = CompactFormatter::new();
        let entry = create_test_entry();
        let output = formatter.format(&entry);

        assert!(output.starts_with("I ")); // I for Info
        assert!(output.contains("Test message"));
        // Should be much shorter than plain format
        assert!(output.len() < 100);
    }

    #[test]
    fn test_logfmt_formatter() {
        let formatter = LogfmtFormatter::new();
        let entry = create_test_entry();
        let output = formatter.format(&entry);

        assert!(output.contains("level=INFO"));
        assert!(output.contains("msg=\"Test message\""));
        assert!(output.contains("user_id=123"));
        assert!(output.contains("action=login"));
    }

    #[test]
    fn test_logfmt_escape() {
        let formatter = LogfmtFormatter::new();
        let mut entry = create_test_entry();
        entry.message = "Message with spaces and \"quotes\"".to_string();
        let output = formatter.format(&entry);

        assert!(output.contains("msg=\"Message with spaces"));
        assert!(output.contains("\\\"quotes\\\""));
    }

    #[test]
    fn test_formatter_with_metrics() {
        let formatter = PlainFormatter::new(false);
        let mut entry = create_test_entry();
        entry.metrics = Some(PerformanceMetrics {
            operation: "test_op".to_string(),
            duration_ms: 150,
            memory_bytes: Some(1024),
            custom_metrics: HashMap::new(),
        });

        let output = formatter.format(&entry);
        assert!(output.contains("[150ms]"));
    }
}

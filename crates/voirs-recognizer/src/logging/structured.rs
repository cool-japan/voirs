//! Structured logging support with JSON formatting.
//!
//! This module provides structured logging capabilities for machine-readable logs
//! that can be easily parsed and analyzed by log aggregation systems.

use super::{HashMap, LogLevel, Utc};
use serde_json::{json, Value};

/// Structured log builder
pub struct StructuredLogBuilder {
    level: LogLevel,
    message: String,
    fields: HashMap<String, Value>,
}

impl StructuredLogBuilder {
    /// Create new structured log builder
    pub fn new(level: LogLevel, message: impl Into<String>) -> Self {
        Self {
            level,
            message: message.into(),
            fields: HashMap::new(),
        }
    }

    /// Add a string field
    pub fn field(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.fields.insert(key.into(), Value::String(value.into()));
        self
    }

    /// Add a numeric field
    pub fn field_num(mut self, key: impl Into<String>, value: f64) -> Self {
        self.fields.insert(key.into(), json!(value));
        self
    }

    /// Add a boolean field
    pub fn field_bool(mut self, key: impl Into<String>, value: bool) -> Self {
        self.fields.insert(key.into(), Value::Bool(value));
        self
    }

    /// Add a JSON value field
    pub fn field_json(mut self, key: impl Into<String>, value: Value) -> Self {
        self.fields.insert(key.into(), value);
        self
    }

    /// Build and return JSON string
    pub fn build(&self) -> Result<String, serde_json::Error> {
        let mut log_obj = json!({
            "level": self.level.as_str(),
            "message": self.message,
            "timestamp": Utc::now().to_rfc3339(),
        });

        if let Some(obj) = log_obj.as_object_mut() {
            for (k, v) in &self.fields {
                obj.insert(k.clone(), v.clone());
            }
        }

        serde_json::to_string(&log_obj)
    }

    /// Build and output to stdout
    pub fn emit(&self) {
        if let Ok(json_str) = self.build() {
            println!("{json_str}");
        }
    }
}

/// Structured logger for specific operations
pub struct OperationLogger {
    operation_name: String,
    start_time: std::time::Instant,
    fields: HashMap<String, Value>,
}

impl OperationLogger {
    /// Start logging an operation
    pub fn start(operation_name: impl Into<String>) -> Self {
        Self {
            operation_name: operation_name.into(),
            start_time: std::time::Instant::now(),
            fields: HashMap::new(),
        }
    }

    /// Add field to operation log
    pub fn field(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.fields.insert(key.into(), Value::String(value.into()));
        self
    }

    /// Add numeric field
    pub fn field_num(mut self, key: impl Into<String>, value: f64) -> Self {
        self.fields.insert(key.into(), json!(value));
        self
    }

    /// Complete operation and log result
    pub fn complete(self, success: bool) {
        let duration = self.start_time.elapsed();
        let mut builder = StructuredLogBuilder::new(
            if success {
                LogLevel::Info
            } else {
                LogLevel::Error
            },
            format!(
                "Operation {} {}",
                self.operation_name,
                if success { "completed" } else { "failed" }
            ),
        );

        builder = builder
            .field("operation", &self.operation_name)
            .field_num("duration_ms", duration.as_millis() as f64)
            .field_bool("success", success);

        for (k, v) in self.fields {
            builder = builder.field_json(k, v);
        }

        builder.emit();
    }
}

/// Create structured log
pub fn structured_log(level: LogLevel, message: impl Into<String>) -> StructuredLogBuilder {
    StructuredLogBuilder::new(level, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_structured_log_builder() {
        let log = StructuredLogBuilder::new(LogLevel::Info, "Test message")
            .field("user_id", "123")
            .field("action", "login")
            .field_num("duration", 150.5)
            .field_bool("success", true);

        let json_str = log.build();
        assert!(json_str.is_ok());

        let json = json_str.unwrap();
        assert!(json.contains("Test message"));
        assert!(json.contains("user_id"));
        assert!(json.contains("123"));
    }

    #[test]
    fn test_operation_logger() {
        let op = OperationLogger::start("test_operation")
            .field("user", "test_user")
            .field_num("count", 42.0);

        std::thread::sleep(std::time::Duration::from_millis(10));
        op.complete(true);
    }

    #[test]
    fn test_structured_log_function() {
        let log = structured_log(LogLevel::Debug, "Debug information")
            .field("module", "test")
            .field("version", "1.0");

        assert!(log.build().is_ok());
    }
}

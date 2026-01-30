//! Telemetry event definitions

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Telemetry event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryEvent {
    /// Unique event ID
    pub id: String,

    /// Event type
    pub event_type: EventType,

    /// Event timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,

    /// Event metadata
    pub metadata: EventMetadata,

    /// User ID (anonymized if privacy controls enabled)
    pub user_id: Option<String>,

    /// Session ID
    pub session_id: String,
}

impl TelemetryEvent {
    /// Create a new telemetry event
    pub fn new(event_type: EventType) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            event_type,
            timestamp: chrono::Utc::now(),
            metadata: EventMetadata::default(),
            user_id: None,
            session_id: uuid::Uuid::new_v4().to_string(),
        }
    }

    /// Create a command execution event
    pub fn command_executed(command: String, duration_ms: u64) -> Self {
        let mut event = Self::new(EventType::CommandExecuted);
        event.metadata.set("command", command);
        event.metadata.set("duration_ms", duration_ms.to_string());
        event
    }

    /// Create a synthesis request event
    pub fn synthesis_request(
        voice: String,
        text_length: usize,
        duration_ms: u64,
        success: bool,
    ) -> Self {
        let mut event = Self::new(EventType::SynthesisRequest);
        event.metadata.set("voice", voice);
        event.metadata.set("text_length", text_length.to_string());
        event.metadata.set("duration_ms", duration_ms.to_string());
        event.metadata.set("success", success.to_string());
        event
    }

    /// Create an error event
    pub fn error(error_type: String, message: String, severity: ErrorSeverity) -> Self {
        let mut event = Self::new(EventType::Error);
        event.metadata.set("error_type", error_type);
        event.metadata.set("message", message);
        event.metadata.set("severity", severity.to_string());
        event
    }

    /// Create a performance event
    pub fn performance(metric_name: String, value: f64, unit: String) -> Self {
        let mut event = Self::new(EventType::Performance);
        event.metadata.set("metric_name", metric_name);
        event.metadata.set("value", value.to_string());
        event.metadata.set("unit", unit);
        event
    }

    /// Set user ID
    pub fn with_user_id(mut self, user_id: String) -> Self {
        self.user_id = Some(user_id);
        self
    }

    /// Set session ID
    pub fn with_session_id(mut self, session_id: String) -> Self {
        self.session_id = session_id;
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.set(key, value);
        self
    }
}

/// Event type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub enum EventType {
    /// Command was executed
    CommandExecuted,

    /// Synthesis request
    SynthesisRequest,

    /// Error occurred
    Error,

    /// Performance metric
    Performance,

    /// Configuration changed
    ConfigurationChanged,

    /// Model loaded
    ModelLoaded,

    /// Voice changed
    VoiceChanged,

    /// Application started
    ApplicationStarted,

    /// Application stopped
    ApplicationStopped,

    /// Feature used
    FeatureUsed,

    /// Custom event
    Custom,
}

impl std::fmt::Display for EventType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EventType::CommandExecuted => write!(f, "command_executed"),
            EventType::SynthesisRequest => write!(f, "synthesis_request"),
            EventType::Error => write!(f, "error"),
            EventType::Performance => write!(f, "performance"),
            EventType::ConfigurationChanged => write!(f, "configuration_changed"),
            EventType::ModelLoaded => write!(f, "model_loaded"),
            EventType::VoiceChanged => write!(f, "voice_changed"),
            EventType::ApplicationStarted => write!(f, "application_started"),
            EventType::ApplicationStopped => write!(f, "application_stopped"),
            EventType::FeatureUsed => write!(f, "feature_used"),
            EventType::Custom => write!(f, "custom"),
        }
    }
}

/// Event metadata container
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EventMetadata {
    data: HashMap<String, String>,
}

impl EventMetadata {
    /// Create new empty metadata
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    /// Set a metadata value
    pub fn set(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.data.insert(key.into(), value.into());
    }

    /// Get a metadata value
    pub fn get(&self, key: &str) -> Option<&String> {
        self.data.get(key)
    }

    /// Check if metadata contains a key
    pub fn contains(&self, key: &str) -> bool {
        self.data.contains_key(key)
    }

    /// Get all metadata keys
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.data.keys()
    }

    /// Get all metadata as hashmap
    pub fn as_map(&self) -> &HashMap<String, String> {
        &self.data
    }

    /// Remove a metadata value
    pub fn remove(&mut self, key: &str) -> Option<String> {
        self.data.remove(key)
    }

    /// Clear all metadata
    pub fn clear(&mut self) {
        self.data.clear();
    }

    /// Number of metadata entries
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if metadata is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorSeverity {
    /// Informational message
    Info,

    /// Warning condition
    Warning,

    /// Error condition
    Error,

    /// Critical error
    Critical,

    /// Fatal error
    Fatal,
}

impl std::fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ErrorSeverity::Info => write!(f, "info"),
            ErrorSeverity::Warning => write!(f, "warning"),
            ErrorSeverity::Error => write!(f, "error"),
            ErrorSeverity::Critical => write!(f, "critical"),
            ErrorSeverity::Fatal => write!(f, "fatal"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_creation() {
        let event = TelemetryEvent::new(EventType::CommandExecuted);
        assert_eq!(event.event_type, EventType::CommandExecuted);
        assert!(!event.id.is_empty());
        assert!(!event.session_id.is_empty());
    }

    #[test]
    fn test_command_executed_event() {
        let event = TelemetryEvent::command_executed("synthesize".to_string(), 1500);
        assert_eq!(event.event_type, EventType::CommandExecuted);
        assert_eq!(
            event
                .metadata
                .get("command")
                .expect("command metadata should exist"),
            "synthesize"
        );
        assert_eq!(
            event
                .metadata
                .get("duration_ms")
                .expect("duration_ms metadata should exist"),
            "1500"
        );
    }

    #[test]
    fn test_synthesis_request_event() {
        let event = TelemetryEvent::synthesis_request("kokoro-en".to_string(), 100, 2000, true);
        assert_eq!(event.event_type, EventType::SynthesisRequest);
        assert_eq!(
            event
                .metadata
                .get("voice")
                .expect("voice metadata should exist"),
            "kokoro-en"
        );
        assert_eq!(
            event
                .metadata
                .get("text_length")
                .expect("text_length metadata should exist"),
            "100"
        );
        assert_eq!(
            event
                .metadata
                .get("duration_ms")
                .expect("duration_ms metadata should exist"),
            "2000"
        );
        assert_eq!(
            event
                .metadata
                .get("success")
                .expect("success metadata should exist"),
            "true"
        );
    }

    #[test]
    fn test_error_event() {
        let event = TelemetryEvent::error(
            "synthesis_error".to_string(),
            "Failed to load model".to_string(),
            ErrorSeverity::Error,
        );
        assert_eq!(event.event_type, EventType::Error);
        assert_eq!(
            event
                .metadata
                .get("error_type")
                .expect("error_type metadata should exist"),
            "synthesis_error"
        );
        assert_eq!(
            event
                .metadata
                .get("message")
                .expect("message metadata should exist"),
            "Failed to load model"
        );
        assert_eq!(
            event
                .metadata
                .get("severity")
                .expect("severity metadata should exist"),
            "error"
        );
    }

    #[test]
    fn test_performance_event() {
        let event = TelemetryEvent::performance("rtf".to_string(), 0.25, "ratio".to_string());
        assert_eq!(event.event_type, EventType::Performance);
        assert_eq!(
            event
                .metadata
                .get("metric_name")
                .expect("metric_name metadata should exist"),
            "rtf"
        );
        assert_eq!(
            event
                .metadata
                .get("value")
                .expect("value metadata should exist"),
            "0.25"
        );
        assert_eq!(
            event
                .metadata
                .get("unit")
                .expect("unit metadata should exist"),
            "ratio"
        );
    }

    #[test]
    fn test_event_builder() {
        let event = TelemetryEvent::new(EventType::Custom)
            .with_user_id("user123".to_string())
            .with_session_id("session456".to_string())
            .with_metadata("key".to_string(), "value".to_string());

        assert_eq!(event.user_id.expect("user_id should be set"), "user123");
        assert_eq!(event.session_id, "session456");
        assert_eq!(
            event
                .metadata
                .get("key")
                .expect("key metadata should exist"),
            "value"
        );
    }

    #[test]
    fn test_event_metadata() {
        let mut metadata = EventMetadata::new();
        assert!(metadata.is_empty());

        metadata.set("key1", "value1");
        metadata.set("key2", "value2");
        assert_eq!(metadata.len(), 2);
        assert!(metadata.contains("key1"));
        assert_eq!(
            metadata.get("key1").expect("key1 metadata should exist"),
            "value1"
        );

        metadata.remove("key1");
        assert_eq!(metadata.len(), 1);
        assert!(!metadata.contains("key1"));

        metadata.clear();
        assert!(metadata.is_empty());
    }

    #[test]
    fn test_event_type_display() {
        assert_eq!(EventType::CommandExecuted.to_string(), "command_executed");
        assert_eq!(EventType::SynthesisRequest.to_string(), "synthesis_request");
        assert_eq!(EventType::Error.to_string(), "error");
    }

    #[test]
    fn test_error_severity_display() {
        assert_eq!(ErrorSeverity::Info.to_string(), "info");
        assert_eq!(ErrorSeverity::Warning.to_string(), "warning");
        assert_eq!(ErrorSeverity::Error.to_string(), "error");
        assert_eq!(ErrorSeverity::Critical.to_string(), "critical");
        assert_eq!(ErrorSeverity::Fatal.to_string(), "fatal");
    }
}

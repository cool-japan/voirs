//! # Integration Traits
//!
//! This module defines traits for VoiRS ecosystem integration, providing
//! standardized interfaces for component coordination and communication.

use crate::RecognitionError;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Trait for VoiRS ecosystem components that can be integrated
#[async_trait]
pub trait EcosystemComponent: Send + Sync {
    /// Get component name
    fn name(&self) -> &str;

    /// Get component version
    fn version(&self) -> &str;

    /// Get component capabilities
    fn capabilities(&self) -> Vec<String>;

    /// Initialize the component
    async fn initialize(&mut self) -> Result<(), RecognitionError>;

    /// Shutdown the component gracefully
    async fn shutdown(&mut self) -> Result<(), RecognitionError>;

    /// Get component health status
    async fn health_check(&self) -> Result<ComponentHealth, RecognitionError>;

    /// Get component configuration
    fn get_config(&self) -> Box<dyn ComponentConfig>;

    /// Update component configuration
    async fn update_config(
        &mut self,
        config: Box<dyn ComponentConfig>,
    ) -> Result<(), RecognitionError>;
}

/// Component health information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    /// Is component healthy
    pub is_healthy: bool,
    /// Health status message
    pub status_message: String,
    /// Last health check time
    pub last_check: std::time::SystemTime,
    /// Resource usage
    pub resource_usage: ResourceMetrics,
    /// Error count since last reset
    pub error_count: u64,
    /// Uptime
    pub uptime: std::time::Duration,
}

/// Resource metrics for components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMetrics {
    /// Memory usage in MB
    pub memory_mb: f32,
    /// CPU usage percentage
    pub cpu_percent: f32,
    /// GPU usage percentage
    pub gpu_percent: Option<f32>,
    /// Network usage in MB
    pub network_mb: f32,
    /// Disk usage in MB
    pub disk_mb: f32,
}

/// Trait for component configuration
pub trait ComponentConfig: Send + Sync {
    /// Get configuration as key-value pairs
    fn as_map(&self) -> HashMap<String, String>;

    /// Update configuration from key-value pairs
    fn from_map(&mut self, map: HashMap<String, String>) -> Result<(), RecognitionError>;

    /// Validate configuration
    fn validate(&self) -> Result<(), RecognitionError>;

    /// Get configuration schema
    fn schema(&self) -> ConfigSchema;
}

/// Configuration schema for validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigSchema {
    /// Required fields
    pub required_fields: Vec<String>,
    /// Optional fields with defaults
    pub optional_fields: HashMap<String, String>,
    /// Field types
    pub field_types: HashMap<String, FieldType>,
    /// Field descriptions
    pub descriptions: HashMap<String, String>,
}

/// Configuration field types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FieldType {
    /// String value
    String,
    /// Integer value
    Integer,
    /// Float value
    Float,
    /// Boolean value
    Boolean,
    /// Array of values
    Array(Box<FieldType>),
    /// Object with nested fields
    Object(HashMap<String, FieldType>),
}

/// Trait for component coordination
#[async_trait]
pub trait ComponentCoordinator: Send + Sync {
    /// Register a component
    async fn register_component(
        &mut self,
        component: Arc<dyn EcosystemComponent>,
    ) -> Result<String, RecognitionError>;

    /// Unregister a component
    async fn unregister_component(&mut self, component_id: &str) -> Result<(), RecognitionError>;

    /// Get component by ID
    async fn get_component(&self, component_id: &str) -> Option<Arc<dyn EcosystemComponent>>;

    /// List all registered components
    async fn list_components(&self) -> Vec<String>;

    /// Send message between components
    async fn send_message(
        &self,
        from: &str,
        to: &str,
        message: ComponentMessage,
    ) -> Result<(), RecognitionError>;

    /// Broadcast message to all components
    async fn broadcast_message(
        &self,
        from: &str,
        message: ComponentMessage,
    ) -> Result<(), RecognitionError>;

    /// Start coordination services
    async fn start(&mut self) -> Result<(), RecognitionError>;

    /// Stop coordination services
    async fn stop(&mut self) -> Result<(), RecognitionError>;
}

/// Message for component communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentMessage {
    /// Message ID
    pub id: String,
    /// Message type
    pub message_type: MessageType,
    /// Message payload
    pub payload: serde_json::Value,
    /// Timestamp
    pub timestamp: std::time::SystemTime,
    /// Priority
    pub priority: MessagePriority,
    /// Reply-to message ID
    pub reply_to: Option<String>,
}

/// Message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageType {
    /// Configuration update
    ConfigUpdate,
    /// Health check request
    HealthCheck,
    /// Performance metrics
    PerformanceMetrics,
    /// Error notification
    Error,
    /// Information/status update
    Info,
    /// Request for data/action
    Request,
    /// Response to request
    Response,
    /// Custom message type
    Custom(String),
}

/// Message priority levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum MessagePriority {
    /// Low priority
    Low,
    /// Normal priority
    Normal,
    /// High priority
    High,
    /// Critical priority
    Critical,
}

/// Trait for performance monitoring integration
#[async_trait]
pub trait PerformanceMonitor: Send + Sync {
    /// Start monitoring a component
    async fn start_monitoring(&mut self, component_id: &str) -> Result<(), RecognitionError>;

    /// Stop monitoring a component
    async fn stop_monitoring(&mut self, component_id: &str) -> Result<(), RecognitionError>;

    /// Get performance metrics for a component
    async fn get_metrics(&self, component_id: &str)
        -> Result<PerformanceMetrics, RecognitionError>;

    /// Get performance metrics for all components
    async fn get_all_metrics(
        &self,
    ) -> Result<HashMap<String, PerformanceMetrics>, RecognitionError>;

    /// Set performance thresholds
    async fn set_thresholds(
        &mut self,
        component_id: &str,
        thresholds: PerformanceThresholds,
    ) -> Result<(), RecognitionError>;

    /// Get performance alerts
    async fn get_alerts(&self) -> Result<Vec<PerformanceAlert>, RecognitionError>;
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Component ID
    pub component_id: String,
    /// Throughput (operations per second)
    pub throughput: f32,
    /// Average latency in milliseconds
    pub avg_latency_ms: f32,
    /// Error rate (0.0 to 1.0)
    pub error_rate: f32,
    /// Resource usage
    pub resource_usage: ResourceMetrics,
    /// Timestamp
    pub timestamp: std::time::SystemTime,
}

/// Performance thresholds for alerting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceThresholds {
    /// Maximum acceptable latency in milliseconds
    pub max_latency_ms: f32,
    /// Maximum acceptable error rate
    pub max_error_rate: f32,
    /// Maximum memory usage in MB
    pub max_memory_mb: f32,
    /// Maximum CPU usage percentage
    pub max_cpu_percent: f32,
}

/// Performance alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAlert {
    /// Alert ID
    pub id: String,
    /// Component ID
    pub component_id: String,
    /// Alert type
    pub alert_type: AlertType,
    /// Alert message
    pub message: String,
    /// Severity level
    pub severity: AlertSeverity,
    /// Timestamp
    pub timestamp: std::time::SystemTime,
    /// Current value that triggered alert
    pub current_value: f32,
    /// Threshold that was exceeded
    pub threshold: f32,
}

/// Alert types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertType {
    /// High latency
    HighLatency,
    /// High error rate
    HighErrorRate,
    /// High memory usage
    HighMemoryUsage,
    /// High CPU usage
    HighCpuUsage,
    /// Component unresponsive
    ComponentUnresponsive,
    /// Custom alert type
    Custom(String),
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum AlertSeverity {
    /// Information
    Info,
    /// Warning
    Warning,
    /// Error
    Error,
    /// Critical
    Critical,
}

/// Trait for configuration management
#[async_trait]
pub trait ConfigurationManager: Send + Sync {
    /// Load configuration for a component
    async fn load_config(
        &self,
        component_id: &str,
    ) -> Result<Box<dyn ComponentConfig>, RecognitionError>;

    /// Save configuration for a component
    async fn save_config(
        &self,
        component_id: &str,
        config: Box<dyn ComponentConfig>,
    ) -> Result<(), RecognitionError>;

    /// Get configuration schema for a component
    async fn get_schema(&self, component_id: &str) -> Result<ConfigSchema, RecognitionError>;

    /// Validate configuration
    async fn validate_config(
        &self,
        component_id: &str,
        config: Box<dyn ComponentConfig>,
    ) -> Result<(), RecognitionError>;

    /// Watch for configuration changes
    async fn watch_config(
        &self,
        component_id: &str,
    ) -> Result<tokio::sync::mpsc::Receiver<Box<dyn ComponentConfig>>, RecognitionError>;
}

/// Trait for logging integration
#[async_trait]
pub trait LoggingIntegration: Send + Sync {
    /// Initialize logging for a component
    async fn initialize_logging(&self, component_id: &str) -> Result<(), RecognitionError>;

    /// Log a message
    async fn log(
        &self,
        component_id: &str,
        level: LogLevel,
        message: &str,
    ) -> Result<(), RecognitionError>;

    /// Log structured data
    async fn log_structured(
        &self,
        component_id: &str,
        level: LogLevel,
        data: serde_json::Value,
    ) -> Result<(), RecognitionError>;

    /// Get log entries for a component
    async fn get_logs(
        &self,
        component_id: &str,
        filter: LogFilter,
    ) -> Result<Vec<LogEntry>, RecognitionError>;
}

/// Log levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    /// Trace level
    Trace,
    /// Debug level
    Debug,
    /// Info level
    Info,
    /// Warning level
    Warn,
    /// Error level
    Error,
    /// Critical level
    Critical,
}

/// Log filter for querying logs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogFilter {
    /// Start time
    pub start_time: Option<std::time::SystemTime>,
    /// End time
    pub end_time: Option<std::time::SystemTime>,
    /// Minimum log level
    pub min_level: Option<LogLevel>,
    /// Maximum number of entries
    pub limit: Option<usize>,
    /// Search query
    pub query: Option<String>,
}

/// Log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    /// Entry ID
    pub id: String,
    /// Component ID
    pub component_id: String,
    /// Log level
    pub level: LogLevel,
    /// Message
    pub message: String,
    /// Structured data
    pub data: Option<serde_json::Value>,
    /// Timestamp
    pub timestamp: std::time::SystemTime,
}

/// Default implementations for common types
impl Default for ComponentHealth {
    fn default() -> Self {
        Self {
            is_healthy: true,
            status_message: "OK".to_string(),
            last_check: std::time::SystemTime::now(),
            resource_usage: ResourceMetrics::default(),
            error_count: 0,
            uptime: std::time::Duration::default(),
        }
    }
}

impl Default for ResourceMetrics {
    fn default() -> Self {
        Self {
            memory_mb: 0.0,
            cpu_percent: 0.0,
            gpu_percent: None,
            network_mb: 0.0,
            disk_mb: 0.0,
        }
    }
}

impl Default for MessagePriority {
    fn default() -> Self {
        Self::Normal
    }
}

impl Default for PerformanceThresholds {
    fn default() -> Self {
        Self {
            max_latency_ms: 1000.0,
            max_error_rate: 0.05,
            max_memory_mb: 2048.0,
            max_cpu_percent: 80.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_component_health_default() {
        let health = ComponentHealth::default();
        assert!(health.is_healthy);
        assert_eq!(health.status_message, "OK");
        assert_eq!(health.error_count, 0);
    }

    #[test]
    fn test_resource_metrics_default() {
        let metrics = ResourceMetrics::default();
        assert_eq!(metrics.memory_mb, 0.0);
        assert_eq!(metrics.cpu_percent, 0.0);
        assert_eq!(metrics.gpu_percent, None);
    }

    #[test]
    fn test_message_priority_ordering() {
        assert!(MessagePriority::Critical > MessagePriority::High);
        assert!(MessagePriority::High > MessagePriority::Normal);
        assert!(MessagePriority::Normal > MessagePriority::Low);
    }

    #[test]
    fn test_alert_severity_ordering() {
        assert!(AlertSeverity::Critical > AlertSeverity::Error);
        assert!(AlertSeverity::Error > AlertSeverity::Warning);
        assert!(AlertSeverity::Warning > AlertSeverity::Info);
    }

    #[test]
    fn test_log_level_ordering() {
        assert!(LogLevel::Critical > LogLevel::Error);
        assert!(LogLevel::Error > LogLevel::Warn);
        assert!(LogLevel::Warn > LogLevel::Info);
        assert!(LogLevel::Info > LogLevel::Debug);
        assert!(LogLevel::Debug > LogLevel::Trace);
    }

    #[test]
    fn test_performance_thresholds_default() {
        let thresholds = PerformanceThresholds::default();
        assert_eq!(thresholds.max_latency_ms, 1000.0);
        assert_eq!(thresholds.max_error_rate, 0.05);
        assert_eq!(thresholds.max_memory_mb, 2048.0);
        assert_eq!(thresholds.max_cpu_percent, 80.0);
    }
}

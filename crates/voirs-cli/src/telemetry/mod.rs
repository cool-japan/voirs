//! Telemetry system for VoiRS CLI
//!
//! Provides privacy-focused usage tracking, performance monitoring, and error reporting.
//! All telemetry is opt-in and respects user privacy with anonymization and local-first storage.

pub mod collector;
pub mod config;
pub mod events;
pub mod export;
pub mod privacy;
pub mod storage;

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

pub use collector::TelemetryCollector;
pub use config::{TelemetryConfig, TelemetryLevel};
pub use events::{EventMetadata, EventType, TelemetryEvent};
pub use export::{ExportFormat, TelemetryExporter};
pub use privacy::{AnonymizationLevel, PrivacyControl};
pub use storage::TelemetryStorage;

/// Telemetry system coordinator
pub struct TelemetrySystem {
    config: Arc<RwLock<TelemetryConfig>>,
    collector: Arc<TelemetryCollector>,
    storage: Arc<RwLock<TelemetryStorage>>,
    exporter: Arc<TelemetryExporter>,
}

impl TelemetrySystem {
    /// Create a new telemetry system
    pub async fn new(config: TelemetryConfig) -> Result<Self, TelemetryError> {
        let config = Arc::new(RwLock::new(config));
        let storage = Arc::new(RwLock::new(TelemetryStorage::new(
            &config.read().await.storage_path,
        )?));
        let collector = Arc::new(TelemetryCollector::new(Arc::clone(&config)).await);
        let exporter = Arc::new(TelemetryExporter::new(Arc::clone(&config)));

        Ok(Self {
            config,
            collector,
            storage,
            exporter,
        })
    }

    /// Record a telemetry event
    pub async fn record_event(&self, event: TelemetryEvent) -> Result<(), TelemetryError> {
        if !self.config.read().await.enabled {
            return Ok(());
        }

        // Apply privacy controls
        let event = self.collector.apply_privacy(event).await?;

        // Store event
        self.storage.write().await.store_event(&event).await?;

        Ok(())
    }

    /// Get telemetry statistics
    pub async fn get_statistics(&self) -> Result<TelemetryStatistics, TelemetryError> {
        self.storage.read().await.get_statistics().await
    }

    /// Export telemetry data
    pub async fn export(
        &self,
        format: ExportFormat,
        output: &std::path::Path,
    ) -> Result<(), TelemetryError> {
        let events = self.storage.read().await.get_all_events().await?;
        self.exporter.export(&events, format, output).await
    }

    /// Clear telemetry data
    pub async fn clear_data(&self) -> Result<(), TelemetryError> {
        self.storage.write().await.clear().await
    }

    /// Update telemetry configuration
    pub async fn update_config(&self, new_config: TelemetryConfig) -> Result<(), TelemetryError> {
        *self.config.write().await = new_config;
        Ok(())
    }

    /// Check if telemetry is enabled
    pub async fn is_enabled(&self) -> bool {
        self.config.read().await.enabled
    }
}

/// Telemetry error types
#[derive(Debug, thiserror::Error)]
pub enum TelemetryError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Privacy violation: {0}")]
    PrivacyViolation(String),

    #[error("Configuration error: {0}")]
    Configuration(String),

    #[error("Storage error: {0}")]
    Storage(String),

    #[error("Export error: {0}")]
    Export(String),
}

/// Telemetry statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryStatistics {
    /// Total number of events
    pub total_events: u64,

    /// Events by type
    pub events_by_type: std::collections::HashMap<String, u64>,

    /// Total synthesis requests
    pub synthesis_requests: u64,

    /// Average synthesis duration (ms)
    pub avg_synthesis_duration: f64,

    /// Total errors
    pub total_errors: u64,

    /// Most used commands
    pub most_used_commands: Vec<(String, u64)>,

    /// Most used voices
    pub most_used_voices: Vec<(String, u64)>,

    /// Time range
    pub start_time: Option<chrono::DateTime<chrono::Utc>>,
    pub end_time: Option<chrono::DateTime<chrono::Utc>>,

    /// Storage size (bytes)
    pub storage_size_bytes: u64,
}

impl Default for TelemetryStatistics {
    fn default() -> Self {
        Self {
            total_events: 0,
            events_by_type: std::collections::HashMap::new(),
            synthesis_requests: 0,
            avg_synthesis_duration: 0.0,
            total_errors: 0,
            most_used_commands: Vec::new(),
            most_used_voices: Vec::new(),
            start_time: None,
            end_time: None,
            storage_size_bytes: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[tokio::test]
    async fn test_telemetry_system_creation() {
        let temp_dir = std::env::temp_dir().join("voirs_telemetry_test");
        let config = TelemetryConfig {
            enabled: true,
            level: TelemetryLevel::Standard,
            storage_path: temp_dir.clone(),
            anonymization: AnonymizationLevel::Medium,
            remote_endpoint: None,
            batch_size: 100,
            flush_interval_secs: 60,
        };

        let system = TelemetrySystem::new(config).await;
        assert!(system.is_ok());

        // Cleanup
        let _ = std::fs::remove_dir_all(temp_dir);
    }

    #[tokio::test]
    async fn test_telemetry_disabled() {
        let temp_dir = std::env::temp_dir().join("voirs_telemetry_test_disabled");
        let config = TelemetryConfig {
            enabled: false,
            level: TelemetryLevel::Minimal,
            storage_path: temp_dir.clone(),
            anonymization: AnonymizationLevel::High,
            remote_endpoint: None,
            batch_size: 100,
            flush_interval_secs: 60,
        };

        let system = TelemetrySystem::new(config).await.unwrap();
        assert!(!system.is_enabled().await);

        // Cleanup
        let _ = std::fs::remove_dir_all(temp_dir);
    }

    #[tokio::test]
    async fn test_record_event() {
        let temp_dir = std::env::temp_dir().join("voirs_telemetry_test_record");
        let config = TelemetryConfig {
            enabled: true,
            level: TelemetryLevel::Standard,
            storage_path: temp_dir.clone(),
            anonymization: AnonymizationLevel::Low,
            remote_endpoint: None,
            batch_size: 100,
            flush_interval_secs: 60,
        };

        let system = TelemetrySystem::new(config).await.unwrap();

        let event = TelemetryEvent {
            id: uuid::Uuid::new_v4().to_string(),
            event_type: EventType::CommandExecuted,
            timestamp: chrono::Utc::now(),
            metadata: EventMetadata::default(),
            user_id: Some("test_user".to_string()),
            session_id: uuid::Uuid::new_v4().to_string(),
        };

        let result = system.record_event(event).await;
        assert!(result.is_ok());

        // Cleanup
        let _ = std::fs::remove_dir_all(temp_dir);
    }

    #[tokio::test]
    async fn test_get_statistics() {
        let temp_dir = std::env::temp_dir().join("voirs_telemetry_test_stats");
        let config = TelemetryConfig {
            enabled: true,
            level: TelemetryLevel::Standard,
            storage_path: temp_dir.clone(),
            anonymization: AnonymizationLevel::Low,
            remote_endpoint: None,
            batch_size: 100,
            flush_interval_secs: 60,
        };

        let system = TelemetrySystem::new(config).await.unwrap();
        let stats = system.get_statistics().await;
        assert!(stats.is_ok());

        // Cleanup
        let _ = std::fs::remove_dir_all(temp_dir);
    }

    #[tokio::test]
    async fn test_clear_data() {
        let temp_dir = std::env::temp_dir().join("voirs_telemetry_test_clear");
        let config = TelemetryConfig {
            enabled: true,
            level: TelemetryLevel::Standard,
            storage_path: temp_dir.clone(),
            anonymization: AnonymizationLevel::Low,
            remote_endpoint: None,
            batch_size: 100,
            flush_interval_secs: 60,
        };

        let system = TelemetrySystem::new(config).await.unwrap();
        let result = system.clear_data().await;
        assert!(result.is_ok());

        // Cleanup
        let _ = std::fs::remove_dir_all(temp_dir);
    }
}

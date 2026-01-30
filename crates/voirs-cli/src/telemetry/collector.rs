//! Telemetry event collector

use std::sync::Arc;
use tokio::sync::RwLock;

use super::{
    config::TelemetryConfig, events::TelemetryEvent, privacy::PrivacyControl, TelemetryError,
};

/// Telemetry collector
pub struct TelemetryCollector {
    config: Arc<RwLock<TelemetryConfig>>,
    privacy: Arc<RwLock<PrivacyControl>>,
}

impl TelemetryCollector {
    /// Create a new telemetry collector
    pub async fn new(config: Arc<RwLock<TelemetryConfig>>) -> Self {
        let anonymization = config.read().await.anonymization;
        let privacy = Arc::new(RwLock::new(PrivacyControl::new(anonymization)));

        Self { config, privacy }
    }

    /// Apply privacy controls to an event
    pub async fn apply_privacy(
        &self,
        event: TelemetryEvent,
    ) -> Result<TelemetryEvent, TelemetryError> {
        let config = self.config.read().await;

        // Check if telemetry is enabled
        if !config.enabled {
            return Err(TelemetryError::Configuration(
                "Telemetry is disabled".to_string(),
            ));
        }

        // Check if this event type should be collected based on level
        if !self.should_collect_event(&event, &config) {
            return Err(TelemetryError::PrivacyViolation(
                "Event type not allowed at current telemetry level".to_string(),
            ));
        }

        // Apply privacy controls
        let privacy = self.privacy.read().await;
        Ok(privacy.anonymize_event(event))
    }

    /// Check if an event should be collected based on configuration
    fn should_collect_event(&self, event: &TelemetryEvent, config: &TelemetryConfig) -> bool {
        use super::events::EventType;

        match event.event_type {
            EventType::Error => config.level.includes_errors(),
            EventType::CommandExecuted | EventType::FeatureUsed => config.level.includes_commands(),
            EventType::Performance => config.level.includes_performance(),
            EventType::Custom => config.level.includes_debug(),
            _ => true, // Other events are always collected
        }
    }

    /// Update privacy control level
    pub async fn update_privacy_level(&self, level: super::privacy::AnonymizationLevel) {
        *self.privacy.write().await = PrivacyControl::new(level);
    }

    /// Get current privacy level
    pub async fn privacy_level(&self) -> super::privacy::AnonymizationLevel {
        self.privacy.read().await.level()
    }

    /// Validate event before collection
    pub fn validate_event(&self, event: &TelemetryEvent) -> Result<(), TelemetryError> {
        // Check event ID is not empty
        if event.id.is_empty() {
            return Err(TelemetryError::Configuration(
                "Event ID cannot be empty".to_string(),
            ));
        }

        // Check session ID is not empty
        if event.session_id.is_empty() {
            return Err(TelemetryError::Configuration(
                "Session ID cannot be empty".to_string(),
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::telemetry::{
        config::{TelemetryConfig, TelemetryLevel},
        events::{EventType, TelemetryEvent},
        privacy::AnonymizationLevel,
    };

    #[tokio::test]
    async fn test_collector_creation() {
        let config = Arc::new(RwLock::new(TelemetryConfig::default()));
        let collector = TelemetryCollector::new(config).await;
        let level = collector.privacy_level().await;
        assert_eq!(level, AnonymizationLevel::Medium);
    }

    #[tokio::test]
    async fn test_apply_privacy() {
        let config = Arc::new(RwLock::new(TelemetryConfig::enabled()));
        let collector = TelemetryCollector::new(config).await;

        let event =
            TelemetryEvent::new(EventType::CommandExecuted).with_user_id("test_user".to_string());

        let result = collector.apply_privacy(event).await;
        assert!(result.is_ok());

        let anonymized = result.unwrap();
        assert_ne!(anonymized.user_id.as_ref().unwrap(), "test_user");
    }

    #[tokio::test]
    async fn test_disabled_telemetry() {
        let config = Arc::new(RwLock::new(TelemetryConfig::disabled()));
        let collector = TelemetryCollector::new(config).await;

        let event = TelemetryEvent::new(EventType::CommandExecuted);
        let result = collector.apply_privacy(event).await;

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            TelemetryError::Configuration(_)
        ));
    }

    #[tokio::test]
    async fn test_event_filtering_by_level() {
        let mut config = TelemetryConfig::enabled();
        config.level = TelemetryLevel::Minimal;
        let config = Arc::new(RwLock::new(config));
        let collector = TelemetryCollector::new(config).await;

        // Error should be allowed
        let error_event = TelemetryEvent::new(EventType::Error);
        assert!(collector.apply_privacy(error_event).await.is_ok());

        // Command should not be allowed at Minimal level
        let command_event = TelemetryEvent::new(EventType::CommandExecuted);
        assert!(collector.apply_privacy(command_event).await.is_err());
    }

    #[tokio::test]
    async fn test_update_privacy_level() {
        let config = Arc::new(RwLock::new(TelemetryConfig::enabled()));
        let collector = TelemetryCollector::new(config).await;

        assert_eq!(collector.privacy_level().await, AnonymizationLevel::Medium);

        collector
            .update_privacy_level(AnonymizationLevel::High)
            .await;
        assert_eq!(collector.privacy_level().await, AnonymizationLevel::High);
    }

    #[tokio::test]
    async fn test_validate_event() {
        let config = Arc::new(RwLock::new(TelemetryConfig::enabled()));
        let collector = TelemetryCollector::new(config).await;

        let valid_event = TelemetryEvent::new(EventType::CommandExecuted);
        assert!(collector.validate_event(&valid_event).is_ok());

        let mut invalid_event = TelemetryEvent::new(EventType::CommandExecuted);
        invalid_event.id = String::new();
        assert!(collector.validate_event(&invalid_event).is_err());

        let mut invalid_session = TelemetryEvent::new(EventType::CommandExecuted);
        invalid_session.session_id = String::new();
        assert!(collector.validate_event(&invalid_session).is_err());
    }

    #[tokio::test]
    async fn test_telemetry_levels() {
        // Test Standard level
        let mut config = TelemetryConfig::enabled();
        config.level = TelemetryLevel::Standard;
        let config = Arc::new(RwLock::new(config));
        let collector = TelemetryCollector::new(config).await;

        assert!(collector
            .apply_privacy(TelemetryEvent::new(EventType::CommandExecuted))
            .await
            .is_ok());
        assert!(collector
            .apply_privacy(TelemetryEvent::new(EventType::Performance))
            .await
            .is_ok());
        assert!(collector
            .apply_privacy(TelemetryEvent::new(EventType::Error))
            .await
            .is_ok());
    }
}

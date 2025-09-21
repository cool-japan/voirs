//! Configuration management and system settings

use super::types::*;
use crate::FeedbackError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Configuration manager for real-time feedback system
#[derive(Debug, Clone)]
pub struct RealtimeConfigManager {
    config: Arc<RwLock<RealtimeConfig>>,
    settings: Arc<RwLock<HashMap<String, ConfigValue>>>,
}

/// Configuration value types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConfigValue {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Array(Vec<ConfigValue>),
}

impl RealtimeConfigManager {
    /// Create a new configuration manager
    pub fn new(config: RealtimeConfig) -> Self {
        Self {
            config: Arc::new(RwLock::new(config)),
            settings: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get the current real-time configuration
    pub async fn get_config(&self) -> RealtimeConfig {
        self.config.read().await.clone()
    }

    /// Update the real-time configuration
    pub async fn update_config(&self, config: RealtimeConfig) -> Result<(), FeedbackError> {
        let mut current_config = self.config.write().await;
        *current_config = config;
        Ok(())
    }

    /// Set a configuration value
    pub async fn set_setting(&self, key: String, value: ConfigValue) -> Result<(), FeedbackError> {
        let mut settings = self.settings.write().await;
        settings.insert(key, value);
        Ok(())
    }

    /// Get a configuration value
    pub async fn get_setting(&self, key: &str) -> Option<ConfigValue> {
        let settings = self.settings.read().await;
        settings.get(key).cloned()
    }

    /// Get all configuration settings
    pub async fn get_all_settings(&self) -> HashMap<String, ConfigValue> {
        self.settings.read().await.clone()
    }

    /// Validate configuration
    pub async fn validate_config(&self) -> Result<(), FeedbackError> {
        let config = self.config.read().await;

        if config.max_latency_ms == 0 {
            return Err(FeedbackError::ConfigurationError {
                message: "max_latency_ms must be greater than 0".to_string(),
            });
        }

        if config.audio_buffer_size == 0 {
            return Err(FeedbackError::ConfigurationError {
                message: "audio_buffer_size must be greater than 0".to_string(),
            });
        }

        // sample_rate validation removed as it's not part of RealtimeConfig

        Ok(())
    }

    /// Reset configuration to defaults
    pub async fn reset_to_defaults(&self) -> Result<(), FeedbackError> {
        let mut config = self.config.write().await;
        *config = RealtimeConfig::default();

        let mut settings = self.settings.write().await;
        settings.clear();

        Ok(())
    }
}

impl Default for RealtimeConfigManager {
    fn default() -> Self {
        Self::new(RealtimeConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_config_manager_creation() {
        let manager = RealtimeConfigManager::default();
        let config = manager.get_config().await;
        assert!(config.max_latency_ms > 0);
    }

    #[tokio::test]
    async fn test_config_update() {
        let manager = RealtimeConfigManager::default();
        let mut new_config = RealtimeConfig::default();
        new_config.max_latency_ms = 200;

        manager.update_config(new_config).await.unwrap();
        let updated_config = manager.get_config().await;
        assert_eq!(updated_config.max_latency_ms, 200);
    }

    #[tokio::test]
    async fn test_setting_management() {
        let manager = RealtimeConfigManager::default();

        manager
            .set_setting(
                "test_key".to_string(),
                ConfigValue::String("test_value".to_string()),
            )
            .await
            .unwrap();

        let value = manager.get_setting("test_key").await;
        assert!(value.is_some());

        if let Some(ConfigValue::String(s)) = value {
            assert_eq!(s, "test_value");
        } else {
            assert!(false, "Expected string value but got: {:?}", value);
        }
    }

    #[tokio::test]
    async fn test_config_validation() {
        let manager = RealtimeConfigManager::default();

        // Valid config should pass
        assert!(manager.validate_config().await.is_ok());

        // Invalid config should fail
        let mut invalid_config = RealtimeConfig::default();
        invalid_config.max_latency_ms = 0;
        manager.update_config(invalid_config).await.unwrap();
        assert!(manager.validate_config().await.is_err());
    }

    #[tokio::test]
    async fn test_reset_to_defaults() {
        let manager = RealtimeConfigManager::default();

        // Modify config and settings
        let mut config = RealtimeConfig::default();
        config.max_latency_ms = 500;
        manager.update_config(config).await.unwrap();
        manager
            .set_setting("test".to_string(), ConfigValue::Boolean(true))
            .await
            .unwrap();

        // Reset to defaults
        manager.reset_to_defaults().await.unwrap();

        let reset_config = manager.get_config().await;
        let default_config = RealtimeConfig::default();
        assert_eq!(reset_config.max_latency_ms, default_config.max_latency_ms);

        let settings = manager.get_all_settings().await;
        assert!(settings.is_empty());
    }
}

//! Configuration manager for runtime configuration management.
//!
//! This module provides a thread-safe configuration manager that can be used
//! to access and update configuration at runtime.

use super::{
    AsrConfig, ConfigError, LoggingConfig, PerformanceConfig, PreprocessingConfig, RecognizerConfig,
};
use parking_lot::RwLock;
use std::sync::Arc;

/// Thread-safe configuration manager
#[derive(Clone)]
pub struct ConfigManager {
    config: Arc<RwLock<RecognizerConfig>>,
}

impl ConfigManager {
    /// Create new configuration manager with given config
    #[must_use]
    pub fn new(config: RecognizerConfig) -> Self {
        Self {
            config: Arc::new(RwLock::new(config)),
        }
    }

    /// Create with default configuration
    #[must_use]
    pub fn default() -> Self {
        Self::new(RecognizerConfig::default())
    }

    /// Get a copy of the current configuration
    #[must_use]
    pub fn get(&self) -> RecognizerConfig {
        self.config.read().clone()
    }

    /// Update configuration
    pub fn update(&self, config: RecognizerConfig) -> Result<(), ConfigError> {
        // Validate before updating
        super::validation::ConfigValidator::validate(&config)?;
        *self.config.write() = config;
        Ok(())
    }

    /// Update specific field
    pub fn update_asr<F>(&self, f: F) -> Result<(), ConfigError>
    where
        F: FnOnce(&mut AsrConfig),
    {
        let mut config = self.config.write();
        f(&mut config.asr);
        super::validation::ConfigValidator::validate_asr(&config.asr)?;
        Ok(())
    }

    /// Update preprocessing configuration
    pub fn update_preprocessing<F>(&self, f: F) -> Result<(), ConfigError>
    where
        F: FnOnce(&mut PreprocessingConfig),
    {
        let mut config = self.config.write();
        f(&mut config.preprocessing);
        super::validation::ConfigValidator::validate_preprocessing(&config.preprocessing)?;
        Ok(())
    }

    /// Update logging configuration
    pub fn update_logging<F>(&self, f: F) -> Result<(), ConfigError>
    where
        F: FnOnce(&mut LoggingConfig),
    {
        let mut config = self.config.write();
        f(&mut config.logging);
        super::validation::ConfigValidator::validate_logging(&config.logging)?;
        Ok(())
    }

    /// Update performance configuration
    pub fn update_performance<F>(&self, f: F) -> Result<(), ConfigError>
    where
        F: FnOnce(&mut PerformanceConfig),
    {
        let mut config = self.config.write();
        f(&mut config.performance);
        super::validation::ConfigValidator::validate_performance(&config.performance)?;
        Ok(())
    }

    /// Get ASR configuration
    #[must_use]
    pub fn get_asr(&self) -> AsrConfig {
        self.config.read().asr.clone()
    }

    /// Get preprocessing configuration
    #[must_use]
    pub fn get_preprocessing(&self) -> PreprocessingConfig {
        self.config.read().preprocessing.clone()
    }

    /// Get logging configuration
    #[must_use]
    pub fn get_logging(&self) -> LoggingConfig {
        self.config.read().logging.clone()
    }

    /// Get performance configuration
    #[must_use]
    pub fn get_performance(&self) -> PerformanceConfig {
        self.config.read().performance.clone()
    }

    /// Save current configuration to file
    pub fn save_to_file(&self, path: impl AsRef<std::path::Path>) -> Result<(), ConfigError> {
        let config = self.get();
        let path = path.as_ref();

        if path.extension().and_then(|s| s.to_str()) == Some("toml") {
            super::loader::ConfigLoader::save_toml(&config, path)
        } else {
            super::loader::ConfigLoader::save_json(&config, path)
        }
    }

    /// Reload configuration from file
    pub fn reload_from_file(&self, path: impl AsRef<std::path::Path>) -> Result<(), ConfigError> {
        let path = path.as_ref();
        let config = if path.extension().and_then(|s| s.to_str()) == Some("toml") {
            super::loader::ConfigLoader::from_toml_file(path)?
        } else {
            super::loader::ConfigLoader::from_json_file(path)?
        };

        self.update(config)
    }
}

impl Default for ConfigManager {
    fn default() -> Self {
        Self::default()
    }
}

/// Global configuration manager
static GLOBAL_CONFIG: once_cell::sync::Lazy<ConfigManager> =
    once_cell::sync::Lazy::new(ConfigManager::default);

/// Get global configuration manager
#[must_use]
pub fn global_config() -> &'static ConfigManager {
    &GLOBAL_CONFIG
}

/// Initialize global configuration
pub fn init_global_config(config: RecognizerConfig) -> Result<(), ConfigError> {
    GLOBAL_CONFIG.update(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_manager_creation() {
        let manager = ConfigManager::default();
        let config = manager.get();
        assert_eq!(config.general.app_name, "voirs-recognizer");
    }

    #[test]
    fn test_config_manager_update() {
        let manager = ConfigManager::default();
        let mut config = RecognizerConfig::default();
        config.asr.default_model = "whisper-large".to_string();

        assert!(manager.update(config).is_ok());

        let updated = manager.get();
        assert_eq!(updated.asr.default_model, "whisper-large");
    }

    #[test]
    fn test_config_manager_update_asr() {
        let manager = ConfigManager::default();

        let result = manager.update_asr(|asr| {
            asr.default_model = "whisper-tiny".to_string();
            asr.enable_gpu = true;
        });

        assert!(result.is_ok());

        let asr = manager.get_asr();
        assert_eq!(asr.default_model, "whisper-tiny");
        assert!(asr.enable_gpu);
    }

    #[test]
    fn test_config_manager_invalid_update() {
        let manager = ConfigManager::default();

        let result = manager.update_asr(|asr| {
            asr.batch_size = 0; // Invalid
        });

        assert!(result.is_err());
    }

    #[test]
    fn test_config_manager_update_preprocessing() {
        let manager = ConfigManager::default();

        let result = manager.update_preprocessing(|preprocessing| {
            preprocessing.target_sample_rate = 22050;
        });

        assert!(result.is_ok());

        let preprocessing = manager.get_preprocessing();
        assert_eq!(preprocessing.target_sample_rate, 22050);
    }

    #[test]
    fn test_config_manager_clone() {
        let manager1 = ConfigManager::default();
        let manager2 = manager1.clone();

        manager1
            .update_asr(|asr| {
                asr.default_model = "test".to_string();
            })
            .unwrap();

        let config1 = manager1.get();
        let config2 = manager2.get();

        // Both should see the update (shared state)
        assert_eq!(config1.asr.default_model, config2.asr.default_model);
    }

    #[test]
    fn test_global_config() {
        let global = global_config();
        let config = global.get();
        assert_eq!(config.general.app_name, "voirs-recognizer");
    }

    #[test]
    fn test_save_and_reload_json() {
        use tempfile::NamedTempFile;

        let manager = ConfigManager::default();
        let temp_file = NamedTempFile::new().unwrap();

        // Modify and save
        manager
            .update_asr(|asr| {
                asr.default_model = "test-model".to_string();
            })
            .unwrap();

        assert!(manager.save_to_file(temp_file.path()).is_ok());

        // Reset
        manager
            .update_asr(|asr| {
                asr.default_model = "original".to_string();
            })
            .unwrap();

        // Reload
        assert!(manager.reload_from_file(temp_file.path()).is_ok());

        let config = manager.get();
        assert_eq!(config.asr.default_model, "test-model");
    }
}

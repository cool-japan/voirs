//! Configuration loader module.
//!
//! This module provides utilities for loading configuration from various sources
//! including files, environment variables, and command-line arguments.

use super::{ConfigError, Environment, RecognizerConfig};
use std::fs;
use std::path::Path;

/// Configuration loader
pub struct ConfigLoader;

impl ConfigLoader {
    /// Load configuration from JSON file
    pub fn from_json_file(path: impl AsRef<Path>) -> Result<RecognizerConfig, ConfigError> {
        let path = path.as_ref();

        if !path.exists() {
            return Err(ConfigError::FileNotFound(path.display().to_string()));
        }

        let contents = fs::read_to_string(path)?;
        let config: RecognizerConfig = serde_json::from_str(&contents)?;

        // Validate loaded configuration
        super::validation::ConfigValidator::validate(&config)?;

        Ok(config)
    }

    /// Load configuration from TOML file
    pub fn from_toml_file(path: impl AsRef<Path>) -> Result<RecognizerConfig, ConfigError> {
        let path = path.as_ref();

        if !path.exists() {
            return Err(ConfigError::FileNotFound(path.display().to_string()));
        }

        let contents = fs::read_to_string(path)?;
        let config: RecognizerConfig =
            toml::from_str(&contents).map_err(|e| ConfigError::InvalidFormat(e.to_string()))?;

        // Validate loaded configuration
        super::validation::ConfigValidator::validate(&config)?;

        Ok(config)
    }

    /// Load configuration from environment variables
    #[must_use]
    pub fn from_env(prefix: &str) -> RecognizerConfig {
        let mut config = RecognizerConfig::default();

        // Load general settings
        if let Ok(val) = std::env::var(format!("{prefix}_APP_NAME")) {
            config.general.app_name = val;
        }

        if let Ok(val) = std::env::var(format!("{prefix}_ENVIRONMENT")) {
            config.general.environment = match val.to_lowercase().as_str() {
                "development" => Environment::Development,
                "staging" => Environment::Staging,
                "production" => Environment::Production,
                _ => Environment::Development,
            };
        }

        // Load ASR settings
        if let Ok(val) = std::env::var(format!("{prefix}_ASR_MODEL")) {
            config.asr.default_model = val;
        }

        if let Ok(val) = std::env::var(format!("{prefix}_ASR_GPU")) {
            config.asr.enable_gpu = val.to_lowercase() == "true" || val == "1";
        }

        if let Ok(val) = std::env::var(format!("{prefix}_ASR_BATCH_SIZE")) {
            if let Ok(size) = val.parse() {
                config.asr.batch_size = size;
            }
        }

        // Load logging settings
        if let Ok(val) = std::env::var(format!("{prefix}_LOG_LEVEL")) {
            config.logging.min_level = val.to_lowercase();
        }

        if let Ok(val) = std::env::var(format!("{prefix}_LOG_FORMAT")) {
            config.logging.format = val.to_lowercase();
        }

        // Load performance settings
        if let Ok(val) = std::env::var(format!("{prefix}_NUM_THREADS")) {
            if let Ok(threads) = val.parse() {
                config.performance.num_threads = threads;
            }
        }

        config
    }

    /// Merge two configurations with override taking precedence
    #[must_use]
    pub fn merge(base: RecognizerConfig, override_config: RecognizerConfig) -> RecognizerConfig {
        RecognizerConfig {
            general: override_config.general,
            asr: override_config.asr,
            preprocessing: override_config.preprocessing,
            multimodal: override_config.multimodal.or(base.multimodal),
            logging: override_config.logging,
            performance: override_config.performance,
            privacy: override_config.privacy.or(base.privacy),
            custom: {
                let mut custom = base.custom;
                custom.extend(override_config.custom);
                custom
            },
        }
    }

    /// Save configuration to JSON file
    pub fn save_json(config: &RecognizerConfig, path: impl AsRef<Path>) -> Result<(), ConfigError> {
        let json = serde_json::to_string_pretty(config)?;
        fs::write(path, json)?;
        Ok(())
    }

    /// Save configuration to TOML file
    pub fn save_toml(config: &RecognizerConfig, path: impl AsRef<Path>) -> Result<(), ConfigError> {
        let toml = toml::to_string_pretty(config)
            .map_err(|e| ConfigError::InvalidFormat(e.to_string()))?;
        fs::write(path, toml)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_save_and_load_json() {
        let config = RecognizerConfig::default();
        let temp_file = NamedTempFile::new().unwrap();

        // Save
        assert!(ConfigLoader::save_json(&config, temp_file.path()).is_ok());

        // Load
        let loaded = ConfigLoader::from_json_file(temp_file.path());
        assert!(loaded.is_ok());

        let loaded_config = loaded.unwrap();
        assert_eq!(loaded_config.general.app_name, config.general.app_name);
    }

    #[test]
    fn test_save_and_load_toml() {
        let config = RecognizerConfig::default();
        let temp_file = NamedTempFile::new().unwrap();

        // Save
        assert!(ConfigLoader::save_toml(&config, temp_file.path()).is_ok());

        // Load
        let loaded = ConfigLoader::from_toml_file(temp_file.path());
        assert!(loaded.is_ok());

        let loaded_config = loaded.unwrap();
        assert_eq!(loaded_config.general.app_name, config.general.app_name);
    }

    #[test]
    fn test_load_nonexistent_file() {
        let result = ConfigLoader::from_json_file("/nonexistent/file.json");
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ConfigError::FileNotFound(_)));
    }

    #[test]
    fn test_from_env() {
        std::env::set_var("TEST_APP_NAME", "test-app");
        std::env::set_var("TEST_ENVIRONMENT", "production");
        std::env::set_var("TEST_ASR_GPU", "true");
        std::env::set_var("TEST_LOG_LEVEL", "debug");

        let config = ConfigLoader::from_env("TEST");

        assert_eq!(config.general.app_name, "test-app");
        assert_eq!(config.general.environment, Environment::Production);
        assert!(config.asr.enable_gpu);
        assert_eq!(config.logging.min_level, "debug");

        // Cleanup
        std::env::remove_var("TEST_APP_NAME");
        std::env::remove_var("TEST_ENVIRONMENT");
        std::env::remove_var("TEST_ASR_GPU");
        std::env::remove_var("TEST_LOG_LEVEL");
    }

    #[test]
    fn test_merge_configs() {
        let mut base = RecognizerConfig::default();
        base.asr.default_model = "whisper-base".to_string();
        base.asr.enable_gpu = false;

        let mut override_config = RecognizerConfig::default();
        override_config.asr.default_model = "whisper-large".to_string();
        override_config.asr.enable_gpu = true;

        let merged = ConfigLoader::merge(base, override_config);

        assert_eq!(merged.asr.default_model, "whisper-large");
        assert!(merged.asr.enable_gpu);
    }
}

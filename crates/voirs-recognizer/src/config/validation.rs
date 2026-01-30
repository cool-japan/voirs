//! Configuration validation module.
//!
//! This module provides comprehensive validation for configuration settings
//! to ensure they meet requirements before being used.

use super::{
    AsrConfig, ConfigError, GeneralConfig, LoggingConfig, MultiModalConfig, PerformanceConfig,
    PreprocessingConfig, PrivacyConfig, RecognizerConfig,
};

/// Configuration validator
pub struct ConfigValidator;

impl ConfigValidator {
    /// Validate complete recognizer configuration
    pub fn validate(config: &RecognizerConfig) -> Result<(), ConfigError> {
        Self::validate_general(&config.general)?;
        Self::validate_asr(&config.asr)?;
        Self::validate_preprocessing(&config.preprocessing)?;

        if let Some(ref multimodal) = config.multimodal {
            Self::validate_multimodal(multimodal)?;
        }

        Self::validate_logging(&config.logging)?;
        Self::validate_performance(&config.performance)?;

        if let Some(ref privacy) = config.privacy {
            Self::validate_privacy(privacy)?;
        }

        Ok(())
    }

    /// Validate general configuration
    pub fn validate_general(config: &GeneralConfig) -> Result<(), ConfigError> {
        if config.app_name.is_empty() {
            return Err(ConfigError::ValidationFailed(
                "app_name cannot be empty".to_string(),
            ));
        }

        if config.app_version.is_empty() {
            return Err(ConfigError::ValidationFailed(
                "app_version cannot be empty".to_string(),
            ));
        }

        Ok(())
    }

    /// Validate ASR configuration
    pub fn validate_asr(config: &AsrConfig) -> Result<(), ConfigError> {
        if config.default_model.is_empty() {
            return Err(ConfigError::ValidationFailed(
                "default_model cannot be empty".to_string(),
            ));
        }

        if config.batch_size == 0 {
            return Err(ConfigError::ValidationFailed(
                "batch_size must be greater than 0".to_string(),
            ));
        }

        if config.max_audio_duration <= 0.0 {
            return Err(ConfigError::ValidationFailed(
                "max_audio_duration must be positive".to_string(),
            ));
        }

        if !(0.0..=1.0).contains(&config.confidence_threshold) {
            return Err(ConfigError::ValidationFailed(
                "confidence_threshold must be between 0.0 and 1.0".to_string(),
            ));
        }

        Ok(())
    }

    /// Validate preprocessing configuration
    pub fn validate_preprocessing(config: &PreprocessingConfig) -> Result<(), ConfigError> {
        if config.target_sample_rate == 0 {
            return Err(ConfigError::ValidationFailed(
                "target_sample_rate must be greater than 0".to_string(),
            ));
        }

        if config.vad_aggressiveness > 3 {
            return Err(ConfigError::ValidationFailed(
                "vad_aggressiveness must be between 0 and 3".to_string(),
            ));
        }

        Ok(())
    }

    /// Validate multi-modal configuration
    pub fn validate_multimodal(config: &MultiModalConfig) -> Result<(), ConfigError> {
        if config.fusion_strategy.is_empty() {
            return Err(ConfigError::ValidationFailed(
                "fusion_strategy cannot be empty".to_string(),
            ));
        }

        // Validate weights sum to approximately 1.0
        let total_weight: f32 = config.modality_weights.values().sum();
        if (total_weight - 1.0).abs() > 0.01 {
            return Err(ConfigError::ValidationFailed(format!(
                "modality_weights must sum to 1.0, got {total_weight}"
            )));
        }

        // Validate all weights are non-negative
        for (modality, weight) in &config.modality_weights {
            if *weight < 0.0 {
                return Err(ConfigError::ValidationFailed(format!(
                    "modality weight for '{modality}' cannot be negative"
                )));
            }
        }

        Ok(())
    }

    /// Validate logging configuration
    pub fn validate_logging(config: &LoggingConfig) -> Result<(), ConfigError> {
        let valid_levels = ["trace", "debug", "info", "warn", "error", "fatal"];
        if !valid_levels.contains(&config.min_level.as_str()) {
            return Err(ConfigError::ValidationFailed(format!(
                "invalid log level: {}",
                config.min_level
            )));
        }

        if config.targets.is_empty() {
            return Err(ConfigError::ValidationFailed(
                "at least one log target must be specified".to_string(),
            ));
        }

        Ok(())
    }

    /// Validate performance configuration
    pub fn validate_performance(config: &PerformanceConfig) -> Result<(), ConfigError> {
        if config.num_threads == 0 {
            return Err(ConfigError::ValidationFailed(
                "num_threads must be greater than 0".to_string(),
            ));
        }

        if config.cache_size_mb == 0 {
            return Err(ConfigError::ValidationFailed(
                "cache_size_mb must be greater than 0".to_string(),
            ));
        }

        if let Some(limit) = config.memory_limit_mb {
            if limit == 0 {
                return Err(ConfigError::ValidationFailed(
                    "memory_limit_mb must be greater than 0 if specified".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Validate privacy configuration
    pub fn validate_privacy(config: &PrivacyConfig) -> Result<(), ConfigError> {
        if config.epsilon <= 0.0 {
            return Err(ConfigError::ValidationFailed(
                "epsilon must be positive".to_string(),
            ));
        }

        if !(0.0..1.0).contains(&config.delta) {
            return Err(ConfigError::ValidationFailed(
                "delta must be between 0.0 and 1.0".to_string(),
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_valid_config() {
        let config = RecognizerConfig::default();
        assert!(ConfigValidator::validate(&config).is_ok());
    }

    #[test]
    fn test_validate_invalid_batch_size() {
        let mut config = RecognizerConfig::default();
        config.asr.batch_size = 0;
        assert!(ConfigValidator::validate(&config).is_err());
    }

    #[test]
    fn test_validate_invalid_confidence() {
        let mut config = RecognizerConfig::default();
        config.asr.confidence_threshold = 1.5;
        assert!(ConfigValidator::validate(&config).is_err());
    }

    #[test]
    fn test_validate_invalid_sample_rate() {
        let mut config = RecognizerConfig::default();
        config.preprocessing.target_sample_rate = 0;
        assert!(ConfigValidator::validate(&config).is_err());
    }

    #[test]
    fn test_validate_invalid_vad_aggressiveness() {
        let mut config = RecognizerConfig::default();
        config.preprocessing.vad_aggressiveness = 4;
        assert!(ConfigValidator::validate(&config).is_err());
    }

    #[test]
    fn test_validate_invalid_modality_weights() {
        let mut config = RecognizerConfig::default();
        let mut multimodal = MultiModalConfig::default();
        multimodal.modality_weights.clear();
        multimodal.modality_weights.insert("audio".to_string(), 0.5);
        config.multimodal = Some(multimodal);

        assert!(ConfigValidator::validate(&config).is_err());
    }

    #[test]
    fn test_validate_invalid_log_level() {
        let mut config = RecognizerConfig::default();
        config.logging.min_level = "invalid".to_string();
        assert!(ConfigValidator::validate(&config).is_err());
    }

    #[test]
    fn test_validate_invalid_num_threads() {
        let mut config = RecognizerConfig::default();
        config.performance.num_threads = 0;
        assert!(ConfigValidator::validate(&config).is_err());
    }

    #[test]
    fn test_validate_invalid_privacy_epsilon() {
        let mut config = RecognizerConfig::default();
        let mut privacy = PrivacyConfig::default();
        privacy.epsilon = -1.0;
        config.privacy = Some(privacy);

        assert!(ConfigValidator::validate(&config).is_err());
    }

    #[test]
    fn test_validate_invalid_privacy_delta() {
        let mut config = RecognizerConfig::default();
        let mut privacy = PrivacyConfig::default();
        privacy.delta = 1.5;
        config.privacy = Some(privacy);

        assert!(ConfigValidator::validate(&config).is_err());
    }
}

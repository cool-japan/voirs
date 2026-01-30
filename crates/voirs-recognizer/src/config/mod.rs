//! Shared configuration management for `VoiRS` recognizer.
//!
//! This module provides unified configuration management that integrates with
//! `VoiRS` SDK and allows for consistent configuration across all modules.

pub mod loader;
pub mod manager;
pub mod validation;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use thiserror::Error;

/// Configuration errors
#[derive(Debug, Error)]
pub enum ConfigError {
    /// Configuration file not found
    #[error("Configuration file not found: {0}")]
    FileNotFound(String),

    /// Invalid configuration format
    #[error("Invalid configuration: {0}")]
    InvalidFormat(String),

    /// Missing required field
    #[error("Missing required field: {0}")]
    MissingField(String),

    /// Validation failed
    #[error("Validation error: {0}")]
    ValidationFailed(String),

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
}

/// Master configuration combining all subsystem configurations
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RecognizerConfig {
    /// General settings
    pub general: GeneralConfig,

    /// ASR configuration
    pub asr: AsrConfig,

    /// Audio preprocessing configuration
    pub preprocessing: PreprocessingConfig,

    /// Multi-modal configuration
    pub multimodal: Option<MultiModalConfig>,

    /// Logging configuration
    pub logging: LoggingConfig,

    /// Performance settings
    pub performance: PerformanceConfig,

    /// Privacy settings
    pub privacy: Option<PrivacyConfig>,

    /// Custom application settings
    pub custom: HashMap<String, serde_json::Value>,
}

/// General configuration settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneralConfig {
    /// Application name
    pub app_name: String,

    /// Application version
    pub app_version: String,

    /// Environment (development, staging, production)
    pub environment: Environment,

    /// Data directory
    pub data_dir: PathBuf,

    /// Cache directory
    pub cache_dir: PathBuf,

    /// Model directory
    pub model_dir: PathBuf,

    /// Enable telemetry
    pub enable_telemetry: bool,
}

impl Default for GeneralConfig {
    fn default() -> Self {
        Self {
            app_name: "voirs-recognizer".to_string(),
            app_version: env!("CARGO_PKG_VERSION").to_string(),
            environment: Environment::Development,
            data_dir: PathBuf::from("./data"),
            cache_dir: PathBuf::from("./cache"),
            model_dir: PathBuf::from("./models"),
            enable_telemetry: false,
        }
    }
}

/// Environment type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Environment {
    /// Development environment
    Development,
    /// Staging environment
    Staging,
    /// Production environment
    Production,
}

/// ASR configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsrConfig {
    /// Default ASR model
    pub default_model: String,

    /// Model-specific settings
    pub model_settings: HashMap<String, serde_json::Value>,

    /// Enable GPU acceleration
    pub enable_gpu: bool,

    /// Batch size for processing
    pub batch_size: usize,

    /// Maximum audio duration in seconds
    pub max_audio_duration: f32,

    /// Language code (e.g., "en", "ja", "auto")
    pub language: String,

    /// Enable word timestamps
    pub enable_word_timestamps: bool,

    /// Confidence threshold
    pub confidence_threshold: f32,
}

impl Default for AsrConfig {
    fn default() -> Self {
        Self {
            default_model: "whisper-base".to_string(),
            model_settings: HashMap::new(),
            enable_gpu: false,
            batch_size: 1,
            max_audio_duration: 30.0,
            language: "auto".to_string(),
            enable_word_timestamps: true,
            confidence_threshold: 0.5,
        }
    }
}

/// Audio preprocessing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingConfig {
    /// Enable noise reduction
    pub enable_noise_reduction: bool,

    /// Enable automatic gain control
    pub enable_agc: bool,

    /// Enable echo cancellation
    pub enable_echo_cancellation: bool,

    /// Sample rate for processing
    pub target_sample_rate: u32,

    /// Enable voice activity detection
    pub enable_vad: bool,

    /// VAD aggressiveness (0-3)
    pub vad_aggressiveness: u8,
}

impl Default for PreprocessingConfig {
    fn default() -> Self {
        Self {
            enable_noise_reduction: true,
            enable_agc: true,
            enable_echo_cancellation: false,
            target_sample_rate: 16000,
            enable_vad: true,
            vad_aggressiveness: 2,
        }
    }
}

/// Multi-modal configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalConfig {
    /// Enable audio-visual processing
    pub enable_audio_visual: bool,

    /// Enable gesture recognition
    pub enable_gesture: bool,

    /// Enable context-aware processing
    pub enable_context: bool,

    /// Fusion strategy
    pub fusion_strategy: String,

    /// Confidence weights for each modality
    pub modality_weights: HashMap<String, f32>,
}

impl Default for MultiModalConfig {
    fn default() -> Self {
        let mut weights = HashMap::new();
        weights.insert("audio".to_string(), 0.5);
        weights.insert("visual".to_string(), 0.3);
        weights.insert("gesture".to_string(), 0.1);
        weights.insert("context".to_string(), 0.1);

        Self {
            enable_audio_visual: false,
            enable_gesture: false,
            enable_context: true,
            fusion_strategy: "late_fusion".to_string(),
            modality_weights: weights,
        }
    }
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Minimum log level
    pub min_level: String,

    /// Enable structured logging
    pub structured: bool,

    /// Log output format
    pub format: String,

    /// Log targets
    pub targets: Vec<String>,

    /// Enable performance logging
    pub enable_performance_logging: bool,

    /// Performance threshold in milliseconds
    pub performance_threshold_ms: u64,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            min_level: "info".to_string(),
            structured: true,
            format: "json".to_string(),
            targets: vec!["stdout".to_string()],
            enable_performance_logging: true,
            performance_threshold_ms: 100,
        }
    }
}

/// Performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Number of worker threads
    pub num_threads: usize,

    /// Enable SIMD optimizations
    pub enable_simd: bool,

    /// Memory limit in MB
    pub memory_limit_mb: Option<usize>,

    /// Enable caching
    pub enable_caching: bool,

    /// Cache size in MB
    pub cache_size_mb: usize,

    /// Enable model quantization
    pub enable_quantization: bool,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            num_threads: num_cpus::get(),
            enable_simd: true,
            memory_limit_mb: Some(2048),
            enable_caching: true,
            cache_size_mb: 512,
            enable_quantization: false,
        }
    }
}

/// Privacy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyConfig {
    /// Enable differential privacy
    pub enable_differential_privacy: bool,

    /// Privacy budget epsilon
    pub epsilon: f32,

    /// Privacy budget delta
    pub delta: f32,

    /// Enable federated learning
    pub enable_federated_learning: bool,

    /// Enable encrypted inference
    pub enable_encrypted_inference: bool,

    /// Enable on-device processing only
    pub on_device_only: bool,
}

impl Default for PrivacyConfig {
    fn default() -> Self {
        Self {
            enable_differential_privacy: false,
            epsilon: 1.0,
            delta: 1e-5,
            enable_federated_learning: false,
            enable_encrypted_inference: false,
            on_device_only: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = RecognizerConfig::default();
        assert_eq!(config.general.app_name, "voirs-recognizer");
        assert_eq!(config.asr.default_model, "whisper-base");
        assert_eq!(config.preprocessing.target_sample_rate, 16000);
    }

    #[test]
    fn test_general_config() {
        let config = GeneralConfig::default();
        assert_eq!(config.environment, Environment::Development);
        assert!(!config.enable_telemetry);
    }

    #[test]
    fn test_asr_config() {
        let config = AsrConfig::default();
        assert!(config.enable_word_timestamps);
        assert_eq!(config.confidence_threshold, 0.5);
        assert_eq!(config.batch_size, 1);
    }

    #[test]
    fn test_preprocessing_config() {
        let config = PreprocessingConfig::default();
        assert!(config.enable_noise_reduction);
        assert!(config.enable_agc);
        assert_eq!(config.vad_aggressiveness, 2);
    }

    #[test]
    fn test_multimodal_config() {
        let config = MultiModalConfig::default();
        assert!(!config.enable_audio_visual);
        assert!(!config.enable_gesture);
        assert!(config.enable_context);
        assert_eq!(config.fusion_strategy, "late_fusion");
        assert_eq!(config.modality_weights.len(), 4);
    }

    #[test]
    fn test_logging_config() {
        let config = LoggingConfig::default();
        assert_eq!(config.min_level, "info");
        assert!(config.structured);
        assert_eq!(config.format, "json");
    }

    #[test]
    fn test_performance_config() {
        let config = PerformanceConfig::default();
        assert!(config.enable_simd);
        assert!(config.enable_caching);
        assert_eq!(config.cache_size_mb, 512);
    }

    #[test]
    fn test_privacy_config() {
        let config = PrivacyConfig::default();
        assert!(!config.enable_differential_privacy);
        assert!(!config.enable_federated_learning);
        assert!(!config.on_device_only);
    }

    #[test]
    fn test_config_serialization() {
        let config = RecognizerConfig::default();
        let json = serde_json::to_string(&config);
        assert!(json.is_ok());

        let deserialized: Result<RecognizerConfig, _> = serde_json::from_str(&json.unwrap());
        assert!(deserialized.is_ok());
    }

    #[test]
    fn test_environment_variants() {
        assert_eq!(Environment::Development, Environment::Development);
        assert_ne!(Environment::Development, Environment::Production);
    }
}

//! Configuration system for VoiRS G2P library.

use crate::{G2pError, LanguageCode, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use tracing::debug;

/// Main configuration structure
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct G2pConfig {
    /// General configuration
    pub general: GeneralConfig,
    /// Backend configurations
    pub backends: BackendConfigs,
    /// Language detection settings
    pub language_detection: LanguageDetectionConfig,
    /// Preprocessing settings
    pub preprocessing: PreprocessingConfig,
    /// Performance settings
    pub performance: PerformanceConfig,
}

/// General configuration settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneralConfig {
    /// Default language for text processing
    pub default_language: LanguageCode,
    /// Default backend to use
    pub default_backend: String,
    /// Fallback backends in order of preference
    pub fallback_backends: Vec<String>,
    /// Whether to enable load balancing
    pub enable_load_balancing: bool,
    /// Log level for tracing
    pub log_level: String,
}

/// Backend configuration settings
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BackendConfigs {
    /// Rule-based backend config
    pub rule_based: RuleBasedConfig,
    /// Neural backend config
    pub neural: NeuralConfig,
    /// Phonetisaurus backend config
    pub phonetisaurus: PhonetisaurusConfig,
    /// OpenJTalk backend config
    pub openjtalk: OpenJTalkConfig,
}

/// Rule-based backend configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleBasedConfig {
    /// Whether the backend is enabled
    pub enabled: bool,
    /// Backend priority
    pub priority: u32,
    /// Rule files to load
    pub rule_files: HashMap<LanguageCode, String>,
    /// Custom rules
    pub custom_rules: Vec<CustomRule>,
}

/// Neural backend configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralConfig {
    /// Whether the backend is enabled
    pub enabled: bool,
    /// Backend priority
    pub priority: u32,
    /// Model files for different languages
    pub model_files: HashMap<LanguageCode, String>,
    /// Model cache directory
    pub model_cache_dir: String,
    /// Maximum batch size for inference
    pub batch_size: usize,
    /// Whether to use GPU acceleration
    pub use_gpu: bool,
}

/// Phonetisaurus backend configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhonetisaurusConfig {
    /// Whether the backend is enabled
    pub enabled: bool,
    /// Backend priority
    pub priority: u32,
    /// FST model files
    pub fst_models: HashMap<LanguageCode, String>,
    /// Model download URLs
    pub model_urls: HashMap<LanguageCode, String>,
    /// Maximum number of pronunciation variants
    pub max_variants: usize,
}

/// OpenJTalk backend configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenJTalkConfig {
    /// Whether the backend is enabled
    pub enabled: bool,
    /// Backend priority
    pub priority: u32,
    /// Dictionary directory
    pub dictionary_dir: String,
    /// Voice model file
    pub voice_model: String,
    /// Custom dictionary files
    pub custom_dictionaries: Vec<String>,
}

/// Custom rule definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomRule {
    /// Rule pattern
    pub pattern: String,
    /// Phoneme output
    pub phoneme: String,
    /// Left context
    pub left_context: Option<String>,
    /// Right context
    pub right_context: Option<String>,
    /// Rule priority
    pub priority: u32,
}

/// Language detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageDetectionConfig {
    /// Whether to enable automatic language detection
    pub enabled: bool,
    /// Detection method: "rule", "statistical", "mixed"
    pub method: String,
    /// Minimum confidence threshold
    pub confidence_threshold: f32,
    /// Maximum text length for detection
    pub max_text_length: usize,
    /// Languages to consider for detection
    pub target_languages: Vec<LanguageCode>,
}

/// Preprocessing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingConfig {
    /// Whether to enable text normalization
    pub enable_normalization: bool,
    /// Whether to expand numbers
    pub expand_numbers: bool,
    /// Whether to expand abbreviations
    pub expand_abbreviations: bool,
    /// Whether to handle URLs
    pub handle_urls: bool,
    /// Whether to remove punctuation
    pub remove_punctuation: bool,
    /// Custom abbreviations
    pub custom_abbreviations: HashMap<String, String>,
}

/// Performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Maximum number of worker threads
    pub max_threads: usize,
    /// Model cache size in MB
    pub cache_size_mb: usize,
    /// Request timeout in seconds
    pub timeout_seconds: u64,
    /// Whether to enable memory profiling
    pub enable_profiling: bool,
}

impl Default for GeneralConfig {
    fn default() -> Self {
        Self {
            default_language: LanguageCode::EnUs,
            default_backend: "rule_based".to_string(),
            fallback_backends: vec!["rule_based".to_string(), "dummy".to_string()],
            enable_load_balancing: false,
            log_level: "info".to_string(),
        }
    }
}

impl Default for RuleBasedConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            priority: 10,
            rule_files: HashMap::new(),
            custom_rules: Vec::new(),
        }
    }
}

impl Default for NeuralConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            priority: 20,
            model_files: HashMap::new(),
            model_cache_dir: "models".to_string(),
            batch_size: 32,
            use_gpu: false,
        }
    }
}

impl Default for PhonetisaurusConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            priority: 15,
            fst_models: HashMap::new(),
            model_urls: HashMap::new(),
            max_variants: 3,
        }
    }
}

impl Default for OpenJTalkConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            priority: 25,
            dictionary_dir: "openjtalk".to_string(),
            voice_model: "default.htsvoice".to_string(),
            custom_dictionaries: Vec::new(),
        }
    }
}

impl Default for LanguageDetectionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            method: "mixed".to_string(),
            confidence_threshold: 0.7,
            max_text_length: 1000,
            target_languages: vec![
                LanguageCode::EnUs,
                LanguageCode::EnGb,
                LanguageCode::De,
                LanguageCode::Fr,
                LanguageCode::Es,
                LanguageCode::Ja,
                LanguageCode::ZhCn,
                LanguageCode::Ko,
            ],
        }
    }
}

impl Default for PreprocessingConfig {
    fn default() -> Self {
        Self {
            enable_normalization: true,
            expand_numbers: true,
            expand_abbreviations: true,
            handle_urls: true,
            remove_punctuation: false,
            custom_abbreviations: HashMap::new(),
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            max_threads: num_cpus::get(),
            cache_size_mb: 100,
            timeout_seconds: 30,
            enable_profiling: false,
        }
    }
}

/// Configuration manager for loading and managing G2P configurations
pub struct ConfigManager {
    config: G2pConfig,
    config_file_path: Option<String>,
}

impl ConfigManager {
    /// Create a new configuration manager with default settings
    pub fn new() -> Self {
        Self {
            config: G2pConfig::default(),
            config_file_path: None,
        }
    }

    /// Load configuration from TOML file
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path)
            .map_err(|e| G2pError::ConfigError(format!("Failed to read config file: {e}")))?;

        let config: G2pConfig = toml::from_str(&content)
            .map_err(|e| G2pError::ConfigError(format!("Failed to parse config file: {e}")))?;

        debug!("Loaded configuration from: {}", path.display());
        Ok(Self {
            config,
            config_file_path: Some(path.to_string_lossy().to_string()),
        })
    }

    /// Load configuration from environment variables
    pub fn load_from_env(&mut self) -> Result<()> {
        // Load general settings
        if let Ok(default_lang) = std::env::var("VOIRS_G2P_DEFAULT_LANGUAGE") {
            if let Ok(lang) = self.parse_language_code(&default_lang) {
                self.config.general.default_language = lang;
            }
        }

        if let Ok(default_backend) = std::env::var("VOIRS_G2P_DEFAULT_BACKEND") {
            self.config.general.default_backend = default_backend;
        }

        if let Ok(log_level) = std::env::var("VOIRS_G2P_LOG_LEVEL") {
            self.config.general.log_level = log_level;
        }

        // Load backend settings
        if let Ok(enable_rule_based) = std::env::var("VOIRS_G2P_ENABLE_RULE_BASED") {
            self.config.backends.rule_based.enabled = enable_rule_based.parse().unwrap_or(true);
        }

        if let Ok(enable_neural) = std::env::var("VOIRS_G2P_ENABLE_NEURAL") {
            self.config.backends.neural.enabled = enable_neural.parse().unwrap_or(false);
        }

        // Load performance settings
        if let Ok(max_threads) = std::env::var("VOIRS_G2P_MAX_THREADS") {
            if let Ok(threads) = max_threads.parse::<usize>() {
                self.config.performance.max_threads = threads;
            }
        }

        if let Ok(cache_size) = std::env::var("VOIRS_G2P_CACHE_SIZE_MB") {
            if let Ok(size) = cache_size.parse::<usize>() {
                self.config.performance.cache_size_mb = size;
            }
        }

        debug!("Loaded configuration from environment variables");
        Ok(())
    }

    /// Save configuration to TOML file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        let content = toml::to_string_pretty(&self.config)
            .map_err(|e| G2pError::ConfigError(format!("Failed to serialize config: {e}")))?;

        std::fs::write(path, content)
            .map_err(|e| G2pError::ConfigError(format!("Failed to write config file: {e}")))?;

        debug!("Saved configuration to: {}", path.display());
        Ok(())
    }

    /// Get the current configuration
    pub fn config(&self) -> &G2pConfig {
        &self.config
    }

    /// Get mutable reference to configuration
    pub fn config_mut(&mut self) -> &mut G2pConfig {
        &mut self.config
    }

    /// Update configuration at runtime
    pub fn update_config<F>(&mut self, update_fn: F) -> Result<()>
    where
        F: FnOnce(&mut G2pConfig) -> Result<()>,
    {
        update_fn(&mut self.config)?;
        self.validate_config()?;
        debug!("Updated configuration at runtime");
        Ok(())
    }

    /// Validate configuration
    pub fn validate_config(&self) -> Result<()> {
        // Validate general settings
        if self.config.general.default_backend.is_empty() {
            return Err(G2pError::ConfigError(
                "Default backend cannot be empty".to_string(),
            ));
        }

        // Validate performance settings
        if self.config.performance.max_threads == 0 {
            return Err(G2pError::ConfigError(
                "Max threads must be greater than 0".to_string(),
            ));
        }

        if self.config.performance.cache_size_mb == 0 {
            return Err(G2pError::ConfigError(
                "Cache size must be greater than 0".to_string(),
            ));
        }

        // Validate language detection settings
        if self.config.language_detection.confidence_threshold < 0.0
            || self.config.language_detection.confidence_threshold > 1.0
        {
            return Err(G2pError::ConfigError(
                "Confidence threshold must be between 0.0 and 1.0".to_string(),
            ));
        }

        // Validate backend priorities
        let priorities = vec![
            self.config.backends.rule_based.priority,
            self.config.backends.neural.priority,
            self.config.backends.phonetisaurus.priority,
            self.config.backends.openjtalk.priority,
        ];

        for priority in priorities {
            if priority > 100 {
                return Err(G2pError::ConfigError(
                    "Backend priority cannot exceed 100".to_string(),
                ));
            }
        }

        debug!("Configuration validation passed");
        Ok(())
    }

    /// Get configuration file path
    pub fn config_file_path(&self) -> Option<&String> {
        self.config_file_path.as_ref()
    }

    /// Generate default configuration file
    pub fn generate_default_config_file<P: AsRef<Path>>(path: P) -> Result<()> {
        let config = G2pConfig::default();
        let content = toml::to_string_pretty(&config).map_err(|e| {
            G2pError::ConfigError(format!("Failed to serialize default config: {e}"))
        })?;

        std::fs::write(path.as_ref(), content).map_err(|e| {
            G2pError::ConfigError(format!("Failed to write default config file: {e}"))
        })?;

        debug!(
            "Generated default configuration file: {}",
            path.as_ref().display()
        );
        Ok(())
    }

    /// Parse language code from string
    fn parse_language_code(&self, lang_str: &str) -> Result<LanguageCode> {
        match lang_str.to_lowercase().as_str() {
            "en-us" | "en_us" | "en" => Ok(LanguageCode::EnUs),
            "en-gb" | "en_gb" => Ok(LanguageCode::EnGb),
            "ja" | "jp" => Ok(LanguageCode::Ja),
            "zh-cn" | "zh_cn" | "zh" => Ok(LanguageCode::ZhCn),
            "ko" | "kr" => Ok(LanguageCode::Ko),
            "de" => Ok(LanguageCode::De),
            "fr" => Ok(LanguageCode::Fr),
            "es" => Ok(LanguageCode::Es),
            _ => Err(G2pError::ConfigError(format!(
                "Unsupported language code: {lang_str}"
            ))),
        }
    }
}

impl Default for ConfigManager {
    fn default() -> Self {
        Self::new()
    }
}

// Add missing num_cpus dependency placeholder
mod num_cpus {
    pub fn get() -> usize {
        4 // Default to 4 threads
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_default_config_creation() {
        let config = G2pConfig::default();
        assert_eq!(config.general.default_language, LanguageCode::EnUs);
        assert_eq!(config.general.default_backend, "rule_based");
        assert!(config.backends.rule_based.enabled);
        assert!(!config.backends.neural.enabled);
    }

    #[test]
    fn test_config_manager_creation() {
        let manager = ConfigManager::new();
        assert_eq!(manager.config.general.default_language, LanguageCode::EnUs);
        assert!(manager.config_file_path.is_none());
    }

    #[test]
    fn test_config_validation() {
        let manager = ConfigManager::new();
        assert!(manager.validate_config().is_ok());
    }

    #[test]
    fn test_invalid_config_validation() {
        let mut manager = ConfigManager::new();
        manager.config.general.default_backend = "".to_string();
        assert!(manager.validate_config().is_err());
    }

    #[test]
    fn test_config_update() {
        let mut manager = ConfigManager::new();

        let result = manager.update_config(|config| {
            config.general.default_language = LanguageCode::De;
            Ok(())
        });

        assert!(result.is_ok());
        assert_eq!(manager.config.general.default_language, LanguageCode::De);
    }

    #[test]
    fn test_language_code_parsing() {
        let manager = ConfigManager::new();

        assert_eq!(
            manager.parse_language_code("en-us").unwrap(),
            LanguageCode::EnUs
        );
        assert_eq!(
            manager.parse_language_code("EN_US").unwrap(),
            LanguageCode::EnUs
        );
        assert_eq!(manager.parse_language_code("ja").unwrap(), LanguageCode::Ja);
        assert_eq!(manager.parse_language_code("de").unwrap(), LanguageCode::De);
        assert!(manager.parse_language_code("invalid").is_err());
    }

    #[test]
    fn test_config_file_generation() {
        let temp_dir = std::env::temp_dir();
        let config_path = temp_dir.join("test_config.toml");

        let result = ConfigManager::generate_default_config_file(&config_path);
        assert!(result.is_ok());

        // Verify file was created
        assert!(config_path.exists());

        // Clean up
        let _ = fs::remove_file(config_path);
    }

    #[test]
    fn test_config_file_load_save() {
        let temp_dir = std::env::temp_dir();
        let config_path = temp_dir.join("test_load_save.toml");

        // Create and save config
        let mut manager = ConfigManager::new();
        manager.config.general.default_language = LanguageCode::De;
        manager.save_to_file(&config_path).unwrap();

        // Load config
        let loaded_manager = ConfigManager::load_from_file(&config_path).unwrap();
        assert_eq!(
            loaded_manager.config.general.default_language,
            LanguageCode::De
        );

        // Clean up
        let _ = fs::remove_file(config_path);
    }

    #[test]
    fn test_env_var_loading() {
        std::env::set_var("VOIRS_G2P_DEFAULT_LANGUAGE", "de");
        std::env::set_var("VOIRS_G2P_DEFAULT_BACKEND", "neural");
        std::env::set_var("VOIRS_G2P_LOG_LEVEL", "debug");

        let mut manager = ConfigManager::new();
        manager.load_from_env().unwrap();

        assert_eq!(manager.config.general.default_language, LanguageCode::De);
        assert_eq!(manager.config.general.default_backend, "neural");
        assert_eq!(manager.config.general.log_level, "debug");

        // Clean up
        std::env::remove_var("VOIRS_G2P_DEFAULT_LANGUAGE");
        std::env::remove_var("VOIRS_G2P_DEFAULT_BACKEND");
        std::env::remove_var("VOIRS_G2P_LOG_LEVEL");
    }
}

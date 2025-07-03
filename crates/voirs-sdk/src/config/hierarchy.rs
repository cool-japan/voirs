//! Configuration hierarchy and inheritance system.
//!
//! This module handles the hierarchical structure of configurations, allowing
//! for composition, inheritance, and override mechanisms. It supports:
//!
//! - Configuration merging and inheritance
//! - Default value resolution
//! - Nested configuration structures
//! - Configuration profiles and environments
//! - Override and fallback mechanisms

use crate::types::{QualityLevel, SynthesisConfig, AudioFormat};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    path::PathBuf,
};

/// Trait for configuration types that support hierarchical composition
pub trait ConfigHierarchy: Clone + Default {
    /// Merge this configuration with another, giving priority to the other
    fn merge_with(&mut self, other: &Self);
    
    /// Create a configuration that inherits from a parent
    fn inherit_from(parent: &Self) -> Self {
        let mut config = parent.clone();
        config.merge_with(&Self::default());
        config
    }
    
    /// Validate the configuration and return any errors
    fn validate(&self) -> Result<(), ConfigValidationError> {
        Ok(())
    }
}

/// Configuration validation error
#[derive(Debug, Clone)]
pub struct ConfigValidationError {
    pub field: String,
    pub message: String,
}

impl std::fmt::Display for ConfigValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Configuration validation error in '{}': {}", self.field, self.message)
    }
}

impl std::error::Error for ConfigValidationError {}

/// Main pipeline configuration with hierarchical support
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PipelineConfig {
    /// Device to use for computation (cpu, cuda, metal, etc.)
    pub device: String,
    
    /// Enable GPU acceleration if available
    pub use_gpu: bool,
    
    /// Number of threads for CPU computation
    pub num_threads: Option<usize>,
    
    /// Cache directory for models and temporary files
    pub cache_dir: Option<PathBuf>,
    
    /// Maximum cache size in MB
    pub max_cache_size_mb: u32,
    
    /// Default synthesis configuration
    pub default_synthesis: SynthesisConfig,
    
    /// Model loading configuration
    pub model_loading: ModelLoadingConfig,
    
    /// Audio processing configuration
    pub audio_processing: AudioProcessingConfig,
    
    /// Logging configuration
    pub logging: LoggingConfig,
    
    /// Configuration profile name (for environment-specific configs)
    pub profile: Option<String>,
    
    /// Parent configuration to inherit from
    pub inherits_from: Option<String>,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            device: "cpu".to_string(),
            use_gpu: false,
            num_threads: None,
            cache_dir: None,
            max_cache_size_mb: 1024,
            default_synthesis: SynthesisConfig::default(),
            model_loading: ModelLoadingConfig::default(),
            audio_processing: AudioProcessingConfig::default(),
            logging: LoggingConfig::default(),
            profile: None,
            inherits_from: None,
        }
    }
}

impl ConfigHierarchy for PipelineConfig {
    fn merge_with(&mut self, other: &Self) {
        // Merge primitive fields (other takes precedence if not default)
        if other.device != "cpu" {
            self.device = other.device.clone();
        }
        if other.use_gpu {
            self.use_gpu = other.use_gpu;
        }
        if other.num_threads.is_some() {
            self.num_threads = other.num_threads;
        }
        if other.cache_dir.is_some() {
            self.cache_dir = other.cache_dir.clone();
        }
        if other.max_cache_size_mb != 1024 {
            self.max_cache_size_mb = other.max_cache_size_mb;
        }
        if other.profile.is_some() {
            self.profile = other.profile.clone();
        }
        if other.inherits_from.is_some() {
            self.inherits_from = other.inherits_from.clone();
        }
        
        // Merge nested configurations
        self.default_synthesis.merge_with(&other.default_synthesis);
        self.model_loading.merge_with(&other.model_loading);
        self.audio_processing.merge_with(&other.audio_processing);
        self.logging.merge_with(&other.logging);
    }
    
    fn validate(&self) -> Result<(), ConfigValidationError> {
        // Validate device
        let valid_devices = ["cpu", "cuda", "metal", "vulkan", "opencl"];
        if !valid_devices.contains(&self.device.as_str()) {
            return Err(ConfigValidationError {
                field: "device".to_string(),
                message: format!("Invalid device '{}'. Must be one of: {}", 
                    self.device, valid_devices.join(", ")),
            });
        }
        
        // Validate thread count
        if let Some(threads) = self.num_threads {
            if threads == 0 || threads > 256 {
                return Err(ConfigValidationError {
                    field: "num_threads".to_string(),
                    message: "Thread count must be between 1 and 256".to_string(),
                });
            }
        }
        
        // Validate cache size
        if self.max_cache_size_mb > 100_000 {
            return Err(ConfigValidationError {
                field: "max_cache_size_mb".to_string(),
                message: "Cache size must be less than 100GB".to_string(),
            });
        }
        
        // Validate nested configurations
        self.default_synthesis.validate()?;
        self.model_loading.validate()?;
        self.audio_processing.validate()?;
        self.logging.validate()?;
        
        Ok(())
    }
}

impl PipelineConfig {
    /// Create new configuration with defaults
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Create configuration for a specific profile
    pub fn for_profile(profile: &str) -> Self {
        let mut config = Self::default();
        config.profile = Some(profile.to_string());
        
        match profile {
            "development" => {
                config.logging.level = "debug".to_string();
                config.logging.metrics = true;
                config.model_loading.auto_download = true;
            }
            "production" => {
                config.logging.level = "warn".to_string();
                config.logging.structured = true;
                config.model_loading.verify_checksums = true;
                config.model_loading.preload_models = true;
            }
            "testing" => {
                config.max_cache_size_mb = 256;
                config.logging.level = "error".to_string();
                config.model_loading.auto_download = false;
            }
            "high_performance" => {
                config.use_gpu = true;
                config.device = "cuda".to_string();
                config.audio_processing.buffer_size = 16384;
                config.default_synthesis.quality = QualityLevel::Ultra;
            }
            "low_latency" => {
                config.audio_processing.buffer_size = 2048;
                config.default_synthesis.quality = QualityLevel::Medium;
                config.model_loading.preload_models = true;
            }
            _ => {}
        }
        
        config
    }
    
    /// Get effective cache directory
    pub fn effective_cache_dir(&self) -> PathBuf {
        self.cache_dir
            .clone()
            .unwrap_or_else(|| std::env::temp_dir().join("voirs-cache"))
    }
    
    /// Get effective thread count
    pub fn effective_thread_count(&self) -> usize {
        self.num_threads
            .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4))
    }
    
    /// Check if GPU is available and enabled
    pub fn should_use_gpu(&self) -> bool {
        self.use_gpu && self.device != "cpu"
    }
}

/// Model loading configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelLoadingConfig {
    /// Timeout for model downloads in seconds
    pub download_timeout_secs: u64,
    
    /// Number of download retries
    pub download_retries: u32,
    
    /// Automatically download missing models
    pub auto_download: bool,
    
    /// Verify model checksums
    pub verify_checksums: bool,
    
    /// Model repositories to use
    pub repositories: Vec<String>,
    
    /// Preload models on startup
    pub preload_models: bool,
    
    /// Model-specific overrides
    pub model_overrides: HashMap<String, ModelOverride>,
}

impl Default for ModelLoadingConfig {
    fn default() -> Self {
        Self {
            download_timeout_secs: 300,
            download_retries: 3,
            auto_download: true,
            verify_checksums: true,
            repositories: vec![
                "https://huggingface.co/voirs/models".to_string(),
                "https://github.com/cool-japan/voirs-models".to_string(),
            ],
            preload_models: false,
            model_overrides: HashMap::new(),
        }
    }
}

impl ConfigHierarchy for ModelLoadingConfig {
    fn merge_with(&mut self, other: &Self) {
        if other.download_timeout_secs != 300 {
            self.download_timeout_secs = other.download_timeout_secs;
        }
        if other.download_retries != 3 {
            self.download_retries = other.download_retries;
        }
        if !other.auto_download {
            self.auto_download = other.auto_download;
        }
        if !other.verify_checksums {
            self.verify_checksums = other.verify_checksums;
        }
        if other.preload_models {
            self.preload_models = other.preload_models;
        }
        
        // Merge repositories (union)
        for repo in &other.repositories {
            if !self.repositories.contains(repo) {
                self.repositories.push(repo.clone());
            }
        }
        
        // Merge model overrides
        for (key, value) in &other.model_overrides {
            self.model_overrides.insert(key.clone(), value.clone());
        }
    }
    
    fn validate(&self) -> Result<(), ConfigValidationError> {
        if self.download_timeout_secs > 3600 {
            return Err(ConfigValidationError {
                field: "download_timeout_secs".to_string(),
                message: "Download timeout must be less than 1 hour".to_string(),
            });
        }
        
        if self.download_retries > 10 {
            return Err(ConfigValidationError {
                field: "download_retries".to_string(),
                message: "Download retries must be 10 or less".to_string(),
            });
        }
        
        if self.repositories.is_empty() {
            return Err(ConfigValidationError {
                field: "repositories".to_string(),
                message: "At least one repository must be configured".to_string(),
            });
        }
        
        Ok(())
    }
}

/// Model-specific configuration override
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelOverride {
    /// Custom download URL
    pub url: Option<String>,
    
    /// Expected checksum
    pub checksum: Option<String>,
    
    /// Local path override
    pub local_path: Option<PathBuf>,
    
    /// Priority for this model
    pub priority: Option<u32>,
}

/// Audio processing configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AudioProcessingConfig {
    /// Buffer size for audio processing
    pub buffer_size: usize,
    
    /// Enable audio enhancement by default
    pub enable_enhancement: bool,
    
    /// Normalization target level (0.0-1.0)
    pub normalization_level: f32,
    
    /// Apply noise reduction
    pub noise_reduction: bool,
    
    /// Dynamic range compression
    pub compression: bool,
    
    /// High-pass filter frequency (Hz)
    pub highpass_freq: Option<f32>,
    
    /// Low-pass filter frequency (Hz)
    pub lowpass_freq: Option<f32>,
    
    /// Effect chain configuration
    pub effects: Vec<EffectConfig>,
}

impl Default for AudioProcessingConfig {
    fn default() -> Self {
        Self {
            buffer_size: 8192,
            enable_enhancement: true,
            normalization_level: 0.95,
            noise_reduction: false,
            compression: false,
            highpass_freq: None,
            lowpass_freq: None,
            effects: Vec::new(),
        }
    }
}

impl ConfigHierarchy for AudioProcessingConfig {
    fn merge_with(&mut self, other: &Self) {
        if other.buffer_size != 8192 {
            self.buffer_size = other.buffer_size;
        }
        if !other.enable_enhancement {
            self.enable_enhancement = other.enable_enhancement;
        }
        if (other.normalization_level - 0.95).abs() > f32::EPSILON {
            self.normalization_level = other.normalization_level;
        }
        if other.noise_reduction {
            self.noise_reduction = other.noise_reduction;
        }
        if other.compression {
            self.compression = other.compression;
        }
        if other.highpass_freq.is_some() {
            self.highpass_freq = other.highpass_freq;
        }
        if other.lowpass_freq.is_some() {
            self.lowpass_freq = other.lowpass_freq;
        }
        
        // Merge effects (append, allowing duplicates for complex chains)
        self.effects.extend(other.effects.clone());
    }
    
    fn validate(&self) -> Result<(), ConfigValidationError> {
        if self.buffer_size < 512 || self.buffer_size > 65536 {
            return Err(ConfigValidationError {
                field: "buffer_size".to_string(),
                message: "Buffer size must be between 512 and 65536".to_string(),
            });
        }
        
        if !self.buffer_size.is_power_of_two() {
            return Err(ConfigValidationError {
                field: "buffer_size".to_string(),
                message: "Buffer size must be a power of 2".to_string(),
            });
        }
        
        if self.normalization_level <= 0.0 || self.normalization_level > 1.0 {
            return Err(ConfigValidationError {
                field: "normalization_level".to_string(),
                message: "Normalization level must be between 0.0 and 1.0".to_string(),
            });
        }
        
        if let Some(freq) = self.highpass_freq {
            if freq <= 0.0 || freq > 20000.0 {
                return Err(ConfigValidationError {
                    field: "highpass_freq".to_string(),
                    message: "High-pass frequency must be between 0 and 20000 Hz".to_string(),
                });
            }
        }
        
        if let Some(freq) = self.lowpass_freq {
            if freq <= 0.0 || freq > 20000.0 {
                return Err(ConfigValidationError {
                    field: "lowpass_freq".to_string(),
                    message: "Low-pass frequency must be between 0 and 20000 Hz".to_string(),
                });
            }
        }
        
        Ok(())
    }
}

/// Audio effect configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EffectConfig {
    /// Effect type
    pub effect_type: String,
    
    /// Effect parameters
    pub parameters: HashMap<String, f32>,
    
    /// Whether effect is enabled
    pub enabled: bool,
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LoggingConfig {
    /// Log level (trace, debug, info, warn, error)
    pub level: String,
    
    /// Enable structured logging (JSON format)
    pub structured: bool,
    
    /// Log to file
    pub file_path: Option<PathBuf>,
    
    /// Log file rotation size in MB
    pub max_file_size_mb: u32,
    
    /// Number of log files to keep
    pub max_files: u32,
    
    /// Enable performance metrics logging
    pub metrics: bool,
    
    /// Module-specific log levels
    pub module_levels: HashMap<String, String>,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            structured: false,
            file_path: None,
            max_file_size_mb: 10,
            max_files: 5,
            metrics: false,
            module_levels: HashMap::new(),
        }
    }
}

impl ConfigHierarchy for LoggingConfig {
    fn merge_with(&mut self, other: &Self) {
        if other.level != "info" {
            self.level = other.level.clone();
        }
        if other.structured {
            self.structured = other.structured;
        }
        if other.file_path.is_some() {
            self.file_path = other.file_path.clone();
        }
        if other.max_file_size_mb != 10 {
            self.max_file_size_mb = other.max_file_size_mb;
        }
        if other.max_files != 5 {
            self.max_files = other.max_files;
        }
        if other.metrics {
            self.metrics = other.metrics;
        }
        
        // Merge module levels
        for (module, level) in &other.module_levels {
            self.module_levels.insert(module.clone(), level.clone());
        }
    }
    
    fn validate(&self) -> Result<(), ConfigValidationError> {
        let valid_levels = ["trace", "debug", "info", "warn", "error", "off"];
        if !valid_levels.contains(&self.level.as_str()) {
            return Err(ConfigValidationError {
                field: "level".to_string(),
                message: format!("Invalid log level '{}'. Must be one of: {}", 
                    self.level, valid_levels.join(", ")),
            });
        }
        
        if self.max_file_size_mb == 0 || self.max_file_size_mb > 1000 {
            return Err(ConfigValidationError {
                field: "max_file_size_mb".to_string(),
                message: "Max file size must be between 1 and 1000 MB".to_string(),
            });
        }
        
        if self.max_files == 0 || self.max_files > 100 {
            return Err(ConfigValidationError {
                field: "max_files".to_string(),
                message: "Max files must be between 1 and 100".to_string(),
            });
        }
        
        // Validate module levels
        for (module, level) in &self.module_levels {
            if !valid_levels.contains(&level.as_str()) {
                return Err(ConfigValidationError {
                    field: format!("module_levels.{}", module),
                    message: format!("Invalid log level '{}' for module '{}'", level, module),
                });
            }
        }
        
        Ok(())
    }
}

/// Application-wide configuration with hierarchy support
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AppConfig {
    /// Pipeline configuration
    pub pipeline: PipelineConfig,
    
    /// CLI-specific configuration
    pub cli: CliConfig,
    
    /// Server configuration
    pub server: ServerConfig,
    
    /// Environment name
    pub environment: Option<String>,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            pipeline: PipelineConfig::default(),
            cli: CliConfig::default(),
            server: ServerConfig::default(),
            environment: None,
        }
    }
}

impl ConfigHierarchy for AppConfig {
    fn merge_with(&mut self, other: &Self) {
        self.pipeline.merge_with(&other.pipeline);
        self.cli.merge_with(&other.cli);
        self.server.merge_with(&other.server);
        
        if other.environment.is_some() {
            self.environment = other.environment.clone();
        }
    }
    
    fn validate(&self) -> Result<(), ConfigValidationError> {
        self.pipeline.validate()?;
        self.cli.validate()?;
        self.server.validate()?;
        Ok(())
    }
}

/// CLI configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CliConfig {
    /// Default output directory
    pub output_dir: Option<PathBuf>,
    
    /// Default voice to use
    pub default_voice: Option<String>,
    
    /// Default output format
    pub default_format: AudioFormat,
    
    /// Show progress bars
    pub show_progress: bool,
    
    /// Colored output
    pub colored_output: bool,
    
    /// Command aliases
    pub aliases: HashMap<String, String>,
}

impl Default for CliConfig {
    fn default() -> Self {
        Self {
            output_dir: None,
            default_voice: None,
            default_format: AudioFormat::Wav,
            show_progress: true,
            colored_output: true,
            aliases: HashMap::new(),
        }
    }
}

impl ConfigHierarchy for CliConfig {
    fn merge_with(&mut self, other: &Self) {
        if other.output_dir.is_some() {
            self.output_dir = other.output_dir.clone();
        }
        if other.default_voice.is_some() {
            self.default_voice = other.default_voice.clone();
        }
        if other.default_format != AudioFormat::Wav {
            self.default_format = other.default_format;
        }
        if !other.show_progress {
            self.show_progress = other.show_progress;
        }
        if !other.colored_output {
            self.colored_output = other.colored_output;
        }
        
        // Merge aliases
        for (alias, command) in &other.aliases {
            self.aliases.insert(alias.clone(), command.clone());
        }
    }
    
    fn validate(&self) -> Result<(), ConfigValidationError> {
        // Validate output directory exists if specified
        if let Some(ref path) = self.output_dir {
            if path.exists() && !path.is_dir() {
                return Err(ConfigValidationError {
                    field: "output_dir".to_string(),
                    message: "Output directory path exists but is not a directory".to_string(),
                });
            }
        }
        
        Ok(())
    }
}

/// Server configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ServerConfig {
    /// Server bind address
    pub bind_address: String,
    
    /// Server port
    pub port: u16,
    
    /// Maximum concurrent requests
    pub max_concurrent_requests: usize,
    
    /// Request timeout in seconds
    pub request_timeout_secs: u64,
    
    /// Enable CORS
    pub enable_cors: bool,
    
    /// API key for authentication
    pub api_key: Option<String>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind_address: "127.0.0.1".to_string(),
            port: 8080,
            max_concurrent_requests: 100,
            request_timeout_secs: 30,
            enable_cors: true,
            api_key: None,
        }
    }
}

impl ConfigHierarchy for ServerConfig {
    fn merge_with(&mut self, other: &Self) {
        if other.bind_address != "127.0.0.1" {
            self.bind_address = other.bind_address.clone();
        }
        if other.port != 8080 {
            self.port = other.port;
        }
        if other.max_concurrent_requests != 100 {
            self.max_concurrent_requests = other.max_concurrent_requests;
        }
        if other.request_timeout_secs != 30 {
            self.request_timeout_secs = other.request_timeout_secs;
        }
        if !other.enable_cors {
            self.enable_cors = other.enable_cors;
        }
        if other.api_key.is_some() {
            self.api_key = other.api_key.clone();
        }
    }
    
    fn validate(&self) -> Result<(), ConfigValidationError> {
        if self.port == 0 {
            return Err(ConfigValidationError {
                field: "port".to_string(),
                message: "Port must be greater than 0".to_string(),
            });
        }
        
        if self.max_concurrent_requests == 0 {
            return Err(ConfigValidationError {
                field: "max_concurrent_requests".to_string(),
                message: "Max concurrent requests must be greater than 0".to_string(),
            });
        }
        
        if self.request_timeout_secs == 0 || self.request_timeout_secs > 3600 {
            return Err(ConfigValidationError {
                field: "request_timeout_secs".to_string(),
                message: "Request timeout must be between 1 and 3600 seconds".to_string(),
            });
        }
        
        Ok(())
    }
}

/// Configuration resolver for handling inheritance and profiles
pub struct ConfigResolver {
    /// Registered configuration profiles
    profiles: HashMap<String, AppConfig>,
    
    /// Base configuration
    base_config: AppConfig,
}

impl ConfigResolver {
    /// Create new configuration resolver
    pub fn new() -> Self {
        Self {
            profiles: HashMap::new(),
            base_config: AppConfig::default(),
        }
    }
    
    /// Register a configuration profile
    pub fn register_profile(&mut self, name: &str, config: AppConfig) {
        self.profiles.insert(name.to_string(), config);
    }
    
    /// Set base configuration
    pub fn set_base_config(&mut self, config: AppConfig) {
        self.base_config = config;
    }
    
    /// Resolve configuration for a given profile with inheritance
    pub fn resolve(&self, profile: Option<&str>) -> Result<AppConfig, ConfigValidationError> {
        let mut resolved = self.base_config.clone();
        
        if let Some(profile_name) = profile {
            if let Some(profile_config) = self.profiles.get(profile_name) {
                let inherits_from = profile_config.pipeline.inherits_from.clone();
                resolved.merge_with(profile_config);
                
                // Handle pipeline profile inheritance
                if let Some(inherits_from) = inherits_from {
                    if let Some(parent_config) = self.profiles.get(&inherits_from) {
                        let mut pipeline_with_inheritance = parent_config.pipeline.clone();
                        pipeline_with_inheritance.merge_with(&resolved.pipeline);
                        resolved.pipeline = pipeline_with_inheritance;
                    }
                }
            }
        }
        
        resolved.validate()?;
        Ok(resolved)
    }
    
    /// Get available profiles
    pub fn available_profiles(&self) -> Vec<&String> {
        self.profiles.keys().collect()
    }
}

impl Default for ConfigResolver {
    fn default() -> Self {
        let mut resolver = Self::new();
        
        // Register standard profiles
        resolver.register_profile("development", AppConfig {
            pipeline: PipelineConfig::for_profile("development"),
            ..Default::default()
        });
        
        resolver.register_profile("production", AppConfig {
            pipeline: PipelineConfig::for_profile("production"),
            ..Default::default()
        });
        
        resolver.register_profile("testing", AppConfig {
            pipeline: PipelineConfig::for_profile("testing"),
            ..Default::default()
        });
        
        resolver
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_config_hierarchy_merge() {
        let mut base = PipelineConfig::default();
        let override_config = PipelineConfig {
            device: "cuda".to_string(),
            use_gpu: true,
            ..Default::default()
        };
        
        base.merge_with(&override_config);
        
        assert_eq!(base.device, "cuda");
        assert_eq!(base.use_gpu, true);
        assert_eq!(base.max_cache_size_mb, 1024); // Should remain default
    }
    
    #[test]
    fn test_config_validation() {
        let invalid_config = PipelineConfig {
            device: "invalid_device".to_string(),
            ..Default::default()
        };
        
        assert!(invalid_config.validate().is_err());
        
        let valid_config = PipelineConfig {
            device: "cuda".to_string(),
            ..Default::default()
        };
        
        assert!(valid_config.validate().is_ok());
    }
    
    #[test]
    fn test_profile_creation() {
        let dev_config = PipelineConfig::for_profile("development");
        assert_eq!(dev_config.logging.level, "debug");
        assert_eq!(dev_config.logging.metrics, true);
        
        let prod_config = PipelineConfig::for_profile("production");
        assert_eq!(prod_config.logging.level, "warn");
        assert_eq!(prod_config.logging.structured, true);
    }
    
    #[test]
    fn test_config_resolver() {
        let resolver = ConfigResolver::default();
        
        let dev_config = resolver.resolve(Some("development")).unwrap();
        assert_eq!(dev_config.pipeline.logging.level, "debug");
        
        let prod_config = resolver.resolve(Some("production")).unwrap();
        assert_eq!(prod_config.pipeline.logging.level, "warn");
    }
    
    #[test]
    fn test_audio_config_validation() {
        let invalid_buffer = AudioProcessingConfig {
            buffer_size: 1000, // Not power of 2
            ..Default::default()
        };
        assert!(invalid_buffer.validate().is_err());
        
        let invalid_normalization = AudioProcessingConfig {
            normalization_level: 1.5, // > 1.0
            ..Default::default()
        };
        assert!(invalid_normalization.validate().is_err());
        
        let valid_config = AudioProcessingConfig::default();
        assert!(valid_config.validate().is_ok());
    }
    
    #[test]
    fn test_logging_config_merge() {
        let mut base = LoggingConfig::default();
        let override_config = LoggingConfig {
            level: "debug".to_string(),
            module_levels: [("voirs".to_string(), "trace".to_string())].into_iter().collect(),
            ..Default::default()
        };
        
        base.merge_with(&override_config);
        
        assert_eq!(base.level, "debug");
        assert_eq!(base.module_levels.get("voirs").unwrap(), "trace");
    }
    
    #[test]
    fn test_model_overrides() {
        let mut config = ModelLoadingConfig::default();
        config.model_overrides.insert(
            "test-model".to_string(),
            ModelOverride {
                url: Some("https://example.com/model.bin".to_string()),
                checksum: Some("abc123".to_string()),
                local_path: None,
                priority: Some(1),
            },
        );
        
        let other = ModelLoadingConfig {
            model_overrides: [(
                "other-model".to_string(),
                ModelOverride {
                    url: Some("https://example.com/other.bin".to_string()),
                    checksum: None,
                    local_path: None,
                    priority: Some(2),
                },
            )].into_iter().collect(),
            ..Default::default()
        };
        
        config.merge_with(&other);
        
        assert_eq!(config.model_overrides.len(), 2);
        assert!(config.model_overrides.contains_key("test-model"));
        assert!(config.model_overrides.contains_key("other-model"));
    }
}
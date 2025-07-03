//! Configuration management for VoiRS.
//!
//! This module provides a comprehensive configuration system with support for:
//! - Hierarchical configuration with inheritance
//! - Multiple file formats (JSON, TOML, YAML)
//! - Environment variable overrides
//! - Runtime configuration updates
//! - Configuration validation and migration
//!
//! # Quick Start
//!
//! ```no_run
//! use voirs_sdk::config::{ConfigBuilder, ConfigLoader, presets};
//!
//! // Create configuration using builder pattern
//! let config = ConfigBuilder::new()
//!     .device("cuda")
//!     .gpu(true)
//!     .threads(8)
//!     .build();
//!
//! // Load configuration from files and environment
//! let config = ConfigLoader::new().load().unwrap();
//!
//! // Use preset configurations
//! let dev_config = presets::development();
//! let prod_config = presets::production();
//! ```
//!
//! # Migration from Old API
//!
//! The configuration system has been refactored into a modular structure.
//! All existing APIs are still available for backward compatibility:
//!
//! ```no_run
//! use voirs_sdk::config::PipelineConfig;
//!
//! // Old API still works
//! let config = PipelineConfig::new()
//!     .with_device("cuda")
//!     .with_gpu(true);
//! ```
//!
//! # Configuration Hierarchy
//!
//! Configurations support inheritance and merging:
//!
//! ```no_run
//! use voirs_sdk::config::{PipelineConfig, ConfigHierarchy};
//!
//! let base_config = PipelineConfig::default();
//! let mut dev_config = PipelineConfig::for_profile("development");
//!
//! // Merge configurations
//! dev_config.merge_with(&base_config);
//! ```
//!
//! # File Persistence
//!
//! Multiple configuration formats are supported:
//!
//! ```no_run
//! use voirs_sdk::config::{ConfigPersistence, TemplateType};
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Load from file
//! let config = ConfigPersistence::load_from_file("config.toml")?;
//!
//! // Save to file
//! ConfigPersistence::save_to_file(&config, "config.json")?;
//!
//! // Create template
//! ConfigPersistence::create_template(TemplateType::Production, "prod-config.toml")?;
//! # Ok(())
//! # }
//! ```

pub mod hierarchy;
pub mod persistence;
pub mod dynamic;

// Re-export main types for convenience
pub use hierarchy::{
    AppConfig, PipelineConfig, ModelLoadingConfig, AudioProcessingConfig, 
    LoggingConfig, CliConfig, ServerConfig, ConfigHierarchy, ConfigResolver,
    ConfigValidationError, ModelOverride, EffectConfig,
};

pub use persistence::{
    ConfigLoader, ConfigWatcher, ConfigPersistence,
    ConfigFormat, TemplateType, ConfigLoadError, ConfigSaveError,
};

pub use dynamic::{
    DynamicConfigManager, ConfigChangeEvent, ConfigChangeType,
    ConfigValidator, ConfigMigrator, ConfigUpdateError, ConfigManagement,
    ResourceValidator, DeviceValidator, ConfigHistoryEntry,
};

/// Configuration builder for fluent API construction
pub struct ConfigBuilder {
    config: PipelineConfig,
}

impl ConfigBuilder {
    /// Create new configuration builder
    pub fn new() -> Self {
        Self {
            config: PipelineConfig::default(),
        }
    }
    
    /// Set device for computation
    pub fn device(mut self, device: impl Into<String>) -> Self {
        self.config.device = device.into();
        self
    }
    
    /// Enable or disable GPU acceleration
    pub fn gpu(mut self, enabled: bool) -> Self {
        self.config.use_gpu = enabled;
        self
    }
    
    /// Set number of CPU threads
    pub fn threads(mut self, count: usize) -> Self {
        self.config.num_threads = Some(count);
        self
    }
    
    /// Set cache directory
    pub fn cache_dir(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.config.cache_dir = Some(path.into());
        self
    }
    
    /// Set maximum cache size in MB
    pub fn cache_size_mb(mut self, size: u32) -> Self {
        self.config.max_cache_size_mb = size;
        self
    }
    
    /// Set default quality level
    pub fn quality(mut self, quality: crate::types::QualityLevel) -> Self {
        self.config.default_synthesis.quality = quality;
        self
    }
    
    /// Configure model loading
    pub fn model_loading<F>(mut self, f: F) -> Self 
    where
        F: FnOnce(ModelLoadingConfigBuilder) -> ModelLoadingConfigBuilder,
    {
        let builder = ModelLoadingConfigBuilder::new(self.config.model_loading);
        self.config.model_loading = f(builder).build();
        self
    }
    
    /// Configure audio processing
    pub fn audio_processing<F>(mut self, f: F) -> Self 
    where
        F: FnOnce(AudioProcessingConfigBuilder) -> AudioProcessingConfigBuilder,
    {
        let builder = AudioProcessingConfigBuilder::new(self.config.audio_processing);
        self.config.audio_processing = f(builder).build();
        self
    }
    
    /// Configure logging
    pub fn logging<F>(mut self, f: F) -> Self 
    where
        F: FnOnce(LoggingConfigBuilder) -> LoggingConfigBuilder,
    {
        let builder = LoggingConfigBuilder::new(self.config.logging);
        self.config.logging = f(builder).build();
        self
    }
    
    /// Set configuration profile
    pub fn profile(mut self, profile: impl Into<String>) -> Self {
        self.config.profile = Some(profile.into());
        self
    }
    
    /// Build the configuration
    pub fn build(self) -> PipelineConfig {
        self.config
    }
    
    /// Validate and build the configuration
    pub fn build_validated(self) -> Result<PipelineConfig, ConfigValidationError> {
        let config = self.config;
        config.validate()?;
        Ok(config)
    }
}

impl Default for ConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for model loading configuration
pub struct ModelLoadingConfigBuilder {
    config: ModelLoadingConfig,
}

impl ModelLoadingConfigBuilder {
    fn new(config: ModelLoadingConfig) -> Self {
        Self { config }
    }
    
    /// Set download timeout
    pub fn download_timeout_secs(mut self, timeout: u64) -> Self {
        self.config.download_timeout_secs = timeout;
        self
    }
    
    /// Set download retries
    pub fn download_retries(mut self, retries: u32) -> Self {
        self.config.download_retries = retries;
        self
    }
    
    /// Enable or disable auto download
    pub fn auto_download(mut self, enabled: bool) -> Self {
        self.config.auto_download = enabled;
        self
    }
    
    /// Enable or disable checksum verification
    pub fn verify_checksums(mut self, enabled: bool) -> Self {
        self.config.verify_checksums = enabled;
        self
    }
    
    /// Add repository
    pub fn add_repository(mut self, repo: impl Into<String>) -> Self {
        self.config.repositories.push(repo.into());
        self
    }
    
    /// Set repositories
    pub fn repositories(mut self, repos: Vec<String>) -> Self {
        self.config.repositories = repos;
        self
    }
    
    /// Enable or disable model preloading
    pub fn preload_models(mut self, enabled: bool) -> Self {
        self.config.preload_models = enabled;
        self
    }
    
    /// Add model override
    pub fn add_model_override(
        mut self, 
        model_name: impl Into<String>, 
        override_config: ModelOverride
    ) -> Self {
        self.config.model_overrides.insert(model_name.into(), override_config);
        self
    }
    
    fn build(self) -> ModelLoadingConfig {
        self.config
    }
}

/// Builder for audio processing configuration
pub struct AudioProcessingConfigBuilder {
    config: AudioProcessingConfig,
}

impl AudioProcessingConfigBuilder {
    fn new(config: AudioProcessingConfig) -> Self {
        Self { config }
    }
    
    /// Set buffer size
    pub fn buffer_size(mut self, size: usize) -> Self {
        self.config.buffer_size = size;
        self
    }
    
    /// Enable or disable audio enhancement
    pub fn enhancement(mut self, enabled: bool) -> Self {
        self.config.enable_enhancement = enabled;
        self
    }
    
    /// Set normalization level
    pub fn normalization_level(mut self, level: f32) -> Self {
        self.config.normalization_level = level;
        self
    }
    
    /// Enable or disable noise reduction
    pub fn noise_reduction(mut self, enabled: bool) -> Self {
        self.config.noise_reduction = enabled;
        self
    }
    
    /// Enable or disable compression
    pub fn compression(mut self, enabled: bool) -> Self {
        self.config.compression = enabled;
        self
    }
    
    /// Set high-pass filter frequency
    pub fn highpass_freq(mut self, freq: f32) -> Self {
        self.config.highpass_freq = Some(freq);
        self
    }
    
    /// Set low-pass filter frequency
    pub fn lowpass_freq(mut self, freq: f32) -> Self {
        self.config.lowpass_freq = Some(freq);
        self
    }
    
    /// Add effect
    pub fn add_effect(mut self, effect: EffectConfig) -> Self {
        self.config.effects.push(effect);
        self
    }
    
    fn build(self) -> AudioProcessingConfig {
        self.config
    }
}

/// Builder for logging configuration
pub struct LoggingConfigBuilder {
    config: LoggingConfig,
}

impl LoggingConfigBuilder {
    fn new(config: LoggingConfig) -> Self {
        Self { config }
    }
    
    /// Set log level
    pub fn level(mut self, level: impl Into<String>) -> Self {
        self.config.level = level.into();
        self
    }
    
    /// Enable or disable structured logging
    pub fn structured(mut self, enabled: bool) -> Self {
        self.config.structured = enabled;
        self
    }
    
    /// Set log file path
    pub fn file_path(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.config.file_path = Some(path.into());
        self
    }
    
    /// Set maximum file size
    pub fn max_file_size_mb(mut self, size: u32) -> Self {
        self.config.max_file_size_mb = size;
        self
    }
    
    /// Set maximum number of files
    pub fn max_files(mut self, count: u32) -> Self {
        self.config.max_files = count;
        self
    }
    
    /// Enable or disable metrics logging
    pub fn metrics(mut self, enabled: bool) -> Self {
        self.config.metrics = enabled;
        self
    }
    
    /// Set module log level
    pub fn module_level(mut self, module: impl Into<String>, level: impl Into<String>) -> Self {
        self.config.module_levels.insert(module.into(), level.into());
        self
    }
    
    fn build(self) -> LoggingConfig {
        self.config
    }
}

/// Application configuration builder
pub struct AppConfigBuilder {
    config: AppConfig,
}

impl AppConfigBuilder {
    /// Create new application config builder
    pub fn new() -> Self {
        Self {
            config: AppConfig::default(),
        }
    }
    
    /// Configure pipeline
    pub fn pipeline<F>(mut self, f: F) -> Self 
    where
        F: FnOnce(ConfigBuilder) -> ConfigBuilder,
    {
        let builder = ConfigBuilder::new();
        self.config.pipeline = f(builder).build();
        self
    }
    
    /// Configure CLI
    pub fn cli<F>(mut self, f: F) -> Self 
    where
        F: FnOnce(CliConfigBuilder) -> CliConfigBuilder,
    {
        let builder = CliConfigBuilder::new(self.config.cli);
        self.config.cli = f(builder).build();
        self
    }
    
    /// Configure server
    pub fn server<F>(mut self, f: F) -> Self 
    where
        F: FnOnce(ServerConfigBuilder) -> ServerConfigBuilder,
    {
        let builder = ServerConfigBuilder::new(self.config.server);
        self.config.server = f(builder).build();
        self
    }
    
    /// Set environment
    pub fn environment(mut self, env: impl Into<String>) -> Self {
        self.config.environment = Some(env.into());
        self
    }
    
    /// Build the configuration
    pub fn build(self) -> AppConfig {
        self.config
    }
    
    /// Validate and build the configuration
    pub fn build_validated(self) -> Result<AppConfig, ConfigValidationError> {
        let config = self.config;
        config.validate()?;
        Ok(config)
    }
}

impl Default for AppConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for CLI configuration
pub struct CliConfigBuilder {
    config: CliConfig,
}

impl CliConfigBuilder {
    fn new(config: CliConfig) -> Self {
        Self { config }
    }
    
    /// Set output directory
    pub fn output_dir(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.config.output_dir = Some(path.into());
        self
    }
    
    /// Set default voice
    pub fn default_voice(mut self, voice: impl Into<String>) -> Self {
        self.config.default_voice = Some(voice.into());
        self
    }
    
    /// Set default format
    pub fn default_format(mut self, format: crate::types::AudioFormat) -> Self {
        self.config.default_format = format;
        self
    }
    
    /// Enable or disable progress bars
    pub fn show_progress(mut self, enabled: bool) -> Self {
        self.config.show_progress = enabled;
        self
    }
    
    /// Enable or disable colored output
    pub fn colored_output(mut self, enabled: bool) -> Self {
        self.config.colored_output = enabled;
        self
    }
    
    /// Add command alias
    pub fn add_alias(mut self, alias: impl Into<String>, command: impl Into<String>) -> Self {
        self.config.aliases.insert(alias.into(), command.into());
        self
    }
    
    fn build(self) -> CliConfig {
        self.config
    }
}

/// Builder for server configuration
pub struct ServerConfigBuilder {
    config: ServerConfig,
}

impl ServerConfigBuilder {
    fn new(config: ServerConfig) -> Self {
        Self { config }
    }
    
    /// Set bind address
    pub fn bind_address(mut self, address: impl Into<String>) -> Self {
        self.config.bind_address = address.into();
        self
    }
    
    /// Set port
    pub fn port(mut self, port: u16) -> Self {
        self.config.port = port;
        self
    }
    
    /// Set maximum concurrent requests
    pub fn max_concurrent_requests(mut self, count: usize) -> Self {
        self.config.max_concurrent_requests = count;
        self
    }
    
    /// Set request timeout
    pub fn request_timeout_secs(mut self, timeout: u64) -> Self {
        self.config.request_timeout_secs = timeout;
        self
    }
    
    /// Enable or disable CORS
    pub fn enable_cors(mut self, enabled: bool) -> Self {
        self.config.enable_cors = enabled;
        self
    }
    
    /// Set API key
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.config.api_key = Some(key.into());
        self
    }
    
    fn build(self) -> ServerConfig {
        self.config
    }
}

/// Extension traits for enhanced configuration functionality
pub trait PipelineConfigExt {
    /// Create optimized configuration for GPU usage
    fn for_gpu(device: &str) -> PipelineConfig;
    
    /// Create optimized configuration for CPU usage
    fn for_cpu(threads: Option<usize>) -> PipelineConfig;
    
    /// Create minimal configuration for testing
    fn for_testing() -> PipelineConfig;
}

impl PipelineConfigExt for PipelineConfig {
    fn for_gpu(device: &str) -> PipelineConfig {
        ConfigBuilder::new()
            .device(device)
            .gpu(true)
            .quality(crate::types::QualityLevel::High)
            .audio_processing(|audio| {
                audio.buffer_size(16384)
                     .enhancement(true)
            })
            .build()
    }
    
    fn for_cpu(threads: Option<usize>) -> PipelineConfig {
        let mut builder = ConfigBuilder::new()
            .device("cpu")
            .gpu(false)
            .quality(crate::types::QualityLevel::Medium);
        
        if let Some(thread_count) = threads {
            builder = builder.threads(thread_count);
        }
        
        builder.build()
    }
    
    fn for_testing() -> PipelineConfig {
        ConfigBuilder::new()
            .device("cpu")
            .gpu(false)
            .cache_size_mb(256)
            .quality(crate::types::QualityLevel::Low)
            .model_loading(|model| {
                model.auto_download(false)
                     .preload_models(false)
            })
            .logging(|log| {
                log.level("error")
                   .structured(false)
                   .metrics(false)
            })
            .build()
    }
}

/// Convenience functions for quick configuration setup
pub mod presets {
    use super::*;
    
    /// Create development configuration preset
    pub fn development() -> AppConfig {
        AppConfigBuilder::new()
            .pipeline(|p| p
                .device("cpu")
                .quality(crate::types::QualityLevel::Medium)
                .logging(|log| log
                    .level("debug")
                    .metrics(true)
                    .structured(false)
                )
                .model_loading(|model| model
                    .auto_download(true)
                    .verify_checksums(false)
                )
            )
            .cli(|cli| cli
                .show_progress(true)
                .colored_output(true)
            )
            .environment("development")
            .build()
    }
    
    /// Create production configuration preset
    pub fn production() -> AppConfig {
        AppConfigBuilder::new()
            .pipeline(|p| p
                .device("cpu")
                .quality(crate::types::QualityLevel::High)
                .logging(|log| log
                    .level("warn")
                    .metrics(true)
                    .structured(true)
                )
                .model_loading(|model| model
                    .auto_download(false)
                    .verify_checksums(true)
                    .preload_models(true)
                )
            )
            .server(|server| server
                .bind_address("0.0.0.0")
                .port(8080)
                .max_concurrent_requests(100)
                .enable_cors(false)
            )
            .environment("production")
            .build()
    }
    
    /// Create high-performance configuration preset
    pub fn high_performance() -> AppConfig {
        AppConfigBuilder::new()
            .pipeline(|p| p
                .device("cuda")
                .gpu(true)
                .quality(crate::types::QualityLevel::Ultra)
                .audio_processing(|audio| audio
                    .buffer_size(16384)
                    .enhancement(true)
                    .compression(false)
                )
                .model_loading(|model| model
                    .preload_models(true)
                    .verify_checksums(true)
                )
            )
            .environment("high_performance")
            .build()
    }
    
    /// Create low-latency configuration preset
    pub fn low_latency() -> AppConfig {
        AppConfigBuilder::new()
            .pipeline(|p| p
                .device("cpu")
                .quality(crate::types::QualityLevel::Medium)
                .audio_processing(|audio| audio
                    .buffer_size(2048)
                    .enhancement(false)
                )
                .model_loading(|model| model
                    .preload_models(true)
                )
            )
            .environment("low_latency")
            .build()
    }
    
    /// Create testing configuration preset
    pub fn testing() -> AppConfig {
        AppConfigBuilder::new()
            .pipeline(|p| p
                .device("cpu")
                .cache_size_mb(256)
                .quality(crate::types::QualityLevel::Low)
                .logging(|log| log
                    .level("error")
                    .metrics(false)
                )
                .model_loading(|model| model
                    .auto_download(false)
                    .preload_models(false)
                )
            )
            .cli(|cli| cli
                .show_progress(false)
                .colored_output(false)
            )
            .environment("testing")
            .build()
    }
}

// Maintain backward compatibility for the old PipelineConfig API
impl PipelineConfig {
    
    /// Set device for computation (legacy API)
    pub fn with_device(mut self, device: impl Into<String>) -> Self {
        self.device = device.into();
        self
    }
    
    /// Enable or disable GPU acceleration (legacy API)
    pub fn with_gpu(mut self, use_gpu: bool) -> Self {
        self.use_gpu = use_gpu;
        self
    }
    
    /// Set number of CPU threads (legacy API)
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.num_threads = Some(threads);
        self
    }
    
    /// Set cache directory (legacy API)
    pub fn with_cache_dir(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.cache_dir = Some(path.into());
        self
    }
    
    /// Set default quality level (legacy API)
    pub fn with_quality(mut self, quality: crate::types::QualityLevel) -> Self {
        self.default_synthesis.quality = quality;
        self
    }
    
    /// Load configuration from file (legacy API)
    pub fn from_file(path: impl AsRef<std::path::Path>) -> crate::Result<Self> {
        use crate::config::persistence::{ConfigLoader, ConfigLoadError};
        
        ConfigLoader::new()
            .load_from_file(path.as_ref())
            .map(|app_config| app_config.pipeline)
            .map_err(|e| match e {
                ConfigLoadError::Io { error, .. } => crate::VoirsError::from(error),
                ConfigLoadError::Parse { error, .. } => {
                    crate::VoirsError::config_error(format!("Failed to parse config file: {}", error))
                }
                ConfigLoadError::UnsupportedFormat { .. } => {
                    crate::VoirsError::config_error("Unsupported config file format".to_string())
                }
                ConfigLoadError::Validation(validation_error) => {
                    crate::VoirsError::config_error(validation_error.to_string())
                }
                ConfigLoadError::EnvVar { error, .. } => {
                    crate::VoirsError::config_error(format!("Environment variable error: {}", error))
                }
            })
    }
    
    /// Save configuration to file (legacy API)
    pub fn save_to_file(&self, path: impl AsRef<std::path::Path>) -> crate::Result<()> {
        use crate::config::persistence::{ConfigLoader, ConfigSaveError};
        
        let app_config = AppConfig {
            pipeline: self.clone(),
            ..Default::default()
        };
        
        ConfigLoader::new()
            .save_to_file(&app_config, path.as_ref())
            .map_err(|e| match e {
                ConfigSaveError::Io { error, .. } => crate::VoirsError::from(error),
                ConfigSaveError::Serialize { error, .. } => {
                    crate::VoirsError::config_error(format!("Failed to serialize config: {}", error))
                }
                ConfigSaveError::UnsupportedFormat { .. } => {
                    crate::VoirsError::config_error("Unsupported config file format".to_string())
                }
            })
    }
}

// Legacy ConfigBuilder for backward compatibility
#[deprecated(since = "0.2.0", note = "Use ConfigBuilder from the new config system")]
pub struct LegacyConfigBuilder {
    config: PipelineConfig,
}

#[allow(deprecated)]
impl LegacyConfigBuilder {
    /// Create new configuration builder
    pub fn new() -> Self {
        Self {
            config: PipelineConfig::default(),
        }
    }
    
    /// Set device
    pub fn device(mut self, device: impl Into<String>) -> Self {
        self.config.device = device.into();
        self
    }
    
    /// Enable GPU
    pub fn gpu(mut self, enabled: bool) -> Self {
        self.config.use_gpu = enabled;
        self
    }
    
    /// Set thread count
    pub fn threads(mut self, count: usize) -> Self {
        self.config.num_threads = Some(count);
        self
    }
    
    /// Set cache directory
    pub fn cache_dir(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.config.cache_dir = Some(path.into());
        self
    }
    
    /// Set quality level
    pub fn quality(mut self, quality: crate::types::QualityLevel) -> Self {
        self.config.default_synthesis.quality = quality;
        self
    }
    
    /// Build the configuration
    pub fn build(self) -> PipelineConfig {
        self.config
    }
}

#[allow(deprecated)]
impl Default for LegacyConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_legacy_config_creation() {
        let config = PipelineConfig::new();
        assert_eq!(config.device, "cpu");
        assert!(!config.use_gpu);
    }

    #[test]
    fn test_legacy_config_fluent_api() {
        let config = PipelineConfig::new()
            .with_device("cuda")
            .with_gpu(true)
            .with_threads(8)
            .with_quality(crate::types::QualityLevel::Ultra);
        
        assert_eq!(config.device, "cuda");
        assert!(config.use_gpu);
        assert_eq!(config.num_threads, Some(8));
        assert_eq!(config.default_synthesis.quality, crate::types::QualityLevel::Ultra);
    }

    #[test]
    fn test_legacy_file_operations() {
        let config = PipelineConfig::default();
        let temp_dir = tempdir().unwrap();
        
        // Test JSON file
        let json_path = temp_dir.path().join("config.json");
        config.save_to_file(&json_path).unwrap();
        let loaded = PipelineConfig::from_file(&json_path).unwrap();
        assert_eq!(config.device, loaded.device);
        
        // Test TOML file
        let toml_path = temp_dir.path().join("config.toml");
        config.save_to_file(&toml_path).unwrap();
        let loaded = PipelineConfig::from_file(&toml_path).unwrap();
        assert_eq!(config.device, loaded.device);
    }
    
    #[test]
    fn test_config_builder() {
        let config = ConfigBuilder::new()
            .device("cuda")
            .gpu(true)
            .threads(8)
            .cache_size_mb(2048)
            .quality(crate::types::QualityLevel::Ultra)
            .build();
        
        assert_eq!(config.device, "cuda");
        assert_eq!(config.use_gpu, true);
        assert_eq!(config.num_threads, Some(8));
        assert_eq!(config.max_cache_size_mb, 2048);
        assert_eq!(config.default_synthesis.quality, crate::types::QualityLevel::Ultra);
    }
    
    #[test]
    fn test_app_config_builder() {
        let config = AppConfigBuilder::new()
            .pipeline(|p| p
                .device("cuda")
                .gpu(true)
            )
            .cli(|cli| cli
                .show_progress(false)
                .colored_output(false)
            )
            .server(|server| server
                .port(9090)
                .bind_address("0.0.0.0")
            )
            .environment("test")
            .build();
        
        assert_eq!(config.pipeline.device, "cuda");
        assert_eq!(config.pipeline.use_gpu, true);
        assert_eq!(config.cli.show_progress, false);
        assert_eq!(config.server.port, 9090);
        assert_eq!(config.environment, Some("test".to_string()));
    }
    
    #[test]
    fn test_nested_builder() {
        let config = ConfigBuilder::new()
            .model_loading(|model| model
                .auto_download(false)
                .download_retries(5)
                .add_repository("https://example.com/models")
            )
            .audio_processing(|audio| audio
                .buffer_size(4096)
                .enhancement(false)
                .highpass_freq(80.0)
            )
            .logging(|log| log
                .level("debug")
                .structured(true)
                .module_level("voirs", "trace")
            )
            .build();
        
        assert_eq!(config.model_loading.auto_download, false);
        assert_eq!(config.model_loading.download_retries, 5);
        assert!(config.model_loading.repositories.contains(&"https://example.com/models".to_string()));
        
        assert_eq!(config.audio_processing.buffer_size, 4096);
        assert_eq!(config.audio_processing.enable_enhancement, false);
        assert_eq!(config.audio_processing.highpass_freq, Some(80.0));
        
        assert_eq!(config.logging.level, "debug");
        assert_eq!(config.logging.structured, true);
        assert_eq!(config.logging.module_levels.get("voirs"), Some(&"trace".to_string()));
    }
    
    #[test]
    fn test_pipeline_config_extensions() {
        let gpu_config = PipelineConfig::for_gpu("cuda");
        assert_eq!(gpu_config.device, "cuda");
        assert_eq!(gpu_config.use_gpu, true);
        assert_eq!(gpu_config.default_synthesis.quality, crate::types::QualityLevel::High);
        
        let cpu_config = PipelineConfig::for_cpu(Some(16));
        assert_eq!(cpu_config.device, "cpu");
        assert_eq!(cpu_config.use_gpu, false);
        assert_eq!(cpu_config.num_threads, Some(16));
        
        let test_config = PipelineConfig::for_testing();
        assert_eq!(test_config.device, "cpu");
        assert_eq!(test_config.max_cache_size_mb, 256);
        assert_eq!(test_config.default_synthesis.quality, crate::types::QualityLevel::Low);
    }
    
    #[test]
    fn test_preset_configurations() {
        let dev_config = presets::development();
        assert_eq!(dev_config.pipeline.logging.level, "debug");
        assert_eq!(dev_config.environment, Some("development".to_string()));
        
        let prod_config = presets::production();
        assert_eq!(prod_config.pipeline.logging.level, "warn");
        assert_eq!(prod_config.pipeline.logging.structured, true);
        assert_eq!(prod_config.server.bind_address, "0.0.0.0");
        
        let hp_config = presets::high_performance();
        assert_eq!(hp_config.pipeline.device, "cuda");
        assert_eq!(hp_config.pipeline.use_gpu, true);
        assert_eq!(hp_config.pipeline.default_synthesis.quality, crate::types::QualityLevel::Ultra);
        
        let ll_config = presets::low_latency();
        assert_eq!(ll_config.pipeline.audio_processing.buffer_size, 2048);
        assert_eq!(ll_config.pipeline.audio_processing.enable_enhancement, false);
        
        let test_config = presets::testing();
        assert_eq!(test_config.pipeline.max_cache_size_mb, 256);
        assert_eq!(test_config.cli.show_progress, false);
    }
    
    #[test]
    fn test_config_validation() {
        let invalid_config = ConfigBuilder::new()
            .device("invalid_device")
            .build();
        
        assert!(invalid_config.validate().is_err());
        
        let valid_config = ConfigBuilder::new()
            .device("cuda")
            .gpu(true)
            .build();
        
        assert!(valid_config.validate().is_ok());
    }
    
    #[test]
    fn test_validated_build() {
        let result = ConfigBuilder::new()
            .device("invalid_device")
            .build_validated();
        
        assert!(result.is_err());
        
        let result = ConfigBuilder::new()
            .device("cuda")
            .build_validated();
        
        assert!(result.is_ok());
    }
}
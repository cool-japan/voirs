//! Configuration persistence and file operations.
//!
//! This module handles loading and saving configurations from various sources:
//!
//! - File-based configurations (JSON, TOML, YAML)
//! - Environment variable overrides
//! - Configuration search and discovery
//! - Hot-reloading and file watching
//! - Configuration validation and error handling

use super::hierarchy::{AppConfig, PipelineConfig, ConfigValidationError, ConfigHierarchy};
use crate::VoirsError;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    env,
    fs,
    path::{Path, PathBuf},
    time::SystemTime,
};

/// Configuration file format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigFormat {
    /// JSON format
    Json,
    /// TOML format
    Toml,
    /// YAML format
    Yaml,
}

impl ConfigFormat {
    /// Detect format from file extension
    pub fn from_extension(path: &Path) -> Option<Self> {
        match path.extension()?.to_str()? {
            "json" => Some(ConfigFormat::Json),
            "toml" => Some(ConfigFormat::Toml),
            "yaml" | "yml" => Some(ConfigFormat::Yaml),
            _ => None,
        }
    }
    
    /// Get file extension for this format
    pub fn extension(&self) -> &'static str {
        match self {
            ConfigFormat::Json => "json",
            ConfigFormat::Toml => "toml",
            ConfigFormat::Yaml => "yaml",
        }
    }
    
    /// Get MIME type for this format
    pub fn mime_type(&self) -> &'static str {
        match self {
            ConfigFormat::Json => "application/json",
            ConfigFormat::Toml => "application/toml",
            ConfigFormat::Yaml => "application/yaml",
        }
    }
}

/// Configuration loader for various sources
pub struct ConfigLoader {
    /// Search paths for configuration files
    search_paths: Vec<PathBuf>,
    
    /// Environment variable prefix
    env_prefix: String,
    
    /// Default configuration name
    config_name: String,
    
    /// Supported formats (in order of preference)
    supported_formats: Vec<ConfigFormat>,
}

impl ConfigLoader {
    /// Create new configuration loader
    pub fn new() -> Self {
        Self {
            search_paths: Self::default_search_paths(),
            env_prefix: "VOIRS".to_string(),
            config_name: "voirs".to_string(),
            supported_formats: vec![ConfigFormat::Toml, ConfigFormat::Json, ConfigFormat::Yaml],
        }
    }
    
    /// Set search paths for configuration files
    pub fn with_search_paths(mut self, paths: Vec<PathBuf>) -> Self {
        self.search_paths = paths;
        self
    }
    
    /// Add search path
    pub fn add_search_path(mut self, path: PathBuf) -> Self {
        self.search_paths.push(path);
        self
    }
    
    /// Set environment variable prefix
    pub fn with_env_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.env_prefix = prefix.into();
        self
    }
    
    /// Set configuration name
    pub fn with_config_name(mut self, name: impl Into<String>) -> Self {
        self.config_name = name.into();
        self
    }
    
    /// Load configuration from discovered files and environment
    pub fn load(&self) -> Result<AppConfig, ConfigLoadError> {
        let mut config = AppConfig::default();
        
        // Load from configuration files
        for config_file in self.discover_config_files()? {
            let file_config = self.load_from_file(&config_file)?;
            config.merge_with(&file_config);
        }
        
        // Apply environment variable overrides
        self.apply_env_overrides(&mut config)?;
        
        // Validate final configuration
        config.validate().map_err(ConfigLoadError::Validation)?;
        
        Ok(config)
    }
    
    /// Load configuration from specific file
    pub fn load_from_file(&self, path: &Path) -> Result<AppConfig, ConfigLoadError> {
        let content = fs::read_to_string(path)
            .map_err(|e| ConfigLoadError::Io {
                path: path.to_path_buf(),
                error: e,
            })?;
        
        let format = ConfigFormat::from_extension(path)
            .ok_or_else(|| ConfigLoadError::UnsupportedFormat {
                path: path.to_path_buf(),
            })?;
        
        self.parse_config(&content, format)
            .map_err(|e| ConfigLoadError::Parse {
                path: path.to_path_buf(),
                format,
                error: e,
            })
    }
    
    /// Save configuration to file
    pub fn save_to_file(&self, config: &AppConfig, path: &Path) -> Result<(), ConfigSaveError> {
        let format = ConfigFormat::from_extension(path)
            .ok_or_else(|| ConfigSaveError::UnsupportedFormat {
                path: path.to_path_buf(),
            })?;
        
        let content = self.serialize_config(config, format)
            .map_err(|e| ConfigSaveError::Serialize {
                path: path.to_path_buf(),
                format,
                error: e,
            })?;
        
        // Create parent directories if they don't exist
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| ConfigSaveError::Io {
                    path: path.to_path_buf(),
                    error: e,
                })?;
        }
        
        fs::write(path, content)
            .map_err(|e| ConfigSaveError::Io {
                path: path.to_path_buf(),
                error: e,
            })
    }
    
    /// Discover configuration files in search paths
    pub fn discover_config_files(&self) -> Result<Vec<PathBuf>, ConfigLoadError> {
        let mut found_files = Vec::new();
        
        for search_path in &self.search_paths {
            if !search_path.exists() {
                continue;
            }
            
            for format in &self.supported_formats {
                let config_file = search_path.join(format!("{}.{}", self.config_name, format.extension()));
                if config_file.exists() {
                    found_files.push(config_file);
                }
            }
        }
        
        Ok(found_files)
    }
    
    /// Get default search paths
    fn default_search_paths() -> Vec<PathBuf> {
        let mut paths = Vec::new();
        
        // Current directory
        if let Ok(current_dir) = env::current_dir() {
            paths.push(current_dir);
        }
        
        paths
    }
    
    /// Parse configuration content
    fn parse_config(&self, content: &str, format: ConfigFormat) -> Result<AppConfig, String> {
        match format {
            ConfigFormat::Json => {
                serde_json::from_str(content)
                    .map_err(|e| format!("JSON parse error: {}", e))
            }
            ConfigFormat::Toml => {
                toml::from_str(content)
                    .map_err(|e| format!("TOML parse error: {}", e))
            }
            ConfigFormat::Yaml => {
                Err("YAML support not implemented".to_string())
            }
        }
    }
    
    /// Serialize configuration to string
    fn serialize_config(&self, config: &AppConfig, format: ConfigFormat) -> Result<String, String> {
        match format {
            ConfigFormat::Json => {
                serde_json::to_string_pretty(config)
                    .map_err(|e| format!("JSON serialize error: {}", e))
            }
            ConfigFormat::Toml => {
                toml::to_string_pretty(config)
                    .map_err(|e| format!("TOML serialize error: {}", e))
            }
            ConfigFormat::Yaml => {
                Err("YAML support not implemented".to_string())
            }
        }
    }
    
    /// Apply environment variable overrides
    fn apply_env_overrides(&self, config: &mut AppConfig) -> Result<(), ConfigLoadError> {
        // Apply environment variable overrides
        if let Ok(device) = env::var(format!("{}_DEVICE", self.env_prefix)) {
            config.pipeline.device = device;
        }
        
        if let Ok(gpu) = env::var(format!("{}_USE_GPU", self.env_prefix)) {
            config.pipeline.use_gpu = gpu.parse().unwrap_or(false);
        }
        
        if let Ok(threads) = env::var(format!("{}_THREADS", self.env_prefix)) {
            if let Ok(thread_count) = threads.parse::<usize>() {
                config.pipeline.num_threads = Some(thread_count);
            }
        }
        
        Ok(())
    }
}

impl Default for ConfigLoader {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration loading error
#[derive(Debug, thiserror::Error)]
pub enum ConfigLoadError {
    /// I/O error reading configuration file
    #[error("I/O error reading config file {path}: {error}")]
    Io { path: PathBuf, error: std::io::Error },
    
    /// Configuration parsing error
    #[error("Error parsing config file {path} as {format:?}: {error}")]
    Parse {
        path: PathBuf,
        format: ConfigFormat,
        error: String,
    },
    
    /// Unsupported file format
    #[error("Unsupported config file format for {path}")]
    UnsupportedFormat { path: PathBuf },
    
    /// Configuration validation error
    #[error("Configuration validation error: {0}")]
    Validation(#[from] ConfigValidationError),
    
    /// Environment variable error
    #[error("Environment variable error for {var}: {error}")]
    EnvVar { var: String, error: String },
}

/// Configuration saving error
#[derive(Debug, thiserror::Error)]
pub enum ConfigSaveError {
    /// I/O error writing configuration file
    #[error("I/O error writing config file {path}: {error}")]
    Io { path: PathBuf, error: std::io::Error },
    
    /// Configuration serialization error
    #[error("Error serializing config to {path} as {format:?}: {error}")]
    Serialize {
        path: PathBuf,
        format: ConfigFormat,
        error: String,
    },
    
    /// Unsupported file format
    #[error("Unsupported config file format for {path}")]
    UnsupportedFormat { path: PathBuf },
}

/// Configuration file watcher (placeholder)
pub struct ConfigWatcher {
    path: PathBuf,
    loader: ConfigLoader,
}

impl ConfigWatcher {
    pub fn new(path: PathBuf, loader: ConfigLoader) -> Self {
        Self { path, loader }
    }
    
    pub fn reload_if_changed(&mut self) -> Result<Option<AppConfig>, ConfigLoadError> {
        // Placeholder implementation
        Ok(None)
    }
}

/// Configuration template types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemplateType {
    Development,
    Production,
    Testing,
}

/// Convenience functions for common operations
pub struct ConfigPersistence;

impl ConfigPersistence {
    /// Load configuration with default settings
    pub fn load() -> Result<AppConfig, ConfigLoadError> {
        ConfigLoader::new().load()
    }
    
    /// Load configuration from specific file
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<AppConfig, ConfigLoadError> {
        ConfigLoader::new().load_from_file(path.as_ref())
    }
    
    /// Save configuration to file
    pub fn save_to_file<P: AsRef<Path>>(config: &AppConfig, path: P) -> Result<(), ConfigSaveError> {
        ConfigLoader::new().save_to_file(config, path.as_ref())
    }
    
    /// Create configuration template
    pub fn create_template<P: AsRef<Path>>(template_type: TemplateType, path: P) -> Result<(), ConfigSaveError> {
        let config = match template_type {
            TemplateType::Development => crate::config::presets::development(),
            TemplateType::Production => crate::config::presets::production(),
            TemplateType::Testing => crate::config::presets::testing(),
        };
        Self::save_to_file(&config, path)
    }
}
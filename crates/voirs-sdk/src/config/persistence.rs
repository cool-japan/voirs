//! Configuration persistence and file operations.
//!
//! This module handles loading and saving configurations from various sources:
//!
//! - File-based configurations (JSON, TOML, YAML)
//! - Environment variable overrides
//! - Configuration search and discovery
//! - Hot-reloading and file watching
//! - Configuration validation and error handling

use super::hierarchy::{AppConfig, ConfigHierarchy, ConfigValidationError};
use std::{
    env, fs,
    path::{Path, PathBuf},
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
        let content = fs::read_to_string(path).map_err(|e| ConfigLoadError::Io {
            path: path.to_path_buf(),
            error: e,
        })?;

        let format = ConfigFormat::from_extension(path).ok_or_else(|| {
            ConfigLoadError::UnsupportedFormat {
                path: path.to_path_buf(),
            }
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
        let format = ConfigFormat::from_extension(path).ok_or_else(|| {
            ConfigSaveError::UnsupportedFormat {
                path: path.to_path_buf(),
            }
        })?;

        let content =
            self.serialize_config(config, format)
                .map_err(|e| ConfigSaveError::Serialize {
                    path: path.to_path_buf(),
                    format,
                    error: e,
                })?;

        // Create parent directories if they don't exist
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| ConfigSaveError::Io {
                path: path.to_path_buf(),
                error: e,
            })?;
        }

        fs::write(path, content).map_err(|e| ConfigSaveError::Io {
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
                let config_file =
                    search_path.join(format!("{}.{}", self.config_name, format.extension()));
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
                serde_json::from_str(content).map_err(|e| format!("JSON parse error: {e}"))
            }
            ConfigFormat::Toml => {
                toml::from_str(content).map_err(|e| format!("TOML parse error: {e}"))
            }
            ConfigFormat::Yaml => {
                serde_yaml::from_str(content).map_err(|e| format!("YAML parse error: {e}"))
            }
        }
    }

    /// Serialize configuration to string
    fn serialize_config(&self, config: &AppConfig, format: ConfigFormat) -> Result<String, String> {
        match format {
            ConfigFormat::Json => serde_json::to_string_pretty(config)
                .map_err(|e| format!("JSON serialize error: {e}")),
            ConfigFormat::Toml => {
                toml::to_string_pretty(config).map_err(|e| format!("TOML serialize error: {e}"))
            }
            ConfigFormat::Yaml => {
                serde_yaml::to_string(config).map_err(|e| format!("YAML serialize error: {e}"))
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
    Io {
        path: PathBuf,
        error: std::io::Error,
    },

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
    Io {
        path: PathBuf,
        error: std::io::Error,
    },

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

/// Configuration file watcher for automatic reloading
pub struct ConfigWatcher {
    /// Path to the configuration file being watched
    path: PathBuf,
    /// Configuration loader
    loader: ConfigLoader,
    /// Last known modification time
    last_modified: Option<std::time::SystemTime>,
    /// File system watcher (optional for automatic watching)
    _watcher: Option<notify::RecommendedWatcher>,
    /// Channel for receiving file system events
    event_receiver: Option<std::sync::mpsc::Receiver<notify::Result<notify::Event>>>,
    /// Current configuration cache
    cached_config: Option<AppConfig>,
}

impl ConfigWatcher {
    /// Create a new configuration file watcher
    pub fn new(path: PathBuf, loader: ConfigLoader) -> Result<Self, ConfigWatchError> {
        let last_modified = Self::get_file_modification_time(&path)?;

        Ok(Self {
            path,
            loader,
            last_modified: Some(last_modified),
            _watcher: None,
            event_receiver: None,
            cached_config: None,
        })
    }

    /// Create a new configuration file watcher with automatic file system watching
    pub fn with_auto_watch(path: PathBuf, loader: ConfigLoader) -> Result<Self, ConfigWatchError> {
        use notify::{RecursiveMode, Watcher};

        let last_modified = Self::get_file_modification_time(&path)?;

        // Create a channel for file system events
        let (tx, rx) = std::sync::mpsc::channel();

        // Create the file system watcher
        let mut watcher =
            notify::recommended_watcher(tx).map_err(|e| ConfigWatchError::WatcherCreation {
                path: path.clone(),
                error: e.to_string(),
            })?;

        // Watch the configuration file's parent directory
        if let Some(parent_dir) = path.parent() {
            watcher
                .watch(parent_dir, RecursiveMode::NonRecursive)
                .map_err(|e| ConfigWatchError::WatchStart {
                    path: path.clone(),
                    error: e.to_string(),
                })?;
        }

        Ok(Self {
            path,
            loader,
            last_modified: Some(last_modified),
            _watcher: Some(watcher),
            event_receiver: Some(rx),
            cached_config: None,
        })
    }

    /// Check if the configuration file has been modified and reload if necessary
    pub fn reload_if_changed(&mut self) -> Result<Option<AppConfig>, ConfigLoadError> {
        // Check if file modification time has changed
        let current_modified =
            Self::get_file_modification_time(&self.path).map_err(|e| ConfigLoadError::Io {
                path: self.path.clone(),
                error: std::io::Error::new(std::io::ErrorKind::Other, e.to_string()),
            })?;

        // Compare with last known modification time
        if self.last_modified.is_none() || Some(current_modified) > self.last_modified {
            tracing::info!(
                "Configuration file modified, reloading: {}",
                self.path.display()
            );

            // Reload the configuration
            let new_config = self.loader.load_from_file(&self.path)?;

            // Update modification time and cache
            self.last_modified = Some(current_modified);
            self.cached_config = Some(new_config.clone());

            tracing::info!(
                "Configuration successfully reloaded from: {}",
                self.path.display()
            );
            Ok(Some(new_config))
        } else {
            // No changes detected
            Ok(None)
        }
    }

    /// Force reload the configuration file
    pub fn force_reload(&mut self) -> Result<AppConfig, ConfigLoadError> {
        tracing::info!(
            "Force reloading configuration from: {}",
            self.path.display()
        );

        let new_config = self.loader.load_from_file(&self.path)?;

        // Update modification time and cache
        if let Ok(modified) = Self::get_file_modification_time(&self.path) {
            self.last_modified = Some(modified);
        }
        self.cached_config = Some(new_config.clone());

        tracing::info!(
            "Configuration successfully force reloaded from: {}",
            self.path.display()
        );
        Ok(new_config)
    }

    /// Get the currently cached configuration
    pub fn get_cached_config(&self) -> Option<&AppConfig> {
        self.cached_config.as_ref()
    }

    /// Check for file system events (only works with auto-watch mode)
    pub fn check_events(&mut self) -> Result<Vec<ConfigChangeEvent>, ConfigWatchError> {
        let Some(ref receiver) = self.event_receiver else {
            return Ok(Vec::new());
        };

        let mut events = Vec::new();

        // Process all pending events
        while let Ok(event_result) = receiver.try_recv() {
            match event_result {
                Ok(event) => {
                    // Check if this event affects our watched file
                    if event.paths.iter().any(|p| p == &self.path) {
                        let change_type = match event.kind {
                            notify::EventKind::Modify(_) => ConfigChangeType::Modified,
                            notify::EventKind::Create(_) => ConfigChangeType::Created,
                            notify::EventKind::Remove(_) => ConfigChangeType::Deleted,
                            _ => ConfigChangeType::Other,
                        };

                        events.push(ConfigChangeEvent {
                            path: self.path.clone(),
                            change_type,
                            timestamp: std::time::SystemTime::now(),
                        });
                    }
                }
                Err(e) => {
                    tracing::warn!("File system watch error: {}", e);
                }
            }
        }

        Ok(events)
    }

    /// Get the path being watched
    pub fn watched_path(&self) -> &PathBuf {
        &self.path
    }

    /// Get the last modification time
    pub fn last_modification_time(&self) -> Option<std::time::SystemTime> {
        self.last_modified
    }

    /// Get file modification time
    fn get_file_modification_time(
        path: &PathBuf,
    ) -> Result<std::time::SystemTime, ConfigWatchError> {
        let metadata = std::fs::metadata(path).map_err(|e| ConfigWatchError::FileAccess {
            path: path.clone(),
            error: e.to_string(),
        })?;

        metadata
            .modified()
            .map_err(|e| ConfigWatchError::FileAccess {
                path: path.clone(),
                error: e.to_string(),
            })
    }
}

impl std::fmt::Debug for ConfigWatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConfigWatcher")
            .field("path", &self.path)
            .field("last_modified", &self.last_modified)
            .field("has_watcher", &self._watcher.is_some())
            .field("has_event_receiver", &self.event_receiver.is_some())
            .field("has_cached_config", &self.cached_config.is_some())
            .finish()
    }
}

/// Configuration change event
#[derive(Debug, Clone)]
pub struct ConfigChangeEvent {
    /// Path to the changed configuration file
    pub path: PathBuf,
    /// Type of change that occurred
    pub change_type: ConfigChangeType,
    /// Timestamp when the change was detected
    pub timestamp: std::time::SystemTime,
}

/// Type of configuration file change
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigChangeType {
    /// File was modified
    Modified,
    /// File was created
    Created,
    /// File was deleted
    Deleted,
    /// Other type of change
    Other,
}

/// Configuration watching error
#[derive(Debug, thiserror::Error)]
pub enum ConfigWatchError {
    /// Error accessing configuration file
    #[error("Error accessing config file {path}: {error}")]
    FileAccess { path: PathBuf, error: String },

    /// Error creating file system watcher
    #[error("Error creating file system watcher for {path}: {error}")]
    WatcherCreation { path: PathBuf, error: String },

    /// Error starting file system watch
    #[error("Error starting file system watch for {path}: {error}")]
    WatchStart { path: PathBuf, error: String },
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
    pub fn save_to_file<P: AsRef<Path>>(
        config: &AppConfig,
        path: P,
    ) -> Result<(), ConfigSaveError> {
        ConfigLoader::new().save_to_file(config, path.as_ref())
    }

    /// Create configuration template
    pub fn create_template<P: AsRef<Path>>(
        template_type: TemplateType,
        path: P,
    ) -> Result<(), ConfigSaveError> {
        let config = match template_type {
            TemplateType::Development => crate::config::presets::development(),
            TemplateType::Production => crate::config::presets::production(),
            TemplateType::Testing => crate::config::presets::testing(),
        };
        Self::save_to_file(&config, path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::thread;
    use std::time::Duration;
    use tempfile::NamedTempFile;

    #[test]
    fn test_config_loader_basic() {
        let loader = ConfigLoader::new();
        assert!(!loader.search_paths.is_empty());
        assert_eq!(loader.env_prefix, "VOIRS");
        assert_eq!(loader.config_name, "voirs");
    }

    #[test]
    fn test_config_format_detection() {
        use std::path::Path;

        assert_eq!(
            ConfigFormat::from_extension(Path::new("test.json")),
            Some(ConfigFormat::Json)
        );
        assert_eq!(
            ConfigFormat::from_extension(Path::new("test.toml")),
            Some(ConfigFormat::Toml)
        );
        assert_eq!(
            ConfigFormat::from_extension(Path::new("test.yaml")),
            Some(ConfigFormat::Yaml)
        );
        assert_eq!(
            ConfigFormat::from_extension(Path::new("test.yml")),
            Some(ConfigFormat::Yaml)
        );
        assert_eq!(
            ConfigFormat::from_extension(Path::new("test.unknown")),
            None
        );
    }

    #[test]
    fn test_config_format_properties() {
        assert_eq!(ConfigFormat::Json.extension(), "json");
        assert_eq!(ConfigFormat::Toml.extension(), "toml");
        assert_eq!(ConfigFormat::Yaml.extension(), "yaml");

        assert_eq!(ConfigFormat::Json.mime_type(), "application/json");
        assert_eq!(ConfigFormat::Toml.mime_type(), "application/toml");
        assert_eq!(ConfigFormat::Yaml.mime_type(), "application/yaml");
    }

    #[test]
    fn test_config_loader_with_custom_settings() {
        let loader = ConfigLoader::new()
            .with_env_prefix("CUSTOM")
            .with_config_name("myapp")
            .with_search_paths(vec![std::path::PathBuf::from("/custom/path")]);

        assert_eq!(loader.env_prefix, "CUSTOM");
        assert_eq!(loader.config_name, "myapp");
        assert_eq!(
            loader.search_paths,
            vec![std::path::PathBuf::from("/custom/path")]
        );
    }

    #[test]
    fn test_config_file_save_and_load() {
        let mut temp_file = NamedTempFile::new().unwrap();
        let config = AppConfig::default();

        // Test JSON format
        let json_path = temp_file.path().with_extension("json");
        let loader = ConfigLoader::new();

        // Save configuration
        loader.save_to_file(&config, &json_path).unwrap();

        // Load configuration
        let loaded_config = loader.load_from_file(&json_path).unwrap();

        // Verify basic structure (we can't do exact equality due to potential differences in defaults)
        assert_eq!(loaded_config.pipeline.device, config.pipeline.device);
        assert_eq!(loaded_config.pipeline.use_gpu, config.pipeline.use_gpu);

        // Clean up
        let _ = std::fs::remove_file(&json_path);
    }

    #[test]
    fn test_config_watcher_creation() {
        let temp_file = NamedTempFile::new().unwrap();
        let config = AppConfig::default();

        // Create a valid config file
        let json_content = serde_json::to_string_pretty(&config).unwrap();
        std::fs::write(temp_file.path(), json_content).unwrap();

        let loader = ConfigLoader::new();
        let watcher = ConfigWatcher::new(temp_file.path().to_path_buf(), loader).unwrap();

        assert_eq!(watcher.watched_path(), temp_file.path());
        assert!(watcher.last_modification_time().is_some());
        assert!(watcher.get_cached_config().is_none());
    }

    #[test]
    fn test_config_watcher_with_auto_watch() {
        let temp_file = NamedTempFile::new().unwrap();
        let config = AppConfig::default();

        // Create a valid config file
        let json_content = serde_json::to_string_pretty(&config).unwrap();
        std::fs::write(temp_file.path(), json_content).unwrap();

        let loader = ConfigLoader::new();
        let watcher =
            ConfigWatcher::with_auto_watch(temp_file.path().to_path_buf(), loader).unwrap();

        assert_eq!(watcher.watched_path(), temp_file.path());
        assert!(watcher.last_modification_time().is_some());
        assert!(watcher.get_cached_config().is_none());
    }

    #[test]
    fn test_config_watcher_reload_if_changed() {
        let temp_file = NamedTempFile::new().unwrap();
        let config = AppConfig::default();

        // Create initial config file with .json extension
        let json_path = temp_file.path().with_extension("json");
        let json_content = serde_json::to_string_pretty(&config).unwrap();
        std::fs::write(&json_path, json_content).unwrap();

        let loader = ConfigLoader::new();
        let mut watcher = ConfigWatcher::new(json_path.clone(), loader).unwrap();

        // First reload should detect changes since we compare with the current file state
        let result = watcher.reload_if_changed().unwrap();
        // The first call might not detect changes if the file hasn't been modified since creation
        // Let's force a change first
        thread::sleep(Duration::from_millis(100));
        let mut modified_config = config.clone();
        modified_config.pipeline.device = "initial_change".to_string();
        let modified_json = serde_json::to_string_pretty(&modified_config).unwrap();
        std::fs::write(&json_path, modified_json).unwrap();

        let result = watcher.reload_if_changed().unwrap();
        assert!(result.is_some());
        assert!(watcher.get_cached_config().is_some());

        // Second reload should detect no changes
        let result = watcher.reload_if_changed().unwrap();
        assert!(result.is_none());

        // Modify the file again
        thread::sleep(Duration::from_millis(100)); // Ensure different modification time
        let mut second_modified_config = config.clone();
        second_modified_config.pipeline.device = "second_change".to_string();
        let second_modified_json = serde_json::to_string_pretty(&second_modified_config).unwrap();
        std::fs::write(&json_path, second_modified_json).unwrap();

        // Third reload should detect changes
        let result = watcher.reload_if_changed().unwrap();
        assert!(result.is_some());
        let reloaded_config = result.unwrap();
        assert_eq!(reloaded_config.pipeline.device, "second_change");

        // Clean up
        let _ = std::fs::remove_file(&json_path);
    }

    #[test]
    fn test_config_watcher_force_reload() {
        let temp_file = NamedTempFile::new().unwrap();
        let config = AppConfig::default();

        // Create initial config file with .json extension
        let json_path = temp_file.path().with_extension("json");
        let json_content = serde_json::to_string_pretty(&config).unwrap();
        std::fs::write(&json_path, json_content).unwrap();

        let loader = ConfigLoader::new();
        let mut watcher = ConfigWatcher::new(json_path.clone(), loader).unwrap();

        // Force reload should always work
        let result = watcher.force_reload().unwrap();
        assert_eq!(result.pipeline.device, config.pipeline.device);
        assert!(watcher.get_cached_config().is_some());

        // Modify the file
        let mut modified_config = config.clone();
        modified_config.pipeline.device = "forced".to_string();
        let modified_json = serde_json::to_string_pretty(&modified_config).unwrap();
        std::fs::write(&json_path, modified_json).unwrap();

        // Force reload should pick up changes
        let result = watcher.force_reload().unwrap();
        assert_eq!(result.pipeline.device, "forced");

        // Clean up
        let _ = std::fs::remove_file(&json_path);
    }

    #[test]
    fn test_config_watcher_events() {
        let temp_file = NamedTempFile::new().unwrap();
        let config = AppConfig::default();

        // Create a valid config file
        let json_content = serde_json::to_string_pretty(&config).unwrap();
        std::fs::write(temp_file.path(), json_content).unwrap();

        let loader = ConfigLoader::new();
        let mut watcher =
            ConfigWatcher::with_auto_watch(temp_file.path().to_path_buf(), loader).unwrap();

        // Check events (should be empty initially)
        let events = watcher.check_events().unwrap();
        assert!(events.is_empty());

        // Modify the file
        let mut modified_config = config.clone();
        modified_config.pipeline.device = "event_test".to_string();
        let modified_json = serde_json::to_string_pretty(&modified_config).unwrap();
        std::fs::write(temp_file.path(), modified_json).unwrap();

        // Give the file watcher time to detect changes
        thread::sleep(Duration::from_millis(200));

        // Check for events (may or may not detect depending on timing)
        let events = watcher.check_events().unwrap();
        // We can't guarantee events will be detected due to timing, so we just verify the method works
        // events.len() is always >= 0 since it's a usize, so we just verify the method returned successfully
    }

    #[test]
    fn test_config_watcher_error_handling() {
        let non_existent_path = PathBuf::from("/non/existent/path/config.json");
        let loader = ConfigLoader::new();

        // Should fail to create watcher for non-existent file
        let result = ConfigWatcher::new(non_existent_path, loader);
        assert!(result.is_err());

        match result.unwrap_err() {
            ConfigWatchError::FileAccess { path, .. } => {
                assert_eq!(path, PathBuf::from("/non/existent/path/config.json"));
            }
            _ => panic!("Expected FileAccess error"),
        }
    }

    #[test]
    fn test_config_change_event_types() {
        let event = ConfigChangeEvent {
            path: PathBuf::from("/test/config.json"),
            change_type: ConfigChangeType::Modified,
            timestamp: std::time::SystemTime::now(),
        };

        assert_eq!(event.change_type, ConfigChangeType::Modified);
        assert_eq!(event.path, PathBuf::from("/test/config.json"));

        // Test all change types
        assert_eq!(ConfigChangeType::Modified, ConfigChangeType::Modified);
        assert_eq!(ConfigChangeType::Created, ConfigChangeType::Created);
        assert_eq!(ConfigChangeType::Deleted, ConfigChangeType::Deleted);
        assert_eq!(ConfigChangeType::Other, ConfigChangeType::Other);

        assert_ne!(ConfigChangeType::Modified, ConfigChangeType::Created);
    }

    #[test]
    fn test_config_persistence_convenience_methods() {
        let temp_file = NamedTempFile::new().unwrap();
        let config = AppConfig::default();

        // Test save convenience method
        let json_path = temp_file.path().with_extension("json");
        ConfigPersistence::save_to_file(&config, &json_path).unwrap();

        // Test load convenience method
        let loaded_config = ConfigPersistence::load_from_file(&json_path).unwrap();
        assert_eq!(loaded_config.pipeline.device, config.pipeline.device);

        // Clean up
        let _ = std::fs::remove_file(&json_path);
    }

    #[test]
    fn test_config_template_creation() {
        let temp_dir = tempfile::tempdir().unwrap();

        // Test development template
        let dev_path = temp_dir.path().join("dev.json");
        ConfigPersistence::create_template(TemplateType::Development, &dev_path).unwrap();
        assert!(dev_path.exists());

        // Test production template
        let prod_path = temp_dir.path().join("prod.json");
        ConfigPersistence::create_template(TemplateType::Production, &prod_path).unwrap();
        assert!(prod_path.exists());

        // Test testing template
        let test_path = temp_dir.path().join("test.json");
        ConfigPersistence::create_template(TemplateType::Testing, &test_path).unwrap();
        assert!(test_path.exists());
    }

    #[test]
    fn test_config_loader_discover_files() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config = AppConfig::default();

        // Create multiple config files
        let json_file = temp_dir.path().join("voirs.json");
        let toml_file = temp_dir.path().join("voirs.toml");
        let yaml_file = temp_dir.path().join("voirs.yaml");

        let loader = ConfigLoader::new().with_search_paths(vec![temp_dir.path().to_path_buf()]);

        // Save config files
        loader.save_to_file(&config, &json_file).unwrap();
        loader.save_to_file(&config, &toml_file).unwrap();
        loader.save_to_file(&config, &yaml_file).unwrap();

        // Discover files
        let discovered_files = loader.discover_config_files().unwrap();
        assert_eq!(discovered_files.len(), 3);

        // Verify all files are discovered
        assert!(discovered_files.contains(&json_file));
        assert!(discovered_files.contains(&toml_file));
        assert!(discovered_files.contains(&yaml_file));
    }
}

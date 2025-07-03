//! CLI-specific configuration utilities.

pub mod profiles;

use std::path::{Path, PathBuf};
use std::env;
use std::fs;
use serde::{Deserialize, Serialize};
use voirs::config::AppConfig;
use crate::error::{CliError, CliResult};

/// CLI-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliConfig {
    /// Core VoiRS configuration
    #[serde(flatten)]
    pub core: AppConfig,
    
    /// CLI-specific settings
    pub cli: CliSettings,
}

/// Alias for compatibility with interactive modules
pub type Config = CliConfig;

/// CLI-specific settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliSettings {
    /// Default output format
    pub default_output_format: String,
    
    /// Default voice
    pub default_voice: Option<String>,
    
    /// Default quality level
    pub default_quality: String,
    
    /// Enable colored output
    pub colored_output: bool,
    
    /// Show progress bars
    pub show_progress: bool,
    
    /// Auto-play synthesized audio
    pub auto_play: bool,
    
    /// Preferred output directory
    pub output_directory: Option<PathBuf>,
    
    /// SSML validation level
    pub ssml_validation: SsmlValidationLevel,
    
    /// Recent files history size
    pub history_size: usize,
    
    /// Voice download preferences
    pub download: DownloadSettings,
}

/// SSML validation levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SsmlValidationLevel {
    /// No validation
    None,
    /// Warn on issues
    Warn,
    /// Error on issues
    Strict,
}

/// Download settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadSettings {
    /// Parallel downloads
    pub parallel_downloads: usize,
    
    /// Retry attempts
    pub retry_attempts: usize,
    
    /// Auto-verify checksums
    pub verify_checksums: bool,
    
    /// Preferred download mirrors
    pub preferred_mirrors: Vec<String>,
}

impl Default for CliConfig {
    fn default() -> Self {
        Self {
            core: AppConfig::default(),
            cli: CliSettings::default(),
        }
    }
}

impl Default for CliSettings {
    fn default() -> Self {
        Self {
            default_output_format: "wav".to_string(),
            default_voice: None,
            default_quality: "high".to_string(),
            colored_output: true,
            show_progress: true,
            auto_play: false,
            output_directory: None,
            ssml_validation: SsmlValidationLevel::Warn,
            history_size: 100,
            download: DownloadSettings::default(),
        }
    }
}

impl Default for DownloadSettings {
    fn default() -> Self {
        Self {
            parallel_downloads: 3,
            retry_attempts: 3,
            verify_checksums: true,
            preferred_mirrors: vec![
                "https://huggingface.co".to_string(),
                "https://github.com".to_string(),
            ],
        }
    }
}

/// Configuration manager for the CLI
pub struct ConfigManager {
    config_path: PathBuf,
    config: CliConfig,
}

impl ConfigManager {
    /// Create a new configuration manager
    pub fn new() -> CliResult<Self> {
        let config_path = Self::find_config_file()
            .unwrap_or_else(|| Self::default_config_path());
        
        let config = if config_path.exists() {
            Self::load_from_file(&config_path)?
        } else {
            CliConfig::default()
        };
        
        Ok(Self { config_path, config })
    }
    
    /// Create configuration manager with specific path
    pub fn with_path<P: AsRef<Path>>(path: P) -> CliResult<Self> {
        let config_path = path.as_ref().to_path_buf();
        
        let config = if config_path.exists() {
            Self::load_from_file(&config_path)?
        } else {
            CliConfig::default()
        };
        
        Ok(Self { config_path, config })
    }
    
    /// Get the current configuration
    pub fn config(&self) -> &CliConfig {
        &self.config
    }
    
    /// Get mutable reference to configuration
    pub fn config_mut(&mut self) -> &mut CliConfig {
        &mut self.config
    }
    
    /// Save configuration to file
    pub fn save(&self) -> CliResult<()> {
        // Create parent directory if it doesn't exist
        if let Some(parent) = self.config_path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| CliError::file_operation("create directory", &parent.display().to_string(), e))?;
        }
        
        let content = toml::to_string_pretty(&self.config)
            .map_err(CliError::from)?;
        
        fs::write(&self.config_path, content)
            .map_err(|e| CliError::file_operation("write", &self.config_path.display().to_string(), e))?;
        
        Ok(())
    }
    
    /// Update configuration value
    pub fn set_value(&mut self, key: &str, value: &str) -> CliResult<()> {
        match key {
            "default_output_format" => {
                self.config.cli.default_output_format = value.to_string();
            }
            "default_voice" => {
                self.config.cli.default_voice = if value.is_empty() { 
                    None 
                } else { 
                    Some(value.to_string()) 
                };
            }
            "default_quality" => {
                if ["low", "medium", "high", "ultra"].contains(&value) {
                    self.config.cli.default_quality = value.to_string();
                } else {
                    return Err(CliError::invalid_parameter(key, "must be one of: low, medium, high, ultra"));
                }
            }
            "colored_output" => {
                self.config.cli.colored_output = value.parse()
                    .map_err(|_| CliError::invalid_parameter(key, "must be true or false"))?;
            }
            "show_progress" => {
                self.config.cli.show_progress = value.parse()
                    .map_err(|_| CliError::invalid_parameter(key, "must be true or false"))?;
            }
            "auto_play" => {
                self.config.cli.auto_play = value.parse()
                    .map_err(|_| CliError::invalid_parameter(key, "must be true or false"))?;
            }
            "output_directory" => {
                self.config.cli.output_directory = if value.is_empty() { 
                    None 
                } else { 
                    Some(PathBuf::from(value)) 
                };
            }
            _ => {
                return Err(CliError::invalid_parameter(key, "unknown configuration key"));
            }
        }
        
        Ok(())
    }
    
    /// Get configuration value as string
    pub fn get_value(&self, key: &str) -> Option<String> {
        match key {
            "default_output_format" => Some(self.config.cli.default_output_format.clone()),
            "default_voice" => self.config.cli.default_voice.clone(),
            "default_quality" => Some(self.config.cli.default_quality.clone()),
            "colored_output" => Some(self.config.cli.colored_output.to_string()),
            "show_progress" => Some(self.config.cli.show_progress.to_string()),
            "auto_play" => Some(self.config.cli.auto_play.to_string()),
            "output_directory" => self.config.cli.output_directory.as_ref().map(|p| p.display().to_string()),
            _ => None,
        }
    }
    
    /// Apply environment variable overrides
    pub fn apply_env_overrides(&mut self) {
        if let Ok(format) = env::var("VOIRS_OUTPUT_FORMAT") {
            self.config.cli.default_output_format = format;
        }
        
        if let Ok(voice) = env::var("VOIRS_DEFAULT_VOICE") {
            self.config.cli.default_voice = Some(voice);
        }
        
        if let Ok(quality) = env::var("VOIRS_QUALITY") {
            if ["low", "medium", "high", "ultra"].contains(&quality.as_str()) {
                self.config.cli.default_quality = quality;
            }
        }
        
        if let Ok(colored) = env::var("VOIRS_COLORED_OUTPUT") {
            if let Ok(value) = colored.parse() {
                self.config.cli.colored_output = value;
            }
        }
        
        if let Ok(progress) = env::var("VOIRS_SHOW_PROGRESS") {
            if let Ok(value) = progress.parse() {
                self.config.cli.show_progress = value;
            }
        }
        
        if let Ok(output_dir) = env::var("VOIRS_OUTPUT_DIR") {
            self.config.cli.output_directory = Some(PathBuf::from(output_dir));
        }
    }
    
    /// Validate configuration
    pub fn validate(&self) -> CliResult<Vec<String>> {
        let mut warnings = Vec::new();
        
        // Check if default voice exists
        if let Some(ref voice) = self.config.cli.default_voice {
            // This would require voice list lookup, skip for now
            warnings.push(format!("Default voice '{}' existence not verified", voice));
        }
        
        // Check output directory
        if let Some(ref output_dir) = self.config.cli.output_directory {
            if !output_dir.exists() {
                warnings.push(format!("Output directory '{}' does not exist", output_dir.display()));
            } else if !output_dir.is_dir() {
                return Err(CliError::config(format!("Output directory '{}' is not a directory", output_dir.display())));
            }
        }
        
        // Validate download settings
        if self.config.cli.download.parallel_downloads == 0 {
            return Err(CliError::config("parallel_downloads must be greater than 0"));
        }
        
        if self.config.cli.download.parallel_downloads > 10 {
            warnings.push("parallel_downloads > 10 may cause server rate limiting".to_string());
        }
        
        Ok(warnings)
    }
    
    /// Get configuration path
    pub fn config_path(&self) -> &Path {
        &self.config_path
    }
    
    /// Load configuration from file
    fn load_from_file<P: AsRef<Path>>(path: P) -> CliResult<CliConfig> {
        let content = fs::read_to_string(path.as_ref())
            .map_err(|e| CliError::file_operation("read", &path.as_ref().display().to_string(), e))?;
        
        // Try TOML first, then JSON for backward compatibility
        if let Ok(config) = toml::from_str::<CliConfig>(&content) {
            Ok(config)
        } else {
            serde_json::from_str::<CliConfig>(&content)
                .map_err(|e| CliError::config(format!("Invalid configuration format: {}", e)))
        }
    }
    
    /// Find configuration file in standard locations
    fn find_config_file() -> Option<PathBuf> {
        let possible_paths = [
            env::current_dir().ok().map(|d| d.join("voirs.toml")),
            env::current_dir().ok().map(|d| d.join("voirs.json")),
            Self::config_dir().map(|d| d.join("voirs.toml")),
            Self::config_dir().map(|d| d.join("voirs.json")),
            env::var("VOIRS_CONFIG").ok().map(PathBuf::from),
        ];
        
        for path in possible_paths.into_iter().flatten() {
            if path.exists() {
                return Some(path);
            }
        }
        
        None
    }
    
    /// Get default configuration path
    fn default_config_path() -> PathBuf {
        Self::config_dir()
            .unwrap_or_else(|| env::current_dir().unwrap())
            .join("voirs.toml")
    }
    
    /// Get configuration directory
    fn config_dir() -> Option<PathBuf> {
        if let Some(config_dir) = env::var_os("XDG_CONFIG_HOME") {
            Some(PathBuf::from(config_dir).join("voirs"))
        } else if let Some(home_dir) = env::var_os("HOME") {
            Some(PathBuf::from(home_dir).join(".config").join("voirs"))
        } else if let Some(app_data) = env::var_os("APPDATA") {
            Some(PathBuf::from(app_data).join("voirs"))
        } else {
            None
        }
    }
}

/// Configuration utilities
pub mod utils {
    use super::*;
    
    /// Create a default configuration file
    pub fn create_default_config<P: AsRef<Path>>(path: P) -> CliResult<()> {
        let config = CliConfig::default();
        let content = toml::to_string_pretty(&config)
            .map_err(CliError::from)?;
        
        if let Some(parent) = path.as_ref().parent() {
            fs::create_dir_all(parent)
                .map_err(|e| CliError::file_operation("create directory", &parent.display().to_string(), e))?;
        }
        
        fs::write(path.as_ref(), content)
            .map_err(|e| CliError::file_operation("write", &path.as_ref().display().to_string(), e))?;
        
        Ok(())
    }
    
    /// Migrate old configuration format to new format
    pub fn migrate_config<P: AsRef<Path>>(old_path: P, new_path: P) -> CliResult<()> {
        let old_content = fs::read_to_string(old_path.as_ref())
            .map_err(|e| CliError::file_operation("read", &old_path.as_ref().display().to_string(), e))?;
        
        // Try to parse as old format (assuming it was JSON)
        let old_config: serde_json::Value = serde_json::from_str(&old_content)
            .map_err(|e| CliError::config(format!("Cannot parse old config: {}", e)))?;
        
        // Create new config with migrated values
        let mut new_config = CliConfig::default();
        
        // Migrate known fields (this is a simplified example)
        if let Some(output_format) = old_config.get("output_format") {
            if let Some(format_str) = output_format.as_str() {
                new_config.cli.default_output_format = format_str.to_string();
            }
        }
        
        // Save migrated config
        let content = toml::to_string_pretty(&new_config)
            .map_err(CliError::from)?;
        
        fs::write(new_path.as_ref(), content)
            .map_err(|e| CliError::file_operation("write", &new_path.as_ref().display().to_string(), e))?;
        
        Ok(())
    }
    
    /// Export configuration for sharing
    pub fn export_config<P: AsRef<Path>>(config: &CliConfig, path: P, format: ConfigFormat) -> CliResult<()> {
        let content = match format {
            ConfigFormat::Toml => toml::to_string_pretty(config)?,
            ConfigFormat::Json => serde_json::to_string_pretty(config)?,
            ConfigFormat::Yaml => serde_yaml::to_string(config)
                .map_err(|e| CliError::config(format!("YAML serialization error: {}", e)))?,
        };
        
        fs::write(path.as_ref(), content)
            .map_err(|e| CliError::file_operation("write", &path.as_ref().display().to_string(), e))?;
        
        Ok(())
    }
}

/// Configuration export formats
pub enum ConfigFormat {
    Toml,
    Json,
    Yaml,
}
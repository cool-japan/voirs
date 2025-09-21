//! Unified Configuration Management System
//!
//! This module provides a centralized configuration management system that coordinates
//! all the different configuration structures across the voice cloning system modules.
//! It includes configuration validation, hot-reloading, versioning, and runtime updates.

use crate::{
    config::CloningConfig,
    embedding::EmbeddingConfig,
    gpu_acceleration::GpuAccelerationConfig,
    memory_optimization::MemoryOptimizationConfig,
    // ab_testing::ABTestConfig, // Temporarily commented out due to serialization issues
    // age_gender_adaptation::AgeGenderAdaptationConfig,
    // perceptual_evaluation::PerceptualEvaluationConfig,
    // performance_monitoring::PerformanceTargets,
    model_loading::ModelLoadingConfig,
    quality::QualityConfig,
    quantization::QuantizationConfig,
    streaming_adaptation::StreamingAdaptationConfig,
    Error,
    Result,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::fs;
use tokio::sync::{broadcast, RwLock};
use tracing::{debug, error, info, warn};

/// Unified configuration manager for the entire voice cloning system
pub struct UnifiedConfigManager {
    /// Current configuration state
    config: Arc<RwLock<SystemConfiguration>>,
    /// Configuration file watchers for hot reloading
    file_watchers: Arc<RwLock<HashMap<String, ConfigFileWatcher>>>,
    /// Configuration change broadcast channel
    change_broadcaster: broadcast::Sender<ConfigChangeEvent>,
    /// Configuration validation cache
    validation_cache: Arc<RwLock<HashMap<String, ValidationResult>>>,
    /// Configuration history for rollback
    config_history: Arc<RwLock<Vec<ConfigSnapshot>>>,
    /// Manager settings
    manager_config: ConfigManagerSettings,
}

/// Complete system configuration containing all module configurations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemConfiguration {
    /// Configuration metadata
    pub metadata: ConfigMetadata,
    /// Core cloning configuration
    pub cloning: CloningConfig,
    /// Speaker embedding configuration
    pub embedding: EmbeddingConfig,
    /// Quality assessment configuration
    pub quality: QualityConfig,
    /// GPU acceleration configuration
    pub gpu: GpuAccelerationConfig,
    /// Memory optimization configuration
    pub memory: MemoryOptimizationConfig,
    /// Model quantization configuration
    pub quantization: QuantizationConfig,
    /// Streaming adaptation configuration
    pub streaming: StreamingAdaptationConfig,
    // Note: These configurations temporarily commented out until they have proper Serialize/Deserialize
    // /// A/B testing configuration
    // pub ab_testing: ABTestConfig,
    // /// Age/gender adaptation configuration
    // pub age_gender: AgeGenderAdaptationConfig,
    // /// Perceptual evaluation configuration
    // pub perceptual: PerceptualEvaluationConfig,
    // /// Performance monitoring configuration
    // pub performance: PerformanceTargets,
    /// Model loading configuration
    pub model_loading: ModelLoadingConfig,
    /// Custom configuration extensions
    pub extensions: HashMap<String, serde_json::Value>,
}

/// Configuration metadata for tracking and versioning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigMetadata {
    /// Configuration version
    pub version: String,
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Last modification timestamp
    pub modified_at: SystemTime,
    /// Configuration source (file, API, default, etc.)
    pub source: ConfigSource,
    /// Configuration checksum for integrity verification
    pub checksum: String,
    /// Configuration description
    pub description: Option<String>,
    /// Environment this config is intended for
    pub environment: Environment,
}

/// Configuration source types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConfigSource {
    /// Loaded from file
    File(PathBuf),
    /// Received via API
    Api,
    /// Default configuration
    Default,
    /// Environment variables
    Environment,
    /// Merged from multiple sources
    Merged(Vec<ConfigSource>),
}

/// Deployment environments
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Environment {
    Development,
    Testing,
    Staging,
    Production,
    Custom(String),
}

/// Configuration manager settings
#[derive(Debug, Clone)]
pub struct ConfigManagerSettings {
    /// Enable configuration file watching for hot reloading
    pub enable_hot_reload: bool,
    /// Configuration validation on load
    pub enable_validation: bool,
    /// Maximum configuration history to keep
    pub max_history_size: usize,
    /// Configuration change broadcast buffer size
    pub change_buffer_size: usize,
    /// File watch debounce interval
    pub watch_debounce: Duration,
    /// Configuration backup directory
    pub backup_directory: Option<PathBuf>,
    /// Enable configuration encryption
    pub enable_encryption: bool,
}

impl Default for ConfigManagerSettings {
    fn default() -> Self {
        Self {
            enable_hot_reload: true,
            enable_validation: true,
            max_history_size: 100,
            change_buffer_size: 1000,
            watch_debounce: Duration::from_millis(500),
            backup_directory: None,
            enable_encryption: false,
        }
    }
}

/// Configuration file watcher for hot reloading
#[derive(Debug)]
pub struct ConfigFileWatcher {
    /// File path being watched
    pub file_path: PathBuf,
    /// Last known modification time
    pub last_modified: SystemTime,
    /// Watch handle (placeholder - would use notify crate in production)
    pub watch_handle: Option<String>,
}

/// Configuration change event for broadcasting
#[derive(Debug, Clone)]
pub struct ConfigChangeEvent {
    /// Type of change that occurred
    pub change_type: ConfigChangeType,
    /// Configuration section that changed
    pub section: String,
    /// Previous configuration (if applicable)
    pub previous_config: Option<serde_json::Value>,
    /// New configuration
    pub new_config: serde_json::Value,
    /// Timestamp of the change
    pub timestamp: SystemTime,
    /// Source of the change
    pub source: ConfigSource,
}

/// Types of configuration changes
#[derive(Debug, Clone, PartialEq)]
pub enum ConfigChangeType {
    /// Configuration was loaded for the first time
    Loaded,
    /// Configuration was updated
    Updated,
    /// Configuration was reloaded from file
    Reloaded,
    /// Configuration was reset to default
    Reset,
    /// Configuration was validated
    Validated,
    /// Configuration was rolled back
    RolledBack,
}

/// Configuration validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether the configuration is valid
    pub is_valid: bool,
    /// Validation errors
    pub errors: Vec<ValidationError>,
    /// Validation warnings
    pub warnings: Vec<ValidationWarning>,
    /// Validation timestamp
    pub validated_at: SystemTime,
}

/// Configuration validation error
#[derive(Debug, Clone)]
pub struct ValidationError {
    /// Configuration path where error occurred
    pub path: String,
    /// Error message
    pub message: String,
    /// Error severity
    pub severity: ValidationSeverity,
}

/// Configuration validation warning
#[derive(Debug, Clone)]
pub struct ValidationWarning {
    /// Configuration path where warning occurred
    pub path: String,
    /// Warning message
    pub message: String,
    /// Suggested fix
    pub suggestion: Option<String>,
}

/// Validation severity levels
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum ValidationSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Configuration snapshot for history and rollback
#[derive(Debug, Clone)]
pub struct ConfigSnapshot {
    /// Snapshot timestamp
    pub timestamp: SystemTime,
    /// Configuration at the time of snapshot
    pub config: SystemConfiguration,
    /// Reason for taking the snapshot
    pub reason: String,
    /// Snapshot ID
    pub id: String,
}

impl Default for SystemConfiguration {
    fn default() -> Self {
        Self {
            metadata: ConfigMetadata {
                version: "1.0.0".to_string(),
                created_at: SystemTime::now(),
                modified_at: SystemTime::now(),
                source: ConfigSource::Default,
                checksum: "default".to_string(),
                description: Some("Default system configuration".to_string()),
                environment: Environment::Development,
            },
            cloning: CloningConfig::default(),
            embedding: EmbeddingConfig::default(),
            quality: QualityConfig::default(),
            gpu: GpuAccelerationConfig::default(),
            memory: MemoryOptimizationConfig::default(),
            quantization: QuantizationConfig::default(),
            streaming: StreamingAdaptationConfig::default(),
            // Note: These configurations temporarily commented out
            // ab_testing: ABTestConfig::default(),
            // age_gender: AgeGenderAdaptationConfig::default(),
            // perceptual: PerceptualEvaluationConfig::default(),
            // performance: PerformanceTargets::default(),
            model_loading: ModelLoadingConfig::default(),
            extensions: HashMap::new(),
        }
    }
}

impl UnifiedConfigManager {
    /// Create new unified configuration manager
    pub fn new(settings: ConfigManagerSettings) -> Self {
        let (change_broadcaster, _) = broadcast::channel(settings.change_buffer_size);

        Self {
            config: Arc::new(RwLock::new(SystemConfiguration::default())),
            file_watchers: Arc::new(RwLock::new(HashMap::new())),
            change_broadcaster,
            validation_cache: Arc::new(RwLock::new(HashMap::new())),
            config_history: Arc::new(RwLock::new(Vec::new())),
            manager_config: settings,
        }
    }

    /// Load configuration from file
    pub async fn load_from_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref().to_path_buf();
        info!("Loading configuration from file: {:?}", path);

        // Read configuration file
        let content = fs::read_to_string(&path)
            .await
            .map_err(|e| Error::Validation(format!("Failed to read config file: {e}")))?;

        // Parse configuration (JSON only for now)
        let mut new_config: SystemConfiguration = match path.extension().and_then(|s| s.to_str()) {
            Some("json") => serde_json::from_str(&content)
                .map_err(|e| Error::Validation(format!("Invalid JSON config: {e}")))?,
            _ => {
                return Err(Error::Validation(
                    "Only JSON config files are supported".to_string(),
                ))
            }
        };

        // Update metadata
        new_config.metadata.source = ConfigSource::File(path.clone());
        new_config.metadata.modified_at = SystemTime::now();
        new_config.metadata.checksum = self.calculate_checksum(&new_config).await?;

        // Validate configuration if enabled
        if self.manager_config.enable_validation {
            let validation = self.validate_configuration(&new_config).await?;
            if !validation.is_valid {
                let errors: Vec<String> = validation
                    .errors
                    .iter()
                    .map(|e| format!("{path}: {message}", path = e.path, message = e.message))
                    .collect();
                return Err(Error::Validation(format!(
                    "Configuration validation failed: {}",
                    errors.join(", ")
                )));
            }
        }

        // Take snapshot before update
        self.take_snapshot("Before file load", &*self.config.read().await)
            .await?;

        // Update configuration
        let mut config = self.config.write().await;
        *config = new_config.clone();
        drop(config);

        // Set up file watcher if hot reload is enabled
        if self.manager_config.enable_hot_reload {
            self.setup_file_watcher(path.clone()).await?;
        }

        // Broadcast change event
        let change_event = ConfigChangeEvent {
            change_type: ConfigChangeType::Loaded,
            section: "system".to_string(),
            previous_config: None,
            new_config: serde_json::to_value(&new_config)
                .map_err(|e| Error::Validation(format!("Failed to serialize config: {e}")))?,
            timestamp: SystemTime::now(),
            source: ConfigSource::File(path),
        };

        let _ = self.change_broadcaster.send(change_event);

        info!("Configuration loaded successfully");
        Ok(())
    }

    /// Save configuration to file
    pub async fn save_to_file<P: AsRef<Path>>(
        &self,
        path: P,
        format: ConfigFileFormat,
    ) -> Result<()> {
        let path = path.as_ref();
        let config = self.config.read().await;

        let content = match format {
            ConfigFileFormat::Json => serde_json::to_string_pretty(&*config)
                .map_err(|e| Error::Validation(format!("Failed to serialize to JSON: {e}")))?,
            _ => {
                return Err(Error::Validation(
                    "Only JSON format is currently supported".to_string(),
                ))
            }
        };

        // Create backup if backup directory is configured
        if let Some(backup_dir) = &self.manager_config.backup_directory {
            self.create_backup(&*config, backup_dir).await?;
        }

        fs::write(path, content)
            .await
            .map_err(|e| Error::Validation(format!("Failed to write config file: {e}")))?;

        info!("Configuration saved to: {:?}", path);
        Ok(())
    }

    /// Get current configuration
    pub async fn get_config(&self) -> SystemConfiguration {
        self.config.read().await.clone()
    }

    /// Update specific configuration section
    pub async fn update_section<T>(&self, section: &str, new_config: T) -> Result<()>
    where
        T: Serialize + Clone,
    {
        let new_value = serde_json::to_value(&new_config)
            .map_err(|e| Error::Validation(format!("Failed to serialize config: {}", e)))?;

        // Take snapshot before update
        self.take_snapshot(
            &format!("Before updating section: {section}"),
            &*self.config.read().await,
        )
        .await?;

        let mut config = self.config.write().await;
        let previous_config = self.get_section_value(&*config, section)?;

        // Update the specific section
        self.set_section_value(&mut *config, section, new_value.clone())?;
        config.metadata.modified_at = SystemTime::now();
        config.metadata.checksum = self.calculate_checksum(&*config).await?;

        // Broadcast change event
        let change_event = ConfigChangeEvent {
            change_type: ConfigChangeType::Updated,
            section: section.to_string(),
            previous_config: Some(previous_config),
            new_config: new_value,
            timestamp: SystemTime::now(),
            source: ConfigSource::Api,
        };

        let _ = self.change_broadcaster.send(change_event);

        info!("Configuration section '{}' updated", section);
        Ok(())
    }

    /// Subscribe to configuration changes
    pub fn subscribe_to_changes(&self) -> broadcast::Receiver<ConfigChangeEvent> {
        self.change_broadcaster.subscribe()
    }

    /// Validate configuration
    pub async fn validate_configuration(
        &self,
        config: &SystemConfiguration,
    ) -> Result<ValidationResult> {
        let mut validation = ValidationResult {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            validated_at: SystemTime::now(),
        };

        // Validate cloning configuration
        self.validate_cloning_config(&config.cloning, &mut validation)
            .await;

        // Validate embedding configuration
        self.validate_embedding_config(&config.embedding, &mut validation)
            .await;

        // Validate GPU configuration
        self.validate_gpu_config(&config.gpu, &mut validation).await;

        // Validate memory configuration
        self.validate_memory_config(&config.memory, &mut validation)
            .await;

        // Cross-validation between modules
        self.validate_cross_module_consistency(config, &mut validation)
            .await;

        validation.is_valid = validation.errors.is_empty();
        Ok(validation)
    }

    /// Rollback to previous configuration
    pub async fn rollback(&self, snapshot_id: Option<String>) -> Result<()> {
        let history = self.config_history.read().await;

        let snapshot = if let Some(id) = snapshot_id {
            history
                .iter()
                .find(|s| s.id == id)
                .ok_or_else(|| Error::Validation(format!("Snapshot {id} not found")))?
        } else {
            history.last().ok_or_else(|| {
                Error::Validation("No snapshots available for rollback".to_string())
            })?
        };

        let mut config = self.config.write().await;
        *config = snapshot.config.clone();
        config.metadata.modified_at = SystemTime::now();

        info!("Configuration rolled back to snapshot: {}", snapshot.id);

        // Broadcast rollback event
        let change_event = ConfigChangeEvent {
            change_type: ConfigChangeType::RolledBack,
            section: "system".to_string(),
            previous_config: None,
            new_config: serde_json::to_value(&*config)
                .map_err(|e| Error::Validation(format!("Failed to serialize config: {e}")))?,
            timestamp: SystemTime::now(),
            source: ConfigSource::Api,
        };

        let _ = self.change_broadcaster.send(change_event);
        Ok(())
    }

    /// Get configuration history
    pub async fn get_history(&self) -> Vec<ConfigSnapshot> {
        self.config_history.read().await.clone()
    }

    /// Reset configuration to default
    pub async fn reset_to_default(&self) -> Result<()> {
        // Take snapshot before reset
        self.take_snapshot("Before reset to default", &*self.config.read().await)
            .await?;

        let mut config = self.config.write().await;
        *config = SystemConfiguration::default();

        info!("Configuration reset to default");

        // Broadcast reset event
        let change_event = ConfigChangeEvent {
            change_type: ConfigChangeType::Reset,
            section: "system".to_string(),
            previous_config: None,
            new_config: serde_json::to_value(&*config)
                .map_err(|e| Error::Validation(format!("Failed to serialize config: {e}")))?,
            timestamp: SystemTime::now(),
            source: ConfigSource::Default,
        };

        let _ = self.change_broadcaster.send(change_event);
        Ok(())
    }

    /// Private helper methods
    async fn setup_file_watcher(&self, path: PathBuf) -> Result<()> {
        let metadata = fs::metadata(&path)
            .await
            .map_err(|e| Error::Validation(format!("Failed to get file metadata: {e}")))?;

        let watcher = ConfigFileWatcher {
            file_path: path.clone(),
            last_modified: metadata
                .modified()
                .map_err(|e| Error::Validation(format!("Failed to get modification time: {e}")))?,
            watch_handle: Some("placeholder".to_string()), // Would use notify crate
        };

        let mut watchers = self.file_watchers.write().await;
        watchers.insert(path.to_string_lossy().to_string(), watcher);

        Ok(())
    }

    async fn take_snapshot(&self, reason: &str, config: &SystemConfiguration) -> Result<()> {
        let snapshot = ConfigSnapshot {
            timestamp: SystemTime::now(),
            config: config.clone(),
            reason: reason.to_string(),
            id: format!(
                "snapshot_{}",
                SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs()
            ),
        };

        let mut history = self.config_history.write().await;
        history.push(snapshot);

        // Limit history size
        while history.len() > self.manager_config.max_history_size {
            history.remove(0);
        }

        Ok(())
    }

    async fn calculate_checksum(&self, config: &SystemConfiguration) -> Result<String> {
        let serialized = serde_json::to_string(config)
            .map_err(|e| Error::Validation(format!("Failed to serialize for checksum: {e}")))?;

        // Simple checksum - in production would use proper hash
        Ok(format!("{:x}", serialized.len()))
    }

    async fn create_backup(&self, config: &SystemConfiguration, backup_dir: &Path) -> Result<()> {
        let timestamp = SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let backup_path = backup_dir.join(format!("config_backup_{timestamp}.json"));

        let content = serde_json::to_string_pretty(config)
            .map_err(|e| Error::Validation(format!("Failed to serialize backup: {e}")))?;

        fs::write(&backup_path, content)
            .await
            .map_err(|e| Error::Validation(format!("Failed to write backup: {e}")))?;

        debug!("Configuration backup created: {:?}", backup_path);
        Ok(())
    }

    fn get_section_value(
        &self,
        config: &SystemConfiguration,
        section: &str,
    ) -> Result<serde_json::Value> {
        match section {
            "cloning" => serde_json::to_value(&config.cloning),
            "embedding" => serde_json::to_value(&config.embedding),
            "quality" => serde_json::to_value(&config.quality),
            "gpu" => serde_json::to_value(&config.gpu),
            "memory" => serde_json::to_value(&config.memory),
            "quantization" => serde_json::to_value(&config.quantization),
            "streaming" => serde_json::to_value(&config.streaming),
            // "ab_testing" => serde_json::to_value(&config.ab_testing),
            // "age_gender" => serde_json::to_value(&config.age_gender),
            // "perceptual" => serde_json::to_value(&config.perceptual),
            // "performance" => serde_json::to_value(&config.performance),
            "model_loading" => serde_json::to_value(&config.model_loading),
            _ => {
                return Err(Error::Validation(format!(
                    "Unknown config section: {}",
                    section
                )))
            }
        }
        .map_err(|e| Error::Validation(format!("Failed to serialize section: {e}")))
    }

    fn set_section_value(
        &self,
        config: &mut SystemConfiguration,
        section: &str,
        value: serde_json::Value,
    ) -> Result<()> {
        match section {
            "cloning" => config.cloning = serde_json::from_value(value)?,
            "embedding" => config.embedding = serde_json::from_value(value)?,
            "quality" => config.quality = serde_json::from_value(value)?,
            "gpu" => config.gpu = serde_json::from_value(value)?,
            "memory" => config.memory = serde_json::from_value(value)?,
            "quantization" => config.quantization = serde_json::from_value(value)?,
            "streaming" => config.streaming = serde_json::from_value(value)?,
            // "ab_testing" => config.ab_testing = serde_json::from_value(value)?,
            // "age_gender" => config.age_gender = serde_json::from_value(value)?,
            // "perceptual" => config.perceptual = serde_json::from_value(value)?,
            // "performance" => config.performance = serde_json::from_value(value)?,
            "model_loading" => config.model_loading = serde_json::from_value(value)?,
            _ => {
                return Err(Error::Validation(format!(
                    "Unknown config section: {}",
                    section
                )))
            }
        }
        Ok(())
    }

    // Validation helper methods
    async fn validate_cloning_config(
        &self,
        config: &CloningConfig,
        validation: &mut ValidationResult,
    ) {
        if config.quality_level < 0.0 || config.quality_level > 1.0 {
            validation.errors.push(ValidationError {
                path: "cloning.quality_level".to_string(),
                message: "Quality level must be between 0.0 and 1.0".to_string(),
                severity: ValidationSeverity::Error,
            });
        }
    }

    async fn validate_embedding_config(
        &self,
        config: &EmbeddingConfig,
        validation: &mut ValidationResult,
    ) {
        if config.dimension == 0 {
            validation.errors.push(ValidationError {
                path: "embedding.embedding_dim".to_string(),
                message: "Embedding dimension must be greater than 0".to_string(),
                severity: ValidationSeverity::Error,
            });
        }
    }

    async fn validate_gpu_config(
        &self,
        config: &GpuAccelerationConfig,
        validation: &mut ValidationResult,
    ) {
        if config.batch_size == 0 {
            validation.warnings.push(ValidationWarning {
                path: "gpu.batch_size".to_string(),
                message: "Batch size of 0 may impact performance".to_string(),
                suggestion: Some("Consider setting batch_size to at least 1".to_string()),
            });
        }
    }

    async fn validate_memory_config(
        &self,
        config: &MemoryOptimizationConfig,
        validation: &mut ValidationResult,
    ) {
        if config.max_memory_usage == 0 {
            validation.warnings.push(ValidationWarning {
                path: "memory.max_memory_usage".to_string(),
                message: "Max memory usage of 0 may impact performance".to_string(),
                suggestion: Some("Consider setting max_memory_usage to at least 64MB".to_string()),
            });
        }
    }

    async fn validate_cross_module_consistency(
        &self,
        config: &SystemConfiguration,
        validation: &mut ValidationResult,
    ) {
        // Example: GPU batch size should be compatible with memory limits
        if config.gpu.batch_size > 32 && config.memory.max_memory_usage < 512 {
            validation.warnings.push(ValidationWarning {
                path: "gpu.batch_size,memory.max_memory_usage".to_string(),
                message:
                    "Large GPU batch size with small memory cache may cause performance issues"
                        .to_string(),
                suggestion: Some("Increase memory cache size or reduce GPU batch size".to_string()),
            });
        }
    }
}

/// Configuration file formats
#[derive(Debug, Clone, Copy)]
pub enum ConfigFileFormat {
    Json,
    // Toml and Yaml support can be added when those dependencies are available
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;

    #[tokio::test]
    async fn test_config_manager_creation() {
        let settings = ConfigManagerSettings::default();
        let manager = UnifiedConfigManager::new(settings);

        let config = manager.get_config().await;
        assert_eq!(config.metadata.version, "1.0.0");
        assert_eq!(config.metadata.source, ConfigSource::Default);
    }

    #[tokio::test]
    async fn test_config_validation() {
        let settings = ConfigManagerSettings::default();
        let manager = UnifiedConfigManager::new(settings);

        let config = manager.get_config().await;
        let validation = manager.validate_configuration(&config).await.unwrap();

        assert!(validation.is_valid);
        assert!(validation.errors.is_empty());
    }

    #[tokio::test]
    async fn test_config_section_update() {
        let settings = ConfigManagerSettings::default();
        let manager = UnifiedConfigManager::new(settings);

        let mut new_cloning_config = CloningConfig::default();
        new_cloning_config.quality_level = 0.8;

        let result = manager
            .update_section("cloning", new_cloning_config.clone())
            .await;
        assert!(result.is_ok());

        let updated_config = manager.get_config().await;
        assert_eq!(updated_config.cloning.quality_level, 0.8);
    }

    #[tokio::test]
    async fn test_config_history() {
        let settings = ConfigManagerSettings::default();
        let manager = UnifiedConfigManager::new(settings);

        // Make a change to create history
        let new_config = CloningConfig::default();
        manager.update_section("cloning", new_config).await.unwrap();

        let history = manager.get_history().await;
        assert!(!history.is_empty());
    }

    #[tokio::test]
    async fn test_invalid_config_validation() {
        let settings = ConfigManagerSettings::default();
        let manager = UnifiedConfigManager::new(settings);

        let mut config = manager.get_config().await;
        config.cloning.quality_level = 2.0; // Invalid value > 1.0

        let validation = manager.validate_configuration(&config).await.unwrap();
        assert!(!validation.is_valid);
        assert!(!validation.errors.is_empty());
    }

    #[tokio::test]
    async fn test_config_change_subscription() {
        let settings = ConfigManagerSettings::default();
        let manager = UnifiedConfigManager::new(settings);

        let mut receiver = manager.subscribe_to_changes();

        // Make a change
        let new_config = CloningConfig::default();
        manager.update_section("cloning", new_config).await.unwrap();

        // Should receive change event
        let event = receiver.recv().await.unwrap();
        assert_eq!(event.change_type, ConfigChangeType::Updated);
        assert_eq!(event.section, "cloning");
    }
}

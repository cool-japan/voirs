//! Dynamic configuration management and runtime updates.
//!
//! This module provides facilities for:
//!
//! - Runtime configuration updates
//! - Configuration validation and verification
//! - Configuration change notifications
//! - Hot-reloading capabilities
//! - Configuration migration and versioning
//! - Thread-safe configuration management

use super::{
    hierarchy::{AppConfig, PipelineConfig, ConfigValidationError, ConfigHierarchy},
    persistence::{ConfigLoader, ConfigLoadError},
};
use crate::VoirsError;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{Arc, RwLock},
    time::{Duration, Instant},
};
use tokio::sync::{broadcast, watch};

/// Dynamic configuration manager with hot-reloading support
pub struct DynamicConfigManager {
    /// Current configuration
    config: Arc<RwLock<AppConfig>>,
    
    /// Configuration change notifier
    change_notifier: broadcast::Sender<ConfigChangeEvent>,
    
    /// Validation rules
    validators: Vec<Box<dyn ConfigValidator + Send + Sync>>,
    
    /// Configuration history for rollback
    history: Arc<RwLock<ConfigHistory>>,
}

impl DynamicConfigManager {
    /// Create new dynamic configuration manager
    pub fn new(initial_config: AppConfig) -> Self {
        let (change_notifier, _) = broadcast::channel(100);
        
        Self {
            config: Arc::new(RwLock::new(initial_config)),
            change_notifier,
            validators: Vec::new(),
            history: Arc::new(RwLock::new(ConfigHistory::new())),
        }
    }
    
    /// Add configuration validator
    pub fn add_validator<V: ConfigValidator + Send + Sync + 'static>(mut self, validator: V) -> Self {
        self.validators.push(Box::new(validator));
        self
    }
    
    /// Get current configuration (read-only)
    pub fn get_config(&self) -> AppConfig {
        self.config.read().unwrap().clone()
    }
    
    /// Update configuration with validation
    pub async fn update_config(&self, new_config: AppConfig) -> Result<(), ConfigUpdateError> {
        // Validate new configuration
        self.validate_config(&new_config).await?;
        
        // Store current config for potential rollback
        let previous_config = {
            let current = self.config.read().unwrap();
            current.clone()
        };
        
        // Apply update
        {
            let mut config = self.config.write().unwrap();
            *config = new_config.clone();
        }
        
        // Record in history
        {
            let mut history = self.history.write().unwrap();
            history.record_change(previous_config.clone(), new_config.clone());
        }
        
        // Notify subscribers
        let event = ConfigChangeEvent {
            timestamp: Instant::now(),
            change_type: ConfigChangeType::Update,
            previous: Some(previous_config),
            current: new_config,
        };
        
        let _ = self.change_notifier.send(event);
        
        Ok(())
    }
    
    /// Merge configuration changes
    pub async fn merge_config(&self, config_update: AppConfig) -> Result<(), ConfigUpdateError> {
        let mut new_config = self.get_config();
        new_config.merge_with(&config_update);
        self.update_config(new_config).await
    }
    
    /// Update specific configuration section
    pub async fn update_pipeline_config(&self, pipeline_config: PipelineConfig) -> Result<(), ConfigUpdateError> {
        let mut new_config = self.get_config();
        new_config.pipeline = pipeline_config;
        self.update_config(new_config).await
    }
    
    /// Rollback to previous configuration
    pub async fn rollback(&self) -> Result<(), ConfigUpdateError> {
        let previous_config = {
            let history = self.history.read().unwrap();
            history.get_previous()
                .ok_or(ConfigUpdateError::NoHistory)?
                .clone()
        };
        
        self.update_config(previous_config).await
    }
    
    /// Subscribe to configuration changes
    pub fn subscribe_to_changes(&self) -> broadcast::Receiver<ConfigChangeEvent> {
        self.change_notifier.subscribe()
    }
    
    /// Validate configuration against all registered validators
    async fn validate_config(&self, config: &AppConfig) -> Result<(), ConfigUpdateError> {
        // Built-in validation
        config.validate().map_err(ConfigUpdateError::ValidationError)?;
        
        // Custom validators
        for validator in &self.validators {
            validator.validate(config)
                .map_err(ConfigUpdateError::CustomValidation)?;
        }
        
        Ok(())
    }
    
    /// Get configuration history
    pub fn get_history(&self) -> Vec<ConfigHistoryEntry> {
        let history = self.history.read().unwrap();
        history.entries.clone()
    }
    
    /// Clear configuration history
    pub fn clear_history(&self) {
        let mut history = self.history.write().unwrap();
        history.clear();
    }
    
    /// Export current configuration
    pub fn export_config(&self) -> AppConfig {
        self.get_config()
    }
    
    /// Import and apply configuration
    pub async fn import_config(&self, config: AppConfig) -> Result<(), ConfigUpdateError> {
        self.update_config(config).await
    }
}

/// Configuration change event
#[derive(Debug, Clone)]
pub struct ConfigChangeEvent {
    /// When the change occurred
    pub timestamp: Instant,
    
    /// Type of change
    pub change_type: ConfigChangeType,
    
    /// Previous configuration (if applicable)
    pub previous: Option<AppConfig>,
    
    /// Current configuration
    pub current: AppConfig,
}

/// Type of configuration change
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigChangeType {
    /// Configuration was updated
    Update,
    /// Configuration was reloaded from file
    Reload,
    /// Configuration was rolled back
    Rollback,
    /// Configuration was migrated
    Migration,
}

/// Configuration validator trait
pub trait ConfigValidator {
    /// Validate configuration
    fn validate(&self, config: &AppConfig) -> Result<(), String>;
}

/// Configuration migrator trait
pub trait ConfigMigrator {
    /// Check if configuration needs migration
    fn needs_migration(&self, config: &AppConfig) -> bool;
    
    /// Migrate configuration to new version
    fn migrate(&self, config: AppConfig) -> Result<AppConfig, Box<dyn std::error::Error>>;
}

/// Configuration history manager
#[derive(Debug, Clone)]
struct ConfigHistory {
    entries: Vec<ConfigHistoryEntry>,
    max_entries: usize,
}

impl ConfigHistory {
    fn new() -> Self {
        Self {
            entries: Vec::new(),
            max_entries: 50, // Keep last 50 configurations
        }
    }
    
    fn record_change(&mut self, previous: AppConfig, current: AppConfig) {
        let entry = ConfigHistoryEntry {
            timestamp: Instant::now(),
            previous,
            current,
        };
        
        self.entries.push(entry);
        
        // Maintain max entries limit
        if self.entries.len() > self.max_entries {
            self.entries.remove(0);
        }
    }
    
    fn get_previous(&self) -> Option<&AppConfig> {
        self.entries.last().map(|entry| &entry.previous)
    }
    
    fn clear(&mut self) {
        self.entries.clear();
    }
}

/// Configuration history entry
#[derive(Debug, Clone)]
pub struct ConfigHistoryEntry {
    /// When the change occurred
    pub timestamp: Instant,
    
    /// Previous configuration
    pub previous: AppConfig,
    
    /// New configuration
    pub current: AppConfig,
}

/// Configuration update error
#[derive(Debug, thiserror::Error)]
pub enum ConfigUpdateError {
    /// Configuration validation failed
    #[error("Configuration validation failed: {0}")]
    ValidationError(#[from] ConfigValidationError),
    
    /// Custom validation failed
    #[error("Custom validation failed: {0}")]
    CustomValidation(String),
    
    /// Configuration loading failed
    #[error("Configuration loading failed: {0}")]
    LoadError(#[from] ConfigLoadError),
    
    /// No configuration history available for rollback
    #[error("No configuration history available for rollback")]
    NoHistory,
}

/// Configuration validator for resource constraints
pub struct ResourceValidator {
    max_cache_size_mb: u32,
    max_threads: usize,
}

impl ResourceValidator {
    pub fn new(max_cache_size_mb: u32, max_threads: usize) -> Self {
        Self {
            max_cache_size_mb,
            max_threads,
        }
    }
}

impl ConfigValidator for ResourceValidator {
    fn validate(&self, config: &AppConfig) -> Result<(), String> {
        if config.pipeline.max_cache_size_mb > self.max_cache_size_mb {
            return Err(format!(
                "Cache size {} MB exceeds maximum allowed {} MB",
                config.pipeline.max_cache_size_mb, self.max_cache_size_mb
            ));
        }
        
        if let Some(threads) = config.pipeline.num_threads {
            if threads > self.max_threads {
                return Err(format!(
                    "Thread count {} exceeds maximum allowed {}",
                    threads, self.max_threads
                ));
            }
        }
        
        Ok(())
    }
}

/// Configuration validator for device availability
pub struct DeviceValidator;

impl ConfigValidator for DeviceValidator {
    fn validate(&self, config: &AppConfig) -> Result<(), String> {
        if config.pipeline.use_gpu && config.pipeline.device == "cpu" {
            return Err("Cannot enable GPU acceleration with CPU device".to_string());
        }
        
        // In a real implementation, you might check actual device availability
        match config.pipeline.device.as_str() {
            "cpu" => Ok(()),
            "cuda" => {
                // Check CUDA availability
                if !Self::is_cuda_available() {
                    Err("CUDA device requested but not available".to_string())
                } else {
                    Ok(())
                }
            }
            "metal" => {
                // Check Metal availability (macOS only)
                #[cfg(target_os = "macos")]
                {
                    Ok(()) // Assume Metal is available on macOS
                }
                #[cfg(not(target_os = "macos"))]
                {
                    Err("Metal device not available on this platform".to_string())
                }
            }
            _ => Err(format!("Unknown device: {}", config.pipeline.device)),
        }
    }
}

impl DeviceValidator {
    fn is_cuda_available() -> bool {
        // In a real implementation, this would check for CUDA installation
        // and available devices
        false // Placeholder
    }
}

/// Convenience functions for common operations
pub struct ConfigManagement;

impl ConfigManagement {
    /// Create a new dynamic configuration manager with common validators
    pub fn create_manager(config: AppConfig) -> DynamicConfigManager {
        DynamicConfigManager::new(config)
            .add_validator(ResourceValidator::new(10_000, 64)) // 10GB cache, 64 threads max
            .add_validator(DeviceValidator)
    }
    
    /// Validate configuration without applying it
    pub async fn validate_config(config: &AppConfig) -> Result<(), ConfigUpdateError> {
        let manager = Self::create_manager(AppConfig::default());
        manager.validate_config(config).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    
    #[tokio::test]
    async fn test_config_update() {
        let initial_config = AppConfig::default();
        let manager = DynamicConfigManager::new(initial_config.clone());
        
        let mut new_config = initial_config;
        new_config.pipeline.device = "cuda".to_string();
        
        assert!(manager.update_config(new_config).await.is_ok());
        
        let updated = manager.get_config();
        assert_eq!(updated.pipeline.device, "cuda");
    }
    
    #[tokio::test]
    async fn test_config_validation() {
        let manager = DynamicConfigManager::new(AppConfig::default())
            .add_validator(ResourceValidator::new(1000, 8));
        
        let mut invalid_config = AppConfig::default();
        invalid_config.pipeline.max_cache_size_mb = 2000; // Exceeds limit
        
        let result = manager.update_config(invalid_config).await;
        assert!(result.is_err());
    }
    
    #[tokio::test]
    async fn test_config_rollback() {
        let initial_config = AppConfig::default();
        let manager = DynamicConfigManager::new(initial_config.clone());
        
        // Update configuration
        let mut new_config = initial_config.clone();
        new_config.pipeline.device = "cuda".to_string();
        manager.update_config(new_config).await.unwrap();
        
        // Rollback
        manager.rollback().await.unwrap();
        
        let current = manager.get_config();
        assert_eq!(current.pipeline.device, initial_config.pipeline.device);
    }
    
    #[tokio::test]
    async fn test_config_merge() {
        let manager = DynamicConfigManager::new(AppConfig::default());
        
        let mut update = AppConfig::default();
        update.pipeline.device = "cuda".to_string();
        update.pipeline.use_gpu = true;
        
        manager.merge_config(update).await.unwrap();
        
        let current = manager.get_config();
        assert_eq!(current.pipeline.device, "cuda");
        assert_eq!(current.pipeline.use_gpu, true);
        // Other fields should remain default
        assert_eq!(current.pipeline.max_cache_size_mb, 1024);
    }
    
    #[test]
    fn test_device_validator() {
        let validator = DeviceValidator;
        
        let mut config = AppConfig::default();
        config.pipeline.device = "cpu".to_string();
        config.pipeline.use_gpu = false;
        assert!(validator.validate(&config).is_ok());
        
        config.pipeline.use_gpu = true; // Invalid combination
        assert!(validator.validate(&config).is_err());
    }
    
    #[test]
    fn test_resource_validator() {
        let validator = ResourceValidator::new(1000, 8);
        
        let mut config = AppConfig::default();
        config.pipeline.max_cache_size_mb = 500;
        config.pipeline.num_threads = Some(4);
        assert!(validator.validate(&config).is_ok());
        
        config.pipeline.max_cache_size_mb = 2000; // Exceeds limit
        assert!(validator.validate(&config).is_err());
        
        config.pipeline.max_cache_size_mb = 500;
        config.pipeline.num_threads = Some(16); // Exceeds limit
        assert!(validator.validate(&config).is_err());
    }
    
    #[test]
    fn test_config_history() {
        let mut history = ConfigHistory::new();
        
        let config1 = AppConfig::default();
        let mut config2 = AppConfig::default();
        config2.pipeline.device = "cuda".to_string();
        
        history.record_change(config1.clone(), config2.clone());
        
        assert!(history.get_previous().is_some());
        assert_eq!(history.get_previous().unwrap().pipeline.device, "cpu");
    }
    
    #[tokio::test]
    async fn test_change_notifications() {
        let manager = DynamicConfigManager::new(AppConfig::default());
        let mut receiver = manager.subscribe_to_changes();
        
        // Update config in background
        let manager_clone = manager;
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(10)).await;
            let mut new_config = AppConfig::default();
            new_config.pipeline.device = "cuda".to_string();
            let _ = manager_clone.update_config(new_config).await;
        });
        
        // Wait for notification
        let event = receiver.recv().await.unwrap();
        assert_eq!(event.change_type, ConfigChangeType::Update);
        assert_eq!(event.current.pipeline.device, "cuda");
    }
}
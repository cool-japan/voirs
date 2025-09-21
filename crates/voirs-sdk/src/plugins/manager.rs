//! Plugin manager for loading and managing plugins.

use crate::{
    error::{Result, VoirsError},
    plugins::{
        AudioEffect, PluginConfig, PluginMetadata, PluginType, TextProcessor, VoiceEffect,
        VoirsPlugin,
    },
};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{Arc, RwLock},
    time::Instant,
};
use tracing::{error, info, warn};

/// Plugin manager for loading and managing plugins
pub struct PluginManager {
    /// Loaded plugins by name
    plugins: Arc<RwLock<HashMap<String, Arc<dyn VoirsPlugin>>>>,

    /// Audio effect plugins
    audio_effects: Arc<RwLock<HashMap<String, Arc<dyn AudioEffect>>>>,

    /// Voice effect plugins
    voice_effects: Arc<RwLock<HashMap<String, Arc<dyn VoiceEffect>>>>,

    /// Text processor plugins
    text_processors: Arc<RwLock<HashMap<String, Arc<dyn TextProcessor>>>>,

    /// Plugin configurations
    configs: Arc<RwLock<HashMap<String, PluginConfig>>>,

    /// Plugin metadata cache
    metadata_cache: Arc<RwLock<HashMap<String, PluginMetadata>>>,

    /// Plugin load order for dependency resolution
    load_order: Arc<RwLock<Vec<String>>>,

    /// Plugin directories
    plugin_directories: Vec<PathBuf>,

    /// Plugin loading statistics
    stats: Arc<RwLock<PluginStats>>,
}

/// Plugin loading and management statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PluginStats {
    /// Total plugins loaded
    pub total_loaded: usize,

    /// Failed plugin loads
    pub failed_loads: usize,

    /// Plugin load times in milliseconds
    pub load_times: HashMap<String, u64>,

    /// Memory usage per plugin (estimated)
    pub memory_usage: HashMap<String, usize>,

    /// Plugin activation counts
    pub activation_counts: HashMap<String, usize>,
}

/// Plugin dependency information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginDependency {
    /// Dependency name
    pub name: String,

    /// Minimum version requirement
    pub min_version: String,

    /// Maximum version requirement (optional)
    pub max_version: Option<String>,

    /// Whether dependency is optional
    pub optional: bool,
}

/// Plugin loading options
#[derive(Debug, Clone, Default)]
pub struct LoadOptions {
    /// Enable debug logging for plugin loading
    pub debug_logging: bool,

    /// Timeout for plugin initialization in milliseconds
    pub init_timeout_ms: Option<u64>,

    /// Load plugins in parallel
    pub parallel_loading: bool,

    /// Skip dependency checks
    pub skip_dependency_checks: bool,

    /// Maximum number of retry attempts
    pub max_retries: usize,
}

impl PluginManager {
    /// Create new plugin manager
    pub fn new() -> Self {
        Self {
            plugins: Arc::new(RwLock::new(HashMap::new())),
            audio_effects: Arc::new(RwLock::new(HashMap::new())),
            voice_effects: Arc::new(RwLock::new(HashMap::new())),
            text_processors: Arc::new(RwLock::new(HashMap::new())),
            configs: Arc::new(RwLock::new(HashMap::new())),
            metadata_cache: Arc::new(RwLock::new(HashMap::new())),
            load_order: Arc::new(RwLock::new(Vec::new())),
            plugin_directories: Vec::new(),
            stats: Arc::new(RwLock::new(PluginStats::default())),
        }
    }

    /// Create plugin manager with specific directories
    pub fn with_directories(directories: Vec<PathBuf>) -> Self {
        let mut manager = Self::new();
        manager.plugin_directories = directories;
        manager
    }

    /// Add plugin directory
    pub fn add_directory(&mut self, directory: impl Into<PathBuf>) {
        self.plugin_directories.push(directory.into());
    }

    /// Register an audio effect plugin with the manager
    pub async fn register_audio_effect(
        &self,
        name: String,
        plugin: Arc<dyn AudioEffect>,
        config: Option<PluginConfig>,
    ) -> Result<()> {
        // First register as a general plugin
        self.register_plugin_internal(name.clone(), plugin.clone(), config.clone())
            .await?;

        // Then add to audio effects collection
        {
            let mut effects = self.audio_effects.write().unwrap();
            effects.insert(name, plugin);
        }

        Ok(())
    }

    /// Register a voice effect plugin with the manager
    pub async fn register_voice_effect(
        &self,
        name: String,
        plugin: Arc<dyn VoiceEffect>,
        config: Option<PluginConfig>,
    ) -> Result<()> {
        // First register as a general plugin
        self.register_plugin_internal(name.clone(), plugin.clone(), config.clone())
            .await?;

        // Then add to voice effects collection
        {
            let mut effects = self.voice_effects.write().unwrap();
            effects.insert(name, plugin);
        }

        Ok(())
    }

    /// Register a text processor plugin with the manager
    pub async fn register_text_processor(
        &self,
        name: String,
        plugin: Arc<dyn TextProcessor>,
        config: Option<PluginConfig>,
    ) -> Result<()> {
        // First register as a general plugin
        self.register_plugin_internal(name.clone(), plugin.clone(), config.clone())
            .await?;

        // Then add to text processors collection
        {
            let mut processors = self.text_processors.write().unwrap();
            processors.insert(name, plugin);
        }

        Ok(())
    }

    /// Register a plugin with the manager (generic method)
    pub async fn register_plugin(
        &self,
        name: String,
        plugin: Arc<dyn VoirsPlugin>,
        config: Option<PluginConfig>,
    ) -> Result<()> {
        self.register_plugin_internal(name, plugin, config).await
    }

    /// Internal plugin registration logic
    async fn register_plugin_internal(
        &self,
        name: String,
        plugin: Arc<dyn VoirsPlugin>,
        config: Option<PluginConfig>,
    ) -> Result<()> {
        let start_time = Instant::now();

        // Get or create default config
        let config = config.unwrap_or_default();

        // Initialize plugin
        if let Err(e) = plugin.initialize(&config) {
            error!("Failed to initialize plugin '{}': {}", name, e);

            // Update stats
            {
                let mut stats = self.stats.write().unwrap();
                stats.failed_loads += 1;
            }

            return Err(VoirsError::plugin_error(format!(
                "Plugin initialization failed: {e}"
            )));
        }

        // Get metadata and cache it
        let metadata = plugin.metadata();
        {
            let mut cache = self.metadata_cache.write().unwrap();
            cache.insert(name.clone(), metadata.clone());
        }

        // Store plugin in main collection
        {
            let mut plugins = self.plugins.write().unwrap();
            plugins.insert(name.clone(), plugin.clone());
        }

        // Note: Specialized collections are managed by the typed registration methods

        // Store configuration
        {
            let mut configs = self.configs.write().unwrap();
            configs.insert(name.clone(), config);
        }

        // Update load order
        {
            let mut load_order = self.load_order.write().unwrap();
            load_order.push(name.clone());
        }

        // Update statistics
        let load_time = start_time.elapsed().as_millis() as u64;
        {
            let mut stats = self.stats.write().unwrap();
            stats.total_loaded += 1;
            stats.load_times.insert(name.clone(), load_time);
            stats.activation_counts.insert(name.clone(), 0);
        }

        info!(
            "Successfully registered plugin '{}' (v{}) in {}ms",
            name, metadata.version, load_time
        );

        Ok(())
    }

    /// Get a plugin by name
    pub fn get_plugin(&self, name: &str) -> Option<Arc<dyn VoirsPlugin>> {
        let plugins = self.plugins.read().unwrap();
        plugins.get(name).cloned()
    }

    /// Get an audio effect plugin by name
    pub fn get_audio_effect(&self, name: &str) -> Option<Arc<dyn AudioEffect>> {
        let effects = self.audio_effects.read().unwrap();
        effects.get(name).cloned()
    }

    /// Get a voice effect plugin by name
    pub fn get_voice_effect(&self, name: &str) -> Option<Arc<dyn VoiceEffect>> {
        let effects = self.voice_effects.read().unwrap();
        effects.get(name).cloned()
    }

    /// Get a text processor plugin by name
    pub fn get_text_processor(&self, name: &str) -> Option<Arc<dyn TextProcessor>> {
        let processors = self.text_processors.read().unwrap();
        processors.get(name).cloned()
    }

    /// List all registered plugins
    pub fn list_plugins(&self) -> Vec<String> {
        let plugins = self.plugins.read().unwrap();
        plugins.keys().cloned().collect()
    }

    /// List plugins by type
    pub fn list_plugins_by_type(&self, plugin_type: PluginType) -> Vec<String> {
        let metadata_cache = self.metadata_cache.read().unwrap();
        metadata_cache
            .iter()
            .filter(|(_, metadata)| metadata.plugin_type == plugin_type)
            .map(|(name, _)| name.clone())
            .collect()
    }

    /// Get plugin metadata
    pub fn get_plugin_metadata(&self, name: &str) -> Option<PluginMetadata> {
        let cache = self.metadata_cache.read().unwrap();
        cache.get(name).cloned()
    }

    /// Get plugin configuration
    pub fn get_plugin_config(&self, name: &str) -> Option<PluginConfig> {
        let configs = self.configs.read().unwrap();
        configs.get(name).cloned()
    }

    /// Update plugin configuration
    pub async fn update_plugin_config(&self, name: &str, config: PluginConfig) -> Result<()> {
        // Get plugin and reconfigure it
        if let Some(plugin) = self.get_plugin(name) {
            if let Err(e) = plugin.initialize(&config) {
                return Err(VoirsError::plugin_error(format!(
                    "Failed to reconfigure plugin '{name}': {e}"
                )));
            }

            // Update stored configuration
            let mut configs = self.configs.write().unwrap();
            configs.insert(name.to_string(), config);

            info!("Updated configuration for plugin '{}'", name);
            Ok(())
        } else {
            Err(VoirsError::plugin_error(format!(
                "Plugin '{name}' not found"
            )))
        }
    }

    /// Unregister a plugin
    pub async fn unregister_plugin(&self, name: &str) -> Result<()> {
        // Remove from all collections
        let plugin = {
            let mut plugins = self.plugins.write().unwrap();
            plugins.remove(name)
        };

        if let Some(plugin) = plugin {
            // Shutdown plugin
            if let Err(e) = plugin.shutdown() {
                warn!("Error during plugin '{}' shutdown: {}", name, e);
            }

            // Remove from specialized collections
            {
                let mut effects = self.audio_effects.write().unwrap();
                effects.remove(name);
            }
            {
                let mut effects = self.voice_effects.write().unwrap();
                effects.remove(name);
            }
            {
                let mut processors = self.text_processors.write().unwrap();
                processors.remove(name);
            }

            // Remove from metadata cache
            {
                let mut cache = self.metadata_cache.write().unwrap();
                cache.remove(name);
            }

            // Remove from configurations
            {
                let mut configs = self.configs.write().unwrap();
                configs.remove(name);
            }

            // Update load order
            {
                let mut load_order = self.load_order.write().unwrap();
                load_order.retain(|n| n != name);
            }

            // Update statistics
            {
                let mut stats = self.stats.write().unwrap();
                stats.total_loaded = stats.total_loaded.saturating_sub(1);
                stats.load_times.remove(name);
                stats.memory_usage.remove(name);
                stats.activation_counts.remove(name);
            }

            info!("Unregistered plugin '{}'", name);
            Ok(())
        } else {
            Err(VoirsError::plugin_error(format!(
                "Plugin '{name}' not found"
            )))
        }
    }

    /// Enable a plugin
    pub async fn enable_plugin(&self, name: &str) -> Result<()> {
        if let Some(mut config) = self.get_plugin_config(name) {
            config.enabled = true;
            self.update_plugin_config(name, config).await
        } else {
            Err(VoirsError::plugin_error(format!(
                "Plugin '{name}' not found"
            )))
        }
    }

    /// Disable a plugin
    pub async fn disable_plugin(&self, name: &str) -> Result<()> {
        if let Some(mut config) = self.get_plugin_config(name) {
            config.enabled = false;
            self.update_plugin_config(name, config).await
        } else {
            Err(VoirsError::plugin_error(format!(
                "Plugin '{name}' not found"
            )))
        }
    }

    /// Check if plugin is enabled
    pub fn is_plugin_enabled(&self, name: &str) -> bool {
        self.get_plugin_config(name)
            .map(|config| config.enabled)
            .unwrap_or(false)
    }

    /// Get plugin statistics
    pub fn get_stats(&self) -> PluginStats {
        let stats = self.stats.read().unwrap();
        stats.clone()
    }

    /// Get enabled plugins in load order
    pub fn get_enabled_plugins_ordered(&self) -> Vec<String> {
        let load_order = self.load_order.read().unwrap();
        load_order
            .iter()
            .filter(|name| self.is_plugin_enabled(name))
            .cloned()
            .collect()
    }

    /// Validate plugin dependencies
    pub fn validate_dependencies(&self, name: &str) -> Result<()> {
        if let Some(metadata) = self.get_plugin_metadata(name) {
            for dep_name in &metadata.dependencies {
                if !self.plugins.read().unwrap().contains_key(dep_name) {
                    return Err(VoirsError::plugin_error(format!(
                        "Plugin '{name}' requires dependency '{dep_name}' which is not loaded"
                    )));
                }
            }
            Ok(())
        } else {
            Err(VoirsError::plugin_error(format!(
                "Plugin '{name}' metadata not found"
            )))
        }
    }

    /// Shutdown all plugins
    pub async fn shutdown_all(&self) -> Result<()> {
        let plugins: Vec<_> = {
            let plugins = self.plugins.read().unwrap();
            plugins
                .iter()
                .map(|(name, plugin)| (name.clone(), plugin.clone()))
                .collect()
        };

        let mut errors = Vec::new();

        for (name, plugin) in plugins {
            if let Err(e) = plugin.shutdown() {
                errors.push(format!("Plugin '{name}': {e}"));
            }
        }

        if !errors.is_empty() {
            Err(VoirsError::plugin_error(format!(
                "Errors during shutdown: {}",
                errors.join(", ")
            )))
        } else {
            info!("All plugins shut down successfully");
            Ok(())
        }
    }
}

impl Default for PluginManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test plugin implementation
    struct TestPlugin {
        name: String,
        version: String,
    }

    impl TestPlugin {
        fn new(name: &str, version: &str) -> Self {
            Self {
                name: name.to_string(),
                version: version.to_string(),
            }
        }
    }

    impl VoirsPlugin for TestPlugin {
        fn name(&self) -> &str {
            &self.name
        }

        fn version(&self) -> &str {
            &self.version
        }

        fn description(&self) -> &str {
            "Test plugin for plugin manager"
        }

        fn author(&self) -> &str {
            "VoiRS Test Team"
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    #[tokio::test]
    async fn test_plugin_registration() {
        let manager = PluginManager::new();
        let plugin = Arc::new(TestPlugin::new("test_plugin", "1.0.0"));

        // Register plugin
        manager
            .register_plugin("test_plugin".to_string(), plugin, None)
            .await
            .unwrap();

        // Verify plugin is registered
        assert!(manager.get_plugin("test_plugin").is_some());
        assert_eq!(manager.list_plugins().len(), 1);

        // Check metadata
        let metadata = manager.get_plugin_metadata("test_plugin").unwrap();
        assert_eq!(metadata.name, "test_plugin");
        assert_eq!(metadata.version, "1.0.0");
    }

    #[tokio::test]
    async fn test_plugin_unregistration() {
        let manager = PluginManager::new();
        let plugin = Arc::new(TestPlugin::new("test_plugin", "1.0.0"));

        // Register and then unregister plugin
        manager
            .register_plugin("test_plugin".to_string(), plugin, None)
            .await
            .unwrap();

        manager.unregister_plugin("test_plugin").await.unwrap();

        // Verify plugin is unregistered
        assert!(manager.get_plugin("test_plugin").is_none());
        assert_eq!(manager.list_plugins().len(), 0);
    }

    #[tokio::test]
    async fn test_plugin_enable_disable() {
        let manager = PluginManager::new();
        let plugin = Arc::new(TestPlugin::new("test_plugin", "1.0.0"));

        // Register plugin
        manager
            .register_plugin("test_plugin".to_string(), plugin, None)
            .await
            .unwrap();

        // Plugin should be enabled by default
        assert!(manager.is_plugin_enabled("test_plugin"));

        // Disable plugin
        manager.disable_plugin("test_plugin").await.unwrap();
        assert!(!manager.is_plugin_enabled("test_plugin"));

        // Re-enable plugin
        manager.enable_plugin("test_plugin").await.unwrap();
        assert!(manager.is_plugin_enabled("test_plugin"));
    }

    #[tokio::test]
    async fn test_plugin_stats() {
        let manager = PluginManager::new();
        let plugin = Arc::new(TestPlugin::new("test_plugin", "1.0.0"));

        // Check initial stats
        let stats = manager.get_stats();
        assert_eq!(stats.total_loaded, 0);

        // Register plugin
        manager
            .register_plugin("test_plugin".to_string(), plugin, None)
            .await
            .unwrap();

        // Check updated stats
        let stats = manager.get_stats();
        assert_eq!(stats.total_loaded, 1);
        assert!(stats.load_times.contains_key("test_plugin"));
    }

    #[tokio::test]
    async fn test_typed_plugin_registration() {
        use crate::audio::AudioBuffer;
        use crate::plugins::{AudioEffect, ParameterDefinition, ParameterValue};
        use async_trait::async_trait;
        use std::collections::HashMap;

        // Create a test audio effect
        struct TestAudioEffect;

        impl VoirsPlugin for TestAudioEffect {
            fn name(&self) -> &str {
                "test_audio_effect"
            }
            fn version(&self) -> &str {
                "1.0.0"
            }
            fn description(&self) -> &str {
                "Test audio effect"
            }
            fn author(&self) -> &str {
                "Test"
            }
            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
        }

        #[async_trait]
        impl AudioEffect for TestAudioEffect {
            async fn process_audio(
                &self,
                audio: &AudioBuffer,
            ) -> crate::error::Result<AudioBuffer> {
                Ok(audio.clone())
            }

            fn get_parameters(&self) -> HashMap<String, ParameterValue> {
                HashMap::new()
            }

            fn set_parameter(
                &self,
                _name: &str,
                _value: ParameterValue,
            ) -> crate::error::Result<()> {
                Ok(())
            }

            fn get_parameter_definition(&self, _name: &str) -> Option<ParameterDefinition> {
                None
            }
        }

        let manager = PluginManager::new();
        let audio_effect = Arc::new(TestAudioEffect);

        // Test typed registration
        manager
            .register_audio_effect("test_effect".to_string(), audio_effect, None)
            .await
            .unwrap();

        // Verify the plugin is in both collections
        assert!(manager.get_plugin("test_effect").is_some());
        assert!(manager.get_audio_effect("test_effect").is_some());
    }
}

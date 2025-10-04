//! Plugin Architecture for Custom Emotion Models and Processing Extensions
//!
//! This module provides a comprehensive plugin system that allows developers to extend
//! the emotion processing capabilities through custom emotion models, processors,
//! and audio effects. The plugin architecture is designed to be flexible, type-safe,
//! and easy to use while maintaining high performance.
//!
//! ## Key Features
//!
//! - **Custom Emotion Models**: Define entirely new emotion types and behaviors
//! - **Processing Plugins**: Add custom audio processing and effects
//! - **Hook System**: Integrate custom logic at various processing stages
//! - **Dynamic Loading**: Load plugins at runtime (with feature flag)
//! - **Type Safety**: Full compile-time safety with trait-based design
//! - **Performance**: Minimal overhead plugin execution
//!
//! ## Example Usage
//!
//! ```rust
//! use voirs_emotion::plugins::*;
//! use voirs_emotion::types::*;
//! use std::any::Any;
//!
//! // Create a custom emotion model
//! #[derive(Debug, Clone)]
//! struct AnxietyModel;
//!
//! impl Plugin for AnxietyModel {
//!     fn metadata(&self) -> &PluginMetadata {
//!         static METADATA: std::sync::OnceLock<PluginMetadata> = std::sync::OnceLock::new();
//!         METADATA.get_or_init(|| PluginMetadata {
//!             name: "anxiety_v1".to_string(),
//!             version: "1.0.0".to_string(),
//!             description: "Anxiety emotion model".to_string(),
//!             author: "Example Author".to_string(),
//!             api_version: "1.0".to_string(),
//!             dependencies: vec![],
//!             tags: vec!["emotion".to_string(), "model".to_string()],
//!         })
//!     }
//!     fn initialize(&mut self, _config: &PluginConfig) -> PluginResult<()> { Ok(()) }
//!     fn is_enabled(&self) -> bool { true }
//!     fn shutdown(&mut self) -> PluginResult<()> { Ok(()) }
//!     fn as_any(&self) -> &dyn Any { self }
//!     fn as_any_mut(&mut self) -> &mut dyn Any { self }
//! }
//!
//! impl EmotionModel for AnxietyModel {
//!     fn model_name(&self) -> &str { "anxiety_v1" }
//!     fn model_version(&self) -> &str { "1.0.0" }
//!     
//!     fn compute_emotion(&self, _params: &EmotionParameters) -> PluginResult<EmotionVector> {
//!         // Custom anxiety computation logic
//!         let mut emotion = EmotionVector::new();
//!         emotion.add_emotion(Emotion::Fear, EmotionIntensity::new(0.8));
//!         Ok(emotion)
//!     }
//!     
//!     fn supported_emotions(&self) -> Vec<Emotion> {
//!         vec![Emotion::Fear]
//!     }
//! }
//!
//! // Register and use the plugin
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut manager = PluginManager::new();
//! manager.register_emotion_model(Box::new(AnxietyModel)).unwrap();
//! # Ok(())
//! # }
//! ```

use crate::types::{Emotion, EmotionDimensions, EmotionParameters, EmotionVector};
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Result type for plugin operations
pub type PluginResult<T> = std::result::Result<T, PluginError>;

/// Errors that can occur during plugin operations
#[derive(Debug, thiserror::Error)]
pub enum PluginError {
    /// Plugin not found
    #[error("Plugin not found: {name}")]
    NotFound {
        /// Name of the missing plugin
        name: String,
    },

    /// Plugin already registered
    #[error("Plugin already registered: {name} version {version}")]
    AlreadyRegistered {
        /// Name of the plugin
        name: String,
        /// Version string of the plugin
        version: String,
    },

    /// Plugin initialization failed
    #[error("Plugin initialization failed: {reason}")]
    InitializationFailed {
        /// Reason for initialization failure
        reason: String,
    },

    /// Plugin execution error
    #[error("Plugin execution error in {plugin}: {reason}")]
    ExecutionError {
        /// Name of the plugin that failed
        plugin: String,
        /// Reason for execution failure
        reason: String,
    },

    /// Invalid plugin configuration
    #[error("Invalid plugin configuration: {reason}")]
    InvalidConfiguration {
        /// Reason for configuration invalidity
        reason: String,
    },

    /// Plugin API version mismatch
    #[error("Plugin API version mismatch: expected {expected}, got {actual}")]
    VersionMismatch {
        /// Expected API version
        expected: String,
        /// Actual API version found
        actual: String,
    },

    /// Plugin dependency missing
    #[error("Plugin dependency missing: {dependency}")]
    DependencyMissing {
        /// Name of the missing dependency
        dependency: String,
    },
}

/// Plugin metadata information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginMetadata {
    /// Plugin name
    pub name: String,
    /// Plugin version
    pub version: String,
    /// Plugin description
    pub description: String,
    /// Plugin author
    pub author: String,
    /// API version compatibility
    pub api_version: String,
    /// Plugin dependencies
    pub dependencies: Vec<String>,
    /// Plugin tags/categories
    pub tags: Vec<String>,
}

/// Plugin initialization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginConfig {
    /// Whether the plugin should be enabled by default
    pub enabled: bool,
    /// Plugin-specific configuration parameters
    pub parameters: HashMap<String, serde_json::Value>,
    /// Priority for plugin execution (higher = earlier)
    pub priority: i32,
    /// Maximum execution time in milliseconds
    pub timeout_ms: u64,
}

impl Default for PluginConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            parameters: HashMap::new(),
            priority: 0,
            timeout_ms: 1000, // 1 second default timeout
        }
    }
}

/// Core trait for all plugins
pub trait Plugin: Send + Sync {
    /// Get plugin metadata
    fn metadata(&self) -> &PluginMetadata;

    /// Initialize the plugin with configuration
    fn initialize(&mut self, config: &PluginConfig) -> PluginResult<()>;

    /// Check if plugin is enabled and ready
    fn is_enabled(&self) -> bool;

    /// Shutdown the plugin and cleanup resources
    fn shutdown(&mut self) -> PluginResult<()>;

    /// Get plugin as Any for downcasting
    fn as_any(&self) -> &dyn Any;

    /// Get mutable plugin as Any for downcasting
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Trait for custom emotion models
pub trait EmotionModel: Plugin {
    /// Get the model name (must be unique)
    fn model_name(&self) -> &str;

    /// Get the model version
    fn model_version(&self) -> &str;

    /// Compute emotion vector from parameters
    fn compute_emotion(&self, params: &EmotionParameters) -> PluginResult<EmotionVector>;

    /// Get supported emotions for this model
    fn supported_emotions(&self) -> Vec<Emotion>;

    /// Validate emotion parameters for this model
    fn validate_parameters(&self, params: &EmotionParameters) -> PluginResult<()> {
        // Default implementation - accept all parameters
        Ok(())
    }

    /// Update model with training data (optional)
    fn update_model(
        &mut self,
        training_data: &[(EmotionParameters, EmotionVector)],
    ) -> PluginResult<()> {
        // Default implementation - no learning
        Ok(())
    }
}

/// Audio processing plugin trait
pub trait AudioProcessor: Plugin {
    /// Process audio data with emotion parameters
    fn process_audio(
        &self,
        audio_data: &mut [f32],
        emotion: &EmotionVector,
        sample_rate: u32,
    ) -> PluginResult<()>;

    /// Get the processing latency in samples
    fn get_latency(&self) -> u32 {
        0
    }

    /// Get required buffer size
    fn get_buffer_size(&self) -> Option<usize> {
        None
    }

    /// Check if processor supports in-place processing
    fn supports_in_place(&self) -> bool {
        true
    }
}

/// Emotion analysis plugin trait
pub trait EmotionAnalyzer: Plugin {
    /// Analyze text and extract emotion parameters
    fn analyze_text(&self, text: &str) -> PluginResult<EmotionParameters>;

    /// Analyze audio and extract emotion features
    fn analyze_audio(
        &self,
        audio_data: &[f32],
        sample_rate: u32,
    ) -> PluginResult<EmotionParameters>;

    /// Get confidence score for analysis
    fn get_confidence(&self, analysis: &EmotionParameters) -> f32;
}

/// Hook system for custom processing stages
pub trait ProcessingHook: Plugin {
    /// Called before emotion processing
    fn pre_process(&self, emotion: &mut EmotionVector) -> PluginResult<()> {
        Ok(())
    }

    /// Called after emotion processing
    fn post_process(&self, emotion: &mut EmotionVector) -> PluginResult<()> {
        Ok(())
    }

    /// Called during emotion transitions
    fn on_transition(&self, from: &EmotionVector, to: &EmotionVector) -> PluginResult<()> {
        Ok(())
    }

    /// Called when audio processing starts
    fn on_audio_start(&self, sample_rate: u32, buffer_size: usize) -> PluginResult<()> {
        Ok(())
    }

    /// Called when audio processing ends
    fn on_audio_end(&self) -> PluginResult<()> {
        Ok(())
    }
}

/// Plugin registry for managing all loaded plugins
pub struct PluginRegistry {
    /// Registered emotion models
    emotion_models: HashMap<String, Box<dyn EmotionModel>>,
    /// Registered audio processors
    audio_processors: HashMap<String, Box<dyn AudioProcessor>>,
    /// Registered emotion analyzers
    emotion_analyzers: HashMap<String, Box<dyn EmotionAnalyzer>>,
    /// Registered processing hooks
    processing_hooks: Vec<Box<dyn ProcessingHook>>,
    /// Plugin configurations
    configs: HashMap<String, PluginConfig>,
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl PluginRegistry {
    /// Create a new plugin registry
    pub fn new() -> Self {
        Self {
            emotion_models: HashMap::new(),
            audio_processors: HashMap::new(),
            emotion_analyzers: HashMap::new(),
            processing_hooks: Vec::new(),
            configs: HashMap::new(),
        }
    }

    /// Register an emotion model plugin
    pub fn register_emotion_model(&mut self, mut model: Box<dyn EmotionModel>) -> PluginResult<()> {
        let name = model.model_name().to_string();
        let version = model.model_version().to_string();

        if self.emotion_models.contains_key(&name) {
            return Err(PluginError::AlreadyRegistered { name, version });
        }

        // Initialize with default config if not present
        let config = self.configs.get(&name).cloned().unwrap_or_default();
        model.initialize(&config)?;

        self.emotion_models.insert(name, model);
        Ok(())
    }

    /// Register an audio processor plugin
    pub fn register_audio_processor(
        &mut self,
        mut processor: Box<dyn AudioProcessor>,
    ) -> PluginResult<()> {
        let name = processor.metadata().name.clone();

        if self.audio_processors.contains_key(&name) {
            return Err(PluginError::AlreadyRegistered {
                name: name.clone(),
                version: processor.metadata().version.clone(),
            });
        }

        let config = self.configs.get(&name).cloned().unwrap_or_default();
        processor.initialize(&config)?;

        self.audio_processors.insert(name, processor);
        Ok(())
    }

    /// Register an emotion analyzer plugin
    pub fn register_emotion_analyzer(
        &mut self,
        mut analyzer: Box<dyn EmotionAnalyzer>,
    ) -> PluginResult<()> {
        let name = analyzer.metadata().name.clone();

        if self.emotion_analyzers.contains_key(&name) {
            return Err(PluginError::AlreadyRegistered {
                name: name.clone(),
                version: analyzer.metadata().version.clone(),
            });
        }

        let config = self.configs.get(&name).cloned().unwrap_or_default();
        analyzer.initialize(&config)?;

        self.emotion_analyzers.insert(name, analyzer);
        Ok(())
    }

    /// Register a processing hook plugin
    pub fn register_processing_hook(
        &mut self,
        mut hook: Box<dyn ProcessingHook>,
    ) -> PluginResult<()> {
        let config = self
            .configs
            .get(&hook.metadata().name)
            .cloned()
            .unwrap_or_default();
        hook.initialize(&config)?;

        self.processing_hooks.push(hook);
        // Sort by priority (higher priority first)
        self.processing_hooks.sort_by(|a, b| {
            let a_priority = self
                .configs
                .get(&a.metadata().name)
                .map(|c| c.priority)
                .unwrap_or(0);
            let b_priority = self
                .configs
                .get(&b.metadata().name)
                .map(|c| c.priority)
                .unwrap_or(0);
            b_priority.cmp(&a_priority)
        });

        Ok(())
    }

    /// Get emotion model by name
    pub fn get_emotion_model(&self, name: &str) -> Option<&dyn EmotionModel> {
        self.emotion_models.get(name).map(|m| m.as_ref())
    }

    /// Get mutable emotion model by name
    pub fn get_emotion_model_mut(&mut self, name: &str) -> Option<&mut (dyn EmotionModel + '_)> {
        match self.emotion_models.get_mut(name) {
            Some(model) => Some(model.as_mut()),
            None => None,
        }
    }

    /// Get audio processor by name
    pub fn get_audio_processor(&self, name: &str) -> Option<&dyn AudioProcessor> {
        self.audio_processors.get(name).map(|p| p.as_ref())
    }

    /// Get emotion analyzer by name
    pub fn get_emotion_analyzer(&self, name: &str) -> Option<&dyn EmotionAnalyzer> {
        self.emotion_analyzers.get(name).map(|a| a.as_ref())
    }

    /// Get all processing hooks
    pub fn get_processing_hooks(&self) -> &[Box<dyn ProcessingHook>] {
        &self.processing_hooks
    }

    /// List all registered plugins
    pub fn list_plugins(&self) -> Vec<&PluginMetadata> {
        let mut plugins = Vec::new();

        for model in self.emotion_models.values() {
            plugins.push(model.metadata());
        }
        for processor in self.audio_processors.values() {
            plugins.push(processor.metadata());
        }
        for analyzer in self.emotion_analyzers.values() {
            plugins.push(analyzer.metadata());
        }
        for hook in &self.processing_hooks {
            plugins.push(hook.metadata());
        }

        plugins
    }

    /// Set plugin configuration
    pub fn set_plugin_config(&mut self, plugin_name: &str, config: PluginConfig) {
        self.configs.insert(plugin_name.to_string(), config);
    }

    /// Get plugin configuration
    pub fn get_plugin_config(&self, plugin_name: &str) -> Option<&PluginConfig> {
        self.configs.get(plugin_name)
    }

    /// Shutdown all plugins
    pub fn shutdown_all(&mut self) -> PluginResult<()> {
        let mut errors = Vec::new();

        for (name, model) in &mut self.emotion_models {
            if let Err(e) = model.shutdown() {
                errors.push(format!("Emotion model '{}': {}", name, e));
            }
        }

        for (name, processor) in &mut self.audio_processors {
            if let Err(e) = processor.shutdown() {
                errors.push(format!("Audio processor '{}': {}", name, e));
            }
        }

        for (name, analyzer) in &mut self.emotion_analyzers {
            if let Err(e) = analyzer.shutdown() {
                errors.push(format!("Emotion analyzer '{}': {}", name, e));
            }
        }

        for hook in &mut self.processing_hooks {
            if let Err(e) = hook.shutdown() {
                errors.push(format!("Processing hook '{}': {}", hook.metadata().name, e));
            }
        }

        if !errors.is_empty() {
            return Err(PluginError::ExecutionError {
                plugin: "shutdown".to_string(),
                reason: errors.join("; "),
            });
        }

        Ok(())
    }
}

/// Thread-safe plugin manager
pub struct PluginManager {
    registry: Arc<RwLock<PluginRegistry>>,
}

impl Default for PluginManager {
    fn default() -> Self {
        Self::new()
    }
}

impl PluginManager {
    /// Create a new plugin manager
    pub fn new() -> Self {
        Self {
            registry: Arc::new(RwLock::new(PluginRegistry::new())),
        }
    }

    /// Register an emotion model plugin
    pub fn register_emotion_model(&self, model: Box<dyn EmotionModel>) -> PluginResult<()> {
        self.registry
            .write()
            .map_err(|_| PluginError::ExecutionError {
                plugin: "manager".to_string(),
                reason: "Failed to acquire write lock".to_string(),
            })?
            .register_emotion_model(model)
    }

    /// Register an audio processor plugin
    pub fn register_audio_processor(&self, processor: Box<dyn AudioProcessor>) -> PluginResult<()> {
        self.registry
            .write()
            .map_err(|_| PluginError::ExecutionError {
                plugin: "manager".to_string(),
                reason: "Failed to acquire write lock".to_string(),
            })?
            .register_audio_processor(processor)
    }

    /// Register an emotion analyzer plugin
    pub fn register_emotion_analyzer(
        &self,
        analyzer: Box<dyn EmotionAnalyzer>,
    ) -> PluginResult<()> {
        self.registry
            .write()
            .map_err(|_| PluginError::ExecutionError {
                plugin: "manager".to_string(),
                reason: "Failed to acquire write lock".to_string(),
            })?
            .register_emotion_analyzer(analyzer)
    }

    /// Register a processing hook plugin
    pub fn register_processing_hook(&self, hook: Box<dyn ProcessingHook>) -> PluginResult<()> {
        self.registry
            .write()
            .map_err(|_| PluginError::ExecutionError {
                plugin: "manager".to_string(),
                reason: "Failed to acquire write lock".to_string(),
            })?
            .register_processing_hook(hook)
    }

    /// Execute emotion computation using a plugin
    pub fn compute_emotion(
        &self,
        model_name: &str,
        params: &EmotionParameters,
    ) -> PluginResult<EmotionVector> {
        let registry = self
            .registry
            .read()
            .map_err(|_| PluginError::ExecutionError {
                plugin: "manager".to_string(),
                reason: "Failed to acquire read lock".to_string(),
            })?;

        let model =
            registry
                .get_emotion_model(model_name)
                .ok_or_else(|| PluginError::NotFound {
                    name: model_name.to_string(),
                })?;

        model.compute_emotion(params)
    }

    /// Process audio using a plugin
    pub fn process_audio(
        &self,
        processor_name: &str,
        audio_data: &mut [f32],
        emotion: &EmotionVector,
        sample_rate: u32,
    ) -> PluginResult<()> {
        let registry = self
            .registry
            .read()
            .map_err(|_| PluginError::ExecutionError {
                plugin: "manager".to_string(),
                reason: "Failed to acquire read lock".to_string(),
            })?;

        let processor = registry
            .get_audio_processor(processor_name)
            .ok_or_else(|| PluginError::NotFound {
                name: processor_name.to_string(),
            })?;

        processor.process_audio(audio_data, emotion, sample_rate)
    }

    /// Analyze text using a plugin
    pub fn analyze_text(&self, analyzer_name: &str, text: &str) -> PluginResult<EmotionParameters> {
        let registry = self
            .registry
            .read()
            .map_err(|_| PluginError::ExecutionError {
                plugin: "manager".to_string(),
                reason: "Failed to acquire read lock".to_string(),
            })?;

        let analyzer = registry
            .get_emotion_analyzer(analyzer_name)
            .ok_or_else(|| PluginError::NotFound {
                name: analyzer_name.to_string(),
            })?;

        analyzer.analyze_text(text)
    }

    /// Execute pre-processing hooks
    pub fn execute_pre_process_hooks(&self, emotion: &mut EmotionVector) -> PluginResult<()> {
        let registry = self
            .registry
            .read()
            .map_err(|_| PluginError::ExecutionError {
                plugin: "manager".to_string(),
                reason: "Failed to acquire read lock".to_string(),
            })?;

        for hook in registry.get_processing_hooks() {
            if hook.is_enabled() {
                hook.pre_process(emotion)?;
            }
        }

        Ok(())
    }

    /// Execute post-processing hooks
    pub fn execute_post_process_hooks(&self, emotion: &mut EmotionVector) -> PluginResult<()> {
        let registry = self
            .registry
            .read()
            .map_err(|_| PluginError::ExecutionError {
                plugin: "manager".to_string(),
                reason: "Failed to acquire read lock".to_string(),
            })?;

        for hook in registry.get_processing_hooks() {
            if hook.is_enabled() {
                hook.post_process(emotion)?;
            }
        }

        Ok(())
    }

    /// List all registered plugins
    pub fn list_plugins(&self) -> PluginResult<Vec<PluginMetadata>> {
        let registry = self
            .registry
            .read()
            .map_err(|_| PluginError::ExecutionError {
                plugin: "manager".to_string(),
                reason: "Failed to acquire read lock".to_string(),
            })?;

        Ok(registry.list_plugins().into_iter().cloned().collect())
    }

    /// Set plugin configuration
    pub fn set_plugin_config(&self, plugin_name: &str, config: PluginConfig) -> PluginResult<()> {
        self.registry
            .write()
            .map_err(|_| PluginError::ExecutionError {
                plugin: "manager".to_string(),
                reason: "Failed to acquire write lock".to_string(),
            })?
            .set_plugin_config(plugin_name, config);
        Ok(())
    }

    /// Get plugin configuration
    pub fn get_plugin_config(&self, plugin_name: &str) -> PluginResult<Option<PluginConfig>> {
        let registry = self
            .registry
            .read()
            .map_err(|_| PluginError::ExecutionError {
                plugin: "manager".to_string(),
                reason: "Failed to acquire read lock".to_string(),
            })?;
        Ok(registry.get_plugin_config(plugin_name).cloned())
    }

    /// Shutdown all plugins
    pub fn shutdown_all(&self) -> PluginResult<()> {
        self.registry
            .write()
            .map_err(|_| PluginError::ExecutionError {
                plugin: "manager".to_string(),
                reason: "Failed to acquire write lock".to_string(),
            })?
            .shutdown_all()
    }
}

/// Helper macro for creating plugin metadata
#[macro_export]
macro_rules! plugin_metadata {
    ($name:expr, $version:expr, $description:expr, $author:expr) => {
        $crate::plugins::PluginMetadata {
            name: $name.to_string(),
            version: $version.to_string(),
            description: $description.to_string(),
            author: $author.to_string(),
            api_version: "1.0.0".to_string(),
            dependencies: Vec::new(),
            tags: Vec::new(),
        }
    };
    ($name:expr, $version:expr, $description:expr, $author:expr, $($tag:expr),*) => {
        $crate::plugins::PluginMetadata {
            name: $name.to_string(),
            version: $version.to_string(),
            description: $description.to_string(),
            author: $author.to_string(),
            api_version: "1.0.0".to_string(),
            dependencies: Vec::new(),
            tags: vec![$($tag.to_string()),*],
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::EmotionIntensity;

    // Mock emotion model for testing
    #[derive(Debug)]
    struct MockEmotionModel {
        metadata: PluginMetadata,
        enabled: bool,
    }

    impl MockEmotionModel {
        fn new() -> Self {
            Self {
                metadata: plugin_metadata!(
                    "mock_model",
                    "1.0.0",
                    "Mock emotion model",
                    "Test Author"
                ),
                enabled: false,
            }
        }
    }

    impl Plugin for MockEmotionModel {
        fn metadata(&self) -> &PluginMetadata {
            &self.metadata
        }

        fn initialize(&mut self, _config: &PluginConfig) -> PluginResult<()> {
            self.enabled = true;
            Ok(())
        }

        fn is_enabled(&self) -> bool {
            self.enabled
        }

        fn shutdown(&mut self) -> PluginResult<()> {
            self.enabled = false;
            Ok(())
        }

        fn as_any(&self) -> &dyn Any {
            self
        }

        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
    }

    impl EmotionModel for MockEmotionModel {
        fn model_name(&self) -> &str {
            "mock_model"
        }

        fn model_version(&self) -> &str {
            "1.0.0"
        }

        fn compute_emotion(&self, _params: &EmotionParameters) -> PluginResult<EmotionVector> {
            let mut emotion = EmotionVector::new();
            emotion.add_emotion(Emotion::Happy, EmotionIntensity::MEDIUM);
            Ok(emotion)
        }

        fn supported_emotions(&self) -> Vec<Emotion> {
            vec![Emotion::Happy, Emotion::Sad]
        }
    }

    // Mock audio processor for testing
    #[derive(Debug)]
    struct MockAudioProcessor {
        metadata: PluginMetadata,
        enabled: bool,
    }

    impl MockAudioProcessor {
        fn new() -> Self {
            Self {
                metadata: plugin_metadata!(
                    "mock_processor",
                    "1.0.0",
                    "Mock audio processor",
                    "Test Author"
                ),
                enabled: false,
            }
        }
    }

    impl Plugin for MockAudioProcessor {
        fn metadata(&self) -> &PluginMetadata {
            &self.metadata
        }

        fn initialize(&mut self, _config: &PluginConfig) -> PluginResult<()> {
            self.enabled = true;
            Ok(())
        }

        fn is_enabled(&self) -> bool {
            self.enabled
        }

        fn shutdown(&mut self) -> PluginResult<()> {
            self.enabled = false;
            Ok(())
        }

        fn as_any(&self) -> &dyn Any {
            self
        }

        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
    }

    impl AudioProcessor for MockAudioProcessor {
        fn process_audio(
            &self,
            audio_data: &mut [f32],
            _emotion: &EmotionVector,
            _sample_rate: u32,
        ) -> PluginResult<()> {
            // Simple gain effect for testing
            for sample in audio_data.iter_mut() {
                *sample *= 0.8;
            }
            Ok(())
        }
    }

    #[test]
    fn test_plugin_registry_creation() {
        let registry = PluginRegistry::new();
        assert_eq!(registry.list_plugins().len(), 0);
    }

    #[test]
    fn test_emotion_model_registration() {
        let mut registry = PluginRegistry::new();
        let model = Box::new(MockEmotionModel::new());

        assert!(registry.register_emotion_model(model).is_ok());
        assert!(registry.get_emotion_model("mock_model").is_some());
    }

    #[test]
    fn test_emotion_model_mutable_access() {
        let mut registry = PluginRegistry::new();
        let model = Box::new(MockEmotionModel::new());

        // Register model
        assert!(registry.register_emotion_model(model).is_ok());

        // Test mutable access
        assert!(registry.get_emotion_model_mut("mock_model").is_some());

        // Test non-existent model returns None
        assert!(registry.get_emotion_model_mut("non_existent").is_none());

        // Test that we can actually use the mutable reference to compute emotion
        if let Some(model_mut) = registry.get_emotion_model_mut("mock_model") {
            let params = crate::types::EmotionParameters::default();
            let result = model_mut.compute_emotion(&params);
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_duplicate_registration_error() {
        let mut registry = PluginRegistry::new();
        let model1 = Box::new(MockEmotionModel::new());
        let model2 = Box::new(MockEmotionModel::new());

        assert!(registry.register_emotion_model(model1).is_ok());
        assert!(registry.register_emotion_model(model2).is_err());
    }

    #[test]
    fn test_audio_processor_registration() {
        let mut registry = PluginRegistry::new();
        let processor = Box::new(MockAudioProcessor::new());

        assert!(registry.register_audio_processor(processor).is_ok());
        assert!(registry.get_audio_processor("mock_processor").is_some());
    }

    #[test]
    fn test_emotion_computation() {
        let mut registry = PluginRegistry::new();
        let model = Box::new(MockEmotionModel::new());

        registry.register_emotion_model(model).unwrap();

        let model = registry.get_emotion_model("mock_model").unwrap();
        let params = EmotionParameters::neutral();
        let result = model.compute_emotion(&params);

        assert!(result.is_ok());
        let emotion = result.unwrap();
        assert_eq!(
            emotion.dominant_emotion(),
            Some((Emotion::Happy, EmotionIntensity::MEDIUM))
        );
    }

    #[test]
    fn test_audio_processing() {
        let mut registry = PluginRegistry::new();
        let processor = Box::new(MockAudioProcessor::new());

        registry.register_audio_processor(processor).unwrap();

        let processor = registry.get_audio_processor("mock_processor").unwrap();
        let mut audio_data = vec![1.0, 0.5, -0.5, -1.0];
        let mut emotion = EmotionVector::new();
        emotion.add_emotion(Emotion::Happy, EmotionIntensity::MEDIUM);

        assert!(processor
            .process_audio(&mut audio_data, &emotion, 44100)
            .is_ok());

        // Check that gain was applied
        assert!((audio_data[0] - 0.8).abs() < f32::EPSILON);
        assert!((audio_data[1] - 0.4).abs() < f32::EPSILON);
    }

    #[test]
    fn test_plugin_manager() {
        let manager = PluginManager::new();
        let model = Box::new(MockEmotionModel::new());

        assert!(manager.register_emotion_model(model).is_ok());

        let params = EmotionParameters::neutral();
        let result = manager.compute_emotion("mock_model", &params);

        assert!(result.is_ok());
        let emotion = result.unwrap();
        assert_eq!(
            emotion.dominant_emotion(),
            Some((Emotion::Happy, EmotionIntensity::MEDIUM))
        );
    }

    #[test]
    fn test_plugin_metadata_macro() {
        let metadata = plugin_metadata!(
            "test",
            "1.0.0",
            "Test plugin",
            "Test Author",
            "tag1",
            "tag2"
        );

        assert_eq!(metadata.name, "test");
        assert_eq!(metadata.version, "1.0.0");
        assert_eq!(metadata.description, "Test plugin");
        assert_eq!(metadata.author, "Test Author");
        assert_eq!(metadata.tags, vec!["tag1", "tag2"]);
    }

    #[test]
    fn test_plugin_configuration() {
        let mut registry = PluginRegistry::new();
        let config = PluginConfig {
            enabled: true,
            parameters: {
                let mut params = HashMap::new();
                params.insert(
                    "gain".to_string(),
                    serde_json::Value::Number(serde_json::Number::from_f64(0.8).unwrap()),
                );
                params
            },
            priority: 5,
            timeout_ms: 500,
        };

        registry.set_plugin_config("test_plugin", config.clone());
        let retrieved_config = registry.get_plugin_config("test_plugin");

        assert!(retrieved_config.is_some());
        assert_eq!(retrieved_config.unwrap().priority, 5);
    }

    #[test]
    fn test_plugin_list() {
        let mut registry = PluginRegistry::new();
        let model = Box::new(MockEmotionModel::new());
        let processor = Box::new(MockAudioProcessor::new());

        registry.register_emotion_model(model).unwrap();
        registry.register_audio_processor(processor).unwrap();

        let plugins = registry.list_plugins();
        assert_eq!(plugins.len(), 2);

        let plugin_names: Vec<&str> = plugins.iter().map(|p| p.name.as_str()).collect();
        assert!(plugin_names.contains(&"mock_model"));
        assert!(plugin_names.contains(&"mock_processor"));
    }

    #[test]
    fn test_plugin_shutdown() {
        let mut registry = PluginRegistry::new();
        let model = Box::new(MockEmotionModel::new());

        registry.register_emotion_model(model).unwrap();
        assert!(registry.shutdown_all().is_ok());
    }
}

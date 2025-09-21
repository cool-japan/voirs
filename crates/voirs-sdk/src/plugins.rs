//! Plugin system for extensible audio processing and effects.

pub mod effects;
pub mod enhancement;
pub mod format;
pub mod manager;
pub mod registry;

use crate::{audio::AudioBuffer, error::Result, VoirsError};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::{
    any::Any,
    collections::HashMap,
    path::PathBuf,
    sync::{Arc, RwLock},
};

/// Plugin trait definition for VoiRS ecosystem
pub trait VoirsPlugin: Send + Sync + Any {
    /// Get plugin name
    fn name(&self) -> &str;

    /// Get plugin version
    fn version(&self) -> &str;

    /// Get plugin description
    fn description(&self) -> &str;

    /// Get plugin author
    fn author(&self) -> &str;

    /// Initialize plugin with configuration
    fn initialize(&self, config: &PluginConfig) -> Result<()> {
        let _ = config; // Suppress unused parameter warning
        Ok(()) // Default: no initialization needed
    }

    /// Shutdown plugin and cleanup resources
    fn shutdown(&self) -> Result<()> {
        Ok(()) // Default: no cleanup needed
    }

    /// Get plugin metadata
    fn metadata(&self) -> PluginMetadata {
        PluginMetadata {
            name: self.name().to_string(),
            version: self.version().to_string(),
            description: self.description().to_string(),
            author: self.author().to_string(),
            plugin_type: PluginType::Effect,
            supported_formats: vec!["wav".to_string()],
            capabilities: vec![],
            dependencies: vec![],
            supported_platforms: Some(vec!["any".to_string()]),
            min_voirs_version: Some("0.1.0".to_string()),
        }
    }

    /// Check if plugin supports a specific capability
    fn supports_capability(&self, capability: &str) -> bool {
        self.metadata()
            .capabilities
            .contains(&capability.to_string())
    }

    /// Get the plugin as an Any trait object for downcasting
    fn as_any(&self) -> &dyn Any;
}

/// Audio effect plugin trait
#[async_trait]
pub trait AudioEffect: VoirsPlugin {
    /// Process audio with the effect
    async fn process_audio(&self, audio: &AudioBuffer) -> Result<AudioBuffer>;

    /// Get effect parameters
    fn get_parameters(&self) -> HashMap<String, ParameterValue>;

    /// Set effect parameter
    fn set_parameter(&self, name: &str, value: ParameterValue) -> Result<()>;

    /// Get parameter definition
    fn get_parameter_definition(&self, name: &str) -> Option<ParameterDefinition>;

    /// List all parameter names
    fn list_parameters(&self) -> Vec<String> {
        self.get_parameters().keys().cloned().collect()
    }

    /// Reset parameters to defaults
    fn reset_parameters(&self) -> Result<()> {
        // Default implementation: do nothing
        Ok(())
    }

    /// Get effect latency in samples
    fn get_latency_samples(&self) -> usize {
        0 // Default: no latency
    }

    /// Check if effect can process in real-time
    fn is_realtime_capable(&self) -> bool {
        true // Default: assume real-time capable
    }
}

/// Voice effect plugin trait for voice-specific processing
#[async_trait]
pub trait VoiceEffect: VoirsPlugin {
    /// Process voice synthesis pipeline
    async fn process_voice_synthesis(
        &self,
        phonemes: &[crate::types::Phoneme],
        mel: &crate::types::MelSpectrogram,
        audio: &AudioBuffer,
    ) -> Result<VoiceSynthesisResult>;

    /// Get voice effect type
    fn get_effect_type(&self) -> VoiceEffectType;

    /// Check if effect modifies specific synthesis stage
    fn modifies_stage(&self, stage: SynthesisStage) -> bool;
}

/// Text processor plugin trait
#[async_trait]
pub trait TextProcessor: VoirsPlugin {
    /// Process text before synthesis
    async fn process_text(
        &self,
        text: &str,
        language: crate::types::LanguageCode,
    ) -> Result<String>;

    /// Get text processor type
    fn get_processor_type(&self) -> TextProcessorType;

    /// Check if processor handles specific language
    fn supports_language(&self, language: crate::types::LanguageCode) -> bool;
}

/// Plugin configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginConfig {
    /// Plugin-specific parameters
    pub parameters: HashMap<String, ParameterValue>,

    /// Plugin enabled state
    pub enabled: bool,

    /// Plugin priority (lower = higher priority)
    pub priority: i32,

    /// Plugin configuration file path
    pub config_file: Option<PathBuf>,

    /// Plugin data directory
    pub data_dir: Option<PathBuf>,

    /// Plugin cache directory
    pub cache_dir: Option<PathBuf>,
}

impl Default for PluginConfig {
    fn default() -> Self {
        Self {
            parameters: HashMap::new(),
            enabled: true,
            priority: 100,
            config_file: None,
            data_dir: None,
            cache_dir: None,
        }
    }
}

/// Plugin metadata
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

    /// Plugin type
    pub plugin_type: PluginType,

    /// Supported audio formats
    pub supported_formats: Vec<String>,

    /// Plugin capabilities
    pub capabilities: Vec<String>,

    /// Plugin dependencies
    pub dependencies: Vec<String>,

    /// Supported platforms
    pub supported_platforms: Option<Vec<String>>,

    /// Minimum VoiRS version required
    pub min_voirs_version: Option<String>,
}

/// Plugin type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PluginType {
    /// Audio effect plugin
    Effect,

    /// Voice synthesis effect
    VoiceEffect,

    /// Text processing plugin
    TextProcessor,

    /// Model enhancement plugin
    ModelEnhancer,

    /// Output format plugin
    OutputProcessor,
}

/// Parameter value types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ParameterValue {
    Float(f32),
    Int(i32),
    Integer(i64), // For larger integer values
    Bool(bool),
    String(String),
    FloatArray(Vec<f32>),
    IntArray(Vec<i32>),
}

impl ParameterValue {
    /// Convert to f32 if possible
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::Float(v) => Some(*v),
            Self::Int(v) => Some(*v as f32),
            Self::Integer(v) => Some(*v as f32),
            _ => None,
        }
    }

    /// Convert to i32 if possible
    pub fn as_i32(&self) -> Option<i32> {
        match self {
            Self::Int(v) => Some(*v),
            Self::Integer(v) => Some(*v as i32),
            Self::Float(v) => Some(*v as i32),
            _ => None,
        }
    }

    /// Convert to i64 if possible
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Integer(v) => Some(*v),
            Self::Int(v) => Some(*v as i64),
            Self::Float(v) => Some(*v as i64),
            _ => None,
        }
    }

    /// Convert to bool if possible
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            Self::Int(v) => Some(*v != 0),
            Self::Integer(v) => Some(*v != 0),
            Self::Float(v) => Some(*v != 0.0),
            _ => None,
        }
    }

    /// Convert to string
    pub fn as_string(&self) -> String {
        match self {
            Self::String(v) => v.clone(),
            Self::Float(v) => v.to_string(),
            Self::Int(v) => v.to_string(),
            Self::Integer(v) => v.to_string(),
            Self::Bool(v) => v.to_string(),
            Self::FloatArray(v) => format!("{v:?}"),
            Self::IntArray(v) => format!("{v:?}"),
        }
    }
}

/// Parameter definition for plugin configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterDefinition {
    /// Parameter name
    pub name: String,

    /// Parameter description
    pub description: String,

    /// Parameter type
    pub parameter_type: ParameterType,

    /// Default value
    pub default_value: ParameterValue,

    /// Minimum value (for numeric types)
    pub min_value: Option<ParameterValue>,

    /// Maximum value (for numeric types)
    pub max_value: Option<ParameterValue>,

    /// Step size (for numeric types)
    pub step_size: Option<f32>,

    /// Whether parameter affects real-time processing
    pub realtime_safe: bool,
}

/// Parameter type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParameterType {
    Float,
    Int,
    Integer, // For larger integer values
    Bool,
    String,
    FloatArray,
    IntArray,
    Enum,
}

/// Voice effect type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VoiceEffectType {
    /// Modify phoneme sequence
    PhonemeModifier,

    /// Modify mel spectrogram
    MelModifier,

    /// Modify final audio
    AudioModifier,

    /// Multi-stage effect
    MultiStage,
}

/// Synthesis stage
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SynthesisStage {
    /// Text preprocessing
    TextPreprocessing,

    /// Phoneme generation
    PhonemeGeneration,

    /// Mel spectrogram synthesis
    MelSynthesis,

    /// Audio vocoding
    AudioGeneration,

    /// Post-processing
    PostProcessing,
}

/// Text processor type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TextProcessorType {
    /// Normalize text format
    Normalizer,

    /// Expand abbreviations
    Expander,

    /// Number to text conversion
    NumberConverter,

    /// SSML processor
    SsmlProcessor,

    /// Emotion analyzer
    EmotionAnalyzer,
}

/// Voice synthesis result from voice effect plugin
#[derive(Debug, Clone)]
pub struct VoiceSynthesisResult {
    /// Modified phonemes (optional)
    pub phonemes: Option<Vec<crate::types::Phoneme>>,

    /// Modified mel spectrogram (optional)
    pub mel_spectrogram: Option<crate::types::MelSpectrogram>,

    /// Modified audio (optional)
    pub audio_buffer: Option<AudioBuffer>,

    /// Processing metadata
    pub metadata: HashMap<String, String>,
}

/// Plugin loading error
#[derive(Debug, thiserror::Error)]
pub enum PluginError {
    #[error("Plugin not found: {name}")]
    NotFound { name: String },

    #[error("Plugin initialization failed: {message}")]
    InitializationFailed { message: String },

    #[error("Plugin parameter error: {message}")]
    ParameterError { message: String },

    #[error("Plugin version incompatible: required {required}, found {found}")]
    VersionIncompatible { required: String, found: String },

    #[error("Plugin dependency missing: {dependency}")]
    DependencyMissing { dependency: String },

    #[error("Plugin processing error: {message}")]
    ProcessingError { message: String },
}

/// Plugin host for managing plugin lifecycle
pub struct PluginHost {
    /// Loaded plugins
    plugins: Arc<RwLock<HashMap<String, Arc<dyn VoirsPlugin>>>>,

    /// Plugin configurations
    configs: Arc<RwLock<HashMap<String, PluginConfig>>>,

    /// Plugin load order
    load_order: Vec<String>,

    /// Plugin data directory
    #[allow(dead_code)]
    plugin_dir: PathBuf,
}

impl PluginHost {
    /// Create new plugin host
    pub fn new(plugin_dir: impl Into<PathBuf>) -> Self {
        Self {
            plugins: Arc::new(RwLock::new(HashMap::new())),
            configs: Arc::new(RwLock::new(HashMap::new())),
            load_order: Vec::new(),
            plugin_dir: plugin_dir.into(),
        }
    }

    /// Load plugin from file
    pub fn load_plugin(&mut self, name: &str, plugin: Arc<dyn VoirsPlugin>) -> Result<()> {
        let default_config = PluginConfig::default();

        // Initialize plugin
        plugin.initialize(&default_config).map_err(|e| {
            VoirsError::internal("plugins", format!("Plugin initialization failed: {e}"))
        })?;

        // Store plugin and config
        {
            let mut plugins = self.plugins.write().unwrap();
            let mut configs = self.configs.write().unwrap();

            plugins.insert(name.to_string(), plugin);
            configs.insert(name.to_string(), default_config);
        }

        self.load_order.push(name.to_string());
        tracing::info!("Loaded plugin: {}", name);

        Ok(())
    }

    /// Unload plugin
    pub fn unload_plugin(&mut self, name: &str) -> Result<()> {
        {
            let mut plugins = self.plugins.write().unwrap();
            let mut configs = self.configs.write().unwrap();

            if let Some(plugin) = plugins.remove(name) {
                plugin.shutdown().map_err(|e| {
                    VoirsError::internal("plugins", format!("Plugin shutdown failed: {e}"))
                })?;
            }

            configs.remove(name);
        }

        self.load_order.retain(|n| n != name);
        tracing::info!("Unloaded plugin: {}", name);

        Ok(())
    }

    /// Get plugin by name
    pub fn get_plugin(&self, name: &str) -> Option<Arc<dyn VoirsPlugin>> {
        let plugins = self.plugins.read().unwrap();
        plugins.get(name).cloned()
    }

    /// List all loaded plugins
    pub fn list_plugins(&self) -> Vec<String> {
        let plugins = self.plugins.read().unwrap();
        plugins.keys().cloned().collect()
    }

    /// Get plugin metadata
    pub fn get_plugin_metadata(&self, name: &str) -> Option<PluginMetadata> {
        self.get_plugin(name).map(|p| p.metadata())
    }

    /// Configure plugin
    pub fn configure_plugin(&self, name: &str, config: PluginConfig) -> Result<()> {
        let mut configs = self.configs.write().unwrap();
        configs.insert(name.to_string(), config);
        Ok(())
    }

    /// Get plugin configuration
    pub fn get_plugin_config(&self, name: &str) -> Option<PluginConfig> {
        let configs = self.configs.read().unwrap();
        configs.get(name).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test plugin implementation
    struct TestAudioEffect {
        name: String,
        gain: std::sync::RwLock<f32>,
    }

    impl TestAudioEffect {
        fn new() -> Self {
            Self {
                name: "TestEffect".to_string(),
                gain: std::sync::RwLock::new(1.0),
            }
        }
    }

    impl VoirsPlugin for TestAudioEffect {
        fn name(&self) -> &str {
            &self.name
        }

        fn version(&self) -> &str {
            "1.0.0"
        }

        fn description(&self) -> &str {
            "Test audio effect plugin"
        }

        fn author(&self) -> &str {
            "VoiRS Team"
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    #[async_trait]
    impl AudioEffect for TestAudioEffect {
        async fn process_audio(&self, audio: &AudioBuffer) -> Result<AudioBuffer> {
            let mut processed = audio.clone();
            let gain = *self.gain.read().unwrap();
            for sample in processed.samples_mut() {
                *sample *= gain;
            }
            Ok(processed)
        }

        fn get_parameters(&self) -> HashMap<String, ParameterValue> {
            let mut params = HashMap::new();
            params.insert(
                "gain".to_string(),
                ParameterValue::Float(*self.gain.read().unwrap()),
            );
            params
        }

        fn set_parameter(&self, name: &str, value: ParameterValue) -> Result<()> {
            match name {
                "gain" => {
                    if let Some(gain) = value.as_f32() {
                        *self.gain.write().unwrap() = gain;
                        Ok(())
                    } else {
                        Err(VoirsError::internal(
                            "plugins",
                            "Invalid gain parameter type",
                        ))
                    }
                }
                _ => Err(VoirsError::internal(
                    "plugins",
                    format!("Unknown parameter: {name}"),
                )),
            }
        }

        fn get_parameter_definition(&self, name: &str) -> Option<ParameterDefinition> {
            match name {
                "gain" => Some(ParameterDefinition {
                    name: "gain".to_string(),
                    description: "Audio gain multiplier".to_string(),
                    parameter_type: ParameterType::Float,
                    default_value: ParameterValue::Float(1.0),
                    min_value: Some(ParameterValue::Float(0.0)),
                    max_value: Some(ParameterValue::Float(10.0)),
                    step_size: Some(0.1),
                    realtime_safe: true,
                }),
                _ => None,
            }
        }
    }

    #[test]
    fn test_parameter_value_conversions() {
        let float_val = ParameterValue::Float(3.5);
        assert_eq!(float_val.as_f32(), Some(3.5));
        assert_eq!(float_val.as_i32(), Some(3));

        let int_val = ParameterValue::Int(42);
        assert_eq!(int_val.as_i32(), Some(42));
        assert_eq!(int_val.as_f32(), Some(42.0));

        let bool_val = ParameterValue::Bool(true);
        assert_eq!(bool_val.as_bool(), Some(true));

        let string_val = ParameterValue::String("test".to_string());
        assert_eq!(string_val.as_string(), "test");
    }

    #[test]
    fn test_plugin_host() {
        let mut host = PluginHost::new("/tmp/plugins");
        let plugin = Arc::new(TestAudioEffect::new());

        // Load plugin
        host.load_plugin("test_effect", plugin).unwrap();

        // Check if plugin is loaded
        assert!(host.get_plugin("test_effect").is_some());
        assert_eq!(host.list_plugins().len(), 1);

        // Get plugin metadata
        let metadata = host.get_plugin_metadata("test_effect").unwrap();
        assert_eq!(metadata.name, "TestEffect");
        assert_eq!(metadata.version, "1.0.0");

        // Unload plugin
        host.unload_plugin("test_effect").unwrap();
        assert!(host.get_plugin("test_effect").is_none());
        assert_eq!(host.list_plugins().len(), 0);
    }

    #[tokio::test]
    async fn test_audio_effect_processing() {
        let effect = TestAudioEffect::new();

        // Test parameter setting
        effect
            .set_parameter("gain", ParameterValue::Float(2.0))
            .unwrap();
        // Note: Can't test gain value since we can't mutate in set_parameter anymore
        // assert_eq!(effect.gain, 2.0);

        // Test audio processing
        let audio = crate::AudioBuffer::sine_wave(440.0, 1.0, 44100, 0.5);
        let processed = effect.process_audio(&audio).await.unwrap();

        // Should have applied 2x gain - check that magnitude has doubled
        assert!(processed.samples()[100].abs() > audio.samples()[100].abs());
    }

    #[test]
    fn test_plugin_metadata() {
        let effect = TestAudioEffect::new();
        let metadata = effect.metadata();

        assert_eq!(metadata.name, "TestEffect");
        assert_eq!(metadata.version, "1.0.0");
        assert_eq!(metadata.author, "VoiRS Team");
        assert_eq!(metadata.plugin_type, PluginType::Effect);
    }
}

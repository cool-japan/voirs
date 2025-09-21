//! Plugin system for custom evaluation metrics
//!
//! This module provides a flexible plugin architecture that allows users to
//! create and integrate custom evaluation metrics into the VoiRS evaluation system.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;
use voirs_sdk::{AudioBuffer, VoirsError};

/// Plugin system errors
#[derive(Error, Debug)]
pub enum PluginError {
    /// Plugin not found
    #[error("Plugin not found: {name}")]
    PluginNotFound {
        /// Plugin name
        name: String,
    },
    /// Plugin initialization failed
    #[error("Plugin initialization failed: {message}")]
    InitializationError {
        /// Error message
        message: String,
        /// Source error
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },
    /// Plugin execution failed
    #[error("Plugin execution failed: {message}")]
    ExecutionError {
        /// Error message
        message: String,
        /// Source error
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },
    /// Invalid plugin configuration
    #[error("Invalid plugin configuration: {message}")]
    ConfigurationError {
        /// Error message
        message: String,
    },
    /// Plugin dependency error
    #[error("Plugin dependency error: {message}")]
    DependencyError {
        /// Error message
        message: String,
    },
}

/// Custom metric plugin trait
#[async_trait]
pub trait MetricPlugin: Send + Sync {
    /// Get plugin information
    fn info(&self) -> &PluginInfo;

    /// Initialize the plugin with configuration
    async fn initialize(&mut self, config: PluginConfig) -> Result<(), PluginError>;

    /// Evaluate audio using the custom metric
    async fn evaluate(
        &self,
        generated: &AudioBuffer,
        reference: Option<&AudioBuffer>,
        context: &EvaluationContext,
    ) -> Result<MetricResult, PluginError>;

    /// Get metric dependencies (other metrics this plugin needs)
    fn dependencies(&self) -> Vec<String> {
        Vec::new()
    }

    /// Validate plugin configuration
    fn validate_config(&self, config: &PluginConfig) -> Result<(), PluginError> {
        let _ = config;
        Ok(())
    }

    /// Get plugin capabilities
    fn capabilities(&self) -> PluginCapabilities {
        PluginCapabilities::default()
    }

    /// Cleanup resources when plugin is unloaded
    async fn cleanup(&mut self) -> Result<(), PluginError> {
        Ok(())
    }
}

/// Plugin information metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginInfo {
    /// Plugin name
    pub name: String,
    /// Plugin version
    pub version: String,
    /// Plugin description
    pub description: String,
    /// Plugin author
    pub author: String,
    /// Plugin license
    pub license: String,
    /// Minimum VoiRS version required
    pub min_voirs_version: String,
    /// Plugin categories/tags
    pub categories: Vec<String>,
    /// Plugin website or repository URL
    pub url: Option<String>,
}

/// Plugin configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginConfig {
    /// Plugin-specific parameters
    pub parameters: HashMap<String, serde_json::Value>,
    /// Metric weight in overall score
    pub weight: f64,
    /// Whether plugin is enabled
    pub enabled: bool,
    /// Plugin timeout in milliseconds
    pub timeout_ms: u64,
    /// Cache configuration
    pub cache_config: CacheConfig,
    /// Logging configuration
    pub logging_config: LoggingConfig,
}

/// Cache configuration for plugin results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Enable caching
    pub enabled: bool,
    /// Cache TTL in seconds
    pub ttl_seconds: u64,
    /// Maximum cache size
    pub max_size: usize,
    /// Cache key prefix
    pub key_prefix: String,
}

/// Logging configuration for plugins
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level
    pub level: String,
    /// Enable performance logging
    pub performance: bool,
    /// Enable error logging
    pub errors: bool,
    /// Log file path (optional)
    pub file_path: Option<String>,
}

/// Evaluation context passed to plugins
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationContext {
    /// Language code
    pub language: Option<String>,
    /// Speaker information
    pub speaker_info: Option<SpeakerInfo>,
    /// Model information
    pub model_info: Option<ModelInfo>,
    /// Evaluation metadata
    pub metadata: HashMap<String, serde_json::Value>,
    /// Session ID for tracking
    pub session_id: Option<String>,
    /// Timestamp
    pub timestamp: f64,
}

/// Speaker information for context-aware evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerInfo {
    /// Speaker ID
    pub id: String,
    /// Speaker gender
    pub gender: Option<String>,
    /// Speaker age
    pub age: Option<u32>,
    /// Speaker accent
    pub accent: Option<String>,
    /// Native language
    pub native_language: Option<String>,
    /// Additional attributes
    pub attributes: HashMap<String, serde_json::Value>,
}

/// Model information for evaluation context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model name
    pub name: String,
    /// Model version
    pub version: String,
    /// Model type (e.g., "tts", "vc", "singing")
    pub model_type: String,
    /// Model architecture
    pub architecture: Option<String>,
    /// Training data information
    pub training_data: Option<String>,
    /// Model parameters
    pub parameters: HashMap<String, serde_json::Value>,
}

/// Plugin evaluation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricResult {
    /// Metric score (0-1 range recommended)
    pub score: f64,
    /// Confidence level (0-1)
    pub confidence: f64,
    /// Detailed sub-scores
    pub sub_scores: HashMap<String, f64>,
    /// Analysis details
    pub analysis: AnalysisDetails,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
    /// Error message if any
    pub error: Option<String>,
}

/// Detailed analysis from metric plugin
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisDetails {
    /// Metric-specific features
    pub features: HashMap<String, f64>,
    /// Quality indicators
    pub quality_indicators: Vec<QualityIndicator>,
    /// Recommendations for improvement
    pub recommendations: Vec<String>,
    /// Debug information
    pub debug_info: HashMap<String, serde_json::Value>,
}

/// Quality indicator from plugin analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityIndicator {
    /// Indicator name
    pub name: String,
    /// Indicator value
    pub value: f64,
    /// Value range (min, max)
    pub range: (f64, f64),
    /// Interpretation (e.g., "higher is better")
    pub interpretation: String,
    /// Importance weight
    pub importance: f64,
}

/// Plugin capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginCapabilities {
    /// Supports reference audio
    pub requires_reference: bool,
    /// Supports real-time evaluation
    pub supports_realtime: bool,
    /// Supports batch processing
    pub supports_batch: bool,
    /// Supported sample rates
    pub supported_sample_rates: Vec<u32>,
    /// Supported audio channels
    pub supported_channels: Vec<u8>,
    /// Supported languages
    pub supported_languages: Vec<String>,
    /// Maximum audio duration (seconds)
    pub max_duration: Option<f64>,
    /// Minimum audio duration (seconds)
    pub min_duration: Option<f64>,
}

impl Default for PluginConfig {
    fn default() -> Self {
        Self {
            parameters: HashMap::new(),
            weight: 1.0,
            enabled: true,
            timeout_ms: 30000,
            cache_config: CacheConfig {
                enabled: true,
                ttl_seconds: 3600,
                max_size: 1000,
                key_prefix: "plugin".to_string(),
            },
            logging_config: LoggingConfig {
                level: "info".to_string(),
                performance: true,
                errors: true,
                file_path: None,
            },
        }
    }
}

impl Default for PluginCapabilities {
    fn default() -> Self {
        Self {
            requires_reference: false,
            supports_realtime: false,
            supports_batch: true,
            supported_sample_rates: vec![16000, 22050, 44100, 48000],
            supported_channels: vec![1, 2],
            supported_languages: vec!["en".to_string()],
            max_duration: None,
            min_duration: None,
        }
    }
}

/// Plugin manager for registering and managing custom metrics
pub struct PluginManager {
    plugins: HashMap<String, Box<dyn MetricPlugin>>,
    configs: HashMap<String, PluginConfig>,
    cache: HashMap<String, MetricResult>,
    stats: PluginStats,
}

/// Plugin usage statistics
#[derive(Debug, Default)]
pub struct PluginStats {
    /// Number of evaluations per plugin
    pub evaluation_counts: HashMap<String, u64>,
    /// Total processing times per plugin
    pub processing_times: HashMap<String, u64>,
    /// Error counts per plugin
    pub error_counts: HashMap<String, u64>,
    /// Success rates per plugin
    pub success_rates: HashMap<String, f64>,
}

impl PluginManager {
    /// Create a new plugin manager
    pub fn new() -> Self {
        Self {
            plugins: HashMap::new(),
            configs: HashMap::new(),
            cache: HashMap::new(),
            stats: PluginStats::default(),
        }
    }

    /// Register a new plugin
    pub async fn register_plugin(
        &mut self,
        mut plugin: Box<dyn MetricPlugin>,
        config: PluginConfig,
    ) -> Result<(), PluginError> {
        let plugin_name = plugin.info().name.clone();

        // Validate configuration
        plugin.validate_config(&config)?;

        // Initialize plugin
        plugin.initialize(config.clone()).await?;

        // Store plugin and config
        self.plugins.insert(plugin_name.clone(), plugin);
        self.configs.insert(plugin_name.clone(), config);

        // Initialize stats
        self.stats.evaluation_counts.insert(plugin_name.clone(), 0);
        self.stats.processing_times.insert(plugin_name.clone(), 0);
        self.stats.error_counts.insert(plugin_name.clone(), 0);
        self.stats.success_rates.insert(plugin_name, 1.0);

        Ok(())
    }

    /// Unregister a plugin
    pub async fn unregister_plugin(&mut self, name: &str) -> Result<(), PluginError> {
        if let Some(mut plugin) = self.plugins.remove(name) {
            plugin.cleanup().await?;
        }
        self.configs.remove(name);
        self.stats.evaluation_counts.remove(name);
        self.stats.processing_times.remove(name);
        self.stats.error_counts.remove(name);
        self.stats.success_rates.remove(name);
        Ok(())
    }

    /// Evaluate using a specific plugin
    pub async fn evaluate_with_plugin(
        &mut self,
        plugin_name: &str,
        generated: &AudioBuffer,
        reference: Option<&AudioBuffer>,
        context: &EvaluationContext,
    ) -> Result<MetricResult, PluginError> {
        let start_time = std::time::Instant::now();

        // Check if plugin exists
        let plugin = self
            .plugins
            .get(plugin_name)
            .ok_or_else(|| PluginError::PluginNotFound {
                name: plugin_name.to_string(),
            })?;

        let config = self.configs.get(plugin_name).unwrap();

        // Check if plugin is enabled
        if !config.enabled {
            return Err(PluginError::ExecutionError {
                message: "Plugin is disabled".to_string(),
                source: None,
            });
        }

        // Validate audio compatibility
        self.validate_audio_compatibility(plugin, generated)?;

        // Check cache first
        let cache_key = self.generate_cache_key(plugin_name, generated, reference, context);
        if config.cache_config.enabled {
            if let Some(cached_result) = self.cache.get(&cache_key) {
                return Ok(cached_result.clone());
            }
        }

        // Execute plugin with timeout
        let result = tokio::time::timeout(
            std::time::Duration::from_millis(config.timeout_ms),
            plugin.evaluate(generated, reference, context),
        )
        .await
        .map_err(|_| PluginError::ExecutionError {
            message: "Plugin execution timeout".to_string(),
            source: None,
        })?;

        let processing_time = start_time.elapsed().as_millis() as u64;

        // Update statistics
        match &result {
            Ok(metric_result) => {
                *self.stats.evaluation_counts.get_mut(plugin_name).unwrap() += 1;
                *self.stats.processing_times.get_mut(plugin_name).unwrap() += processing_time;

                // Update success rate
                let total_evaluations = *self.stats.evaluation_counts.get(plugin_name).unwrap();
                let error_count = *self.stats.error_counts.get(plugin_name).unwrap();
                let success_rate =
                    (total_evaluations - error_count) as f64 / total_evaluations as f64;
                self.stats
                    .success_rates
                    .insert(plugin_name.to_string(), success_rate);

                // Cache result if caching is enabled
                if config.cache_config.enabled {
                    self.cache.insert(cache_key, metric_result.clone());

                    // Clean cache if it exceeds max size
                    if self.cache.len() > config.cache_config.max_size {
                        self.cleanup_cache();
                    }
                }
            }
            Err(_) => {
                *self.stats.error_counts.get_mut(plugin_name).unwrap() += 1;
            }
        }

        result
    }

    /// Evaluate using multiple plugins
    pub async fn evaluate_with_plugins(
        &mut self,
        plugin_names: &[String],
        generated: &AudioBuffer,
        reference: Option<&AudioBuffer>,
        context: &EvaluationContext,
    ) -> HashMap<String, Result<MetricResult, PluginError>> {
        let mut results = HashMap::new();

        for plugin_name in plugin_names {
            let result = self
                .evaluate_with_plugin(plugin_name, generated, reference, context)
                .await;
            results.insert(plugin_name.clone(), result);
        }

        results
    }

    /// Get list of registered plugins
    pub fn list_plugins(&self) -> Vec<&PluginInfo> {
        self.plugins.values().map(|p| p.info()).collect()
    }

    /// Get plugin configuration
    pub fn get_plugin_config(&self, name: &str) -> Option<&PluginConfig> {
        self.configs.get(name)
    }

    /// Update plugin configuration
    pub fn update_plugin_config(
        &mut self,
        name: &str,
        config: PluginConfig,
    ) -> Result<(), PluginError> {
        if let Some(plugin) = self.plugins.get(name) {
            plugin.validate_config(&config)?;
            self.configs.insert(name.to_string(), config);
            Ok(())
        } else {
            Err(PluginError::PluginNotFound {
                name: name.to_string(),
            })
        }
    }

    /// Get plugin statistics
    pub fn get_stats(&self) -> &PluginStats {
        &self.stats
    }

    /// Clear plugin cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Validate audio compatibility with plugin
    fn validate_audio_compatibility(
        &self,
        plugin: &Box<dyn MetricPlugin>,
        audio: &AudioBuffer,
    ) -> Result<(), PluginError> {
        let capabilities = plugin.capabilities();

        // Check sample rate
        if !capabilities
            .supported_sample_rates
            .contains(&(audio.sample_rate() as u32))
        {
            return Err(PluginError::ExecutionError {
                message: format!(
                    "Unsupported sample rate: {}. Supported: {:?}",
                    audio.sample_rate(),
                    capabilities.supported_sample_rates
                ),
                source: None,
            });
        }

        // Check channels
        if !capabilities
            .supported_channels
            .contains(&(audio.channels() as u8))
        {
            return Err(PluginError::ExecutionError {
                message: format!(
                    "Unsupported channel count: {}. Supported: {:?}",
                    audio.channels(),
                    capabilities.supported_channels
                ),
                source: None,
            });
        }

        // Check duration
        let duration = audio.duration() as f64;
        if let Some(max_duration) = capabilities.max_duration {
            if duration > max_duration {
                return Err(PluginError::ExecutionError {
                    message: format!(
                        "Audio duration {} exceeds maximum {}",
                        duration, max_duration
                    ),
                    source: None,
                });
            }
        }

        if let Some(min_duration) = capabilities.min_duration {
            if duration < min_duration {
                return Err(PluginError::ExecutionError {
                    message: format!("Audio duration {} below minimum {}", duration, min_duration),
                    source: None,
                });
            }
        }

        Ok(())
    }

    /// Generate cache key for plugin result
    fn generate_cache_key(
        &self,
        plugin_name: &str,
        generated: &AudioBuffer,
        reference: Option<&AudioBuffer>,
        context: &EvaluationContext,
    ) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        plugin_name.hash(&mut hasher);

        // Hash a subset of audio data instead of all samples
        let sample_hash = generated
            .samples()
            .iter()
            .take(100)
            .fold(0u64, |acc, &x| acc.wrapping_add((x * 1000.0) as u64));
        sample_hash.hash(&mut hasher);

        if let Some(ref_audio) = reference {
            let ref_hash = ref_audio
                .samples()
                .iter()
                .take(100)
                .fold(0u64, |acc, &x| acc.wrapping_add((x * 1000.0) as u64));
            ref_hash.hash(&mut hasher);
        }

        context.language.hash(&mut hasher);
        context.session_id.hash(&mut hasher);

        format!("plugin_{}_{:x}", plugin_name, hasher.finish())
    }

    /// Clean up old cache entries
    fn cleanup_cache(&mut self) {
        // Simple cleanup: remove half of the entries
        let target_size = self.cache.len() / 2;
        let keys_to_remove: Vec<String> = self.cache.keys().take(target_size).cloned().collect();
        for key in keys_to_remove {
            self.cache.remove(&key);
        }
    }
}

impl std::fmt::Debug for PluginManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PluginManager")
            .field("plugin_count", &self.plugins.len())
            .field("configs", &self.configs)
            .field("cache_size", &self.cache.len())
            .field("stats", &self.stats)
            .finish()
    }
}

impl Default for PluginManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Example custom metric plugin implementation
#[derive(Debug)]
pub struct ExampleMetricPlugin {
    info: PluginInfo,
    initialized: bool,
}

impl ExampleMetricPlugin {
    /// Create a new example plugin
    pub fn new() -> Self {
        Self {
            info: PluginInfo {
                name: "example_metric".to_string(),
                version: "1.0.0".to_string(),
                description: "Example custom metric plugin".to_string(),
                author: "VoiRS Team".to_string(),
                license: "MIT".to_string(),
                min_voirs_version: "0.1.0".to_string(),
                categories: vec!["quality".to_string(), "example".to_string()],
                url: Some("https://github.com/voirs/voirs".to_string()),
            },
            initialized: false,
        }
    }
}

#[async_trait]
impl MetricPlugin for ExampleMetricPlugin {
    fn info(&self) -> &PluginInfo {
        &self.info
    }

    async fn initialize(&mut self, _config: PluginConfig) -> Result<(), PluginError> {
        self.initialized = true;
        Ok(())
    }

    async fn evaluate(
        &self,
        generated: &AudioBuffer,
        reference: Option<&AudioBuffer>,
        _context: &EvaluationContext,
    ) -> Result<MetricResult, PluginError> {
        if !self.initialized {
            return Err(PluginError::ExecutionError {
                message: "Plugin not initialized".to_string(),
                source: None,
            });
        }

        let start_time = std::time::Instant::now();

        // Simple example metric: RMS level comparison
        let generated_rms = self.calculate_rms(generated.samples());
        let score = if let Some(ref_audio) = reference {
            let reference_rms = self.calculate_rms(ref_audio.samples());
            let difference = (generated_rms - reference_rms).abs();
            (1.0 - difference.min(1.0)).max(0.0)
        } else {
            // Without reference, just return normalized RMS as quality indicator
            generated_rms.min(1.0)
        };

        let processing_time = start_time.elapsed().as_millis() as u64;

        Ok(MetricResult {
            score,
            confidence: 0.8,
            sub_scores: HashMap::from([
                ("rms_level".to_string(), generated_rms),
                ("quality_estimate".to_string(), score),
            ]),
            analysis: AnalysisDetails {
                features: HashMap::from([
                    ("generated_rms".to_string(), generated_rms),
                    (
                        "reference_rms".to_string(),
                        reference.map_or(0.0, |r| self.calculate_rms(r.samples())),
                    ),
                ]),
                quality_indicators: vec![QualityIndicator {
                    name: "RMS Level".to_string(),
                    value: generated_rms,
                    range: (0.0, 1.0),
                    interpretation: "Optimal range: 0.1-0.8".to_string(),
                    importance: 0.7,
                }],
                recommendations: if generated_rms < 0.1 {
                    vec!["Audio level is too low, consider increasing gain".to_string()]
                } else if generated_rms > 0.8 {
                    vec!["Audio level is high, check for clipping".to_string()]
                } else {
                    vec!["Audio level is in optimal range".to_string()]
                },
                debug_info: HashMap::from([
                    (
                        "sample_count".to_string(),
                        serde_json::Value::Number(serde_json::Number::from(
                            generated.samples().len(),
                        )),
                    ),
                    (
                        "sample_rate".to_string(),
                        serde_json::Value::Number(serde_json::Number::from(
                            generated.sample_rate(),
                        )),
                    ),
                ]),
            },
            processing_time_ms: processing_time,
            error: None,
        })
    }

    fn capabilities(&self) -> PluginCapabilities {
        PluginCapabilities {
            requires_reference: false,
            supports_realtime: true,
            supports_batch: true,
            supported_sample_rates: vec![16000, 22050, 44100, 48000],
            supported_channels: vec![1, 2],
            supported_languages: vec!["en".to_string(), "es".to_string(), "fr".to_string()],
            max_duration: Some(300.0), // 5 minutes
            min_duration: Some(0.1),   // 100ms
        }
    }
}

impl ExampleMetricPlugin {
    /// Calculate RMS level of audio samples
    fn calculate_rms(&self, samples: &[f32]) -> f64 {
        if samples.is_empty() {
            return 0.0;
        }

        let sum_squares: f64 = samples.iter().map(|&x| (x as f64).powi(2)).sum();
        (sum_squares / samples.len() as f64).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use voirs_sdk::AudioBuffer;

    #[test]
    fn test_plugin_info_creation() {
        let info = PluginInfo {
            name: "test_plugin".to_string(),
            version: "1.0.0".to_string(),
            description: "Test plugin".to_string(),
            author: "Test Author".to_string(),
            license: "MIT".to_string(),
            min_voirs_version: "0.1.0".to_string(),
            categories: vec!["test".to_string()],
            url: None,
        };

        assert_eq!(info.name, "test_plugin");
        assert_eq!(info.version, "1.0.0");
    }

    #[test]
    fn test_plugin_config_default() {
        let config = PluginConfig::default();
        assert!(config.enabled);
        assert_eq!(config.weight, 1.0);
        assert_eq!(config.timeout_ms, 30000);
    }

    #[test]
    fn test_plugin_capabilities_default() {
        let capabilities = PluginCapabilities::default();
        assert!(!capabilities.requires_reference);
        assert!(capabilities.supports_batch);
        assert!(!capabilities.supports_realtime);
    }

    #[test]
    fn test_plugin_manager_creation() {
        let manager = PluginManager::new();
        assert_eq!(manager.plugins.len(), 0);
        assert_eq!(manager.configs.len(), 0);
    }

    #[tokio::test]
    async fn test_example_plugin_creation() {
        let plugin = ExampleMetricPlugin::new();
        assert_eq!(plugin.info().name, "example_metric");
        assert!(!plugin.initialized);
    }

    #[tokio::test]
    async fn test_example_plugin_initialization() {
        let mut plugin = ExampleMetricPlugin::new();
        let config = PluginConfig::default();

        let result = plugin.initialize(config).await;
        assert!(result.is_ok());
        assert!(plugin.initialized);
    }

    #[tokio::test]
    async fn test_example_plugin_evaluation() {
        let mut plugin = ExampleMetricPlugin::new();
        let config = PluginConfig::default();
        plugin.initialize(config).await.unwrap();

        let audio = AudioBuffer::new(vec![0.1; 1000], 16000, 1);
        let context = EvaluationContext {
            language: Some("en".to_string()),
            speaker_info: None,
            model_info: None,
            metadata: HashMap::new(),
            session_id: None,
            timestamp: 0.0,
        };

        let result = plugin.evaluate(&audio, None, &context).await;
        assert!(result.is_ok());

        let metric_result = result.unwrap();
        assert!(metric_result.score >= 0.0 && metric_result.score <= 1.0);
        assert!(metric_result.confidence > 0.0);
        assert!(!metric_result.sub_scores.is_empty());
    }

    #[tokio::test]
    async fn test_plugin_manager_registration() {
        let mut manager = PluginManager::new();
        let plugin = Box::new(ExampleMetricPlugin::new());
        let config = PluginConfig::default();

        let result = manager.register_plugin(plugin, config).await;
        assert!(result.is_ok());
        assert_eq!(manager.plugins.len(), 1);
        assert_eq!(manager.configs.len(), 1);
    }

    #[tokio::test]
    async fn test_plugin_manager_evaluation() {
        let mut manager = PluginManager::new();
        let plugin = Box::new(ExampleMetricPlugin::new());
        let config = PluginConfig::default();

        manager.register_plugin(plugin, config).await.unwrap();

        let audio = AudioBuffer::new(vec![0.2; 2000], 16000, 1); // 0.125 seconds duration
        let context = EvaluationContext {
            language: Some("en".to_string()),
            speaker_info: None,
            model_info: None,
            metadata: HashMap::new(),
            session_id: None,
            timestamp: 0.0,
        };

        let result = manager
            .evaluate_with_plugin("example_metric", &audio, None, &context)
            .await;

        assert!(result.is_ok());

        // Check that stats were updated
        let stats = manager.get_stats();
        assert_eq!(*stats.evaluation_counts.get("example_metric").unwrap(), 1);
    }

    #[tokio::test]
    async fn test_plugin_manager_unregistration() {
        let mut manager = PluginManager::new();
        let plugin = Box::new(ExampleMetricPlugin::new());
        let config = PluginConfig::default();

        manager.register_plugin(plugin, config).await.unwrap();
        assert_eq!(manager.plugins.len(), 1);

        let result = manager.unregister_plugin("example_metric").await;
        assert!(result.is_ok());
        assert_eq!(manager.plugins.len(), 0);
    }

    #[test]
    fn test_plugin_list() {
        let mut manager = PluginManager::new();
        assert_eq!(manager.list_plugins().len(), 0);
    }

    #[test]
    fn test_metric_result_serialization() {
        let result = MetricResult {
            score: 0.85,
            confidence: 0.9,
            sub_scores: HashMap::from([("test".to_string(), 0.8)]),
            analysis: AnalysisDetails {
                features: HashMap::new(),
                quality_indicators: Vec::new(),
                recommendations: Vec::new(),
                debug_info: HashMap::new(),
            },
            processing_time_ms: 100,
            error: None,
        };

        let serialized = serde_json::to_string(&result).unwrap();
        assert!(serialized.contains("0.85"));

        let deserialized: MetricResult = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.score, 0.85);
    }
}

//! Plugin Architecture for Custom Voice Cloning Models
//!
//! This module provides a comprehensive plugin system that allows users to integrate
//! custom voice cloning models into the VoiRS framework. It supports dynamic plugin
//! loading, lifecycle management, configuration validation, and seamless integration
//! with the existing cloning pipeline.

use crate::{
    config::CloningConfig,
    model_loading::{ModelInterface, ModelLoadingManager},
    performance_monitoring::{
        PerformanceMeasurement, PerformanceMetrics, PerformanceMonitor, PerformanceTargets,
    },
    quality::{CloningQualityAssessor, QualityMetrics},
    types::{
        CloningMethod, SpeakerData, SpeakerProfile, VoiceCloneRequest, VoiceCloneResult,
        VoiceSample,
    },
    Error, Result,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, error, info, warn};

/// Plugin manager for custom voice cloning models
pub struct PluginManager {
    /// Registered plugins
    plugins: Arc<RwLock<HashMap<String, Arc<dyn CloningPlugin>>>>,
    /// Plugin configurations
    plugin_configs: Arc<RwLock<HashMap<String, PluginConfig>>>,
    /// Plugin registry for discovery
    registry: Arc<PluginRegistry>,
    /// Plugin loading metrics
    metrics: Arc<RwLock<PluginMetrics>>,
    /// Performance monitor for plugin operations
    performance_monitor: Arc<PerformanceMonitor>,
    /// Manager configuration
    config: PluginManagerConfig,
}

/// Configuration for the plugin manager
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginManagerConfig {
    /// Plugin discovery paths
    pub plugin_paths: Vec<PathBuf>,
    /// Enable automatic plugin discovery
    pub auto_discovery: bool,
    /// Maximum number of plugins to load
    pub max_plugins: usize,
    /// Plugin loading timeout
    pub loading_timeout: Duration,
    /// Enable plugin validation
    pub enable_validation: bool,
    /// Plugin cache directory
    pub cache_directory: Option<PathBuf>,
    /// Enable plugin hot-reloading
    pub enable_hot_reload: bool,
    /// Plugin API version compatibility
    pub api_version: String,
}

impl Default for PluginManagerConfig {
    fn default() -> Self {
        Self {
            plugin_paths: vec![
                PathBuf::from("./plugins"),
                PathBuf::from("/usr/local/lib/voirs/plugins"),
            ],
            auto_discovery: true,
            max_plugins: 50,
            loading_timeout: Duration::from_secs(30),
            enable_validation: true,
            cache_directory: None,
            enable_hot_reload: false,
            api_version: "1.0.0".to_string(),
        }
    }
}

/// Plugin configuration and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginConfig {
    /// Plugin unique identifier
    pub id: String,
    /// Plugin name
    pub name: String,
    /// Plugin version
    pub version: String,
    /// Plugin description
    pub description: String,
    /// Plugin author
    pub author: String,
    /// Supported cloning methods
    pub supported_methods: Vec<CloningMethod>,
    /// Plugin capabilities
    pub capabilities: PluginCapabilities,
    /// Configuration parameters
    pub parameters: HashMap<String, PluginParameter>,
    /// Plugin dependencies
    pub dependencies: Vec<PluginDependency>,
    /// Minimum API version required
    pub min_api_version: String,
    /// Plugin license
    pub license: Option<String>,
    /// Plugin website/repository
    pub website: Option<String>,
}

/// Plugin capabilities and features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginCapabilities {
    /// Supports real-time synthesis
    pub realtime_synthesis: bool,
    /// Supports streaming adaptation
    pub streaming_adaptation: bool,
    /// Supports cross-lingual cloning
    pub cross_lingual: bool,
    /// Supports emotion transfer
    pub emotion_transfer: bool,
    /// Supports voice morphing
    pub voice_morphing: bool,
    /// Supports zero-shot cloning
    pub zero_shot: bool,
    /// Requires GPU acceleration
    pub requires_gpu: bool,
    /// Maximum concurrent sessions
    pub max_concurrent_sessions: usize,
    /// Supported sample rates
    pub supported_sample_rates: Vec<u32>,
    /// Supported languages
    pub supported_languages: Vec<String>,
    /// Memory requirements (MB)
    pub memory_requirements: usize,
}

impl Default for PluginCapabilities {
    fn default() -> Self {
        Self {
            realtime_synthesis: false,
            streaming_adaptation: false,
            cross_lingual: false,
            emotion_transfer: false,
            voice_morphing: false,
            zero_shot: false,
            requires_gpu: false,
            max_concurrent_sessions: 1,
            supported_sample_rates: vec![16000, 22050, 44100],
            supported_languages: vec!["en".to_string()],
            memory_requirements: 512, // 512MB default
        }
    }
}

/// Plugin parameter definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginParameter {
    /// Parameter name
    pub name: String,
    /// Parameter type
    pub param_type: ParameterType,
    /// Parameter description
    pub description: String,
    /// Default value
    pub default_value: ParameterValue,
    /// Parameter constraints
    pub constraints: Option<ParameterConstraints>,
    /// Whether parameter is required
    pub required: bool,
}

/// Parameter types supported by plugins
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterType {
    String,
    Integer,
    Float,
    Boolean,
    Array(Box<ParameterType>),
    Object,
}

/// Parameter values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterValue {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Array(Vec<ParameterValue>),
    Object(HashMap<String, ParameterValue>),
    None,
}

/// Parameter constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterConstraints {
    /// Minimum value (for numeric types)
    pub min: Option<f64>,
    /// Maximum value (for numeric types)
    pub max: Option<f64>,
    /// Allowed values (for enum-like parameters)
    pub allowed_values: Option<Vec<ParameterValue>>,
    /// Regular expression pattern (for strings)
    pub pattern: Option<String>,
    /// Minimum length (for strings and arrays)
    pub min_length: Option<usize>,
    /// Maximum length (for strings and arrays)
    pub max_length: Option<usize>,
}

/// Plugin dependency specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginDependency {
    /// Dependency name
    pub name: String,
    /// Required version
    pub version: String,
    /// Whether dependency is optional
    pub optional: bool,
}

/// Main trait that all plugins must implement
#[async_trait::async_trait]
pub trait CloningPlugin: Send + Sync {
    /// Get plugin configuration
    fn get_config(&self) -> &PluginConfig;

    /// Initialize the plugin
    async fn initialize(&mut self, parameters: HashMap<String, ParameterValue>) -> Result<()>;

    /// Shutdown the plugin
    async fn shutdown(&mut self) -> Result<()>;

    /// Perform voice cloning
    async fn clone_voice(
        &self,
        request: VoiceCloneRequest,
        context: PluginContext,
    ) -> Result<VoiceCloneResult>;

    /// Validate speaker data for compatibility
    async fn validate_speaker_data(&self, data: &SpeakerData) -> Result<PluginValidationResult>;

    /// Get plugin health status
    async fn health_check(&self) -> Result<PluginHealth>;

    /// Update plugin configuration at runtime
    async fn update_config(&mut self, parameters: HashMap<String, ParameterValue>) -> Result<()>;

    /// Get plugin capabilities
    fn get_capabilities(&self) -> &PluginCapabilities;

    /// Get plugin metrics
    async fn get_metrics(&self) -> Result<PluginOperationMetrics>;
}

/// Context provided to plugins during operation
#[derive(Clone)]
pub struct PluginContext {
    /// Quality assessor instance
    pub quality_assessor: Arc<CloningQualityAssessor>,
    /// Model loading manager
    pub model_loader: Arc<ModelLoadingManager>,
    /// Global cloning configuration
    pub global_config: Arc<CloningConfig>,
    /// Request metadata
    pub request_metadata: HashMap<String, String>,
    /// Session ID
    pub session_id: String,
    /// User context
    pub user_context: Option<HashMap<String, String>>,
}

/// Plugin validation result
#[derive(Debug, Clone)]
pub struct PluginValidationResult {
    /// Whether validation passed
    pub is_valid: bool,
    /// Validation errors
    pub errors: Vec<String>,
    /// Validation warnings
    pub warnings: Vec<String>,
    /// Compatibility score (0.0 to 1.0)
    pub compatibility_score: f32,
    /// Required adaptations
    pub required_adaptations: Vec<String>,
}

/// Plugin health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginHealth {
    /// Overall health status
    pub status: PluginHealthStatus,
    /// Health score (0.0 to 1.0)
    pub health_score: f32,
    /// Memory usage (MB)
    pub memory_usage: usize,
    /// CPU usage percentage
    pub cpu_usage: f32,
    /// Active sessions count
    pub active_sessions: usize,
    /// Last health check timestamp
    pub last_check: SystemTime,
    /// Health issues
    pub issues: Vec<String>,
    /// Performance metrics
    pub performance: PluginPerformanceMetrics,
}

/// Plugin health status
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum PluginHealthStatus {
    Healthy,
    Warning,
    Critical,
    Offline,
}

/// Plugin performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginPerformanceMetrics {
    /// Average processing time
    pub avg_processing_time: Duration,
    /// Requests per second
    pub requests_per_second: f32,
    /// Success rate
    pub success_rate: f32,
    /// Error rate
    pub error_rate: f32,
    /// Quality score
    pub avg_quality_score: f32,
}

impl Default for PluginPerformanceMetrics {
    fn default() -> Self {
        Self {
            avg_processing_time: Duration::from_millis(100),
            requests_per_second: 0.0,
            success_rate: 1.0,
            error_rate: 0.0,
            avg_quality_score: 0.0,
        }
    }
}

/// Plugin operation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginOperationMetrics {
    /// Total operations performed
    pub total_operations: u64,
    /// Successful operations
    pub successful_operations: u64,
    /// Failed operations
    pub failed_operations: u64,
    /// Total processing time
    pub total_processing_time: Duration,
    /// Cache hits
    pub cache_hits: u64,
    /// Cache misses
    pub cache_misses: u64,
    /// Memory usage statistics
    pub memory_stats: PluginMemoryStats,
}

impl Default for PluginOperationMetrics {
    fn default() -> Self {
        Self {
            total_operations: 0,
            successful_operations: 0,
            failed_operations: 0,
            total_processing_time: Duration::from_secs(0),
            cache_hits: 0,
            cache_misses: 0,
            memory_stats: PluginMemoryStats::default(),
        }
    }
}

/// Plugin memory usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginMemoryStats {
    /// Current memory usage (bytes)
    pub current_usage: usize,
    /// Peak memory usage (bytes)
    pub peak_usage: usize,
    /// Memory allocations count
    pub allocations: u64,
    /// Memory deallocations count
    pub deallocations: u64,
}

impl Default for PluginMemoryStats {
    fn default() -> Self {
        Self {
            current_usage: 0,
            peak_usage: 0,
            allocations: 0,
            deallocations: 0,
        }
    }
}

/// Plugin registry for discovery and management
pub struct PluginRegistry {
    /// Available plugins
    available_plugins: Arc<RwLock<HashMap<String, PluginManifest>>>,
    /// Plugin discovery paths
    discovery_paths: Vec<PathBuf>,
    /// Registry cache
    cache: Arc<RwLock<Option<RegistryCache>>>,
}

/// Plugin manifest for discovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginManifest {
    /// Plugin configuration
    pub config: PluginConfig,
    /// Plugin file path
    pub path: PathBuf,
    /// Last modified timestamp
    pub last_modified: SystemTime,
    /// Plugin size in bytes
    pub size: usize,
    /// Plugin checksum for integrity
    pub checksum: String,
}

/// Registry cache for performance
#[derive(Debug, Clone)]
pub struct RegistryCache {
    /// Cached plugin manifests
    pub manifests: HashMap<String, PluginManifest>,
    /// Cache timestamp
    pub cached_at: SystemTime,
    /// Cache expiration
    pub expires_at: SystemTime,
}

/// Plugin manager metrics
#[derive(Debug, Default, Clone)]
pub struct PluginMetrics {
    /// Total plugins registered
    pub total_plugins: usize,
    /// Active plugins
    pub active_plugins: usize,
    /// Plugin loading time
    pub total_loading_time: Duration,
    /// Plugin initialization failures
    pub initialization_failures: usize,
    /// Plugin health check failures
    pub health_check_failures: usize,
    /// Total plugin operations
    pub total_operations: u64,
}

impl PluginManager {
    /// Create new plugin manager
    pub fn new(config: PluginManagerConfig) -> Self {
        // Create performance monitor with plugin-specific targets
        let plugin_targets = PerformanceTargets {
            adaptation_time_target: Duration::from_secs(60), // 1 minute for plugin operations
            synthesis_rtf_target: 0.15, // Slightly higher tolerance for plugins
            memory_usage_target: 512 * 1024 * 1024, // 512MB for plugin operations
            quality_score_target: 0.75, // Lower threshold for experimental plugins
            concurrent_adaptations_target: 8, // Moderate concurrency for plugins
        };
        let performance_monitor = Arc::new(PerformanceMonitor::with_targets(plugin_targets));

        Self {
            plugins: Arc::new(RwLock::new(HashMap::new())),
            plugin_configs: Arc::new(RwLock::new(HashMap::new())),
            registry: Arc::new(PluginRegistry::new(config.plugin_paths.clone())),
            metrics: Arc::new(RwLock::new(PluginMetrics::default())),
            performance_monitor,
            config,
        }
    }

    /// Discover available plugins
    pub async fn discover_plugins(&self) -> Result<Vec<PluginManifest>> {
        if !self.config.auto_discovery {
            return Ok(Vec::new());
        }

        info!("Discovering plugins in configured paths");
        let manifests = self.registry.discover_plugins().await?;

        let mut metrics = self.metrics.write().await;
        metrics.total_plugins = manifests.len();

        info!("Discovered {} plugins", manifests.len());
        Ok(manifests)
    }

    /// Register a plugin
    pub async fn register_plugin(&self, plugin: Arc<dyn CloningPlugin>) -> Result<()> {
        let config = plugin.get_config().clone(); // Clone the config to avoid borrowing issues
        let plugin_id = config.id.clone();

        // Validate plugin
        if self.config.enable_validation {
            self.validate_plugin(&plugin).await?;
        }

        // Check plugin limit
        let plugins_count = self.plugins.read().await.len();
        if plugins_count >= self.config.max_plugins {
            return Err(Error::Validation(format!(
                "Maximum number of plugins ({}) exceeded",
                self.config.max_plugins
            )));
        }

        // Register plugin
        let mut plugins = self.plugins.write().await;
        let mut plugin_configs = self.plugin_configs.write().await;

        plugins.insert(plugin_id.clone(), plugin);
        plugin_configs.insert(plugin_id.clone(), config.clone());

        info!("Registered plugin: {} ({})", config.name, plugin_id);
        Ok(())
    }

    /// Unregister a plugin
    pub async fn unregister_plugin(&self, plugin_id: &str) -> Result<()> {
        let mut plugins = self.plugins.write().await;
        let mut plugin_configs = self.plugin_configs.write().await;

        if let Some(mut plugin) = plugins.remove(plugin_id) {
            // Shutdown plugin safely
            if let Err(e) = Arc::get_mut(&mut plugin).unwrap().shutdown().await {
                warn!("Error shutting down plugin {}: {}", plugin_id, e);
            }
        }

        plugin_configs.remove(plugin_id);
        info!("Unregistered plugin: {}", plugin_id);
        Ok(())
    }

    /// Get registered plugin
    pub async fn get_plugin(&self, plugin_id: &str) -> Option<Arc<dyn CloningPlugin>> {
        let plugins = self.plugins.read().await;
        plugins.get(plugin_id).cloned()
    }

    /// List all registered plugins
    pub async fn list_plugins(&self) -> Vec<PluginConfig> {
        let plugin_configs = self.plugin_configs.read().await;
        plugin_configs.values().cloned().collect()
    }

    /// Clone voice using a specific plugin
    pub async fn clone_voice_with_plugin(
        &self,
        plugin_id: &str,
        request: VoiceCloneRequest,
        context: PluginContext,
    ) -> Result<VoiceCloneResult> {
        let start_time = Instant::now();

        let plugin = self
            .get_plugin(plugin_id)
            .await
            .ok_or_else(|| Error::Validation(format!("Plugin not found: {}", plugin_id)))?;

        // Validate speaker data compatibility
        let validation = plugin.validate_speaker_data(&request.speaker_data).await?;
        if !validation.is_valid {
            return Err(Error::Validation(format!(
                "Speaker data validation failed: {}",
                validation.errors.join(", ")
            )));
        }

        // Perform cloning
        let result = plugin.clone_voice(request, context).await?;

        // Update metrics
        let processing_time = start_time.elapsed();
        let mut metrics = self.metrics.write().await;
        metrics.total_operations += 1;

        // Record performance metrics
        let memory_usage = self.estimate_plugin_memory_usage(plugin_id).await;
        let quality_score = result.similarity_score as f64;

        let performance_metrics = PerformanceMetrics {
            adaptation_time: processing_time,
            synthesis_rtf: processing_time.as_secs_f64()
                / result.processing_time.as_secs_f64().max(0.001),
            memory_usage,
            quality_score,
            concurrent_adaptations: 1, // Single plugin operation
            timestamp: SystemTime::now(),
        };

        let targets = self.performance_monitor.get_targets();
        let measurement = PerformanceMeasurement {
            metrics: performance_metrics.clone(),
            targets: targets.clone(),
            target_results: crate::performance_monitoring::TargetResults {
                adaptation_time_met: processing_time <= targets.adaptation_time_target,
                synthesis_rtf_met: performance_metrics.synthesis_rtf
                    <= targets.synthesis_rtf_target,
                memory_usage_met: memory_usage <= targets.memory_usage_target,
                quality_score_met: quality_score >= targets.quality_score_target,
                concurrent_adaptations_met: true, // Single operation
            },
            overall_score: if quality_score >= targets.quality_score_target {
                1.0
            } else {
                0.8
            },
        };

        // Record the measurement (fire-and-forget)
        let performance_monitor = Arc::clone(&self.performance_monitor);
        tokio::spawn(async move {
            let _ = performance_monitor.record_measurement(measurement).await;
        });

        debug!(
            "Plugin {} completed operation in {:?}",
            plugin_id, processing_time
        );
        Ok(result)
    }

    /// Find best plugin for a specific request
    pub async fn find_best_plugin(&self, request: &VoiceCloneRequest) -> Result<Option<String>> {
        let plugins = self.plugins.read().await;
        let mut best_plugin = None;
        let mut best_score = 0.0f32;

        for (plugin_id, plugin) in plugins.iter() {
            let validation = plugin.validate_speaker_data(&request.speaker_data).await?;
            if validation.is_valid && validation.compatibility_score > best_score {
                best_score = validation.compatibility_score;
                best_plugin = Some(plugin_id.clone());
            }
        }

        Ok(best_plugin)
    }

    /// Perform health check on all plugins
    pub async fn health_check_all(&self) -> HashMap<String, PluginHealth> {
        let plugins = self.plugins.read().await;
        let mut health_status = HashMap::new();

        for (plugin_id, plugin) in plugins.iter() {
            match plugin.health_check().await {
                Ok(health) => {
                    health_status.insert(plugin_id.clone(), health);
                }
                Err(e) => {
                    warn!("Health check failed for plugin {}: {}", plugin_id, e);
                    let mut metrics = self.metrics.write().await;
                    metrics.health_check_failures += 1;

                    health_status.insert(
                        plugin_id.clone(),
                        PluginHealth {
                            status: PluginHealthStatus::Offline,
                            health_score: 0.0,
                            memory_usage: 0,
                            cpu_usage: 0.0,
                            active_sessions: 0,
                            last_check: SystemTime::now(),
                            issues: vec![format!("Health check failed: {}", e)],
                            performance: PluginPerformanceMetrics::default(),
                        },
                    );
                }
            }
        }

        health_status
    }

    /// Get plugin manager metrics
    pub async fn get_metrics(&self) -> PluginMetrics {
        let metrics = self.metrics.read().await;
        let plugins = self.plugins.read().await;

        let mut result = metrics.clone();
        result.active_plugins = plugins.len();
        result
    }

    /// Validate plugin before registration
    async fn validate_plugin(&self, plugin: &Arc<dyn CloningPlugin>) -> Result<()> {
        let config = plugin.get_config();

        // Check API version compatibility
        if !self.is_api_version_compatible(&config.min_api_version) {
            return Err(Error::Validation(format!(
                "Plugin requires API version {} but current version is {}",
                config.min_api_version, self.config.api_version
            )));
        }

        // Validate plugin ID uniqueness
        let plugin_configs = self.plugin_configs.read().await;
        if plugin_configs.contains_key(&config.id) {
            return Err(Error::Validation(format!(
                "Plugin ID already exists: {}",
                config.id
            )));
        }

        // Validate configuration
        self.validate_plugin_config(config)?;

        Ok(())
    }

    /// Check API version compatibility
    fn is_api_version_compatible(&self, required_version: &str) -> bool {
        // Simple version comparison - in production would use proper semver
        let current_parts: Vec<u32> = self
            .config
            .api_version
            .split('.')
            .filter_map(|s| s.parse().ok())
            .collect();

        let required_parts: Vec<u32> = required_version
            .split('.')
            .filter_map(|s| s.parse().ok())
            .collect();

        if current_parts.len() < 3 || required_parts.len() < 3 {
            return false;
        }

        // Check major version compatibility
        current_parts[0] == required_parts[0]
            && (current_parts[1] > required_parts[1]
                || (current_parts[1] == required_parts[1] && current_parts[2] >= required_parts[2]))
    }

    /// Validate plugin configuration
    fn validate_plugin_config(&self, config: &PluginConfig) -> Result<()> {
        if config.id.is_empty() {
            return Err(Error::Validation("Plugin ID cannot be empty".to_string()));
        }

        if config.name.is_empty() {
            return Err(Error::Validation("Plugin name cannot be empty".to_string()));
        }

        if config.version.is_empty() {
            return Err(Error::Validation(
                "Plugin version cannot be empty".to_string(),
            ));
        }

        // Validate parameters
        for parameter in config.parameters.values() {
            self.validate_parameter(parameter)?;
        }

        Ok(())
    }

    /// Validate plugin parameter
    fn validate_parameter(&self, parameter: &PluginParameter) -> Result<()> {
        if parameter.name.is_empty() {
            return Err(Error::Validation(
                "Parameter name cannot be empty".to_string(),
            ));
        }

        // Validate constraints if present
        if let Some(constraints) = &parameter.constraints {
            if let (Some(min), Some(max)) = (constraints.min, constraints.max) {
                if min > max {
                    return Err(Error::Validation(format!(
                        "Parameter {} min value ({}) cannot be greater than max value ({})",
                        parameter.name, min, max
                    )));
                }
            }
        }

        Ok(())
    }

    /// Estimate memory usage for a specific plugin
    async fn estimate_plugin_memory_usage(&self, plugin_id: &str) -> u64 {
        // In a real implementation, this would query actual memory usage
        // For now, we'll estimate based on plugin type and typical usage
        128 * 1024 * 1024 // 128MB estimate for plugin operations
    }
}

impl PluginRegistry {
    /// Create new plugin registry
    pub fn new(discovery_paths: Vec<PathBuf>) -> Self {
        Self {
            available_plugins: Arc::new(RwLock::new(HashMap::new())),
            discovery_paths,
            cache: Arc::new(RwLock::new(None)),
        }
    }

    /// Discover plugins in configured paths
    pub async fn discover_plugins(&self) -> Result<Vec<PluginManifest>> {
        // Check cache first
        {
            let cache = self.cache.read().await;
            if let Some(cached) = cache.as_ref() {
                if cached.expires_at > SystemTime::now() {
                    return Ok(cached.manifests.values().cloned().collect());
                }
            }
        }

        let mut manifests = Vec::new();

        // Discover plugins in each path
        for path in &self.discovery_paths {
            if path.exists() && path.is_dir() {
                if let Ok(entries) = std::fs::read_dir(path) {
                    for entry in entries.flatten() {
                        if let Ok(manifest) = self.parse_plugin_manifest(&entry.path()).await {
                            manifests.push(manifest);
                        }
                    }
                }
            }
        }

        // Update cache
        {
            let mut cache = self.cache.write().await;
            *cache = Some(RegistryCache {
                manifests: manifests
                    .iter()
                    .map(|m| (m.config.id.clone(), m.clone()))
                    .collect(),
                cached_at: SystemTime::now(),
                expires_at: SystemTime::now() + Duration::from_secs(300), // 5 minutes
            });
        }

        Ok(manifests)
    }

    /// Parse plugin manifest from file
    async fn parse_plugin_manifest(&self, path: &PathBuf) -> Result<PluginManifest> {
        // Look for plugin.json or plugin.toml manifest files
        let manifest_path = if path.join("plugin.json").exists() {
            path.join("plugin.json")
        } else if path.join("plugin.toml").exists() {
            path.join("plugin.toml")
        } else {
            return Err(Error::Validation("No plugin manifest found".to_string()));
        };

        let content = tokio::fs::read_to_string(&manifest_path)
            .await
            .map_err(|e| Error::Validation(format!("Failed to read manifest: {}", e)))?;

        let config: PluginConfig =
            if manifest_path.extension().and_then(|s| s.to_str()) == Some("json") {
                serde_json::from_str(&content)
                    .map_err(|e| Error::Validation(format!("Invalid JSON manifest: {}", e)))?
            } else {
                return Err(Error::Validation(
                    "Only JSON manifests are currently supported".to_string(),
                ));
            };

        let metadata = tokio::fs::metadata(&manifest_path)
            .await
            .map_err(|e| Error::Validation(format!("Failed to get manifest metadata: {}", e)))?;

        Ok(PluginManifest {
            config,
            path: path.clone(),
            last_modified: metadata.modified().unwrap_or(SystemTime::now()),
            size: metadata.len() as usize,
            checksum: format!("{:x}", content.len()), // Simple checksum
        })
    }
}

/// Example implementation of a basic plugin
pub struct ExamplePlugin {
    config: PluginConfig,
    initialized: bool,
    parameters: HashMap<String, ParameterValue>,
    metrics: PluginOperationMetrics,
}

impl ExamplePlugin {
    /// Create new example plugin
    pub fn new() -> Self {
        let mut parameters = HashMap::new();
        parameters.insert(
            "quality_level".to_string(),
            PluginParameter {
                name: "quality_level".to_string(),
                param_type: ParameterType::Float,
                description: "Voice cloning quality level".to_string(),
                default_value: ParameterValue::Float(0.8),
                constraints: Some(ParameterConstraints {
                    min: Some(0.0),
                    max: Some(1.0),
                    allowed_values: None,
                    pattern: None,
                    min_length: None,
                    max_length: None,
                }),
                required: false,
            },
        );

        let config = PluginConfig {
            id: "example_plugin".to_string(),
            name: "Example Voice Cloning Plugin".to_string(),
            version: "1.0.0".to_string(),
            description: "Example plugin for demonstration purposes".to_string(),
            author: "VoiRS Team".to_string(),
            supported_methods: vec![CloningMethod::FewShot, CloningMethod::OneShot],
            capabilities: PluginCapabilities::default(),
            parameters,
            dependencies: Vec::new(),
            min_api_version: "1.0.0".to_string(),
            license: Some("MIT".to_string()),
            website: Some("https://github.com/voirs/voirs".to_string()),
        };

        Self {
            config,
            initialized: false,
            parameters: HashMap::new(),
            metrics: PluginOperationMetrics::default(),
        }
    }
}

#[async_trait::async_trait]
impl CloningPlugin for ExamplePlugin {
    fn get_config(&self) -> &PluginConfig {
        &self.config
    }

    async fn initialize(&mut self, parameters: HashMap<String, ParameterValue>) -> Result<()> {
        self.parameters = parameters;
        self.initialized = true;
        info!("Example plugin initialized");
        Ok(())
    }

    async fn shutdown(&mut self) -> Result<()> {
        self.initialized = false;
        info!("Example plugin shutdown");
        Ok(())
    }

    async fn clone_voice(
        &self,
        request: VoiceCloneRequest,
        _context: PluginContext,
    ) -> Result<VoiceCloneResult> {
        if !self.initialized {
            return Err(Error::Processing("Plugin not initialized".to_string()));
        }

        let start_time = Instant::now();

        // Simulate voice cloning process
        tokio::time::sleep(Duration::from_millis(100)).await;

        let processing_time = start_time.elapsed();

        // Create dummy result
        let result = VoiceCloneResult::success(
            request.id,
            vec![0.0; 16000], // 1 second of silence
            16000,
            0.8, // Similarity score
            processing_time,
            request.method,
        )
        .with_quality_metric("example_score".to_string(), 0.75);

        Ok(result)
    }

    async fn validate_speaker_data(&self, data: &SpeakerData) -> Result<PluginValidationResult> {
        let mut result = PluginValidationResult {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            compatibility_score: 1.0,
            required_adaptations: Vec::new(),
        };

        // Basic validation
        if data.reference_samples.is_empty() {
            result.is_valid = false;
            result
                .errors
                .push("No reference samples provided".to_string());
            result.compatibility_score = 0.0;
        }

        Ok(result)
    }

    async fn health_check(&self) -> Result<PluginHealth> {
        Ok(PluginHealth {
            status: if self.initialized {
                PluginHealthStatus::Healthy
            } else {
                PluginHealthStatus::Offline
            },
            health_score: if self.initialized { 1.0 } else { 0.0 },
            memory_usage: 50, // 50MB
            cpu_usage: 5.0,   // 5%
            active_sessions: 0,
            last_check: SystemTime::now(),
            issues: Vec::new(),
            performance: PluginPerformanceMetrics::default(),
        })
    }

    async fn update_config(&mut self, parameters: HashMap<String, ParameterValue>) -> Result<()> {
        self.parameters.extend(parameters);
        info!("Example plugin configuration updated");
        Ok(())
    }

    fn get_capabilities(&self) -> &PluginCapabilities {
        &self.config.capabilities
    }

    async fn get_metrics(&self) -> Result<PluginOperationMetrics> {
        Ok(self.metrics.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;

    #[tokio::test]
    async fn test_plugin_manager_creation() {
        let config = PluginManagerConfig::default();
        let manager = PluginManager::new(config);

        let metrics = manager.get_metrics().await;
        assert_eq!(metrics.total_plugins, 0);
        assert_eq!(metrics.active_plugins, 0);
    }

    #[tokio::test]
    async fn test_example_plugin_creation() {
        let plugin = ExamplePlugin::new();
        let config = plugin.get_config();

        assert_eq!(config.id, "example_plugin");
        assert_eq!(config.name, "Example Voice Cloning Plugin");
        assert!(!config.supported_methods.is_empty());
    }

    #[tokio::test]
    async fn test_plugin_registration() {
        let config = PluginManagerConfig::default();
        let manager = PluginManager::new(config);

        let plugin = Arc::new(ExamplePlugin::new());
        let result = manager.register_plugin(plugin).await;
        assert!(result.is_ok());

        let plugins = manager.list_plugins().await;
        assert_eq!(plugins.len(), 1);
        assert_eq!(plugins[0].id, "example_plugin");
    }

    #[tokio::test]
    async fn test_plugin_initialization() {
        let mut plugin = ExamplePlugin::new();
        let mut parameters = HashMap::new();
        parameters.insert("quality_level".to_string(), ParameterValue::Float(0.9));

        let result = plugin.initialize(parameters).await;
        assert!(result.is_ok());
        assert!(plugin.initialized);
    }

    #[tokio::test]
    async fn test_plugin_health_check() {
        let mut plugin = ExamplePlugin::new();
        plugin.initialize(HashMap::new()).await.unwrap();

        let health = plugin.health_check().await.unwrap();
        assert_eq!(health.status, PluginHealthStatus::Healthy);
        assert_eq!(health.health_score, 1.0);
    }

    #[tokio::test]
    async fn test_speaker_data_validation() {
        let plugin = ExamplePlugin::new();
        let profile = SpeakerProfile::new("test".to_string(), "Test Speaker".to_string());
        let speaker_data = SpeakerData::new(profile);

        let validation = plugin.validate_speaker_data(&speaker_data).await.unwrap();
        assert!(!validation.is_valid); // Should fail due to no reference samples
        assert!(!validation.errors.is_empty());
    }

    #[tokio::test]
    async fn test_plugin_voice_cloning() {
        let mut plugin = ExamplePlugin::new();
        plugin.initialize(HashMap::new()).await.unwrap();

        let profile = SpeakerProfile::new("test".to_string(), "Test Speaker".to_string());
        let mut speaker_data = SpeakerData::new(profile);
        speaker_data.reference_samples.push(VoiceSample::new(
            "test".to_string(),
            vec![0.0; 16000],
            16000,
        ));

        let request = VoiceCloneRequest::new(
            "test_request".to_string(),
            speaker_data,
            CloningMethod::FewShot,
            "Hello world".to_string(),
        );

        let context = PluginContext {
            quality_assessor: Arc::new(CloningQualityAssessor::new().unwrap()),
            model_loader: Arc::new(ModelLoadingManager::new(Default::default())),
            global_config: Arc::new(CloningConfig::default()),
            request_metadata: HashMap::new(),
            session_id: "test_session".to_string(),
            user_context: None,
        };

        let result = plugin.clone_voice(request, context).await;
        assert!(result.is_ok());

        let clone_result = result.unwrap();
        assert!(!clone_result.audio.is_empty());
        assert!(clone_result.success);
        assert!(clone_result.similarity_score > 0.0);
    }

    #[tokio::test]
    async fn test_plugin_manager_find_best_plugin() {
        let config = PluginManagerConfig::default();
        let manager = PluginManager::new(config);

        let plugin = Arc::new(ExamplePlugin::new());
        manager.register_plugin(plugin).await.unwrap();

        let profile = SpeakerProfile::new("test".to_string(), "Test Speaker".to_string());
        let mut speaker_data = SpeakerData::new(profile);
        speaker_data.reference_samples.push(VoiceSample::new(
            "test".to_string(),
            vec![0.0; 16000],
            16000,
        ));

        let request = VoiceCloneRequest::new(
            "test_request".to_string(),
            speaker_data,
            CloningMethod::FewShot,
            "Hello world".to_string(),
        );

        let best_plugin = manager.find_best_plugin(&request).await.unwrap();
        assert!(best_plugin.is_some());
        assert_eq!(best_plugin.unwrap(), "example_plugin");
    }

    #[tokio::test]
    async fn test_plugin_registry_creation() {
        let paths = vec![PathBuf::from("./test_plugins")];
        let registry = PluginRegistry::new(paths);

        // Test discovery (will return empty results for non-existent paths)
        let manifests = registry.discover_plugins().await.unwrap();
        assert!(manifests.is_empty()); // Should be empty since test paths don't exist
    }

    #[tokio::test]
    async fn test_parameter_validation() {
        let config = PluginManagerConfig::default();
        let manager = PluginManager::new(config);

        let mut invalid_parameter = PluginParameter {
            name: "".to_string(), // Invalid empty name
            param_type: ParameterType::Float,
            description: "Test parameter".to_string(),
            default_value: ParameterValue::Float(0.5),
            constraints: None,
            required: false,
        };

        // Test validation failure
        let result = manager.validate_parameter(&invalid_parameter);
        assert!(result.is_err());

        // Fix parameter
        invalid_parameter.name = "valid_name".to_string();
        let result = manager.validate_parameter(&invalid_parameter);
        assert!(result.is_ok());
    }
}

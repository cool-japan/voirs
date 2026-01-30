//! # `VoiRS` SDK Bridge
//!
//! Deep integration bridge between voirs-recognizer and voirs-sdk.
//! Provides standardized interfaces for:
//! - Cross-crate configuration synchronization
//! - Common error handling patterns
//! - Unified performance optimization
//! - Shared resource management

use crate::config::{AsrConfig, PreprocessingConfig, RecognizerConfig};
use crate::RecognitionError;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;
use voirs_sdk::config::PipelineConfig;

/// SDK Bridge errors
#[derive(Debug, Error)]
pub enum SdkBridgeError {
    /// Configuration synchronization failed
    #[error("Configuration sync failed: {0}")]
    ConfigSyncFailed(String),

    /// SDK communication error
    #[error("SDK communication error: {0}")]
    CommunicationError(String),

    /// Resource conflict
    #[error("Resource conflict: {0}")]
    ResourceConflict(String),

    /// Compatibility issue
    #[error("Compatibility issue: {0}")]
    CompatibilityIssue(String),

    /// Recognition error wrapper
    #[error("Recognition error: {0}")]
    RecognitionError(#[from] RecognitionError),
}

/// SDK Bridge for deep ecosystem integration
#[derive(Debug)]
pub struct VoirsSdkBridge {
    /// Recognizer configuration
    recognizer_config: Arc<RwLock<RecognizerConfig>>,

    /// SDK pipeline configuration
    sdk_config: Arc<RwLock<PipelineConfig>>,

    /// Cross-crate shared state
    shared_state: Arc<RwLock<SharedState>>,

    /// Performance optimization settings
    optimization_settings: Arc<RwLock<OptimizationSettings>>,

    /// Error mapping registry
    error_registry: Arc<RwLock<ErrorRegistry>>,
}

/// Shared state across `VoiRS` crates
#[derive(Debug, Clone, Default)]
pub struct SharedState {
    /// Shared model cache paths
    pub model_cache_paths: HashMap<String, String>,

    /// Shared audio buffer pools
    pub audio_buffer_pool_size: usize,

    /// Shared GPU device IDs
    pub gpu_device_ids: Vec<i32>,

    /// Shared thread pool size
    pub thread_pool_size: usize,

    /// Feature flags enabled across crates
    pub feature_flags: HashMap<String, bool>,

    /// Resource quotas
    pub resource_quotas: ResourceQuotas,
}

/// Resource quotas for cross-crate resource management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceQuotas {
    /// Maximum total memory allocation in MB
    pub max_memory_mb: f32,

    /// Maximum GPU memory allocation in MB
    pub max_gpu_memory_mb: Option<f32>,

    /// Maximum concurrent operations
    pub max_concurrent_operations: usize,

    /// Maximum cache size in MB
    pub max_cache_size_mb: f32,
}

impl Default for ResourceQuotas {
    fn default() -> Self {
        Self {
            max_memory_mb: 2048.0,
            max_gpu_memory_mb: Some(4096.0),
            max_concurrent_operations: 4,
            max_cache_size_mb: 512.0,
        }
    }
}

/// Cross-crate optimization settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSettings {
    /// Enable cross-crate model sharing
    pub enable_model_sharing: bool,

    /// Enable unified memory pooling
    pub enable_unified_memory_pool: bool,

    /// Enable cross-crate caching
    pub enable_cross_crate_cache: bool,

    /// Enable GPU resource pooling
    pub enable_gpu_pooling: bool,

    /// Batch size coordination
    pub coordinated_batch_size: Option<usize>,

    /// Unified precision policy (fp32, fp16, int8)
    pub unified_precision: PrecisionPolicy,

    /// Enable parallel pipeline execution
    pub enable_parallel_pipelines: bool,

    /// Memory pressure threshold (0.0-1.0)
    pub memory_pressure_threshold: f32,
}

impl Default for OptimizationSettings {
    fn default() -> Self {
        Self {
            enable_model_sharing: true,
            enable_unified_memory_pool: true,
            enable_cross_crate_cache: true,
            enable_gpu_pooling: true,
            coordinated_batch_size: Some(4),
            unified_precision: PrecisionPolicy::Mixed,
            enable_parallel_pipelines: false,
            memory_pressure_threshold: 0.8,
        }
    }
}

/// Unified precision policy across crates
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrecisionPolicy {
    /// Full precision (FP32)
    Full,
    /// Half precision (FP16)
    Half,
    /// Integer quantization (INT8)
    Quantized,
    /// Mixed precision (FP32 + FP16)
    Mixed,
}

/// Error registry for standardized error handling
#[derive(Default)]
pub struct ErrorRegistry {
    /// Error code mappings
    error_codes: HashMap<String, ErrorCode>,

    /// Error handlers
    handlers: HashMap<String, ErrorHandler>,
}

impl std::fmt::Debug for ErrorRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ErrorRegistry")
            .field("error_codes", &self.error_codes)
            .field("handlers_count", &self.handlers.len())
            .finish()
    }
}

/// Standardized error codes across `VoiRS` crates
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ErrorCode {
    /// Success
    Success = 0,

    /// Generic error
    GenericError = 1000,

    /// Configuration error
    ConfigError = 2000,

    /// Model loading error
    ModelLoadError = 3000,

    /// Inference error
    InferenceError = 4000,

    /// Audio processing error
    AudioProcessingError = 5000,

    /// Memory allocation error
    MemoryError = 6000,

    /// GPU error
    GpuError = 7000,

    /// I/O error
    IoError = 8000,

    /// Timeout error
    TimeoutError = 9000,

    /// Resource exhausted error
    ResourceExhausted = 10000,
}

/// Error handler function type
pub type ErrorHandler = Arc<dyn Fn(&str) + Send + Sync>;

impl VoirsSdkBridge {
    /// Create a new SDK bridge
    pub fn new(
        recognizer_config: RecognizerConfig,
        sdk_config: PipelineConfig,
    ) -> Result<Self, SdkBridgeError> {
        Ok(Self {
            recognizer_config: Arc::new(RwLock::new(recognizer_config)),
            sdk_config: Arc::new(RwLock::new(sdk_config)),
            shared_state: Arc::new(RwLock::new(SharedState::default())),
            optimization_settings: Arc::new(RwLock::new(OptimizationSettings::default())),
            error_registry: Arc::new(RwLock::new(ErrorRegistry::default())),
        })
    }

    /// Synchronize configuration from SDK to recognizer
    pub fn sync_config_from_sdk(&self) -> Result<(), SdkBridgeError> {
        let sdk_cfg = self.sdk_config.read();
        let mut recognizer_cfg = self.recognizer_config.write();

        // Synchronize common settings
        let device_str = sdk_cfg.device.to_lowercase();
        recognizer_cfg.asr.enable_gpu = sdk_cfg.use_gpu
            || device_str.contains("gpu")
            || device_str.contains("cuda")
            || device_str.contains("metal");

        // Sync thread count if available
        if let Some(num_threads) = sdk_cfg.num_threads {
            recognizer_cfg.performance.num_threads = num_threads;
        }

        Ok(())
    }

    /// Synchronize configuration from recognizer to SDK
    pub fn sync_config_to_sdk(&self) -> Result<(), SdkBridgeError> {
        let recognizer_cfg = self.recognizer_config.read();
        let mut _sdk_cfg = self.sdk_config.write();

        // Synchronize ASR settings back to SDK
        // This would update the SDK configuration based on recognizer settings
        tracing::debug!(
            "Syncing recognizer config to SDK: GPU={}, batch_size={}",
            recognizer_cfg.asr.enable_gpu,
            recognizer_cfg.asr.batch_size
        );

        Ok(())
    }

    /// Get shared state
    #[must_use]
    pub fn shared_state(&self) -> Arc<RwLock<SharedState>> {
        Arc::clone(&self.shared_state)
    }

    /// Get optimization settings
    #[must_use]
    pub fn optimization_settings(&self) -> Arc<RwLock<OptimizationSettings>> {
        Arc::clone(&self.optimization_settings)
    }

    /// Update shared state
    pub fn update_shared_state<F>(&self, updater: F) -> Result<(), SdkBridgeError>
    where
        F: FnOnce(&mut SharedState),
    {
        let mut state = self.shared_state.write();
        updater(&mut state);
        Ok(())
    }

    /// Update optimization settings
    pub fn update_optimization_settings<F>(&self, updater: F) -> Result<(), SdkBridgeError>
    where
        F: FnOnce(&mut OptimizationSettings),
    {
        let mut settings = self.optimization_settings.write();
        updater(&mut settings);
        Ok(())
    }

    /// Register error code mapping
    pub fn register_error_code(&self, error_name: String, code: ErrorCode) {
        let mut registry = self.error_registry.write();
        registry.error_codes.insert(error_name, code);
    }

    /// Register error handler
    pub fn register_error_handler(&self, error_type: String, handler: ErrorHandler) {
        let mut registry = self.error_registry.write();
        registry.handlers.insert(error_type, handler);
    }

    /// Get error code for error name
    #[must_use]
    pub fn get_error_code(&self, error_name: &str) -> Option<ErrorCode> {
        let registry = self.error_registry.read();
        registry.error_codes.get(error_name).copied()
    }

    /// Handle error with registered handlers
    pub fn handle_error(&self, error_type: &str, message: &str) {
        let registry = self.error_registry.read();
        if let Some(handler) = registry.handlers.get(error_type) {
            handler(message);
        } else {
            tracing::error!("Unhandled error [{}]: {}", error_type, message);
        }
    }

    /// Get recognizer configuration (read-only)
    #[must_use]
    pub fn recognizer_config(&self) -> Arc<RwLock<RecognizerConfig>> {
        Arc::clone(&self.recognizer_config)
    }

    /// Get SDK configuration (read-only)
    #[must_use]
    pub fn sdk_config(&self) -> Arc<RwLock<PipelineConfig>> {
        Arc::clone(&self.sdk_config)
    }

    /// Check resource availability
    #[must_use]
    pub fn check_resource_availability(&self) -> ResourceAvailability {
        let state = self.shared_state.read();
        let quotas = &state.resource_quotas;

        ResourceAvailability {
            memory_available_mb: quotas.max_memory_mb * 0.5, // Simplified calculation
            gpu_memory_available_mb: quotas.max_gpu_memory_mb.map(|m| m * 0.5),
            concurrent_slots_available: quotas.max_concurrent_operations / 2,
            cache_space_available_mb: quotas.max_cache_size_mb * 0.5,
        }
    }

    /// Allocate resources for an operation
    pub fn allocate_resources(
        &self,
        requirements: &ResourceRequirements,
    ) -> Result<ResourceAllocation, SdkBridgeError> {
        let availability = self.check_resource_availability();

        // Check if resources are available
        if requirements.memory_mb > availability.memory_available_mb {
            return Err(SdkBridgeError::ResourceConflict(
                "Insufficient memory".to_string(),
            ));
        }

        if let (Some(required_gpu), Some(available_gpu)) = (
            requirements.gpu_memory_mb,
            availability.gpu_memory_available_mb,
        ) {
            if required_gpu > available_gpu {
                return Err(SdkBridgeError::ResourceConflict(
                    "Insufficient GPU memory".to_string(),
                ));
            }
        }

        Ok(ResourceAllocation {
            allocation_id: uuid::Uuid::new_v4().to_string(),
            memory_allocated_mb: requirements.memory_mb,
            gpu_memory_allocated_mb: requirements.gpu_memory_mb,
            concurrent_slot: true,
        })
    }

    /// Release allocated resources
    pub fn release_resources(&self, allocation: &ResourceAllocation) {
        tracing::debug!(
            "Releasing resources: allocation_id={}, memory={} MB",
            allocation.allocation_id,
            allocation.memory_allocated_mb
        );
        // In a real implementation, this would update resource tracking
    }
}

/// Resource availability information
#[derive(Debug, Clone)]
pub struct ResourceAvailability {
    /// Available memory in MB
    pub memory_available_mb: f32,

    /// Available GPU memory in MB
    pub gpu_memory_available_mb: Option<f32>,

    /// Available concurrent operation slots
    pub concurrent_slots_available: usize,

    /// Available cache space in MB
    pub cache_space_available_mb: f32,
}

/// Resource requirements
#[derive(Debug, Clone)]
pub struct ResourceRequirements {
    /// Memory requirement in MB
    pub memory_mb: f32,

    /// GPU memory requirement in MB
    pub gpu_memory_mb: Option<f32>,

    /// Requires concurrent slot
    pub requires_concurrent_slot: bool,
}

/// Resource allocation result
#[derive(Debug, Clone)]
pub struct ResourceAllocation {
    /// Allocation ID
    pub allocation_id: String,

    /// Memory allocated in MB
    pub memory_allocated_mb: f32,

    /// GPU memory allocated in MB
    pub gpu_memory_allocated_mb: Option<f32>,

    /// Concurrent slot allocated
    pub concurrent_slot: bool,
}

/// Builder for SDK bridge configuration
pub struct SdkBridgeBuilder {
    recognizer_config: Option<RecognizerConfig>,
    sdk_config: Option<PipelineConfig>,
    optimization_settings: OptimizationSettings,
    resource_quotas: ResourceQuotas,
}

impl SdkBridgeBuilder {
    /// Create a new builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            recognizer_config: None,
            sdk_config: None,
            optimization_settings: OptimizationSettings::default(),
            resource_quotas: ResourceQuotas::default(),
        }
    }

    /// Set recognizer configuration
    #[must_use]
    pub fn with_recognizer_config(mut self, config: RecognizerConfig) -> Self {
        self.recognizer_config = Some(config);
        self
    }

    /// Set SDK configuration
    #[must_use]
    pub fn with_sdk_config(mut self, config: PipelineConfig) -> Self {
        self.sdk_config = Some(config);
        self
    }

    /// Set optimization settings
    #[must_use]
    pub fn with_optimization_settings(mut self, settings: OptimizationSettings) -> Self {
        self.optimization_settings = settings;
        self
    }

    /// Set resource quotas
    #[must_use]
    pub fn with_resource_quotas(mut self, quotas: ResourceQuotas) -> Self {
        self.resource_quotas = quotas;
        self
    }

    /// Build the SDK bridge
    pub fn build(self) -> Result<VoirsSdkBridge, SdkBridgeError> {
        let recognizer_config = self.recognizer_config.unwrap_or_default();
        let sdk_config = self.sdk_config.unwrap_or_default();

        let bridge = VoirsSdkBridge::new(recognizer_config, sdk_config)?;

        // Apply optimization settings
        bridge.update_optimization_settings(|settings| {
            *settings = self.optimization_settings.clone();
        })?;

        // Apply resource quotas
        bridge.update_shared_state(|state| {
            state.resource_quotas = self.resource_quotas.clone();
        })?;

        Ok(bridge)
    }
}

impl Default for SdkBridgeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sdk_bridge_creation() {
        let bridge = VoirsSdkBridge::new(RecognizerConfig::default(), PipelineConfig::default());
        assert!(bridge.is_ok());
    }

    #[test]
    fn test_config_synchronization() {
        let bridge =
            VoirsSdkBridge::new(RecognizerConfig::default(), PipelineConfig::default()).unwrap();

        assert!(bridge.sync_config_from_sdk().is_ok());
        assert!(bridge.sync_config_to_sdk().is_ok());
    }

    #[test]
    fn test_shared_state_update() {
        let bridge =
            VoirsSdkBridge::new(RecognizerConfig::default(), PipelineConfig::default()).unwrap();

        let result = bridge.update_shared_state(|state| {
            state.thread_pool_size = 8;
        });

        assert!(result.is_ok());
        assert_eq!(bridge.shared_state().read().thread_pool_size, 8);
    }

    #[test]
    fn test_optimization_settings_update() {
        let bridge =
            VoirsSdkBridge::new(RecognizerConfig::default(), PipelineConfig::default()).unwrap();

        let result = bridge.update_optimization_settings(|settings| {
            settings.enable_model_sharing = false;
        });

        assert!(result.is_ok());
        assert!(!bridge.optimization_settings().read().enable_model_sharing);
    }

    #[test]
    fn test_error_registry() {
        let bridge =
            VoirsSdkBridge::new(RecognizerConfig::default(), PipelineConfig::default()).unwrap();

        bridge.register_error_code("test_error".to_string(), ErrorCode::GenericError);

        let code = bridge.get_error_code("test_error");
        assert_eq!(code, Some(ErrorCode::GenericError));
    }

    #[test]
    fn test_resource_allocation() {
        let bridge =
            VoirsSdkBridge::new(RecognizerConfig::default(), PipelineConfig::default()).unwrap();

        let requirements = ResourceRequirements {
            memory_mb: 100.0,
            gpu_memory_mb: Some(200.0),
            requires_concurrent_slot: true,
        };

        let allocation = bridge.allocate_resources(&requirements);
        assert!(allocation.is_ok());

        if let Ok(alloc) = allocation {
            bridge.release_resources(&alloc);
        }
    }

    #[test]
    fn test_resource_exhaustion() {
        let bridge =
            VoirsSdkBridge::new(RecognizerConfig::default(), PipelineConfig::default()).unwrap();

        // Try to allocate more memory than available
        let requirements = ResourceRequirements {
            memory_mb: 10000.0, // More than default quota
            gpu_memory_mb: None,
            requires_concurrent_slot: true,
        };

        let allocation = bridge.allocate_resources(&requirements);
        assert!(allocation.is_err());
    }

    #[test]
    fn test_builder_pattern() {
        let builder = SdkBridgeBuilder::new()
            .with_recognizer_config(RecognizerConfig::default())
            .with_sdk_config(PipelineConfig::default())
            .with_optimization_settings(OptimizationSettings {
                enable_model_sharing: false,
                ..OptimizationSettings::default()
            });

        let bridge = builder.build();
        assert!(bridge.is_ok());

        if let Ok(b) = bridge {
            assert!(!b.optimization_settings().read().enable_model_sharing);
        }
    }
}

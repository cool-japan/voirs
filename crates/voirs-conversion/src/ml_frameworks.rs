//! # ML Frameworks Integration Module
//!
//! This module provides integration with the latest machine learning frameworks
//! for voice conversion, including Candle, ONNX Runtime, TensorFlow Lite, and PyTorch.

use crate::{Error, Result};
use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

/// Supported ML framework types
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum MLFramework {
    /// Candle framework (Rust-native)
    Candle,
    /// ONNX Runtime
    OnnxRuntime,
    /// TensorFlow Lite
    TensorFlowLite,
    /// PyTorch (via Candle integration)
    PyTorch,
    /// Custom framework implementation
    Custom,
}

/// ML framework configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLFrameworkConfig {
    /// Primary framework to use
    pub primary_framework: MLFramework,
    /// Fallback frameworks in order of preference
    pub fallback_frameworks: Vec<MLFramework>,
    /// Device preference (CPU, GPU, etc.)
    pub device_preference: DevicePreference,
    /// Model optimization settings
    pub optimization: ModelOptimization,
    /// Memory management settings
    pub memory_config: MemoryConfig,
    /// Performance tuning settings
    pub performance_config: PerformanceConfig,
    /// Framework-specific settings
    pub framework_settings: HashMap<MLFramework, FrameworkSettings>,
}

/// Device preference for ML computations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DevicePreference {
    /// Prefer CPU computation
    Cpu,
    /// Prefer GPU computation (CUDA, Metal, etc.)
    Gpu {
        /// GPU device index
        device_index: Option<usize>,
        /// Memory limit in MB
        memory_limit_mb: Option<usize>,
    },
    /// Automatic device selection
    Auto,
    /// Custom device specification
    Custom {
        /// Device identifier
        device_id: String,
        /// Device capabilities
        capabilities: HashMap<String, String>,
    },
}

/// Model optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelOptimization {
    /// Enable quantization
    pub quantization_enabled: bool,
    /// Quantization precision
    pub quantization_precision: QuantizationPrecision,
    /// Enable pruning
    pub pruning_enabled: bool,
    /// Pruning ratio (0.0 - 1.0)
    pub pruning_ratio: f32,
    /// Enable knowledge distillation
    pub distillation_enabled: bool,
    /// Enable operator fusion
    pub operator_fusion: bool,
    /// Enable constant folding
    pub constant_folding: bool,
}

/// Quantization precision options
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum QuantizationPrecision {
    /// 8-bit integer quantization
    Int8,
    /// 16-bit integer quantization
    Int16,
    /// 16-bit floating point
    Float16,
    /// Dynamic quantization
    Dynamic,
}

/// Memory management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Maximum memory usage in MB
    pub max_memory_mb: usize,
    /// Memory pool size for intermediate tensors
    pub memory_pool_size_mb: usize,
    /// Enable memory optimization
    pub memory_optimization_enabled: bool,
    /// Garbage collection frequency
    pub gc_frequency: usize,
    /// Enable memory mapping for large models
    pub memory_mapping_enabled: bool,
}

/// Performance tuning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Number of threads for CPU inference
    pub cpu_threads: Option<usize>,
    /// Batch size for inference
    pub batch_size: usize,
    /// Enable asynchronous execution
    pub async_execution: bool,
    /// Prefetch buffer size
    pub prefetch_buffer_size: usize,
    /// Enable pipeline parallelism
    pub pipeline_parallelism: bool,
    /// Cache compiled models
    pub model_caching: bool,
}

/// Framework-specific settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameworkSettings {
    /// Library path or configuration
    pub library_path: Option<PathBuf>,
    /// Custom initialization parameters
    pub init_params: HashMap<String, String>,
    /// Provider-specific options
    pub provider_options: HashMap<String, String>,
    /// Session configuration
    pub session_config: HashMap<String, String>,
}

/// ML model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLModelMetadata {
    /// Model name
    pub name: String,
    /// Model version
    pub version: String,
    /// Framework this model was trained with
    pub framework: MLFramework,
    /// Input tensor specifications
    pub input_specs: Vec<TensorSpec>,
    /// Output tensor specifications
    pub output_specs: Vec<TensorSpec>,
    /// Model file path
    pub model_path: PathBuf,
    /// Model size in bytes
    pub model_size_bytes: u64,
    /// Supported sample rates
    pub supported_sample_rates: Vec<u32>,
    /// Model capabilities
    pub capabilities: ModelCapabilities,
}

/// Tensor specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorSpec {
    /// Tensor name
    pub name: String,
    /// Tensor shape (-1 for dynamic dimensions)
    pub shape: Vec<i64>,
    /// Data type
    pub data_type: TensorDataType,
    /// Optional description
    pub description: Option<String>,
}

/// Supported tensor data types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TensorDataType {
    Float32,
    Float64,
    Int32,
    Int64,
    UInt8,
    Int8,
    Float16,
}

/// Model capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCapabilities {
    /// Supports real-time processing
    pub realtime_capable: bool,
    /// Supports batch processing
    pub batch_capable: bool,
    /// Supports streaming input
    pub streaming_capable: bool,
    /// GPU acceleration support
    pub gpu_accelerated: bool,
    /// Quantization support
    pub quantization_support: bool,
    /// Maximum input length
    pub max_input_length: Option<usize>,
}

/// ML inference session
pub struct MLInferenceSession {
    /// Framework being used
    framework: MLFramework,
    /// Model metadata
    model_metadata: MLModelMetadata,
    /// Candle-specific session
    candle_session: Option<CandleSession>,
    /// Framework configuration
    config: MLFrameworkConfig,
    /// Performance metrics
    metrics: Arc<RwLock<InferenceMetrics>>,
}

/// Candle-specific inference session
pub struct CandleSession {
    /// Candle device
    device: Device,
    /// Loaded model tensors/weights
    model_weights: HashMap<String, Tensor>,
    /// Model architecture
    model_architecture: ModelArchitecture,
}

/// Model architecture for Candle
#[derive(Debug, Clone)]
pub enum ModelArchitecture {
    /// Transformer-based architecture
    Transformer {
        /// Number of layers
        num_layers: usize,
        /// Hidden dimension
        hidden_dim: usize,
        /// Number of attention heads
        num_heads: usize,
    },
    /// Convolutional neural network
    Cnn {
        /// Convolution layers configuration
        conv_layers: Vec<ConvLayerConfig>,
        /// Fully connected layers
        fc_layers: Vec<usize>,
    },
    /// Recurrent neural network
    Rnn {
        /// RNN type (LSTM, GRU, etc.)
        rnn_type: RnnType,
        /// Hidden size
        hidden_size: usize,
        /// Number of layers
        num_layers: usize,
        /// Bidirectional
        bidirectional: bool,
    },
    /// Custom architecture
    Custom {
        /// Architecture description
        description: String,
        /// Layer specifications
        layers: Vec<LayerSpec>,
    },
}

/// Convolution layer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvLayerConfig {
    /// Input channels
    pub in_channels: usize,
    /// Output channels
    pub out_channels: usize,
    /// Kernel size
    pub kernel_size: usize,
    /// Stride
    pub stride: usize,
    /// Padding
    pub padding: usize,
    /// Activation function
    pub activation: ActivationFunction,
}

/// RNN types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum RnnType {
    /// Long Short-Term Memory
    Lstm,
    /// Gated Recurrent Unit
    Gru,
    /// Vanilla RNN
    Vanilla,
}

/// Activation functions
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ActivationFunction {
    ReLU,
    LeakyReLU,
    Tanh,
    Sigmoid,
    Swish,
    GELU,
    Mish,
}

/// Layer specification for custom architectures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerSpec {
    /// Layer type
    pub layer_type: String,
    /// Layer parameters
    pub parameters: HashMap<String, f32>,
    /// Input shape
    pub input_shape: Vec<usize>,
    /// Output shape
    pub output_shape: Vec<usize>,
}

/// Inference performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceMetrics {
    /// Total inference count
    pub inference_count: u64,
    /// Total inference time in milliseconds
    pub total_inference_time_ms: u64,
    /// Average inference time in milliseconds
    pub avg_inference_time_ms: f32,
    /// Minimum inference time in milliseconds
    pub min_inference_time_ms: f32,
    /// Maximum inference time in milliseconds
    pub max_inference_time_ms: f32,
    /// Memory usage statistics
    pub memory_usage: MemoryUsageStats,
    /// Error count
    pub error_count: u64,
    /// Last update timestamp
    pub last_update: std::time::SystemTime,
}

impl Default for InferenceMetrics {
    fn default() -> Self {
        Self {
            inference_count: 0,
            total_inference_time_ms: 0,
            avg_inference_time_ms: 0.0,
            min_inference_time_ms: 0.0,
            max_inference_time_ms: 0.0,
            memory_usage: MemoryUsageStats::default(),
            error_count: 0,
            last_update: std::time::SystemTime::now(),
        }
    }
}

/// Memory usage statistics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct MemoryUsageStats {
    /// Peak memory usage in bytes
    pub peak_usage_bytes: u64,
    /// Current memory usage in bytes
    pub current_usage_bytes: u64,
    /// Average memory usage in bytes
    pub avg_usage_bytes: u64,
    /// Memory allocations count
    pub allocation_count: u64,
}

/// ML framework manager
pub struct MLFrameworkManager {
    /// Available frameworks
    frameworks: HashMap<MLFramework, FrameworkInfo>,
    /// Active sessions
    active_sessions: Arc<RwLock<HashMap<String, MLInferenceSession>>>,
    /// Configuration
    config: MLFrameworkConfig,
    /// Model registry
    model_registry: Arc<RwLock<HashMap<String, MLModelMetadata>>>,
}

/// Framework information
#[derive(Debug, Clone)]
pub struct FrameworkInfo {
    /// Framework version
    version: String,
    /// Available providers
    providers: Vec<String>,
    /// Initialization status
    initialized: bool,
    /// Capabilities
    capabilities: FrameworkCapabilities,
}

/// Framework capabilities
#[derive(Debug, Clone)]
pub struct FrameworkCapabilities {
    /// GPU support
    gpu_support: bool,
    /// Quantization support
    quantization_support: bool,
    /// Dynamic shapes support
    dynamic_shapes: bool,
    /// Streaming support
    streaming_support: bool,
}

impl Default for MLFrameworkConfig {
    fn default() -> Self {
        Self {
            primary_framework: MLFramework::Candle,
            fallback_frameworks: vec![MLFramework::OnnxRuntime],
            device_preference: DevicePreference::Auto,
            optimization: ModelOptimization::default(),
            memory_config: MemoryConfig::default(),
            performance_config: PerformanceConfig::default(),
            framework_settings: HashMap::new(),
        }
    }
}

impl Default for ModelOptimization {
    fn default() -> Self {
        Self {
            quantization_enabled: true,
            quantization_precision: QuantizationPrecision::Int8,
            pruning_enabled: false,
            pruning_ratio: 0.1,
            distillation_enabled: false,
            operator_fusion: true,
            constant_folding: true,
        }
    }
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            max_memory_mb: 4096,      // 4GB
            memory_pool_size_mb: 512, // 512MB
            memory_optimization_enabled: true,
            gc_frequency: 100,
            memory_mapping_enabled: true,
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            cpu_threads: None, // Auto-detect
            batch_size: 1,
            async_execution: true,
            prefetch_buffer_size: 4,
            pipeline_parallelism: false,
            model_caching: true,
        }
    }
}

impl MLFrameworkManager {
    /// Create new ML framework manager
    pub fn new(config: MLFrameworkConfig) -> Result<Self> {
        let mut frameworks = HashMap::new();

        // Initialize Candle framework (always available)
        frameworks.insert(
            MLFramework::Candle,
            FrameworkInfo {
                version: env!("CARGO_PKG_VERSION").to_string(),
                providers: vec!["CPU".to_string(), "CUDA".to_string(), "Metal".to_string()],
                initialized: true,
                capabilities: FrameworkCapabilities {
                    gpu_support: true,
                    quantization_support: true,
                    dynamic_shapes: true,
                    streaming_support: true,
                },
            },
        );

        // Initialize other frameworks (placeholder implementations)
        frameworks.insert(
            MLFramework::OnnxRuntime,
            FrameworkInfo {
                version: "1.16.0".to_string(),
                providers: vec!["CPU".to_string(), "CUDA".to_string()],
                initialized: false, // Would check if ONNX Runtime is available
                capabilities: FrameworkCapabilities {
                    gpu_support: true,
                    quantization_support: true,
                    dynamic_shapes: true,
                    streaming_support: false,
                },
            },
        );

        Ok(Self {
            frameworks,
            active_sessions: Arc::new(RwLock::new(HashMap::new())),
            config,
            model_registry: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Register a new model
    pub fn register_model(&self, metadata: MLModelMetadata) -> Result<()> {
        let mut registry = self.model_registry.write().map_err(|_| {
            Error::runtime("Failed to acquire write lock on model registry".to_string())
        })?;

        registry.insert(metadata.name.clone(), metadata);
        Ok(())
    }

    /// Create inference session for a model
    pub fn create_session(&self, model_name: &str, session_id: String) -> Result<()> {
        let model_metadata = {
            let registry = self.model_registry.read().map_err(|_| {
                Error::runtime("Failed to acquire read lock on model registry".to_string())
            })?;

            registry.get(model_name).cloned().ok_or_else(|| {
                Error::model(format!("Model '{model_name}' not found in registry"))
            })?
        };

        // Select framework based on configuration and model requirements
        let framework = self.select_framework(&model_metadata)?;

        let session = match framework {
            MLFramework::Candle => self.create_candle_session(&model_metadata)?,
            MLFramework::OnnxRuntime => self.create_onnx_session(&model_metadata)?,
            MLFramework::TensorFlowLite => self.create_tflite_session(&model_metadata)?,
            MLFramework::PyTorch => self.create_pytorch_session(&model_metadata)?,
            MLFramework::Custom => self.create_custom_session(&model_metadata)?,
        };

        let mut sessions = self.active_sessions.write().map_err(|_| {
            Error::runtime("Failed to acquire write lock on active sessions".to_string())
        })?;

        sessions.insert(session_id, session);
        Ok(())
    }

    /// Run inference on a session
    pub fn run_inference(&self, session_id: &str, inputs: &[Tensor]) -> Result<Vec<Tensor>> {
        let mut sessions = self.active_sessions.write().map_err(|_| {
            Error::runtime("Failed to acquire write lock on active sessions".to_string())
        })?;

        let session = sessions
            .get_mut(session_id)
            .ok_or_else(|| Error::runtime(format!("Session '{session_id}' not found")))?;

        let start_time = std::time::Instant::now();

        let outputs = match session.framework {
            MLFramework::Candle => self.run_candle_inference(session, inputs)?,
            MLFramework::OnnxRuntime => self.run_onnx_inference(session, inputs)?,
            MLFramework::TensorFlowLite => self.run_tflite_inference(session, inputs)?,
            MLFramework::PyTorch => self.run_pytorch_inference(session, inputs)?,
            MLFramework::Custom => self.run_custom_inference(session, inputs)?,
        };

        let inference_time = start_time.elapsed();

        // Update metrics
        {
            let mut metrics = session.metrics.write().map_err(|_| {
                Error::runtime("Failed to acquire write lock on metrics".to_string())
            })?;

            metrics.inference_count += 1;
            let inference_time_ms = inference_time.as_millis() as u64;
            metrics.total_inference_time_ms += inference_time_ms;
            metrics.avg_inference_time_ms =
                metrics.total_inference_time_ms as f32 / metrics.inference_count as f32;

            if metrics.inference_count == 1 {
                metrics.min_inference_time_ms = inference_time_ms as f32;
                metrics.max_inference_time_ms = inference_time_ms as f32;
            } else {
                metrics.min_inference_time_ms =
                    metrics.min_inference_time_ms.min(inference_time_ms as f32);
                metrics.max_inference_time_ms =
                    metrics.max_inference_time_ms.max(inference_time_ms as f32);
            }

            metrics.last_update = std::time::SystemTime::now();
        }

        Ok(outputs)
    }

    /// Select appropriate framework for a model
    fn select_framework(&self, model_metadata: &MLModelMetadata) -> Result<MLFramework> {
        // Check if primary framework supports the model
        if self.framework_supports_model(self.config.primary_framework, model_metadata)? {
            return Ok(self.config.primary_framework);
        }

        // Try fallback frameworks
        for &framework in &self.config.fallback_frameworks {
            if self.framework_supports_model(framework, model_metadata)? {
                return Ok(framework);
            }
        }

        Err(Error::model(format!(
            "No compatible framework found for model '{}'",
            model_metadata.name
        )))
    }

    /// Check if framework supports a model
    fn framework_supports_model(
        &self,
        framework: MLFramework,
        model_metadata: &MLModelMetadata,
    ) -> Result<bool> {
        let framework_info = self
            .frameworks
            .get(&framework)
            .ok_or_else(|| Error::model(format!("Framework {framework:?} not available")))?;

        if !framework_info.initialized {
            return Ok(false);
        }

        // Check framework compatibility with model
        match (framework, model_metadata.framework) {
            (MLFramework::Candle, _) => Ok(true), // Candle can handle most formats
            (a, b) if a == b => Ok(true),         // Same framework
            (MLFramework::OnnxRuntime, MLFramework::PyTorch) => Ok(true), // ONNX can run PyTorch models
            (MLFramework::OnnxRuntime, MLFramework::TensorFlowLite) => Ok(true), // ONNX can run TF models
            _ => Ok(false),
        }
    }

    /// Create Candle inference session
    fn create_candle_session(
        &self,
        model_metadata: &MLModelMetadata,
    ) -> Result<MLInferenceSession> {
        let device = match &self.config.device_preference {
            DevicePreference::Cpu => Device::Cpu,
            DevicePreference::Gpu { device_index, .. } => match device_index {
                Some(idx) => Device::cuda_if_available(*idx).map_err(|e| {
                    Error::model(format!("Failed to create CUDA device {idx}: {e}"))
                })?,
                None => Device::cuda_if_available(0)
                    .map_err(|e| Error::model(format!("Failed to create CUDA device: {e}")))?,
            },
            DevicePreference::Auto => {
                if Device::cuda_if_available(0).is_ok() {
                    Device::cuda_if_available(0)
                        .map_err(|e| Error::model(format!("Failed to create CUDA device: {e}")))?
                } else {
                    Device::Cpu
                }
            }
            DevicePreference::Custom { .. } => Device::Cpu, // Fallback to CPU for custom
        };

        // Load model weights (placeholder - would load actual model file)
        let model_weights = HashMap::new();

        // Create model architecture (placeholder - would parse from model file)
        let model_architecture = ModelArchitecture::Transformer {
            num_layers: 12,
            hidden_dim: 768,
            num_heads: 12,
        };

        let candle_session = CandleSession {
            device,
            model_weights,
            model_architecture,
        };

        Ok(MLInferenceSession {
            framework: MLFramework::Candle,
            model_metadata: model_metadata.clone(),
            candle_session: Some(candle_session),
            config: self.config.clone(),
            metrics: Arc::new(RwLock::new(InferenceMetrics::default())),
        })
    }

    /// Create ONNX Runtime session (placeholder)
    fn create_onnx_session(&self, model_metadata: &MLModelMetadata) -> Result<MLInferenceSession> {
        // Placeholder implementation - would use actual ONNX Runtime bindings
        Ok(MLInferenceSession {
            framework: MLFramework::OnnxRuntime,
            model_metadata: model_metadata.clone(),
            candle_session: None,
            config: self.config.clone(),
            metrics: Arc::new(RwLock::new(InferenceMetrics::default())),
        })
    }

    /// Create TensorFlow Lite session (placeholder)
    fn create_tflite_session(
        &self,
        model_metadata: &MLModelMetadata,
    ) -> Result<MLInferenceSession> {
        // Placeholder implementation - would use actual TensorFlow Lite bindings
        Ok(MLInferenceSession {
            framework: MLFramework::TensorFlowLite,
            model_metadata: model_metadata.clone(),
            candle_session: None,
            config: self.config.clone(),
            metrics: Arc::new(RwLock::new(InferenceMetrics::default())),
        })
    }

    /// Create PyTorch session (placeholder)
    fn create_pytorch_session(
        &self,
        model_metadata: &MLModelMetadata,
    ) -> Result<MLInferenceSession> {
        // Placeholder implementation - would use actual PyTorch bindings
        Ok(MLInferenceSession {
            framework: MLFramework::PyTorch,
            model_metadata: model_metadata.clone(),
            candle_session: None,
            config: self.config.clone(),
            metrics: Arc::new(RwLock::new(InferenceMetrics::default())),
        })
    }

    /// Create custom framework session (placeholder)
    fn create_custom_session(
        &self,
        model_metadata: &MLModelMetadata,
    ) -> Result<MLInferenceSession> {
        // Placeholder implementation - would use custom framework
        Ok(MLInferenceSession {
            framework: MLFramework::Custom,
            model_metadata: model_metadata.clone(),
            candle_session: None,
            config: self.config.clone(),
            metrics: Arc::new(RwLock::new(InferenceMetrics::default())),
        })
    }

    /// Run Candle inference
    fn run_candle_inference(
        &self,
        session: &MLInferenceSession,
        inputs: &[Tensor],
    ) -> Result<Vec<Tensor>> {
        // Placeholder implementation - would run actual model inference
        let candle_session = session
            .candle_session
            .as_ref()
            .ok_or_else(|| Error::model("Candle session not initialized".to_string()))?;

        // Simple passthrough for now - would implement actual model forward pass
        Ok(inputs.to_vec())
    }

    /// Run ONNX Runtime inference (placeholder)
    fn run_onnx_inference(
        &self,
        _session: &MLInferenceSession,
        inputs: &[Tensor],
    ) -> Result<Vec<Tensor>> {
        // Placeholder implementation
        Ok(inputs.to_vec())
    }

    /// Run TensorFlow Lite inference (placeholder)
    fn run_tflite_inference(
        &self,
        _session: &MLInferenceSession,
        inputs: &[Tensor],
    ) -> Result<Vec<Tensor>> {
        // Placeholder implementation
        Ok(inputs.to_vec())
    }

    /// Run PyTorch inference (placeholder)
    fn run_pytorch_inference(
        &self,
        _session: &MLInferenceSession,
        inputs: &[Tensor],
    ) -> Result<Vec<Tensor>> {
        // Placeholder implementation
        Ok(inputs.to_vec())
    }

    /// Run custom framework inference (placeholder)
    fn run_custom_inference(
        &self,
        _session: &MLInferenceSession,
        inputs: &[Tensor],
    ) -> Result<Vec<Tensor>> {
        // Placeholder implementation
        Ok(inputs.to_vec())
    }

    /// Get inference metrics for a session
    pub fn get_metrics(&self, session_id: &str) -> Result<InferenceMetrics> {
        let sessions = self.active_sessions.read().map_err(|_| {
            Error::runtime("Failed to acquire read lock on active sessions".to_string())
        })?;

        let session = sessions
            .get(session_id)
            .ok_or_else(|| Error::runtime(format!("Session '{session_id}' not found")))?;

        let metrics = session
            .metrics
            .read()
            .map_err(|_| Error::runtime("Failed to acquire read lock on metrics".to_string()))?;

        Ok(metrics.clone())
    }

    /// Close inference session
    pub fn close_session(&self, session_id: &str) -> Result<()> {
        let mut sessions = self.active_sessions.write().map_err(|_| {
            Error::runtime("Failed to acquire write lock on active sessions".to_string())
        })?;

        sessions
            .remove(session_id)
            .ok_or_else(|| Error::runtime(format!("Session '{session_id}' not found")))?;

        Ok(())
    }

    /// List available frameworks
    pub fn list_frameworks(&self) -> Vec<(MLFramework, &FrameworkInfo)> {
        self.frameworks
            .iter()
            .map(|(&framework, info)| (framework, info))
            .collect()
    }

    /// Get framework capabilities
    pub fn get_framework_capabilities(
        &self,
        framework: MLFramework,
    ) -> Option<&FrameworkCapabilities> {
        self.frameworks
            .get(&framework)
            .map(|info| &info.capabilities)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_ml_framework_config_default() {
        let config = MLFrameworkConfig::default();
        assert_eq!(config.primary_framework, MLFramework::Candle);
        assert!(config.optimization.quantization_enabled);
        assert_eq!(config.performance_config.batch_size, 1);
    }

    #[test]
    fn test_ml_framework_manager_creation() {
        let config = MLFrameworkConfig::default();
        let manager = MLFrameworkManager::new(config).unwrap();

        let frameworks = manager.list_frameworks();
        assert!(!frameworks.is_empty());

        // Candle should always be available
        assert!(frameworks
            .iter()
            .any(|(framework, _)| *framework == MLFramework::Candle));
    }

    #[test]
    fn test_model_registration() {
        let config = MLFrameworkConfig::default();
        let manager = MLFrameworkManager::new(config).unwrap();

        let model_metadata = MLModelMetadata {
            name: "test-model".to_string(),
            version: "1.0.0".to_string(),
            framework: MLFramework::Candle,
            input_specs: vec![TensorSpec {
                name: "input".to_string(),
                shape: vec![1, -1, 80],
                data_type: TensorDataType::Float32,
                description: Some("Audio features".to_string()),
            }],
            output_specs: vec![TensorSpec {
                name: "output".to_string(),
                shape: vec![1, -1, 80],
                data_type: TensorDataType::Float32,
                description: Some("Converted features".to_string()),
            }],
            model_path: PathBuf::from("test_model.safetensors"),
            model_size_bytes: 1024 * 1024, // 1MB
            supported_sample_rates: vec![22050, 44100],
            capabilities: ModelCapabilities {
                realtime_capable: true,
                batch_capable: true,
                streaming_capable: true,
                gpu_accelerated: true,
                quantization_support: true,
                max_input_length: Some(1000),
            },
        };

        manager.register_model(model_metadata).unwrap();
    }

    #[test]
    fn test_quantization_precision() {
        let mut config = MLFrameworkConfig::default();
        config.optimization.quantization_precision = QuantizationPrecision::Float16;

        assert_eq!(
            config.optimization.quantization_precision,
            QuantizationPrecision::Float16
        );
    }

    #[test]
    fn test_device_preference() {
        let cpu_preference = DevicePreference::Cpu;
        let gpu_preference = DevicePreference::Gpu {
            device_index: Some(0),
            memory_limit_mb: Some(4096),
        };

        match cpu_preference {
            DevicePreference::Cpu => assert!(true),
            _ => panic!("Expected CPU preference"),
        }

        match gpu_preference {
            DevicePreference::Gpu {
                device_index: Some(0),
                memory_limit_mb: Some(4096),
            } => assert!(true),
            _ => panic!("Expected GPU preference with specific settings"),
        }
    }

    #[test]
    fn test_inference_metrics() {
        let mut metrics = InferenceMetrics::default();

        // Simulate some inference runs
        metrics.inference_count = 10;
        metrics.total_inference_time_ms = 1000;
        metrics.avg_inference_time_ms = 100.0;
        metrics.min_inference_time_ms = 50.0;
        metrics.max_inference_time_ms = 200.0;

        assert_eq!(metrics.inference_count, 10);
        assert_eq!(metrics.avg_inference_time_ms, 100.0);
    }
}

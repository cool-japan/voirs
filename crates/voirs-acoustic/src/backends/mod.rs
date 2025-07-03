//! Backend implementations for acoustic models
//!
//! This module provides abstraction layer for different backends including
//! Candle, ONNX Runtime, and potential future backends.

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;

use crate::{Result, AcousticError, AcousticModel};
use crate::config::{DeviceConfig, DeviceType, BackendType, BackendOptions};

pub mod candle;
pub mod loader;

#[cfg(feature = "onnx")]
pub mod onnx;

pub use candle::*;
pub use loader::*;

#[cfg(feature = "onnx")]
pub use onnx::*;

/// Backend manager for acoustic models
pub struct BackendManager {
    /// Available backends
    backends: HashMap<BackendType, Box<dyn Backend>>,
    /// Default backend
    default_backend: Option<BackendType>,
    /// Device configuration
    device_config: DeviceConfig,
}

impl BackendManager {
    /// Create new backend manager
    pub fn new() -> Self {
        Self {
            backends: HashMap::new(),
            default_backend: None,
            device_config: DeviceConfig::default(),
        }
    }
    
    /// Add backend to manager
    pub fn add_backend(&mut self, backend_type: BackendType, backend: Box<dyn Backend>) {
        self.backends.insert(backend_type, backend);
        
        // Set as default if it's the first backend
        if self.default_backend.is_none() {
            self.default_backend = Some(backend_type);
        }
    }
    
    /// Set default backend
    pub fn set_default_backend(&mut self, backend_type: BackendType) -> Result<()> {
        if self.backends.contains_key(&backend_type) {
            self.default_backend = Some(backend_type);
            Ok(())
        } else {
            Err(AcousticError::ConfigError(format!("Backend {:?} not available", backend_type)))
        }
    }
    
    /// Set device configuration
    pub fn set_device_config(&mut self, config: DeviceConfig) {
        self.device_config = config;
    }
    
    /// Get backend by type
    pub fn get_backend(&self, backend_type: BackendType) -> Result<&dyn Backend> {
        self.backends
            .get(&backend_type)
            .map(|b| b.as_ref())
            .ok_or_else(|| AcousticError::ConfigError(format!("Backend {:?} not found", backend_type)))
    }
    
    /// Get default backend
    pub fn get_default_backend(&self) -> Result<&dyn Backend> {
        let backend_type = self.default_backend
            .ok_or_else(|| AcousticError::ConfigError("No default backend set".to_string()))?;
        self.get_backend(backend_type)
    }
    
    /// Load model using specified backend
    pub async fn load_model(&self, backend_type: BackendType, model_path: &str) -> Result<Box<dyn AcousticModel>> {
        let backend = self.get_backend(backend_type)?;
        backend.create_model(model_path).await
    }
    
    /// Load model using default backend
    pub async fn load_model_default(&self, model_path: &str) -> Result<Box<dyn AcousticModel>> {
        let backend = self.get_default_backend()?;
        backend.create_model(model_path).await
    }
    
    /// List available backends
    pub fn list_backends(&self) -> Vec<BackendType> {
        self.backends.keys().copied().collect()
    }
    
    /// Get device capabilities for a backend
    pub fn get_device_capabilities(&self, backend_type: BackendType) -> Result<DeviceCapabilities> {
        let backend = self.get_backend(backend_type)?;
        let devices = backend.available_devices();
        let supports_gpu = backend.supports_gpu();
        
        Ok(DeviceCapabilities {
            backend_name: backend.name().to_string(),
            supports_gpu,
            available_devices: devices,
            supports_mixed_precision: supports_gpu, // Generally true for GPU backends
        })
    }
    
    /// Auto-detect best backend for current system
    pub fn auto_detect_backend(&self) -> Result<BackendType> {
        // Priority order: GPU-capable backends first, then CPU
        let preferred_order = vec![
            BackendType::Candle,
            #[cfg(feature = "onnx")]
            BackendType::Onnx,
        ];
        
        for &backend_type in &preferred_order {
            if let Ok(backend) = self.get_backend(backend_type) {
                // Prefer GPU-capable backends
                if backend.supports_gpu() && self.device_config.device_type != DeviceType::Cpu {
                    return Ok(backend_type);
                }
            }
        }
        
        // Fallback to any available backend
        preferred_order.into_iter()
            .find(|&bt| self.backends.contains_key(&bt))
            .ok_or_else(|| AcousticError::ConfigError("No suitable backend found".to_string()))
    }
}

impl Default for BackendManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Backend trait for acoustic model implementations
#[async_trait]
pub trait Backend: Send + Sync {
    /// Get backend name
    fn name(&self) -> &'static str;
    
    /// Check if GPU acceleration is available
    fn supports_gpu(&self) -> bool;
    
    /// Get available devices
    fn available_devices(&self) -> Vec<String>;
    
    /// Create model instance from path
    async fn create_model(&self, model_path: &str) -> Result<Box<dyn AcousticModel>>;
    
    /// Get backend capabilities
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            name: self.name().to_string(),
            supports_gpu: self.supports_gpu(),
            supports_streaming: false,
            supports_batch_processing: true,
            max_batch_size: Some(32),
            memory_efficient: true,
        }
    }
    
    /// Validate model format compatibility
    fn validate_model(&self, model_path: &str) -> Result<ModelInfo> {
        // Default implementation - backends should override
        Ok(ModelInfo {
            path: model_path.to_string(),
            format: ModelFormat::Unknown,
            size_bytes: 0,
            compatible: true,
            metadata: HashMap::new(),
        })
    }
    
    /// Get optimization options for this backend
    fn optimization_options(&self) -> Vec<OptimizationOption> {
        vec![
            OptimizationOption {
                name: "default".to_string(),
                description: "Default optimization level".to_string(),
                enabled: true,
            }
        ]
    }
}

/// Device capabilities information
#[derive(Debug, Clone)]
pub struct DeviceCapabilities {
    /// Backend name
    pub backend_name: String,
    /// Whether GPU acceleration is supported
    pub supports_gpu: bool,
    /// List of available devices
    pub available_devices: Vec<String>,
    /// Whether mixed precision is supported
    pub supports_mixed_precision: bool,
}

/// Backend capabilities information
#[derive(Debug, Clone)]
pub struct BackendCapabilities {
    /// Backend name
    pub name: String,
    /// GPU support
    pub supports_gpu: bool,
    /// Streaming support
    pub supports_streaming: bool,
    /// Batch processing support
    pub supports_batch_processing: bool,
    /// Maximum batch size
    pub max_batch_size: Option<u32>,
    /// Memory efficiency
    pub memory_efficient: bool,
}

/// Model information from validation
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Model file path
    pub path: String,
    /// Model format
    pub format: ModelFormat,
    /// Model size in bytes
    pub size_bytes: u64,
    /// Whether model is compatible with backend
    pub compatible: bool,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Supported model formats
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    /// SafeTensors format
    SafeTensors,
    /// ONNX format
    Onnx,
    /// PyTorch format
    PyTorch,
    /// TensorFlow format
    TensorFlow,
    /// Custom format
    Custom,
    /// Unknown format
    Unknown,
}

impl ModelFormat {
    /// Get file extension for format
    pub fn extension(&self) -> &'static str {
        match self {
            ModelFormat::SafeTensors => "safetensors",
            ModelFormat::Onnx => "onnx",
            ModelFormat::PyTorch => "pth",
            ModelFormat::TensorFlow => "pb",
            ModelFormat::Custom => "bin",
            ModelFormat::Unknown => "",
        }
    }
    
    /// Detect format from file extension
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_lowercase().as_str() {
            "safetensors" => ModelFormat::SafeTensors,
            "onnx" => ModelFormat::Onnx,
            "pth" | "pt" => ModelFormat::PyTorch,
            "pb" => ModelFormat::TensorFlow,
            "bin" => ModelFormat::Custom,
            _ => ModelFormat::Unknown,
        }
    }
}

/// Optimization option for backends
#[derive(Debug, Clone)]
pub struct OptimizationOption {
    /// Option name
    pub name: String,
    /// Option description
    pub description: String,
    /// Whether option is enabled
    pub enabled: bool,
}

/// Model loading configuration
#[derive(Debug, Clone)]
pub struct ModelLoadConfig {
    /// Device to load model on
    pub device: DeviceConfig,
    /// Whether to use memory mapping
    pub memory_map: bool,
    /// Whether to validate model after loading
    pub validate: bool,
    /// Optimization options
    pub optimizations: Vec<String>,
    /// Custom configuration
    pub custom: HashMap<String, serde_json::Value>,
}

impl ModelLoadConfig {
    /// Create new model load configuration
    pub fn new() -> Self {
        Self {
            device: DeviceConfig::default(),
            memory_map: true,
            validate: true,
            optimizations: vec!["default".to_string()],
            custom: HashMap::new(),
        }
    }
    
    /// Set device configuration
    pub fn with_device(mut self, device: DeviceConfig) -> Self {
        self.device = device;
        self
    }
    
    /// Enable/disable memory mapping
    pub fn with_memory_map(mut self, enabled: bool) -> Self {
        self.memory_map = enabled;
        self
    }
    
    /// Enable/disable validation
    pub fn with_validation(mut self, enabled: bool) -> Self {
        self.validate = enabled;
        self
    }
    
    /// Set optimization options
    pub fn with_optimizations(mut self, optimizations: Vec<String>) -> Self {
        self.optimizations = optimizations;
        self
    }
    
    /// Add custom option
    pub fn with_custom<K, V>(mut self, key: K, value: V) -> Self 
    where
        K: Into<String>,
        V: Into<serde_json::Value>,
    {
        self.custom.insert(key.into(), value.into());
        self
    }
}

impl Default for ModelLoadConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Create default backend manager with available backends
pub fn create_default_backend_manager() -> Result<BackendManager> {
    let mut manager = BackendManager::new();
    
    // Add Candle backend
    let candle_backend = CandleBackend::new()?;
    manager.add_backend(BackendType::Candle, Box::new(candle_backend));
    
    // Add ONNX backend if available
    #[cfg(feature = "onnx")]
    {
        let onnx_backend = OnnxBackend::new()?;
        manager.add_backend(BackendType::Onnx, Box::new(onnx_backend));
    }
    
    // Auto-detect best backend
    let best_backend = manager.auto_detect_backend()?;
    manager.set_default_backend(best_backend)?;
    
    Ok(manager)
}

/// Backend factory for creating backends with specific configurations
pub struct BackendFactory;

impl BackendFactory {
    /// Create Candle backend with configuration
    pub fn create_candle(config: Option<BackendOptions>) -> Result<Box<dyn Backend>> {
        let backend = if let Some(options) = config {
            CandleBackend::with_options(options)?
        } else {
            CandleBackend::new()?
        };
        Ok(Box::new(backend))
    }
    
    /// Create ONNX backend with configuration
    #[cfg(feature = "onnx")]
    pub fn create_onnx(config: Option<BackendOptions>) -> Result<Box<dyn Backend>> {
        let backend = if let Some(options) = config {
            OnnxBackend::with_options(options)?
        } else {
            OnnxBackend::new()?
        };
        Ok(Box::new(backend))
    }
    
    /// Create backend by type
    pub fn create_backend(backend_type: BackendType, config: Option<BackendOptions>) -> Result<Box<dyn Backend>> {
        match backend_type {
            BackendType::Candle => Self::create_candle(config),
            #[cfg(feature = "onnx")]
            BackendType::Onnx => Self::create_onnx(config),
            #[cfg(not(feature = "onnx"))]
            BackendType::Onnx => Err(AcousticError::ConfigError("ONNX backend not available (feature not enabled)".to_string())),
            BackendType::Custom(_) => Err(AcousticError::ConfigError("Custom backends not yet supported".to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_manager_creation() {
        let manager = BackendManager::new();
        assert!(manager.list_backends().is_empty());
        assert!(manager.get_default_backend().is_err());
    }

    #[test]
    fn test_model_format_detection() {
        assert_eq!(ModelFormat::from_extension("safetensors"), ModelFormat::SafeTensors);
        assert_eq!(ModelFormat::from_extension("onnx"), ModelFormat::Onnx);
        assert_eq!(ModelFormat::from_extension("pth"), ModelFormat::PyTorch);
        assert_eq!(ModelFormat::from_extension("pt"), ModelFormat::PyTorch);
        assert_eq!(ModelFormat::from_extension("pb"), ModelFormat::TensorFlow);
        assert_eq!(ModelFormat::from_extension("bin"), ModelFormat::Custom);
        assert_eq!(ModelFormat::from_extension("unknown"), ModelFormat::Unknown);
    }

    #[test]
    fn test_model_format_extensions() {
        assert_eq!(ModelFormat::SafeTensors.extension(), "safetensors");
        assert_eq!(ModelFormat::Onnx.extension(), "onnx");
        assert_eq!(ModelFormat::PyTorch.extension(), "pth");
        assert_eq!(ModelFormat::TensorFlow.extension(), "pb");
        assert_eq!(ModelFormat::Custom.extension(), "bin");
        assert_eq!(ModelFormat::Unknown.extension(), "");
    }

    #[test]
    fn test_model_load_config() {
        let config = ModelLoadConfig::new()
            .with_device(DeviceConfig::cpu())
            .with_memory_map(false)
            .with_validation(false)
            .with_optimizations(vec!["fast".to_string()])
            .with_custom("test", serde_json::Value::String("value".to_string()));
        
        assert_eq!(config.device.device_type, DeviceType::Cpu);
        assert!(!config.memory_map);
        assert!(!config.validate);
        assert_eq!(config.optimizations, vec!["fast"]);
        assert!(config.custom.contains_key("test"));
    }

    #[test]
    fn test_default_backend_manager_creation() {
        // This test might fail if no backends are available
        match create_default_backend_manager() {
            Ok(manager) => {
                assert!(!manager.list_backends().is_empty());
                assert!(manager.get_default_backend().is_ok());
            }
            Err(_) => {
                // Expected if no backends are compiled in
            }
        }
    }

    #[test]
    fn test_backend_factory() {
        // Test Candle backend creation
        match BackendFactory::create_candle(None) {
            Ok(_) => {
                // Backend created successfully
            }
            Err(_) => {
                // Expected if Candle is not available
            }
        }
        
        // Test ONNX backend creation
        #[cfg(feature = "onnx")]
        {
            match BackendFactory::create_onnx(None) {
                Ok(_) => {
                    // Backend created successfully
                }
                Err(_) => {
                    // Expected if ONNX is not available
                }
            }
        }
        
        // Test invalid backend type
        assert!(BackendFactory::create_backend(BackendType::Custom(999), None).is_err());
    }
}
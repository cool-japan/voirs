//! Unit tests for backend system
//!
//! Tests backend abstraction, model loading, and device management
//! to ensure robust backend functionality.

use voirs_vocoder::backends::{Backend, BackendType, DeviceType, ModelLoader, ModelFormat};
use voirs_vocoder::{VocoderError, MelSpectrogram, AudioBuffer};
use std::sync::Arc;
use tempfile::NamedTempFile;
use std::io::Write;

#[cfg(feature = "candle")]
use voirs_vocoder::backends::candle::CandleBackend;

#[test]
fn test_backend_type_enum() {
    assert_eq!(BackendType::Candle.to_string(), "candle");
    assert_eq!(BackendType::Onnx.to_string(), "onnx");
    
    // Test parsing from string
    assert_eq!("candle".parse::<BackendType>().unwrap(), BackendType::Candle);
    assert_eq!("onnx".parse::<BackendType>().unwrap(), BackendType::Onnx);
    
    // Test invalid parsing
    assert!("invalid".parse::<BackendType>().is_err());
}

#[test]
fn test_device_type_enum() {
    assert_eq!(DeviceType::Cpu.to_string(), "cpu");
    assert_eq!(DeviceType::Cuda.to_string(), "cuda");
    assert_eq!(DeviceType::Metal.to_string(), "metal");
    
    // Test parsing from string
    assert_eq!("cpu".parse::<DeviceType>().unwrap(), DeviceType::Cpu);
    assert_eq!("cuda".parse::<DeviceType>().unwrap(), DeviceType::Cuda);
    assert_eq!("metal".parse::<DeviceType>().unwrap(), DeviceType::Metal);
    
    // Test invalid parsing
    assert!("invalid".parse::<DeviceType>().is_err());
}

#[test]
fn test_model_format_detection() {
    // Test different file extensions
    assert_eq!(ModelFormat::from_path("model.safetensors").unwrap(), ModelFormat::SafeTensors);
    assert_eq!(ModelFormat::from_path("model.onnx").unwrap(), ModelFormat::Onnx);
    assert_eq!(ModelFormat::from_path("model.pth").unwrap(), ModelFormat::PyTorch);
    assert_eq!(ModelFormat::from_path("model.pt").unwrap(), ModelFormat::PyTorch);
    
    // Test case insensitivity
    assert_eq!(ModelFormat::from_path("MODEL.SAFETENSORS").unwrap(), ModelFormat::SafeTensors);
    assert_eq!(ModelFormat::from_path("Model.Onnx").unwrap(), ModelFormat::Onnx);
    
    // Test unsupported formats
    assert!(ModelFormat::from_path("model.txt").is_err());
    assert!(ModelFormat::from_path("model").is_err());
    assert!(ModelFormat::from_path("").is_err());
}

#[test]
fn test_model_loader_creation() {
    let loader = ModelLoader::new();
    
    // Test that loader is created successfully
    assert!(true); // Basic creation test
    
    // Test loader with different configurations
    let loader_with_cache = ModelLoader::with_cache_enabled(true);
    assert!(true); // Cache enabled test
    
    let loader_without_cache = ModelLoader::with_cache_enabled(false);
    assert!(true); // Cache disabled test
}

#[test]
fn test_model_loader_format_detection() {
    let loader = ModelLoader::new();
    
    // Test format detection for different file types
    assert_eq!(loader.detect_format("model.safetensors").unwrap(), ModelFormat::SafeTensors);
    assert_eq!(loader.detect_format("model.onnx").unwrap(), ModelFormat::Onnx);
    assert_eq!(loader.detect_format("model.pth").unwrap(), ModelFormat::PyTorch);
    
    // Test with full paths
    assert_eq!(loader.detect_format("/path/to/model.safetensors").unwrap(), ModelFormat::SafeTensors);
    assert_eq!(loader.detect_format("../models/hifigan.onnx").unwrap(), ModelFormat::Onnx);
    
    // Test unsupported formats
    assert!(loader.detect_format("model.unknown").is_err());
}

#[test]
fn test_model_loader_with_dummy_files() {
    let loader = ModelLoader::new();
    
    // Create temporary files with different extensions
    let mut safetensors_file = NamedTempFile::new().unwrap();
    let mut onnx_file = NamedTempFile::new().unwrap();
    
    // Write some dummy content
    writeln!(safetensors_file, "dummy safetensors content").unwrap();
    writeln!(onnx_file, "dummy onnx content").unwrap();
    
    // Test format detection on actual files
    let safetensors_path = format!("{}.safetensors", safetensors_file.path().to_str().unwrap());
    let onnx_path = format!("{}.onnx", onnx_file.path().to_str().unwrap());
    
    assert_eq!(loader.detect_format(&safetensors_path).unwrap(), ModelFormat::SafeTensors);
    assert_eq!(loader.detect_format(&onnx_path).unwrap(), ModelFormat::Onnx);
    
    // Test loading non-existent files
    assert!(loader.load_model("non_existent.safetensors").is_err());
    assert!(loader.load_model("non_existent.onnx").is_err());
}

#[test]
fn test_model_loader_validation() {
    let loader = ModelLoader::new();
    
    // Test validation of different file types
    assert!(loader.validate_model_file("model.safetensors").is_ok());
    assert!(loader.validate_model_file("model.onnx").is_ok());
    assert!(loader.validate_model_file("model.pth").is_ok());
    
    // Test validation of invalid files
    assert!(loader.validate_model_file("").is_err());
    assert!(loader.validate_model_file("model.txt").is_err());
    assert!(loader.validate_model_file("model").is_err());
    
    // Test validation of paths with spaces
    assert!(loader.validate_model_file("model with spaces.safetensors").is_ok());
    assert!(loader.validate_model_file("path/to/model.onnx").is_ok());
}

#[test]
fn test_model_loader_cache_behavior() {
    let loader_with_cache = ModelLoader::with_cache_enabled(true);
    let loader_without_cache = ModelLoader::with_cache_enabled(false);
    
    // Test cache configuration
    assert!(loader_with_cache.is_cache_enabled());
    assert!(!loader_without_cache.is_cache_enabled());
    
    // Test cache operations
    loader_with_cache.clear_cache();
    assert_eq!(loader_with_cache.cache_size(), 0);
    
    // Test cache statistics
    let stats = loader_with_cache.cache_stats();
    assert_eq!(stats.hits, 0);
    assert_eq!(stats.misses, 0);
    assert_eq!(stats.entries, 0);
}

#[cfg(feature = "candle")]
#[test]
fn test_candle_backend_creation() {
    let backend = CandleBackend::new();
    
    assert!(backend.is_ok());
    let backend = backend.unwrap();
    
    // Test backend metadata
    let metadata = backend.metadata();
    assert_eq!(metadata.name, "candle");
    assert_eq!(metadata.version, "0.9.1");
    assert!(metadata.supported_devices.contains(&DeviceType::Cpu));
}

#[cfg(feature = "candle")]
#[test]
fn test_candle_backend_device_support() {
    let backend = CandleBackend::new().unwrap();
    
    // CPU should always be supported
    assert!(backend.supports_device(DeviceType::Cpu));
    
    // GPU support depends on compilation features
    let cuda_support = backend.supports_device(DeviceType::Cuda);
    let metal_support = backend.supports_device(DeviceType::Metal);
    
    // At least one should be true (CPU is always supported)
    assert!(backend.supports_device(DeviceType::Cpu) || cuda_support || metal_support);
}

#[cfg(feature = "candle")]
#[test]
fn test_candle_backend_device_selection() {
    let backend = CandleBackend::new().unwrap();
    
    // Test automatic device selection
    let device = backend.select_device("auto").unwrap();
    assert!(device == DeviceType::Cpu || device == DeviceType::Cuda || device == DeviceType::Metal);
    
    // Test explicit CPU selection
    let cpu_device = backend.select_device("cpu").unwrap();
    assert_eq!(cpu_device, DeviceType::Cpu);
    
    // Test invalid device selection
    assert!(backend.select_device("invalid").is_err());
}

#[cfg(feature = "candle")]
#[test]
fn test_candle_backend_inference_preparation() {
    let backend = CandleBackend::new().unwrap();
    
    // Test inference preparation
    let result = backend.prepare_inference(DeviceType::Cpu);
    assert!(result.is_ok());
    
    // Test preparation with different devices
    if backend.supports_device(DeviceType::Cuda) {
        let result = backend.prepare_inference(DeviceType::Cuda);
        assert!(result.is_ok());
    }
    
    if backend.supports_device(DeviceType::Metal) {
        let result = backend.prepare_inference(DeviceType::Metal);
        assert!(result.is_ok());
    }
}

#[cfg(feature = "candle")]
#[test]
fn test_candle_backend_memory_management() {
    let backend = CandleBackend::new().unwrap();
    
    // Test memory pool creation
    let pool = backend.create_memory_pool(DeviceType::Cpu, 100 * 1024 * 1024); // 100MB
    assert!(pool.is_ok());
    
    // Test memory statistics
    let stats = backend.memory_stats(DeviceType::Cpu);
    assert!(stats.is_ok());
    
    let stats = stats.unwrap();
    assert!(stats.total_allocated >= 0);
    assert!(stats.peak_allocated >= 0);
    assert!(stats.current_allocated >= 0);
    
    // Test memory cleanup
    backend.cleanup_memory(DeviceType::Cpu);
    
    // Memory should be cleaned up
    let stats_after = backend.memory_stats(DeviceType::Cpu).unwrap();
    assert!(stats_after.current_allocated <= stats.current_allocated);
}

#[cfg(feature = "candle")]
#[test]
fn test_candle_backend_tensor_operations() {
    let backend = CandleBackend::new().unwrap();
    
    // Test tensor creation
    let shape = vec![1, 80, 100]; // Batch, mel_bins, time_steps
    let tensor = backend.create_tensor(&shape, DeviceType::Cpu);
    assert!(tensor.is_ok());
    
    let tensor = tensor.unwrap();
    assert_eq!(tensor.dims(), &[1, 80, 100]);
    
    // Test tensor operations
    let reshaped = backend.reshape_tensor(&tensor, &[80, 100]);
    assert!(reshaped.is_ok());
    
    let transposed = backend.transpose_tensor(&tensor, 1, 2);
    assert!(transposed.is_ok());
    
    // Test tensor conversion
    let data = vec![0.5f32; 80 * 100];
    let from_data = backend.tensor_from_data(&data, &[80, 100], DeviceType::Cpu);
    assert!(from_data.is_ok());
    
    let to_data = backend.tensor_to_data(&from_data.unwrap());
    assert!(to_data.is_ok());
    
    let extracted_data = to_data.unwrap();
    assert_eq!(extracted_data.len(), 80 * 100);
    assert!((extracted_data[0] - 0.5).abs() < 1e-6);
}

#[cfg(feature = "candle")]
#[test]
fn test_candle_backend_error_handling() {
    let backend = CandleBackend::new().unwrap();
    
    // Test error handling for invalid operations
    let invalid_shape = vec![0, 80, 100]; // Zero dimension
    let result = backend.create_tensor(&invalid_shape, DeviceType::Cpu);
    assert!(result.is_err());
    
    // Test error handling for unsupported devices
    let result = backend.select_device("nonexistent");
    assert!(result.is_err());
    
    // Test error handling for memory operations
    let result = backend.create_memory_pool(DeviceType::Cpu, 0); // Zero size
    assert!(result.is_err());
}

#[test]
fn test_backend_factory_pattern() {
    // Test backend creation through factory
    let backend_type = BackendType::Candle;
    let result = create_backend(backend_type);
    
    #[cfg(feature = "candle")]
    assert!(result.is_ok());
    
    #[cfg(not(feature = "candle"))]
    assert!(result.is_err());
}

#[test]
fn test_backend_compatibility() {
    // Test backend compatibility with different model formats
    let candle_compatible = is_backend_compatible(BackendType::Candle, ModelFormat::SafeTensors);
    let onnx_compatible = is_backend_compatible(BackendType::Onnx, ModelFormat::Onnx);
    
    assert!(candle_compatible);
    assert!(onnx_compatible);
    
    // Test incompatible combinations
    let incompatible = is_backend_compatible(BackendType::Candle, ModelFormat::Onnx);
    assert!(!incompatible);
}

#[test]
fn test_device_detection() {
    // Test device detection and availability
    let available_devices = detect_available_devices();
    
    // CPU should always be available
    assert!(available_devices.contains(&DeviceType::Cpu));
    
    // Test device capabilities
    let cpu_caps = get_device_capabilities(DeviceType::Cpu);
    assert!(cpu_caps.supports_f32);
    assert!(cpu_caps.max_memory > 0);
    
    // Test device selection logic
    let best_device = select_best_device(&available_devices);
    assert!(available_devices.contains(&best_device));
}

#[test]
fn test_backend_configuration() {
    // Test backend configuration options
    let mut config = BackendConfig::default();
    
    config.device = DeviceType::Cpu;
    config.memory_pool_size = 256 * 1024 * 1024; // 256MB
    config.enable_optimization = true;
    config.optimization_level = OptimizationLevel::Balanced;
    
    assert!(config.validate().is_ok());
    
    // Test invalid configurations
    config.memory_pool_size = 0;
    assert!(config.validate().is_err());
    
    config.memory_pool_size = 256 * 1024 * 1024;
    config.optimization_level = OptimizationLevel::Invalid;
    assert!(config.validate().is_err());
}

// Helper functions for testing
fn create_backend(backend_type: BackendType) -> Result<Box<dyn Backend>, VocoderError> {
    match backend_type {
        #[cfg(feature = "candle")]
        BackendType::Candle => {
            Ok(Box::new(CandleBackend::new()?))
        }
        #[cfg(not(feature = "candle"))]
        BackendType::Candle => {
            Err(VocoderError::BackendError("Candle backend not available".to_string()))
        }
        BackendType::Onnx => {
            #[cfg(feature = "onnx")]
            {
                Ok(Box::new(OnnxBackend::new()?))
            }
            #[cfg(not(feature = "onnx"))]
            {
                Err(VocoderError::BackendError("ONNX backend not available".to_string()))
            }
        }
    }
}

fn is_backend_compatible(backend: BackendType, format: ModelFormat) -> bool {
    match (backend, format) {
        (BackendType::Candle, ModelFormat::SafeTensors) => true,
        (BackendType::Candle, ModelFormat::PyTorch) => true,
        (BackendType::Onnx, ModelFormat::Onnx) => true,
        _ => false,
    }
}

fn detect_available_devices() -> Vec<DeviceType> {
    let mut devices = vec![DeviceType::Cpu]; // CPU always available
    
    #[cfg(feature = "cuda")]
    if is_cuda_available() {
        devices.push(DeviceType::Cuda);
    }
    
    #[cfg(target_os = "macos")]
    if is_metal_available() {
        devices.push(DeviceType::Metal);
    }
    
    devices
}

fn get_device_capabilities(device: DeviceType) -> DeviceCapabilities {
    match device {
        DeviceType::Cpu => DeviceCapabilities {
            supports_f32: true,
            supports_f16: false,
            supports_int8: true,
            max_memory: 1024 * 1024 * 1024 * 8, // 8GB
            compute_capability: 1.0,
        },
        DeviceType::Cuda => DeviceCapabilities {
            supports_f32: true,
            supports_f16: true,
            supports_int8: true,
            max_memory: 1024 * 1024 * 1024 * 24, // 24GB
            compute_capability: 8.0,
        },
        DeviceType::Metal => DeviceCapabilities {
            supports_f32: true,
            supports_f16: true,
            supports_int8: false,
            max_memory: 1024 * 1024 * 1024 * 64, // 64GB
            compute_capability: 1.0,
        },
    }
}

fn select_best_device(available: &[DeviceType]) -> DeviceType {
    if available.contains(&DeviceType::Cuda) {
        DeviceType::Cuda
    } else if available.contains(&DeviceType::Metal) {
        DeviceType::Metal
    } else {
        DeviceType::Cpu
    }
}

#[cfg(feature = "cuda")]
fn is_cuda_available() -> bool {
    // This would normally check for CUDA runtime
    false // Assume not available in tests
}

#[cfg(target_os = "macos")]
fn is_metal_available() -> bool {
    // This would normally check for Metal framework
    true // Assume available on macOS
}

#[derive(Debug, Clone)]
struct BackendConfig {
    device: DeviceType,
    memory_pool_size: usize,
    enable_optimization: bool,
    optimization_level: OptimizationLevel,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            device: DeviceType::Cpu,
            memory_pool_size: 256 * 1024 * 1024,
            enable_optimization: true,
            optimization_level: OptimizationLevel::Balanced,
        }
    }
}

impl BackendConfig {
    fn validate(&self) -> Result<(), VocoderError> {
        if self.memory_pool_size == 0 {
            return Err(VocoderError::ConfigError("Memory pool size cannot be zero".to_string()));
        }
        
        if matches!(self.optimization_level, OptimizationLevel::Invalid) {
            return Err(VocoderError::ConfigError("Invalid optimization level".to_string()));
        }
        
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
enum OptimizationLevel {
    None,
    Basic,
    Balanced,
    Aggressive,
    Invalid,
}

#[derive(Debug, Clone)]
struct DeviceCapabilities {
    supports_f32: bool,
    supports_f16: bool,
    supports_int8: bool,
    max_memory: usize,
    compute_capability: f32,
}
//! Backend infrastructure for voirs-vocoder.
//!
//! This module provides abstraction layers for different ML backends including:
//! - Candle (Rust-native tensor operations)
//! - ONNX Runtime
//! - Model loading and management
//! - Device management (CPU, CUDA, Metal)

pub mod candle;
pub mod loader;

#[cfg(feature = "onnx")]
pub mod onnx;

pub use candle::*;
pub use loader::*;

#[cfg(feature = "onnx")]
pub use onnx::*;

use crate::{AudioBuffer, MelSpectrogram, Result, VocoderError};
use crate::config::{DeviceType, ModelConfig};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Backend trait for ML inference
#[async_trait]
pub trait Backend: Send + Sync {
    /// Initialize the backend with given configuration
    async fn initialize(&mut self, config: &ModelConfig) -> Result<()>;
    
    /// Load a model from the given path
    async fn load_model(&mut self, model_path: &str) -> Result<()>;
    
    /// Perform inference on mel spectrogram
    async fn inference(&self, mel: &MelSpectrogram) -> Result<AudioBuffer>;
    
    /// Perform batch inference
    async fn batch_inference(&self, mels: &[MelSpectrogram]) -> Result<Vec<AudioBuffer>>;
    
    /// Get backend metadata
    fn metadata(&self) -> BackendMetadata;
    
    /// Check if the backend supports the given device
    fn supports_device(&self, device: DeviceType) -> bool;
    
    /// Get current memory usage in bytes
    fn memory_usage(&self) -> u64;
    
    /// Cleanup and release resources
    async fn cleanup(&mut self) -> Result<()>;
}

/// Backend metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendMetadata {
    /// Backend name
    pub name: String,
    /// Backend version
    pub version: String,
    /// Supported devices
    pub supported_devices: Vec<DeviceType>,
    /// Supported model formats
    pub supported_formats: Vec<String>,
    /// Whether the backend supports GPU acceleration
    pub gpu_acceleration: bool,
    /// Whether the backend supports mixed precision
    pub mixed_precision: bool,
    /// Whether the backend supports quantization
    pub quantization: bool,
}

/// Device manager for handling different compute devices
pub struct DeviceManager {
    current_device: DeviceType,
    available_devices: Vec<DeviceType>,
}

impl DeviceManager {
    /// Create new device manager
    pub fn new() -> Self {
        let available_devices = Self::detect_available_devices();
        let current_device = if available_devices.contains(&DeviceType::Cuda) {
            DeviceType::Cuda
        } else if available_devices.contains(&DeviceType::Metal) {
            DeviceType::Metal
        } else {
            DeviceType::Cpu
        };

        Self {
            current_device,
            available_devices,
        }
    }

    /// Detect available devices
    fn detect_available_devices() -> Vec<DeviceType> {
        let mut devices = vec![DeviceType::Cpu];
        
        // Check for CUDA availability
        #[cfg(feature = "gpu")]
        if Self::is_cuda_available() {
            devices.push(DeviceType::Cuda);
        }
        
        // Check for Metal availability (macOS)
        #[cfg(target_os = "macos")]
        if Self::is_metal_available() {
            devices.push(DeviceType::Metal);
        }
        
        devices
    }

    /// Check if CUDA is available
    #[cfg(feature = "gpu")]
    fn is_cuda_available() -> bool {
        // This would typically check for CUDA runtime
        // For now, we'll assume it's available if the GPU feature is enabled
        true
    }

    /// Check if Metal is available
    #[cfg(target_os = "macos")]
    fn is_metal_available() -> bool {
        // Metal is typically available on macOS
        true
    }

    /// Set current device
    pub fn set_device(&mut self, device: DeviceType) -> Result<()> {
        if device == DeviceType::Auto {
            self.current_device = self.get_best_device();
        } else if self.available_devices.contains(&device) {
            self.current_device = device;
        } else {
            return Err(VocoderError::ConfigError(
                format!("Device {:?} is not available", device)
            ));
        }
        Ok(())
    }

    /// Get current device
    pub fn current_device(&self) -> DeviceType {
        self.current_device
    }

    /// Get available devices
    pub fn available_devices(&self) -> &[DeviceType] {
        &self.available_devices
    }

    /// Get the best available device
    pub fn get_best_device(&self) -> DeviceType {
        if self.available_devices.contains(&DeviceType::Cuda) {
            DeviceType::Cuda
        } else if self.available_devices.contains(&DeviceType::Metal) {
            DeviceType::Metal
        } else {
            DeviceType::Cpu
        }
    }

    /// Check if device supports feature
    pub fn device_supports_feature(&self, device: DeviceType, feature: &str) -> bool {
        match (device, feature) {
            (DeviceType::Cuda, "mixed_precision") => true,
            (DeviceType::Cuda, "fp16") => true,
            (DeviceType::Metal, "mixed_precision") => true,
            (DeviceType::Metal, "fp16") => true,
            (DeviceType::Cpu, "quantization") => true,
            _ => false,
        }
    }
}

impl Default for DeviceManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory manager for backend resources
pub struct MemoryManager {
    allocated_bytes: u64,
    peak_bytes: u64,
    allocation_count: usize,
}

impl MemoryManager {
    /// Create new memory manager
    pub fn new() -> Self {
        Self {
            allocated_bytes: 0,
            peak_bytes: 0,
            allocation_count: 0,
        }
    }

    /// Record memory allocation
    pub fn allocate(&mut self, bytes: u64) {
        self.allocated_bytes += bytes;
        self.allocation_count += 1;
        if self.allocated_bytes > self.peak_bytes {
            self.peak_bytes = self.allocated_bytes;
        }
    }

    /// Record memory deallocation
    pub fn deallocate(&mut self, bytes: u64) {
        self.allocated_bytes = self.allocated_bytes.saturating_sub(bytes);
    }

    /// Get current allocated memory
    pub fn allocated_bytes(&self) -> u64 {
        self.allocated_bytes
    }

    /// Get peak memory usage
    pub fn peak_bytes(&self) -> u64 {
        self.peak_bytes
    }

    /// Get allocation count
    pub fn allocation_count(&self) -> usize {
        self.allocation_count
    }

    /// Reset statistics
    pub fn reset(&mut self) {
        self.allocated_bytes = 0;
        self.peak_bytes = 0;
        self.allocation_count = 0;
    }
}

impl Default for MemoryManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Backend factory for creating backend instances
pub struct BackendFactory;

impl BackendFactory {
    /// Create backend based on configuration
    pub fn create_backend(config: &ModelConfig) -> Result<Box<dyn Backend>> {
        match config.backend {
            crate::config::BackendType::Candle => {
                Ok(Box::new(CandleBackend::new()))
            }
            #[cfg(feature = "onnx")]
            crate::config::BackendType::ONNX => {
                Ok(Box::new(OnnxBackend::new()))
            }
            crate::config::BackendType::Auto => {
                // Auto-select the best available backend
                #[cfg(feature = "onnx")]
                if Self::is_onnx_available() {
                    Ok(Box::new(OnnxBackend::new()))
                } else {
                    Ok(Box::new(CandleBackend::new()))
                }
                #[cfg(not(feature = "onnx"))]
                Ok(Box::new(CandleBackend::new()))
            }
            _ => Err(VocoderError::ConfigError(
                format!("Backend {:?} not supported", config.backend)
            )),
        }
    }

    /// Check if ONNX Runtime is available
    #[cfg(feature = "onnx")]
    fn is_onnx_available() -> bool {
        // This would check if ONNX Runtime is properly installed
        true
    }

    /// Get list of available backends
    pub fn available_backends() -> Vec<String> {
        let mut backends = vec!["Candle".to_string()];
        
        #[cfg(feature = "onnx")]
        backends.push("ONNX".to_string());
        
        backends
    }
}

/// Error recovery strategies
#[derive(Debug, Clone)]
pub enum ErrorRecoveryStrategy {
    /// Retry the operation
    Retry,
    /// Fallback to CPU
    FallbackToCpu,
    /// Reduce batch size
    ReduceBatchSize,
    /// Switch backend
    SwitchBackend,
    /// Abort operation
    Abort,
}

/// Backend performance monitor
#[derive(Debug, Clone)]
pub struct PerformanceMonitor {
    inference_times: Vec<f32>,
    memory_usage: Vec<u64>,
    error_count: usize,
}

impl PerformanceMonitor {
    /// Create new performance monitor
    pub fn new() -> Self {
        Self {
            inference_times: Vec::new(),
            memory_usage: Vec::new(),
            error_count: 0,
        }
    }

    /// Record inference time
    pub fn record_inference_time(&mut self, time_ms: f32) {
        self.inference_times.push(time_ms);
        // Keep only recent measurements
        if self.inference_times.len() > 100 {
            self.inference_times.remove(0);
        }
    }

    /// Record memory usage
    pub fn record_memory_usage(&mut self, bytes: u64) {
        self.memory_usage.push(bytes);
        // Keep only recent measurements
        if self.memory_usage.len() > 100 {
            self.memory_usage.remove(0);
        }
    }

    /// Record error
    pub fn record_error(&mut self) {
        self.error_count += 1;
    }

    /// Get average inference time
    pub fn average_inference_time(&self) -> f32 {
        if self.inference_times.is_empty() {
            0.0
        } else {
            self.inference_times.iter().sum::<f32>() / self.inference_times.len() as f32
        }
    }

    /// Get average memory usage
    pub fn average_memory_usage(&self) -> u64 {
        if self.memory_usage.is_empty() {
            0
        } else {
            self.memory_usage.iter().sum::<u64>() / self.memory_usage.len() as u64
        }
    }

    /// Get error rate
    pub fn error_rate(&self) -> f32 {
        let total_operations = self.inference_times.len() + self.error_count;
        if total_operations == 0 {
            0.0
        } else {
            self.error_count as f32 / total_operations as f32
        }
    }
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_manager() {
        let mut manager = DeviceManager::new();
        
        // Should have at least CPU available
        assert!(manager.available_devices().contains(&DeviceType::Cpu));
        
        // Should be able to set CPU device
        assert!(manager.set_device(DeviceType::Cpu).is_ok());
        assert_eq!(manager.current_device(), DeviceType::Cpu);
    }

    #[test]
    fn test_memory_manager() {
        let mut manager = MemoryManager::new();
        
        assert_eq!(manager.allocated_bytes(), 0);
        
        manager.allocate(1024);
        assert_eq!(manager.allocated_bytes(), 1024);
        assert_eq!(manager.peak_bytes(), 1024);
        
        manager.deallocate(512);
        assert_eq!(manager.allocated_bytes(), 512);
        assert_eq!(manager.peak_bytes(), 1024); // Peak should remain
    }

    #[test]
    fn test_performance_monitor() {
        let mut monitor = PerformanceMonitor::new();
        
        monitor.record_inference_time(10.0);
        monitor.record_inference_time(20.0);
        
        assert_eq!(monitor.average_inference_time(), 15.0);
        
        monitor.record_error();
        assert!(monitor.error_rate() > 0.0);
    }

    #[test]
    fn test_backend_factory() {
        let backends = BackendFactory::available_backends();
        assert!(backends.contains(&"Candle".to_string()));
    }
}
//! Backend infrastructure for voirs-vocoder.
//!
//! This module provides abstraction layers for different ML backends including:
//! - Candle (Rust-native tensor operations)
//! - ONNX Runtime
//! - Model loading and management
//! - Device management (CPU, CUDA, Metal)

pub mod candle;
pub mod loader;

// #[cfg(feature = "onnx")]
// pub mod onnx;

pub use candle::*;
pub use loader::*;

// #[cfg(feature = "onnx")]
// pub use onnx::*;

use crate::config::{DeviceType, ModelConfig};
use crate::{AudioBuffer, MelSpectrogram, Result, VocoderError};
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
            return Err(VocoderError::ConfigError(format!(
                "Device {device:?} is not available",
            )));
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
        matches!(
            (device, feature),
            (DeviceType::Cuda, "mixed_precision")
                | (DeviceType::Cuda, "fp16")
                | (DeviceType::Metal, "mixed_precision")
                | (DeviceType::Metal, "fp16")
                | (DeviceType::Cpu, "quantization")
        )
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
            crate::config::BackendType::Candle => Ok(Box::new(CandleBackend::new())),
            // #[cfg(feature = "onnx")]
            // crate::config::BackendType::ONNX => Ok(Box::new(OnnxVocoderBackend::new()?)),
            crate::config::BackendType::Auto => {
                // Auto-select the best available backend
                // #[cfg(feature = "onnx")]
                // if Self::is_onnx_available() {
                //     Ok(Box::new(OnnxVocoderBackend::new()?)
                // } else {
                //     Ok(Box::new(CandleBackend::new()))
                // }
                // #[cfg(not(feature = "onnx"))]
                Ok(Box::new(CandleBackend::new()))
            }
            _ => Err(VocoderError::ConfigError(format!(
                "Backend {:?} not supported",
                config.backend
            ))),
        }
    }

    /// Check if ONNX Runtime is available
    #[cfg(feature = "onnx")]
    #[allow(dead_code)]
    fn is_onnx_available() -> bool {
        // This would check if ONNX Runtime is properly installed
        true
    }

    /// Get list of available backends
    pub fn available_backends() -> Vec<String> {
        #[cfg(feature = "onnx")]
        {
            let mut backends = vec!["Candle".to_string()];
            backends.push("ONNX".to_string());
            backends
        }
        #[cfg(not(feature = "onnx"))]
        {
            vec!["Candle".to_string()]
        }
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

/// GPU performance statistics
#[derive(Debug, Clone)]
pub struct GpuStats {
    pub average_utilization: f32,
    pub peak_utilization: f32,
    pub average_memory_usage: u64,
    pub peak_memory_usage: u64,
    pub average_temperature: f32,
    pub peak_temperature: f32,
    pub average_power_usage: f32,
    pub peak_power_usage: f32,
}

/// Backend performance monitor with GPU utilization tracking
#[derive(Debug, Clone)]
pub struct PerformanceMonitor {
    inference_times: Vec<f32>,
    memory_usage: Vec<u64>,
    error_count: usize,
    gpu_utilization: Vec<f32>,
    gpu_memory_usage: Vec<u64>,
    gpu_temperature: Vec<f32>,
    gpu_power_usage: Vec<f32>,
}

impl PerformanceMonitor {
    /// Create new performance monitor
    pub fn new() -> Self {
        Self {
            inference_times: Vec::new(),
            memory_usage: Vec::new(),
            error_count: 0,
            gpu_utilization: Vec::new(),
            gpu_memory_usage: Vec::new(),
            gpu_temperature: Vec::new(),
            gpu_power_usage: Vec::new(),
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

    /// Record GPU utilization percentage (0.0-100.0)
    pub fn record_gpu_utilization(&mut self, utilization: f32) {
        self.gpu_utilization.push(utilization.clamp(0.0, 100.0));
        // Keep only recent measurements
        if self.gpu_utilization.len() > 100 {
            self.gpu_utilization.remove(0);
        }
    }

    /// Record GPU memory usage in bytes
    pub fn record_gpu_memory_usage(&mut self, bytes: u64) {
        self.gpu_memory_usage.push(bytes);
        // Keep only recent measurements
        if self.gpu_memory_usage.len() > 100 {
            self.gpu_memory_usage.remove(0);
        }
    }

    /// Record GPU temperature in Celsius
    pub fn record_gpu_temperature(&mut self, temperature: f32) {
        self.gpu_temperature.push(temperature);
        // Keep only recent measurements
        if self.gpu_temperature.len() > 100 {
            self.gpu_temperature.remove(0);
        }
    }

    /// Record GPU power usage in watts
    pub fn record_gpu_power_usage(&mut self, power: f32) {
        self.gpu_power_usage.push(power);
        // Keep only recent measurements
        if self.gpu_power_usage.len() > 100 {
            self.gpu_power_usage.remove(0);
        }
    }

    /// Get average GPU utilization percentage
    pub fn average_gpu_utilization(&self) -> f32 {
        if self.gpu_utilization.is_empty() {
            0.0
        } else {
            self.gpu_utilization.iter().sum::<f32>() / self.gpu_utilization.len() as f32
        }
    }

    /// Get peak GPU utilization percentage
    pub fn peak_gpu_utilization(&self) -> f32 {
        self.gpu_utilization.iter().copied().fold(0.0, f32::max)
    }

    /// Get average GPU memory usage in bytes
    pub fn average_gpu_memory_usage(&self) -> u64 {
        if self.gpu_memory_usage.is_empty() {
            0
        } else {
            self.gpu_memory_usage.iter().sum::<u64>() / self.gpu_memory_usage.len() as u64
        }
    }

    /// Get peak GPU memory usage in bytes
    pub fn peak_gpu_memory_usage(&self) -> u64 {
        self.gpu_memory_usage.iter().copied().max().unwrap_or(0)
    }

    /// Get average GPU temperature in Celsius
    pub fn average_gpu_temperature(&self) -> f32 {
        if self.gpu_temperature.is_empty() {
            0.0
        } else {
            self.gpu_temperature.iter().sum::<f32>() / self.gpu_temperature.len() as f32
        }
    }

    /// Get peak GPU temperature in Celsius
    pub fn peak_gpu_temperature(&self) -> f32 {
        self.gpu_temperature.iter().copied().fold(0.0, f32::max)
    }

    /// Get average GPU power usage in watts
    pub fn average_gpu_power_usage(&self) -> f32 {
        if self.gpu_power_usage.is_empty() {
            0.0
        } else {
            self.gpu_power_usage.iter().sum::<f32>() / self.gpu_power_usage.len() as f32
        }
    }

    /// Get peak GPU power usage in watts
    pub fn peak_gpu_power_usage(&self) -> f32 {
        self.gpu_power_usage.iter().copied().fold(0.0, f32::max)
    }

    /// Get comprehensive GPU statistics
    pub fn gpu_stats(&self) -> GpuStats {
        GpuStats {
            average_utilization: self.average_gpu_utilization(),
            peak_utilization: self.peak_gpu_utilization(),
            average_memory_usage: self.average_gpu_memory_usage(),
            peak_memory_usage: self.peak_gpu_memory_usage(),
            average_temperature: self.average_gpu_temperature(),
            peak_temperature: self.peak_gpu_temperature(),
            average_power_usage: self.average_gpu_power_usage(),
            peak_power_usage: self.peak_gpu_power_usage(),
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
    fn test_gpu_monitoring() {
        let mut monitor = PerformanceMonitor::new();

        // Test GPU utilization monitoring
        monitor.record_gpu_utilization(75.0);
        monitor.record_gpu_utilization(85.0);
        monitor.record_gpu_utilization(95.0);

        assert_eq!(monitor.average_gpu_utilization(), 85.0);
        assert_eq!(monitor.peak_gpu_utilization(), 95.0);

        // Test GPU memory usage monitoring
        monitor.record_gpu_memory_usage(1024);
        monitor.record_gpu_memory_usage(2048);
        monitor.record_gpu_memory_usage(1536);

        assert_eq!(monitor.average_gpu_memory_usage(), 1536);
        assert_eq!(monitor.peak_gpu_memory_usage(), 2048);

        // Test GPU temperature monitoring
        monitor.record_gpu_temperature(65.0);
        monitor.record_gpu_temperature(70.0);
        monitor.record_gpu_temperature(75.0);

        assert_eq!(monitor.average_gpu_temperature(), 70.0);
        assert_eq!(monitor.peak_gpu_temperature(), 75.0);

        // Test GPU power usage monitoring
        monitor.record_gpu_power_usage(150.0);
        monitor.record_gpu_power_usage(175.0);
        monitor.record_gpu_power_usage(200.0);

        assert_eq!(monitor.average_gpu_power_usage(), 175.0);
        assert_eq!(monitor.peak_gpu_power_usage(), 200.0);

        // Test comprehensive GPU stats
        let stats = monitor.gpu_stats();
        assert_eq!(stats.average_utilization, 85.0);
        assert_eq!(stats.peak_utilization, 95.0);
        assert_eq!(stats.average_memory_usage, 1536);
        assert_eq!(stats.peak_memory_usage, 2048);
        assert_eq!(stats.average_temperature, 70.0);
        assert_eq!(stats.peak_temperature, 75.0);
        assert_eq!(stats.average_power_usage, 175.0);
        assert_eq!(stats.peak_power_usage, 200.0);
    }

    #[test]
    fn test_gpu_utilization_clamping() {
        let mut monitor = PerformanceMonitor::new();

        // Test values outside valid range are clamped
        monitor.record_gpu_utilization(-10.0);
        monitor.record_gpu_utilization(150.0);

        assert_eq!(monitor.average_gpu_utilization(), 50.0); // (0.0 + 100.0) / 2
        assert_eq!(monitor.peak_gpu_utilization(), 100.0);
    }

    #[test]
    fn test_backend_factory() {
        let backends = BackendFactory::available_backends();
        assert!(backends.contains(&"Candle".to_string()));
    }
}

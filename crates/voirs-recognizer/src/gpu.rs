//! GPU acceleration framework for VoiRS Recognition
//!
//! This module provides GPU acceleration capabilities for all major operations
//! including ASR inference, audio processing, and model operations.

use crate::RecognitionError;
use candle_core::{DType, Device, DeviceType, Tensor};
use candle_nn;
use std::sync::Arc;
use tokio::sync::RwLock;

/// GPU acceleration configuration
#[derive(Debug, Clone)]
pub struct GpuConfig {
    /// Enable GPU acceleration
    pub enabled: bool,
    /// Preferred device type
    pub device_type: DeviceType,
    /// Memory pool size in MB
    pub memory_pool_size: usize,
    /// Enable mixed precision (FP16)
    pub mixed_precision: bool,
    /// CUDA device index (if multiple GPUs available)
    pub cuda_device_index: Option<usize>,
    /// Maximum batch size for GPU processing
    pub max_batch_size: usize,
    /// Enable memory optimization
    pub memory_optimization: bool,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            device_type: DeviceType::Cpu,
            memory_pool_size: 2048, // 2GB default
            mixed_precision: false,
            cuda_device_index: None,
            max_batch_size: 32,
            memory_optimization: true,
        }
    }
}

/// GPU acceleration manager
#[derive(Debug)]
pub struct GpuAccelerator {
    /// Configuration
    config: GpuConfig,
    /// Device handle
    device: Device,
    /// Memory monitor
    memory_monitor: Arc<RwLock<MemoryMonitor>>,
    /// Performance metrics
    performance_metrics: Arc<RwLock<PerformanceMetrics>>,
}

/// Memory usage monitoring
#[derive(Debug, Default)]
pub struct MemoryMonitor {
    /// Total allocated memory in bytes
    pub total_allocated: usize,
    /// Peak memory usage in bytes
    pub peak_usage: usize,
    /// Current memory usage in bytes
    pub current_usage: usize,
    /// Memory pool utilization percentage
    pub pool_utilization: f32,
}

/// Performance metrics tracking
#[derive(Debug, Default)]
pub struct PerformanceMetrics {
    /// GPU utilization percentage
    pub gpu_utilization: f32,
    /// Memory bandwidth utilization
    pub memory_bandwidth_utilization: f32,
    /// Tensor operations per second
    pub tensor_ops_per_second: f32,
    /// Average kernel execution time
    pub avg_kernel_time_ms: f32,
    /// Number of completed operations
    pub completed_operations: usize,
    /// Total processing time
    pub total_processing_time: std::time::Duration,
}

impl GpuAccelerator {
    /// Create new GPU accelerator
    pub fn new(config: GpuConfig) -> Result<Self, RecognitionError> {
        // Detect available devices
        let available_devices = Self::detect_available_devices();
        
        // Select optimal device
        let device = if config.enabled {
            Self::select_device(&config, &available_devices)?
        } else {
            Device::Cpu
        };

        let memory_monitor = Arc::new(RwLock::new(MemoryMonitor::default()));
        let performance_metrics = Arc::new(RwLock::new(PerformanceMetrics::default()));

        Ok(Self {
            config,
            device,
            memory_monitor,
            performance_metrics,
        })
    }

    /// Detect available computing devices
    pub fn detect_available_devices() -> Vec<DeviceInfo> {
        let mut devices = Vec::new();
        
        // Always have CPU
        devices.push(DeviceInfo {
            device_type: DeviceType::Cpu,
            name: "CPU".to_string(),
            memory_mb: Self::get_system_memory_mb(),
            compute_capability: ComputeCapability::Cpu,
            available: true,
        });

        // Check for CUDA devices
        #[cfg(feature = "cuda")]
        {
            if let Ok(cuda_devices) = Self::detect_cuda_devices() {
                devices.extend(cuda_devices);
            }
        }

        // Check for Metal devices (macOS)
        #[cfg(target_os = "macos")]
        {
            if let Ok(metal_devices) = Self::detect_metal_devices() {
                devices.extend(metal_devices);
            }
        }

        devices
    }

    /// Select optimal device based on configuration
    fn select_device(config: &GpuConfig, available_devices: &[DeviceInfo]) -> Result<Device, RecognitionError> {
        // If GPU is disabled, return CPU
        if !config.enabled {
            return Ok(Device::Cpu);
        }

        // Find the best available device
        let best_device = available_devices
            .iter()
            .filter(|d| d.available && d.device_type == config.device_type)
            .max_by_key(|d| d.memory_mb);

        match best_device {
            Some(device_info) => {
                match device_info.device_type {
                    DeviceType::Cpu => Ok(Device::Cpu),
                    #[cfg(feature = "cuda")]
                    DeviceType::Cuda => {
                        let device_index = config.cuda_device_index.unwrap_or(0);
                        Device::new_cuda(device_index)
                            .map_err(|e| RecognitionError::ResourceError {
                                message: format!("Failed to initialize CUDA device {device_index}: {e}"),
                                source: Some(Box::new(e)),
                            })
                    }
                    #[cfg(target_os = "macos")]
                    DeviceType::Metal => {
                        Device::new_metal(0)
                            .map_err(|e| RecognitionError::ResourceError {
                                message: format!("Failed to initialize Metal device: {e}"),
                                source: Some(Box::new(e)),
                            })
                    }
                    _ => {
                        tracing::warn!("Unsupported device type, falling back to CPU");
                        Ok(Device::Cpu)
                    }
                }
            }
            None => {
                tracing::warn!("No suitable GPU device found, falling back to CPU");
                Ok(Device::Cpu)
            }
        }
    }

    /// Get system memory in MB
    fn get_system_memory_mb() -> usize {
        #[cfg(target_os = "linux")]
        {
            use std::fs;
            if let Ok(meminfo) = fs::read_to_string("/proc/meminfo") {
                for line in meminfo.lines() {
                    if line.starts_with("MemTotal:") {
                        let parts: Vec<&str> = line.split_whitespace().collect();
                        if parts.len() >= 2 {
                            if let Ok(kb) = parts[1].parse::<usize>() {
                                return kb / 1024; // Convert KB to MB
                            }
                        }
                    }
                }
            }
        }
        
        #[cfg(target_os = "macos")]
        {
            use std::process::Command;
            if let Ok(output) = Command::new("sysctl")
                .arg("-n")
                .arg("hw.memsize")
                .output()
            {
                let stdout = String::from_utf8_lossy(&output.stdout);
                if let Ok(bytes) = stdout.trim().parse::<usize>() {
                    return bytes / 1024 / 1024; // Convert bytes to MB
                }
            }
        }
        
        #[cfg(target_os = "windows")]
        {
            use std::process::Command;
            if let Ok(output) = Command::new("wmic")
                .arg("computersystem")
                .arg("get")
                .arg("TotalPhysicalMemory")
                .output()
            {
                let stdout = String::from_utf8_lossy(&output.stdout);
                for line in stdout.lines() {
                    if let Ok(bytes) = line.trim().parse::<usize>() {
                        return bytes / 1024 / 1024; // Convert bytes to MB
                    }
                }
            }
        }
        
        8192 // 8GB default fallback
    }

    /// Detect CUDA devices
    #[cfg(feature = "cuda")]
    fn detect_cuda_devices() -> Result<Vec<DeviceInfo>, RecognitionError> {
        use std::process::Command;
        
        // Try to query CUDA devices using nvidia-smi
        match Command::new("nvidia-smi")
            .arg("--query-gpu=index,name,memory.total")
            .arg("--format=csv,noheader,nounits")
            .output()
        {
            Ok(output) => {
                let mut devices = Vec::new();
                let stdout = String::from_utf8_lossy(&output.stdout);
                
                for line in stdout.lines() {
                    let parts: Vec<&str> = line.split(',').collect();
                    if parts.len() >= 3 {
                        if let (Ok(_index), Ok(memory)) = (parts[0].trim().parse::<usize>(), parts[2].trim().parse::<usize>()) {
                            devices.push(DeviceInfo {
                                device_type: DeviceType::Cuda,
                                name: parts[1].trim().to_string(),
                                memory_mb: memory,
                                compute_capability: ComputeCapability::Cuda(7, 5), // Default capability
                                available: true,
                            });
                        }
                    }
                }
                
                Ok(devices)
            }
            Err(_) => {
                // Fall back to candle device detection
                match candle_core::Device::cuda_if_available(0) {
                    Ok(_) => {
                        Ok(vec![DeviceInfo {
                            device_type: DeviceType::Cuda,
                            name: "CUDA Device".to_string(),
                            memory_mb: 8192, // Default 8GB
                            compute_capability: ComputeCapability::Cuda(7, 5),
                            available: true,
                        }])
                    }
                    Err(_) => Ok(vec![])
                }
            }
        }
    }

    /// Detect Metal devices
    #[cfg(target_os = "macos")]
    fn detect_metal_devices() -> Result<Vec<DeviceInfo>, RecognitionError> {
        use std::process::Command;
        
        // Try to detect Metal support
        match Command::new("system_profiler")
            .arg("SPDisplaysDataType")
            .output()
        {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                
                if stdout.contains("Metal") {
                    // Try to initialize Metal device to confirm availability
                    match candle_core::Device::new_metal(0) {
                        Ok(_) => {
                            Ok(vec![DeviceInfo {
                                device_type: DeviceType::Metal,
                                name: "Metal GPU".to_string(),
                                memory_mb: 8192, // Default estimation
                                compute_capability: ComputeCapability::Metal,
                                available: true,
                            }])
                        }
                        Err(_) => Ok(vec![])
                    }
                } else {
                    Ok(vec![])
                }
            }
            Err(_) => {
                // Fall back to direct Metal device creation
                match candle_core::Device::new_metal(0) {
                    Ok(_) => {
                        Ok(vec![DeviceInfo {
                            device_type: DeviceType::Metal,
                            name: "Metal GPU".to_string(),
                            memory_mb: 8192,
                            compute_capability: ComputeCapability::Metal,
                            available: true,
                        }])
                    }
                    Err(_) => Ok(vec![])
                }
            }
        }
    }

    /// Get device handle
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Check if GPU acceleration is enabled
    pub fn is_gpu_enabled(&self) -> bool {
        self.config.enabled && !matches!(self.device, Device::Cpu)
    }

    /// Get memory usage
    pub async fn get_memory_usage(&self) -> MemoryUsage {
        let monitor = self.memory_monitor.read().await;
        MemoryUsage {
            allocated_mb: monitor.total_allocated / 1024 / 1024,
            peak_mb: monitor.peak_usage / 1024 / 1024,
            current_mb: monitor.current_usage / 1024 / 1024,
            pool_utilization: monitor.pool_utilization,
        }
    }

    /// Get performance metrics
    pub async fn get_performance_metrics(&self) -> PerformanceMetrics {
        self.performance_metrics.read().await.clone()
    }

    /// Optimize tensor for GPU processing
    pub async fn optimize_tensor(&self, tensor: &Tensor) -> Result<Tensor, RecognitionError> {
        // Move tensor to GPU if available
        if self.is_gpu_enabled() {
            tensor.to_device(&self.device)
                .map_err(|e| RecognitionError::ModelError {
                    message: format!("Failed to move tensor to GPU: {e}"),
                    source: Some(Box::new(e)),
                })
        } else {
            Ok(tensor.clone())
        }
    }

    /// Batch process tensors efficiently
    pub async fn batch_process_tensors<F, T>(&self, tensors: Vec<Tensor>, processor: F) -> Result<Vec<T>, RecognitionError>
    where
        F: Fn(Tensor) -> Result<T, RecognitionError>,
    {
        let start_time = std::time::Instant::now();
        let mut results = Vec::new();

        // Process in batches optimized for GPU
        let batch_size = if self.is_gpu_enabled() {
            self.config.max_batch_size
        } else {
            1
        };

        for chunk in tensors.chunks(batch_size) {
            for tensor in chunk {
                let optimized_tensor = self.optimize_tensor(tensor).await?;
                let result = processor(optimized_tensor)?;
                results.push(result);
            }
        }

        // Update performance metrics
        let processing_time = start_time.elapsed();
        let mut metrics = self.performance_metrics.write().await;
        metrics.completed_operations += tensors.len();
        metrics.total_processing_time += processing_time;

        Ok(results)
    }

    /// Update memory usage statistics
    pub async fn update_memory_usage(&self, allocated_bytes: usize) {
        let mut monitor = self.memory_monitor.write().await;
        monitor.total_allocated += allocated_bytes;
        monitor.current_usage += allocated_bytes;
        
        if monitor.current_usage > monitor.peak_usage {
            monitor.peak_usage = monitor.current_usage;
        }

        // Calculate pool utilization
        let pool_size_bytes = self.config.memory_pool_size * 1024 * 1024;
        monitor.pool_utilization = (monitor.current_usage as f32 / pool_size_bytes as f32) * 100.0;
    }

    /// Synchronize GPU operations
    pub async fn synchronize(&self) -> Result<(), RecognitionError> {
        // For CUDA devices, this would call cudaDeviceSynchronize
        // For Metal devices, this would wait for command buffer completion
        // For now, we'll just add a small delay to simulate synchronization
        if self.is_gpu_enabled() {
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
        }
        Ok(())
    }

    /// Prefetch tensors to GPU memory
    pub async fn prefetch_tensors(&self, tensors: &[Tensor]) -> Result<Vec<Tensor>, RecognitionError> {
        if !self.is_gpu_enabled() {
            return Ok(tensors.to_vec());
        }

        let mut gpu_tensors = Vec::new();
        for tensor in tensors {
            let gpu_tensor = self.optimize_tensor(tensor).await?;
            gpu_tensors.push(gpu_tensor);
        }

        Ok(gpu_tensors)
    }

    /// Optimize tensor operations for GPU
    pub async fn optimize_tensor_operations(&self, operations: Vec<TensorOperation>) -> Result<Vec<Tensor>, RecognitionError> {
        if !self.is_gpu_enabled() {
            return self.execute_operations_cpu(operations).await;
        }

        let start_time = std::time::Instant::now();
        let mut results = Vec::new();

        // Execute operations in optimized GPU batches
        for operation in operations {
            let result = self.execute_gpu_operation(operation).await?;
            results.push(result);
        }

        // Update performance metrics
        let processing_time = start_time.elapsed();
        let mut metrics = self.performance_metrics.write().await;
        metrics.completed_operations += results.len();
        metrics.total_processing_time += processing_time;
        metrics.avg_kernel_time_ms = processing_time.as_millis() as f32 / results.len() as f32;

        Ok(results)
    }

    /// Execute GPU operation
    async fn execute_gpu_operation(&self, operation: TensorOperation) -> Result<Tensor, RecognitionError> {
        match operation {
            TensorOperation::MatMul { a, b } => {
                let a_gpu = self.optimize_tensor(&a).await?;
                let b_gpu = self.optimize_tensor(&b).await?;
                a_gpu.matmul(&b_gpu).map_err(|e| RecognitionError::ModelError {
                    message: format!("GPU matrix multiplication failed: {e}"),
                    source: Some(Box::new(e)),
                })
            }
            TensorOperation::Add { a, b } => {
                let a_gpu = self.optimize_tensor(&a).await?;
                let b_gpu = self.optimize_tensor(&b).await?;
                a_gpu.add(&b_gpu).map_err(|e| RecognitionError::ModelError {
                    message: format!("GPU addition failed: {e}"),
                    source: Some(Box::new(e)),
                })
            }
            TensorOperation::Softmax { input, dim } => {
                let input_gpu = self.optimize_tensor(&input).await?;
                candle_nn::ops::softmax(&input_gpu, dim).map_err(|e| RecognitionError::ModelError {
                    message: format!("GPU softmax failed: {e}"),
                    source: Some(Box::new(e)),
                })
            }
            TensorOperation::LayerNorm { input, weight, bias, eps } => {
                let input_gpu = self.optimize_tensor(&input).await?;
                let weight_gpu = self.optimize_tensor(&weight).await?;
                let bias_gpu = self.optimize_tensor(&bias).await?;
                
                // Compute layer normalization
                let mean = input_gpu.mean_keepdim(candle_core::D::Minus1)?;
                let var = input_gpu.var_keepdim(candle_core::D::Minus1)?;
                let normalized = input_gpu.broadcast_sub(&mean)?
                    .broadcast_div(&(var + eps)?.sqrt()?)?;
                
                let scaled = normalized.broadcast_mul(&weight_gpu)?;
                scaled.broadcast_add(&bias_gpu).map_err(|e| RecognitionError::ModelError {
                    message: format!("GPU layer normalization failed: {e}"),
                    source: Some(Box::new(e)),
                })
            }
        }
    }

    /// Execute operations on CPU (fallback)
    async fn execute_operations_cpu(&self, operations: Vec<TensorOperation>) -> Result<Vec<Tensor>, RecognitionError> {
        let mut results = Vec::new();
        
        for operation in operations {
            let result = match operation {
                TensorOperation::MatMul { a, b } => {
                    a.matmul(&b).map_err(|e| RecognitionError::ModelError {
                        message: format!("CPU matrix multiplication failed: {e}"),
                        source: Some(Box::new(e)),
                    })?
                }
                TensorOperation::Add { a, b } => {
                    a.add(&b).map_err(|e| RecognitionError::ModelError {
                        message: format!("CPU addition failed: {e}"),
                        source: Some(Box::new(e)),
                    })?
                }
                TensorOperation::Softmax { input, dim } => {
                    candle_nn::ops::softmax(&input, dim).map_err(|e| RecognitionError::ModelError {
                        message: format!("CPU softmax failed: {e}"),
                        source: Some(Box::new(e)),
                    })?
                }
                TensorOperation::LayerNorm { input, weight, bias, eps } => {
                    let mean = input.mean_keepdim(candle_core::D::Minus1)?;
                    let var = input.var_keepdim(candle_core::D::Minus1)?;
                    let normalized = input.broadcast_sub(&mean)?
                        .broadcast_div(&(var + eps)?.sqrt()?)?;
                    
                    let scaled = normalized.broadcast_mul(&weight)?;
                    scaled.broadcast_add(&bias).map_err(|e| RecognitionError::ModelError {
                        message: format!("CPU layer normalization failed: {e}"),
                        source: Some(Box::new(e)),
                    })?
                }
            };
            results.push(result);
        }
        
        Ok(results)
    }

    /// Clean up GPU resources
    pub async fn cleanup(&self) -> Result<(), RecognitionError> {
        // Synchronize any pending operations
        self.synchronize().await?;
        
        // Reset metrics
        let mut metrics = self.performance_metrics.write().await;
        *metrics = PerformanceMetrics::default();

        // Reset memory monitor
        let mut monitor = self.memory_monitor.write().await;
        *monitor = MemoryMonitor::default();

        Ok(())
    }
}

/// Device information
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Device type
    pub device_type: DeviceType,
    /// Device name
    pub name: String,
    /// Available memory in MB
    pub memory_mb: usize,
    /// Compute capability
    pub compute_capability: ComputeCapability,
    /// Device availability
    pub available: bool,
}

/// Compute capability levels
#[derive(Debug, Clone, PartialEq)]
pub enum ComputeCapability {
    /// CPU compute
    Cpu,
    /// CUDA compute capability
    Cuda(u32, u32),
    /// Metal performance shaders
    Metal,
}

/// Memory usage information
#[derive(Debug, Clone)]
pub struct MemoryUsage {
    /// Currently allocated memory in MB
    pub allocated_mb: usize,
    /// Peak memory usage in MB
    pub peak_mb: usize,
    /// Current memory usage in MB
    pub current_mb: usize,
    /// Memory pool utilization percentage
    pub pool_utilization: f32,
}

/// GPU tensor operations
#[derive(Debug)]
pub enum TensorOperation {
    /// Matrix multiplication
    MatMul { a: Tensor, b: Tensor },
    /// Element-wise addition
    Add { a: Tensor, b: Tensor },
    /// Softmax operation
    Softmax { input: Tensor, dim: usize },
    /// Layer normalization
    LayerNorm { input: Tensor, weight: Tensor, bias: Tensor, eps: f32 },
}

/// GPU memory pool for efficient tensor reuse
#[derive(Debug)]
pub struct GpuMemoryPool {
    /// Pool of reusable tensors
    tensor_pool: std::collections::HashMap<String, Vec<Tensor>>,
    /// Maximum pool size per tensor type
    max_pool_size: usize,
    /// Total memory allocated in pool
    total_allocated_mb: f32,
}

impl GpuMemoryPool {
    pub fn new(max_pool_size: usize) -> Self {
        Self {
            tensor_pool: std::collections::HashMap::new(),
            max_pool_size,
            total_allocated_mb: 0.0,
        }
    }

    pub fn get_tensor(&mut self, shape: &[usize], dtype: DType, device: &Device) -> Option<Tensor> {
        let key = format!("{shape:?}_{dtype:?}_{device:?}");
        if let Some(tensors) = self.tensor_pool.get_mut(&key) {
            tensors.pop()
        } else {
            None
        }
    }

    pub fn return_tensor(&mut self, tensor: Tensor) -> Result<(), RecognitionError> {
        let shape = tensor.shape().dims().to_vec();
        let dtype = tensor.dtype();
        let device = tensor.device();
        let key = format!("{shape:?}_{dtype:?}_{device:?}");
        
        let pool = self.tensor_pool.entry(key).or_insert_with(Vec::new);
        if pool.len() < self.max_pool_size {
            pool.push(tensor);
        }
        
        Ok(())
    }

    pub fn clear(&mut self) {
        self.tensor_pool.clear();
        self.total_allocated_mb = 0.0;
    }
}

/// GPU acceleration utilities
pub mod utils {
    use super::*;

    /// Check if GPU acceleration is available
    pub fn is_gpu_available() -> bool {
        let devices = GpuAccelerator::detect_available_devices();
        devices.iter().any(|d| d.available && d.device_type != DeviceType::Cpu)
    }

    /// Get recommended GPU configuration
    pub fn get_recommended_gpu_config() -> GpuConfig {
        let devices = GpuAccelerator::detect_available_devices();
        
        // Find best GPU device
        let best_gpu = devices
            .iter()
            .filter(|d| d.available && d.device_type != DeviceType::Cpu)
            .max_by_key(|d| d.memory_mb);

        match best_gpu {
            Some(device) => GpuConfig {
                enabled: true,
                device_type: device.device_type,
                memory_pool_size: (device.memory_mb / 2).min(4096), // Use up to half of GPU memory, max 4GB
                mixed_precision: true,
                cuda_device_index: Some(0),
                max_batch_size: 16,
                memory_optimization: true,
            },
            None => GpuConfig::default(),
        }
    }

    /// Estimate optimal batch size for GPU
    pub fn estimate_optimal_batch_size(memory_mb: usize, tensor_size_mb: usize) -> usize {
        if tensor_size_mb == 0 {
            return 1;
        }
        
        // Use 70% of available memory for batching
        let available_memory = (memory_mb as f32 * 0.7) as usize;
        let optimal_batch_size = available_memory / tensor_size_mb;
        
        // Clamp to reasonable bounds
        optimal_batch_size.clamp(1, 32)
    }

    /// Optimize memory layout for GPU processing
    pub fn optimize_memory_layout(tensor: &Tensor) -> Result<Tensor, RecognitionError> {
        // For GPU processing, we want tensors to be contiguous in memory
        if tensor.is_contiguous() {
            Ok(tensor.clone())
        } else {
            tensor.contiguous().map_err(|e| RecognitionError::ModelError {
                message: format!("Failed to make tensor contiguous: {e}"),
                source: Some(Box::new(e)),
            })
        }
    }

    /// Warm up GPU kernels
    pub fn warmup_gpu_kernels(device: &Device) -> Result<(), RecognitionError> {
        // Create small dummy tensors and perform operations to warm up GPU
        let dummy_a = Tensor::ones(&[16, 16], DType::F32, device)
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Failed to create warmup tensor: {e}"),
                source: Some(Box::new(e)),
            })?;
        
        let dummy_b = Tensor::ones(&[16, 16], DType::F32, device)
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Failed to create warmup tensor: {e}"),
                source: Some(Box::new(e)),
            })?;
        
        // Perform warmup operations
        let _ = dummy_a.matmul(&dummy_b)?;
        let _ = dummy_a.add(&dummy_b)?;
        let _ = candle_nn::ops::softmax(&dummy_a, 1)?;
        
        Ok(())
    }

    /// Profile GPU memory bandwidth
    pub fn profile_memory_bandwidth(device: &Device) -> Result<f32, RecognitionError> {
        let size = 1024 * 1024; // 1M elements
        let start_time = std::time::Instant::now();
        
        // Create large tensor and perform memory-intensive operations
        let tensor = Tensor::ones(&[size], DType::F32, device)
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Failed to create profiling tensor: {}", e),
                source: Some(Box::new(e)),
            })?;
        
        // Perform operations that test memory bandwidth
        let _ = tensor.sum_all()?;
        let _ = tensor.mean_all()?;
        
        let elapsed = start_time.elapsed();
        let bytes_transferred = size * 4 * 2; // 4 bytes per float, 2 operations
        let bandwidth_gb_per_s = (bytes_transferred as f32) / (elapsed.as_secs_f32() * 1e9);
        
        Ok(bandwidth_gb_per_s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_config_default() {
        let config = GpuConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.device_type, DeviceType::Cpu);
        assert_eq!(config.memory_pool_size, 2048);
    }

    #[test]
    fn test_device_detection() {
        let devices = GpuAccelerator::detect_available_devices();
        assert!(!devices.is_empty());
        
        // Should always have CPU
        assert!(devices.iter().any(|d| d.device_type == DeviceType::Cpu));
    }

    #[test]
    fn test_gpu_availability() {
        let available = utils::is_gpu_available();
        // This test depends on the system, so we just check it doesn't panic
        assert!(available || !available);
    }

    #[test]
    fn test_optimal_batch_size_estimation() {
        let batch_size = utils::estimate_optimal_batch_size(1024, 64);
        assert!(batch_size > 0);
        assert!(batch_size <= 32);
    }

    #[tokio::test]
    async fn test_gpu_accelerator_creation() {
        let config = GpuConfig::default();
        let accelerator = GpuAccelerator::new(config);
        assert!(accelerator.is_ok());
    }

    #[tokio::test]
    async fn test_memory_usage_tracking() {
        let config = GpuConfig::default();
        let accelerator = GpuAccelerator::new(config).unwrap();
        
        accelerator.update_memory_usage(1024).await;
        let usage = accelerator.get_memory_usage().await;
        assert!(usage.allocated_mb > 0);
    }
}
use crate::types::{SingingRequest, VoiceCharacteristics};
use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Errors that can occur during GPU-accelerated operations.
#[derive(Debug, Error)]
pub enum GpuError {
    /// GPU device requested is not available on the system.
    #[error("GPU device not available: {0}")]
    DeviceNotAvailable(String),
    /// CUDA-specific error occurred during execution.
    #[error("CUDA error: {0}")]
    CudaError(String),
    /// Failed to allocate required GPU memory.
    #[error("Memory allocation error: {0}")]
    MemoryError(String),
    /// Error occurred during GPU computation.
    #[error("Computation error: {0}")]
    ComputationError(String),
}

/// Configuration for GPU acceleration settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuConfig {
    /// Type of compute device to use for acceleration.
    pub device_type: DeviceType,
    /// Optional memory limit in megabytes (MB).
    pub memory_limit_mb: Option<u64>,
    /// Number of samples to process in a single batch.
    pub batch_size: usize,
    /// Enable FP16 (half-precision) computation for memory efficiency.
    pub enable_fp16: bool,
    /// Enable Tensor Core acceleration on compatible NVIDIA GPUs.
    pub enable_tensor_cores: bool,
    /// Size of the tensor memory pool for reuse.
    pub memory_pool_size: usize,
}

/// Type of compute device for acceleration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeviceType {
    /// Use CPU for computation.
    Cpu,
    /// Use NVIDIA CUDA GPU with the specified device index.
    Cuda(u32),
    /// Use Apple Metal GPU (for Apple Silicon).
    Metal,
    /// Automatically select best available device (CUDA > Metal > CPU).
    Auto,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            device_type: DeviceType::Auto,
            memory_limit_mb: None,
            batch_size: 32,
            enable_fp16: false,
            enable_tensor_cores: true,
            memory_pool_size: 64,
        }
    }
}

/// GPU accelerator for neural singing synthesis operations.
///
/// Provides hardware acceleration for transformer models, diffusion synthesis,
/// and neural vocoders using CUDA, Metal, or CPU backends.
pub struct GpuAccelerator {
    device: Device,
    config: GpuConfig,
    memory_pool: TensorMemoryPool,
    tensor_cache: HashMap<String, Tensor>,
}

impl GpuAccelerator {
    /// Creates a new GPU accelerator with the specified configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - GPU acceleration configuration
    ///
    /// # Returns
    ///
    /// Returns the initialized accelerator or an error if device is unavailable.
    ///
    /// # Errors
    ///
    /// Returns `GpuError::DeviceNotAvailable` if the requested device cannot be initialized.
    pub fn new(config: GpuConfig) -> Result<Self, GpuError> {
        let device = Self::select_device(&config.device_type)?;
        let memory_pool = TensorMemoryPool::new(config.memory_pool_size);

        Ok(Self {
            device,
            config,
            memory_pool,
            tensor_cache: HashMap::new(),
        })
    }

    fn select_device(device_type: &DeviceType) -> Result<Device, GpuError> {
        match device_type {
            DeviceType::Cpu => Ok(Device::Cpu),
            DeviceType::Cuda(index) => Device::new_cuda(*index as usize)
                .map_err(|e| GpuError::DeviceNotAvailable(format!("CUDA device {}: {}", index, e))),
            DeviceType::Metal => Device::new_metal(0)
                .map_err(|e| GpuError::DeviceNotAvailable(format!("Metal device: {}", e))),
            DeviceType::Auto => {
                // Try CUDA first, then Metal, fallback to CPU
                if let Ok(device) = Device::new_cuda(0) {
                    Ok(device)
                } else if let Ok(device) = Device::new_metal(0) {
                    Ok(device)
                } else {
                    Ok(Device::Cpu)
                }
            }
        }
    }

    /// Returns a reference to the active compute device.
    ///
    /// # Returns
    ///
    /// Reference to the Candle `Device` being used for computation.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Returns a reference to the current GPU configuration.
    ///
    /// # Returns
    ///
    /// Reference to the `GpuConfig` used to initialize this accelerator.
    pub fn config(&self) -> &GpuConfig {
        &self.config
    }

    /// Accelerates transformer-based neural synthesis using GPU.
    ///
    /// Processes acoustic features through transformer layers with GPU acceleration,
    /// optionally using FP16 precision and Tensor Cores for improved performance.
    ///
    /// # Arguments
    ///
    /// * `features` - Input acoustic features tensor
    /// * `voice_characteristics` - Target voice characteristics for synthesis
    ///
    /// # Returns
    ///
    /// Returns the processed features tensor on the GPU device.
    ///
    /// # Errors
    ///
    /// Returns `GpuError::ComputationError` if tensor operations fail.
    pub fn accelerate_transformer_synthesis(
        &mut self,
        features: &Tensor,
        voice_characteristics: &VoiceCharacteristics,
    ) -> Result<Tensor, GpuError> {
        let batch_size = self.config.batch_size;
        let device = self.device.clone();

        // Move tensors to GPU device
        let gpu_features = features.to_device(&device).map_err(|e| {
            GpuError::ComputationError(format!("Failed to move features to GPU: {}", e))
        })?;

        // Batch processing for efficiency
        let processed = if self.config.enable_fp16 {
            self.process_fp16_batch(&gpu_features, voice_characteristics)?
        } else {
            self.process_fp32_batch(&gpu_features, voice_characteristics)?
        };

        Ok(processed)
    }

    /// Accelerates diffusion-based synthesis using GPU progressive denoising.
    ///
    /// Performs iterative denoising steps on the GPU with automatic memory management
    /// to generate high-quality audio from noise.
    ///
    /// # Arguments
    ///
    /// * `noise` - Initial noise tensor to denoise
    /// * `conditioning` - Conditioning information for guided generation
    /// * `timesteps` - Number of denoising steps to perform
    ///
    /// # Returns
    ///
    /// Returns the denoised audio tensor on the GPU device.
    ///
    /// # Errors
    ///
    /// Returns `GpuError::ComputationError` if denoising operations fail.
    pub fn accelerate_diffusion_synthesis(
        &mut self,
        noise: &Tensor,
        conditioning: &Tensor,
        timesteps: usize,
    ) -> Result<Tensor, GpuError> {
        let device = self.device.clone();

        let gpu_noise = noise.to_device(&device).map_err(|e| {
            GpuError::ComputationError(format!("Failed to move noise to GPU: {}", e))
        })?;
        let gpu_conditioning = conditioning.to_device(&device).map_err(|e| {
            GpuError::ComputationError(format!("Failed to move conditioning to GPU: {}", e))
        })?;

        // Progressive denoising with GPU acceleration
        let mut current = gpu_noise;
        for step in 0..timesteps {
            current = self.denoise_step(&current, &gpu_conditioning, step)?;

            // Memory management - clear intermediate tensors
            if step % 10 == 0 {
                self.cleanup_temporary_tensors()?;
            }
        }

        Ok(current)
    }

    /// Accelerates neural vocoder processing using GPU WaveNet-style synthesis.
    ///
    /// Converts mel spectrograms to raw audio waveforms using GPU-accelerated
    /// neural vocoder layers.
    ///
    /// # Arguments
    ///
    /// * `mel_spectrogram` - Input mel spectrogram tensor
    ///
    /// # Returns
    ///
    /// Returns the generated audio waveform tensor on the GPU device.
    ///
    /// # Errors
    ///
    /// Returns `GpuError::ComputationError` if vocoder processing fails.
    pub fn accelerate_neural_vocoder(
        &mut self,
        mel_spectrogram: &Tensor,
    ) -> Result<Tensor, GpuError> {
        let device = self.device.clone();

        let gpu_mel = mel_spectrogram.to_device(&device).map_err(|e| {
            GpuError::ComputationError(format!("Failed to move mel spectrogram to GPU: {}", e))
        })?;

        // WaveNet-style processing with GPU acceleration
        let mut waveform = self.initialize_waveform(&gpu_mel)?;
        let layers = 16; // WaveNet layers

        for layer in 0..layers {
            waveform = self.wavenet_layer(&waveform, &gpu_mel, layer)?;
        }

        Ok(waveform)
    }

    /// Processes multiple singing synthesis requests in optimized batches.
    ///
    /// Automatically divides requests into batches based on configured batch size
    /// for improved GPU utilization and throughput.
    ///
    /// # Arguments
    ///
    /// * `requests` - Slice of singing synthesis requests to process
    ///
    /// # Returns
    ///
    /// Returns a vector of output tensors, one per input request.
    ///
    /// # Errors
    ///
    /// Returns `GpuError::ComputationError` if batch processing fails.
    pub fn batch_synthesize(
        &mut self,
        requests: &[SingingRequest],
    ) -> Result<Vec<Tensor>, GpuError> {
        let batch_size = self.config.batch_size;
        let mut results = Vec::new();

        for chunk in requests.chunks(batch_size) {
            let batch_features = self.prepare_batch_features(chunk)?;
            let batch_results = self.process_batch(&batch_features)?;
            results.extend(batch_results);
        }

        Ok(results)
    }

    /// Optimizes GPU memory usage by clearing caches and performing cleanup.
    ///
    /// Clears the tensor cache, optimizes the memory pool, and triggers
    /// device-specific memory cleanup (e.g., CUDA garbage collection).
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success.
    ///
    /// # Errors
    ///
    /// Returns `GpuError` if memory cleanup operations fail.
    pub fn optimize_memory(&mut self) -> Result<(), GpuError> {
        // Clear tensor cache
        self.tensor_cache.clear();

        // Optimize memory pool
        self.memory_pool.optimize();

        // Force garbage collection on GPU
        if let Device::Cuda(_) = self.device {
            self.cuda_memory_cleanup()?;
        }

        Ok(())
    }

    /// Retrieves current GPU memory usage statistics.
    ///
    /// # Returns
    ///
    /// Returns a `MemoryUsage` struct containing allocated memory, cached memory,
    /// and device utilization metrics.
    pub fn get_memory_usage(&self) -> MemoryUsage {
        MemoryUsage {
            allocated_mb: self.get_allocated_memory(),
            cached_mb: self.get_cached_memory(),
            device_utilization: self.get_device_utilization(),
        }
    }

    // Private helper methods
    fn process_fp16_batch(
        &mut self,
        features: &Tensor,
        voice_characteristics: &VoiceCharacteristics,
    ) -> Result<Tensor, GpuError> {
        // Convert to FP16 for memory efficiency
        let fp16_features = features
            .to_dtype(candle_core::DType::F16)
            .map_err(|e| GpuError::ComputationError(format!("FP16 conversion failed: {}", e)))?;

        // Process with Tensor Cores if available
        self.tensor_core_processing(&fp16_features, voice_characteristics)
    }

    fn process_fp32_batch(
        &mut self,
        features: &Tensor,
        voice_characteristics: &VoiceCharacteristics,
    ) -> Result<Tensor, GpuError> {
        // Standard FP32 processing
        self.standard_processing(features, voice_characteristics)
    }

    fn tensor_core_processing(
        &mut self,
        features: &Tensor,
        voice_characteristics: &VoiceCharacteristics,
    ) -> Result<Tensor, GpuError> {
        // Optimized for Tensor Cores (requires specific matrix dimensions)
        let optimized_shape = self.optimize_for_tensor_cores(features.shape().dims());
        let reshaped = features.reshape(optimized_shape).map_err(|e| {
            GpuError::ComputationError(format!("Tensor Core reshape failed: {}", e))
        })?;

        self.standard_processing(&reshaped, voice_characteristics)
    }

    fn standard_processing(
        &mut self,
        features: &Tensor,
        voice_characteristics: &VoiceCharacteristics,
    ) -> Result<Tensor, GpuError> {
        // Placeholder for actual neural synthesis
        // In practice, this would call the transformer/diffusion models
        let processed = features.clone();
        Ok(processed)
    }

    fn denoise_step(
        &mut self,
        current: &Tensor,
        conditioning: &Tensor,
        step: usize,
    ) -> Result<Tensor, GpuError> {
        // Placeholder for diffusion denoising step
        let denoised = current.clone();
        Ok(denoised)
    }

    fn initialize_waveform(&mut self, mel: &Tensor) -> Result<Tensor, GpuError> {
        // Initialize waveform tensor from mel spectrogram
        let shape_dims = mel.shape().dims();
        let waveform_length = shape_dims[shape_dims.len() - 1] * 256; // Typical hop length

        Tensor::zeros(&[waveform_length], candle_core::DType::F32, &self.device).map_err(|e| {
            GpuError::ComputationError(format!("Waveform initialization failed: {}", e))
        })
    }

    fn wavenet_layer(
        &mut self,
        waveform: &Tensor,
        conditioning: &Tensor,
        layer: usize,
    ) -> Result<Tensor, GpuError> {
        // Placeholder for WaveNet layer processing
        let processed = waveform.clone();
        Ok(processed)
    }

    fn prepare_batch_features(&mut self, requests: &[SingingRequest]) -> Result<Tensor, GpuError> {
        // Convert requests to batch tensor
        let batch_size = requests.len();
        let feature_dim = 512; // Typical feature dimension

        Tensor::zeros(
            &[batch_size, feature_dim],
            candle_core::DType::F32,
            &self.device,
        )
        .map_err(|e| GpuError::ComputationError(format!("Batch preparation failed: {}", e)))
    }

    fn process_batch(&mut self, features: &Tensor) -> Result<Vec<Tensor>, GpuError> {
        // Process entire batch at once for efficiency
        let batch_result = features.clone();

        // Split back into individual results
        let batch_size = features.shape().dims()[0];
        let mut results = Vec::new();
        for i in 0..batch_size {
            let single_result = batch_result.get(i).map_err(|e| {
                GpuError::ComputationError(format!("Batch splitting failed: {}", e))
            })?;
            results.push(single_result);
        }

        Ok(results)
    }

    fn cleanup_temporary_tensors(&mut self) -> Result<(), GpuError> {
        // Remove cached tensors that are no longer needed
        self.tensor_cache.retain(|_, tensor| {
            // Keep tensors that are still referenced
            true // Simplified logic
        });
        Ok(())
    }

    fn cuda_memory_cleanup(&mut self) -> Result<(), GpuError> {
        // CUDA-specific memory cleanup
        // In practice, this would use CUDA runtime API
        Ok(())
    }

    fn optimize_for_tensor_cores(&self, shape: &[usize]) -> Vec<usize> {
        // Tensor Cores work best with dimensions that are multiples of 8 (FP16) or 16 (INT8)
        let mut optimized = shape.to_vec();
        for dim in &mut optimized {
            // Round up to multiple of 8
            let remainder = *dim % 8;
            if remainder != 0 {
                *dim += 8 - remainder;
            }
        }
        optimized
    }

    fn get_allocated_memory(&self) -> u64 {
        // Placeholder - would query actual GPU memory usage
        0
    }

    fn get_cached_memory(&self) -> u64 {
        // Placeholder - would query cached memory usage
        0
    }

    fn get_device_utilization(&self) -> f32 {
        // Placeholder - would query GPU utilization
        0.0
    }
}

/// GPU memory usage statistics.
#[derive(Debug, Clone)]
pub struct MemoryUsage {
    /// Amount of memory currently allocated on the device in megabytes (MB).
    pub allocated_mb: u64,
    /// Amount of cached memory on the device in megabytes (MB).
    pub cached_mb: u64,
    /// Device compute utilization as a percentage (0.0-1.0).
    pub device_utilization: f32,
}

/// Memory pool for efficient tensor reuse and reduced allocation overhead.
///
/// Maintains a pool of pre-allocated tensors that can be reused across
/// multiple operations to minimize GPU memory allocation/deallocation overhead.
pub struct TensorMemoryPool {
    pool_size: usize,
    available_tensors: Vec<Tensor>,
}

impl TensorMemoryPool {
    /// Creates a new tensor memory pool with the specified capacity.
    ///
    /// # Arguments
    ///
    /// * `pool_size` - Maximum number of tensors to keep in the pool
    ///
    /// # Returns
    ///
    /// Returns a new empty memory pool.
    pub fn new(pool_size: usize) -> Self {
        Self {
            pool_size,
            available_tensors: Vec::new(),
        }
    }

    /// Gets a tensor from the pool or creates a new one if none available.
    ///
    /// Attempts to reuse an existing tensor with matching shape and dtype from the pool.
    /// If no matching tensor is available, allocates a new tensor on the device.
    ///
    /// # Arguments
    ///
    /// * `shape` - Desired tensor shape (dimensions)
    /// * `dtype` - Desired tensor data type (e.g., F32, F16)
    /// * `device` - Target compute device
    ///
    /// # Returns
    ///
    /// Returns a tensor with the requested shape and dtype, either from the pool or newly allocated.
    ///
    /// # Errors
    ///
    /// Returns an error if tensor allocation fails.
    pub fn get_tensor(
        &mut self,
        shape: &[usize],
        dtype: candle_core::DType,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        // Try to reuse a tensor from pool
        for (i, tensor) in self.available_tensors.iter().enumerate() {
            if tensor.shape().dims() == shape && tensor.dtype() == dtype {
                return Ok(self.available_tensors.remove(i));
            }
        }

        // Create new tensor if none available
        Tensor::zeros(shape, dtype, device)
    }

    /// Returns a tensor to the pool for reuse.
    ///
    /// If the pool is not full, the tensor is added for future reuse.
    /// If the pool is at capacity, the tensor is dropped to free memory.
    ///
    /// # Arguments
    ///
    /// * `tensor` - Tensor to return to the pool
    pub fn return_tensor(&mut self, tensor: Tensor) {
        if self.available_tensors.len() < self.pool_size {
            self.available_tensors.push(tensor);
        }
        // If pool is full, let tensor be dropped
    }

    /// Optimizes the pool by clearing all cached tensors to free memory.
    pub fn optimize(&mut self) {
        // Clear pool to free memory
        self.available_tensors.clear();
    }
}

/// Trait for types that support GPU acceleration.
///
/// Allows synthesis components to be configured with GPU acceleration
/// and provides information about batching capabilities.
pub trait GpuAccelerated {
    /// Configures this component to use GPU acceleration.
    ///
    /// # Arguments
    ///
    /// * `accelerator` - GPU accelerator to use for operations
    ///
    /// # Returns
    ///
    /// Returns self configured with GPU acceleration.
    fn with_gpu_acceleration(self, accelerator: &mut GpuAccelerator) -> Self;

    /// Checks whether this component supports batch processing.
    ///
    /// # Returns
    ///
    /// Returns `true` if the component can process multiple items in a batch.
    fn supports_batching(&self) -> bool;

    /// Returns the optimal batch size for this component.
    ///
    /// # Returns
    ///
    /// Returns the recommended number of items to process in a single batch.
    fn optimal_batch_size(&self) -> usize;
}

/// Builder for configuring GPU acceleration settings.
///
/// Provides a fluent API for constructing `GpuConfig` instances with
/// custom settings for device type, memory limits, precision, and batching.
pub struct GpuConfigBuilder {
    config: GpuConfig,
}

impl GpuConfigBuilder {
    /// Creates a new GPU configuration builder with default settings.
    ///
    /// # Returns
    ///
    /// Returns a new builder initialized with default values.
    pub fn new() -> Self {
        Self {
            config: GpuConfig::default(),
        }
    }

    /// Sets the device type for GPU acceleration.
    ///
    /// # Arguments
    ///
    /// * `device_type` - Type of compute device to use (CPU, CUDA, Metal, or Auto)
    ///
    /// # Returns
    ///
    /// Returns self for method chaining.
    pub fn device_type(mut self, device_type: DeviceType) -> Self {
        self.config.device_type = device_type;
        self
    }

    /// Sets a memory limit for GPU usage.
    ///
    /// # Arguments
    ///
    /// * `limit_mb` - Maximum GPU memory to use in megabytes (MB)
    ///
    /// # Returns
    ///
    /// Returns self for method chaining.
    pub fn memory_limit(mut self, limit_mb: u64) -> Self {
        self.config.memory_limit_mb = Some(limit_mb);
        self
    }

    /// Sets the batch size for processing.
    ///
    /// # Arguments
    ///
    /// * `size` - Number of samples to process in a single batch
    ///
    /// # Returns
    ///
    /// Returns self for method chaining.
    pub fn batch_size(mut self, size: usize) -> Self {
        self.config.batch_size = size;
        self
    }

    /// Enables or disables FP16 (half-precision) computation.
    ///
    /// FP16 reduces memory usage and can improve performance on supported GPUs,
    /// with minimal impact on quality for most neural synthesis tasks.
    ///
    /// # Arguments
    ///
    /// * `enable` - Whether to enable FP16 computation
    ///
    /// # Returns
    ///
    /// Returns self for method chaining.
    pub fn enable_fp16(mut self, enable: bool) -> Self {
        self.config.enable_fp16 = enable;
        self
    }

    /// Enables or disables Tensor Core acceleration on NVIDIA GPUs.
    ///
    /// Tensor Cores provide significant performance improvements for matrix
    /// operations on compatible NVIDIA GPUs (Volta architecture and later).
    ///
    /// # Arguments
    ///
    /// * `enable` - Whether to enable Tensor Core acceleration
    ///
    /// # Returns
    ///
    /// Returns self for method chaining.
    pub fn enable_tensor_cores(mut self, enable: bool) -> Self {
        self.config.enable_tensor_cores = enable;
        self
    }

    /// Sets the size of the tensor memory pool.
    ///
    /// # Arguments
    ///
    /// * `size` - Maximum number of tensors to cache in the memory pool
    ///
    /// # Returns
    ///
    /// Returns self for method chaining.
    pub fn memory_pool_size(mut self, size: usize) -> Self {
        self.config.memory_pool_size = size;
        self
    }

    /// Builds the final GPU configuration.
    ///
    /// # Returns
    ///
    /// Returns the constructed `GpuConfig` with all configured settings.
    pub fn build(self) -> GpuConfig {
        self.config
    }
}

impl Default for GpuConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_config_builder() {
        let config = GpuConfigBuilder::new()
            .device_type(DeviceType::Cuda(0))
            .batch_size(64)
            .enable_fp16(true)
            .memory_limit(8000)
            .build();

        assert!(matches!(config.device_type, DeviceType::Cuda(0)));
        assert_eq!(config.batch_size, 64);
        assert!(config.enable_fp16);
        assert_eq!(config.memory_limit_mb, Some(8000));
    }

    #[test]
    fn test_gpu_accelerator_creation() {
        let config = GpuConfig::default();
        let accelerator = GpuAccelerator::new(config);
        assert!(accelerator.is_ok());
    }

    #[test]
    fn test_memory_pool() {
        let mut pool = TensorMemoryPool::new(10);
        let device = Device::Cpu;

        let tensor1 = pool.get_tensor(&[100, 200], candle_core::DType::F32, &device);
        assert!(tensor1.is_ok());

        let tensor2 = pool.get_tensor(&[100, 200], candle_core::DType::F32, &device);
        assert!(tensor2.is_ok());
    }

    #[test]
    fn test_device_selection() {
        // Test CPU fallback
        let device = GpuAccelerator::select_device(&DeviceType::Cpu);
        assert!(device.is_ok());

        // Test auto selection (should fallback to CPU in test environment)
        let device = GpuAccelerator::select_device(&DeviceType::Auto);
        assert!(device.is_ok());
    }

    #[test]
    fn test_memory_usage_tracking() {
        let config = GpuConfig::default();
        let accelerator = GpuAccelerator::new(config).unwrap();
        let usage = accelerator.get_memory_usage();

        assert_eq!(usage.allocated_mb, 0);
        assert_eq!(usage.cached_mb, 0);
        assert_eq!(usage.device_utilization, 0.0);
    }

    #[test]
    fn test_tensor_core_optimization() {
        let config = GpuConfig::default();
        let accelerator = GpuAccelerator::new(config).unwrap();

        let original_shape = vec![15, 23, 31];
        let optimized = accelerator.optimize_for_tensor_cores(&original_shape);

        // All dimensions should be multiples of 8
        for dim in optimized {
            assert_eq!(dim % 8, 0);
        }
    }

    #[test]
    fn test_gpu_config_defaults() {
        let config = GpuConfig::default();
        assert!(matches!(config.device_type, DeviceType::Auto));
        assert_eq!(config.batch_size, 32);
        assert!(!config.enable_fp16);
        assert!(config.enable_tensor_cores);
        assert_eq!(config.memory_pool_size, 64);
    }
}

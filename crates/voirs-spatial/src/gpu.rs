//! GPU Acceleration for Spatial Audio Processing
//!
//! This module provides GPU-accelerated implementations of computationally intensive
//! spatial audio operations using CUDA or other GPU compute platforms.

use crate::{Error, Position3D, Result};
use candle_core::{DType, Device, Tensor};
use scirs2_core::ndarray::{Array1, Array2, Array3};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// GPU device configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuConfig {
    /// Prefer GPU over CPU when available
    pub prefer_gpu: bool,
    /// Device ID to use (for multi-GPU systems)
    pub device_id: usize,
    /// Memory limit in bytes (0 = no limit)
    pub memory_limit: usize,
    /// Batch size for parallel processing
    pub batch_size: usize,
    /// Enable mixed precision computation
    pub mixed_precision: bool,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            prefer_gpu: true,
            device_id: 0,
            memory_limit: 0, // No limit
            batch_size: 32,
            mixed_precision: true,
        }
    }
}

/// GPU device wrapper with automatic fallback
pub struct GpuDevice {
    device: Device,
    config: GpuConfig,
    is_gpu: bool,
}

impl GpuDevice {
    /// Create a new GPU device with automatic selection
    pub fn new(config: GpuConfig) -> Result<Self> {
        let device = if config.prefer_gpu {
            match Device::cuda_if_available(config.device_id) {
                Ok(device) => device,
                Err(_) => {
                    tracing::warn!("GPU not available, falling back to CPU");
                    Device::Cpu
                }
            }
        } else {
            Device::Cpu
        };

        let is_gpu = matches!(device, Device::Cuda(_));

        if is_gpu {
            tracing::info!(
                "Using GPU device {} for spatial audio processing",
                config.device_id
            );
        } else {
            tracing::info!("Using CPU for spatial audio processing");
        }

        Ok(Self {
            device,
            config,
            is_gpu,
        })
    }

    /// Get the underlying device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Check if using GPU
    pub fn is_gpu(&self) -> bool {
        self.is_gpu
    }

    /// Get device configuration
    pub fn config(&self) -> &GpuConfig {
        &self.config
    }
}

/// GPU-accelerated convolution processor for HRTF and reverb
pub struct GpuConvolution {
    device: Arc<GpuDevice>,
    fft_size: usize,
    hop_size: usize,
    // Pre-allocated buffers for efficient processing
    input_buffer: Option<Tensor>,
    output_buffer: Option<Tensor>,
    frequency_domain_buffer: Option<Tensor>,
}

impl GpuConvolution {
    /// Create new GPU convolution processor
    pub fn new(device: Arc<GpuDevice>, fft_size: usize, hop_size: usize) -> Result<Self> {
        Ok(Self {
            device,
            fft_size,
            hop_size,
            input_buffer: None,
            output_buffer: None,
            frequency_domain_buffer: None,
        })
    }

    /// Process convolution with impulse response on GPU
    pub fn convolve(
        &mut self,
        input: &Array1<f32>,
        impulse_response: &Array1<f32>,
    ) -> Result<Array1<f32>> {
        let device = self.device.device();

        // Convert input to tensor
        let input_tensor = Tensor::from_slice(input.as_slice().unwrap(), input.len(), device)
            .map_err(|e| Error::LegacyProcessing(format!("Failed to create input tensor: {e}")))?;

        let ir_tensor = Tensor::from_slice(
            impulse_response.as_slice().unwrap(),
            impulse_response.len(),
            device,
        )
        .map_err(|e| Error::LegacyProcessing(format!("Failed to create IR tensor: {e}")))?;

        // Perform FFT-based convolution
        let result = self.fft_convolve(&input_tensor, &ir_tensor)?;

        // Convert back to ndarray
        let result_vec: Vec<f32> = result.to_vec1().map_err(|e| {
            Error::LegacyProcessing(format!("Failed to convert result tensor: {e}"))
        })?;

        Ok(Array1::from_vec(result_vec))
    }

    /// FFT-based convolution implementation
    #[allow(clippy::single_range_in_vec_init)]
    fn fft_convolve(&self, input: &Tensor, impulse_response: &Tensor) -> Result<Tensor> {
        // For now, implement basic convolution
        // In practice, would use proper FFT operations
        let device = self.device.device();

        let input_len = input
            .dims1()
            .map_err(|e| Error::LegacyProcessing(format!("Invalid input dimensions: {e}")))?;
        let ir_len = impulse_response
            .dims1()
            .map_err(|e| Error::LegacyProcessing(format!("Invalid IR dimensions: {e}")))?;

        let output_len = input_len + ir_len - 1;

        // Create output tensor
        let zeros = Tensor::zeros((output_len,), DType::F32, device)
            .map_err(|e| Error::LegacyProcessing(format!("Failed to create output tensor: {e}")))?;

        // Simple direct convolution for now (would use FFT for efficiency)
        let mut result = zeros;
        for i in 0..input_len {
            for j in 0..ir_len {
                let idx = i + j;
                if idx < output_len {
                    let input_val = input
                        .get(i)
                        .map_err(|e| Error::LegacyProcessing(format!("Input access error: {e}")))?;
                    let ir_val = impulse_response
                        .get(j)
                        .map_err(|e| Error::LegacyProcessing(format!("IR access error: {e}")))?;
                    let current = result.get(idx).map_err(|e| {
                        Error::LegacyProcessing(format!("Result access error: {e}"))
                    })?;
                    let new_val = (current + (input_val * ir_val))?;
                    result = result.slice_assign(&[idx..idx + 1], &new_val)?;
                }
            }
        }

        Ok(result)
    }

    /// Batch process multiple convolutions
    pub fn convolve_batch(
        &mut self,
        inputs: &Array2<f32>,
        impulse_responses: &Array2<f32>,
    ) -> Result<Array2<f32>> {
        let batch_size = inputs.shape()[0];
        let input_len = inputs.shape()[1];
        let ir_len = impulse_responses.shape()[1];
        let output_len = input_len + ir_len - 1;

        let mut results = Array2::zeros((batch_size, output_len));

        // Process in batches for GPU efficiency
        for i in 0..batch_size {
            let input = inputs.row(i).to_owned();
            let ir = impulse_responses.row(i).to_owned();
            let result = self.convolve(&input, &ir)?;
            results.row_mut(i).assign(&result);
        }

        Ok(results)
    }
}

/// GPU-accelerated distance calculations for spatial audio
pub struct GpuSpatialMath {
    device: Arc<GpuDevice>,
}

impl GpuSpatialMath {
    /// Create new GPU spatial math processor
    pub fn new(device: Arc<GpuDevice>) -> Self {
        Self { device }
    }

    /// Calculate distances between listener and multiple sources
    pub fn calculate_distances(
        &self,
        listener_pos: &Position3D,
        source_positions: &[Position3D],
    ) -> Result<Array1<f32>> {
        let device = self.device.device();
        let num_sources = source_positions.len();

        // Convert positions to tensors
        let listener_tensor = Tensor::from_slice(
            &[listener_pos.x, listener_pos.y, listener_pos.z],
            (3,),
            device,
        )
        .map_err(|e| Error::LegacyProcessing(format!("Failed to create listener tensor: {e}")))?;

        let source_data: Vec<f32> = source_positions
            .iter()
            .flat_map(|pos| vec![pos.x, pos.y, pos.z])
            .collect();

        let source_tensor = Tensor::from_slice(&source_data, (num_sources, 3), device)
            .map_err(|e| Error::LegacyProcessing(format!("Failed to create source tensor: {e}")))?;

        // Calculate differences
        let listener_expanded = listener_tensor.unsqueeze(0)?.expand((num_sources, 3))?;

        let differences = (&source_tensor - &listener_expanded)?;

        // Calculate squared distances
        let squared_diffs = differences.sqr()?;
        let distances_squared = squared_diffs.sum(1)?;

        // Take square root
        let distances = distances_squared.sqrt()?;

        // Convert back to ndarray
        let result_vec: Vec<f32> = distances
            .to_vec1()
            .map_err(|e| Error::LegacyProcessing(format!("Failed to convert distances: {e}")))?;

        Ok(Array1::from_vec(result_vec))
    }

    /// Calculate batch of dot products
    pub fn batch_dot_product(
        &self,
        vectors_a: &Array2<f32>,
        vectors_b: &Array2<f32>,
    ) -> Result<Array1<f32>> {
        let device = self.device.device();

        if vectors_a.shape() != vectors_b.shape() {
            return Err(Error::LegacyProcessing(
                "Vector arrays must have same shape".to_string(),
            ));
        }

        let batch_size = vectors_a.shape()[0];
        let vector_len = vectors_a.shape()[1];

        // Convert to tensors
        let tensor_a = Tensor::from_slice(
            vectors_a.as_slice().unwrap(),
            (batch_size, vector_len),
            device,
        )
        .map_err(|e| Error::LegacyProcessing(format!("Failed to create tensor A: {e}")))?;

        let tensor_b = Tensor::from_slice(
            vectors_b.as_slice().unwrap(),
            (batch_size, vector_len),
            device,
        )
        .map_err(|e| Error::LegacyProcessing(format!("Failed to create tensor B: {e}")))?;

        // Element-wise multiplication and sum along vector dimension
        let products = (&tensor_a * &tensor_b)?;
        let dot_products = products.sum(1)?;

        // Convert result
        let result_vec: Vec<f32> = dot_products
            .to_vec1()
            .map_err(|e| Error::LegacyProcessing(format!("Failed to convert dot products: {e}")))?;

        Ok(Array1::from_vec(result_vec))
    }

    /// Normalize batch of vectors
    pub fn normalize_batch(&self, vectors: &Array2<f32>) -> Result<Array2<f32>> {
        let device = self.device.device();
        let batch_size = vectors.shape()[0];
        let vector_len = vectors.shape()[1];

        // Convert to tensor
        let tensor = Tensor::from_slice(
            vectors.as_slice().unwrap(),
            (batch_size, vector_len),
            device,
        )
        .map_err(|e| Error::LegacyProcessing(format!("Failed to create tensor: {e}")))?;

        // Calculate magnitudes
        let squared = tensor.sqr()?;
        let magnitudes_squared = squared.sum_keepdim(1)?;
        let magnitudes = magnitudes_squared.sqrt()?;

        // Avoid division by zero
        let epsilon = Tensor::from_slice(&[1e-8f32], (1,), device)?.expand((batch_size, 1))?;
        let safe_magnitudes = magnitudes.maximum(&epsilon)?;

        // Normalize
        let normalized = tensor.broadcast_div(&safe_magnitudes)?;

        // Convert back
        let result_vec: Vec<f32> = normalized
            .to_vec2()
            .map_err(|e| {
                Error::LegacyProcessing(format!("Failed to convert normalized vectors: {e}"))
            })?
            .into_iter()
            .flatten()
            .collect();

        let result = Array2::from_shape_vec((batch_size, vector_len), result_vec)
            .map_err(|e| Error::LegacyProcessing(format!("Failed to reshape result: {e}")))?;

        Ok(result)
    }
}

/// GPU-accelerated ambisonics processor
pub struct GpuAmbisonics {
    device: Arc<GpuDevice>,
    order: u32,
    encoding_matrices: Option<Tensor>,
    decoding_matrices: Option<Tensor>,
}

impl GpuAmbisonics {
    /// Create new GPU ambisonics processor
    pub fn new(device: Arc<GpuDevice>, order: u32) -> Result<Self> {
        Ok(Self {
            device,
            order,
            encoding_matrices: None,
            decoding_matrices: None,
        })
    }

    /// Pre-compute encoding matrices for common source positions
    pub fn precompute_encoding_matrices(&mut self, source_positions: &[Position3D]) -> Result<()> {
        let device = self.device.device();
        let num_sources = source_positions.len();
        let num_channels = ((self.order + 1) * (self.order + 1)) as usize;

        // Calculate spherical harmonics for all positions
        let mut encoding_data = Vec::with_capacity(num_sources * num_channels);

        for position in source_positions {
            // Convert to spherical coordinates
            let distance =
                (position.x * position.x + position.y * position.y + position.z * position.z)
                    .sqrt();
            let azimuth = position.y.atan2(position.x);
            let elevation = (position.z / distance.max(1e-8)).asin();

            // Calculate spherical harmonics (simplified)
            for l in 0..=self.order {
                for m in -(l as i32)..=(l as i32) {
                    let coeff = self.spherical_harmonic(l, m, azimuth, elevation);
                    encoding_data.push(coeff);
                }
            }
        }

        self.encoding_matrices = Some(
            Tensor::from_slice(&encoding_data, (num_sources, num_channels), device).map_err(
                |e| Error::LegacyProcessing(format!("Failed to create encoding matrices: {e}")),
            )?,
        );

        Ok(())
    }

    /// Simplified spherical harmonic calculation
    fn spherical_harmonic(&self, l: u32, m: i32, azimuth: f32, elevation: f32) -> f32 {
        // Simplified implementation - in practice would use proper spherical harmonics
        let cos_el = elevation.cos();
        let sin_el = elevation.sin();

        match (l, m) {
            (0, 0) => 1.0,
            (1, -1) => sin_el * azimuth.sin(),
            (1, 0) => cos_el,
            (1, 1) => sin_el * azimuth.cos(),
            _ => 0.5, // Placeholder for higher orders
        }
    }

    /// Encode multiple sources to ambisonics using pre-computed matrices
    pub fn encode_batch(&self, audio_samples: &Array2<f32>) -> Result<Array2<f32>> {
        let encoding_matrices = self
            .encoding_matrices
            .as_ref()
            .ok_or_else(|| Error::LegacyProcessing("Encoding matrices not computed".to_string()))?;

        let device = self.device.device();
        let num_sources = audio_samples.shape()[0];
        let num_samples = audio_samples.shape()[1];
        let num_channels = ((self.order + 1) * (self.order + 1)) as usize;

        // Convert audio to tensor
        let audio_tensor = Tensor::from_slice(
            audio_samples.as_slice().unwrap(),
            (num_sources, num_samples),
            device,
        )
        .map_err(|e| Error::LegacyProcessing(format!("Failed to create audio tensor: {e}")))?;

        // Matrix multiplication: [num_channels, num_sources] × [num_sources, num_samples]
        let encoding_transposed = encoding_matrices.transpose(0, 1)?;
        let encoded = encoding_transposed.matmul(&audio_tensor)?;

        // Convert back to ndarray
        let result_vec: Vec<f32> = encoded
            .to_vec2()
            .map_err(|e| Error::LegacyProcessing(format!("Failed to convert encoded audio: {e}")))?
            .into_iter()
            .flatten()
            .collect();

        let result =
            Array2::from_shape_vec((num_channels, num_samples), result_vec).map_err(|e| {
                Error::LegacyProcessing(format!("Failed to reshape encoded audio: {e}"))
            })?;

        Ok(result)
    }
}

/// GPU resource manager for spatial audio
pub struct GpuResourceManager {
    devices: Vec<Arc<GpuDevice>>,
    current_device: usize,
    memory_usage: Vec<usize>,
}

impl GpuResourceManager {
    /// Create new GPU resource manager
    pub fn new(configs: Vec<GpuConfig>) -> Result<Self> {
        let mut devices = Vec::new();
        let mut memory_usage = Vec::new();

        for config in configs {
            let device = Arc::new(GpuDevice::new(config)?);
            devices.push(device);
            memory_usage.push(0);
        }

        if devices.is_empty() {
            // Create default CPU device
            devices.push(Arc::new(GpuDevice::new(GpuConfig {
                prefer_gpu: false,
                ..Default::default()
            })?));
            memory_usage.push(0);
        }

        Ok(Self {
            devices,
            current_device: 0,
            memory_usage,
        })
    }

    /// Get optimal device for processing
    pub fn get_optimal_device(&mut self) -> Arc<GpuDevice> {
        // Simple round-robin for now
        // In practice, would consider memory usage and load
        let device = self.devices[self.current_device].clone();
        self.current_device = (self.current_device + 1) % self.devices.len();
        device
    }

    /// Get all available devices
    pub fn get_all_devices(&self) -> &[Arc<GpuDevice>] {
        &self.devices
    }

    /// Get device count
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// Get memory usage for device
    pub fn get_memory_usage(&self, device_id: usize) -> Option<usize> {
        self.memory_usage.get(device_id).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_config() {
        let config = GpuConfig::default();
        assert_eq!(config.prefer_gpu, true);
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.mixed_precision, true);
    }

    #[test]
    fn test_gpu_device_creation() {
        let config = GpuConfig {
            prefer_gpu: false, // Force CPU for testing
            ..Default::default()
        };
        let device = GpuDevice::new(config).unwrap();
        assert_eq!(device.is_gpu(), false);
    }

    #[test]
    fn test_gpu_spatial_math() {
        let config = GpuConfig {
            prefer_gpu: false,
            ..Default::default()
        };
        let device = Arc::new(GpuDevice::new(config).unwrap());
        let math = GpuSpatialMath::new(device);

        let listener = Position3D::new(0.0, 0.0, 0.0);
        let sources = vec![
            Position3D::new(1.0, 0.0, 0.0),
            Position3D::new(0.0, 1.0, 0.0),
            Position3D::new(0.0, 0.0, 1.0),
        ];

        let distances = math.calculate_distances(&listener, &sources).unwrap();
        assert_eq!(distances.len(), 3);

        // All distances should be approximately 1.0
        for distance in distances.iter() {
            assert!((distance - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_batch_dot_product() {
        let config = GpuConfig {
            prefer_gpu: false,
            ..Default::default()
        };
        let device = Arc::new(GpuDevice::new(config).unwrap());
        let math = GpuSpatialMath::new(device);

        let vectors_a = Array2::from_shape_vec(
            (2, 3),
            vec![
                1.0, 0.0, 0.0, // First vector
                0.0, 1.0, 0.0, // Second vector
            ],
        )
        .unwrap();

        let vectors_b = Array2::from_shape_vec(
            (2, 3),
            vec![
                1.0, 0.0, 0.0, // First vector
                0.0, 1.0, 0.0, // Second vector
            ],
        )
        .unwrap();

        let dot_products = math.batch_dot_product(&vectors_a, &vectors_b).unwrap();
        assert_eq!(dot_products.len(), 2);

        // Both dot products should be 1.0
        for &dot_product in dot_products.iter() {
            assert!((dot_product - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_normalize_batch() {
        let config = GpuConfig {
            prefer_gpu: false,
            ..Default::default()
        };
        let device = Arc::new(GpuDevice::new(config).unwrap());
        let math = GpuSpatialMath::new(device);

        let vectors = Array2::from_shape_vec(
            (2, 3),
            vec![
                2.0, 0.0, 0.0, // First vector
                0.0, 3.0, 0.0, // Second vector
            ],
        )
        .unwrap();

        let normalized = math.normalize_batch(&vectors).unwrap();
        assert_eq!(normalized.shape(), [2, 3]);

        // Check that vectors are normalized
        let first_magnitude =
            (normalized[[0, 0]].powi(2) + normalized[[0, 1]].powi(2) + normalized[[0, 2]].powi(2))
                .sqrt();
        let second_magnitude =
            (normalized[[1, 0]].powi(2) + normalized[[1, 1]].powi(2) + normalized[[1, 2]].powi(2))
                .sqrt();

        assert!((first_magnitude - 1.0).abs() < 1e-6);
        assert!((second_magnitude - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_gpu_convolution_creation() {
        let config = GpuConfig {
            prefer_gpu: false,
            ..Default::default()
        };
        let device = Arc::new(GpuDevice::new(config).unwrap());
        let convolution = GpuConvolution::new(device, 1024, 256).unwrap();
        assert_eq!(convolution.fft_size, 1024);
        assert_eq!(convolution.hop_size, 256);
    }

    #[test]
    fn test_gpu_resource_manager() {
        let configs = vec![GpuConfig {
            prefer_gpu: false,
            ..Default::default()
        }];

        let mut manager = GpuResourceManager::new(configs).unwrap();
        assert_eq!(manager.device_count(), 1);

        let device = manager.get_optimal_device();
        assert!(!device.is_gpu());
    }

    #[test]
    fn test_gpu_ambisonics_creation() {
        let config = GpuConfig {
            prefer_gpu: false,
            ..Default::default()
        };
        let device = Arc::new(GpuDevice::new(config).unwrap());
        let ambisonics = GpuAmbisonics::new(device, 1).unwrap();
        assert_eq!(ambisonics.order, 1);
    }

    #[test]
    fn test_spherical_harmonic_calculation() {
        let config = GpuConfig {
            prefer_gpu: false,
            ..Default::default()
        };
        let device = Arc::new(GpuDevice::new(config).unwrap());
        let ambisonics = GpuAmbisonics::new(device, 1).unwrap();

        // Test basic spherical harmonics
        let coeff = ambisonics.spherical_harmonic(0, 0, 0.0, 0.0);
        assert_eq!(coeff, 1.0);

        let coeff = ambisonics.spherical_harmonic(1, 0, 0.0, 0.0);
        assert_eq!(coeff, 1.0); // cos(0) = 1
    }
}

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
            match std::panic::catch_unwind(|| Device::cuda_if_available(config.device_id)) {
                Ok(Ok(device)) => device,
                _ => {
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

        // Convert input to tensor - handle non-contiguous arrays properly
        let input_slice = input.as_slice().ok_or_else(|| {
            Error::LegacyProcessing(
                "Input array is not contiguous in memory, cannot create tensor efficiently"
                    .to_string(),
            )
        })?;
        let input_tensor = Tensor::from_slice(input_slice, input.len(), device)
            .map_err(|e| Error::LegacyProcessing(format!("Failed to create input tensor: {e}")))?;

        let ir_slice = impulse_response.as_slice().ok_or_else(|| {
            Error::LegacyProcessing(
                "Impulse response array is not contiguous in memory, cannot create tensor efficiently"
                    .to_string(),
            )
        })?;
        let ir_tensor = Tensor::from_slice(ir_slice, impulse_response.len(), device)
            .map_err(|e| Error::LegacyProcessing(format!("Failed to create IR tensor: {e}")))?;

        // Perform FFT-based convolution
        let result = self.fft_convolve(&input_tensor, &ir_tensor)?;

        // Convert back to ndarray
        let result_vec: Vec<f32> = result.to_vec1().map_err(|e| {
            Error::LegacyProcessing(format!("Failed to convert result tensor: {e}"))
        })?;

        Ok(Array1::from_vec(result_vec))
    }

    /// FFT-based linear convolution.
    ///
    /// Computes the full linear convolution of `input` (length `N`) and
    /// `impulse_response` (length `M`) — a tensor of length `N + M − 1` — via the
    /// real FFT. Both operands are zero-padded to
    /// `fft_len = next_power_of_two(N + M − 1)`, transformed with `rfft`, multiplied
    /// bin-by-bin (the convolution theorem), and brought back to the time domain
    /// with `irfft`. This replaces the previous O(N·M) time-domain loop with an
    /// O(L·log L) algorithm while keeping identical input/output semantics.
    fn fft_convolve(&self, input: &Tensor, impulse_response: &Tensor) -> Result<Tensor> {
        let device = self.device.device();

        let input_len = input
            .dims1()
            .map_err(|e| Error::LegacyProcessing(format!("Invalid input dimensions: {e}")))?;
        let ir_len = impulse_response
            .dims1()
            .map_err(|e| Error::LegacyProcessing(format!("Invalid IR dimensions: {e}")))?;

        let output_len = (input_len + ir_len).saturating_sub(1);

        // An empty operand convolves to an all-zero (or empty) result.
        if input_len == 0 || ir_len == 0 {
            return Tensor::zeros((output_len,), DType::F32, device).map_err(|e| {
                Error::LegacyProcessing(format!("Failed to create output tensor: {e}"))
            });
        }

        // Round the transform size up to a power of two so the circular convolution
        // of the zero-padded operands equals their linear convolution.
        let fft_len = output_len.next_power_of_two();

        // Pull the samples out of the tensors and promote to f64 for the transform.
        let input_samples: Vec<f32> = input
            .to_vec1()
            .map_err(|e| Error::LegacyProcessing(format!("Failed to read input tensor: {e}")))?;
        let ir_samples: Vec<f32> = impulse_response
            .to_vec1()
            .map_err(|e| Error::LegacyProcessing(format!("Failed to read IR tensor: {e}")))?;

        let input_f64: Vec<f64> = input_samples.iter().map(|&x| x as f64).collect();
        let ir_f64: Vec<f64> = ir_samples.iter().map(|&x| x as f64).collect();

        // Forward transforms (each yields `fft_len / 2 + 1` complex bins).
        let input_spectrum = scirs2_fft::rfft(&input_f64, Some(fft_len))
            .map_err(|e| Error::LegacyProcessing(format!("FFT error: {e}")))?;
        let ir_spectrum = scirs2_fft::rfft(&ir_f64, Some(fft_len))
            .map_err(|e| Error::LegacyProcessing(format!("FFT error: {e}")))?;

        // Per-bin spectral product (convolution theorem).
        let product: Vec<_> = input_spectrum
            .iter()
            .zip(ir_spectrum.iter())
            .map(|(a, b)| a * b)
            .collect();

        // Inverse transform; `irfft` already applies the 1/N normalisation.
        let time = scirs2_fft::irfft(&product, Some(fft_len))
            .map_err(|e| Error::LegacyProcessing(format!("IFFT error: {e}")))?;

        let result: Vec<f32> = time
            .into_iter()
            .take(output_len)
            .map(|x| x as f32)
            .collect();

        Tensor::from_slice(&result, (output_len,), device)
            .map_err(|e| Error::LegacyProcessing(format!("Failed to create result tensor: {e}")))
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

        // Convert to tensors - handle non-contiguous arrays properly
        let slice_a = vectors_a.as_slice().ok_or_else(|| {
            Error::LegacyProcessing(
                "Vector array A is not contiguous in memory, cannot create tensor efficiently"
                    .to_string(),
            )
        })?;
        let tensor_a = Tensor::from_slice(slice_a, (batch_size, vector_len), device)
            .map_err(|e| Error::LegacyProcessing(format!("Failed to create tensor A: {e}")))?;

        let slice_b = vectors_b.as_slice().ok_or_else(|| {
            Error::LegacyProcessing(
                "Vector array B is not contiguous in memory, cannot create tensor efficiently"
                    .to_string(),
            )
        })?;
        let tensor_b = Tensor::from_slice(slice_b, (batch_size, vector_len), device)
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

        // Convert to tensor - handle non-contiguous arrays properly
        let slice = vectors.as_slice().ok_or_else(|| {
            Error::LegacyProcessing(
                "Vector array is not contiguous in memory, cannot create tensor efficiently"
                    .to_string(),
            )
        })?;
        let tensor = Tensor::from_slice(slice, (batch_size, vector_len), device)
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

    /// Real-valued spherical harmonic `Y_l^m(θ, φ)`.
    ///
    /// Evaluated with the standard associated Legendre recurrence and
    /// **Schmidt semi-normalisation (SN3D)** — the normalisation family used by this
    /// crate's `ambisonics` module — so the order-0/1 harmonics are exactly
    /// `Y_0^0 = 1`, `Y_1^{-1} = sinθ·sinφ`, `Y_1^0 = cosθ`, `Y_1^1 = sinθ·cosφ`
    /// (these reproduce the previous hard-coded low-order cases).
    ///
    /// Conventions:
    /// - The `elevation` argument plays the role of the polar/inclination angle `θ`,
    ///   so the Legendre polynomial is evaluated at `x = cos θ` and `Y_1^0 ∝ cos θ`.
    /// - `azimuth` is the azimuth `φ`; the azimuthal factor is `cos(mφ)` for `m ≥ 0`
    ///   and `sin(|m|φ)` for `m < 0`.
    /// - The Condon–Shortley phase is excluded, matching the ambisonics literature.
    /// - Schmidt semi-normalisation factor:
    ///   `N_l^m = sqrt((2 − δ_{m0}) · (l − |m|)! / (l + |m|)!)`.
    ///
    /// Arbitrary `(l, m)` with `|m| ≤ l` are supported; `|m| > l` returns `0`.
    fn spherical_harmonic(&self, l: u32, m: i32, azimuth: f32, elevation: f32) -> f32 {
        let abs_m = m.unsigned_abs();
        if abs_m > l {
            return 0.0;
        }

        // x = cos θ feeds the associated Legendre polynomial.
        let x = elevation.cos();
        let legendre = Self::associated_legendre(l, abs_m, x);
        let normalization = Self::schmidt_seminormalization(l, abs_m);
        let azimuthal = if m >= 0 {
            (abs_m as f32 * azimuth).cos()
        } else {
            (abs_m as f32 * azimuth).sin()
        };

        normalization * legendre * azimuthal
    }

    /// Associated Legendre polynomial `P_l^m(x)` for `0 ≤ m ≤ l`, computed with the
    /// standard upward recurrence and *without* the Condon–Shortley phase.
    ///
    /// Seeds `P_m^m(x) = (2m−1)!! · (1 − x²)^{m/2}`, then
    /// `P_{m+1}^m(x) = x·(2m+1)·P_m^m(x)`, and recurs in `l` via
    /// `(l − m)·P_l^m = (2l − 1)·x·P_{l−1}^m − (l + m − 1)·P_{l−2}^m`.
    fn associated_legendre(l: u32, m: u32, x: f32) -> f32 {
        if m > l {
            return 0.0;
        }

        // P_m^m(x) = (2m-1)!! * (1 - x^2)^(m/2)  (no Condon–Shortley phase).
        let mut p_mm = 1.0f32;
        if m > 0 {
            let sin_theta = (1.0 - x * x).max(0.0).sqrt();
            let mut double_factorial = 1.0f32;
            for _ in 0..m {
                p_mm *= double_factorial * sin_theta;
                double_factorial += 2.0;
            }
        }
        if l == m {
            return p_mm;
        }

        // P_{m+1}^m(x) = x * (2m + 1) * P_m^m(x).
        let mut p_prev = p_mm;
        let mut p_curr = x * (2.0 * m as f32 + 1.0) * p_mm;
        if l == m + 1 {
            return p_curr;
        }

        // Upward recurrence in l.
        for n in (m + 2)..=l {
            let n_f = n as f32;
            let m_f = m as f32;
            let p_next =
                ((2.0 * n_f - 1.0) * x * p_curr - (n_f + m_f - 1.0) * p_prev) / (n_f - m_f);
            p_prev = p_curr;
            p_curr = p_next;
        }
        p_curr
    }

    /// Schmidt semi-normalisation factor
    /// `N_l^m = sqrt((2 − δ_{m0}) · (l − m)! / (l + m)!)` for `0 ≤ m ≤ l`.
    ///
    /// The factorial ratio is accumulated as `∏_{k=l−m+1}^{l+m} 1/k` to avoid forming
    /// large intermediate factorials.
    fn schmidt_seminormalization(l: u32, m: u32) -> f32 {
        let delta = if m == 0 { 1.0f64 } else { 0.0 };
        let mut ratio = 1.0f64; // (l - m)! / (l + m)!
        for k in (l - m + 1)..=(l + m) {
            ratio /= k as f64;
        }
        (((2.0 - delta) * ratio).sqrt()) as f32
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

        // Convert audio to tensor - handle non-contiguous arrays properly
        let audio_slice = audio_samples.as_slice().ok_or_else(|| {
            Error::LegacyProcessing(
                "Audio samples array is not contiguous in memory, cannot create tensor efficiently"
                    .to_string(),
            )
        })?;
        let audio_tensor = Tensor::from_slice(audio_slice, (num_sources, num_samples), device)
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
        assert!(config.prefer_gpu);
        assert_eq!(config.batch_size, 32);
        assert!(config.mixed_precision);
    }

    #[test]
    fn test_gpu_device_creation() {
        let config = GpuConfig {
            prefer_gpu: false, // Force CPU for testing
            ..Default::default()
        };
        let device = GpuDevice::new(config).expect("Should successfully create GPU device");
        assert!(!device.is_gpu());
    }

    #[test]
    fn test_gpu_spatial_math() {
        let config = GpuConfig {
            prefer_gpu: false,
            ..Default::default()
        };
        let device =
            Arc::new(GpuDevice::new(config).expect("Should successfully create GPU device"));
        let math = GpuSpatialMath::new(device);

        let listener = Position3D::new(0.0, 0.0, 0.0);
        let sources = vec![
            Position3D::new(1.0, 0.0, 0.0),
            Position3D::new(0.0, 1.0, 0.0),
            Position3D::new(0.0, 0.0, 1.0),
        ];

        let distances = math
            .calculate_distances(&listener, &sources)
            .expect("Should successfully calculate distances");
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
        let device =
            Arc::new(GpuDevice::new(config).expect("Should successfully create GPU device"));
        let math = GpuSpatialMath::new(device);

        let vectors_a = Array2::from_shape_vec(
            (2, 3),
            vec![
                1.0, 0.0, 0.0, // First vector
                0.0, 1.0, 0.0, // Second vector
            ],
        )
        .expect("Should successfully create Array2 from shape vec");

        let vectors_b = Array2::from_shape_vec(
            (2, 3),
            vec![
                1.0, 0.0, 0.0, // First vector
                0.0, 1.0, 0.0, // Second vector
            ],
        )
        .expect("Should successfully create Array2 from shape vec");

        let dot_products = math
            .batch_dot_product(&vectors_a, &vectors_b)
            .expect("Should successfully calculate batch dot product");
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
        let device =
            Arc::new(GpuDevice::new(config).expect("Should successfully create GPU device"));
        let math = GpuSpatialMath::new(device);

        let vectors = Array2::from_shape_vec(
            (2, 3),
            vec![
                2.0, 0.0, 0.0, // First vector
                0.0, 3.0, 0.0, // Second vector
            ],
        )
        .expect("Should successfully create Array2 from shape vec");

        let normalized = math
            .normalize_batch(&vectors)
            .expect("Should successfully normalize batch");
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
        let device =
            Arc::new(GpuDevice::new(config).expect("Should successfully create GPU device"));
        let convolution = GpuConvolution::new(device, 1024, 256)
            .expect("Should successfully create GPU convolution");
        assert_eq!(convolution.fft_size, 1024);
        assert_eq!(convolution.hop_size, 256);
    }

    #[test]
    fn test_gpu_resource_manager() {
        let configs = vec![GpuConfig {
            prefer_gpu: false,
            ..Default::default()
        }];

        let mut manager = GpuResourceManager::new(configs)
            .expect("Should successfully create GPU resource manager");
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
        let device =
            Arc::new(GpuDevice::new(config).expect("Should successfully create GPU device"));
        let ambisonics =
            GpuAmbisonics::new(device, 1).expect("Should successfully create GPU ambisonics");
        assert_eq!(ambisonics.order, 1);
    }

    #[test]
    fn test_spherical_harmonic_calculation() {
        let config = GpuConfig {
            prefer_gpu: false,
            ..Default::default()
        };
        let device =
            Arc::new(GpuDevice::new(config).expect("Should successfully create GPU device"));
        let ambisonics =
            GpuAmbisonics::new(device, 1).expect("Should successfully create GPU ambisonics");

        // Test basic spherical harmonics
        let coeff = ambisonics.spherical_harmonic(0, 0, 0.0, 0.0);
        assert_eq!(coeff, 1.0);

        let coeff = ambisonics.spherical_harmonic(1, 0, 0.0, 0.0);
        assert_eq!(coeff, 1.0); // cos(0) = 1
    }

    /// Naive O(N·M) reference convolution used to cross-check the FFT path.
    fn direct_convolution(input: &[f32], ir: &[f32]) -> Vec<f32> {
        if input.is_empty() || ir.is_empty() {
            return vec![0.0; (input.len() + ir.len()).saturating_sub(1)];
        }
        let mut output = vec![0.0f32; input.len() + ir.len() - 1];
        for (i, &x) in input.iter().enumerate() {
            for (j, &h) in ir.iter().enumerate() {
                output[i + j] += x * h;
            }
        }
        output
    }

    fn cpu_convolution() -> GpuConvolution {
        let config = GpuConfig {
            prefer_gpu: false,
            ..Default::default()
        };
        let device = Arc::new(GpuDevice::new(config).expect("create CPU device"));
        GpuConvolution::new(device, 1024, 256).expect("create convolution")
    }

    fn cpu_ambisonics() -> GpuAmbisonics {
        let config = GpuConfig {
            prefer_gpu: false,
            ..Default::default()
        };
        let device = Arc::new(GpuDevice::new(config).expect("create CPU device"));
        GpuAmbisonics::new(device, 3).expect("create ambisonics")
    }

    #[test]
    fn test_fft_convolution_unit_impulse_returns_input() {
        let mut conv = cpu_convolution();
        let input = Array1::from_vec(vec![0.5, -1.0, 2.0, 0.25, -0.75]);
        let impulse = Array1::from_vec(vec![1.0]); // unit impulse: y = x

        let result = conv
            .convolve(&input, &impulse)
            .expect("FFT convolution should succeed");

        assert_eq!(result.len(), input.len());
        for (got, want) in result.iter().zip(input.iter()) {
            assert!((got - want).abs() < 1e-5, "got {got}, want {want}");
        }
    }

    #[test]
    fn test_fft_convolution_matches_direct() {
        let mut conv = cpu_convolution();
        let input = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let ir = Array1::from_vec(vec![0.5, 0.25]);

        let result = conv
            .convolve(&input, &ir)
            .expect("FFT convolution should succeed");
        let expected = direct_convolution(
            input.as_slice().expect("contiguous"),
            ir.as_slice().expect("contiguous"),
        );

        assert_eq!(result.len(), expected.len());
        // Hand-computed reference: [0.5, 1.25, 2.0, 0.75].
        assert_eq!(expected, vec![0.5, 1.25, 2.0, 0.75]);
        for (got, want) in result.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-5, "got {got}, want {want}");
        }
    }

    #[test]
    fn test_spherical_harmonic_closed_forms() {
        let ambisonics = cpu_ambisonics();

        // Y_0^0 is a direction-independent constant (= 1 in Schmidt seminorm).
        for &(az, el) in &[(0.0, 0.0), (1.0, 0.5), (-2.0, -0.3), (3.0, 1.2)] {
            let y00 = ambisonics.spherical_harmonic(0, 0, az, el);
            assert!((y00 - 1.0).abs() < 1e-6, "Y_0^0 not constant: {y00}");
        }

        // Y_1^0 ∝ cos θ (θ is the elevation/inclination argument).
        for &el in &[0.0_f32, 0.3, 0.75, 1.4, -0.6] {
            let y10 = ambisonics.spherical_harmonic(1, 0, 0.0, el);
            assert!(
                (y10 - el.cos()).abs() < 1e-6,
                "Y_1^0 != cosθ: {y10} vs {}",
                el.cos()
            );
        }

        // Y_1^1 = sinθ·cosφ and Y_1^{-1} = sinθ·sinφ.
        let (az, el) = (0.6_f32, 0.4_f32);
        let sin_theta = el.sin();
        let y1p1 = ambisonics.spherical_harmonic(1, 1, az, el);
        let y1m1 = ambisonics.spherical_harmonic(1, -1, az, el);
        assert!((y1p1 - sin_theta * az.cos()).abs() < 1e-6);
        assert!((y1m1 - sin_theta * az.sin()).abs() < 1e-6);
    }

    #[test]
    fn test_spherical_harmonic_finite_and_bounded() {
        let ambisonics = cpu_ambisonics();
        for l in 0..=6u32 {
            for m in -(l as i32)..=(l as i32) {
                let value = ambisonics.spherical_harmonic(l, m, 0.7, 0.4);
                assert!(value.is_finite(), "Y_{l}^{m} not finite: {value}");
            }
        }
        // |m| > l is undefined and must return 0.
        assert_eq!(ambisonics.spherical_harmonic(1, 2, 0.5, 0.5), 0.0);
        assert_eq!(ambisonics.spherical_harmonic(2, -3, 0.5, 0.5), 0.0);
    }
}

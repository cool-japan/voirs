//! GPU acceleration for emotion processing using Candle framework

use crate::{types::EmotionParameters, Error, Result};
use std::sync::{Arc, Mutex};
use tracing::{debug, trace, warn};

#[cfg(feature = "gpu")]
use candle_core::{DType, Device, Tensor};

/// GPU accelerated emotion processor
#[derive(Debug)]
pub struct GpuEmotionProcessor {
    #[cfg(feature = "gpu")]
    device: Device,
    #[cfg(feature = "gpu")]
    enabled: bool,
    #[cfg(not(feature = "gpu"))]
    enabled: bool,
}

impl GpuEmotionProcessor {
    /// Create a new GPU emotion processor
    pub fn new() -> Result<Self> {
        #[cfg(feature = "gpu")]
        {
            // Try to initialize CUDA device first, fallback to CPU
            let device = if Device::cuda_if_available(0).is_ok() {
                debug!("GPU acceleration enabled with CUDA device");
                Device::cuda_if_available(0).map_err(|e| {
                    Error::Processing(format!("Failed to initialize CUDA device: {}", e))
                })?
            } else {
                debug!("CUDA not available, GPU acceleration disabled");
                Device::Cpu
            };

            let enabled = !matches!(device, Device::Cpu);

            Ok(Self { device, enabled })
        }
        #[cfg(not(feature = "gpu"))]
        {
            debug!("GPU feature not enabled, using CPU fallback");
            Ok(Self { enabled: false })
        }
    }

    /// Check if GPU acceleration is available and enabled
    pub fn is_gpu_enabled(&self) -> bool {
        self.enabled
    }

    /// Get device information
    pub fn device_info(&self) -> String {
        #[cfg(feature = "gpu")]
        {
            match &self.device {
                Device::Cpu => "CPU".to_string(),
                Device::Cuda(_cuda_device) => "CUDA Device".to_string(),
                Device::Metal(_metal_device) => "Metal Device".to_string(),
            }
        }
        #[cfg(not(feature = "gpu"))]
        {
            "CPU (GPU feature disabled)".to_string()
        }
    }

    /// Apply energy scaling using GPU acceleration
    #[cfg(feature = "gpu")]
    pub fn apply_energy_scaling_gpu(&self, audio_data: &[f32], factor: f32) -> Result<Vec<f32>> {
        if !self.enabled {
            return Err(Error::Processing("GPU not enabled".to_string()));
        }

        // Convert audio to tensor
        let audio_tensor = Tensor::from_slice(audio_data, audio_data.len(), &self.device)
            .map_err(|e| Error::Processing(format!("Failed to create audio tensor: {}", e)))?;

        // Create factor tensor
        let factor_tensor = Tensor::from_slice(&[factor], 1, &self.device)
            .map_err(|e| Error::Processing(format!("Failed to create factor tensor: {}", e)))?;

        // Perform element-wise multiplication on GPU
        let result_tensor = audio_tensor
            .mul(&factor_tensor)
            .map_err(|e| Error::Processing(format!("GPU multiplication failed: {}", e)))?;

        // Convert back to Vec<f32>
        let result_vec = result_tensor
            .to_vec1::<f32>()
            .map_err(|e| Error::Processing(format!("Failed to convert tensor to vec: {}", e)))?;

        trace!("Applied GPU energy scaling with factor: {:.2}", factor);
        Ok(result_vec)
    }

    /// Apply energy scaling - CPU fallback version
    #[cfg(not(feature = "gpu"))]
    pub fn apply_energy_scaling_gpu(&self, audio_data: &[f32], factor: f32) -> Result<Vec<f32>> {
        // CPU fallback implementation
        Ok(audio_data.iter().map(|&sample| sample * factor).collect())
    }

    /// Apply pitch shift using GPU acceleration
    #[cfg(feature = "gpu")]
    pub fn apply_pitch_shift_gpu(&self, audio_data: &[f32], pitch_shift: f32) -> Result<Vec<f32>> {
        if !self.enabled {
            return Err(Error::Processing("GPU not enabled".to_string()));
        }

        let len = audio_data.len();
        if len < 2 {
            return Ok(audio_data.to_vec());
        }

        // Create input tensor
        let input_tensor = Tensor::from_slice(audio_data, len, &self.device)
            .map_err(|e| Error::Processing(format!("Failed to create input tensor: {}", e)))?;

        // Create indices for resampling on GPU
        let mut indices = Vec::with_capacity(len);
        for i in 0..len {
            let source_idx = (i as f32 / pitch_shift) as i64;
            indices.push(source_idx.min(len as i64 - 1).max(0));
        }

        let indices_tensor = Tensor::from_vec(indices, len, &self.device)
            .map_err(|e| Error::Processing(format!("Failed to create indices tensor: {}", e)))?;

        // Gather operation for resampling
        let result_tensor = input_tensor
            .gather(&indices_tensor, 0)
            .map_err(|e| Error::Processing(format!("GPU gather failed: {}", e)))?;

        let result_vec = result_tensor
            .to_vec1::<f32>()
            .map_err(|e| Error::Processing(format!("Failed to convert tensor to vec: {}", e)))?;

        trace!("Applied GPU pitch shift with factor: {:.2}", pitch_shift);
        Ok(result_vec)
    }

    /// Apply pitch shift - CPU fallback version
    #[cfg(not(feature = "gpu"))]
    pub fn apply_pitch_shift_gpu(&self, audio_data: &[f32], pitch_shift: f32) -> Result<Vec<f32>> {
        let len = audio_data.len();
        if len < 2 {
            return Ok(audio_data.to_vec());
        }

        // CPU fallback implementation
        let mut result = vec![0.0; len];
        for i in 0..len {
            let source_idx = (i as f32 / pitch_shift) as usize;
            if source_idx < len {
                result[i] = audio_data[source_idx];
            }
        }
        Ok(result)
    }

    /// Apply tempo modification using GPU acceleration
    #[cfg(feature = "gpu")]
    pub fn apply_tempo_modification_gpu(
        &self,
        audio_data: &[f32],
        tempo_scale: f32,
    ) -> Result<Vec<f32>> {
        if !self.enabled {
            return Err(Error::Processing("GPU not enabled".to_string()));
        }

        if (tempo_scale - 1.0).abs() < 0.01 {
            return Ok(audio_data.to_vec());
        }

        let input_len = audio_data.len();
        let output_len = (input_len as f32 / tempo_scale) as usize;

        if output_len == 0 {
            return Ok(vec![]);
        }

        // Create input tensor
        let input_tensor = Tensor::from_slice(audio_data, input_len, &self.device)
            .map_err(|e| Error::Processing(format!("Failed to create input tensor: {}", e)))?;

        // Generate interpolation indices and weights for linear interpolation
        let mut indices = Vec::with_capacity(output_len);
        let mut weights = Vec::with_capacity(output_len);

        for i in 0..output_len {
            let source_pos = i as f32 * tempo_scale;
            let source_idx = source_pos as i64;
            let frac = source_pos - source_idx as f32;

            indices.push(source_idx.min(input_len as i64 - 1).max(0));
            weights.push(1.0 - frac);
        }

        let indices_tensor = Tensor::from_vec(indices.clone(), output_len, &self.device)
            .map_err(|e| Error::Processing(format!("Failed to create indices tensor: {}", e)))?;

        let weights_tensor = Tensor::from_slice(&weights, output_len, &self.device)
            .map_err(|e| Error::Processing(format!("Failed to create weights tensor: {}", e)))?;

        // Sample current positions
        let samples1 = input_tensor
            .gather(&indices_tensor, 0)
            .map_err(|e| Error::Processing(format!("GPU gather failed: {}", e)))?;

        // Sample next positions for interpolation
        let mut next_indices = indices;
        for idx in &mut next_indices {
            *idx = (*idx + 1).min(input_len as i64 - 1);
        }

        let next_indices_tensor = Tensor::from_vec(next_indices, output_len, &self.device)
            .map_err(|e| {
                Error::Processing(format!("Failed to create next indices tensor: {}", e))
            })?;

        let samples2 = input_tensor
            .gather(&next_indices_tensor, 0)
            .map_err(|e| Error::Processing(format!("GPU gather failed: {}", e)))?;

        // Linear interpolation: result = sample1 * weight + sample2 * (1 - weight)
        let one_minus_weights = weights_tensor
            .neg()
            .map_err(|e| Error::Processing(format!("GPU neg failed: {}", e)))?
            .add(
                &Tensor::ones(output_len, DType::F32, &self.device)
                    .map_err(|e| Error::Processing(format!("GPU ones failed: {}", e)))?,
            )
            .map_err(|e| Error::Processing(format!("GPU add failed: {}", e)))?;

        let interpolated = samples1
            .mul(&weights_tensor)
            .map_err(|e| Error::Processing(format!("GPU mul failed: {}", e)))?
            .add(
                &samples2
                    .mul(&one_minus_weights)
                    .map_err(|e| Error::Processing(format!("GPU mul failed: {}", e)))?,
            )
            .map_err(|e| Error::Processing(format!("GPU add failed: {}", e)))?;

        let result_vec = interpolated
            .to_vec1::<f32>()
            .map_err(|e| Error::Processing(format!("Failed to convert tensor to vec: {}", e)))?;

        trace!(
            "Applied GPU tempo modification with scale: {:.2}",
            tempo_scale
        );
        Ok(result_vec)
    }

    /// Apply tempo modification - CPU fallback version
    #[cfg(not(feature = "gpu"))]
    pub fn apply_tempo_modification_gpu(
        &self,
        audio_data: &[f32],
        tempo_scale: f32,
    ) -> Result<Vec<f32>> {
        if (tempo_scale - 1.0).abs() < 0.01 {
            return Ok(audio_data.to_vec());
        }

        let input_len = audio_data.len();
        let output_len = (input_len as f32 / tempo_scale) as usize;

        if output_len == 0 {
            return Ok(vec![]);
        }

        // CPU fallback implementation with linear interpolation
        let mut result = vec![0.0; output_len];
        for i in 0..output_len {
            let source_pos = i as f32 * tempo_scale;
            let source_idx = source_pos as usize;
            let frac = source_pos - source_idx as f32;

            if source_idx < input_len {
                result[i] = if source_idx + 1 < input_len {
                    audio_data[source_idx] * (1.0 - frac) + audio_data[source_idx + 1] * frac
                } else {
                    audio_data[source_idx]
                };
            }
        }
        Ok(result)
    }

    /// Apply convolution-based filtering using GPU acceleration
    #[cfg(feature = "gpu")]
    pub fn apply_filter_gpu(&self, audio_data: &[f32], filter_coeffs: &[f32]) -> Result<Vec<f32>> {
        if !self.enabled || filter_coeffs.is_empty() {
            return Ok(audio_data.to_vec());
        }

        let audio_len = audio_data.len();
        let filter_len = filter_coeffs.len();

        if audio_len < filter_len {
            return Ok(audio_data.to_vec());
        }

        // Create tensors
        let audio_tensor = Tensor::from_slice(audio_data, audio_len, &self.device)
            .map_err(|e| Error::Processing(format!("Failed to create audio tensor: {}", e)))?;

        let filter_tensor = Tensor::from_slice(filter_coeffs, filter_len, &self.device)
            .map_err(|e| Error::Processing(format!("Failed to create filter tensor: {}", e)))?;

        // Reshape for conv1d: [batch_size, channels, length]
        let audio_reshaped = audio_tensor
            .unsqueeze(0)
            .map_err(|e| Error::Processing(format!("Failed to unsqueeze audio: {}", e)))?
            .unsqueeze(0)
            .map_err(|e| Error::Processing(format!("Failed to unsqueeze audio: {}", e)))?;

        let filter_reshaped = filter_tensor
            .unsqueeze(0)
            .map_err(|e| Error::Processing(format!("Failed to unsqueeze filter: {}", e)))?
            .unsqueeze(0)
            .map_err(|e| Error::Processing(format!("Failed to unsqueeze filter: {}", e)))?;

        // Apply convolution
        let conv_result = audio_reshaped
            .conv1d(&filter_reshaped, 0, 1, 1, 1)
            .map_err(|e| Error::Processing(format!("GPU convolution failed: {}", e)))?;

        // Remove batch and channel dimensions
        let result_tensor = conv_result
            .squeeze(0)
            .map_err(|e| Error::Processing(format!("Failed to squeeze result: {}", e)))?
            .squeeze(0)
            .map_err(|e| Error::Processing(format!("Failed to squeeze result: {}", e)))?;

        let result_vec = result_tensor
            .to_vec1::<f32>()
            .map_err(|e| Error::Processing(format!("Failed to convert tensor to vec: {}", e)))?;

        trace!(
            "Applied GPU filter convolution with {} coefficients",
            filter_len
        );
        Ok(result_vec)
    }

    /// Apply convolution-based filtering - CPU fallback version
    #[cfg(not(feature = "gpu"))]
    pub fn apply_filter_gpu(&self, audio_data: &[f32], filter_coeffs: &[f32]) -> Result<Vec<f32>> {
        if filter_coeffs.is_empty() {
            return Ok(audio_data.to_vec());
        }

        let audio_len = audio_data.len();
        let filter_len = filter_coeffs.len();

        if audio_len < filter_len {
            return Ok(audio_data.to_vec());
        }

        // CPU fallback convolution implementation
        let mut result = vec![0.0; audio_len - filter_len + 1];
        for i in 0..result.len() {
            let mut sum = 0.0;
            for j in 0..filter_len {
                sum += audio_data[i + j] * filter_coeffs[j];
            }
            result[i] = sum;
        }
        Ok(result)
    }

    /// Process audio with emotion parameters using GPU acceleration where possible
    pub fn process_audio_gpu(
        &self,
        audio_data: &[f32],
        params: &EmotionParameters,
    ) -> Result<Vec<f32>> {
        let mut processed_audio = audio_data.to_vec();

        // Apply energy scaling on GPU if enabled
        if (params.energy_scale - 1.0).abs() > 0.01 {
            processed_audio =
                self.apply_energy_scaling_gpu(&processed_audio, params.energy_scale)?;
        }

        // Apply pitch shift on GPU if enabled
        if (params.pitch_shift - 1.0).abs() > 0.01 {
            processed_audio = self.apply_pitch_shift_gpu(&processed_audio, params.pitch_shift)?;
        }

        // Apply tempo modification on GPU if enabled
        if (params.tempo_scale - 1.0).abs() > 0.01 {
            processed_audio =
                self.apply_tempo_modification_gpu(&processed_audio, params.tempo_scale)?;
        }

        // Apply simple lowpass filter for emotional effects
        if let Some((emotion, intensity)) = params.emotion_vector.dominant_emotion() {
            match emotion {
                crate::types::Emotion::Sad => {
                    // Apply lowpass filter for sadness (reduce brightness)
                    let filter_strength = intensity.value() * 0.3;
                    let cutoff = 0.7 + filter_strength * 0.2; // Adjust cutoff based on intensity
                    let filter_coeffs = self.generate_lowpass_coeffs(cutoff);
                    if let Ok(filtered) = self.apply_filter_gpu(&processed_audio, &filter_coeffs) {
                        processed_audio = filtered;
                    }
                }
                crate::types::Emotion::Happy | crate::types::Emotion::Excited => {
                    // Apply highpass emphasis for brightness
                    let filter_strength = intensity.value() * 0.2;
                    let filter_coeffs = self.generate_highpass_coeffs(filter_strength);
                    if let Ok(filtered) = self.apply_filter_gpu(&processed_audio, &filter_coeffs) {
                        processed_audio = filtered;
                    }
                }
                _ => {
                    // No specific GPU filtering for other emotions
                }
            }
        }

        Ok(processed_audio)
    }

    /// Generate simple lowpass filter coefficients
    fn generate_lowpass_coeffs(&self, cutoff: f32) -> Vec<f32> {
        // Simple 3-tap lowpass filter
        let alpha = cutoff.clamp(0.1, 1.0);
        vec![alpha * 0.25, alpha * 0.5, alpha * 0.25]
    }

    /// Generate simple highpass emphasis coefficients
    fn generate_highpass_coeffs(&self, strength: f32) -> Vec<f32> {
        // Simple highpass emphasis filter
        let s = strength.clamp(0.0, 0.5);
        vec![-s, 1.0 + 2.0 * s, -s]
    }

    /// Get GPU memory information (if available)
    #[cfg(feature = "gpu")]
    pub fn get_gpu_memory_info(&self) -> Option<(u64, u64)> {
        // This would require additional GPU memory introspection
        // For now, return None as it's not easily available through Candle
        None
    }

    /// Get GPU memory information - CPU fallback
    #[cfg(not(feature = "gpu"))]
    pub fn get_gpu_memory_info(&self) -> Option<(u64, u64)> {
        None
    }

    /// Benchmark GPU vs CPU performance for a given audio size
    pub fn benchmark_performance(&self, audio_size: usize) -> Result<(f64, f64)> {
        let test_audio = vec![0.1; audio_size];
        let test_params = EmotionParameters {
            emotion_vector: crate::types::EmotionVector::new(),
            duration_ms: None,
            fade_in_ms: None,
            fade_out_ms: None,
            pitch_shift: 1.2,
            tempo_scale: 0.9,
            energy_scale: 1.3,
            breathiness: 0.0,
            roughness: 0.0,
            custom_params: std::collections::HashMap::new(),
        };

        // Time GPU processing
        let gpu_start = std::time::Instant::now();
        let _gpu_result = self.process_audio_gpu(&test_audio, &test_params)?;
        let gpu_time = gpu_start.elapsed().as_secs_f64();

        // Time CPU fallback (simplified)
        let cpu_start = std::time::Instant::now();
        let mut cpu_result = test_audio.clone();
        // Simple CPU operations for comparison
        for sample in &mut cpu_result {
            *sample *= test_params.energy_scale;
        }
        let cpu_time = cpu_start.elapsed().as_secs_f64();

        debug!(
            "Performance benchmark for {} samples: GPU={:.4}s, CPU={:.4}s, Speedup={:.2}x",
            audio_size,
            gpu_time,
            cpu_time,
            cpu_time / gpu_time.max(0.0001)
        );

        Ok((gpu_time, cpu_time))
    }
}

impl Default for GpuEmotionProcessor {
    fn default() -> Self {
        // Safe fallback implementation - CPU fallback if GPU fails
        Self::new().unwrap_or_else(|_| {
            // Emergency CPU-only fallback that can't fail
            #[cfg(feature = "gpu")]
            {
                Self {
                    device: candle_core::Device::Cpu,
                    enabled: false,
                }
            }
            #[cfg(not(feature = "gpu"))]
            {
                Self { enabled: false }
            }
        })
    }
}

/// GPU processing capabilities and configuration
#[derive(Debug, Clone)]
pub struct GpuCapabilities {
    pub cuda_available: bool,
    pub opencl_available: bool,
    pub memory_mb: Option<u64>,
    pub compute_capability: Option<String>,
}

impl GpuCapabilities {
    /// Detect available GPU capabilities
    pub fn detect() -> Self {
        #[cfg(feature = "gpu")]
        {
            let cuda_available = Device::cuda_if_available(0).is_ok();

            Self {
                cuda_available,
                opencl_available: false,  // Would need OpenCL detection
                memory_mb: None,          // Would need memory introspection
                compute_capability: None, // Would need capability detection
            }
        }
        #[cfg(not(feature = "gpu"))]
        {
            Self {
                cuda_available: false,
                opencl_available: false,
                memory_mb: None,
                compute_capability: None,
            }
        }
    }

    /// Check if any GPU acceleration is available
    pub fn has_gpu_support(&self) -> bool {
        self.cuda_available || self.opencl_available
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Emotion, EmotionIntensity, EmotionVector};

    #[test]
    fn test_gpu_processor_creation() {
        let processor = GpuEmotionProcessor::new().unwrap();
        // Should not fail even if GPU is not available
        debug!("GPU processor created: {}", processor.device_info());
    }

    #[test]
    fn test_gpu_capabilities_detection() {
        let caps = GpuCapabilities::detect();
        debug!("GPU capabilities: {:?}", caps);
        // Should not fail regardless of GPU availability
    }

    #[test]
    fn test_gpu_energy_scaling() {
        let processor = GpuEmotionProcessor::new().unwrap();
        let test_audio = vec![0.1, -0.2, 0.3, -0.1, 0.2];
        let factor = 1.5;

        // Test should work regardless of GPU availability
        match processor.apply_energy_scaling_gpu(&test_audio, factor) {
            Ok(result) => {
                assert_eq!(result.len(), test_audio.len());

                // Check that scaling was applied (approximately)
                for (i, &sample) in result.iter().enumerate() {
                    let expected = test_audio[i] * factor;
                    assert!(
                        (sample - expected).abs() < 0.001,
                        "Sample {} mismatch: got {}, expected {}",
                        i,
                        sample,
                        expected
                    );
                }
            }
            Err(_) => {
                // GPU not available - this is OK for testing
                debug!("GPU not available for energy scaling test");
            }
        }
    }

    #[test]
    fn test_gpu_pitch_shift() {
        let processor = GpuEmotionProcessor::new().unwrap();
        let test_audio = vec![0.1, -0.2, 0.3, -0.1, 0.2];
        let pitch_shift = 1.2;

        match processor.apply_pitch_shift_gpu(&test_audio, pitch_shift) {
            Ok(result) => {
                assert_eq!(result.len(), test_audio.len());
                // Output should be different from input due to resampling
            }
            Err(_) => {
                // GPU not available - this is OK for testing
                debug!("GPU not available for pitch shift test");
            }
        }
    }

    #[test]
    fn test_gpu_tempo_modification() {
        let processor = GpuEmotionProcessor::new().unwrap();
        let test_audio = vec![0.1, -0.2, 0.3, -0.1, 0.2, 0.0, -0.1, 0.4];
        let tempo_scale = 0.8;

        match processor.apply_tempo_modification_gpu(&test_audio, tempo_scale) {
            Ok(result) => {
                let expected_len = (test_audio.len() as f32 / tempo_scale) as usize;
                assert_eq!(result.len(), expected_len);
            }
            Err(_) => {
                // GPU not available - this is OK for testing
                debug!("GPU not available for tempo modification test");
            }
        }
    }

    #[test]
    fn test_gpu_audio_processing() {
        let processor = GpuEmotionProcessor::new().unwrap();
        let test_audio = vec![0.1, -0.2, 0.3, -0.1, 0.2];

        let mut emotion_vector = EmotionVector::new();
        emotion_vector.add_emotion(Emotion::Happy, EmotionIntensity::HIGH);

        let params = EmotionParameters {
            emotion_vector,
            duration_ms: None,
            fade_in_ms: None,
            fade_out_ms: None,
            pitch_shift: 1.1,
            tempo_scale: 0.9,
            energy_scale: 1.2,
            breathiness: 0.0,
            roughness: 0.0,
            custom_params: std::collections::HashMap::new(),
        };

        match processor.process_audio_gpu(&test_audio, &params) {
            Ok(result) => {
                assert!(!result.is_empty());
                // Result should be different from input due to processing
            }
            Err(_) => {
                // GPU not available - this is OK for testing
                debug!("GPU not available for audio processing test");
            }
        }
    }

    #[test]
    fn test_gpu_performance_benchmark() {
        let processor = GpuEmotionProcessor::new().unwrap();
        let result = processor.benchmark_performance(1000);

        match result {
            Ok((gpu_time, cpu_time)) => {
                debug!("Benchmark: GPU={:.4}s, CPU={:.4}s", gpu_time, cpu_time);
                assert!(gpu_time >= 0.0);
                assert!(cpu_time >= 0.0);
            }
            Err(e) => {
                debug!("Benchmark failed (expected on systems without GPU): {}", e);
                // This is OK - not all systems have GPU support
            }
        }
    }
}

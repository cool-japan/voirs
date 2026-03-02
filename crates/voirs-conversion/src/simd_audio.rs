//! SIMD-optimized audio processing operations
//!
//! This module provides high-performance audio processing primitives
//! using SIMD operations from scirs2-core for maximum efficiency.

use crate::Result;

/// SIMD-optimized audio buffer operations
pub struct SimdAudioOps;

impl SimdAudioOps {
    /// Mix two audio buffers with specified weights using SIMD
    ///
    /// # Arguments
    /// * `buf1` - First audio buffer
    /// * `buf2` - Second audio buffer
    /// * `weight1` - Weight for first buffer (0.0-1.0)
    /// * `weight2` - Weight for second buffer (0.0-1.0)
    ///
    /// # Returns
    /// Mixed audio buffer
    ///
    /// # Example
    /// ```
    /// use voirs_conversion::simd_audio::SimdAudioOps;
    ///
    /// let buf1 = vec![1.0, 2.0, 3.0, 4.0];
    /// let buf2 = vec![0.5, 1.0, 1.5, 2.0];
    /// let mixed = SimdAudioOps::mix_buffers(&buf1, &buf2, 0.5, 0.5).expect("operation should succeed");
    /// // Result is weighted average of both buffers
    /// ```
    pub fn mix_buffers(buf1: &[f32], buf2: &[f32], weight1: f32, weight2: f32) -> Result<Vec<f32>> {
        if buf1.len() != buf2.len() {
            return Err(crate::Error::Processing {
                operation: "mix_buffers".to_string(),
                message: "Buffer lengths must match for mixing".to_string(),
                context: None,
                recovery_suggestions: Box::new(vec![
                    "Ensure both buffers have the same length".to_string(),
                    "Resample or trim buffers to match lengths".to_string(),
                ]),
            });
        }

        // Use SIMD-friendly loop for efficient mixing
        // LLVM will auto-vectorize this pattern
        let result: Vec<f32> = buf1
            .iter()
            .zip(buf2.iter())
            .map(|(&a, &b)| a * weight1 + b * weight2)
            .collect();

        Ok(result)
    }

    /// Apply gain to audio buffer using SIMD
    ///
    /// # Arguments
    /// * `buffer` - Input audio buffer
    /// * `gain` - Gain factor to apply
    ///
    /// # Returns
    /// Amplified audio buffer
    ///
    /// # Example
    /// ```
    /// use voirs_conversion::simd_audio::SimdAudioOps;
    ///
    /// let buffer = vec![1.0, 2.0, 3.0, 4.0];
    /// let amplified = SimdAudioOps::apply_gain(&buffer, 2.0);
    /// assert_eq!(amplified[0], 2.0);
    /// assert_eq!(amplified[1], 4.0);
    /// ```
    pub fn apply_gain(buffer: &[f32], gain: f32) -> Vec<f32> {
        // LLVM auto-vectorization friendly pattern
        buffer.iter().map(|&x| x * gain).collect()
    }

    /// Apply gain to audio buffer in-place using SIMD
    ///
    /// # Arguments
    /// * `buffer` - Audio buffer to modify
    /// * `gain` - Gain factor to apply
    ///
    /// # Example
    /// ```
    /// use voirs_conversion::simd_audio::SimdAudioOps;
    ///
    /// let mut buffer = vec![1.0, 2.0, 3.0, 4.0];
    /// SimdAudioOps::apply_gain_inplace(&mut buffer, 2.0);
    /// assert_eq!(buffer[0], 2.0);
    /// assert_eq!(buffer[1], 4.0);
    /// ```
    pub fn apply_gain_inplace(buffer: &mut [f32], gain: f32) {
        // In-place gain application with SIMD-friendly pattern
        for x in buffer.iter_mut() {
            *x *= gain;
        }
    }

    /// Normalize audio buffer to target RMS level using SIMD
    ///
    /// # Arguments
    /// * `buffer` - Input audio buffer
    /// * `target_rms` - Target RMS level (typically 0.0-1.0)
    ///
    /// # Returns
    /// Normalized audio buffer
    ///
    /// # Example
    /// ```
    /// use voirs_conversion::simd_audio::SimdAudioOps;
    ///
    /// let buffer = vec![1.0, 2.0, 3.0, 4.0];
    /// let normalized = SimdAudioOps::normalize_rms(&buffer, 0.5);
    /// // Buffer is scaled to have RMS of 0.5
    /// ```
    pub fn normalize_rms(buffer: &[f32], target_rms: f32) -> Vec<f32> {
        if buffer.is_empty() {
            return buffer.to_vec();
        }

        // Calculate current RMS using SIMD
        let sum_squares: f32 = buffer.iter().map(|x| x * x).sum();
        let rms = (sum_squares / buffer.len() as f32).sqrt();

        if rms < f32::EPSILON {
            return buffer.to_vec();
        }

        // Calculate gain needed for target RMS
        let gain = target_rms / rms;

        // Apply gain using SIMD
        Self::apply_gain(buffer, gain)
    }

    /// Calculate RMS (Root Mean Square) of audio buffer
    ///
    /// # Arguments
    /// * `buffer` - Input audio buffer
    ///
    /// # Returns
    /// RMS value
    ///
    /// # Example
    /// ```
    /// use voirs_conversion::simd_audio::SimdAudioOps;
    ///
    /// let buffer = vec![1.0, 2.0, 3.0, 4.0];
    /// let rms = SimdAudioOps::calculate_rms(&buffer);
    /// assert!(rms > 0.0);
    /// ```
    pub fn calculate_rms(buffer: &[f32]) -> f32 {
        if buffer.is_empty() {
            return 0.0;
        }

        let sum_squares: f32 = buffer.iter().map(|x| x * x).sum();
        (sum_squares / buffer.len() as f32).sqrt()
    }

    /// Apply soft clipping to prevent distortion using SIMD
    ///
    /// # Arguments
    /// * `buffer` - Input audio buffer
    /// * `threshold` - Clipping threshold (0.0-1.0)
    ///
    /// # Returns
    /// Soft-clipped audio buffer
    ///
    /// # Example
    /// ```
    /// use voirs_conversion::simd_audio::SimdAudioOps;
    ///
    /// let buffer = vec![0.5, 1.5, -1.2, 0.8];
    /// let clipped = SimdAudioOps::soft_clip(&buffer, 1.0);
    /// // Values above threshold are soft-clipped using tanh
    /// ```
    pub fn soft_clip(buffer: &[f32], threshold: f32) -> Vec<f32> {
        buffer
            .iter()
            .map(|&x| {
                if x.abs() > threshold {
                    threshold * x.signum() * (1.0 - (-x.abs() / threshold).exp())
                } else {
                    x
                }
            })
            .collect()
    }

    /// Crossfade between two audio buffers using SIMD
    ///
    /// # Arguments
    /// * `buf1` - First audio buffer
    /// * `buf2` - Second audio buffer
    /// * `crossfade_samples` - Number of samples for crossfade
    ///
    /// # Returns
    /// Crossfaded audio buffer
    ///
    /// # Example
    /// ```
    /// use voirs_conversion::simd_audio::SimdAudioOps;
    ///
    /// let buf1 = vec![1.0; 100];
    /// let buf2 = vec![0.5; 100];
    /// let crossfaded = SimdAudioOps::crossfade(&buf1, &buf2, 50).expect("operation should succeed");
    /// // First 50 samples gradually transition from buf1 to buf2
    /// ```
    pub fn crossfade(buf1: &[f32], buf2: &[f32], crossfade_samples: usize) -> Result<Vec<f32>> {
        if buf1.len() != buf2.len() {
            return Err(crate::Error::Processing {
                operation: "crossfade".to_string(),
                message: "Buffer lengths must match for crossfading".to_string(),
                context: None,
                recovery_suggestions: Box::new(vec![
                    "Ensure both buffers have the same length".to_string()
                ]),
            });
        }

        if crossfade_samples > buf1.len() {
            return Err(crate::Error::Processing {
                operation: "crossfade".to_string(),
                message: "Crossfade samples exceed buffer length".to_string(),
                context: None,
                recovery_suggestions: Box::new(vec![
                    "Reduce crossfade samples to not exceed buffer length".to_string(),
                ]),
            });
        }

        let mut result = Vec::with_capacity(buf1.len());

        // Pre-crossfade: use buf1
        result.extend_from_slice(&buf1[..buf1.len().saturating_sub(crossfade_samples)]);

        // Crossfade region
        let start_idx = buf1.len().saturating_sub(crossfade_samples);
        for i in 0..crossfade_samples {
            let t = i as f32 / crossfade_samples as f32;
            let weight1 = 1.0 - t;
            let weight2 = t;
            let idx = start_idx + i;
            result.push(buf1[idx] * weight1 + buf2[idx] * weight2);
        }

        Ok(result)
    }

    /// Compute dot product of two audio buffers using SIMD
    ///
    /// # Arguments
    /// * `buf1` - First audio buffer
    /// * `buf2` - Second audio buffer
    ///
    /// # Returns
    /// Dot product value
    ///
    /// # Example
    /// ```
    /// use voirs_conversion::simd_audio::SimdAudioOps;
    ///
    /// let buf1 = vec![1.0, 2.0, 3.0];
    /// let buf2 = vec![4.0, 5.0, 6.0];
    /// let dot = SimdAudioOps::dot_product(&buf1, &buf2).expect("operation should succeed");
    /// assert_eq!(dot, 32.0); // 1*4 + 2*5 + 3*6
    /// ```
    pub fn dot_product(buf1: &[f32], buf2: &[f32]) -> Result<f32> {
        if buf1.len() != buf2.len() {
            return Err(crate::Error::Processing {
                operation: "dot_product".to_string(),
                message: "Buffer lengths must match for dot product".to_string(),
                context: None,
                recovery_suggestions: Box::new(vec![
                    "Ensure both buffers have the same length".to_string()
                ]),
            });
        }

        Ok(buf1.iter().zip(buf2.iter()).map(|(a, b)| a * b).sum())
    }

    /// Apply DC offset removal (high-pass filter) using SIMD
    ///
    /// # Arguments
    /// * `buffer` - Input audio buffer
    ///
    /// # Returns
    /// DC-offset removed audio buffer
    ///
    /// # Example
    /// ```
    /// use voirs_conversion::simd_audio::SimdAudioOps;
    ///
    /// let buffer = vec![1.5, 1.6, 1.4, 1.5]; // Has DC offset of ~1.5
    /// let filtered = SimdAudioOps::remove_dc_offset(&buffer);
    /// // DC component is removed
    /// ```
    pub fn remove_dc_offset(buffer: &[f32]) -> Vec<f32> {
        if buffer.is_empty() {
            return buffer.to_vec();
        }

        // Calculate mean (DC offset)
        let mean: f32 = buffer.iter().sum::<f32>() / buffer.len() as f32;

        // Subtract mean from all samples
        buffer.iter().map(|&x| x - mean).collect()
    }

    /// Zero-pad buffer to specified length
    ///
    /// # Arguments
    /// * `buffer` - Input audio buffer
    /// * `target_length` - Target length after padding
    ///
    /// # Returns
    /// Zero-padded buffer
    ///
    /// # Example
    /// ```
    /// use voirs_conversion::simd_audio::SimdAudioOps;
    ///
    /// let buffer = vec![1.0, 2.0, 3.0];
    /// let padded = SimdAudioOps::zero_pad(&buffer, 5);
    /// assert_eq!(padded.len(), 5);
    /// assert_eq!(padded[3], 0.0);
    /// assert_eq!(padded[4], 0.0);
    /// ```
    pub fn zero_pad(buffer: &[f32], target_length: usize) -> Vec<f32> {
        if buffer.len() >= target_length {
            return buffer.to_vec();
        }

        let mut result = Vec::with_capacity(target_length);
        result.extend_from_slice(buffer);
        result.resize(target_length, 0.0);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mix_buffers() {
        let buf1 = vec![1.0, 2.0, 3.0, 4.0];
        let buf2 = vec![0.0, 0.0, 0.0, 0.0];

        let mixed = SimdAudioOps::mix_buffers(&buf1, &buf2, 0.5, 0.5).unwrap();
        assert_eq!(mixed.len(), 4);
        assert!((mixed[0] - 0.5).abs() < f32::EPSILON);
        assert!((mixed[1] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_apply_gain() {
        let buffer = vec![1.0, 2.0, 3.0, 4.0];
        let amplified = SimdAudioOps::apply_gain(&buffer, 2.0);

        assert_eq!(amplified.len(), 4);
        assert!((amplified[0] - 2.0).abs() < f32::EPSILON);
        assert!((amplified[1] - 4.0).abs() < f32::EPSILON);
        assert!((amplified[2] - 6.0).abs() < f32::EPSILON);
        assert!((amplified[3] - 8.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_apply_gain_inplace() {
        let mut buffer = vec![1.0, 2.0, 3.0, 4.0];
        SimdAudioOps::apply_gain_inplace(&mut buffer, 2.0);

        assert!((buffer[0] - 2.0).abs() < f32::EPSILON);
        assert!((buffer[1] - 4.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_calculate_rms() {
        let buffer = vec![1.0, 2.0, 3.0, 4.0];
        let rms = SimdAudioOps::calculate_rms(&buffer);

        // Expected RMS: sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.739
        assert!((rms - 2.739).abs() < 0.01);
    }

    #[test]
    fn test_normalize_rms() {
        let buffer = vec![1.0, 2.0, 3.0, 4.0];
        let normalized = SimdAudioOps::normalize_rms(&buffer, 1.0);

        let result_rms = SimdAudioOps::calculate_rms(&normalized);
        assert!((result_rms - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_soft_clip() {
        let buffer = vec![0.5, 1.5, -1.2, 0.8];
        let clipped = SimdAudioOps::soft_clip(&buffer, 1.0);

        // Values below threshold should be unchanged
        assert!((clipped[0] - 0.5).abs() < 0.01);
        assert!((clipped[3] - 0.8).abs() < 0.01);

        // Values above threshold should be clipped
        assert!(clipped[1].abs() < 1.5);
        assert!(clipped[2].abs() < 1.2);
    }

    #[test]
    fn test_crossfade() {
        let buf1 = vec![1.0; 100];
        let buf2 = vec![0.0; 100];

        let crossfaded = SimdAudioOps::crossfade(&buf1, &buf2, 50).unwrap();
        assert_eq!(crossfaded.len(), 100);

        // Start should be buf1
        assert!((crossfaded[0] - 1.0).abs() < f32::EPSILON);
        // End should transition to buf2
        assert!(crossfaded[99] < 1.0);
    }

    #[test]
    fn test_dot_product() {
        let buf1 = vec![1.0, 2.0, 3.0];
        let buf2 = vec![4.0, 5.0, 6.0];

        let dot = SimdAudioOps::dot_product(&buf1, &buf2).unwrap();
        assert!((dot - 32.0).abs() < f32::EPSILON); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_remove_dc_offset() {
        let buffer = vec![1.5, 1.6, 1.4, 1.5];
        let filtered = SimdAudioOps::remove_dc_offset(&buffer);

        // Mean should be close to zero
        let mean: f32 = filtered.iter().sum::<f32>() / filtered.len() as f32;
        assert!(mean.abs() < 0.01);
    }

    #[test]
    fn test_zero_pad() {
        let buffer = vec![1.0, 2.0, 3.0];
        let padded = SimdAudioOps::zero_pad(&buffer, 5);

        assert_eq!(padded.len(), 5);
        assert_eq!(padded[0], 1.0);
        assert_eq!(padded[1], 2.0);
        assert_eq!(padded[2], 3.0);
        assert_eq!(padded[3], 0.0);
        assert_eq!(padded[4], 0.0);
    }
}

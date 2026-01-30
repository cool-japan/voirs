//! Enhanced SIMD operations using SciRS2-Core features
//!
//! This module bridges between raw audio buffer SIMD operations and
//! SciRS2-Core's advanced SIMD abstractions, providing the best of both worlds.

use crate::{DatasetError, Result};
use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_core::simd::{
    simd_add_f32, simd_dot_f32, simd_fma_f32_ultra, simd_mul_f32, simd_scalar_mul_f32, simd_sum_f32,
};

/// Enhanced SIMD processor with SciRS2-Core integration
pub struct EnhancedSimdProcessor;

impl EnhancedSimdProcessor {
    /// Convert slice to ArrayView1 for SciRS2 operations
    #[inline]
    fn slice_to_view(slice: &[f32]) -> ArrayView1<'_, f32> {
        ArrayView1::from(slice)
    }

    /// High-performance dot product using SciRS2-Core
    ///
    /// This uses SciRS2's optimized SIMD dot product implementation which
    /// automatically selects the best instruction set (AVX512, AVX2, SSE, or scalar)
    pub fn dot_product_scirs2(a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(DatasetError::AudioError(format!(
                "Slice length mismatch: {} vs {}",
                a.len(),
                b.len()
            )));
        }

        let a_view = Self::slice_to_view(a);
        let b_view = Self::slice_to_view(b);

        Ok(simd_dot_f32(&a_view, &b_view))
    }

    /// Element-wise addition using SciRS2-Core
    pub fn add_scirs2(a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        if a.len() != b.len() {
            return Err(DatasetError::AudioError(format!(
                "Slice length mismatch: {} vs {}",
                a.len(),
                b.len()
            )));
        }

        let a_view = Self::slice_to_view(a);
        let b_view = Self::slice_to_view(b);

        let result = simd_add_f32(&a_view, &b_view);
        Ok(result.to_vec())
    }

    /// Element-wise multiplication using SciRS2-Core
    pub fn multiply_scirs2(a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        if a.len() != b.len() {
            return Err(DatasetError::AudioError(format!(
                "Slice length mismatch: {} vs {}",
                a.len(),
                b.len()
            )));
        }

        let a_view = Self::slice_to_view(a);
        let b_view = Self::slice_to_view(b);

        let result = simd_mul_f32(&a_view, &b_view);
        Ok(result.to_vec())
    }

    /// Scalar multiplication using SciRS2-Core
    pub fn scalar_multiply_scirs2(samples: &[f32], scalar: f32) -> Vec<f32> {
        let view = Self::slice_to_view(samples);
        let result = simd_scalar_mul_f32(&view, scalar);
        result.to_vec()
    }

    /// Sum reduction using SciRS2-Core
    pub fn sum_scirs2(samples: &[f32]) -> f32 {
        let view = Self::slice_to_view(samples);
        simd_sum_f32(&view)
    }

    /// Fused multiply-add using SciRS2-Core ultra-optimized version
    ///
    /// Computes: result = a * b + c
    /// This is highly optimized for modern CPUs with FMA instructions
    pub fn fma_scirs2(a: &[f32], b: &[f32], c: &[f32]) -> Result<Vec<f32>> {
        if a.len() != b.len() || a.len() != c.len() {
            return Err(DatasetError::AudioError(
                "All slices must have the same length for FMA".to_string(),
            ));
        }

        let a_view = Self::slice_to_view(a);
        let b_view = Self::slice_to_view(b);
        let c_view = Self::slice_to_view(c);

        let result = simd_fma_f32_ultra(&a_view, &b_view, &c_view);
        Ok(result.to_vec())
    }

    /// RMS calculation using SciRS2-Core optimized operations
    pub fn calculate_rms_scirs2(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }

        let view = Self::slice_to_view(samples);
        let sum_squares = simd_dot_f32(&view, &view);
        (sum_squares / samples.len() as f32).sqrt()
    }

    /// Energy calculation using SciRS2-Core (sum of squares)
    pub fn calculate_energy_scirs2(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }

        let view = Self::slice_to_view(samples);
        simd_dot_f32(&view, &view)
    }

    /// Convolution using SciRS2-Core optimized operations
    ///
    /// This performs 1D convolution of input signal with kernel
    /// using SIMD-accelerated dot products for each output sample
    pub fn convolve_scirs2(signal: &[f32], kernel: &[f32]) -> Vec<f32> {
        if signal.is_empty() || kernel.is_empty() {
            return vec![];
        }

        let output_len = signal.len() + kernel.len() - 1;
        let mut output = vec![0.0f32; output_len];

        // Flip kernel for convolution (correlation is without flip)
        let kernel_flipped: Vec<f32> = kernel.iter().rev().copied().collect();
        let kernel_view = ArrayView1::from(&kernel_flipped);

        // Index i is used for mathematical calculations, not just array indexing
        #[allow(clippy::needless_range_loop)]
        for i in 0..output_len {
            let mut sum = 0.0;

            // Determine the range of kernel indices that overlap with signal
            let k_start = if i < signal.len() {
                0
            } else {
                i - signal.len() + 1
            };
            let k_end = (i + 1).min(kernel.len());

            // Extract overlapping signal segment
            let sig_start = if i < kernel.len() {
                0
            } else {
                i - kernel.len() + 1
            };
            let sig_end = (i + 1).min(signal.len());

            if sig_start < sig_end && k_start < k_end {
                let sig_segment = &signal[sig_start..sig_end];
                let ker_segment = &kernel_flipped[k_start..k_end];

                // Use SciRS2 optimized dot product if segments match
                if sig_segment.len() == ker_segment.len() {
                    let sig_view = ArrayView1::from(sig_segment);
                    sum = simd_dot_f32(&sig_view, &ArrayView1::from(ker_segment));
                } else {
                    // Fallback to manual calculation for edge cases
                    sum = sig_segment
                        .iter()
                        .zip(ker_segment.iter())
                        .map(|(s, k)| s * k)
                        .sum();
                }
            }

            output[i] = sum;
        }

        output
    }

    /// Correlation using SciRS2-Core optimized operations
    ///
    /// This performs 1D correlation of two signals
    /// (like convolution but without flipping the kernel)
    pub fn correlate_scirs2(signal: &[f32], template: &[f32]) -> Vec<f32> {
        if signal.is_empty() || template.is_empty() {
            return vec![];
        }

        let output_len = signal.len() - template.len() + 1;
        if output_len == 0 {
            return vec![];
        }

        let mut output = vec![0.0f32; output_len];
        let template_view = ArrayView1::from(template);

        for i in 0..output_len {
            let segment = &signal[i..i + template.len()];
            let segment_view = ArrayView1::from(segment);
            output[i] = simd_dot_f32(&segment_view, &template_view);
        }

        output
    }

    /// Exponential moving average using SciRS2-Core operations
    ///
    /// Computes: y[n] = alpha * x[n] + (1 - alpha) * y[n-1]
    pub fn exponential_moving_average_scirs2(samples: &[f32], alpha: f32) -> Vec<f32> {
        if samples.is_empty() {
            return vec![];
        }

        let mut output = Vec::with_capacity(samples.len());
        output.push(samples[0]);

        let one_minus_alpha = 1.0 - alpha;

        for &sample in &samples[1..] {
            let prev = *output.last().unwrap();
            output.push(alpha * sample + one_minus_alpha * prev);
        }

        output
    }

    /// Compute running variance using SciRS2-Core operations
    ///
    /// Uses Welford's online algorithm for numerical stability
    pub fn running_variance_scirs2(samples: &[f32]) -> Vec<f32> {
        if samples.is_empty() {
            return vec![];
        }

        let mut variances = Vec::with_capacity(samples.len());
        let mut mean = 0.0;
        let mut m2 = 0.0;

        for (i, &sample) in samples.iter().enumerate() {
            let count = (i + 1) as f32;
            let delta = sample - mean;
            mean += delta / count;
            let delta2 = sample - mean;
            m2 += delta * delta2;

            let variance = if i > 0 { m2 / count } else { 0.0 };
            variances.push(variance);
        }

        variances
    }

    /// Batch normalize samples using SciRS2-Core operations
    ///
    /// Normalizes to zero mean and unit variance
    pub fn batch_normalize_scirs2(samples: &[f32]) -> Vec<f32> {
        if samples.is_empty() {
            return vec![];
        }

        let view = Self::slice_to_view(samples);

        // Calculate mean using SciRS2
        let sum = simd_sum_f32(&view);
        let mean = sum / samples.len() as f32;

        // Calculate variance
        let mut variance = 0.0;
        for &sample in samples {
            let diff = sample - mean;
            variance += diff * diff;
        }
        variance /= samples.len() as f32;

        let std_dev = variance.sqrt().max(1e-8); // Avoid division by zero

        // Normalize
        let normalized: Vec<f32> = samples.iter().map(|&x| (x - mean) / std_dev).collect();

        normalized
    }

    /// Windowed RMS calculation with SciRS2-Core
    ///
    /// Computes RMS over sliding windows
    pub fn windowed_rms_scirs2(samples: &[f32], window_size: usize) -> Vec<f32> {
        if samples.is_empty() || window_size == 0 || window_size > samples.len() {
            return vec![];
        }

        let output_len = samples.len() - window_size + 1;
        let mut output = Vec::with_capacity(output_len);

        for i in 0..output_len {
            let window = &samples[i..i + window_size];
            let view = ArrayView1::from(window);
            let sum_squares = simd_dot_f32(&view, &view);
            let rms = (sum_squares / window_size as f32).sqrt();
            output.push(rms);
        }

        output
    }

    /// Peak normalization using SciRS2-Core
    ///
    /// Normalizes signal so that peak amplitude equals target
    pub fn peak_normalize_scirs2(samples: &[f32], target_peak: f32) -> Vec<f32> {
        if samples.is_empty() {
            return vec![];
        }

        // Find peak
        let peak = samples.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);

        if peak < 1e-10 {
            return samples.to_vec();
        }

        // Normalize using SciRS2 scalar multiplication
        let gain = target_peak / peak;
        Self::scalar_multiply_scirs2(samples, gain)
    }

    /// Crossfade between two signals using SciRS2-Core
    ///
    /// Linear crossfade from signal A to signal B
    pub fn crossfade_scirs2(signal_a: &[f32], signal_b: &[f32]) -> Result<Vec<f32>> {
        if signal_a.len() != signal_b.len() {
            return Err(DatasetError::AudioError(
                "Signals must have same length for crossfade".to_string(),
            ));
        }

        let len = signal_a.len();
        if len == 0 {
            return Ok(vec![]);
        }

        let mut output = Vec::with_capacity(len);

        for i in 0..len {
            let alpha = i as f32 / (len - 1).max(1) as f32;
            let sample = (1.0 - alpha) * signal_a[i] + alpha * signal_b[i];
            output.push(sample);
        }

        Ok(output)
    }

    /// Downsample by integer factor using SciRS2-Core
    ///
    /// Simple decimation (no anti-aliasing filter)
    pub fn downsample_scirs2(samples: &[f32], factor: usize) -> Vec<f32> {
        if samples.is_empty() || factor == 0 {
            return vec![];
        }

        samples.iter().step_by(factor).copied().collect()
    }

    /// Upsample by integer factor with zero-insertion
    pub fn upsample_scirs2(samples: &[f32], factor: usize) -> Vec<f32> {
        if samples.is_empty() || factor == 0 {
            return vec![];
        }

        let output_len = samples.len() * factor;
        let mut output = vec![0.0; output_len];

        for (i, &sample) in samples.iter().enumerate() {
            output[i * factor] = sample;
        }

        output
    }

    /// Compute spectrogram magnitude bins using SciRS2-Core
    ///
    /// Given complex FFT output (interleaved real/imag), compute magnitudes
    pub fn compute_magnitude_spectrum_scirs2(complex_spectrum: &[f32]) -> Vec<f32> {
        if !complex_spectrum.len().is_multiple_of(2) {
            return vec![];
        }

        let num_bins = complex_spectrum.len() / 2;
        let mut magnitudes = Vec::with_capacity(num_bins);

        for i in 0..num_bins {
            let real = complex_spectrum[2 * i];
            let imag = complex_spectrum[2 * i + 1];
            let magnitude = (real * real + imag * imag).sqrt();
            magnitudes.push(magnitude);
        }

        magnitudes
    }

    /// Compute power spectrum using SciRS2-Core
    ///
    /// Given complex FFT output (interleaved real/imag), compute power
    pub fn compute_power_spectrum_scirs2(complex_spectrum: &[f32]) -> Vec<f32> {
        if !complex_spectrum.len().is_multiple_of(2) {
            return vec![];
        }

        let num_bins = complex_spectrum.len() / 2;
        let mut power = Vec::with_capacity(num_bins);

        for i in 0..num_bins {
            let real = complex_spectrum[2 * i];
            let imag = complex_spectrum[2 * i + 1];
            let pow = real * real + imag * imag;
            power.push(pow);
        }

        power
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product_scirs2() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];

        let result = EnhancedSimdProcessor::dot_product_scirs2(&a, &b).unwrap();
        let expected = 1.0 * 5.0 + 2.0 * 6.0 + 3.0 * 7.0 + 4.0 * 8.0;

        assert!((result - expected).abs() < 1e-5);
    }

    #[test]
    fn test_add_scirs2() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];

        let result = EnhancedSimdProcessor::add_scirs2(&a, &b).unwrap();

        assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_multiply_scirs2() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];

        let result = EnhancedSimdProcessor::multiply_scirs2(&a, &b).unwrap();

        assert_eq!(result, vec![5.0, 12.0, 21.0, 32.0]);
    }

    #[test]
    fn test_scalar_multiply_scirs2() {
        let samples = vec![1.0, 2.0, 3.0, 4.0];
        let scalar = 2.5;

        let result = EnhancedSimdProcessor::scalar_multiply_scirs2(&samples, scalar);

        assert_eq!(result, vec![2.5, 5.0, 7.5, 10.0]);
    }

    #[test]
    fn test_sum_scirs2() {
        let samples = vec![1.0, 2.0, 3.0, 4.0];

        let result = EnhancedSimdProcessor::sum_scirs2(&samples);

        assert!((result - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_fma_scirs2() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let c = vec![0.1, 0.2, 0.3, 0.4];

        let result = EnhancedSimdProcessor::fma_scirs2(&a, &b, &c).unwrap();

        // Expected: a * b + c
        assert!((result[0] - (1.0 * 5.0 + 0.1)).abs() < 1e-5);
        assert!((result[1] - (2.0 * 6.0 + 0.2)).abs() < 1e-5);
        assert!((result[2] - (3.0 * 7.0 + 0.3)).abs() < 1e-5);
        assert!((result[3] - (4.0 * 8.0 + 0.4)).abs() < 1e-5);
    }

    #[test]
    fn test_rms_scirs2() {
        let samples = vec![1.0, 2.0, 3.0, 4.0];

        let result = EnhancedSimdProcessor::calculate_rms_scirs2(&samples);
        let expected = ((1.0 + 4.0 + 9.0 + 16.0) / 4.0f32).sqrt();

        assert!((result - expected).abs() < 1e-5);
    }

    #[test]
    fn test_energy_scirs2() {
        let samples = vec![1.0, 2.0, 3.0, 4.0];

        let result = EnhancedSimdProcessor::calculate_energy_scirs2(&samples);
        let expected = 1.0 + 4.0 + 9.0 + 16.0;

        assert!((result - expected).abs() < 1e-5);
    }

    #[test]
    fn test_correlate_scirs2() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let template = vec![1.0, 0.5];

        let result = EnhancedSimdProcessor::correlate_scirs2(&signal, &template);

        // Expected: [1*1 + 2*0.5, 2*1 + 3*0.5, 3*1 + 4*0.5, 4*1 + 5*0.5]
        assert_eq!(result.len(), 4);
        assert!((result[0] - 2.0).abs() < 1e-5);
        assert!((result[1] - 3.5).abs() < 1e-5);
        assert!((result[2] - 5.0).abs() < 1e-5);
        assert!((result[3] - 6.5).abs() < 1e-5);
    }

    #[test]
    fn test_ema_scirs2() {
        let samples = vec![1.0, 2.0, 3.0, 4.0];
        let alpha = 0.5;

        let result = EnhancedSimdProcessor::exponential_moving_average_scirs2(&samples, alpha);

        assert_eq!(result.len(), 4);
        assert_eq!(result[0], 1.0);
        // result[1] = 0.5 * 2.0 + 0.5 * 1.0 = 1.5
        assert!((result[1] - 1.5).abs() < 1e-5);
    }

    #[test]
    fn test_batch_normalize_scirs2() {
        let samples = vec![1.0, 2.0, 3.0, 4.0];

        let result = EnhancedSimdProcessor::batch_normalize_scirs2(&samples);

        // Check mean is approximately 0
        let mean: f32 = result.iter().sum::<f32>() / result.len() as f32;
        assert!(mean.abs() < 1e-5);

        // Check variance is approximately 1
        let variance: f32 = result.iter().map(|&x| x * x).sum::<f32>() / result.len() as f32;
        assert!((variance - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_peak_normalize_scirs2() {
        let samples = vec![0.5, -1.0, 0.75, -0.25];
        let target = 0.5;

        let result = EnhancedSimdProcessor::peak_normalize_scirs2(&samples, target);

        let peak = result.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        assert!((peak - target).abs() < 1e-5);
    }

    #[test]
    fn test_crossfade_scirs2() {
        let a = vec![1.0, 1.0, 1.0, 1.0];
        let b = vec![0.0, 0.0, 0.0, 0.0];

        let result = EnhancedSimdProcessor::crossfade_scirs2(&a, &b).unwrap();

        assert_eq!(result.len(), 4);
        assert!((result[0] - 1.0).abs() < 1e-5); // Start with A
        assert!((result[3] - 0.0).abs() < 1e-5); // End with B
    }

    #[test]
    fn test_downsample_scirs2() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let factor = 2;

        let result = EnhancedSimdProcessor::downsample_scirs2(&samples, factor);

        assert_eq!(result, vec![1.0, 3.0, 5.0]);
    }

    #[test]
    fn test_upsample_scirs2() {
        let samples = vec![1.0, 2.0, 3.0];
        let factor = 2;

        let result = EnhancedSimdProcessor::upsample_scirs2(&samples, factor);

        assert_eq!(result, vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0]);
    }

    #[test]
    fn test_magnitude_spectrum_scirs2() {
        // Complex spectrum: (1+0i), (0+1i), (3+4i)
        let complex = vec![1.0, 0.0, 0.0, 1.0, 3.0, 4.0];

        let result = EnhancedSimdProcessor::compute_magnitude_spectrum_scirs2(&complex);

        assert_eq!(result.len(), 3);
        assert!((result[0] - 1.0).abs() < 1e-5);
        assert!((result[1] - 1.0).abs() < 1e-5);
        assert!((result[2] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_power_spectrum_scirs2() {
        // Complex spectrum: (1+0i), (0+1i), (3+4i)
        let complex = vec![1.0, 0.0, 0.0, 1.0, 3.0, 4.0];

        let result = EnhancedSimdProcessor::compute_power_spectrum_scirs2(&complex);

        assert_eq!(result.len(), 3);
        assert!((result[0] - 1.0).abs() < 1e-5);
        assert!((result[1] - 1.0).abs() < 1e-5);
        assert!((result[2] - 25.0).abs() < 1e-5);
    }

    #[test]
    fn test_windowed_rms_scirs2() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let window_size = 3;

        let result = EnhancedSimdProcessor::windowed_rms_scirs2(&samples, window_size);

        assert_eq!(result.len(), 3);
        // First window: [1, 2, 3], RMS = sqrt((1+4+9)/3) = sqrt(14/3)
        let expected_0 = ((1.0 + 4.0 + 9.0) / 3.0f32).sqrt();
        assert!((result[0] - expected_0).abs() < 1e-5);
    }

    #[test]
    fn test_running_variance_scirs2() {
        let samples = vec![1.0, 2.0, 3.0, 4.0];

        let result = EnhancedSimdProcessor::running_variance_scirs2(&samples);

        assert_eq!(result.len(), 4);
        assert!((result[0] - 0.0).abs() < 1e-5); // First sample has zero variance
        assert!(result[1] > 0.0); // Variance increases
    }

    #[test]
    fn test_length_mismatch_errors() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];

        assert!(EnhancedSimdProcessor::dot_product_scirs2(&a, &b).is_err());
        assert!(EnhancedSimdProcessor::add_scirs2(&a, &b).is_err());
        assert!(EnhancedSimdProcessor::multiply_scirs2(&a, &b).is_err());
        assert!(EnhancedSimdProcessor::crossfade_scirs2(&a, &b).is_err());
    }
}

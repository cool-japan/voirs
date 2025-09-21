//! SIMD-optimized audio processing operations
//!
//! This module provides vectorized implementations of common audio processing
//! operations for improved performance on supported platforms.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::{aarch64::*, is_aarch64_feature_detected};

/// SIMD-optimized audio processing utilities
pub struct SimdAudioProcessor;

impl SimdAudioProcessor {
    /// Check if the current CPU supports SSE 4.1 for SIMD operations
    #[cfg(target_arch = "x86_64")]
    pub fn is_simd_supported() -> bool {
        is_x86_feature_detected!("sse4.1")
    }

    /// Check if the current CPU supports AVX2 for enhanced SIMD operations
    #[cfg(target_arch = "x86_64")]
    pub fn is_avx2_supported() -> bool {
        is_x86_feature_detected!("avx2")
    }

    /// Check if the current CPU supports NEON for SIMD operations (ARM)
    #[cfg(target_arch = "aarch64")]
    pub fn is_simd_supported() -> bool {
        is_aarch64_feature_detected!("neon")
    }

    /// ARM processors don't have AVX2, but we can check for advanced NEON features
    #[cfg(target_arch = "aarch64")]
    pub fn is_avx2_supported() -> bool {
        false // AVX2 is x86_64 specific
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    pub fn is_simd_supported() -> bool {
        false
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    pub fn is_avx2_supported() -> bool {
        false
    }

    /// AVX2-optimized RMS calculation (processes 8 floats at once)
    /// Returns the root mean square of the audio samples
    #[cfg(target_arch = "x86_64")]
    pub unsafe fn calculate_rms_avx2(samples: &[f32]) -> f32 {
        if !Self::is_avx2_supported() || samples.len() < 16 {
            return Self::calculate_rms_simd(samples);
        }

        let mut sum_squares = _mm256_setzero_ps();
        let len = samples.len();
        let simd_len = len & !7; // Round down to multiple of 8

        // Process 8 samples at a time
        for i in (0..simd_len).step_by(8) {
            let chunk = _mm256_loadu_ps(samples.as_ptr().add(i));
            let squares = _mm256_mul_ps(chunk, chunk);
            sum_squares = _mm256_add_ps(sum_squares, squares);
        }

        // Horizontal sum of the AVX2 register
        let sum_array = std::mem::transmute::<__m256, [f32; 8]>(sum_squares);
        let mut total_sum = sum_array[0]
            + sum_array[1]
            + sum_array[2]
            + sum_array[3]
            + sum_array[4]
            + sum_array[5]
            + sum_array[6]
            + sum_array[7];

        // Process remaining samples (scalar)
        for &sample in &samples[simd_len..] {
            total_sum += sample * sample;
        }

        (total_sum / len as f32).sqrt()
    }

    /// SIMD-optimized RMS calculation (SSE 4.1)
    /// Returns the root mean square of the audio samples
    #[cfg(target_arch = "x86_64")]
    pub unsafe fn calculate_rms_simd(samples: &[f32]) -> f32 {
        if !Self::is_simd_supported() || samples.len() < 8 {
            return Self::calculate_rms_scalar(samples);
        }

        let mut sum_squares = _mm_setzero_ps();
        let len = samples.len();
        let simd_len = len & !3; // Round down to multiple of 4

        // Process 4 samples at a time
        for i in (0..simd_len).step_by(4) {
            let chunk = _mm_loadu_ps(samples.as_ptr().add(i));
            let squares = _mm_mul_ps(chunk, chunk);
            sum_squares = _mm_add_ps(sum_squares, squares);
        }

        // Horizontal sum of the SIMD register
        let sum_array = std::mem::transmute::<__m128, [f32; 4]>(sum_squares);
        let mut total_sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];

        // Process remaining samples (scalar)
        for &sample in &samples[simd_len..] {
            total_sum += sample * sample;
        }

        (total_sum / len as f32).sqrt()
    }

    /// NEON-optimized RMS calculation for ARM processors (processes 4 floats at once)
    /// Returns the root mean square of the audio samples
    ///
    /// # Safety
    /// This function uses unsafe NEON intrinsics but is safe when:
    /// - NEON support is detected via feature detection
    /// - Input slice length is properly validated
    /// - Memory access is within bounds
    #[cfg(target_arch = "aarch64")]
    pub unsafe fn calculate_rms_neon(samples: &[f32]) -> f32 {
        if !Self::is_simd_supported() || samples.len() < 8 {
            return Self::calculate_rms_scalar(samples);
        }

        let mut sum_squares = vdupq_n_f32(0.0);
        let len = samples.len();
        let simd_len = len & !3; // Round down to multiple of 4

        // Process 4 samples at a time
        for i in (0..simd_len).step_by(4) {
            let chunk = vld1q_f32(samples.as_ptr().add(i));
            let squares = vmulq_f32(chunk, chunk);
            sum_squares = vaddq_f32(sum_squares, squares);
        }

        // Horizontal sum of the vector
        let sum_array: [f32; 4] = std::mem::transmute(sum_squares);
        let mut total = sum_array.iter().sum::<f32>();

        // Process remaining samples (scalar)
        for &sample in &samples[simd_len..] {
            total += sample * sample;
        }

        (total / samples.len() as f32).sqrt()
    }

    /// Fallback scalar RMS calculation
    pub fn calculate_rms_scalar(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }

        let sum_squares: f32 = samples.iter().map(|&x| x * x).sum();
        (sum_squares / samples.len() as f32).sqrt()
    }

    /// Safe wrapper for RMS calculation with automatic AVX2/SIMD/scalar fallback
    pub fn calculate_rms(samples: &[f32]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        {
            if Self::is_avx2_supported() && samples.len() >= 16 {
                unsafe { Self::calculate_rms_avx2(samples) }
            } else if Self::is_simd_supported() && samples.len() >= 8 {
                unsafe { Self::calculate_rms_simd(samples) }
            } else {
                Self::calculate_rms_scalar(samples)
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if Self::is_simd_supported() && samples.len() >= 8 {
                unsafe { Self::calculate_rms_neon(samples) }
            } else {
                Self::calculate_rms_scalar(samples)
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self::calculate_rms_scalar(samples)
        }
    }

    /// SIMD-optimized peak detection
    #[cfg(target_arch = "x86_64")]
    pub unsafe fn find_peak_simd(samples: &[f32]) -> f32 {
        if !Self::is_simd_supported() || samples.len() < 8 {
            return Self::find_peak_scalar(samples);
        }

        let mut max_vec = _mm_setzero_ps();
        let len = samples.len();
        let simd_len = len & !3; // Round down to multiple of 4

        // Process 4 samples at a time
        for i in (0..simd_len).step_by(4) {
            let chunk = _mm_loadu_ps(samples.as_ptr().add(i));
            let abs_chunk = _mm_andnot_ps(_mm_set1_ps(-0.0), chunk); // Fast abs using bit manipulation
            max_vec = _mm_max_ps(max_vec, abs_chunk);
        }

        // Find maximum from SIMD register
        let max_array = std::mem::transmute::<__m128, [f32; 4]>(max_vec);
        let mut max_val = max_array[0]
            .max(max_array[1])
            .max(max_array[2])
            .max(max_array[3]);

        // Process remaining samples (scalar)
        for &sample in &samples[simd_len..] {
            max_val = max_val.max(sample.abs());
        }

        max_val
    }

    /// NEON-optimized peak detection for ARM processors
    ///
    /// # Safety
    /// This function uses unsafe NEON intrinsics but is safe when:
    /// - NEON support is detected via feature detection
    /// - Input slice length is properly validated
    /// - Memory access is within bounds
    #[cfg(target_arch = "aarch64")]
    pub unsafe fn find_peak_neon(samples: &[f32]) -> f32 {
        if !Self::is_simd_supported() || samples.len() < 8 {
            return Self::find_peak_scalar(samples);
        }

        let mut max_vec = vdupq_n_f32(0.0);
        let len = samples.len();
        let simd_len = len & !3; // Round down to multiple of 4

        // Process 4 samples at a time
        for i in (0..simd_len).step_by(4) {
            let chunk = vld1q_f32(samples.as_ptr().add(i));
            let abs_chunk = vabsq_f32(chunk);
            max_vec = vmaxq_f32(max_vec, abs_chunk);
        }

        // Find maximum value in the vector
        let max_array: [f32; 4] = std::mem::transmute(max_vec);
        let mut max_val = max_array.iter().fold(0.0f32, |a, &b| a.max(b));

        // Process remaining samples (scalar)
        for &sample in &samples[simd_len..] {
            max_val = max_val.max(sample.abs());
        }

        max_val
    }

    /// Fallback scalar peak detection
    pub fn find_peak_scalar(samples: &[f32]) -> f32 {
        samples.iter().map(|&x| x.abs()).fold(0.0f32, f32::max)
    }

    /// Safe wrapper for peak detection with automatic SIMD/scalar fallback
    pub fn find_peak(samples: &[f32]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        {
            if Self::is_simd_supported() && samples.len() >= 8 {
                unsafe { Self::find_peak_simd(samples) }
            } else {
                Self::find_peak_scalar(samples)
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if Self::is_simd_supported() && samples.len() >= 8 {
                unsafe { Self::find_peak_neon(samples) }
            } else {
                Self::find_peak_scalar(samples)
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self::find_peak_scalar(samples)
        }
    }

    /// SIMD-optimized scalar multiplication (gain application)
    #[cfg(target_arch = "x86_64")]
    pub unsafe fn apply_gain_simd(samples: &mut [f32], gain: f32) {
        if !Self::is_simd_supported() || samples.len() < 8 {
            return Self::apply_gain_scalar(samples, gain);
        }

        let gain_vec = _mm_set1_ps(gain);
        let len = samples.len();
        let simd_len = len & !3; // Round down to multiple of 4

        // Process 4 samples at a time
        for i in (0..simd_len).step_by(4) {
            let chunk = _mm_loadu_ps(samples.as_ptr().add(i));
            let result = _mm_mul_ps(chunk, gain_vec);
            _mm_storeu_ps(samples.as_mut_ptr().add(i), result);
        }

        // Process remaining samples (scalar)
        for sample in &mut samples[simd_len..] {
            *sample *= gain;
        }
    }

    /// NEON-optimized gain application for ARM processors
    ///
    /// # Safety
    /// This function uses unsafe NEON intrinsics but is safe when:
    /// - NEON support is detected via feature detection
    /// - Input slice length is properly validated
    /// - Memory access is within bounds
    #[cfg(target_arch = "aarch64")]
    pub unsafe fn apply_gain_neon(samples: &mut [f32], gain: f32) {
        if !Self::is_simd_supported() || samples.len() < 8 {
            return Self::apply_gain_scalar(samples, gain);
        }

        let gain_vec = vdupq_n_f32(gain);
        let len = samples.len();
        let simd_len = len & !3; // Round down to multiple of 4

        // Process 4 samples at a time
        for i in (0..simd_len).step_by(4) {
            let chunk = vld1q_f32(samples.as_ptr().add(i));
            let result = vmulq_f32(chunk, gain_vec);
            vst1q_f32(samples.as_mut_ptr().add(i), result);
        }

        // Process remaining samples (scalar)
        for sample in &mut samples[simd_len..] {
            *sample *= gain;
        }
    }

    /// Fallback scalar gain application
    pub fn apply_gain_scalar(samples: &mut [f32], gain: f32) {
        for sample in samples {
            *sample *= gain;
        }
    }

    /// Safe wrapper for gain application with automatic SIMD/scalar fallback
    pub fn apply_gain(samples: &mut [f32], gain: f32) {
        #[cfg(target_arch = "x86_64")]
        {
            if Self::is_simd_supported() && samples.len() >= 8 {
                unsafe { Self::apply_gain_simd(samples, gain) }
            } else {
                Self::apply_gain_scalar(samples, gain);
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if Self::is_simd_supported() && samples.len() >= 8 {
                unsafe { Self::apply_gain_neon(samples, gain) }
            } else {
                Self::apply_gain_scalar(samples, gain);
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self::apply_gain_scalar(samples, gain);
        }
    }

    /// SIMD-optimized sample mixing (addition with scaling)
    #[cfg(target_arch = "x86_64")]
    pub unsafe fn mix_samples_simd(target: &mut [f32], source: &[f32], scale: f32) {
        let len = target.len().min(source.len());

        if !Self::is_simd_supported() || len < 8 {
            return Self::mix_samples_scalar(target, source, scale);
        }

        let scale_vec = _mm_set1_ps(scale);
        let simd_len = len & !3; // Round down to multiple of 4

        // Process 4 samples at a time
        for i in (0..simd_len).step_by(4) {
            let target_chunk = _mm_loadu_ps(target.as_ptr().add(i));
            let source_chunk = _mm_loadu_ps(source.as_ptr().add(i));
            let scaled_source = _mm_mul_ps(source_chunk, scale_vec);
            let result = _mm_add_ps(target_chunk, scaled_source);
            _mm_storeu_ps(target.as_mut_ptr().add(i), result);
        }

        // Process remaining samples (scalar)
        for i in simd_len..len {
            target[i] += source[i] * scale;
        }
    }

    /// Fallback scalar sample mixing
    pub fn mix_samples_scalar(target: &mut [f32], source: &[f32], scale: f32) {
        let len = target.len().min(source.len());
        for i in 0..len {
            target[i] += source[i] * scale;
        }
    }

    /// Safe wrapper for sample mixing with automatic SIMD/scalar fallback
    pub fn mix_samples(target: &mut [f32], source: &[f32], scale: f32) {
        #[cfg(target_arch = "x86_64")]
        {
            let len = target.len().min(source.len());
            if Self::is_simd_supported() && len >= 8 {
                unsafe { Self::mix_samples_simd(target, source, scale) }
            } else {
                Self::mix_samples_scalar(target, source, scale);
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Self::mix_samples_scalar(target, source, scale);
        }
    }

    /// SIMD-optimized threshold detection for silence detection
    #[cfg(target_arch = "x86_64")]
    pub unsafe fn count_above_threshold_simd(samples: &[f32], threshold: f32) -> usize {
        if !Self::is_simd_supported() || samples.len() < 8 {
            return Self::count_above_threshold_scalar(samples, threshold);
        }

        let threshold_vec = _mm_set1_ps(threshold);
        let len = samples.len();
        let simd_len = len & !3; // Round down to multiple of 4
        let mut count = 0usize;

        // Process 4 samples at a time
        for i in (0..simd_len).step_by(4) {
            let chunk = _mm_loadu_ps(samples.as_ptr().add(i));
            let abs_chunk = _mm_andnot_ps(_mm_set1_ps(-0.0), chunk); // Fast abs
            let mask = _mm_cmpgt_ps(abs_chunk, threshold_vec);
            let mask_int = _mm_movemask_ps(mask);
            count += mask_int.count_ones() as usize;
        }

        // Process remaining samples (scalar)
        for &sample in &samples[simd_len..] {
            if sample.abs() > threshold {
                count += 1;
            }
        }

        count
    }

    /// Fallback scalar threshold counting
    pub fn count_above_threshold_scalar(samples: &[f32], threshold: f32) -> usize {
        samples.iter().filter(|&&x| x.abs() > threshold).count()
    }

    /// Safe wrapper for threshold counting with automatic SIMD/scalar fallback
    pub fn count_above_threshold(samples: &[f32], threshold: f32) -> usize {
        #[cfg(target_arch = "x86_64")]
        {
            if Self::is_simd_supported() && samples.len() >= 8 {
                unsafe { Self::count_above_threshold_simd(samples, threshold) }
            } else {
                Self::count_above_threshold_scalar(samples, threshold)
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Self::count_above_threshold_scalar(samples, threshold)
        }
    }

    /// SIMD-optimized int16 to float32 conversion with normalization
    #[cfg(target_arch = "x86_64")]
    pub unsafe fn convert_i16_to_f32_avx2(
        input: &[i16],
        output: &mut [f32],
        normalization_factor: f32,
    ) {
        if !Self::is_avx2_supported() || input.len() < 8 || output.len() < input.len() {
            return Self::convert_i16_to_f32_scalar(input, output, normalization_factor);
        }

        let len = input.len().min(output.len());
        let simd_len = len & !7; // Round down to multiple of 8
        let norm_vec = _mm256_set1_ps(normalization_factor);

        // Process 8 samples at a time
        for i in (0..simd_len).step_by(8) {
            // Load 8 i16 values (128-bit)
            let i16_chunk = _mm_loadu_si128(input.as_ptr().add(i) as *const __m128i);

            // Convert to two 4-element i32 vectors (256-bit each)
            let i32_low = _mm256_cvtepi16_epi32(i16_chunk);
            let i32_high = _mm256_cvtepi16_epi32(_mm_unpackhi_epi64(i16_chunk, i16_chunk));

            // Convert to f32 and normalize
            let f32_low = _mm256_mul_ps(_mm256_cvtepi32_ps(i32_low), norm_vec);
            let f32_high = _mm256_mul_ps(_mm256_cvtepi32_ps(i32_high), norm_vec);

            // Store results
            _mm256_storeu_ps(output.as_mut_ptr().add(i), f32_low);
            if i + 4 < len {
                _mm256_storeu_ps(output.as_mut_ptr().add(i + 4), f32_high);
            }
        }

        // Process remaining samples (scalar)
        for i in simd_len..len {
            output[i] = input[i] as f32 * normalization_factor;
        }
    }

    /// SIMD-optimized int16 to float32 conversion with normalization (SSE version)
    #[cfg(target_arch = "x86_64")]
    pub unsafe fn convert_i16_to_f32_simd(
        input: &[i16],
        output: &mut [f32],
        normalization_factor: f32,
    ) {
        if !Self::is_simd_supported() || input.len() < 4 || output.len() < input.len() {
            return Self::convert_i16_to_f32_scalar(input, output, normalization_factor);
        }

        let len = input.len().min(output.len());
        let simd_len = len & !3; // Round down to multiple of 4
        let norm_vec = _mm_set1_ps(normalization_factor);

        // Process 4 samples at a time
        for i in (0..simd_len).step_by(4) {
            // Load 4 i16 values and zero-extend upper bits
            let i16_vals = [input[i], input[i + 1], input[i + 2], input[i + 3]];
            let i32_vals = _mm_set_epi32(
                i16_vals[3] as i32,
                i16_vals[2] as i32,
                i16_vals[1] as i32,
                i16_vals[0] as i32,
            );

            // Convert to f32 and normalize
            let f32_vals = _mm_mul_ps(_mm_cvtepi32_ps(i32_vals), norm_vec);

            // Store result
            _mm_storeu_ps(output.as_mut_ptr().add(i), f32_vals);
        }

        // Process remaining samples (scalar)
        for i in simd_len..len {
            output[i] = input[i] as f32 * normalization_factor;
        }
    }

    /// Scalar fallback for int16 to float32 conversion
    pub fn convert_i16_to_f32_scalar(input: &[i16], output: &mut [f32], normalization_factor: f32) {
        let len = input.len().min(output.len());
        for i in 0..len {
            output[i] = input[i] as f32 * normalization_factor;
        }
    }

    /// Safe wrapper for int16 to float32 conversion with automatic SIMD/scalar fallback
    pub fn convert_i16_to_f32(input: &[i16], output: &mut [f32], normalization_factor: f32) {
        if input.is_empty() || output.is_empty() {
            return;
        }

        #[cfg(target_arch = "x86_64")]
        {
            if Self::is_avx2_supported() && input.len() >= 8 {
                unsafe { Self::convert_i16_to_f32_avx2(input, output, normalization_factor) }
            } else if Self::is_simd_supported() && input.len() >= 4 {
                unsafe { Self::convert_i16_to_f32_simd(input, output, normalization_factor) }
            } else {
                Self::convert_i16_to_f32_scalar(input, output, normalization_factor)
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Self::convert_i16_to_f32_scalar(input, output, normalization_factor)
        }
    }

    /// Optimized bulk sample processing with pre-allocation and SIMD
    pub fn process_audio_samples_bulk<T, F>(input: &[T], processor: F) -> Vec<f32>
    where
        T: Copy,
        F: Fn(T) -> f32,
    {
        if input.is_empty() {
            return Vec::new();
        }

        // Pre-allocate with exact capacity to avoid reallocations
        let mut output = Vec::with_capacity(input.len());

        // Process samples in chunks to improve cache efficiency
        const CHUNK_SIZE: usize = 1024;

        for chunk in input.chunks(CHUNK_SIZE) {
            for &sample in chunk {
                output.push(processor(sample));
            }
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_calculation_scalar() {
        let samples = vec![1.0, -1.0, 2.0, -2.0];
        let rms = SimdAudioProcessor::calculate_rms_scalar(&samples);
        let expected = ((1.0 + 1.0 + 4.0 + 4.0) / 4.0_f32).sqrt();
        assert!((rms - expected).abs() < 1e-6);
    }

    #[test]
    fn test_rms_calculation_consistency() {
        let samples: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin()).collect();
        let rms_scalar = SimdAudioProcessor::calculate_rms_scalar(&samples);
        let rms_auto = SimdAudioProcessor::calculate_rms(&samples);
        assert!((rms_scalar - rms_auto).abs() < 1e-6);
    }

    #[test]
    fn test_peak_detection_scalar() {
        let samples = vec![0.5, -1.5, 0.8, -0.3];
        let peak = SimdAudioProcessor::find_peak_scalar(&samples);
        assert_eq!(peak, 1.5);
    }

    #[test]
    fn test_peak_detection_consistency() {
        let samples: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin()).collect();
        let peak_scalar = SimdAudioProcessor::find_peak_scalar(&samples);
        let peak_auto = SimdAudioProcessor::find_peak(&samples);
        assert!((peak_scalar - peak_auto).abs() < 1e-6);
    }

    #[test]
    fn test_gain_application() {
        let mut samples = vec![1.0, 2.0, 3.0, 4.0];
        let expected = [2.0, 4.0, 6.0, 8.0];
        SimdAudioProcessor::apply_gain(&mut samples, 2.0);
        for (actual, expected) in samples.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_sample_mixing() {
        let mut target = vec![1.0, 2.0, 3.0, 4.0];
        let source = vec![0.5, 1.0, 1.5, 2.0];
        let expected = [2.0, 4.0, 6.0, 8.0]; // target + source * scale
        SimdAudioProcessor::mix_samples(&mut target, &source, 2.0);
        for (actual, expected) in target.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_threshold_counting() {
        let samples = vec![0.1, -0.5, 1.2, -0.2, 0.8];
        let count = SimdAudioProcessor::count_above_threshold(&samples, 0.4);
        assert_eq!(count, 3); // -0.5, 1.2, 0.8 are above threshold
    }

    #[test]
    fn test_simd_support_detection() {
        // This test will pass regardless of SIMD support
        let simd_supported = SimdAudioProcessor::is_simd_supported();
        let avx2_supported = SimdAudioProcessor::is_avx2_supported();
        println!("SIMD support: {simd_supported}, AVX2 support: {avx2_supported}");

        // These should not panic and should return valid boolean values
        // The actual values depend on the CPU running the tests
    }

    #[test]
    fn test_empty_arrays() {
        let empty: Vec<f32> = vec![];
        assert_eq!(SimdAudioProcessor::calculate_rms(&empty), 0.0);
        assert_eq!(SimdAudioProcessor::find_peak(&empty), 0.0);
        assert_eq!(SimdAudioProcessor::count_above_threshold(&empty, 0.5), 0);
    }

    #[test]
    fn test_large_arrays_performance() {
        let large_samples: Vec<f32> = (0..10000).map(|i| (i as f32 * 0.001).sin()).collect();

        // Test that SIMD and scalar versions produce consistent results
        let rms_scalar = SimdAudioProcessor::calculate_rms_scalar(&large_samples);
        let rms_auto = SimdAudioProcessor::calculate_rms(&large_samples);
        assert!((rms_scalar - rms_auto).abs() < 1e-4);

        let peak_scalar = SimdAudioProcessor::find_peak_scalar(&large_samples);
        let peak_auto = SimdAudioProcessor::find_peak(&large_samples);
        assert!((peak_scalar - peak_auto).abs() < 1e-6);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_avx2_rms_calculation() {
        let samples = vec![
            1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 3.5, -3.5,
        ];

        // Test AVX2 implementation against scalar implementation
        let rms_scalar = SimdAudioProcessor::calculate_rms_scalar(&samples);

        // Only test AVX2 if the CPU supports it
        if SimdAudioProcessor::is_avx2_supported() {
            let rms_avx2 = unsafe { SimdAudioProcessor::calculate_rms_avx2(&samples) };
            assert!(
                (rms_scalar - rms_avx2).abs() < 1e-6,
                "AVX2 RMS calculation should match scalar implementation"
            );
        }

        // Test the automatic selection wrapper
        let rms_auto = SimdAudioProcessor::calculate_rms(&samples);
        assert!(
            (rms_scalar - rms_auto).abs() < 1e-6,
            "Automatic RMS calculation should match scalar implementation"
        );
    }
}

//! Generic scalar implementations for SIMD operations
//!
//! Provides fallback implementations that work on all platforms
//! when SIMD instructions are not available.

/// Scalar implementation of vector addition
pub fn add_f32_scalar(a: &[f32], b: &[f32], output: &mut [f32]) {
    let len = a.len().min(b.len()).min(output.len());
    for i in 0..len {
        output[i] = a[i] + b[i];
    }
}

/// Scalar implementation of vector multiplication
pub fn mul_f32_scalar(a: &[f32], b: &[f32], output: &mut [f32]) {
    let len = a.len().min(b.len()).min(output.len());
    for i in 0..len {
        output[i] = a[i] * b[i];
    }
}

/// Scalar implementation of scalar multiplication
pub fn mul_scalar_f32_scalar(input: &[f32], scalar: f32, output: &mut [f32]) {
    let len = input.len().min(output.len());
    for i in 0..len {
        output[i] = input[i] * scalar;
    }
}

/// Scalar implementation of fused multiply-add
pub fn fma_f32_scalar(a: &[f32], b: &[f32], c: &[f32], output: &mut [f32]) {
    let len = a.len().min(b.len()).min(c.len()).min(output.len());
    for i in 0..len {
        output[i] = a[i] * b[i] + c[i];
    }
}

/// Scalar implementation of sum reduction
pub fn sum_f32_scalar(input: &[f32]) -> f32 {
    input.iter().sum()
}

/// Scalar implementation of dot product
pub fn dot_product_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let mut sum = 0.0;
    for i in 0..len {
        sum += a[i] * b[i];
    }
    sum
}

/// Scalar implementation of RMS calculation
pub fn rms_f32_scalar(input: &[f32]) -> f32 {
    if input.is_empty() {
        return 0.0;
    }

    let sum_squares: f32 = input.iter().map(|&x| x * x).sum();
    (sum_squares / input.len() as f32).sqrt()
}

/// Scalar implementation of peak detection
pub fn find_peak_f32_scalar(input: &[f32]) -> (usize, f32) {
    if input.is_empty() {
        return (0, 0.0);
    }

    let mut max_idx = 0;
    let mut max_val = input[0].abs();

    for (i, &val) in input.iter().enumerate().skip(1) {
        let abs_val = val.abs();
        if abs_val > max_val {
            max_val = abs_val;
            max_idx = i;
        }
    }

    (max_idx, max_val)
}

/// Scalar implementation of normalization
pub fn normalize_f32_scalar(input: &[f32], output: &mut [f32], target_peak: f32) {
    let len = input.len().min(output.len());
    if len == 0 {
        return;
    }

    // Find current peak
    let current_peak = input.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));

    if current_peak > 0.0 {
        let scale = target_peak / current_peak;
        mul_scalar_f32_scalar(input, scale, output);
    } else {
        // Silent input, just copy
        output[..len].copy_from_slice(&input[..len]);
    }
}

/// Scalar implementation of DC removal
pub fn remove_dc_f32_scalar(input: &[f32], output: &mut [f32]) {
    let len = input.len().min(output.len());
    if len == 0 {
        return;
    }

    // Calculate mean
    let mean = sum_f32_scalar(input) / len as f32;

    // Subtract mean from each sample
    for i in 0..len {
        output[i] = input[i] - mean;
    }
}

/// Scalar implementation of high-pass filter (simple one-pole)
pub fn highpass_filter_f32_scalar(
    input: &[f32],
    output: &mut [f32],
    cutoff: f32,
    sample_rate: f32,
) {
    let len = input.len().min(output.len());
    if len == 0 {
        return;
    }

    // Calculate filter coefficient
    let rc = 1.0 / (2.0 * std::f32::consts::PI * cutoff);
    let dt = 1.0 / sample_rate;
    let alpha = rc / (rc + dt);

    output[0] = input[0];

    for i in 1..len {
        output[i] = alpha * (output[i - 1] + input[i] - input[i - 1]);
    }
}

/// Scalar implementation of low-pass filter (simple one-pole)
pub fn lowpass_filter_f32_scalar(input: &[f32], output: &mut [f32], cutoff: f32, sample_rate: f32) {
    let len = input.len().min(output.len());
    if len == 0 {
        return;
    }

    // Calculate filter coefficient
    let rc = 1.0 / (2.0 * std::f32::consts::PI * cutoff);
    let dt = 1.0 / sample_rate;
    let alpha = dt / (rc + dt);

    output[0] = input[0];

    for i in 1..len {
        output[i] = output[i - 1] + alpha * (input[i] - output[i - 1]);
    }
}

/// Scalar implementation of interleaved stereo to mono conversion
pub fn stereo_to_mono_f32_scalar(stereo_input: &[f32], mono_output: &mut [f32]) {
    let mono_len = stereo_input.len() / 2;
    let output_len = mono_output.len().min(mono_len);

    for i in 0..output_len {
        let left = stereo_input[i * 2];
        let right = stereo_input[i * 2 + 1];
        mono_output[i] = (left + right) * 0.5;
    }
}

/// Scalar implementation of mono to interleaved stereo conversion
pub fn mono_to_stereo_f32_scalar(mono_input: &[f32], stereo_output: &mut [f32]) {
    let stereo_len = stereo_output.len() / 2;
    let input_len = mono_input.len().min(stereo_len);

    for i in 0..input_len {
        let sample = mono_input[i];
        stereo_output[i * 2] = sample; // Left
        stereo_output[i * 2 + 1] = sample; // Right
    }
}

/// Scalar implementation of audio mixing with separate gains
pub fn mix_audio_f32_scalar(a: &[f32], b: &[f32], output: &mut [f32], gain_a: f32, gain_b: f32) {
    let len = a.len().min(b.len()).min(output.len());
    for i in 0..len {
        output[i] = a[i] * gain_a + b[i] * gain_b;
    }
}

/// Scalar implementation of fade in/out
pub fn apply_fade_f32_scalar(
    input: &[f32],
    output: &mut [f32],
    fade_in_samples: usize,
    fade_out_samples: usize,
) {
    let len = input.len().min(output.len());
    if len == 0 {
        return;
    }

    for i in 0..len {
        let mut gain = 1.0;

        // Fade in
        if i < fade_in_samples {
            gain *= i as f32 / fade_in_samples as f32;
        }

        // Fade out
        if i >= len - fade_out_samples {
            let remaining = len - i;
            gain *= remaining as f32 / fade_out_samples as f32;
        }

        output[i] = input[i] * gain;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_f32_scalar() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![0.5, 1.5, 2.5, 3.5];
        let mut output = vec![0.0; 4];

        add_f32_scalar(&a, &b, &mut output);
        assert_eq!(output, vec![1.5, 3.5, 5.5, 7.5]);
    }

    #[test]
    fn test_mul_f32_scalar() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![0.5, 1.5, 2.5, 3.5];
        let mut output = vec![0.0; 4];

        mul_f32_scalar(&a, &b, &mut output);
        assert_eq!(output, vec![0.5, 3.0, 7.5, 14.0]);
    }

    #[test]
    fn test_mul_scalar_f32_scalar() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];

        mul_scalar_f32_scalar(&input, 2.0, &mut output);
        assert_eq!(output, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_fma_f32_scalar() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![0.5, 1.5, 2.5, 3.5];
        let c = vec![0.1, 0.2, 0.3, 0.4];
        let mut output = vec![0.0; 4];

        fma_f32_scalar(&a, &b, &c, &mut output);
        assert_eq!(output, vec![0.6, 3.2, 7.8, 14.4]);
    }

    #[test]
    fn test_sum_f32_scalar() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let sum = sum_f32_scalar(&input);
        assert_eq!(sum, 10.0);
    }

    #[test]
    fn test_dot_product_f32_scalar() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![0.5, 1.5, 2.5, 3.5];
        let dot = dot_product_f32_scalar(&a, &b);
        assert_eq!(dot, 25.0);
    }

    #[test]
    fn test_rms_f32_scalar() {
        let input = vec![1.0, -1.0, 1.0, -1.0];
        let rms = rms_f32_scalar(&input);
        assert_eq!(rms, 1.0);
    }

    #[test]
    fn test_find_peak_f32_scalar() {
        let input = vec![0.1, -0.5, 0.3, -0.8, 0.2];
        let (idx, peak) = find_peak_f32_scalar(&input);
        assert_eq!(idx, 3);
        assert_eq!(peak, 0.8);
    }

    #[test]
    fn test_normalize_f32_scalar() {
        let input = vec![0.1, -0.5, 0.3, -0.8, 0.2];
        let mut output = vec![0.0; 5];

        normalize_f32_scalar(&input, &mut output, 1.0);

        // Check that the peak is now 1.0
        let (_, peak) = find_peak_f32_scalar(&output);
        assert!((peak - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_remove_dc_f32_scalar() {
        let input = vec![1.1, 2.1, 3.1, 4.1]; // DC offset of +1.0
        let mut output = vec![0.0; 4];

        remove_dc_f32_scalar(&input, &mut output);

        let mean = sum_f32_scalar(&output) / 4.0;
        assert!(mean.abs() < 1e-6); // Should be close to zero
    }

    #[test]
    fn test_stereo_to_mono_f32_scalar() {
        let stereo = vec![1.0, -1.0, 2.0, -2.0, 3.0, -3.0];
        let mut mono = vec![0.0; 3];

        stereo_to_mono_f32_scalar(&stereo, &mut mono);
        assert_eq!(mono, vec![0.0, 0.0, 0.0]); // Should average to zero
    }

    #[test]
    fn test_mono_to_stereo_f32_scalar() {
        let mono = vec![1.0, 2.0, 3.0];
        let mut stereo = vec![0.0; 6];

        mono_to_stereo_f32_scalar(&mono, &mut stereo);
        assert_eq!(stereo, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
    }

    #[test]
    fn test_mix_audio_f32_scalar() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![0.5, 1.5, 2.5, 3.5];
        let mut output = vec![0.0; 4];

        mix_audio_f32_scalar(&a, &b, &mut output, 0.8, 0.2);

        // Expected: [1.0*0.8 + 0.5*0.2, 2.0*0.8 + 1.5*0.2, ...]
        let expected = [0.9, 1.9, 2.9, 3.9];
        for (actual, expected) in output.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_apply_fade_f32_scalar() {
        let input = vec![1.0; 10];
        let mut output = vec![0.0; 10];

        apply_fade_f32_scalar(&input, &mut output, 3, 3);

        // Check fade in
        assert!(output[0] < output[1]);
        assert!(output[1] < output[2]);

        // Check middle (should be close to 1.0)
        assert!((output[4] - 1.0).abs() < 0.1);

        // Check fade out
        assert!(output[7] > output[8]);
        assert!(output[8] > output[9]);
    }
}

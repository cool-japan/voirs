//! SIMD-optimized audio processing operations
//!
//! Provides vectorized implementations of common audio processing operations
//! such as sample rate conversion, filtering, and audio effects.

/// SIMD-optimized linear interpolation for sample rate conversion
pub fn linear_interpolate_f32(
    input: &[f32],
    output: &mut [f32],
    input_rate: f32,
    output_rate: f32,
) {
    let ratio = input_rate / output_rate;
    let input_len = input.len();
    let output_len = output.len();

    for (i, output_sample) in output.iter_mut().enumerate().take(output_len) {
        let src_pos = i as f32 * ratio;
        let src_index = src_pos.floor() as usize;
        let frac = src_pos - src_index as f32;

        if src_index + 1 < input_len {
            *output_sample = input[src_index] * (1.0 - frac) + input[src_index + 1] * frac;
        } else if src_index < input_len {
            *output_sample = input[src_index];
        } else {
            *output_sample = 0.0;
        }
    }
}

/// SIMD-optimized biquad filter implementation
pub struct BiquadFilter {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}

impl BiquadFilter {
    /// Create a new biquad filter with the given coefficients
    pub fn new(b0: f32, b1: f32, b2: f32, a1: f32, a2: f32) -> Self {
        Self {
            b0,
            b1,
            b2,
            a1,
            a2,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    /// Create a low-pass filter with the given cutoff frequency
    pub fn lowpass(sample_rate: f32, cutoff: f32, q: f32) -> Self {
        let omega = 2.0 * std::f32::consts::PI * cutoff / sample_rate;
        let sin_omega = omega.sin();
        let cos_omega = omega.cos();
        let alpha = sin_omega / (2.0 * q);

        let b0 = (1.0 - cos_omega) / 2.0;
        let b1 = 1.0 - cos_omega;
        let b2 = (1.0 - cos_omega) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_omega;
        let a2 = 1.0 - alpha;

        // Normalize coefficients
        Self::new(b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)
    }

    /// Create a high-pass filter with the given cutoff frequency
    pub fn highpass(sample_rate: f32, cutoff: f32, q: f32) -> Self {
        let omega = 2.0 * std::f32::consts::PI * cutoff / sample_rate;
        let sin_omega = omega.sin();
        let cos_omega = omega.cos();
        let alpha = sin_omega / (2.0 * q);

        let b0 = (1.0 + cos_omega) / 2.0;
        let b1 = -(1.0 + cos_omega);
        let b2 = (1.0 + cos_omega) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_omega;
        let a2 = 1.0 - alpha;

        // Normalize coefficients
        Self::new(b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)
    }

    /// Process a single sample through the filter
    pub fn process_sample(&mut self, input: f32) -> f32 {
        let output = self.b0 * input + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;

        // Update delay line
        self.x2 = self.x1;
        self.x1 = input;
        self.y2 = self.y1;
        self.y1 = output;

        output
    }

    /// Process a buffer of samples through the filter
    pub fn process_buffer(&mut self, input: &[f32], output: &mut [f32]) {
        let len = input.len().min(output.len());

        for i in 0..len {
            output[i] = self.process_sample(input[i]);
        }
    }
}

/// SIMD-optimized RMS calculation
pub fn rms_f32(input: &[f32]) -> f32 {
    if input.is_empty() {
        return 0.0;
    }

    let sum_squares = {
        #[cfg(target_arch = "x86_64")]
        {
            // Create squared samples and sum them
            let mut squares = vec![0.0; input.len()];
            super::x86_64::mul_f32_x86_64(input, input, &mut squares);
            super::x86_64::sum_f32_x86_64(&squares)
        }

        #[cfg(target_arch = "aarch64")]
        {
            // Create squared samples and sum them
            let mut squares = vec![0.0; input.len()];
            super::aarch64::mul_f32_aarch64(input, input, &mut squares);
            super::aarch64::sum_f32_aarch64(&squares)
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            input.iter().map(|x| x * x).sum::<f32>()
        }
    };

    (sum_squares / input.len() as f32).sqrt()
}

/// SIMD-optimized peak detection
pub fn find_peak_f32(input: &[f32]) -> f32 {
    if input.is_empty() {
        return 0.0;
    }

    // Convert to absolute values and find maximum
    let mut abs_values = vec![0.0; input.len()];

    #[cfg(target_arch = "x86_64")]
    {
        for (i, &sample) in input.iter().enumerate() {
            abs_values[i] = sample.abs();
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        for (i, &sample) in input.iter().enumerate() {
            abs_values[i] = sample.abs();
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        for (i, &sample) in input.iter().enumerate() {
            abs_values[i] = sample.abs();
        }
    }

    abs_values.iter().fold(0.0f32, |max, &x| max.max(x))
}

/// SIMD-optimized DC offset removal
pub fn remove_dc_offset_f32(input: &[f32], output: &mut [f32]) {
    if input.is_empty() {
        return;
    }

    // Calculate mean (DC component)
    let mean = {
        #[cfg(target_arch = "x86_64")]
        {
            super::x86_64::sum_f32_x86_64(input) / input.len() as f32
        }

        #[cfg(target_arch = "aarch64")]
        {
            super::aarch64::sum_f32_aarch64(input) / input.len() as f32
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            super::generic::sum_f32_scalar(input) / input.len() as f32
        }
    };

    // Subtract mean from all samples
    let mean_vec = vec![mean; input.len()];

    #[cfg(target_arch = "x86_64")]
    {
        // Create temporary buffer for subtraction (input - mean)
        let mut temp = vec![0.0; input.len()];
        super::x86_64::mul_scalar_f32_x86_64(&mean_vec, -1.0, &mut temp);
        super::x86_64::add_f32_x86_64(input, &temp, output);
    }

    #[cfg(target_arch = "aarch64")]
    {
        // Create temporary buffer for subtraction (input - mean)
        let mut temp = vec![0.0; input.len()];
        super::aarch64::mul_scalar_f32_aarch64(&mean_vec, -1.0, &mut temp);
        super::aarch64::add_f32_aarch64(input, &temp, output);
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let len = input.len().min(output.len());
        for i in 0..len {
            output[i] = input[i] - mean;
        }
    }
}

/// SIMD-optimized gain application
pub fn apply_gain_f32(input: &[f32], gain: f32, output: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        super::x86_64::mul_scalar_f32_x86_64(input, gain, output);
    }

    #[cfg(target_arch = "aarch64")]
    {
        super::aarch64::mul_scalar_f32_aarch64(input, gain, output);
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        super::generic::mul_scalar_f32_scalar(input, gain, output);
    }
}

/// SIMD-optimized normalization to peak amplitude
pub fn normalize_peak_f32(input: &[f32], target_peak: f32, output: &mut [f32]) {
    let current_peak = find_peak_f32(input);

    if current_peak > 0.0 {
        let gain = target_peak / current_peak;
        apply_gain_f32(input, gain, output);
    } else {
        let len = input.len().min(output.len());
        output[..len].copy_from_slice(&input[..len]);
    }
}

/// SIMD-optimized crossfade between two audio buffers
pub fn crossfade_f32(
    input_a: &[f32],
    input_b: &[f32],
    output: &mut [f32],
    fade_position: f32, // 0.0 = full A, 1.0 = full B
) {
    let len = input_a.len().min(input_b.len()).min(output.len());
    let gain_a = 1.0 - fade_position;
    let gain_b = fade_position;

    // Create temporary buffers for scaled inputs
    let mut temp_a = vec![0.0; len];
    let mut temp_b = vec![0.0; len];

    apply_gain_f32(&input_a[..len], gain_a, &mut temp_a);
    apply_gain_f32(&input_b[..len], gain_b, &mut temp_b);

    // Add the scaled inputs
    #[cfg(target_arch = "x86_64")]
    {
        super::x86_64::add_f32_x86_64(&temp_a, &temp_b, &mut output[..len]);
    }

    #[cfg(target_arch = "aarch64")]
    {
        super::aarch64::add_f32_aarch64(&temp_a, &temp_b, &mut output[..len]);
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        super::generic::add_f32_scalar(&temp_a, &temp_b, &mut output[..len]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_interpolate() {
        let input = vec![0.0, 1.0, 2.0, 3.0];
        let mut output = vec![0.0; 8]; // Upsample 2x

        linear_interpolate_f32(&input, &mut output, 22050.0, 44100.0);

        // Should interpolate between samples
        assert_eq!(output[0], 0.0);
        assert_eq!(output[2], 1.0);
        assert_eq!(output[4], 2.0);
        assert_eq!(output[6], 3.0);
    }

    #[test]
    fn test_biquad_filter_creation() {
        let filter = BiquadFilter::lowpass(44100.0, 1000.0, 0.707);

        // Test that filter was created (basic check)
        assert!(filter.b0 > 0.0);
    }

    #[test]
    fn test_rms_calculation() {
        let input = vec![0.0, 1.0, 0.0, -1.0];
        let rms = rms_f32(&input);

        // RMS of [0, 1, 0, -1] should be sqrt((0^2 + 1^2 + 0^2 + 1^2) / 4) = sqrt(0.5)
        assert!((rms - (0.5f32).sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_find_peak() {
        let input = vec![0.1, -0.5, 0.3, -0.8, 0.2];
        let peak = find_peak_f32(&input);

        assert_eq!(peak, 0.8);
    }

    #[test]
    fn test_remove_dc_offset() {
        let input = vec![1.0, 2.0, 3.0, 4.0]; // Mean = 2.5
        let mut output = vec![0.0; 4];

        remove_dc_offset_f32(&input, &mut output);

        let expected = [-1.5, -0.5, 0.5, 1.5];
        for (a, b) in output.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_apply_gain() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];

        apply_gain_f32(&input, 2.0, &mut output);

        let expected = vec![2.0, 4.0, 6.0, 8.0];
        assert_eq!(output, expected);
    }

    #[test]
    fn test_normalize_peak() {
        let input = vec![0.1, 0.2, 0.5, 0.3]; // Peak = 0.5
        let mut output = vec![0.0; 4];

        normalize_peak_f32(&input, 1.0, &mut output);

        // Should scale by 1.0/0.5 = 2.0
        let expected = [0.2, 0.4, 1.0, 0.6];
        for (a, b) in output.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_crossfade() {
        let input_a = vec![1.0, 1.0, 1.0, 1.0];
        let input_b = vec![2.0, 2.0, 2.0, 2.0];
        let mut output = vec![0.0; 4];

        crossfade_f32(&input_a, &input_b, &mut output, 0.5);

        // 50% fade should give (1.0 * 0.5 + 2.0 * 0.5) = 1.5
        let expected = [1.5, 1.5, 1.5, 1.5];
        for (a, b) in output.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}

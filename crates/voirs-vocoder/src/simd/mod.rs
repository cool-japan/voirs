//! SIMD acceleration for audio processing operations
//!
//! Provides vectorized implementations of audio processing operations
//! using platform-specific SIMD instructions (AVX2, AVX-512, NEON).

#[cfg(target_arch = "aarch64")]
pub mod aarch64;
#[cfg(target_arch = "x86_64")]
pub mod x86_64;

pub mod audio_ops;
pub mod convolution;
pub mod generic;

/// SIMD feature detection and capability checking
#[derive(Debug, Clone, Copy)]
pub struct SimdCapabilities {
    pub sse2: bool,
    pub sse3: bool,
    pub sse4_1: bool,
    pub sse4_2: bool,
    pub avx: bool,
    pub avx2: bool,
    pub avx512f: bool,
    pub fma: bool,
    pub neon: bool,
}

impl SimdCapabilities {
    /// Detect available SIMD capabilities at runtime
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                sse2: is_x86_feature_detected!("sse2"),
                sse3: is_x86_feature_detected!("sse3"),
                sse4_1: is_x86_feature_detected!("sse4.1"),
                sse4_2: is_x86_feature_detected!("sse4.2"),
                avx: is_x86_feature_detected!("avx"),
                avx2: is_x86_feature_detected!("avx2"),
                avx512f: is_x86_feature_detected!("avx512f"),
                fma: is_x86_feature_detected!("fma"),
                neon: false,
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            Self {
                sse2: false,
                sse3: false,
                sse4_1: false,
                sse4_2: false,
                avx: false,
                avx2: false,
                avx512f: false,
                fma: false,
                neon: std::arch::is_aarch64_feature_detected!("neon"),
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self {
                sse2: false,
                sse3: false,
                sse4_1: false,
                sse4_2: false,
                avx: false,
                avx2: false,
                avx512f: false,
                fma: false,
                neon: false,
            }
        }
    }

    /// Get the best available SIMD level
    pub fn best_level(&self) -> SimdLevel {
        if self.avx512f {
            SimdLevel::Avx512
        } else if self.avx2 {
            SimdLevel::Avx2
        } else if self.avx {
            SimdLevel::Avx
        } else if self.sse4_2 {
            SimdLevel::Sse42
        } else if self.sse2 {
            SimdLevel::Sse2
        } else if self.neon {
            SimdLevel::Neon
        } else {
            SimdLevel::Scalar
        }
    }
}

/// Available SIMD instruction levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SimdLevel {
    Scalar,
    Sse2,
    Sse42,
    Avx,
    Avx2,
    Avx512,
    Neon,
}

/// Trait for SIMD-accelerated operations
pub trait SimdOps {
    /// Add two f32 slices with SIMD acceleration
    fn add_f32(&self, a: &[f32], b: &[f32], output: &mut [f32]);

    /// Multiply two f32 slices with SIMD acceleration
    fn mul_f32(&self, a: &[f32], b: &[f32], output: &mut [f32]);

    /// Multiply slice by scalar with SIMD acceleration
    fn mul_scalar_f32(&self, input: &[f32], scalar: f32, output: &mut [f32]);

    /// Fused multiply-add operation (a * b + c)
    fn fma_f32(&self, a: &[f32], b: &[f32], c: &[f32], output: &mut [f32]);

    /// Sum all elements in slice
    fn sum_f32(&self, input: &[f32]) -> f32;

    /// Dot product of two slices
    fn dot_product_f32(&self, a: &[f32], b: &[f32]) -> f32;

    /// Apply gain to audio samples
    fn apply_gain(&self, samples: &mut [f32], gain: f32);

    /// Mix two audio buffers
    fn mix_audio(&self, a: &[f32], b: &[f32], output: &mut [f32], gain_a: f32, gain_b: f32);

    /// Fast convolution using SIMD
    fn convolve(&self, signal: &[f32], kernel: &[f32], output: &mut [f32]);

    /// Enhanced audio-specific operations
    /// Normalize audio samples to target RMS level
    fn normalize_rms(&self, samples: &mut [f32], target_rms: f32);

    /// Apply soft clipping to prevent hard clipping artifacts
    fn soft_clip(&self, samples: &mut [f32], threshold: f32);

    /// Calculate RMS (Root Mean Square) value efficiently
    fn calculate_rms(&self, samples: &[f32]) -> f32;

    /// Apply window function (Hann, Hamming, etc.)
    fn apply_window(&self, samples: &mut [f32], window: &[f32]);

    /// Optimized mel spectrogram normalization
    fn normalize_mel_spectrogram(&self, mel_data: &mut [f32], mean: f32, std_dev: f32);
}

/// SIMD processor that dispatches to the best available implementation
pub struct SimdProcessor {
    capabilities: SimdCapabilities,
    level: SimdLevel,
}

impl SimdProcessor {
    /// Create new SIMD processor with automatic capability detection
    pub fn new() -> Self {
        let capabilities = SimdCapabilities::detect();
        let level = capabilities.best_level();

        Self {
            capabilities,
            level,
        }
    }

    /// Create processor with specific SIMD level (for testing)
    pub fn with_level(level: SimdLevel) -> Self {
        Self {
            capabilities: SimdCapabilities::detect(),
            level,
        }
    }

    /// Get available capabilities
    pub fn capabilities(&self) -> SimdCapabilities {
        self.capabilities
    }

    /// Get current SIMD level
    pub fn level(&self) -> SimdLevel {
        self.level
    }
}

impl Default for SimdProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl SimdOps for SimdProcessor {
    fn add_f32(&self, a: &[f32], b: &[f32], output: &mut [f32]) {
        // Intelligent dispatch based on detected capabilities and data size
        let len = a.len().min(b.len()).min(output.len());

        #[cfg(target_arch = "x86_64")]
        {
            // Use highest available SIMD level for better performance
            match self.level {
                SimdLevel::Avx512 | SimdLevel::Avx2 | SimdLevel::Avx if len >= 32 => {
                    x86_64::add_f32_x86_64(a, b, output);
                }
                _ => {
                    // For small arrays or lower SIMD levels, use generic implementation
                    generic::add_f32_scalar(&a[..len], &b[..len], &mut output[..len]);
                }
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            match self.level {
                SimdLevel::Neon if len >= 16 => {
                    aarch64::add_f32_aarch64(a, b, output);
                }
                _ => {
                    generic::add_f32_scalar(&a[..len], &b[..len], &mut output[..len]);
                }
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            generic::add_f32_scalar(&a[..len], &b[..len], &mut output[..len]);
        }
    }

    fn mul_f32(&self, a: &[f32], b: &[f32], output: &mut [f32]) {
        let len = a.len().min(b.len()).min(output.len());

        #[cfg(target_arch = "x86_64")]
        {
            match self.level {
                SimdLevel::Avx512 | SimdLevel::Avx2 | SimdLevel::Avx if len >= 32 => {
                    x86_64::mul_f32_x86_64(a, b, output);
                }
                _ => {
                    generic::mul_f32_scalar(&a[..len], &b[..len], &mut output[..len]);
                }
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            match self.level {
                SimdLevel::Neon if len >= 16 => {
                    aarch64::mul_f32_aarch64(a, b, output);
                }
                _ => {
                    generic::mul_f32_scalar(&a[..len], &b[..len], &mut output[..len]);
                }
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            generic::mul_f32_scalar(&a[..len], &b[..len], &mut output[..len]);
        }
    }

    fn mul_scalar_f32(&self, input: &[f32], scalar: f32, output: &mut [f32]) {
        let len = input.len().min(output.len());

        #[cfg(target_arch = "x86_64")]
        {
            match self.level {
                SimdLevel::Avx512 | SimdLevel::Avx2 | SimdLevel::Avx if len >= 32 => {
                    x86_64::mul_scalar_f32_x86_64(input, scalar, output);
                }
                _ => {
                    generic::mul_scalar_f32_scalar(&input[..len], scalar, &mut output[..len]);
                }
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            match self.level {
                SimdLevel::Neon if len >= 16 => {
                    aarch64::mul_scalar_f32_aarch64(input, scalar, output);
                }
                _ => {
                    generic::mul_scalar_f32_scalar(&input[..len], scalar, &mut output[..len]);
                }
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            generic::mul_scalar_f32_scalar(&input[..len], scalar, &mut output[..len]);
        }
    }

    fn fma_f32(&self, a: &[f32], b: &[f32], c: &[f32], output: &mut [f32]) {
        let len = a.len().min(b.len()).min(c.len()).min(output.len());

        #[cfg(target_arch = "x86_64")]
        {
            match self.level {
                SimdLevel::Avx512 | SimdLevel::Avx2 | SimdLevel::Avx
                    if len >= 32 && self.capabilities.fma =>
                {
                    x86_64::fma_f32_x86_64(a, b, c, output);
                }
                _ => {
                    generic::fma_f32_scalar(&a[..len], &b[..len], &c[..len], &mut output[..len]);
                }
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            match self.level {
                SimdLevel::Neon if len >= 16 => {
                    aarch64::fma_f32_aarch64(a, b, c, output);
                }
                _ => {
                    generic::fma_f32_scalar(&a[..len], &b[..len], &c[..len], &mut output[..len]);
                }
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            generic::fma_f32_scalar(&a[..len], &b[..len], &c[..len], &mut output[..len]);
        }
    }

    fn sum_f32(&self, input: &[f32]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        {
            match self.level {
                SimdLevel::Avx512 | SimdLevel::Avx2 | SimdLevel::Avx if input.len() >= 32 => {
                    x86_64::sum_f32_x86_64(input)
                }
                _ => generic::sum_f32_scalar(input),
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            match self.level {
                SimdLevel::Neon if input.len() >= 16 => aarch64::sum_f32_aarch64(input),
                _ => generic::sum_f32_scalar(input),
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            generic::sum_f32_scalar(input)
        }
    }

    fn dot_product_f32(&self, a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());

        #[cfg(target_arch = "x86_64")]
        {
            match self.level {
                SimdLevel::Avx512 | SimdLevel::Avx2 | SimdLevel::Avx if len >= 32 => {
                    x86_64::dot_product_f32_x86_64(a, b)
                }
                _ => generic::dot_product_f32_scalar(&a[..len], &b[..len]),
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            match self.level {
                SimdLevel::Neon if len >= 16 => aarch64::dot_product_f32_aarch64(a, b),
                _ => generic::dot_product_f32_scalar(&a[..len], &b[..len]),
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            generic::dot_product_f32_scalar(&a[..len], &b[..len])
        }
    }

    fn apply_gain(&self, samples: &mut [f32], gain: f32) {
        let input_copy = samples.to_vec();
        self.mul_scalar_f32(&input_copy, gain, samples);
    }

    fn mix_audio(&self, a: &[f32], b: &[f32], output: &mut [f32], gain_a: f32, gain_b: f32) {
        let len = a.len().min(b.len()).min(output.len());

        // Use crossfade from audio_ops module
        audio_ops::crossfade_f32(
            &a[..len],
            &b[..len],
            &mut output[..len],
            gain_b / (gain_a + gain_b),
        );
    }

    fn convolve(&self, signal: &[f32], kernel: &[f32], output: &mut [f32]) {
        // Use conv1d from convolution module with default stride and no padding
        convolution::conv1d_f32(signal, kernel, output, 1, 0);
    }

    /// Enhanced audio-specific operations
    fn normalize_rms(&self, samples: &mut [f32], target_rms: f32) {
        if samples.is_empty() || target_rms <= 0.0 {
            return;
        }

        // Calculate current RMS
        let current_rms = self.calculate_rms(samples);

        if current_rms > 1e-10 {
            // Avoid division by very small numbers
            let gain = target_rms / current_rms;
            let input_copy = samples.to_vec();
            self.mul_scalar_f32(&input_copy, gain, samples);
        }
    }

    fn soft_clip(&self, samples: &mut [f32], threshold: f32) {
        if threshold <= 0.0 {
            return;
        }

        let inv_threshold = 1.0 / threshold;

        // Apply soft clipping: tanh(x / threshold) * threshold
        for sample in samples.iter_mut() {
            if sample.abs() > threshold {
                *sample = (*sample * inv_threshold).tanh() * threshold;
            }
        }
    }

    fn calculate_rms(&self, samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }

        // Use SIMD for dot product to calculate sum of squares
        let sum_of_squares = self.dot_product_f32(samples, samples);
        (sum_of_squares / samples.len() as f32).sqrt()
    }

    fn apply_window(&self, samples: &mut [f32], window: &[f32]) {
        let len = samples.len().min(window.len());
        let input_copy = samples[..len].to_vec();
        self.mul_f32(&input_copy, &window[..len], &mut samples[..len]);
    }

    fn normalize_mel_spectrogram(&self, mel_data: &mut [f32], mean: f32, std_dev: f32) {
        if std_dev <= 0.0 {
            return;
        }

        let inv_std = 1.0 / std_dev;

        // Optimized normalization: (x - mean) / std_dev
        // This is equivalent to: x * (1/std_dev) - mean * (1/std_dev)
        let mean_factor = mean * inv_std;

        for value in mel_data.iter_mut() {
            *value = *value * inv_std - mean_factor;
        }
    }
}

/// Convenience function to get a global SIMD processor
pub fn get_simd_processor() -> &'static SimdProcessor {
    use std::sync::OnceLock;
    static PROCESSOR: OnceLock<SimdProcessor> = OnceLock::new();
    PROCESSOR.get_or_init(SimdProcessor::new)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_capability_detection() {
        let caps = SimdCapabilities::detect();
        let level = caps.best_level();

        // Should at least support scalar operations
        assert!(level >= SimdLevel::Scalar);

        // Print detected capabilities for debugging
        println!("Detected SIMD capabilities: {caps:?}");
        println!("Best SIMD level: {level:?}");
    }

    #[test]
    fn test_simd_processor_creation() {
        let processor = SimdProcessor::new();
        assert!(processor.level() >= SimdLevel::Scalar);
    }

    #[test]
    fn test_basic_operations() {
        let processor = SimdProcessor::new();

        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![0.5, 1.5, 2.5, 3.5];
        let mut output = vec![0.0; 4];

        // Test addition
        processor.add_f32(&a, &b, &mut output);
        assert_eq!(output, vec![1.5, 3.5, 5.5, 7.5]);

        // Test multiplication
        processor.mul_f32(&a, &b, &mut output);
        assert_eq!(output, vec![0.5, 3.0, 7.5, 14.0]);

        // Test scalar multiplication
        processor.mul_scalar_f32(&a, 2.0, &mut output);
        assert_eq!(output, vec![2.0, 4.0, 6.0, 8.0]);

        // Test sum
        let sum = processor.sum_f32(&a);
        assert_eq!(sum, 10.0);

        // Test dot product
        let dot = processor.dot_product_f32(&a, &b);
        assert_eq!(dot, 25.0); // 1*0.5 + 2*1.5 + 3*2.5 + 4*3.5
    }

    #[test]
    fn test_audio_operations() {
        let processor = SimdProcessor::new();

        let mut samples = vec![0.1, 0.2, 0.3, 0.4];
        processor.apply_gain(&mut samples, 2.0);
        assert_eq!(samples, vec![0.2, 0.4, 0.6, 0.8]);

        let a = vec![0.1, 0.2, 0.3, 0.4];
        let b = vec![0.05, 0.15, 0.25, 0.35];
        let mut output = vec![0.0; 4];

        processor.mix_audio(&a, &b, &mut output, 0.8, 0.2);
        // Expected: [0.1*0.8 + 0.05*0.2, 0.2*0.8 + 0.15*0.2, ...]
        let expected = [0.09, 0.19, 0.29, 0.39];

        for (actual, expected) in output.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_global_processor() {
        let processor1 = get_simd_processor();
        let processor2 = get_simd_processor();

        // Should be the same instance
        assert_eq!(processor1.level(), processor2.level());
    }

    #[test]
    fn test_enhanced_audio_operations() {
        let processor = SimdProcessor::new();

        // Test RMS calculation
        let samples = vec![0.1, -0.2, 0.3, -0.4];
        let rms = processor.calculate_rms(&samples);
        let expected_rms = (0.01f32 + 0.04 + 0.09 + 0.16).sqrt() / 2.0; // sqrt(0.3 / 4)
        assert!((rms - expected_rms).abs() < 1e-6);

        // Test RMS normalization
        let mut samples = vec![0.1, -0.2, 0.3, -0.4];
        processor.normalize_rms(&mut samples, 0.5);
        let new_rms = processor.calculate_rms(&samples);
        assert!((new_rms - 0.5).abs() < 1e-6);

        // Test soft clipping
        let mut samples = vec![-2.0, -0.5, 0.5, 2.0];
        processor.soft_clip(&mut samples, 1.0);
        // Values within threshold should be unchanged
        assert!((samples[1] + 0.5).abs() < 1e-6);
        assert!((samples[2] - 0.5).abs() < 1e-6);
        // Values above threshold should be clipped
        assert!(samples[0] > -2.0 && samples[0] < -0.5);
        assert!(samples[3] < 2.0 && samples[3] > 0.5);

        // Test window application
        let mut samples = vec![1.0, 1.0, 1.0, 1.0];
        let window = vec![0.0, 0.5, 1.0, 0.5];
        processor.apply_window(&mut samples, &window);
        assert_eq!(samples, vec![0.0, 0.5, 1.0, 0.5]);

        // Test mel spectrogram normalization
        let mut mel_data = vec![1.0, 2.0, 3.0, 4.0];
        processor.normalize_mel_spectrogram(&mut mel_data, 2.5, 1.0);
        let expected = [-1.5, -0.5, 0.5, 1.5]; // (x - 2.5) / 1.0
        for (actual, expected) in mel_data.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_enhanced_operations_edge_cases() {
        let processor = SimdProcessor::new();

        // Test empty slice
        let mut empty: Vec<f32> = vec![];
        processor.normalize_rms(&mut empty, 1.0);
        assert!(empty.is_empty());

        let rms = processor.calculate_rms(&empty);
        assert_eq!(rms, 0.0);

        // Test zero target RMS
        let mut samples = vec![0.1, 0.2, 0.3];
        let original = samples.clone();
        processor.normalize_rms(&mut samples, 0.0);
        assert_eq!(samples, original); // Should be unchanged

        // Test zero std dev in mel normalization
        let mut mel_data = vec![1.0, 2.0, 3.0];
        let original = mel_data.clone();
        processor.normalize_mel_spectrogram(&mut mel_data, 1.0, 0.0);
        assert_eq!(mel_data, original); // Should be unchanged
    }
}

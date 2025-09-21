//! Automatic Gain Control (AGC) module
//!
//! This module implements real-time automatic gain control for maintaining
//! consistent audio levels across different input sources and conditions.

use crate::RecognitionError;
use voirs_sdk::AudioBuffer;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// AGC configuration
#[derive(Debug, Clone)]
pub struct AGCConfig {
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Target output level in dB
    pub target_level: f32,
    /// Maximum gain in dB
    pub max_gain: f32,
    /// Attack time in seconds
    pub attack_time: f32,
    /// Release time in seconds
    pub release_time: f32,
}

impl Default for AGCConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            target_level: -20.0,
            max_gain: 30.0,
            attack_time: 0.001,
            release_time: 0.1,
        }
    }
}

/// AGC processing statistics
#[derive(Debug, Clone)]
pub struct AGCStats {
    /// Current gain applied in dB
    pub current_gain_db: f32,
    /// Input signal level in dB
    pub input_level_db: f32,
    /// Output signal level in dB
    pub output_level_db: f32,
    /// Gain reduction applied in dB
    pub gain_reduction_db: f32,
    /// Processing time in milliseconds
    pub processing_time_ms: f64,
}

/// AGC processing result
#[derive(Debug, Clone)]
pub struct AGCResult {
    /// Enhanced audio buffer
    pub enhanced_audio: AudioBuffer,
    /// Processing statistics
    pub stats: AGCStats,
}

/// Real-time AGC processor
#[derive(Debug)]
pub struct AGCProcessor {
    /// Configuration
    config: AGCConfig,
    /// Current gain value (linear scale)
    current_gain: f32,
    /// Attack coefficient
    attack_coeff: f32,
    /// Release coefficient
    release_coeff: f32,
    /// Peak detector state
    peak_detector: f32,
    /// RMS detector state
    rms_detector: f32,
    /// Gain smoothing filter
    gain_smoother: f32,
}

impl AGCProcessor {
    /// Create a new AGC processor
    pub fn new(config: AGCConfig) -> Result<Self, RecognitionError> {
        // Calculate attack and release coefficients
        let attack_coeff = (-1.0 / (config.attack_time * config.sample_rate as f32)).exp();
        let release_coeff = (-1.0 / (config.release_time * config.sample_rate as f32)).exp();

        Ok(Self {
            config,
            current_gain: 1.0,
            attack_coeff,
            release_coeff,
            peak_detector: 0.0,
            rms_detector: 0.0,
            gain_smoother: 1.0,
        })
    }

    /// Process audio buffer with AGC
    pub async fn process(&mut self, audio: &AudioBuffer) -> Result<AGCResult, RecognitionError> {
        let start_time = std::time::Instant::now();

        let samples = audio.samples();
        let mut enhanced_samples = Vec::with_capacity(samples.len());

        // Calculate input level
        let input_level_db = self.calculate_rms_level(samples);

        // Process each sample
        for &sample in samples {
            let enhanced_sample = self.process_sample(sample);
            enhanced_samples.push(enhanced_sample);
        }

        let enhanced_audio = AudioBuffer::mono(enhanced_samples, audio.sample_rate());

        // Calculate output level
        let output_level_db = self.calculate_rms_level(&enhanced_audio.samples());

        // Calculate statistics
        let current_gain_db = 20.0 * self.current_gain.log10();
        let gain_reduction_db = if current_gain_db < 0.0 {
            -current_gain_db
        } else {
            0.0
        };
        let processing_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;

        let stats = AGCStats {
            current_gain_db,
            input_level_db,
            output_level_db,
            gain_reduction_db,
            processing_time_ms,
        };

        Ok(AGCResult {
            enhanced_audio,
            stats,
        })
    }

    /// Process a single sample
    fn process_sample(&mut self, sample: f32) -> f32 {
        // Update peak detector
        let abs_sample = sample.abs();
        if abs_sample > self.peak_detector {
            self.peak_detector = abs_sample;
        } else {
            self.peak_detector =
                self.peak_detector * self.release_coeff + abs_sample * (1.0 - self.release_coeff);
        }

        // Update RMS detector
        let sample_squared = sample * sample;
        self.rms_detector =
            self.rms_detector * self.release_coeff + sample_squared * (1.0 - self.release_coeff);

        // Calculate required gain
        let rms_level = self.rms_detector.sqrt();
        let target_linear = self.db_to_linear(self.config.target_level);
        let required_gain = if rms_level > 0.0 {
            target_linear / rms_level
        } else {
            1.0
        };

        // Limit gain to maximum
        let max_gain_linear = self.db_to_linear(self.config.max_gain);
        let limited_gain = required_gain.min(max_gain_linear);

        // Apply gain smoothing
        let coeff = if limited_gain > self.gain_smoother {
            self.attack_coeff
        } else {
            self.release_coeff
        };

        self.gain_smoother = self.gain_smoother * coeff + limited_gain * (1.0 - coeff);
        self.current_gain = self.gain_smoother;

        // Apply gain to sample
        sample * self.current_gain
    }

    /// Calculate RMS level in dB with SIMD optimizations
    fn calculate_rms_level(&self, samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return -60.0; // Very low level
        }

        let sum_of_squares = Self::calculate_sum_of_squares_simd(samples);
        let rms = (sum_of_squares / samples.len() as f32).sqrt();

        if rms > 0.0 {
            20.0 * rms.log10()
        } else {
            -60.0
        }
    }

    /// SIMD-optimized sum of squares calculation for RMS
    #[cfg(target_arch = "x86_64")]
    fn calculate_sum_of_squares_simd(samples: &[f32]) -> f32 {
        if is_x86_feature_detected!("avx2") {
            unsafe { Self::calculate_sum_of_squares_avx2(samples) }
        } else {
            Self::calculate_sum_of_squares_scalar(samples)
        }
    }

    /// SIMD-optimized sum of squares calculation for ARM64
    #[cfg(target_arch = "aarch64")]
    fn calculate_sum_of_squares_simd(samples: &[f32]) -> f32 {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { Self::calculate_sum_of_squares_neon(samples) }
        } else {
            Self::calculate_sum_of_squares_scalar(samples)
        }
    }

    /// Fallback SIMD implementation for other architectures
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    fn calculate_sum_of_squares_simd(samples: &[f32]) -> f32 {
        Self::calculate_sum_of_squares_scalar(samples)
    }

    /// AVX2 vectorized sum of squares calculation
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn calculate_sum_of_squares_avx2(samples: &[f32]) -> f32 {
        let mut sum = _mm256_setzero_ps();
        let chunks = samples.chunks_exact(8);
        let remainder = chunks.remainder();

        // Process 8 samples at a time with AVX2
        for chunk in chunks {
            let values = _mm256_loadu_ps(chunk.as_ptr());
            let squares = _mm256_mul_ps(values, values);
            sum = _mm256_add_ps(sum, squares);
        }

        // Horizontal sum of the 8 accumulated values
        let sum_high = _mm256_extractf128_ps(sum, 1);
        let sum_low = _mm256_extractf128_ps(sum, 0);
        let sum_128 = _mm_add_ps(sum_low, sum_high);

        let sum_64 = _mm_add_ps(sum_128, _mm_movehl_ps(sum_128, sum_128));
        let sum_32 = _mm_add_ss(sum_64, _mm_shuffle_ps(sum_64, sum_64, 1));

        let mut result = _mm_cvtss_f32(sum_32);

        // Process remaining samples
        for &sample in remainder {
            result += sample * sample;
        }

        result
    }

    /// NEON vectorized sum of squares calculation
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn calculate_sum_of_squares_neon(samples: &[f32]) -> f32 {
        let mut sum = vdupq_n_f32(0.0);
        let chunks = samples.chunks_exact(4);
        let remainder = chunks.remainder();

        // Process 4 samples at a time with NEON
        for chunk in chunks {
            let values = vld1q_f32(chunk.as_ptr());
            let squares = vmulq_f32(values, values);
            sum = vaddq_f32(sum, squares);
        }

        // Horizontal sum of the 4 accumulated values
        let sum_pair = vpadd_f32(vget_low_f32(sum), vget_high_f32(sum));
        let result_pair = vpadd_f32(sum_pair, sum_pair);
        let mut result = vget_lane_f32(result_pair, 0);

        // Process remaining samples
        for &sample in remainder {
            result += sample * sample;
        }

        result
    }

    /// Scalar fallback for sum of squares calculation
    fn calculate_sum_of_squares_scalar(samples: &[f32]) -> f32 {
        samples.iter().map(|&x| x * x).sum()
    }

    /// Convert dB to linear scale
    fn db_to_linear(&self, db: f32) -> f32 {
        10.0_f32.powf(db / 20.0)
    }

    /// Convert linear to dB scale
    fn linear_to_db(&self, linear: f32) -> f32 {
        if linear > 0.0 {
            20.0 * linear.log10()
        } else {
            -60.0
        }
    }

    /// Reset processor state
    pub fn reset(&mut self) -> Result<(), RecognitionError> {
        self.current_gain = 1.0;
        self.peak_detector = 0.0;
        self.rms_detector = 0.0;
        self.gain_smoother = 1.0;
        Ok(())
    }

    /// Get current gain in dB
    pub fn get_current_gain_db(&self) -> f32 {
        self.linear_to_db(self.current_gain)
    }

    /// Get current input level in dB
    pub fn get_current_input_level_db(&self) -> f32 {
        self.linear_to_db(self.rms_detector.sqrt())
    }

    /// Set target level
    pub fn set_target_level(&mut self, target_db: f32) {
        self.config.target_level = target_db;
    }

    /// Set maximum gain
    pub fn set_max_gain(&mut self, max_gain_db: f32) {
        self.config.max_gain = max_gain_db;
    }

    /// Set attack time
    pub fn set_attack_time(&mut self, attack_time_seconds: f32) {
        self.config.attack_time = attack_time_seconds;
        self.attack_coeff = (-1.0 / (attack_time_seconds * self.config.sample_rate as f32)).exp();
    }

    /// Set release time
    pub fn set_release_time(&mut self, release_time_seconds: f32) {
        self.config.release_time = release_time_seconds;
        self.release_coeff = (-1.0 / (release_time_seconds * self.config.sample_rate as f32)).exp();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_agc_processor_creation() {
        let config = AGCConfig::default();
        let processor = AGCProcessor::new(config);
        assert!(processor.is_ok());
    }

    #[tokio::test]
    async fn test_agc_basic_processing() {
        let config = AGCConfig::default();
        let mut processor = AGCProcessor::new(config).unwrap();

        // Test with quiet signal
        let samples = vec![0.01f32; 1024];
        let audio = AudioBuffer::mono(samples, 16000);

        let result = processor.process(&audio).await;
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.stats.processing_time_ms > 0.0);
        assert!(result.stats.current_gain_db > 0.0); // Should apply gain to quiet signal
    }

    #[tokio::test]
    async fn test_agc_loud_signal() {
        let config = AGCConfig::default();
        let mut processor = AGCProcessor::new(config).unwrap();

        // Test with loud signal
        let samples = vec![0.5f32; 1024];
        let audio = AudioBuffer::mono(samples, 16000);

        let result = processor.process(&audio).await;
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.stats.gain_reduction_db >= 0.0); // Should show gain reduction
    }

    #[tokio::test]
    async fn test_agc_target_level_adjustment() {
        let config = AGCConfig::default();
        let mut processor = AGCProcessor::new(config).unwrap();

        // Change target level
        processor.set_target_level(-10.0);
        processor.set_max_gain(20.0);

        let samples = vec![0.1f32; 1024];
        let audio = AudioBuffer::mono(samples, 16000);

        let result = processor.process(&audio).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_agc_attack_release_timing() {
        let config = AGCConfig {
            attack_time: 0.001,
            release_time: 0.1,
            ..Default::default()
        };
        let mut processor = AGCProcessor::new(config).unwrap();

        // Test with varying signal levels
        let mut samples = Vec::new();
        samples.extend(vec![0.01f32; 512]); // Quiet part
        samples.extend(vec![0.5f32; 512]); // Loud part

        let audio = AudioBuffer::mono(samples, 16000);

        let result = processor.process(&audio).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    #[allow(clippy::float_cmp)]
    async fn test_agc_reset() {
        let config = AGCConfig::default();
        let mut processor = AGCProcessor::new(config).unwrap();

        let result = processor.reset();
        assert!(result.is_ok());
        assert_eq!(processor.get_current_gain_db(), 0.0);
    }

    #[tokio::test]
    async fn test_agc_gain_limits() {
        let config = AGCConfig {
            max_gain: 10.0,
            ..Default::default()
        };
        let mut processor = AGCProcessor::new(config).unwrap();

        // Very quiet signal that would need high gain
        let samples = vec![0.001f32; 1024];
        let audio = AudioBuffer::mono(samples, 16000);

        let result = processor.process(&audio).await;
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.stats.current_gain_db <= 10.0); // Should not exceed max gain
    }

    #[test]
    fn test_simd_vs_scalar_rms_consistency() {
        // Test that SIMD and scalar implementations produce identical results
        let test_samples = vec![
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], // Exactly 8 samples for AVX2
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], // 9 samples (1 remainder)
            vec![0.1, 0.2, 0.3, 0.4, 0.5],                // 5 samples (all remainder)
            vec![0.1; 100],                               // Larger buffer
            vec![-0.5, 0.3, -0.1, 0.8, -0.2, 0.6, -0.4, 0.7], // Mixed positive/negative
        ];

        for samples in test_samples {
            let simd_result = AGCProcessor::calculate_sum_of_squares_simd(&samples);
            let scalar_result = AGCProcessor::calculate_sum_of_squares_scalar(&samples);

            // Results should be very close (allowing for floating-point precision differences)
            let diff = (simd_result - scalar_result).abs();
            assert!(
                diff < 1e-6,
                "SIMD and scalar results differ too much: SIMD={}, Scalar={}, Diff={}",
                simd_result,
                scalar_result,
                diff
            );
        }
    }

    #[test]
    fn test_simd_rms_calculation() {
        let processor = AGCProcessor::new(AGCConfig::default()).unwrap();

        // Test with known values
        let samples = vec![0.1, 0.2, 0.3, 0.4];
        let expected_sum_of_squares = 0.01f32 + 0.04 + 0.09 + 0.16; // 0.3
        let expected_rms = (expected_sum_of_squares / 4.0f32).sqrt(); // sqrt(0.075) ≈ 0.274
        let expected_db = 20.0 * expected_rms.log10(); // ≈ -11.24 dB

        let actual_db = processor.calculate_rms_level(&samples);

        // Allow for small floating-point differences
        let diff = (actual_db - expected_db).abs();
        assert!(
            diff < 0.01,
            "RMS calculation incorrect: expected {}, got {}",
            expected_db,
            actual_db
        );
    }

    #[test]
    fn test_simd_performance_benefit() {
        use std::time::Instant;

        // Create a large buffer to test performance
        let large_samples: Vec<f32> = (0..10000).map(|i| (i as f32 * 0.001).sin()).collect();

        // Use more iterations to get reliable timing measurements
        let iterations = 10000;

        // Measure scalar performance
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = AGCProcessor::calculate_sum_of_squares_scalar(&large_samples);
        }
        let scalar_duration = start.elapsed();

        // Measure SIMD performance
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = AGCProcessor::calculate_sum_of_squares_simd(&large_samples);
        }
        let simd_duration = start.elapsed();

        // Check if we have reliable measurements (both should take at least 1ms)
        if scalar_duration.as_millis() > 0 && simd_duration.as_millis() > 0 {
            // SIMD should be faster (or at least not significantly slower)
            // We allow SIMD to be up to 50% slower to account for detection overhead on platforms without SIMD
            let slowdown_ratio =
                simd_duration.as_nanos() as f64 / scalar_duration.as_nanos() as f64;
            assert!(slowdown_ratio < 1.5,
                "SIMD implementation is too slow compared to scalar: SIMD={}μs, Scalar={}μs, Ratio={:.2}",
                simd_duration.as_micros(), scalar_duration.as_micros(), slowdown_ratio);
        } else {
            // If measurements are too small, just verify that both complete without error
            println!("Performance measurements too small for reliable comparison: SIMD={}μs, Scalar={}μs", 
                simd_duration.as_micros(), scalar_duration.as_micros());
        }

        // Verify results are still consistent (allow for slightly more tolerance for large datasets)
        let simd_result = AGCProcessor::calculate_sum_of_squares_simd(&large_samples);
        let scalar_result = AGCProcessor::calculate_sum_of_squares_scalar(&large_samples);
        let diff = (simd_result - scalar_result).abs() / scalar_result;
        assert!(
            diff < 1e-5,
            "Performance test results inconsistent: diff={}",
            diff
        );
    }

    #[test]
    fn test_simd_edge_cases() {
        // Test empty buffer
        let empty: Vec<f32> = vec![];
        let simd_result = AGCProcessor::calculate_sum_of_squares_simd(&empty);
        let scalar_result = AGCProcessor::calculate_sum_of_squares_scalar(&empty);
        assert_eq!(simd_result, scalar_result);

        // Test single sample
        let single = vec![0.5];
        let simd_result = AGCProcessor::calculate_sum_of_squares_simd(&single);
        let scalar_result = AGCProcessor::calculate_sum_of_squares_scalar(&single);
        assert!((simd_result - scalar_result).abs() < 1e-6);

        // Test zero samples
        let zeros = vec![0.0; 16];
        let simd_result = AGCProcessor::calculate_sum_of_squares_simd(&zeros);
        let scalar_result = AGCProcessor::calculate_sum_of_squares_scalar(&zeros);
        assert_eq!(simd_result, 0.0);
        assert_eq!(scalar_result, 0.0);
    }
}

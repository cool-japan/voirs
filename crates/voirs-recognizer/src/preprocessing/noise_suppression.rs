//! Real-time noise suppression module
//!
//! This module implements various noise suppression algorithms for real-time
//! audio processing, including spectral subtraction and Wiener filtering.
//! Includes SIMD optimizations for enhanced performance.

use crate::RecognitionError;
use scirs2_core::Complex;
use std::collections::VecDeque;
use voirs_sdk::AudioBuffer;

#[cfg(target_arch = "aarch64")]
#[allow(unused_imports)]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Noise suppression algorithm types
#[derive(Debug, Clone, PartialEq)]
pub enum NoiseSuppressionAlgorithm {
    /// Spectral subtraction method
    SpectralSubtraction,
    /// Wiener filtering method
    WienerFiltering,
    /// Adaptive noise suppression
    Adaptive,
}

/// Configuration for noise suppression
#[derive(Debug, Clone)]
pub struct NoiseSuppressionConfig {
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Buffer size for processing
    pub buffer_size: usize,
    /// Noise suppression algorithm
    pub algorithm: NoiseSuppressionAlgorithm,
    /// Over-subtraction factor (alpha)
    pub alpha: f32,
    /// Floor factor (beta)
    pub beta: f32,
}

impl Default for NoiseSuppressionConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            buffer_size: 1024,
            algorithm: NoiseSuppressionAlgorithm::SpectralSubtraction,
            alpha: 2.0,
            beta: 0.01,
        }
    }
}

/// Statistics from noise suppression processing
#[derive(Debug, Clone)]
pub struct NoiseSuppressionStats {
    /// Estimated noise floor level (dB)
    pub noise_floor_db: f32,
    /// Signal-to-noise ratio improvement (dB)
    pub snr_improvement_db: f32,
    /// Amount of noise reduced (0.0 to 1.0)
    pub noise_reduction: f32,
    /// Processing time in milliseconds
    pub processing_time_ms: f64,
}

/// Result of noise suppression processing
#[derive(Debug, Clone)]
pub struct NoiseSuppressionResult {
    /// Enhanced audio buffer
    pub enhanced_audio: AudioBuffer,
    /// Processing statistics
    pub stats: NoiseSuppressionStats,
}

/// Real-time noise suppression processor
#[derive(Debug)]
pub struct NoiseSuppressionProcessor {
    /// Configuration
    config: NoiseSuppressionConfig,
    /// Noise profile estimation
    noise_profile: Vec<f32>,
    /// Previous frames for overlap-add
    prev_frames: VecDeque<Vec<f32>>,
    /// FFT buffer
    fft_buffer: Vec<f32>,
    /// Noise estimation window
    noise_estimation_frames: usize,
    /// Frame counter
    frame_counter: usize,
    /// Running noise estimate
    running_noise_estimate: Vec<f32>,
    /// Phase of the most recently analyzed frame (per frequency bin, radians).
    ///
    /// Spectral subtraction / Wiener filtering only modify the magnitude spectrum.
    /// To reconstruct a time-domain signal without phase artifacts we reuse the
    /// original (noisy) phase captured during `compute_spectrum`, which is the
    /// standard spectral-subtraction practice.
    last_phase: Vec<f32>,
}

impl NoiseSuppressionProcessor {
    /// Create a new noise suppression processor
    pub fn new(config: NoiseSuppressionConfig) -> Result<Self, RecognitionError> {
        let fft_size = config.buffer_size;
        let freq_bins = fft_size / 2 + 1;

        Ok(Self {
            config,
            noise_profile: vec![0.0; freq_bins],
            prev_frames: VecDeque::new(),
            fft_buffer: vec![0.0; fft_size],
            noise_estimation_frames: 10, // First 10 frames for noise estimation
            frame_counter: 0,
            running_noise_estimate: vec![0.0; freq_bins],
            last_phase: vec![0.0; freq_bins],
        })
    }

    /// Process audio buffer with noise suppression
    pub async fn process(
        &mut self,
        audio: &AudioBuffer,
    ) -> Result<NoiseSuppressionResult, RecognitionError> {
        let start_time = std::time::Instant::now();

        let samples = audio.samples();
        let mut enhanced_samples = Vec::with_capacity(samples.len());

        // Process audio in chunks
        for chunk in samples.chunks(self.config.buffer_size) {
            let mut chunk_vec = chunk.to_vec();

            // Pad if necessary
            if chunk_vec.len() < self.config.buffer_size {
                chunk_vec.resize(self.config.buffer_size, 0.0);
            }

            let enhanced_chunk = self.process_chunk(&chunk_vec).await?;
            enhanced_samples.extend_from_slice(&enhanced_chunk[..chunk.len()]);
        }

        let enhanced_audio = AudioBuffer::mono(enhanced_samples, audio.sample_rate());

        // Calculate statistics
        let noise_floor_db = self.calculate_noise_floor();
        let snr_improvement_db = self.calculate_snr_improvement(audio, &enhanced_audio);
        let noise_reduction = self.calculate_noise_reduction();
        let processing_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;

        let stats = NoiseSuppressionStats {
            noise_floor_db,
            snr_improvement_db,
            noise_reduction,
            processing_time_ms,
        };

        Ok(NoiseSuppressionResult {
            enhanced_audio,
            stats,
        })
    }

    /// Process a single chunk of audio
    async fn process_chunk(&mut self, chunk: &[f32]) -> Result<Vec<f32>, RecognitionError> {
        match self.config.algorithm {
            NoiseSuppressionAlgorithm::SpectralSubtraction => {
                self.spectral_subtraction(chunk).await
            }
            NoiseSuppressionAlgorithm::WienerFiltering => self.wiener_filtering(chunk).await,
            NoiseSuppressionAlgorithm::Adaptive => self.adaptive_noise_suppression(chunk).await,
        }
    }

    /// Spectral subtraction algorithm
    async fn spectral_subtraction(&mut self, chunk: &[f32]) -> Result<Vec<f32>, RecognitionError> {
        // Simple spectral subtraction implementation
        let mut enhanced_chunk = chunk.to_vec();

        // Apply window function (Hann window)
        self.apply_window(&mut enhanced_chunk);

        // Analyze the frame with a real FFT (also caches per-bin phase for reconstruction).
        let spectrum = self.compute_spectrum(&enhanced_chunk)?;

        // Update noise profile if we're in the initial estimation phase
        if self.frame_counter < self.noise_estimation_frames {
            self.update_noise_profile(&spectrum);
        }

        // Apply spectral subtraction
        let enhanced_spectrum = self.apply_spectral_subtraction(&spectrum);

        // Reconstruct the time-domain frame from processed magnitudes + original phase.
        let enhanced_chunk = self.spectrum_to_time_domain(&enhanced_spectrum)?;

        self.frame_counter += 1;
        Ok(enhanced_chunk)
    }

    /// Wiener filtering algorithm
    async fn wiener_filtering(&mut self, chunk: &[f32]) -> Result<Vec<f32>, RecognitionError> {
        // Simple Wiener filtering implementation
        let mut enhanced_chunk = chunk.to_vec();

        // Apply adaptive filtering based on SNR estimation
        let snr_estimate = self.estimate_local_snr(&enhanced_chunk);
        let wiener_gain = snr_estimate / (snr_estimate + 1.0);

        for sample in &mut enhanced_chunk {
            *sample *= wiener_gain;
        }

        Ok(enhanced_chunk)
    }

    /// Adaptive noise suppression algorithm
    async fn adaptive_noise_suppression(
        &mut self,
        chunk: &[f32],
    ) -> Result<Vec<f32>, RecognitionError> {
        // Adaptive algorithm that combines spectral subtraction and Wiener filtering
        let spectral_result = self.spectral_subtraction(chunk).await?;
        let wiener_result = self.wiener_filtering(&spectral_result).await?;

        Ok(wiener_result)
    }

    /// Apply window function to audio chunk
    fn apply_window(&self, chunk: &mut [f32]) {
        #[cfg(target_arch = "x86_64")]
        {
            if is_simd_available() {
                self.apply_window_simd(chunk);
            } else {
                self.apply_window_scalar(chunk);
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            self.apply_window_scalar(chunk);
        }
    }

    /// SIMD-optimized window function application.
    ///
    /// The Hann window coefficients are computed with the exact same scalar
    /// `f32::cos` formula used by [`Self::apply_window_scalar`], then the
    /// element-wise multiply against the input frame is vectorized with AVX2.
    /// Because both paths derive the coefficients from `f32::cos` and apply an
    /// IEEE-754 f32 multiply, the SIMD and scalar results are bit-for-bit
    /// identical. (The previous implementation used a 4-term Taylor cosine
    /// approximation that diverged from `f32::cos` by ~1e-5, breaking the
    /// SIMD-vs-scalar consistency test.)
    #[cfg(target_arch = "x86_64")]
    fn apply_window_simd(&self, chunk: &mut [f32]) {
        if !is_x86_feature_detected!("avx2") {
            return self.apply_window_scalar(chunk);
        }

        let window = hann_window(chunk.len());

        unsafe {
            let len = chunk.len();
            let simd_len = len - (len % 8);

            // Process 8 samples at a time with AVX2.
            for i in (0..simd_len).step_by(8) {
                let samples = _mm256_loadu_ps(chunk[i..].as_ptr());
                let window_vals = _mm256_loadu_ps(window[i..].as_ptr());
                let windowed = _mm256_mul_ps(samples, window_vals);
                _mm256_storeu_ps(chunk[i..].as_mut_ptr(), windowed);
            }

            // Process remaining samples with the identical scalar multiply.
            for i in simd_len..len {
                chunk[i] *= window[i];
            }
        }
    }

    /// Fallback scalar window function application.
    ///
    /// Shares [`hann_window`] with the SIMD path so the two produce identical
    /// coefficients (and therefore identical output).
    fn apply_window_scalar(&self, chunk: &mut [f32]) {
        let window = hann_window(chunk.len());
        for (sample, &w) in chunk.iter_mut().zip(window.iter()) {
            *sample *= w;
        }
    }

    /// Compute the magnitude spectrum of a frame via a real FFT.
    ///
    /// Uses `scirs2_fft::rfft` (O(N log N)) instead of a naive O(N²) DFT. The
    /// per-bin phase (`atan2(im, re)`) of the complex spectrum is cached in
    /// `self.last_phase` so that `spectrum_to_time_domain` can reconstruct the
    /// signal with the original phase after the magnitudes have been processed.
    /// Returns the `n / 2 + 1` non-negative-frequency magnitudes.
    fn compute_spectrum(&mut self, chunk: &[f32]) -> Result<Vec<f32>, RecognitionError> {
        let n = chunk.len();
        let buf_f64: Vec<f64> = chunk.iter().map(|&s| f64::from(s)).collect();

        let spectrum: Vec<Complex<f64>> = scirs2_fft::rfft(&buf_f64, Some(n)).map_err(|e| {
            RecognitionError::AudioProcessingError {
                message: format!("rfft failed in noise suppression: {e}"),
                source: None,
            }
        })?;

        // Cache the per-bin phase (radians) of the original (noisy) frame so the
        // reconstruction step can reuse it (standard spectral-subtraction practice).
        self.last_phase = spectrum.iter().map(|c| c.im.atan2(c.re) as f32).collect();

        // Return magnitudes for spectral-subtraction / Wiener processing.
        Ok(spectrum.iter().map(|c| c.norm() as f32).collect())
    }

    /// Update noise profile with current spectrum
    fn update_noise_profile(&mut self, spectrum: &[f32]) {
        #[cfg(target_arch = "x86_64")]
        {
            if is_simd_available() {
                self.update_noise_profile_simd(spectrum);
            } else {
                self.update_noise_profile_scalar(spectrum);
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            self.update_noise_profile_scalar(spectrum);
        }
    }

    /// SIMD-optimized noise profile update
    #[cfg(target_arch = "x86_64")]
    fn update_noise_profile_simd(&mut self, spectrum: &[f32]) {
        if !is_x86_feature_detected!("avx2") || spectrum.len() != self.noise_profile.len() {
            return self.update_noise_profile_scalar(spectrum);
        }

        let alpha = 0.95f32;

        if self.frame_counter == 0 {
            self.noise_profile.copy_from_slice(spectrum);
            return;
        }

        unsafe {
            let alpha_vec = _mm256_set1_ps(alpha);
            let one_minus_alpha_vec = _mm256_set1_ps(1.0 - alpha);

            // Process 8 elements at a time
            let len = spectrum.len();
            let simd_len = len & !7; // Round down to nearest multiple of 8

            for i in (0..simd_len).step_by(8) {
                let old_profile = _mm256_loadu_ps(self.noise_profile[i..].as_ptr());
                let new_spectrum = _mm256_loadu_ps(spectrum[i..].as_ptr());

                let smoothed = _mm256_add_ps(
                    _mm256_mul_ps(alpha_vec, old_profile),
                    _mm256_mul_ps(one_minus_alpha_vec, new_spectrum),
                );

                _mm256_storeu_ps(self.noise_profile[i..].as_mut_ptr(), smoothed);
            }

            // Handle remaining elements
            for i in simd_len..len {
                self.noise_profile[i] = alpha * self.noise_profile[i] + (1.0 - alpha) * spectrum[i];
            }
        }
    }

    /// Fallback scalar noise profile update
    fn update_noise_profile_scalar(&mut self, spectrum: &[f32]) {
        let alpha = 0.95; // Smoothing factor

        if self.frame_counter == 0 {
            self.noise_profile = spectrum.to_vec();
        } else {
            for (i, &magnitude) in spectrum.iter().enumerate() {
                if i < self.noise_profile.len() {
                    self.noise_profile[i] =
                        alpha * self.noise_profile[i] + (1.0 - alpha) * magnitude;
                }
            }
        }
    }

    /// Apply spectral subtraction to spectrum
    fn apply_spectral_subtraction(&self, spectrum: &[f32]) -> Vec<f32> {
        if is_simd_available() {
            #[cfg(target_arch = "x86_64")]
            {
                self.apply_spectral_subtraction_simd(spectrum)
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                self.apply_spectral_subtraction_scalar(spectrum)
            }
        } else {
            self.apply_spectral_subtraction_scalar(spectrum)
        }
    }

    /// Reconstruct a time-domain frame from processed magnitudes and cached phase.
    ///
    /// Recombines each (processed) magnitude with the original phase stored in
    /// `self.last_phase` into a complex bin, then applies `scirs2_fft::irfft`
    /// with `Some(n)` so the output length matches `self.config.buffer_size`.
    fn spectrum_to_time_domain(&self, spectrum: &[f32]) -> Result<Vec<f32>, RecognitionError> {
        let n = self.config.buffer_size;

        // Recombine magnitude + original phase into complex bins:
        //   X[k] = |X[k]| * (cos(phi[k]) + i * sin(phi[k]))
        let complex_bins: Vec<Complex<f64>> = spectrum
            .iter()
            .enumerate()
            .map(|(k, &magnitude)| {
                let phase = f64::from(self.last_phase.get(k).copied().unwrap_or(0.0));
                let mag = f64::from(magnitude);
                Complex::new(mag * phase.cos(), mag * phase.sin())
            })
            .collect();

        let time_domain = scirs2_fft::irfft(&complex_bins, Some(n)).map_err(|e| {
            RecognitionError::AudioProcessingError {
                message: format!("irfft failed in noise suppression: {e}"),
                source: None,
            }
        })?;

        Ok(time_domain.iter().map(|&x| x as f32).collect())
    }

    /// Estimate local SNR for Wiener filtering
    fn estimate_local_snr(&self, chunk: &[f32]) -> f32 {
        if is_simd_available() {
            self.estimate_local_snr_simd(chunk)
        } else {
            self.estimate_local_snr_scalar(chunk)
        }
    }

    /// Calculate noise floor in dB
    fn calculate_noise_floor(&self) -> f32 {
        let noise_power = self.noise_profile.iter().sum::<f32>() / self.noise_profile.len() as f32;
        20.0 * noise_power.log10()
    }

    /// Calculate SNR improvement
    fn calculate_snr_improvement(&self, original: &AudioBuffer, enhanced: &AudioBuffer) -> f32 {
        let original_power = original.samples().iter().map(|&x| x * x).sum::<f32>()
            / original.samples().len() as f32;
        let enhanced_power = enhanced.samples().iter().map(|&x| x * x).sum::<f32>()
            / enhanced.samples().len() as f32;

        if original_power > 0.0 && enhanced_power > 0.0 {
            10.0 * (enhanced_power / original_power).log10()
        } else {
            0.0
        }
    }

    /// Calculate noise reduction amount
    fn calculate_noise_reduction(&self) -> f32 {
        // Simple metric based on spectral subtraction effectiveness
        let avg_noise = self.noise_profile.iter().sum::<f32>() / self.noise_profile.len() as f32;
        let noise_reduction = (self.config.alpha - 1.0) * avg_noise;
        noise_reduction.max(0.0).min(1.0)
    }

    /// Reset processor state
    pub fn reset(&mut self) -> Result<(), RecognitionError> {
        self.noise_profile.fill(0.0);
        self.prev_frames.clear();
        self.fft_buffer.fill(0.0);
        self.frame_counter = 0;
        self.running_noise_estimate.fill(0.0);
        self.last_phase.fill(0.0);
        Ok(())
    }
}

/// Check if SIMD optimizations are available
fn is_simd_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2")
    }
    #[cfg(target_arch = "aarch64")]
    {
        std::arch::is_aarch64_feature_detected!("neon")
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        false
    }
}

/// Compute a Hann window of length `n`: `w[i] = 0.5 * (1 - cos(2π i / (n - 1)))`.
///
/// Shared by both the scalar and SIMD window paths so they derive identical
/// coefficients from `f32::cos`, guaranteeing bit-for-bit identical output
/// regardless of which path runs.
fn hann_window(n: usize) -> Vec<f32> {
    if n <= 1 {
        // A single-sample (or empty) frame has no meaningful taper; avoid the
        // division-by-zero that `(n - 1)` would otherwise produce.
        return vec![0.0; n];
    }

    let n_minus_1 = (n - 1) as f32;
    let pi_2 = 2.0 * std::f32::consts::PI;
    (0..n)
        .map(|i| 0.5 * (1.0 - f32::cos(pi_2 * i as f32 / n_minus_1)))
        .collect()
}

/// SIMD power calculation optimization
#[cfg(target_arch = "x86_64")]
unsafe fn simd_power_sum_avx2(samples: &[f32]) -> f32 {
    if !is_x86_feature_detected!("avx2") || samples.len() < 8 {
        return samples.iter().map(|&x| x * x).sum();
    }

    let mut sum_vec = _mm256_setzero_ps();
    let len = samples.len();
    let simd_len = len & !7;

    // Process 8 samples at a time
    for i in (0..simd_len).step_by(8) {
        let vals = _mm256_loadu_ps(samples[i..].as_ptr());
        let squared = _mm256_mul_ps(vals, vals);
        sum_vec = _mm256_add_ps(sum_vec, squared);
    }

    // Extract and sum the 8 values in the vector
    let mut result = [0.0f32; 8];
    _mm256_storeu_ps(result.as_mut_ptr(), sum_vec);
    let mut total = result.iter().sum::<f32>();

    // Add remaining elements
    for &sample in &samples[simd_len..] {
        total += sample * sample;
    }

    total
}

/// Enhanced spectral subtraction with SIMD optimization
impl NoiseSuppressionProcessor {
    /// SIMD-optimized spectral subtraction
    #[cfg(target_arch = "x86_64")]
    fn apply_spectral_subtraction_simd(&self, spectrum: &[f32]) -> Vec<f32> {
        if !is_x86_feature_detected!("avx2") || spectrum.len() != self.noise_profile.len() {
            return self.apply_spectral_subtraction_scalar(spectrum);
        }

        let mut enhanced_spectrum = vec![0.0f32; spectrum.len()];

        unsafe {
            let alpha_vec = _mm256_set1_ps(self.config.alpha);
            let beta_vec = _mm256_set1_ps(self.config.beta);

            let len = spectrum.len();
            let simd_len = len & !7;

            // Process 8 elements at a time
            for i in (0..simd_len).step_by(8) {
                let magnitude = _mm256_loadu_ps(spectrum[i..].as_ptr());
                let noise_magnitude = _mm256_loadu_ps(self.noise_profile[i..].as_ptr());

                // Perform spectral subtraction: magnitude - alpha * noise_magnitude
                let subtracted =
                    _mm256_sub_ps(magnitude, _mm256_mul_ps(alpha_vec, noise_magnitude));

                // Apply floor: max(subtracted, beta * magnitude)
                let floor = _mm256_mul_ps(beta_vec, magnitude);
                let enhanced = _mm256_max_ps(subtracted, floor);

                _mm256_storeu_ps(enhanced_spectrum[i..].as_mut_ptr(), enhanced);
            }

            // Handle remaining elements
            for i in simd_len..len {
                let noise_magnitude = if i < self.noise_profile.len() {
                    self.noise_profile[i]
                } else {
                    0.0
                };

                let subtracted = spectrum[i] - self.config.alpha * noise_magnitude;
                enhanced_spectrum[i] = subtracted.max(self.config.beta * spectrum[i]);
            }
        }

        enhanced_spectrum
    }

    /// Fallback scalar spectral subtraction
    fn apply_spectral_subtraction_scalar(&self, spectrum: &[f32]) -> Vec<f32> {
        let mut enhanced_spectrum = Vec::new();

        for (i, &magnitude) in spectrum.iter().enumerate() {
            let noise_magnitude = if i < self.noise_profile.len() {
                self.noise_profile[i]
            } else {
                0.0
            };

            let subtracted = magnitude - self.config.alpha * noise_magnitude;
            let enhanced_magnitude = subtracted.max(self.config.beta * magnitude);

            enhanced_spectrum.push(enhanced_magnitude);
        }

        enhanced_spectrum
    }

    /// SIMD-optimized SNR estimation
    fn estimate_local_snr_simd(&self, chunk: &[f32]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    let signal_power = simd_power_sum_avx2(chunk) / chunk.len() as f32;
                    let noise_power =
                        simd_power_sum_avx2(&self.noise_profile) / self.noise_profile.len() as f32;

                    if noise_power > 0.0 {
                        signal_power / noise_power
                    } else {
                        10.0
                    }
                }
            } else {
                self.estimate_local_snr_scalar(chunk)
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            self.estimate_local_snr_scalar(chunk)
        }
    }

    /// Fallback scalar SNR estimation
    fn estimate_local_snr_scalar(&self, chunk: &[f32]) -> f32 {
        let signal_power = chunk.iter().map(|&x| x * x).sum::<f32>() / chunk.len() as f32;
        let noise_power = self.noise_profile.iter().sum::<f32>() / self.noise_profile.len() as f32;

        if noise_power > 0.0 {
            signal_power / noise_power
        } else {
            10.0 // Default SNR if no noise estimate
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_noise_suppression_processor_creation() {
        let config = NoiseSuppressionConfig::default();
        let processor = NoiseSuppressionProcessor::new(config);
        assert!(processor.is_ok());
    }

    #[tokio::test]
    async fn test_simd_optimization_available() {
        // Test that SIMD optimizations are detected correctly
        let available = is_simd_available();

        #[cfg(target_arch = "x86_64")]
        {
            // On x86_64, we expect AVX2 to be available on most modern systems
            // but we don't assert it as it depends on the system
            println!("SIMD availability on x86_64: {}", available);
        }

        #[cfg(target_arch = "aarch64")]
        {
            // On ARM64, NEON should be available
            println!("SIMD availability on aarch64: {}", available);
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            // On other architectures, SIMD should not be available
            assert!(!available);
        }
    }

    #[tokio::test]
    async fn test_simd_vs_scalar_consistency() {
        // Test that SIMD and scalar implementations produce equivalent results
        let config = NoiseSuppressionConfig::default();
        let mut processor = NoiseSuppressionProcessor::new(config).unwrap();

        // Create test data
        let test_spectrum = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        processor.noise_profile = vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0];

        // Test spectral subtraction consistency
        #[allow(unused_variables)] // Used in x86_64 conditional compilation
        let scalar_result = processor.apply_spectral_subtraction_scalar(&test_spectrum);

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                let simd_result = processor.apply_spectral_subtraction_simd(&test_spectrum);

                // Results should be very close (allowing for small floating point differences)
                assert_eq!(scalar_result.len(), simd_result.len());
                for (scalar, simd) in scalar_result.iter().zip(simd_result.iter()) {
                    assert!(
                        (scalar - simd).abs() < 1e-6,
                        "SIMD and scalar results differ: {} vs {}",
                        scalar,
                        simd
                    );
                }
            }
        }

        // Test window function consistency
        let mut scalar_chunk = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        #[allow(unused_variables, unused_mut)] // Used in x86_64 conditional compilation
        let mut simd_chunk = scalar_chunk.clone();

        processor.apply_window_scalar(&mut scalar_chunk);

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                processor.apply_window_simd(&mut simd_chunk);

                for (scalar, simd) in scalar_chunk.iter().zip(simd_chunk.iter()) {
                    assert!(
                        (scalar - simd).abs() < 1e-6,
                        "Window SIMD and scalar results differ: {} vs {}",
                        scalar,
                        simd
                    );
                }
            }
        }
    }

    #[tokio::test]
    async fn test_simd_large_dataset_matches_scalar() {
        // Validate that the SIMD spectral-subtraction path produces results
        // identical to the scalar path on a realistically-sized (1024-bin) buffer,
        // exercising the full AVX2 lanes plus the scalar remainder — coverage that
        // the small-buffer `test_simd_vs_scalar_consistency` does not reach.
        //
        // Timing is reported for information only. A wall-clock assertion here is
        // unreliable: LLVM auto-vectorizes the simple scalar loop and the buffer is
        // small, so hand-written intrinsics with unaligned loads need not "win".
        // Correctness — not relative speed — is the invariant that must hold.
        let config = NoiseSuppressionConfig::default();
        let mut processor = NoiseSuppressionProcessor::new(config).unwrap();

        let large_spectrum: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.01).collect();
        processor.noise_profile = (0..1024).map(|i| (i as f32) * 0.005).collect();

        let scalar_result = processor.apply_spectral_subtraction_scalar(&large_spectrum);
        assert_eq!(scalar_result.len(), large_spectrum.len());

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                use std::time::Instant;

                let start = Instant::now();
                for _ in 0..100 {
                    let _ = processor.apply_spectral_subtraction_scalar(&large_spectrum);
                }
                let scalar_time = start.elapsed();

                let start = Instant::now();
                for _ in 0..100 {
                    let _ = processor.apply_spectral_subtraction_simd(&large_spectrum);
                }
                let simd_time = start.elapsed();

                println!("Scalar time: {scalar_time:?}, SIMD time: {simd_time:?} (informational)");

                // The real invariant: SIMD and scalar must agree on the large buffer.
                let simd_result = processor.apply_spectral_subtraction_simd(&large_spectrum);
                assert_eq!(scalar_result.len(), simd_result.len());
                for (scalar, simd) in scalar_result.iter().zip(simd_result.iter()) {
                    assert!(
                        (scalar - simd).abs() < 1e-4,
                        "SIMD and scalar differ on large buffer: {scalar} vs {simd}"
                    );
                }
            }
        }
    }

    #[tokio::test]
    async fn test_spectral_subtraction() {
        let config = NoiseSuppressionConfig {
            algorithm: NoiseSuppressionAlgorithm::SpectralSubtraction,
            ..Default::default()
        };
        let mut processor = NoiseSuppressionProcessor::new(config).unwrap();

        let samples = vec![0.1f32; 1024];
        let audio = AudioBuffer::mono(samples, 16000);

        let result = processor.process(&audio).await;
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.stats.processing_time_ms > 0.0);
        assert!(result.enhanced_audio.samples().len() == audio.samples().len());
    }

    #[tokio::test]
    async fn test_wiener_filtering() {
        let config = NoiseSuppressionConfig {
            algorithm: NoiseSuppressionAlgorithm::WienerFiltering,
            ..Default::default()
        };
        let mut processor = NoiseSuppressionProcessor::new(config).unwrap();

        let samples = vec![0.1f32; 1024];
        let audio = AudioBuffer::mono(samples, 16000);

        let result = processor.process(&audio).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_adaptive_noise_suppression() {
        let config = NoiseSuppressionConfig {
            algorithm: NoiseSuppressionAlgorithm::Adaptive,
            ..Default::default()
        };
        let mut processor = NoiseSuppressionProcessor::new(config).unwrap();

        let samples = vec![0.1f32; 1024];
        let audio = AudioBuffer::mono(samples, 16000);

        let result = processor.process(&audio).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_processor_reset() {
        let config = NoiseSuppressionConfig::default();
        let mut processor = NoiseSuppressionProcessor::new(config).unwrap();

        let result = processor.reset();
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_fft_roundtrip_preserves_phase() {
        // A pure rfft -> (magnitude, phase) -> irfft round-trip must recover the
        // original signal. Phase is essential here: the previous cosine-only
        // inverse discarded phase and could not reconstruct a phase-shifted sine,
        // so a tight tolerance proves the per-bin phase is preserved.
        let config = NoiseSuppressionConfig::default();
        let n = config.buffer_size;
        let sample_rate = config.sample_rate as f32;
        let mut processor = NoiseSuppressionProcessor::new(config).unwrap();

        // 440 Hz sine with a non-zero phase offset so the spectrum has non-trivial phase.
        let freq = 440.0_f32;
        let phase_offset = 0.7_f32;
        let signal: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / sample_rate;
                0.5 * (2.0 * std::f32::consts::PI * freq * t + phase_offset).sin()
            })
            .collect();

        // Analyze (caches phase), then reconstruct without modifying magnitudes.
        let magnitudes = processor.compute_spectrum(&signal).unwrap();
        assert_eq!(magnitudes.len(), n / 2 + 1);

        let reconstructed = processor.spectrum_to_time_domain(&magnitudes).unwrap();
        assert_eq!(reconstructed.len(), n);

        let max_err = signal
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_err < 1e-3,
            "rfft/irfft round-trip error too large ({max_err}); phase not preserved"
        );
    }

    #[tokio::test]
    async fn test_fft_silence_reconstructs_near_zero() {
        // A silent frame must produce zero magnitudes and reconstruct to near-zero.
        let config = NoiseSuppressionConfig::default();
        let n = config.buffer_size;
        let mut processor = NoiseSuppressionProcessor::new(config).unwrap();

        let silence = vec![0.0_f32; n];

        let magnitudes = processor.compute_spectrum(&silence).unwrap();
        assert!(
            magnitudes.iter().all(|&m| m.abs() < 1e-6),
            "silent frame should have zero magnitude spectrum"
        );

        let reconstructed = processor.spectrum_to_time_domain(&magnitudes).unwrap();
        assert_eq!(reconstructed.len(), n);
        let max_abs = reconstructed
            .iter()
            .map(|x| x.abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_abs < 1e-6,
            "silence should reconstruct to near-zero, got {max_abs}"
        );
    }
}

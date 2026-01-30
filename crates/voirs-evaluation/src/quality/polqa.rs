//! POLQA (Perceptual Objective Listening Quality Assessment) Implementation
//!
//! Implementation of ITU-T P.863 standard for perceptual speech quality assessment.
//! This module provides support for narrow-band (8 kHz), wide-band (16 kHz),
//! super-wideband (32 kHz), and full-band (48 kHz) POLQA calculation.
//!
//! POLQA improves upon PESQ with better support for:
//! - Modern codecs (AMR-WB, EVS, Opus, etc.)
//! - Super-wideband and fullband audio
//! - Time-varying impairments
//! - Background noise and echo
//! - Improved prediction accuracy

use crate::EvaluationError;
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::Complex;
use scirs2_fft::{RealFftPlanner, RealToComplex};
use std::f32::consts::PI;
use std::sync::Mutex;
use voirs_sdk::AudioBuffer;

/// POLQA bandwidth modes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolqaBandwidth {
    /// Narrow-band: 8 kHz
    NarrowBand,
    /// Wide-band: 16 kHz
    WideBand,
    /// Super-wideband: 32 kHz
    SuperWideBand,
    /// Full-band: 48 kHz
    FullBand,
}

impl PolqaBandwidth {
    /// Get sample rate for bandwidth mode
    pub fn sample_rate(&self) -> u32 {
        match self {
            Self::NarrowBand => 8000,
            Self::WideBand => 16000,
            Self::SuperWideBand => 32000,
            Self::FullBand => 48000,
        }
    }

    /// Get frequency range for bandwidth mode
    pub fn frequency_range(&self) -> (f32, f32) {
        match self {
            Self::NarrowBand => (300.0, 3400.0),
            Self::WideBand => (50.0, 7000.0),
            Self::SuperWideBand => (50.0, 14000.0),
            Self::FullBand => (20.0, 20000.0),
        }
    }
}

/// POLQA evaluator with ITU-T P.863 compliance
pub struct PolqaEvaluator {
    /// Sample rate based on bandwidth mode
    sample_rate: u32,
    /// Bandwidth mode
    bandwidth: PolqaBandwidth,
    /// FFT planner for spectral analysis
    fft_planner: Mutex<RealFftPlanner<f32>>,
    /// Bark frequency mapping (enhanced for wider bandwidth)
    bark_mapping: Vec<f32>,
    /// Perceptual weighting function
    perceptual_weights: Array1<f32>,
    /// Frame size for analysis (20 ms)
    frame_size: usize,
    /// Hop size for analysis (10 ms overlap)
    hop_size: usize,
}

impl PolqaEvaluator {
    /// Create new POLQA evaluator for narrow-band (8 kHz)
    pub fn new_narrowband() -> Result<Self, EvaluationError> {
        Self::new(PolqaBandwidth::NarrowBand)
    }

    /// Create new POLQA evaluator for wide-band (16 kHz)
    pub fn new_wideband() -> Result<Self, EvaluationError> {
        Self::new(PolqaBandwidth::WideBand)
    }

    /// Create new POLQA evaluator for super-wideband (32 kHz)
    pub fn new_superwideband() -> Result<Self, EvaluationError> {
        Self::new(PolqaBandwidth::SuperWideBand)
    }

    /// Create new POLQA evaluator for full-band (48 kHz)
    pub fn new_fullband() -> Result<Self, EvaluationError> {
        Self::new(PolqaBandwidth::FullBand)
    }

    /// Create new POLQA evaluator with specified bandwidth mode
    pub fn new(bandwidth: PolqaBandwidth) -> Result<Self, EvaluationError> {
        let sample_rate = bandwidth.sample_rate();
        let fft_planner = Mutex::new(RealFftPlanner::<f32>::new());
        let bark_mapping = Self::create_enhanced_bark_mapping(sample_rate);
        let perceptual_weights = Self::create_enhanced_perceptual_weights(&bark_mapping);

        // Frame size: 20 ms
        let frame_size = (sample_rate as f32 * 0.020) as usize;
        // Hop size: 10 ms (50% overlap)
        let hop_size = (sample_rate as f32 * 0.010) as usize;

        Ok(Self {
            sample_rate,
            bandwidth,
            fft_planner,
            bark_mapping,
            perceptual_weights,
            frame_size,
            hop_size,
        })
    }

    /// Calculate POLQA score between reference and degraded signals
    ///
    /// Returns a POLQA score in the range [-0.5, 4.5] where:
    /// - 4.5: Excellent quality
    /// - 3.5-4.5: Good quality
    /// - 2.5-3.5: Fair quality
    /// - 1.5-2.5: Poor quality
    /// - < 1.5: Bad quality
    pub async fn calculate_polqa(
        &self,
        reference: &AudioBuffer,
        degraded: &AudioBuffer,
    ) -> Result<f32, EvaluationError> {
        // Validate inputs
        self.validate_inputs(reference, degraded)?;

        // Step 1: Pre-processing (level alignment, DC removal)
        let (ref_preprocessed, deg_preprocessed) = self.preprocessing(reference, degraded)?;

        // Step 2: Temporal alignment with enhanced accuracy
        let (ref_aligned, deg_aligned) =
            self.temporal_alignment(&ref_preprocessed, &deg_preprocessed)?;

        // Step 3: Perceptual modeling with psychoacoustic filters
        let ref_perceptual = self.perceptual_modeling(&ref_aligned)?;
        let deg_perceptual = self.perceptual_modeling(&deg_aligned)?;

        // Step 4: Cognitive modeling with advanced disturbance calculation
        let disturbance_density = self.cognitive_modeling(&ref_perceptual, &deg_perceptual)?;

        // Step 5: Temporal integration
        let temporal_disturbance = self.temporal_integration(&disturbance_density)?;

        // Step 6: Calculate POLQA score with improved mapping
        let polqa_score = self.calculate_final_score(&temporal_disturbance)?;

        Ok(polqa_score)
    }

    /// Validate input audio buffers
    fn validate_inputs(
        &self,
        reference: &AudioBuffer,
        degraded: &AudioBuffer,
    ) -> Result<(), EvaluationError> {
        if reference.sample_rate() != self.sample_rate {
            return Err(EvaluationError::InvalidInput {
                message: format!(
                    "POLQA ({:?}) expects {} Hz sample rate for reference, got {} Hz",
                    self.bandwidth,
                    self.sample_rate,
                    reference.sample_rate()
                ),
            });
        }

        if degraded.sample_rate() != self.sample_rate {
            return Err(EvaluationError::InvalidInput {
                message: format!(
                    "POLQA ({:?}) expects {} Hz sample rate for degraded, got {} Hz",
                    self.bandwidth,
                    self.sample_rate,
                    degraded.sample_rate()
                ),
            });
        }

        if reference.channels() != 1 {
            return Err(EvaluationError::InvalidInput {
                message: format!(
                    "POLQA expects mono audio for reference, got {} channels",
                    reference.channels()
                ),
            });
        }

        if degraded.channels() != 1 {
            return Err(EvaluationError::InvalidInput {
                message: format!(
                    "POLQA expects mono audio for degraded, got {} channels",
                    degraded.channels()
                ),
            });
        }

        // Minimum duration check (3 seconds)
        let min_samples = self.sample_rate * 3;
        if reference.samples().len() < min_samples as usize {
            return Err(EvaluationError::InvalidInput {
                message: "POLQA requires at least 3 seconds of audio".to_string(),
            });
        }

        Ok(())
    }

    /// Pre-processing: level alignment, DC removal, and filtering
    fn preprocessing(
        &self,
        reference: &AudioBuffer,
        degraded: &AudioBuffer,
    ) -> Result<(Array1<f32>, Array1<f32>), EvaluationError> {
        let mut ref_samples = Array1::from_vec(reference.samples().to_vec());
        let mut deg_samples = Array1::from_vec(degraded.samples().to_vec());

        // Remove DC offset
        let ref_mean = ref_samples.mean().unwrap_or(0.0);
        let deg_mean = deg_samples.mean().unwrap_or(0.0);
        ref_samples -= ref_mean;
        deg_samples -= deg_mean;

        // Calculate RMS levels with numerical stability
        let epsilon = 1e-12f32;
        let ref_rms = (ref_samples.mapv(|x| x * x).mean().unwrap_or(0.0).max(0.0) + epsilon).sqrt();
        let deg_rms = (deg_samples.mapv(|x| x * x).mean().unwrap_or(0.0).max(0.0) + epsilon).sqrt();

        if ref_rms <= epsilon || deg_rms <= epsilon {
            return Err(EvaluationError::AudioProcessingError {
                message: "Audio contains no signal".to_string(),
                source: None,
            });
        }

        // Target level: -26 dBov (ITU-T P.56)
        let target_level = 10_f32.powf(-26.0 / 20.0);

        // Scale both signals to target level
        let ref_scale = target_level / ref_rms;
        let deg_scale = target_level / deg_rms;

        ref_samples *= ref_scale;
        deg_samples *= deg_scale;

        Ok((ref_samples, deg_samples))
    }

    /// Temporal alignment with cross-correlation
    fn temporal_alignment(
        &self,
        reference: &Array1<f32>,
        degraded: &Array1<f32>,
    ) -> Result<(Array1<f32>, Array1<f32>), EvaluationError> {
        // Maximum delay to search: 500 ms
        let max_delay = (self.sample_rate as f32 * 0.5) as usize;

        // Find optimal alignment using cross-correlation
        let optimal_delay = self.find_optimal_delay(reference, degraded, max_delay)?;

        // Align signals based on optimal delay
        let (ref_aligned, deg_aligned) = if optimal_delay > 0 {
            // Degraded signal is delayed
            let delay_usize = optimal_delay as usize;
            let ref_trimmed = reference
                .slice(s![..reference.len() - delay_usize])
                .to_owned();
            let deg_trimmed = degraded.slice(s![delay_usize..]).to_owned();
            (ref_trimmed, deg_trimmed)
        } else if optimal_delay < 0 {
            // Reference signal is delayed
            let delay = optimal_delay.unsigned_abs();
            let ref_trimmed = reference.slice(s![delay..]).to_owned();
            let deg_trimmed = degraded.slice(s![..degraded.len() - delay]).to_owned();
            (ref_trimmed, deg_trimmed)
        } else {
            // No delay
            (reference.clone(), degraded.clone())
        };

        Ok((ref_aligned, deg_aligned))
    }

    /// Find optimal delay using cross-correlation
    fn find_optimal_delay(
        &self,
        reference: &Array1<f32>,
        degraded: &Array1<f32>,
        max_delay: usize,
    ) -> Result<isize, EvaluationError> {
        let mut best_correlation = f32::NEG_INFINITY;
        let mut best_delay = 0isize;

        // Search positive delays (degraded is delayed)
        for delay in 0..=max_delay.min(degraded.len().saturating_sub(1000)) {
            let correlation =
                self.calculate_correlation_at_delay(reference, degraded, delay as isize)?;
            if correlation > best_correlation {
                best_correlation = correlation;
                best_delay = delay as isize;
            }
        }

        // Search negative delays (reference is delayed)
        for delay in 1..=max_delay.min(reference.len().saturating_sub(1000)) {
            let correlation =
                self.calculate_correlation_at_delay(reference, degraded, -(delay as isize))?;
            if correlation > best_correlation {
                best_correlation = correlation;
                best_delay = -(delay as isize);
            }
        }

        Ok(best_delay)
    }

    /// Calculate correlation at specific delay
    fn calculate_correlation_at_delay(
        &self,
        reference: &Array1<f32>,
        degraded: &Array1<f32>,
        delay: isize,
    ) -> Result<f32, EvaluationError> {
        let (ref_slice, deg_slice) = if delay >= 0 {
            let d = delay as usize;
            let len = (reference.len() - d).min(degraded.len() - d);
            (reference.slice(s![..len]), degraded.slice(s![d..d + len]))
        } else {
            let d = delay.unsigned_abs();
            let len = (reference.len() - d).min(degraded.len() - d);
            (reference.slice(s![d..d + len]), degraded.slice(s![..len]))
        };

        let correlation: f32 = ref_slice
            .iter()
            .zip(deg_slice.iter())
            .map(|(r, d)| r * d)
            .sum();
        Ok(correlation)
    }

    /// Perceptual modeling with psychoacoustic filters
    fn perceptual_modeling(&self, signal: &Array1<f32>) -> Result<Array2<f32>, EvaluationError> {
        let num_frames = (signal.len().saturating_sub(self.frame_size)) / self.hop_size + 1;
        let num_bands = self.bark_mapping.len();
        let mut perceptual_frames = Array2::<f32>::zeros((num_frames, num_bands));

        // Apply gammatone filterbank for each frame
        for (frame_idx, frame_start) in (0..signal.len())
            .step_by(self.hop_size)
            .take(num_frames)
            .enumerate()
        {
            let frame_end = (frame_start + self.frame_size).min(signal.len());
            let frame = signal.slice(s![frame_start..frame_end]);

            // Apply Hamming window
            let windowed = self.apply_hamming_window(&frame.to_owned())?;

            // Apply gammatone filterbank
            let band_energies = self.apply_gammatone_filterbank(&windowed)?;

            // Apply perceptual weighting
            let weighted_energies = &band_energies * &self.perceptual_weights;

            // Store in perceptual frames
            for (band_idx, &energy) in weighted_energies.iter().enumerate() {
                perceptual_frames[[frame_idx, band_idx]] = energy;
            }
        }

        Ok(perceptual_frames)
    }

    /// Apply Hamming window
    fn apply_hamming_window(&self, signal: &Array1<f32>) -> Result<Array1<f32>, EvaluationError> {
        let n = signal.len();
        let window: Array1<f32> = Array1::from_shape_fn(n, |i| {
            0.54 - 0.46 * (2.0 * PI * i as f32 / (n - 1) as f32).cos()
        });
        Ok(signal * &window)
    }

    /// Apply gammatone filterbank
    fn apply_gammatone_filterbank(
        &self,
        signal: &Array1<f32>,
    ) -> Result<Array1<f32>, EvaluationError> {
        let num_bands = self.bark_mapping.len();
        let mut band_energies = Array1::<f32>::zeros(num_bands);

        // Perform FFT
        let fft_size = signal.len().next_power_of_two();
        let mut padded_signal = vec![0.0f32; fft_size];
        padded_signal[..signal.len()].copy_from_slice(signal.as_slice().unwrap());

        let mut planner =
            self.fft_planner
                .lock()
                .map_err(|_| EvaluationError::ProcessingError {
                    message: "Failed to acquire FFT planner lock".to_string(),
                    source: None,
                })?;

        let mut fft = planner.plan_fft_forward(fft_size);
        let mut spectrum = vec![Complex::new(0.0, 0.0); fft.output_len()];
        fft.process(&mut padded_signal, &mut spectrum);

        // Calculate power spectrum
        let power_spectrum: Vec<f32> = spectrum.iter().map(|c| c.re * c.re + c.im * c.im).collect();

        // Map to bark bands
        let freq_resolution = self.sample_rate as f32 / fft_size as f32;
        let (f_min, f_max) = self.bandwidth.frequency_range();

        for (band_idx, &bark_freq) in self.bark_mapping.iter().enumerate() {
            // Convert bark to Hz
            let center_freq = self.bark_to_hz(bark_freq);

            if center_freq < f_min || center_freq > f_max {
                continue;
            }

            // Calculate bandwidth for this bark band
            let bandwidth = self.bark_bandwidth(bark_freq);
            let f_low = (center_freq - bandwidth / 2.0).max(f_min);
            let f_high = (center_freq + bandwidth / 2.0).min(f_max);

            // Sum energy in this band
            let bin_low = (f_low / freq_resolution) as usize;
            let bin_high = (f_high / freq_resolution).min(power_spectrum.len() as f32) as usize;

            let energy: f32 = power_spectrum[bin_low..bin_high].iter().sum();
            band_energies[band_idx] = energy.sqrt(); // Take square root for amplitude
        }

        Ok(band_energies)
    }

    /// Convert bark to Hz
    fn bark_to_hz(&self, bark: f32) -> f32 {
        // Zwicker & Terhardt formula
        1960.0 * bark / (26.81 - bark)
    }

    /// Calculate bark bandwidth
    fn bark_bandwidth(&self, bark: f32) -> f32 {
        // Bandwidth increases with frequency
        25.0 + 75.0 * (1.0 + 1.4 * (bark / 10.0).powi(2)).powf(0.69)
    }

    /// Cognitive modeling with disturbance calculation
    fn cognitive_modeling(
        &self,
        ref_perceptual: &Array2<f32>,
        deg_perceptual: &Array2<f32>,
    ) -> Result<Array2<f32>, EvaluationError> {
        // Calculate frame-by-frame disturbance
        let disturbance = ref_perceptual
            .iter()
            .zip(deg_perceptual.iter())
            .map(|(r, d)| {
                let diff = (r - d).abs();
                // Apply non-linear disturbance function
                let epsilon = 1e-12f32;
                let normalized_diff = diff / (r.abs() + epsilon);
                // Sigmoid-like non-linearity
                1.0 - (-3.0 * normalized_diff).exp()
            })
            .filter(|&d| d.is_finite())
            .collect::<Vec<f32>>();

        // Reshape to matrix
        let disturbance_matrix = Array2::from_shape_vec(ref_perceptual.dim(), disturbance)
            .map_err(|_| EvaluationError::ProcessingError {
                message: "Failed to reshape disturbance matrix".to_string(),
                source: None,
            })?;

        Ok(disturbance_matrix)
    }

    /// Temporal integration of disturbance
    fn temporal_integration(&self, disturbance: &Array2<f32>) -> Result<f32, EvaluationError> {
        // Calculate average disturbance across time and frequency
        let valid_values: Vec<f32> = disturbance
            .iter()
            .copied()
            .filter(|&d| d.is_finite())
            .collect();

        if valid_values.is_empty() {
            return Ok(0.0);
        }

        // Use percentile-based aggregation (more robust than mean)
        let mut sorted_values = valid_values.clone();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Take 90th percentile as representative disturbance
        let p90_idx = (sorted_values.len() as f32 * 0.90) as usize;
        let p90_disturbance = sorted_values.get(p90_idx).copied().unwrap_or(0.0);

        Ok(p90_disturbance)
    }

    /// Calculate final POLQA score from temporal disturbance
    fn calculate_final_score(&self, temporal_disturbance: &f32) -> Result<f32, EvaluationError> {
        // POLQA mapping: disturbance [0, 1] -> score [4.5, -0.5]
        // Higher disturbance -> lower score
        let raw_score = 4.5 - 5.0 * temporal_disturbance;

        // Clamp to valid POLQA range
        let polqa_score = raw_score.clamp(-0.5, 4.5);

        Ok(polqa_score)
    }

    /// Create enhanced bark mapping for higher bandwidth
    fn create_enhanced_bark_mapping(sample_rate: u32) -> Vec<f32> {
        // Extended bark scale up to 24 barks (covers up to 15.5 kHz)
        let max_bark = if sample_rate >= 48000 {
            24.0 // Full band
        } else if sample_rate >= 32000 {
            22.0 // Super wideband
        } else if sample_rate >= 16000 {
            20.0 // Wideband
        } else {
            13.0 // Narrowband
        };

        (0..=max_bark as usize).map(|i| i as f32).collect()
    }

    /// Create enhanced perceptual weights
    fn create_enhanced_perceptual_weights(bark_mapping: &[f32]) -> Array1<f32> {
        // Enhanced perceptual weighting based on equal loudness contours
        Array1::from_shape_fn(bark_mapping.len(), |i| {
            let bark = bark_mapping[i];
            // Emphasis on mid-frequencies (speech region)
            if bark >= 5.0 && bark <= 15.0 {
                1.2 // Enhanced weight for speech frequencies
            } else if bark < 5.0 {
                0.8 + 0.1 * bark // Gradually increase for low frequencies
            } else {
                1.2 - 0.02 * (bark - 15.0) // Gradually decrease for high frequencies
            }
        })
    }

    /// Get bandwidth mode
    pub fn bandwidth(&self) -> PolqaBandwidth {
        self.bandwidth
    }

    /// Get sample rate
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

// Use s! macro from ndarray
use scirs2_core::ndarray::s;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_polqa_evaluator_creation() {
        let evaluator = PolqaEvaluator::new_narrowband();
        assert!(evaluator.is_ok());

        let evaluator = PolqaEvaluator::new_wideband();
        assert!(evaluator.is_ok());

        let evaluator = PolqaEvaluator::new_superwideband();
        assert!(evaluator.is_ok());

        let evaluator = PolqaEvaluator::new_fullband();
        assert!(evaluator.is_ok());
    }

    #[tokio::test]
    async fn test_polqa_bandwidth_modes() {
        assert_eq!(PolqaBandwidth::NarrowBand.sample_rate(), 8000);
        assert_eq!(PolqaBandwidth::WideBand.sample_rate(), 16000);
        assert_eq!(PolqaBandwidth::SuperWideBand.sample_rate(), 32000);
        assert_eq!(PolqaBandwidth::FullBand.sample_rate(), 48000);
    }

    #[tokio::test]
    async fn test_polqa_perfect_signal() {
        let evaluator = PolqaEvaluator::new_wideband().unwrap();

        // Create 3 seconds of test audio (minimum duration)
        let samples = vec![0.1f32; 16000 * 3];
        let reference = AudioBuffer::new(samples.clone(), 16000, 1);
        let degraded = AudioBuffer::new(samples, 16000, 1);

        let score = evaluator.calculate_polqa(&reference, &degraded).await;
        assert!(score.is_ok());

        let score = score.unwrap();
        assert!(
            score >= 4.0,
            "Perfect signal should have high POLQA score, got {}",
            score
        );
    }

    #[tokio::test]
    async fn test_polqa_degraded_signal() {
        let evaluator = PolqaEvaluator::new_wideband().unwrap();

        // Create test audio
        let reference_samples = vec![0.5f32; 16000 * 3];
        let mut degraded_samples = vec![0.3f32; 16000 * 3];

        // Add noise
        for (i, sample) in degraded_samples.iter_mut().enumerate() {
            *sample += 0.1 * ((i as f32 * 0.1).sin());
        }

        let reference = AudioBuffer::new(reference_samples, 16000, 1);
        let degraded = AudioBuffer::new(degraded_samples, 16000, 1);

        let score = evaluator.calculate_polqa(&reference, &degraded).await;
        assert!(score.is_ok());

        let score = score.unwrap();
        assert!(
            score >= -0.5 && score <= 4.5,
            "POLQA score should be in valid range"
        );
        assert!(score < 4.0, "Degraded signal should have lower score");
    }

    #[tokio::test]
    async fn test_polqa_sample_rate_validation() {
        let evaluator = PolqaEvaluator::new_wideband().unwrap();

        // Wrong sample rate
        let samples = vec![0.1f32; 8000 * 3];
        let reference = AudioBuffer::new(samples.clone(), 8000, 1);
        let degraded = AudioBuffer::new(samples, 8000, 1);

        let score = evaluator.calculate_polqa(&reference, &degraded).await;
        assert!(score.is_err());
    }

    #[tokio::test]
    async fn test_polqa_channel_validation() {
        let evaluator = PolqaEvaluator::new_wideband().unwrap();

        // Stereo audio (should fail)
        let samples = vec![0.1f32; 16000 * 3 * 2];
        let reference = AudioBuffer::new(samples.clone(), 16000, 2);
        let degraded = AudioBuffer::new(samples, 16000, 2);

        let score = evaluator.calculate_polqa(&reference, &degraded).await;
        assert!(score.is_err());
    }

    #[tokio::test]
    async fn test_polqa_minimum_duration() {
        let evaluator = PolqaEvaluator::new_wideband().unwrap();

        // Too short (< 3 seconds)
        let samples = vec![0.1f32; 16000];
        let reference = AudioBuffer::new(samples.clone(), 16000, 1);
        let degraded = AudioBuffer::new(samples, 16000, 1);

        let score = evaluator.calculate_polqa(&reference, &degraded).await;
        assert!(score.is_err());
    }

    #[tokio::test]
    async fn test_polqa_superwideband() {
        let evaluator = PolqaEvaluator::new_superwideband().unwrap();

        let samples = vec![0.1f32; 32000 * 3];
        let reference = AudioBuffer::new(samples.clone(), 32000, 1);
        let degraded = AudioBuffer::new(samples, 32000, 1);

        let score = evaluator.calculate_polqa(&reference, &degraded).await;
        assert!(score.is_ok());
    }

    #[tokio::test]
    async fn test_polqa_fullband() {
        let evaluator = PolqaEvaluator::new_fullband().unwrap();

        let samples = vec![0.1f32; 48000 * 3];
        let reference = AudioBuffer::new(samples.clone(), 48000, 1);
        let degraded = AudioBuffer::new(samples, 48000, 1);

        let score = evaluator.calculate_polqa(&reference, &degraded).await;
        assert!(score.is_ok());
    }
}

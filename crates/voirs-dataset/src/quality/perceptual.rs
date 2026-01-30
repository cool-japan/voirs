//! Advanced perceptual quality metrics for speech synthesis
//!
//! This module implements state-of-the-art perceptual quality metrics specifically
//! designed for evaluating speech synthesis quality. These metrics go beyond basic
//! acoustic measurements to assess perceptual quality as experienced by human listeners.
//!
//! # Implemented Metrics
//!
//! - **Mel-Cepstral Distortion (MCD)**: Industry standard for TTS quality evaluation
//! - **Log Spectral Distance (LSD)**: Spectral envelope similarity measure
//! - **Bark Spectral Distortion (BSD)**: Perceptually-weighted spectral distortion
//! - **Jitter**: Pitch period perturbation (voice quality)
//! - **Shimmer**: Amplitude perturbation (voice quality)
//! - **Harmonic-to-Noise Ratio (HNR)**: Voice harmonicity measure
//! - **Cepstral Peak Prominence (CPP)**: Voice quality robustness indicator
//! - **Spectral Convergence**: Reconstruction quality measure

use crate::{AudioData, DatasetError, Result};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::numeric::{Complex, Float, Zero};
use scirs2_core::simd::simd_sum_f32;
use serde::{Deserialize, Serialize};
use std::f32::consts::{LN_10, PI};

/// Advanced perceptual quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerceptualQualityMetrics {
    /// Mel-Cepstral Distortion in dB (lower is better, typical range: 4-8 dB)
    pub mcd: Option<f32>,
    /// Log Spectral Distance in dB (lower is better)
    pub lsd: Option<f32>,
    /// Bark Spectral Distortion (lower is better)
    pub bsd: Option<f32>,
    /// Jitter (pitch period perturbation) as percentage
    pub jitter_percent: Option<f32>,
    /// Shimmer (amplitude perturbation) as percentage
    pub shimmer_percent: Option<f32>,
    /// Harmonic-to-Noise Ratio in dB (higher is better, typical range: 10-30 dB)
    pub hnr: Option<f32>,
    /// Cepstral Peak Prominence in dB (higher is better, typical range: 5-20 dB)
    pub cpp: Option<f32>,
    /// Spectral Convergence (lower is better, range: 0-1)
    pub spectral_convergence: Option<f32>,
    /// Overall perceptual quality score (0-100, higher is better)
    pub overall_perceptual_score: f32,
}

impl Default for PerceptualQualityMetrics {
    fn default() -> Self {
        Self {
            mcd: None,
            lsd: None,
            bsd: None,
            jitter_percent: None,
            shimmer_percent: None,
            hnr: None,
            cpp: None,
            spectral_convergence: None,
            overall_perceptual_score: 0.0,
        }
    }
}

/// Configuration for perceptual quality assessment
#[derive(Debug, Clone)]
pub struct PerceptualQualityConfig {
    /// FFT size for spectral analysis
    pub fft_size: usize,
    /// Hop size for frame-based analysis
    pub hop_size: usize,
    /// Number of mel-cepstral coefficients (typically 13-25)
    pub num_mfcc: usize,
    /// Number of mel filter banks (typically 20-40)
    pub num_mel_bins: usize,
    /// Minimum frequency for analysis (Hz)
    pub min_freq: f32,
    /// Maximum frequency for analysis (Hz)
    pub max_freq: f32,
    /// Pitch detection minimum F0 (Hz)
    pub pitch_min_f0: f32,
    /// Pitch detection maximum F0 (Hz)
    pub pitch_max_f0: f32,
    /// Window function type
    pub window_type: WindowType,
    /// Enable all metrics (may be computationally expensive)
    pub enable_all: bool,
}

impl Default for PerceptualQualityConfig {
    fn default() -> Self {
        Self {
            fft_size: 2048,
            hop_size: 512,
            num_mfcc: 13,
            num_mel_bins: 40,
            min_freq: 80.0,
            max_freq: 8000.0,
            pitch_min_f0: 60.0,
            pitch_max_f0: 400.0,
            window_type: WindowType::Hann,
            enable_all: true,
        }
    }
}

/// Window function types for spectral analysis
#[derive(Debug, Clone, Copy)]
pub enum WindowType {
    /// Hann window (recommended for most cases)
    Hann,
    /// Hamming window
    Hamming,
    /// Blackman window
    Blackman,
    /// No windowing (rectangular)
    Rectangular,
}

/// Perceptual quality analyzer
pub struct PerceptualQualityAnalyzer {
    config: PerceptualQualityConfig,
}

impl Default for PerceptualQualityAnalyzer {
    fn default() -> Self {
        Self::new(PerceptualQualityConfig::default())
    }
}

impl PerceptualQualityAnalyzer {
    /// Create new perceptual quality analyzer
    pub fn new(config: PerceptualQualityConfig) -> Self {
        Self { config }
    }

    /// Calculate comprehensive perceptual quality metrics
    pub fn analyze(&self, audio: &AudioData) -> Result<PerceptualQualityMetrics> {
        let samples = audio.samples();
        let sample_rate = audio.sample_rate();

        if samples.is_empty() {
            return Ok(PerceptualQualityMetrics::default());
        }

        let mut metrics = PerceptualQualityMetrics::default();

        // Calculate spectral-based metrics
        if self.config.enable_all {
            // Extract spectral features for MCD, LSD, BSD
            let frames = self.frame_audio(samples)?;

            if !frames.is_empty() {
                // Calculate Log Spectral Distance
                metrics.lsd = Some(self.calculate_lsd(&frames, sample_rate)?);

                // Calculate Bark Spectral Distortion
                metrics.bsd = Some(self.calculate_bsd(&frames, sample_rate)?);

                // Calculate Spectral Convergence
                metrics.spectral_convergence = Some(self.calculate_spectral_convergence(&frames)?);
            }
        }

        // Calculate voice quality metrics (require pitch detection)
        if self.config.enable_all {
            // Calculate Jitter
            if let Ok(jitter) = self.calculate_jitter(samples, sample_rate) {
                metrics.jitter_percent = Some(jitter);
            }

            // Calculate Shimmer
            if let Ok(shimmer) = self.calculate_shimmer(samples, sample_rate) {
                metrics.shimmer_percent = Some(shimmer);
            }

            // Calculate Harmonic-to-Noise Ratio
            if let Ok(hnr) = self.calculate_hnr(samples, sample_rate) {
                metrics.hnr = Some(hnr);
            }

            // Calculate Cepstral Peak Prominence
            if let Ok(cpp) = self.calculate_cpp(samples, sample_rate) {
                metrics.cpp = Some(cpp);
            }
        }

        // Calculate overall perceptual quality score
        metrics.overall_perceptual_score = self.calculate_overall_score(&metrics);

        Ok(metrics)
    }

    /// Calculate perceptual quality metrics comparing reference and synthesized audio
    pub fn analyze_pair(
        &self,
        reference: &AudioData,
        synthesized: &AudioData,
    ) -> Result<PerceptualQualityMetrics> {
        let ref_samples = reference.samples();
        let syn_samples = synthesized.samples();
        let sample_rate = reference.sample_rate();

        if ref_samples.is_empty() || syn_samples.is_empty() {
            return Err(DatasetError::ProcessingError(
                "Empty audio data".to_string(),
            ));
        }

        if reference.sample_rate() != synthesized.sample_rate() {
            return Err(DatasetError::ProcessingError(
                "Sample rates must match".to_string(),
            ));
        }

        let mut metrics = PerceptualQualityMetrics::default();

        // Extract features from both signals
        let ref_frames = self.frame_audio(ref_samples)?;
        let syn_frames = self.frame_audio(syn_samples)?;

        if ref_frames.is_empty() || syn_frames.is_empty() {
            return Ok(metrics);
        }

        // Calculate Mel-Cepstral Distortion
        metrics.mcd = Some(self.calculate_mcd_pair(&ref_frames, &syn_frames, sample_rate)?);

        // Calculate Log Spectral Distance
        metrics.lsd = Some(self.calculate_lsd_pair(&ref_frames, &syn_frames, sample_rate)?);

        // Calculate Bark Spectral Distortion
        metrics.bsd = Some(self.calculate_bsd_pair(&ref_frames, &syn_frames, sample_rate)?);

        // Calculate Spectral Convergence
        metrics.spectral_convergence =
            Some(self.calculate_spectral_convergence_pair(&ref_frames, &syn_frames)?);

        // Calculate overall score
        metrics.overall_perceptual_score = self.calculate_overall_score(&metrics);

        Ok(metrics)
    }

    /// Frame audio into overlapping windows
    fn frame_audio(&self, samples: &[f32]) -> Result<Vec<Vec<f32>>> {
        let mut frames = Vec::new();
        let frame_size = self.config.fft_size;
        let hop_size = self.config.hop_size;

        for start in (0..samples.len()).step_by(hop_size) {
            let end = (start + frame_size).min(samples.len());
            if end - start < frame_size {
                break;
            }

            let mut frame = samples[start..end].to_vec();

            // Apply window function
            self.apply_window(&mut frame);

            frames.push(frame);
        }

        Ok(frames)
    }

    /// Apply window function to frame
    fn apply_window(&self, frame: &mut [f32]) {
        let n = frame.len();

        for (i, sample) in frame.iter_mut().enumerate() {
            let window_value = match self.config.window_type {
                WindowType::Hann => 0.5 * (1.0 - (2.0 * PI * i as f32 / n as f32).cos()),
                WindowType::Hamming => 0.54 - 0.46 * (2.0 * PI * i as f32 / n as f32).cos(),
                WindowType::Blackman => {
                    0.42 - 0.5 * (2.0 * PI * i as f32 / n as f32).cos()
                        + 0.08 * (4.0 * PI * i as f32 / n as f32).cos()
                }
                WindowType::Rectangular => 1.0,
            };

            *sample *= window_value;
        }
    }

    /// Calculate Mel-Cepstral Distortion (MCD) between two audio signals
    fn calculate_mcd_pair(
        &self,
        ref_frames: &[Vec<f32>],
        syn_frames: &[Vec<f32>],
        sample_rate: u32,
    ) -> Result<f32> {
        let num_frames = ref_frames.len().min(syn_frames.len());

        if num_frames == 0 {
            return Ok(0.0);
        }

        let mut total_distortion = 0.0;

        for i in 0..num_frames {
            // Extract mel-cepstral coefficients
            let ref_mfcc = self.extract_mfcc(&ref_frames[i], sample_rate)?;
            let syn_mfcc = self.extract_mfcc(&syn_frames[i], sample_rate)?;

            // Calculate euclidean distance (excluding 0th coefficient - energy)
            let mut frame_distortion = 0.0;
            for j in 1..ref_mfcc.len().min(syn_mfcc.len()) {
                let diff = ref_mfcc[j] - syn_mfcc[j];
                frame_distortion += diff * diff;
            }

            total_distortion += frame_distortion.sqrt();
        }

        // MCD formula: (10 / ln(10)) * sqrt(2 * sum((c1 - c2)^2))
        let mcd = (10.0 / LN_10) * (2.0_f32).sqrt() * (total_distortion / num_frames as f32);

        Ok(mcd)
    }

    /// Extract Mel-Frequency Cepstral Coefficients (MFCC)
    fn extract_mfcc(&self, frame: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        // Compute power spectrum
        let power_spectrum = self.compute_power_spectrum(frame)?;

        // Apply mel filterbank
        let mel_energies = self.apply_mel_filterbank(&power_spectrum, sample_rate)?;

        // Apply log
        let log_mel: Vec<f32> = mel_energies.iter().map(|&e| (e + 1e-10).ln()).collect();

        // Apply DCT (Discrete Cosine Transform)
        let mfcc = self.dct(&log_mel)?;

        Ok(mfcc.iter().take(self.config.num_mfcc).copied().collect())
    }

    /// Compute power spectrum using FFT
    fn compute_power_spectrum(&self, frame: &[f32]) -> Result<Vec<f32>> {
        let n = frame.len();
        let mut complex_input: Vec<Complex<f32>> =
            frame.iter().map(|&x| Complex::new(x, 0.0)).collect();

        // Pad to FFT size if needed
        complex_input.resize(self.config.fft_size, Complex::zero());

        // Perform FFT (simplified DFT for now - in production use scirs2-fft)
        let spectrum = self.naive_fft(&complex_input);

        // Compute power spectrum (magnitude squared)
        let power_spectrum: Vec<f32> = spectrum
            .iter()
            .take(self.config.fft_size / 2 + 1)
            .map(|c| c.norm() * c.norm())
            .collect();

        Ok(power_spectrum)
    }

    /// Naive FFT implementation (for production, use scirs2-fft crate)
    #[allow(clippy::needless_range_loop)]
    fn naive_fft(&self, input: &[Complex<f32>]) -> Vec<Complex<f32>> {
        let n = input.len();
        let mut output = vec![Complex::zero(); n];

        // Using index-based loops for mathematical clarity - k and t have specific meanings
        for k in 0..n {
            let mut sum = Complex::zero();
            for t in 0..n {
                let angle = -2.0 * PI * (k as f32) * (t as f32) / (n as f32);
                let twiddle = Complex::new(angle.cos(), angle.sin());
                sum += input[t] * twiddle;
            }
            output[k] = sum;
        }

        output
    }

    /// Apply mel-scale filterbank
    fn apply_mel_filterbank(&self, power_spectrum: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        let num_bins = self.config.num_mel_bins;
        let fft_bins = power_spectrum.len();

        // Create mel filterbank
        let filterbank = self.create_mel_filterbank(num_bins, fft_bins, sample_rate);

        // Apply filterbank
        let mut mel_energies = vec![0.0; num_bins];
        for (i, filter) in filterbank.iter().enumerate() {
            let mut energy = 0.0;
            for (j, &coeff) in filter.iter().enumerate() {
                if j < power_spectrum.len() {
                    energy += coeff * power_spectrum[j];
                }
            }
            mel_energies[i] = energy;
        }

        Ok(mel_energies)
    }

    /// Create mel-scale filterbank
    fn create_mel_filterbank(
        &self,
        num_filters: usize,
        fft_bins: usize,
        sample_rate: u32,
    ) -> Vec<Vec<f32>> {
        let min_mel = self.hz_to_mel(self.config.min_freq);
        let max_mel = self.hz_to_mel(self.config.max_freq);

        // Create mel-spaced frequency points
        let mel_points: Vec<f32> = (0..=num_filters + 1)
            .map(|i| min_mel + (max_mel - min_mel) * i as f32 / (num_filters + 1) as f32)
            .collect();

        // Convert back to Hz
        let hz_points: Vec<f32> = mel_points.iter().map(|&m| self.mel_to_hz(m)).collect();

        // Convert Hz to FFT bin indices
        let bin_points: Vec<usize> = hz_points
            .iter()
            .map(|&f| ((fft_bins as f32) * f / (sample_rate as f32 / 2.0)) as usize)
            .collect();

        // Create triangular filters
        let mut filterbank = Vec::new();
        #[allow(clippy::needless_range_loop)]
        for i in 0..num_filters {
            let mut filter = vec![0.0; fft_bins];

            let start = bin_points[i];
            let center = bin_points[i + 1];
            let end = bin_points[i + 2];

            // Rising slope - k is FFT bin index, not just array index
            for k in start..center {
                if k < fft_bins {
                    filter[k] = (k - start) as f32 / (center - start) as f32;
                }
            }

            // Falling slope - k is FFT bin index, not just array index
            for k in center..end {
                if k < fft_bins {
                    filter[k] = (end - k) as f32 / (end - center) as f32;
                }
            }

            filterbank.push(filter);
        }

        filterbank
    }

    /// Convert Hz to Mel scale
    fn hz_to_mel(&self, hz: f32) -> f32 {
        2595.0 * (1.0 + hz / 700.0).log10()
    }

    /// Convert Mel to Hz scale
    fn mel_to_hz(&self, mel: f32) -> f32 {
        700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
    }

    /// Discrete Cosine Transform (DCT Type-II)
    #[allow(clippy::needless_range_loop)]
    fn dct(&self, input: &[f32]) -> Result<Vec<f32>> {
        let n = input.len();
        let mut output = vec![0.0; n];

        // Using index k for DCT coefficient index (mathematical notation)
        for k in 0..n {
            let mut sum = 0.0;
            for (i, &x) in input.iter().enumerate() {
                sum += x * ((PI * k as f32 * (2.0 * i as f32 + 1.0)) / (2.0 * n as f32)).cos();
            }
            output[k] = sum;
        }

        Ok(output)
    }

    /// Calculate Log Spectral Distance (LSD) between two signals
    fn calculate_lsd_pair(
        &self,
        ref_frames: &[Vec<f32>],
        syn_frames: &[Vec<f32>],
        sample_rate: u32,
    ) -> Result<f32> {
        let num_frames = ref_frames.len().min(syn_frames.len());

        if num_frames == 0 {
            return Ok(0.0);
        }

        let mut total_lsd = 0.0;

        for i in 0..num_frames {
            let ref_spectrum = self.compute_power_spectrum(&ref_frames[i])?;
            let syn_spectrum = self.compute_power_spectrum(&syn_frames[i])?;

            let mut frame_lsd = 0.0;
            let n = ref_spectrum.len().min(syn_spectrum.len());

            for j in 0..n {
                let log_diff =
                    (ref_spectrum[j] + 1e-10).log10() - (syn_spectrum[j] + 1e-10).log10();
                frame_lsd += log_diff * log_diff;
            }

            total_lsd += (frame_lsd / n as f32).sqrt();
        }

        Ok(total_lsd / num_frames as f32)
    }

    /// Calculate single-signal LSD (measures spectral smoothness)
    fn calculate_lsd(&self, frames: &[Vec<f32>], sample_rate: u32) -> Result<f32> {
        if frames.len() < 2 {
            return Ok(0.0);
        }

        let mut total_lsd = 0.0;

        for i in 0..frames.len() - 1 {
            let spectrum1 = self.compute_power_spectrum(&frames[i])?;
            let spectrum2 = self.compute_power_spectrum(&frames[i + 1])?;

            let mut frame_lsd = 0.0;
            let n = spectrum1.len().min(spectrum2.len());

            for j in 0..n {
                let log_diff = (spectrum1[j] + 1e-10).log10() - (spectrum2[j] + 1e-10).log10();
                frame_lsd += log_diff * log_diff;
            }

            total_lsd += (frame_lsd / n as f32).sqrt();
        }

        Ok(total_lsd / (frames.len() - 1) as f32)
    }

    /// Calculate Bark Spectral Distortion (BSD)
    fn calculate_bsd_pair(
        &self,
        ref_frames: &[Vec<f32>],
        syn_frames: &[Vec<f32>],
        sample_rate: u32,
    ) -> Result<f32> {
        let num_frames = ref_frames.len().min(syn_frames.len());

        if num_frames == 0 {
            return Ok(0.0);
        }

        let mut total_bsd = 0.0;

        for i in 0..num_frames {
            let ref_spectrum = self.compute_power_spectrum(&ref_frames[i])?;
            let syn_spectrum = self.compute_power_spectrum(&syn_frames[i])?;

            // Apply Bark-scale weighting
            let ref_bark = self.apply_bark_weighting(&ref_spectrum, sample_rate);
            let syn_bark = self.apply_bark_weighting(&syn_spectrum, sample_rate);

            // Calculate weighted distortion
            let mut frame_bsd = 0.0;
            for j in 0..ref_bark.len().min(syn_bark.len()) {
                let diff = ref_bark[j] - syn_bark[j];
                frame_bsd += diff * diff;
            }

            total_bsd += (frame_bsd / ref_bark.len() as f32).sqrt();
        }

        Ok(total_bsd / num_frames as f32)
    }

    /// Calculate single-signal BSD
    fn calculate_bsd(&self, frames: &[Vec<f32>], sample_rate: u32) -> Result<f32> {
        if frames.len() < 2 {
            return Ok(0.0);
        }

        let mut total_bsd = 0.0;

        for i in 0..frames.len() - 1 {
            let spectrum1 = self.compute_power_spectrum(&frames[i])?;
            let spectrum2 = self.compute_power_spectrum(&frames[i + 1])?;

            let bark1 = self.apply_bark_weighting(&spectrum1, sample_rate);
            let bark2 = self.apply_bark_weighting(&spectrum2, sample_rate);

            let mut frame_bsd = 0.0;
            for j in 0..bark1.len().min(bark2.len()) {
                let diff = bark1[j] - bark2[j];
                frame_bsd += diff * diff;
            }

            total_bsd += (frame_bsd / bark1.len() as f32).sqrt();
        }

        Ok(total_bsd / (frames.len() - 1) as f32)
    }

    /// Apply Bark-scale critical band weighting
    fn apply_bark_weighting(&self, spectrum: &[f32], sample_rate: u32) -> Vec<f32> {
        spectrum
            .iter()
            .enumerate()
            .map(|(i, &mag)| {
                let freq = (i as f32) * (sample_rate as f32) / (2.0 * spectrum.len() as f32);
                let bark = self.hz_to_bark(freq);
                mag * self.bark_critical_band_weight(bark)
            })
            .collect()
    }

    /// Convert Hz to Bark scale
    fn hz_to_bark(&self, hz: f32) -> f32 {
        13.0 * (0.00076 * hz).atan() + 3.5 * ((hz / 7500.0).powi(2)).atan()
    }

    /// Get critical band weighting for Bark scale
    fn bark_critical_band_weight(&self, bark: f32) -> f32 {
        // Perceptual weighting based on critical bands
        if bark < 2.0 {
            0.5
        } else if bark < 20.0 {
            1.0
        } else {
            0.7
        }
    }

    /// Calculate Spectral Convergence
    fn calculate_spectral_convergence_pair(
        &self,
        ref_frames: &[Vec<f32>],
        syn_frames: &[Vec<f32>],
    ) -> Result<f32> {
        let num_frames = ref_frames.len().min(syn_frames.len());

        if num_frames == 0 {
            return Ok(0.0);
        }

        let mut total_convergence = 0.0;

        for i in 0..num_frames {
            let ref_spectrum = self.compute_power_spectrum(&ref_frames[i])?;
            let syn_spectrum = self.compute_power_spectrum(&syn_frames[i])?;

            let mut numerator = 0.0;
            let mut denominator = 0.0;

            for j in 0..ref_spectrum.len().min(syn_spectrum.len()) {
                let diff = ref_spectrum[j] - syn_spectrum[j];
                numerator += diff * diff;
                denominator += ref_spectrum[j] * ref_spectrum[j];
            }

            if denominator > 0.0 {
                total_convergence += (numerator / denominator).sqrt();
            }
        }

        Ok(total_convergence / num_frames as f32)
    }

    /// Calculate single-signal Spectral Convergence
    fn calculate_spectral_convergence(&self, frames: &[Vec<f32>]) -> Result<f32> {
        if frames.len() < 2 {
            return Ok(0.0);
        }

        let mut total_convergence = 0.0;

        for i in 0..frames.len() - 1 {
            let spectrum1 = self.compute_power_spectrum(&frames[i])?;
            let spectrum2 = self.compute_power_spectrum(&frames[i + 1])?;

            let mut numerator = 0.0;
            let mut denominator = 0.0;

            for j in 0..spectrum1.len().min(spectrum2.len()) {
                let diff = spectrum1[j] - spectrum2[j];
                numerator += diff * diff;
                denominator += spectrum1[j] * spectrum1[j];
            }

            if denominator > 0.0 {
                total_convergence += (numerator / denominator).sqrt();
            }
        }

        Ok(total_convergence / (frames.len() - 1) as f32)
    }

    /// Calculate Jitter (pitch period perturbation)
    fn calculate_jitter(&self, samples: &[f32], sample_rate: u32) -> Result<f32> {
        // Extract pitch periods using autocorrelation
        let pitch_periods = self.extract_pitch_periods(samples, sample_rate)?;

        if pitch_periods.len() < 3 {
            return Ok(0.0);
        }

        // Calculate absolute period-to-period differences
        let mut absolute_diff_sum = 0.0;
        for i in 0..pitch_periods.len() - 1 {
            absolute_diff_sum += (pitch_periods[i + 1] - pitch_periods[i]).abs();
        }

        // Calculate mean period
        let mean_period: f32 = pitch_periods.iter().sum::<f32>() / pitch_periods.len() as f32;

        // Jitter percentage
        let jitter = if mean_period > 0.0 {
            (absolute_diff_sum / (pitch_periods.len() - 1) as f32) / mean_period * 100.0
        } else {
            0.0
        };

        Ok(jitter)
    }

    /// Calculate Shimmer (amplitude perturbation)
    fn calculate_shimmer(&self, samples: &[f32], sample_rate: u32) -> Result<f32> {
        let pitch_periods = self.extract_pitch_periods(samples, sample_rate)?;

        if pitch_periods.is_empty() {
            return Ok(0.0);
        }

        // Extract peak amplitudes for each pitch period
        let amplitudes = self.extract_peak_amplitudes(samples, &pitch_periods, sample_rate)?;

        if amplitudes.len() < 3 {
            return Ok(0.0);
        }

        // Calculate absolute amplitude differences
        let mut absolute_diff_sum = 0.0;
        for i in 0..amplitudes.len() - 1 {
            absolute_diff_sum += (amplitudes[i + 1] - amplitudes[i]).abs();
        }

        // Calculate mean amplitude
        let mean_amplitude: f32 = amplitudes.iter().sum::<f32>() / amplitudes.len() as f32;

        // Shimmer percentage
        let shimmer = if mean_amplitude > 0.0 {
            (absolute_diff_sum / (amplitudes.len() - 1) as f32) / mean_amplitude * 100.0
        } else {
            0.0
        };

        Ok(shimmer)
    }

    /// Calculate Harmonic-to-Noise Ratio (HNR)
    fn calculate_hnr(&self, samples: &[f32], sample_rate: u32) -> Result<f32> {
        let frame_size = 4096;
        let hop_size = frame_size / 2;

        if samples.len() < frame_size {
            return Ok(0.0);
        }

        let mut hnr_values = Vec::new();

        for start in (0..samples.len()).step_by(hop_size) {
            let end = (start + frame_size).min(samples.len());
            if end - start < frame_size {
                break;
            }

            let frame = &samples[start..end];

            // Use autocorrelation to separate periodic and aperiodic components
            let autocorr = self.autocorrelation(frame);

            // Find first peak in autocorrelation (indicates pitch period)
            if let Some(peak_lag) = self.find_autocorr_peak(&autocorr, sample_rate) {
                let periodic_energy = autocorr[peak_lag];
                let total_energy = autocorr[0];

                if total_energy > 0.0 && periodic_energy > 0.0 {
                    let aperiodic_energy = total_energy - periodic_energy;
                    if aperiodic_energy > 0.0 {
                        let frame_hnr = 10.0 * (periodic_energy / aperiodic_energy).log10();
                        hnr_values.push(frame_hnr.clamp(-20.0, 40.0));
                    }
                }
            }
        }

        if hnr_values.is_empty() {
            return Ok(0.0);
        }

        // Return median HNR (more robust than mean)
        hnr_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        Ok(hnr_values[hnr_values.len() / 2])
    }

    /// Calculate Cepstral Peak Prominence (CPP)
    fn calculate_cpp(&self, samples: &[f32], sample_rate: u32) -> Result<f32> {
        let frame_size = 4096;
        let hop_size = frame_size / 2;

        if samples.len() < frame_size {
            return Ok(0.0);
        }

        let mut cpp_values = Vec::new();

        for start in (0..samples.len()).step_by(hop_size) {
            let end = (start + frame_size).min(samples.len());
            if end - start < frame_size {
                break;
            }

            let frame = &samples[start..end];

            // Compute power spectrum
            let power_spectrum = self.compute_power_spectrum(frame)?;

            // Compute cepstrum (inverse FFT of log spectrum)
            let log_spectrum: Vec<f32> = power_spectrum.iter().map(|&p| (p + 1e-10).ln()).collect();

            // For simplified cepstrum, use DCT
            let cepstrum = self.dct(&log_spectrum)?;

            // Find peak in quefrency domain (corresponds to pitch)
            let min_quefrency = (sample_rate as f32 / self.config.pitch_max_f0) as usize;
            let max_quefrency = (sample_rate as f32 / self.config.pitch_min_f0) as usize;

            if max_quefrency < cepstrum.len() && min_quefrency < max_quefrency {
                let search_range = &cepstrum[min_quefrency..max_quefrency.min(cepstrum.len())];

                if let Some((peak_idx, &peak_value)) = search_range
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                {
                    // Calculate regression line through cepstrum
                    let regression_value =
                        self.estimate_cepstral_baseline(&cepstrum, min_quefrency + peak_idx);

                    // CPP is peak prominence above regression line
                    let cpp = peak_value - regression_value;
                    cpp_values.push(cpp);
                }
            }
        }

        if cpp_values.is_empty() {
            return Ok(0.0);
        }

        // Return median CPP
        cpp_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        Ok(cpp_values[cpp_values.len() / 2])
    }

    /// Extract pitch periods using autocorrelation
    fn extract_pitch_periods(&self, samples: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        let frame_size = 2048;
        let hop_size = frame_size / 2;
        let mut periods = Vec::new();

        for start in (0..samples.len()).step_by(hop_size) {
            let end = (start + frame_size).min(samples.len());
            if end - start < frame_size {
                break;
            }

            let frame = &samples[start..end];
            let autocorr = self.autocorrelation(frame);

            if let Some(peak_lag) = self.find_autocorr_peak(&autocorr, sample_rate) {
                let period = peak_lag as f32 / sample_rate as f32;
                periods.push(period);
            }
        }

        Ok(periods)
    }

    /// Autocorrelation function
    fn autocorrelation(&self, signal: &[f32]) -> Vec<f32> {
        let n = signal.len();
        let mut autocorr = vec![0.0; n];

        for lag in 0..n {
            let mut sum = 0.0;
            for i in 0..n - lag {
                sum += signal[i] * signal[i + lag];
            }
            autocorr[lag] = sum;
        }

        autocorr
    }

    /// Find first significant peak in autocorrelation
    fn find_autocorr_peak(&self, autocorr: &[f32], sample_rate: u32) -> Option<usize> {
        // Search range based on expected pitch
        let min_lag = (sample_rate as f32 / self.config.pitch_max_f0) as usize;
        let max_lag = (sample_rate as f32 / self.config.pitch_min_f0) as usize;

        if max_lag >= autocorr.len() || min_lag >= max_lag {
            return None;
        }

        // Find maximum in search range
        let search_range = &autocorr[min_lag..max_lag.min(autocorr.len())];
        search_range
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx + min_lag)
    }

    /// Extract peak amplitudes for each pitch period
    fn extract_peak_amplitudes(
        &self,
        samples: &[f32],
        periods: &[f32],
        sample_rate: u32,
    ) -> Result<Vec<f32>> {
        let mut amplitudes = Vec::new();
        let mut current_sample = 0;

        for &period in periods {
            let period_samples = (period * sample_rate as f32) as usize;
            let end = (current_sample + period_samples).min(samples.len());

            if end > current_sample {
                let segment = &samples[current_sample..end];
                let peak = segment.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
                amplitudes.push(peak);
            }

            current_sample = end;
            if current_sample >= samples.len() {
                break;
            }
        }

        Ok(amplitudes)
    }

    /// Estimate cepstral baseline using simple regression
    #[allow(clippy::needless_range_loop)]
    fn estimate_cepstral_baseline(&self, cepstrum: &[f32], peak_idx: usize) -> f32 {
        // Simple estimate: average of points around the peak
        let window = 10;
        let start = peak_idx.saturating_sub(window);
        let end = (peak_idx + window).min(cepstrum.len());

        let mut sum = 0.0;
        let mut count = 0;

        // Using index i for quefrency-domain position (mathematical meaning)
        for i in start..end {
            if (i as isize - peak_idx as isize).abs() > 3 {
                sum += cepstrum[i];
                count += 1;
            }
        }

        if count > 0 {
            sum / count as f32
        } else {
            0.0
        }
    }

    /// Calculate overall perceptual quality score
    fn calculate_overall_score(&self, metrics: &PerceptualQualityMetrics) -> f32 {
        let mut score = 0.0;
        let mut weight_sum = 0.0;

        // MCD: Lower is better (target: 4-6 dB is excellent, 6-8 is good)
        if let Some(mcd) = metrics.mcd {
            let mcd_score = (10.0 - mcd.clamp(3.0, 10.0)) / 7.0 * 100.0;
            score += mcd_score * 0.3;
            weight_sum += 0.3;
        }

        // HNR: Higher is better (target: >15 dB is good)
        if let Some(hnr) = metrics.hnr {
            let hnr_score = (hnr.clamp(0.0, 30.0) / 30.0) * 100.0;
            score += hnr_score * 0.25;
            weight_sum += 0.25;
        }

        // CPP: Higher is better (target: >10 dB is good)
        if let Some(cpp) = metrics.cpp {
            let cpp_score = (cpp.clamp(0.0, 20.0) / 20.0) * 100.0;
            score += cpp_score * 0.15;
            weight_sum += 0.15;
        }

        // Jitter: Lower is better (target: <1% is excellent)
        if let Some(jitter) = metrics.jitter_percent {
            let jitter_score = (5.0 - jitter.clamp(0.0, 5.0)) / 5.0 * 100.0;
            score += jitter_score * 0.1;
            weight_sum += 0.1;
        }

        // Shimmer: Lower is better (target: <3% is excellent)
        if let Some(shimmer) = metrics.shimmer_percent {
            let shimmer_score = (10.0 - shimmer.clamp(0.0, 10.0)) / 10.0 * 100.0;
            score += shimmer_score * 0.1;
            weight_sum += 0.1;
        }

        // LSD: Lower is better
        if let Some(lsd) = metrics.lsd {
            let lsd_score = (5.0 - lsd.clamp(0.0, 5.0)) / 5.0 * 100.0;
            score += lsd_score * 0.05;
            weight_sum += 0.05;
        }

        // BSD: Lower is better
        if let Some(bsd) = metrics.bsd {
            let bsd_score = (5.0 - bsd.clamp(0.0, 5.0)) / 5.0 * 100.0;
            score += bsd_score * 0.05;
            weight_sum += 0.05;
        }

        if weight_sum > 0.0 {
            score / weight_sum
        } else {
            50.0 // Default neutral score if no metrics available
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hz_to_mel_conversion() {
        let analyzer = PerceptualQualityAnalyzer::default();

        let mel_1000 = analyzer.hz_to_mel(1000.0);
        assert!(mel_1000 > 0.0 && mel_1000 < 3000.0);

        let hz_back = analyzer.mel_to_hz(mel_1000);
        assert!((hz_back - 1000.0).abs() < 1.0);
    }

    #[test]
    fn test_hz_to_bark_conversion() {
        let analyzer = PerceptualQualityAnalyzer::default();

        let bark_1000 = analyzer.hz_to_bark(1000.0);
        assert!(bark_1000 > 0.0 && bark_1000 < 25.0);
    }

    #[test]
    fn test_window_functions() {
        let analyzer = PerceptualQualityAnalyzer::default();
        let mut frame = vec![1.0; 512];

        analyzer.apply_window(&mut frame);

        // Window should be symmetric
        assert!((frame[0] - frame[frame.len() - 1]).abs() < 0.01);

        // Center should have maximum value
        assert!(frame[256] > frame[0]);
    }

    #[test]
    fn test_empty_audio_handling() {
        let analyzer = PerceptualQualityAnalyzer::default();
        let audio = AudioData::new(vec![], 22050, 1);

        let result = analyzer.analyze(&audio);
        assert!(result.is_ok());

        let metrics = result.unwrap();
        assert_eq!(metrics.overall_perceptual_score, 0.0);
    }

    #[test]
    fn test_autocorrelation() {
        let analyzer = PerceptualQualityAnalyzer::default();

        // Simple sine wave
        let signal: Vec<f32> = (0..100)
            .map(|i| (2.0 * PI * i as f32 / 10.0).sin())
            .collect();

        let autocorr = analyzer.autocorrelation(&signal);

        // Autocorrelation at lag 0 should be maximum
        assert!(autocorr[0] >= autocorr[1]);
    }

    #[test]
    fn test_perceptual_quality_config_default() {
        let config = PerceptualQualityConfig::default();

        assert_eq!(config.fft_size, 2048);
        assert_eq!(config.hop_size, 512);
        assert_eq!(config.num_mfcc, 13);
        assert!(config.enable_all);
    }

    #[test]
    fn test_mel_filterbank_creation() {
        let analyzer = PerceptualQualityAnalyzer::default();
        let filterbank = analyzer.create_mel_filterbank(40, 1025, 22050);

        assert_eq!(filterbank.len(), 40);
        assert_eq!(filterbank[0].len(), 1025);

        // Filters should sum to positive values
        for filter in filterbank.iter() {
            let sum: f32 = filter.iter().sum();
            assert!(sum > 0.0);
        }
    }

    #[test]
    fn test_overall_score_calculation() {
        let analyzer = PerceptualQualityAnalyzer::default();

        let metrics = PerceptualQualityMetrics {
            mcd: Some(5.0),
            hnr: Some(20.0),
            cpp: Some(12.0),
            jitter_percent: Some(0.5),
            shimmer_percent: Some(2.0),
            ..Default::default()
        };

        let score = analyzer.calculate_overall_score(&metrics);
        assert!(score > 0.0 && score <= 100.0);
    }
}

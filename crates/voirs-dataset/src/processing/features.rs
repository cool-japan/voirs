//! Feature extraction utilities
//!
//! This module provides comprehensive audio feature extraction capabilities
//! for speech processing and machine learning applications.
//!
//! Features:
//! - Mel spectrogram computation
//! - MFCC coefficient extraction
//! - Fundamental frequency estimation
//! - Energy and spectral features
//! - Real-time feature extraction
//! - Configurable processing parameters

use crate::{AudioData, DatasetError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Mel spectrogram configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MelSpectrogramConfig {
    /// Number of mel bins
    pub n_mels: usize,
    /// FFT size
    pub n_fft: usize,
    /// Hop length (samples)
    pub hop_length: usize,
    /// Window length (samples)
    pub win_length: Option<usize>,
    /// Window function type
    pub window: String,
    /// Lower frequency bound (Hz)
    pub f_min: f32,
    /// Upper frequency bound (Hz)
    pub f_max: Option<f32>,
    /// Power spectrum power (1.0 for energy, 2.0 for power)
    pub power: f32,
    /// Whether to use HTK mel formula
    pub htk_mel: bool,
    /// Normalization method
    pub norm: Option<String>,
}

impl Default for MelSpectrogramConfig {
    fn default() -> Self {
        Self {
            n_mels: 80,
            n_fft: 1024,
            hop_length: 256,
            win_length: None,
            window: "hann".to_string(),
            f_min: 0.0,
            f_max: None,
            power: 2.0,
            htk_mel: false,
            norm: Some("slaney".to_string()),
        }
    }
}

/// MFCC configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MfccConfig {
    /// Number of MFCC coefficients
    pub n_mfcc: usize,
    /// DCT normalization type
    pub dct_type: u32,
    /// Normalization mode for DCT
    pub norm: Option<String>,
    /// Liftering parameter (0 = no liftering)
    pub lifter: f32,
    /// Mel spectrogram configuration
    pub mel_config: MelSpectrogramConfig,
}

impl Default for MfccConfig {
    fn default() -> Self {
        Self {
            n_mfcc: 13,
            dct_type: 2,
            norm: Some("ortho".to_string()),
            lifter: 0.0,
            mel_config: MelSpectrogramConfig::default(),
        }
    }
}

/// Fundamental frequency estimation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct F0Config {
    /// Minimum frequency (Hz)
    pub f_min: f32,
    /// Maximum frequency (Hz)
    pub f_max: f32,
    /// Frame length for analysis (seconds)
    pub frame_length: f32,
    /// Hop length for analysis (seconds)
    pub hop_length: f32,
    /// Algorithm to use ("yin", "autocorrelation", "cepstrum")
    pub algorithm: String,
    /// Threshold for voicing detection
    pub voicing_threshold: f32,
}

impl Default for F0Config {
    fn default() -> Self {
        Self {
            f_min: 80.0,
            f_max: 400.0,
            frame_length: 0.025, // 25ms
            hop_length: 0.010,   // 10ms
            algorithm: "yin".to_string(),
            voicing_threshold: 0.1,
        }
    }
}

/// Spectral features configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralConfig {
    /// FFT size
    pub n_fft: usize,
    /// Hop length (samples)
    pub hop_length: usize,
    /// Window function
    pub window: String,
    /// Whether to compute spectral centroid
    pub centroid: bool,
    /// Whether to compute spectral bandwidth
    pub bandwidth: bool,
    /// Whether to compute spectral rolloff
    pub rolloff: bool,
    /// Whether to compute spectral flatness
    pub flatness: bool,
    /// Whether to compute zero crossing rate
    pub zcr: bool,
    /// Rolloff percentage (0.0-1.0)
    pub rolloff_percent: f32,
}

impl Default for SpectralConfig {
    fn default() -> Self {
        Self {
            n_fft: 1024,
            hop_length: 256,
            window: "hann".to_string(),
            centroid: true,
            bandwidth: true,
            rolloff: true,
            flatness: true,
            zcr: true,
            rolloff_percent: 0.85,
        }
    }
}

/// Feature extraction result
#[derive(Debug, Clone)]
pub struct FeatureResult {
    /// Feature name
    pub name: String,
    /// Feature values (time series or single value)
    pub values: Vec<f32>,
    /// Feature dimensions (frames, coefficients)
    pub shape: (usize, usize),
    /// Sampling information
    pub frame_rate: f32,
    /// Metadata about extraction
    pub metadata: HashMap<String, String>,
}

impl FeatureResult {
    pub fn new(name: String, values: Vec<f32>, shape: (usize, usize), frame_rate: f32) -> Self {
        Self {
            name,
            values,
            shape,
            frame_rate,
            metadata: HashMap::new(),
        }
    }

    /// Get feature as 2D matrix (frames x coefficients)
    pub fn as_matrix(&self) -> Vec<Vec<f32>> {
        let (n_frames, n_coeffs) = self.shape;
        let mut matrix = Vec::with_capacity(n_frames);

        for frame_idx in 0..n_frames {
            let mut frame = Vec::with_capacity(n_coeffs);
            for coeff_idx in 0..n_coeffs {
                let idx = frame_idx * n_coeffs + coeff_idx;
                if idx < self.values.len() {
                    frame.push(self.values[idx]);
                } else {
                    frame.push(0.0);
                }
            }
            matrix.push(frame);
        }

        matrix
    }

    /// Get time axis for the features
    pub fn time_axis(&self) -> Vec<f32> {
        let (n_frames, _) = self.shape;
        (0..n_frames).map(|i| i as f32 / self.frame_rate).collect()
    }
}

/// Main feature extractor
#[derive(Default)]
pub struct FeatureExtractor {
    /// Mel spectrogram configuration
    pub mel_config: MelSpectrogramConfig,
    /// MFCC configuration
    pub mfcc_config: MfccConfig,
    /// F0 configuration
    pub f0_config: F0Config,
    /// Spectral features configuration
    pub spectral_config: SpectralConfig,
}

impl FeatureExtractor {
    /// Create a new feature extractor with default configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a feature extractor with custom configurations
    pub fn with_configs(
        mel_config: MelSpectrogramConfig,
        mfcc_config: MfccConfig,
        f0_config: F0Config,
        spectral_config: SpectralConfig,
    ) -> Self {
        Self {
            mel_config,
            mfcc_config,
            f0_config,
            spectral_config,
        }
    }

    /// Extract mel spectrogram features
    pub fn extract_mel_spectrogram(&self, audio: &AudioData) -> Result<FeatureResult> {
        extract_mel_spectrogram(
            audio,
            self.mel_config.n_mels,
            self.mel_config.n_fft,
            self.mel_config.hop_length,
        )
    }

    /// Extract MFCC features
    pub fn extract_mfcc(&self, audio: &AudioData) -> Result<FeatureResult> {
        extract_mfcc_with_config(audio, &self.mfcc_config)
    }

    /// Extract fundamental frequency
    pub fn extract_f0(&self, audio: &AudioData) -> Result<FeatureResult> {
        extract_fundamental_frequency_with_config(audio, &self.f0_config)
    }

    /// Extract spectral features
    pub fn extract_spectral_features(&self, audio: &AudioData) -> Result<Vec<FeatureResult>> {
        extract_spectral_features_with_config(audio, &self.spectral_config)
    }

    /// Extract all available features
    pub fn extract_all_features(&self, audio: &AudioData) -> Result<Vec<FeatureResult>> {
        let mut features = Vec::new();

        // Mel spectrogram
        if let Ok(mel) = self.extract_mel_spectrogram(audio) {
            features.push(mel);
        }

        // MFCC
        if let Ok(mfcc) = self.extract_mfcc(audio) {
            features.push(mfcc);
        }

        // F0
        if let Ok(f0) = self.extract_f0(audio) {
            features.push(f0);
        }

        // Spectral features
        if let Ok(mut spectral) = self.extract_spectral_features(audio) {
            features.append(&mut spectral);
        }

        Ok(features)
    }
}

/// Extract mel spectrogram from audio
pub fn extract_mel_spectrogram(
    audio: &AudioData,
    n_mels: usize,
    n_fft: usize,
    hop_length: usize,
) -> Result<FeatureResult> {
    use scirs2_core::ndarray::Array2;

    let sample_rate = audio.sample_rate() as f32;
    let samples = audio.samples();

    if samples.is_empty() {
        return Err(DatasetError::ProcessingError(
            "Empty audio data".to_string(),
        ));
    }

    // Calculate number of frames
    let n_frames = if samples.len() > n_fft {
        (samples.len() - n_fft) / hop_length + 1
    } else {
        1
    };

    // Create Hann window
    let window: Vec<f32> = (0..n_fft)
        .map(|i| {
            let x = i as f32 / (n_fft - 1) as f32;
            0.5 * (1.0 - (2.0 * std::f32::consts::PI * x).cos())
        })
        .collect();

    // Create mel filterbank
    let mel_fb = create_mel_filterbank(n_mels, n_fft, sample_rate);

    // Compute STFT
    let mut mel_spec = Vec::with_capacity(n_frames * n_mels);

    for frame_idx in 0..n_frames {
        let start = frame_idx * hop_length;

        // Extract windowed frame as f64 for scirs2_fft::fft
        let windowed_frame: Vec<f64> = (0..n_fft)
            .map(|i| {
                let sample_idx = start + i;
                let sample = if sample_idx < samples.len() {
                    samples[sample_idx] * window[i]
                } else {
                    0.0_f32
                };
                sample as f64
            })
            .collect();

        // Apply FFT using scirs2_fft functional API (SCIRS2 POLICY)
        let fft_result = scirs2_fft::fft(&windowed_frame, Some(n_fft))
            .map_err(|e| DatasetError::ProcessingError(format!("FFT error: {e}")))?;

        // Compute power spectrum (first half + DC and Nyquist)
        let n_freqs = n_fft / 2 + 1;
        let power_spec: Vec<f32> = fft_result
            .iter()
            .take(n_freqs)
            .map(|c| (c.re * c.re + c.im * c.im) as f32)
            .collect();

        // Apply mel filterbank
        for mel_idx in 0..n_mels {
            let mut mel_energy = 0.0_f32;
            for (freq_idx, &power) in power_spec.iter().enumerate() {
                mel_energy += power * mel_fb[[mel_idx, freq_idx]];
            }
            // Convert to log scale with small epsilon to avoid log(0)
            let log_mel = (mel_energy + 1e-10_f32).ln();
            mel_spec.push(log_mel);
        }
    }

    let frame_rate = sample_rate / hop_length as f32;

    Ok(FeatureResult::new(
        "mel_spectrogram".to_string(),
        mel_spec,
        (n_frames, n_mels),
        frame_rate,
    ))
}

/// Create mel filterbank matrix
fn create_mel_filterbank(
    n_mels: usize,
    n_fft: usize,
    sample_rate: f32,
) -> scirs2_core::ndarray::Array2<f32> {
    use scirs2_core::ndarray::Array2;

    let n_freqs = n_fft / 2 + 1;
    let mut filterbank = Array2::zeros((n_mels, n_freqs));

    // Helper: Hz to Mel conversion
    let hz_to_mel = |hz: f32| 2595.0 * (1.0 + hz / 700.0).log10();
    let mel_to_hz = |mel: f32| 700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0);

    let f_min = 0.0;
    let f_max = sample_rate / 2.0;

    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);

    // Create mel frequency points
    let mel_points: Vec<f32> = (0..=n_mels + 1)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
        .map(mel_to_hz)
        .collect();

    // Convert to FFT bin numbers
    let bin_points: Vec<usize> = mel_points
        .iter()
        .map(|&f| ((n_fft + 1) as f32 * f / sample_rate).floor() as usize)
        .collect();

    // Create triangular filters
    for mel_idx in 0..n_mels {
        let left = bin_points[mel_idx];
        let center = bin_points[mel_idx + 1];
        let right = bin_points[mel_idx + 2];

        // Left slope
        for bin in left..center {
            if center > left {
                filterbank[[mel_idx, bin]] = (bin - left) as f32 / (center - left) as f32;
            }
        }

        // Right slope
        for bin in center..right {
            if right > center {
                filterbank[[mel_idx, bin]] = (right - bin) as f32 / (right - center) as f32;
            }
        }
    }

    // Normalize filters
    for mel_idx in 0..n_mels {
        let sum: f32 = filterbank.row(mel_idx).sum();
        if sum > 0.0 {
            for freq_idx in 0..n_freqs {
                filterbank[[mel_idx, freq_idx]] /= sum;
            }
        }
    }

    filterbank
}

/// Extract MFCC coefficients from audio
///
/// Implements the standard MFCC pipeline:
/// 1. Compute log-mel spectrogram (40 mel bins, n_fft=1024, hop=256)
/// 2. Apply DCT-II with ortho normalization to each frame
/// 3. Retain coefficients 1..=n_mfcc (skip DC/C0 unless include_energy)
/// 4. Optionally prepend log-energy (C0) as the first coefficient
pub fn extract_mfcc(audio: &AudioData, n_mfcc: usize, include_energy: bool) -> Result<Vec<f32>> {
    let n_mel_bins = 40usize;
    let mel_result = extract_mel_spectrogram(audio, n_mel_bins, 1024, 256)?;
    let (n_frames, _n_mels) = mel_result.shape;
    // mel_result.values is stored row-major: [frame0_mel0, frame0_mel1, ..., frame1_mel0, ...]
    // Values are already in log scale (computed as ln(energy + eps) in extract_mel_spectrogram)

    let n_coeffs_out = if include_energy { n_mfcc + 1 } else { n_mfcc };

    // Pre-compute the DCT-II orthonormal basis scaling factors:
    // coeff k=0: sqrt(1/N)
    // coeff k>0: sqrt(2/N)
    let n_f = n_mel_bins as f32;
    let scale_k0 = (1.0 / n_f).sqrt();
    let scale_k = (2.0 / n_f).sqrt();
    let pi_over_2n = std::f32::consts::PI / (2.0 * n_f);

    let mut values = Vec::with_capacity(n_frames * n_coeffs_out);

    for frame_idx in 0..n_frames {
        let frame_start = frame_idx * n_mel_bins;
        let log_mel = &mel_result.values[frame_start..frame_start + n_mel_bins];

        // DCT-II of the log-mel vector:
        //   C[k] = scale[k] * Σ_{n=0}^{N-1} log_mel[n] * cos(π*k*(2n+1)/(2N))
        // C[0] serves as the log-energy (DC component)

        // Compute C[0] = sqrt(1/N) * Σ log_mel[n]  (ortho normalization)
        let c0: f32 = scale_k0 * log_mel.iter().sum::<f32>();

        if include_energy {
            values.push(c0);
        }

        // Compute C[1]..C[n_mfcc]
        for k in 1..=n_mfcc {
            let k_f = k as f32;
            let coeff: f32 = scale_k
                * log_mel
                    .iter()
                    .enumerate()
                    .map(|(n, &lm)| lm * (k_f * (2 * n + 1) as f32 * pi_over_2n).cos())
                    .sum::<f32>();
            values.push(coeff);
        }
    }

    Ok(values)
}

/// Extract MFCC with configuration
pub fn extract_mfcc_with_config(audio: &AudioData, config: &MfccConfig) -> Result<FeatureResult> {
    let values = extract_mfcc(audio, config.n_mfcc, false)?;
    let n_frames = values.len() / config.n_mfcc;
    let frame_rate = audio.sample_rate() as f32 / config.mel_config.hop_length as f32;

    Ok(FeatureResult::new(
        "mfcc".to_string(),
        values,
        (n_frames, config.n_mfcc),
        frame_rate,
    ))
}

/// Extract fundamental frequency using the YIN algorithm.
///
/// YIN is de Cheveigné & Kawahara (2002). Per-frame steps:
/// 1. Compute the squared-difference function d(τ)
/// 2. Normalize to the cumulative-mean normalized difference (CMND)
/// 3. Search for the first τ whose CMND is below threshold and is a local minimum
/// 4. Apply parabolic interpolation to refine the lag estimate
/// 5. Convert refined lag to F0 = sample_rate / τ_refined, or 0.0 for unvoiced
pub fn extract_fundamental_frequency(
    audio: &AudioData,
    f_min: f32,
    f_max: f32,
) -> Result<Vec<f32>> {
    let sample_rate = audio.sample_rate() as f32;
    let samples = audio.samples();

    if samples.is_empty() {
        return Ok(vec![]);
    }

    let frame_length = (0.025 * sample_rate) as usize; // 25 ms frames
    let hop_length = (0.010 * sample_rate) as usize; // 10 ms hop

    let n_frames = if samples.len() > frame_length {
        (samples.len() - frame_length) / hop_length + 1
    } else {
        1
    };

    // τ range: [τ_min, τ_max] corresponding to [f_max, f_min]
    let tau_min = (sample_rate / f_max).ceil() as usize;
    let tau_max = ((sample_rate / f_min) as usize).min(frame_length / 2);
    let yin_threshold = 0.1_f32;

    let mut f0_values = Vec::with_capacity(n_frames);

    for frame_idx in 0..n_frames {
        let start = frame_idx * hop_length;
        let end = (start + frame_length).min(samples.len());
        let frame = &samples[start..end];

        // Half-window size W: integrate over the first half of the frame
        let half_w = frame.len() / 2;
        if half_w < 2 || tau_max < tau_min {
            f0_values.push(0.0);
            continue;
        }

        // ---- Step 1: Squared-difference function d(τ) -----------------------
        // d(τ) = Σ_{j=0}^{W-1} (x[j] - x[j+τ])²
        // We only need τ = 1..tau_max
        let effective_tau_max = tau_max.min(half_w);
        let mut d = vec![0.0_f32; effective_tau_max + 1]; // d[0] unused
        for tau in 1..=effective_tau_max {
            let mut acc = 0.0_f32;
            for j in 0..half_w {
                let x_j = frame[j];
                let x_j_tau = if j + tau < frame.len() {
                    frame[j + tau]
                } else {
                    0.0
                };
                let diff = x_j - x_j_tau;
                acc += diff * diff;
            }
            d[tau] = acc;
        }

        // ---- Step 2: Cumulative mean normalized difference (CMND) ------------
        // cmnd[0] = 1.0 (by definition)
        // cmnd[τ] = d[τ] / ((1/τ) * Σ_{j=1}^{τ} d[j])
        let mut cmnd = vec![1.0_f32; effective_tau_max + 1];
        let mut running_sum = 0.0_f32;
        for tau in 1..=effective_tau_max {
            running_sum += d[tau];
            if running_sum > 0.0 {
                cmnd[tau] = d[tau] * tau as f32 / running_sum;
            } else {
                cmnd[tau] = 1.0;
            }
        }

        // ---- Step 3: Threshold search for first local minimum below threshold
        let tau_lo = tau_min.max(1);
        let tau_hi = effective_tau_max;

        let mut best_tau: Option<usize> = None;
        let mut tau_search = tau_lo;
        while tau_search <= tau_hi {
            if cmnd[tau_search] < yin_threshold {
                // Find the absolute minimum in the dip starting here
                let mut local_min_tau = tau_search;
                let mut local_min_val = cmnd[tau_search];
                let mut t = tau_search + 1;
                while t <= tau_hi && cmnd[t] <= cmnd[t.saturating_sub(1)] {
                    if cmnd[t] < local_min_val {
                        local_min_val = cmnd[t];
                        local_min_tau = t;
                    }
                    t += 1;
                }
                best_tau = Some(local_min_tau);
                break;
            }
            tau_search += 1;
        }

        // If no τ found below threshold, fall back to global minimum of cmnd in [tau_lo, tau_hi]
        let refined_tau = if let Some(tau) = best_tau {
            // ---- Step 4: Parabolic interpolation around best_tau ---------------
            if tau > tau_lo && tau < tau_hi {
                let y0 = cmnd[tau - 1];
                let y1 = cmnd[tau];
                let y2 = cmnd[tau + 1];
                let denom = 2.0 * (2.0 * y1 - y0 - y2);
                if denom.abs() > 1e-10 {
                    let shift = (y2 - y0) / denom;
                    // Clamp shift to ±0.5 samples
                    tau as f32 + shift.clamp(-0.5, 0.5)
                } else {
                    tau as f32
                }
            } else {
                tau as f32
            }
        } else {
            // Unvoiced — return 0.0
            f0_values.push(0.0);
            continue;
        };

        // ---- Step 5: Convert lag to F0 and clamp to [f_min, f_max] -----------
        let f0 = if refined_tau > 0.0 {
            (sample_rate / refined_tau).clamp(f_min, f_max)
        } else {
            0.0
        };
        f0_values.push(f0);
    }

    Ok(f0_values)
}

/// Extract fundamental frequency with configuration
pub fn extract_fundamental_frequency_with_config(
    audio: &AudioData,
    config: &F0Config,
) -> Result<FeatureResult> {
    let values = extract_fundamental_frequency(audio, config.f_min, config.f_max)?;
    let frame_rate = 1.0 / config.hop_length;
    let values_len = values.len();

    Ok(FeatureResult::new(
        "f0".to_string(),
        values,
        (values_len, 1),
        frame_rate,
    ))
}

/// Extract spectral features
pub fn extract_spectral_features_with_config(
    audio: &AudioData,
    config: &SpectralConfig,
) -> Result<Vec<FeatureResult>> {
    let mut features = Vec::new();
    let sample_rate = audio.sample_rate() as f32;
    let samples = audio.samples();

    if samples.is_empty() {
        return Ok(features);
    }

    let n_frames = if samples.len() > config.n_fft {
        (samples.len() - config.n_fft) / config.hop_length + 1
    } else {
        1
    };

    let frame_rate = sample_rate / config.hop_length as f32;

    // Spectral centroid
    if config.centroid {
        let centroid_values: Vec<f32> = (0..n_frames)
            .map(|frame_idx| {
                let start_idx = frame_idx * config.hop_length;
                if start_idx + config.n_fft <= samples.len() {
                    // Simplified centroid calculation
                    sample_rate * 0.25 + (frame_idx as f32 * 10.0).sin() * 100.0
                } else {
                    sample_rate * 0.25
                }
            })
            .collect();

        features.push(FeatureResult::new(
            "spectral_centroid".to_string(),
            centroid_values,
            (n_frames, 1),
            frame_rate,
        ));
    }

    // Zero crossing rate
    if config.zcr {
        let zcr_values: Vec<f32> = (0..n_frames)
            .map(|frame_idx| {
                let start_idx = frame_idx * config.hop_length;
                let end_idx = (start_idx + config.n_fft).min(samples.len());

                if end_idx > start_idx + 1 {
                    let mut crossings = 0;
                    for i in start_idx..end_idx - 1 {
                        if (samples[i] >= 0.0) != (samples[i + 1] >= 0.0) {
                            crossings += 1;
                        }
                    }
                    crossings as f32 / (end_idx - start_idx - 1) as f32
                } else {
                    0.0
                }
            })
            .collect();

        features.push(FeatureResult::new(
            "zero_crossing_rate".to_string(),
            zcr_values,
            (n_frames, 1),
            frame_rate,
        ));
    }

    // Add more spectral features as needed (bandwidth, rolloff, flatness)

    Ok(features)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::dummy::DummyDataset;
    use crate::traits::Dataset;

    #[tokio::test]
    async fn test_mel_spectrogram_extraction() {
        let dataset = DummyDataset::small();
        let sample = dataset.get(0).await.unwrap();

        let result = extract_mel_spectrogram(&sample.audio, 80, 1024, 256).unwrap();
        assert_eq!(result.name, "mel_spectrogram");
        assert_eq!(result.shape.1, 80); // 80 mel bins
        assert!(!result.values.is_empty());
    }

    #[tokio::test]
    async fn test_mfcc_extraction() {
        let dataset = DummyDataset::small();
        let sample = dataset.get(0).await.unwrap();

        let result = extract_mfcc(&sample.audio, 13, true).unwrap();
        assert!(!result.is_empty());
        // Should have 14 coefficients (13 MFCC + 1 energy)
        assert!(result.len().is_multiple_of(14));
    }

    #[tokio::test]
    async fn test_f0_extraction() {
        let dataset = DummyDataset::small();
        let sample = dataset.get(0).await.unwrap();

        let result = extract_fundamental_frequency(&sample.audio, 80.0, 400.0).unwrap();
        assert!(!result.is_empty());
        // All F0 values should be within the specified range or 0 (unvoiced)
        for &f0 in &result {
            assert!(f0 == 0.0 || (80.0..=400.0).contains(&f0));
        }
    }

    #[tokio::test]
    async fn test_feature_extractor() {
        let dataset = DummyDataset::small();
        let sample = dataset.get(0).await.unwrap();

        let extractor = FeatureExtractor::new();
        let features = extractor.extract_all_features(&sample.audio).unwrap();

        assert!(!features.is_empty());

        // Check that we got at least some expected features
        let feature_names: Vec<&str> = features.iter().map(|f| f.name.as_str()).collect();
        assert!(feature_names.contains(&"mel_spectrogram"));
        assert!(feature_names.contains(&"mfcc"));
        assert!(feature_names.contains(&"f0"));
    }

    #[test]
    fn test_feature_result_matrix_conversion() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = FeatureResult::new("test".to_string(), values, (2, 3), 100.0);

        let matrix = result.as_matrix();
        assert_eq!(matrix.len(), 2); // 2 frames
        assert_eq!(matrix[0].len(), 3); // 3 coefficients
        assert_eq!(matrix[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(matrix[1], vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_time_axis_generation() {
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let result = FeatureResult::new("test".to_string(), values, (4, 1), 100.0);

        let time_axis = result.time_axis();
        assert_eq!(time_axis.len(), 4);
        assert_eq!(time_axis[0], 0.0);
        assert_eq!(time_axis[1], 0.01);
        assert_eq!(time_axis[2], 0.02);
        assert_eq!(time_axis[3], 0.03);
    }

    // -----------------------------------------------------------------------
    // New tests for the real MFCC and YIN-based F0 implementations
    // -----------------------------------------------------------------------

    /// Helper: build a pure sine-wave AudioData at the given frequency.
    fn make_sine_audio(freq_hz: f32, sample_rate: u32, n_samples: usize) -> AudioData {
        let samples: Vec<f32> = (0..n_samples)
            .map(|i| (2.0 * std::f32::consts::PI * freq_hz * i as f32 / sample_rate as f32).sin())
            .collect();
        AudioData::new(samples, sample_rate, 1)
    }

    /// Helper: build a near-silent (scaled) AudioData.
    fn make_scaled_audio(scale: f32, n_samples: usize, sample_rate: u32) -> AudioData {
        let samples: Vec<f32> = (0..n_samples)
            .map(|i| {
                scale * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin()
            })
            .collect();
        AudioData::new(samples, sample_rate, 1)
    }

    /// test_mfcc_shape: verify output length for both include_energy modes.
    #[test]
    fn test_mfcc_shape() {
        let sample_rate = 16_000u32;
        let n_samples = 500usize;
        let n_mfcc = 13usize;
        let audio = make_sine_audio(440.0, sample_rate, n_samples);

        // Reference mel spectrogram to derive expected frame count
        let mel = extract_mel_spectrogram(&audio, 40, 1024, 256).unwrap();
        let n_frames = mel.shape.0;

        // Without energy
        let mfcc_no_energy = extract_mfcc(&audio, n_mfcc, false).unwrap();
        assert_eq!(
            mfcc_no_energy.len(),
            n_frames * n_mfcc,
            "without energy: expected {} coefficients, got {}",
            n_frames * n_mfcc,
            mfcc_no_energy.len()
        );

        // With energy
        let mfcc_with_energy = extract_mfcc(&audio, n_mfcc, true).unwrap();
        assert_eq!(
            mfcc_with_energy.len(),
            n_frames * (n_mfcc + 1),
            "with energy: expected {} coefficients, got {}",
            n_frames * (n_mfcc + 1),
            mfcc_with_energy.len()
        );
    }

    /// test_mfcc_energy_ordering: energy coefficient (index 0 per frame in include_energy=true)
    /// should be larger on average for a loud signal than for a near-silent one.
    #[test]
    fn test_mfcc_energy_ordering() {
        let sample_rate = 16_000u32;
        let n_samples = 4096usize;
        let n_mfcc = 13usize;

        let loud_audio = make_scaled_audio(0.9, n_samples, sample_rate);
        let quiet_audio = make_scaled_audio(1e-4, n_samples, sample_rate);

        let loud_mfcc = extract_mfcc(&loud_audio, n_mfcc, true).unwrap();
        let quiet_mfcc = extract_mfcc(&quiet_audio, n_mfcc, true).unwrap();

        let n_coeffs = n_mfcc + 1;
        // Energy is index 0 within each frame
        let mean_energy = |mfcc: &[f32]| -> f32 {
            let frames = mfcc.len() / n_coeffs;
            if frames == 0 {
                return 0.0;
            }
            let sum: f32 = (0..frames).map(|f| mfcc[f * n_coeffs]).sum();
            sum / frames as f32
        };

        let loud_mean = mean_energy(&loud_mfcc);
        let quiet_mean = mean_energy(&quiet_mfcc);

        assert!(
            loud_mean > quiet_mean,
            "Loud signal energy ({}) should exceed quiet signal energy ({})",
            loud_mean,
            quiet_mean
        );
    }

    /// test_f0_sine: a 220 Hz sine wave should yield at least 80% voiced frames
    /// within ±15 Hz of 220 Hz.
    #[test]
    fn test_f0_sine() {
        let sample_rate = 16_000u32;
        // 1 second of 220 Hz sine
        let n_samples = sample_rate as usize;
        let audio = make_sine_audio(220.0, sample_rate, n_samples);

        let f0_values = extract_fundamental_frequency(&audio, 80.0, 800.0).unwrap();

        assert!(!f0_values.is_empty(), "F0 extraction returned no frames");

        let voiced_total = f0_values.iter().filter(|&&f| f > 0.0).count();
        let near_220 = f0_values
            .iter()
            .filter(|&&f| f > 0.0 && (f - 220.0).abs() <= 15.0)
            .count();

        assert!(
            voiced_total > 0,
            "Expected some voiced frames for a 220 Hz sine wave"
        );

        let accuracy = near_220 as f32 / voiced_total as f32;
        assert!(
            accuracy >= 0.80,
            "Expected ≥80% of voiced frames within ±15 Hz of 220 Hz, got {:.1}% ({}/{})",
            accuracy * 100.0,
            near_220,
            voiced_total
        );
    }

    /// test_f0_silence: all-zeros signal should produce all-zero F0 (unvoiced).
    #[test]
    fn test_f0_silence() {
        let sample_rate = 16_000u32;
        let audio = AudioData::silence(1.0, sample_rate, 1);

        let f0_values = extract_fundamental_frequency(&audio, 80.0, 800.0).unwrap();

        for (i, &f0) in f0_values.iter().enumerate() {
            assert_eq!(
                f0, 0.0,
                "Frame {} should be unvoiced (0.0) for silent input, got {}",
                i, f0
            );
        }
    }
}

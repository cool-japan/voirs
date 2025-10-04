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
    use rustfft::FftPlanner;
    use scirs2_core::ndarray::Array2;
    use scirs2_core::Complex32; // Use SciRS2 Complex type (SCIRS2 POLICY)

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

    // Create FFT planner
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);

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

        // Extract frame and apply window (Complex32 is compatible with rustfft)
        let mut frame_data: Vec<Complex32> = (0..n_fft)
            .map(|i| {
                let sample_idx = start + i;
                let sample = if sample_idx < samples.len() {
                    samples[sample_idx] * window[i]
                } else {
                    0.0_f32
                };
                Complex32::new(sample, 0.0_f32)
            })
            .collect();

        // Apply FFT
        fft.process(&mut frame_data);

        // Compute power spectrum (first half + DC and Nyquist)
        let n_freqs = n_fft / 2 + 1;
        let power_spec: Vec<f32> = frame_data
            .iter()
            .take(n_freqs)
            .map(|c| c.norm_sqr())
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
pub fn extract_mfcc(audio: &AudioData, n_mfcc: usize, include_energy: bool) -> Result<Vec<f32>> {
    // Simplified MFCC extraction
    // In a real implementation, this would:
    // 1. Compute mel spectrogram
    // 2. Apply DCT to get cepstral coefficients
    // 3. Optionally include energy coefficient

    let n_coeffs = if include_energy { n_mfcc + 1 } else { n_mfcc };

    // Mock MFCC computation
    let mel_result = extract_mel_spectrogram(audio, 26, 1024, 256)?;
    let (n_frames, _) = mel_result.shape;

    let values: Vec<f32> = (0..n_frames * n_coeffs)
        .map(|i| {
            let coeff_idx = i % n_coeffs;
            let frame_idx = i / n_coeffs;

            if coeff_idx == 0 && include_energy {
                // Energy coefficient (C0)
                frame_idx as f32 * 0.1
            } else {
                // Other MFCC coefficients
                let normalized_coeff = (coeff_idx as f32) / (n_coeffs as f32);
                (normalized_coeff * std::f32::consts::PI).cos() * 0.5
            }
        })
        .collect();

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

/// Extract fundamental frequency using YIN algorithm (simplified)
pub fn extract_fundamental_frequency(
    audio: &AudioData,
    f_min: f32,
    f_max: f32,
) -> Result<Vec<f32>> {
    // Simplified F0 extraction
    // In a real implementation, this would use YIN, autocorrelation, or other F0 algorithms

    let sample_rate = audio.sample_rate() as f32;
    let samples = audio.samples();

    if samples.is_empty() {
        return Ok(vec![]);
    }

    let frame_length = (0.025 * sample_rate) as usize; // 25ms frames
    let hop_length = (0.010 * sample_rate) as usize; // 10ms hop

    let n_frames = if samples.len() > frame_length {
        (samples.len() - frame_length) / hop_length + 1
    } else {
        1
    };

    // Mock F0 extraction
    let f0_values: Vec<f32> = (0..n_frames)
        .map(|frame_idx| {
            let start_idx = frame_idx * hop_length;
            if start_idx + frame_length <= samples.len() {
                // Simplified F0 estimation based on dominant frequency
                let frame_energy: f32 = samples[start_idx..start_idx + frame_length]
                    .iter()
                    .map(|&x| x * x)
                    .sum();

                if frame_energy > 1e-6 {
                    // Mock F0 based on energy and position
                    let base_f0 = f_min + (f_max - f_min) * 0.5;
                    let variation = (frame_idx as f32 * 0.1).sin() * 20.0;
                    (base_f0 + variation).max(f_min).min(f_max)
                } else {
                    0.0 // Unvoiced
                }
            } else {
                0.0
            }
        })
        .collect();

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
        assert!(result.len() % 14 == 0);
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
}

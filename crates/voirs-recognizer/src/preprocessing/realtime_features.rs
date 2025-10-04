//! Real-time Feature Extraction Module
//!
//! Extracts acoustic features from audio streams in real-time for improved
//! recognition accuracy and analysis.

use crate::RecognitionError;
use std::collections::HashMap;
use voirs_sdk::AudioBuffer;

/// Configuration for real-time feature extraction
#[derive(Debug, Clone)]
pub struct RealTimeFeatureConfig {
    /// Window size for feature extraction
    pub window_size: usize,
    /// Hop length between windows
    pub hop_length: usize,
    /// Number of mel filterbank channels
    pub n_mels: usize,
    /// Enable MFCC extraction
    pub extract_mfcc: bool,
    /// Enable spectral centroid
    pub extract_spectral_centroid: bool,
    /// Enable zero crossing rate
    pub extract_zcr: bool,
    /// Enable spectral rolloff
    pub extract_spectral_rolloff: bool,
    /// Enable energy features
    pub extract_energy: bool,
}

impl Default for RealTimeFeatureConfig {
    fn default() -> Self {
        Self {
            window_size: 512,
            hop_length: 256,
            n_mels: 13,
            extract_mfcc: true,
            extract_spectral_centroid: true,
            extract_zcr: true,
            extract_spectral_rolloff: true,
            extract_energy: true,
        }
    }
}

/// Feature types that can be extracted
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FeatureType {
    /// Mel-frequency cepstral coefficients
    MFCC,
    /// Spectral centroid
    SpectralCentroid,
    /// Zero crossing rate
    ZeroCrossingRate,
    /// Spectral rolloff
    SpectralRolloff,
    /// RMS energy
    Energy,
    /// Pitch/F0
    Pitch,
    /// Spectral bandwidth
    SpectralBandwidth,
}

/// Result of real-time feature extraction
#[derive(Debug, Clone)]
pub struct RealTimeFeatureResult {
    /// Extracted features by type
    pub features: HashMap<FeatureType, Vec<f32>>,
    /// Number of frames processed
    pub num_frames: usize,
    /// Processing time in milliseconds
    pub processing_time_ms: f32,
    /// Feature quality metrics
    pub quality_metrics: HashMap<String, f32>,
}

impl Default for RealTimeFeatureResult {
    fn default() -> Self {
        Self {
            features: HashMap::new(),
            num_frames: 0,
            processing_time_ms: 0.0,
            quality_metrics: HashMap::new(),
        }
    }
}

/// Real-time feature extractor
#[derive(Debug)]
pub struct RealTimeFeatureExtractor {
    config: RealTimeFeatureConfig,
    window: Vec<f32>,
    mel_filterbank: Vec<Vec<f32>>,
    dct_matrix: Vec<Vec<f32>>,
}

impl RealTimeFeatureExtractor {
    /// Create a new real-time feature extractor
    pub fn new(config: RealTimeFeatureConfig) -> Result<Self, RecognitionError> {
        let window = Self::create_hann_window(config.window_size);
        let mel_filterbank = Self::create_mel_filterbank(config.n_mels, config.window_size / 2 + 1);
        let dct_matrix = Self::create_dct_matrix(config.n_mels);

        Ok(Self {
            config,
            window,
            mel_filterbank,
            dct_matrix,
        })
    }

    /// Create Hann window
    fn create_hann_window(size: usize) -> Vec<f32> {
        (0..size)
            .map(|i| {
                0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (size - 1) as f32).cos())
            })
            .collect()
    }

    /// Create mel filterbank
    fn create_mel_filterbank(n_mels: usize, n_fft: usize) -> Vec<Vec<f32>> {
        // Simplified mel filterbank creation
        (0..n_mels)
            .map(|i| {
                (0..n_fft)
                    .map(|j| {
                        let mel_freq = 2595.0 * (1.0 + j as f32 / n_fft as f32).ln();
                        let filter_val = if i == 0 {
                            1.0 - (j as f32 / n_fft as f32)
                        } else {
                            (mel_freq / (i + 1) as f32).sin().abs()
                        };
                        filter_val
                    })
                    .collect()
            })
            .collect()
    }

    /// Create DCT matrix for MFCC
    fn create_dct_matrix(n_mels: usize) -> Vec<Vec<f32>> {
        (0..n_mels)
            .map(|i| {
                (0..n_mels)
                    .map(|j| {
                        ((2.0 * j as f32 + 1.0) * i as f32 * std::f32::consts::PI
                            / (2.0 * n_mels as f32))
                            .cos()
                    })
                    .collect()
            })
            .collect()
    }

    /// Extract features from audio buffer
    pub fn extract_features(
        &self,
        audio: &AudioBuffer,
    ) -> Result<RealTimeFeatureResult, RecognitionError> {
        let start_time = std::time::Instant::now();
        let mut result = RealTimeFeatureResult::default();

        let samples = audio.samples();
        let num_frames = (samples.len() - self.config.window_size) / self.config.hop_length + 1;
        result.num_frames = num_frames;

        for frame_idx in 0..num_frames {
            let start = frame_idx * self.config.hop_length;
            let end = (start + self.config.window_size).min(samples.len());
            let frame = &samples[start..end];

            if frame.len() == self.config.window_size {
                // Apply window
                let windowed: Vec<f32> = frame
                    .iter()
                    .zip(self.window.iter())
                    .map(|(s, w)| s * w)
                    .collect();

                // Extract requested features
                if self.config.extract_mfcc {
                    let mfcc = self.extract_mfcc(&windowed)?;
                    result
                        .features
                        .entry(FeatureType::MFCC)
                        .or_insert_with(Vec::new)
                        .extend(mfcc);
                }

                if self.config.extract_spectral_centroid {
                    let centroid = self.extract_spectral_centroid(&windowed)?;
                    result
                        .features
                        .entry(FeatureType::SpectralCentroid)
                        .or_insert_with(Vec::new)
                        .push(centroid);
                }

                if self.config.extract_zcr {
                    let zcr = self.extract_zero_crossing_rate(frame)?;
                    result
                        .features
                        .entry(FeatureType::ZeroCrossingRate)
                        .or_insert_with(Vec::new)
                        .push(zcr);
                }

                if self.config.extract_spectral_rolloff {
                    let rolloff = self.extract_spectral_rolloff(&windowed)?;
                    result
                        .features
                        .entry(FeatureType::SpectralRolloff)
                        .or_insert_with(Vec::new)
                        .push(rolloff);
                }

                if self.config.extract_energy {
                    let energy = self.extract_energy(frame)?;
                    result
                        .features
                        .entry(FeatureType::Energy)
                        .or_insert_with(Vec::new)
                        .push(energy);
                }
            }
        }

        result.processing_time_ms = start_time.elapsed().as_secs_f32() * 1000.0;

        // Calculate quality metrics
        result
            .quality_metrics
            .insert("snr_estimate".to_string(), self.estimate_snr(samples));
        result.quality_metrics.insert(
            "spectral_flatness".to_string(),
            self.calculate_spectral_flatness(samples),
        );

        Ok(result)
    }

    /// Extract MFCC features
    fn extract_mfcc(&self, windowed_frame: &[f32]) -> Result<Vec<f32>, RecognitionError> {
        // Simplified MFCC extraction
        let fft = self.simple_fft(windowed_frame);
        let power_spectrum: Vec<f32> = fft.iter().map(|c| c.norm_sqr()).collect();

        // Apply mel filterbank
        let mel_energies: Vec<f32> = self
            .mel_filterbank
            .iter()
            .map(|filter| {
                filter
                    .iter()
                    .zip(power_spectrum.iter())
                    .map(|(f, p)| f * p)
                    .sum::<f32>()
                    .max(1e-10)
                    .ln()
            })
            .collect();

        // Apply DCT
        let mfcc: Vec<f32> = self
            .dct_matrix
            .iter()
            .map(|dct_row| {
                dct_row
                    .iter()
                    .zip(mel_energies.iter())
                    .map(|(d, m)| d * m)
                    .sum()
            })
            .collect();

        Ok(mfcc)
    }

    /// Extract spectral centroid
    fn extract_spectral_centroid(&self, windowed_frame: &[f32]) -> Result<f32, RecognitionError> {
        let fft = self.simple_fft(windowed_frame);
        let power_spectrum: Vec<f32> = fft.iter().map(|c| c.norm_sqr()).collect();

        let numerator: f32 = power_spectrum
            .iter()
            .enumerate()
            .map(|(i, p)| i as f32 * p)
            .sum();

        let denominator: f32 = power_spectrum.iter().sum();

        Ok(if denominator > 0.0 {
            numerator / denominator
        } else {
            0.0
        })
    }

    /// Extract zero crossing rate
    fn extract_zero_crossing_rate(&self, frame: &[f32]) -> Result<f32, RecognitionError> {
        let crossings = frame
            .windows(2)
            .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
            .count();

        Ok(crossings as f32 / frame.len() as f32)
    }

    /// Extract spectral rolloff
    fn extract_spectral_rolloff(&self, windowed_frame: &[f32]) -> Result<f32, RecognitionError> {
        let fft = self.simple_fft(windowed_frame);
        let power_spectrum: Vec<f32> = fft.iter().map(|c| c.norm_sqr()).collect();

        let total_energy: f32 = power_spectrum.iter().sum();
        let threshold = 0.85 * total_energy;

        let mut cumsum = 0.0;
        for (i, power) in power_spectrum.iter().enumerate() {
            cumsum += power;
            if cumsum >= threshold {
                return Ok(i as f32 / power_spectrum.len() as f32);
            }
        }

        Ok(1.0)
    }

    /// Extract RMS energy
    fn extract_energy(&self, frame: &[f32]) -> Result<f32, RecognitionError> {
        let energy: f32 = frame.iter().map(|s| s * s).sum();
        Ok((energy / frame.len() as f32).sqrt())
    }

    /// Simple FFT implementation (for demonstration)
    fn simple_fft(&self, input: &[f32]) -> Vec<scirs2_core::Complex<f32>> {
        // Very simplified FFT - in practice, use a proper FFT library
        input
            .iter()
            .enumerate()
            .map(|(i, &sample)| {
                let angle = -2.0 * std::f32::consts::PI * i as f32 / input.len() as f32;
                scirs2_core::Complex::new(sample * angle.cos(), sample * angle.sin())
            })
            .collect()
    }

    /// Estimate signal-to-noise ratio
    fn estimate_snr(&self, samples: &[f32]) -> f32 {
        let signal_power: f32 = samples.iter().map(|s| s * s).sum();
        let mean_power = signal_power / samples.len() as f32;

        // Simple noise floor estimation
        let sorted_powers: Vec<f32> = samples.iter().map(|s| s * s).collect::<Vec<_>>();

        let noise_floor = sorted_powers.iter().take(samples.len() / 10).sum::<f32>()
            / (samples.len() / 10) as f32;

        if noise_floor > 0.0 {
            10.0 * (mean_power / noise_floor).log10()
        } else {
            60.0 // High SNR if no detectable noise
        }
    }

    /// Calculate spectral flatness
    fn calculate_spectral_flatness(&self, samples: &[f32]) -> f32 {
        let fft = self.simple_fft(samples);
        let power_spectrum: Vec<f32> = fft.iter().map(|c| c.norm_sqr()).collect();

        let geometric_mean = power_spectrum
            .iter()
            .map(|p| p.max(1e-10).ln())
            .sum::<f32>()
            / power_spectrum.len() as f32;

        let arithmetic_mean = power_spectrum.iter().sum::<f32>() / power_spectrum.len() as f32;

        if arithmetic_mean > 0.0 {
            geometric_mean.exp() / arithmetic_mean
        } else {
            0.0
        }
    }

    /// Get current configuration
    pub fn config(&self) -> &RealTimeFeatureConfig {
        &self.config
    }

    /// Update configuration
    pub fn set_config(&mut self, config: RealTimeFeatureConfig) -> Result<(), RecognitionError> {
        self.window = Self::create_hann_window(config.window_size);
        self.mel_filterbank =
            Self::create_mel_filterbank(config.n_mels, config.window_size / 2 + 1);
        self.dct_matrix = Self::create_dct_matrix(config.n_mels);
        self.config = config;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_realtime_feature_config_default() {
        let config = RealTimeFeatureConfig::default();
        assert_eq!(config.window_size, 512);
        assert_eq!(config.hop_length, 256);
        assert_eq!(config.n_mels, 13);
        assert!(config.extract_mfcc);
        assert!(config.extract_spectral_centroid);
    }

    #[test]
    fn test_feature_extractor_creation() {
        let config = RealTimeFeatureConfig::default();
        let extractor = RealTimeFeatureExtractor::new(config);
        assert!(extractor.is_ok());
    }

    #[test]
    fn test_feature_extraction() {
        let config = RealTimeFeatureConfig::default();
        let extractor = RealTimeFeatureExtractor::new(config).unwrap();

        let samples = vec![0.1; 1024]; // 1024 samples
        let audio = AudioBuffer::new(samples, 16000, 1);

        let result = extractor.extract_features(&audio);
        assert!(result.is_ok());

        let features = result.unwrap();
        assert!(features.features.contains_key(&FeatureType::MFCC));
        assert!(features
            .features
            .contains_key(&FeatureType::SpectralCentroid));
        assert!(features.num_frames > 0);
        assert!(features.processing_time_ms >= 0.0);
    }

    #[test]
    fn test_feature_types() {
        let types = vec![
            FeatureType::MFCC,
            FeatureType::SpectralCentroid,
            FeatureType::ZeroCrossingRate,
            FeatureType::SpectralRolloff,
            FeatureType::Energy,
            FeatureType::Pitch,
            FeatureType::SpectralBandwidth,
        ];

        for feature_type in types {
            // Test that feature types are properly comparable
            assert_eq!(feature_type.clone(), feature_type);
        }
    }

    #[test]
    fn test_feature_result_default() {
        let result = RealTimeFeatureResult::default();
        assert!(result.features.is_empty());
        assert_eq!(result.num_frames, 0);
        assert!((result.processing_time_ms - 0.0).abs() < f32::EPSILON);
        assert!(result.quality_metrics.is_empty());
    }
}

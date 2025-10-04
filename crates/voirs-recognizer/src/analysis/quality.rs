//! Audio quality analysis implementation
//!
//! This module provides comprehensive audio quality metrics including:
//! - Signal-to-Noise Ratio (SNR)
//! - Total Harmonic Distortion (THD)
//! - Spectral features
//! - Perceptual quality metrics

#![allow(clippy::cast_precision_loss)]

use crate::traits::AudioMetric;
use crate::RecognitionError;
use std::collections::HashMap;
use voirs_sdk::AudioBuffer;

/// Audio quality analyzer
pub struct QualityAnalyzer {
    /// Sample rate for analysis
    _sample_rate: f32,
    /// Frame size for spectral analysis
    _frame_size: usize,
    /// Hop size for overlapping frames
    _hop_size: usize,
    /// Window function for FFT
    window: Vec<f32>,
}

impl QualityAnalyzer {
    /// Create a new quality analyzer
    ///
    /// # Errors
    ///
    /// Returns an error if the quality analyzer cannot be initialized.
    pub async fn new() -> Result<Self, RecognitionError> {
        let frame_size = 1024;
        let hop_size = 512;
        let window = Self::hann_window(frame_size);

        Ok(Self {
            _sample_rate: 16000.0,
            _frame_size: frame_size,
            _hop_size: hop_size,
            window,
        })
    }

    /// Analyze audio quality metrics
    ///
    /// # Errors
    ///
    /// Returns an error if the quality analysis fails or if the audio buffer is invalid.
    pub async fn analyze_quality(
        &self,
        audio: &AudioBuffer,
        metrics: &[AudioMetric],
    ) -> Result<HashMap<String, f32>, RecognitionError> {
        let mut results = HashMap::new();

        for metric in metrics {
            let value = match metric {
                AudioMetric::SNR => self.calculate_snr(audio).await?,
                AudioMetric::THD => self.calculate_thd(audio).await?,
                AudioMetric::SpectralCentroid => self.calculate_spectral_centroid(audio).await?,
                AudioMetric::SpectralRolloff => self.calculate_spectral_rolloff(audio).await?,
                AudioMetric::ZeroCrossingRate => self.calculate_zero_crossing_rate(audio).await?,
                AudioMetric::MelFrequencyCepstralCoefficients => {
                    // For MFCC, we'll return the first coefficient as a representative value
                    let mfccs = self.calculate_mfcc(audio).await?;
                    mfccs.first().copied().unwrap_or(0.0)
                }
                AudioMetric::ChromaFeatures => {
                    // For chroma, return the energy in the first chroma bin
                    let chroma = self.calculate_chroma(audio).await?;
                    chroma.first().copied().unwrap_or(0.0)
                }
                AudioMetric::SpectralContrast => self.calculate_spectral_contrast(audio).await?,
                AudioMetric::TonnetzFeatures => {
                    // For tonnetz, return the first harmonic coordinate
                    let tonnetz = self.calculate_tonnetz(audio).await?;
                    tonnetz.first().copied().unwrap_or(0.0)
                }
                AudioMetric::RootMeanSquare => self.calculate_rms(audio).await?,
            };

            results.insert(format!("{metric:?}"), value);
        }

        Ok(results)
    }

    /// Calculate Signal-to-Noise Ratio
    async fn calculate_snr(&self, audio: &AudioBuffer) -> Result<f32, RecognitionError> {
        let samples = audio.samples();

        // Simple SNR calculation: assume the signal is the main component
        // and noise is the variation around the signal
        let signal_power: f32 = samples.iter().map(|x| x * x).sum::<f32>() / samples.len() as f32;

        // Estimate noise power using high-frequency content
        let noise_power = self.estimate_noise_power(samples).await?;

        if noise_power > 0.0 {
            let snr_linear = signal_power / noise_power;
            Ok(10.0 * snr_linear.log10())
        } else {
            Ok(60.0) // Very high SNR if no noise detected
        }
    }

    /// Estimate noise power from high-frequency content
    async fn estimate_noise_power(&self, samples: &[f32]) -> Result<f32, RecognitionError> {
        // Simple noise estimation using differences between adjacent samples
        if samples.len() < 2 {
            return Ok(0.0);
        }

        let differences: Vec<f32> = samples.windows(2).map(|w| w[1] - w[0]).collect();

        let noise_power = differences.iter().map(|x| x * x).sum::<f32>() / differences.len() as f32;
        Ok(noise_power)
    }

    /// Calculate Total Harmonic Distortion
    async fn calculate_thd(&self, audio: &AudioBuffer) -> Result<f32, RecognitionError> {
        let samples = audio.samples();

        // For THD calculation, we need to analyze harmonics in frequency domain
        let spectrum = self.compute_fft(samples).await?;

        // Find fundamental frequency (simplified approach)
        let fundamental_bin = Self::find_fundamental_frequency(&spectrum);

        if fundamental_bin == 0 {
            return Ok(0.0);
        }

        let fundamental_power = spectrum[fundamental_bin];

        // Calculate harmonic powers (2nd, 3rd, 4th, 5th harmonics)
        let mut harmonic_power = 0.0;
        for harmonic in 2..=5 {
            let harmonic_bin = fundamental_bin * harmonic;
            if harmonic_bin < spectrum.len() {
                harmonic_power += spectrum[harmonic_bin];
            }
        }

        if fundamental_power > 0.0 {
            let thd = (harmonic_power / fundamental_power).sqrt();
            Ok(thd * 100.0) // Convert to percentage
        } else {
            Ok(0.0)
        }
    }

    /// Calculate spectral centroid
    async fn calculate_spectral_centroid(
        &self,
        audio: &AudioBuffer,
    ) -> Result<f32, RecognitionError> {
        let samples = audio.samples();
        let spectrum = self.compute_fft(samples).await?;

        let mut weighted_sum = 0.0;
        let mut magnitude_sum = 0.0;

        for (i, &magnitude) in spectrum.iter().enumerate() {
            let frequency = i as f32 * audio.sample_rate() as f32 / spectrum.len() as f32;
            weighted_sum += frequency * magnitude;
            magnitude_sum += magnitude;
        }

        if magnitude_sum > 0.0 {
            Ok(weighted_sum / magnitude_sum)
        } else {
            Ok(0.0)
        }
    }

    /// Calculate spectral rolloff
    async fn calculate_spectral_rolloff(
        &self,
        audio: &AudioBuffer,
    ) -> Result<f32, RecognitionError> {
        let samples = audio.samples();
        let spectrum = self.compute_fft(samples).await?;

        let total_energy: f32 = spectrum.iter().sum();
        let rolloff_threshold = total_energy * 0.85; // 85% rolloff point

        let mut cumulative_energy = 0.0;
        for (i, &magnitude) in spectrum.iter().enumerate() {
            cumulative_energy += magnitude;
            if cumulative_energy >= rolloff_threshold {
                let frequency = i as f32 * audio.sample_rate() as f32 / spectrum.len() as f32;
                return Ok(frequency);
            }
        }

        Ok(audio.sample_rate() as f32 / 2.0) // Nyquist frequency as fallback
    }

    /// Calculate zero crossing rate
    async fn calculate_zero_crossing_rate(
        &self,
        audio: &AudioBuffer,
    ) -> Result<f32, RecognitionError> {
        let samples = audio.samples();

        if samples.len() < 2 {
            return Ok(0.0);
        }

        let mut zero_crossings = 0;
        for window in samples.windows(2) {
            if (window[0] >= 0.0 && window[1] < 0.0) || (window[0] < 0.0 && window[1] >= 0.0) {
                zero_crossings += 1;
            }
        }

        let zcr = zero_crossings as f32 / (samples.len() - 1) as f32;
        Ok(zcr)
    }

    /// Calculate MFCC (Mel-Frequency Cepstral Coefficients)
    async fn calculate_mfcc(&self, audio: &AudioBuffer) -> Result<Vec<f32>, RecognitionError> {
        let samples = audio.samples();
        let spectrum = self.compute_fft(samples).await?;

        // Convert to mel scale (simplified)
        let mel_spectrum = Self::linear_to_mel_spectrum(&spectrum, audio.sample_rate());

        // Apply DCT to get cepstral coefficients
        let mfcc = Self::dct(&mel_spectrum);

        // Return first 13 coefficients
        Ok(mfcc.into_iter().take(13).collect())
    }

    /// Calculate chroma features
    async fn calculate_chroma(&self, audio: &AudioBuffer) -> Result<Vec<f32>, RecognitionError> {
        let samples = audio.samples();
        let spectrum = self.compute_fft(samples).await?;

        // Map frequency bins to chroma classes (12 semitones)
        let mut chroma = vec![0.0; 12];

        for (i, &magnitude) in spectrum.iter().enumerate() {
            let frequency = i as f32 * audio.sample_rate() as f32 / spectrum.len() as f32;
            if frequency > 0.0 {
                let midi_note = 12.0 * (frequency / 440.0).log2() + 69.0;
                let chroma_value = midi_note % 12.0;
                if (0.0..12.0).contains(&chroma_value) {
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let chroma_class = chroma_value as usize;
                    chroma[chroma_class] += magnitude;
                }
            }
        }

        // Normalize
        let total: f32 = chroma.iter().sum();
        if total > 0.0 {
            for value in &mut chroma {
                *value /= total;
            }
        }

        Ok(chroma)
    }

    /// Calculate spectral contrast
    async fn calculate_spectral_contrast(
        &self,
        audio: &AudioBuffer,
    ) -> Result<f32, RecognitionError> {
        let samples = audio.samples();
        let spectrum = self.compute_fft(samples).await?;

        // Simple spectral contrast: ratio of high to low frequency energy
        let mid_point = spectrum.len() / 2;
        let low_freq_energy: f32 = spectrum[..mid_point].iter().sum();
        let high_freq_energy: f32 = spectrum[mid_point..].iter().sum();

        if low_freq_energy > 0.0 {
            Ok(high_freq_energy / low_freq_energy)
        } else {
            Ok(0.0)
        }
    }

    /// Calculate tonnetz features
    async fn calculate_tonnetz(&self, _audio: &AudioBuffer) -> Result<Vec<f32>, RecognitionError> {
        // Simplified tonnetz calculation
        // In a real implementation, this would compute harmonic network coordinates
        Ok(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) // 6D tonnetz space
    }

    /// Calculate RMS (Root Mean Square) energy
    async fn calculate_rms(&self, audio: &AudioBuffer) -> Result<f32, RecognitionError> {
        let samples = audio.samples();

        if samples.is_empty() {
            return Ok(0.0);
        }

        let mean_square: f32 = samples.iter().map(|x| x * x).sum::<f32>() / samples.len() as f32;
        Ok(mean_square.sqrt())
    }

    /// Compute FFT magnitude spectrum
    async fn compute_fft(&self, samples: &[f32]) -> Result<Vec<f32>, RecognitionError> {
        use scirs2_fft::{FftPlanner, RealFftPlanner};

        let mut padded_samples = samples.to_vec();

        // Pad to power of 2 if necessary
        let fft_size = padded_samples.len().next_power_of_two();
        padded_samples.resize(fft_size, 0.0);

        // Apply window
        if padded_samples.len() >= self.window.len() {
            for (i, sample) in padded_samples
                .iter_mut()
                .enumerate()
                .take(self.window.len())
            {
                *sample *= self.window[i];
            }
        }

        // Convert to f64 for FFT computation
        let padded_samples_f64: Vec<f64> = padded_samples.iter().map(|&x| x as f64).collect();

        // Perform the real FFT using scirs2_fft functional API
        let spectrum_complex = scirs2_fft::rfft(&padded_samples_f64, None).map_err(|e| {
            RecognitionError::AudioProcessingError {
                message: format!("FFT computation failed: {e}"),
                source: None,
            }
        })?;

        // Convert complex values to magnitude spectrum
        let spectrum: Vec<f32> = spectrum_complex
            .iter()
            .map(|c| {
                let magnitude = (c.re * c.re + c.im * c.im).sqrt();
                magnitude as f32
            })
            .collect();

        Ok(spectrum)
    }

    /// Find fundamental frequency bin in spectrum
    fn find_fundamental_frequency(spectrum: &[f32]) -> usize {
        // Find the peak in the spectrum (simplified)
        spectrum
            .iter()
            .enumerate()
            .skip(1) // Skip DC component
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i)
    }

    /// Convert linear spectrum to mel spectrum
    fn linear_to_mel_spectrum(spectrum: &[f32], sample_rate: u32) -> Vec<f32> {
        // Simplified mel filterbank
        let num_mel_bins = 26;
        let mut mel_spectrum = vec![0.0; num_mel_bins];

        // Linear to mel conversion: mel = 2595 * log10(1 + f/700)
        let mel_low = 0.0;
        let mel_high = 2595.0 * (1.0 + sample_rate as f32 / 2.0 / 700.0).log10();

        let mel_points: Vec<f32> = (0..=num_mel_bins + 1)
            .map(|i| mel_low + i as f32 * (mel_high - mel_low) / (num_mel_bins + 1) as f32)
            .collect();

        // Convert mel points back to Hz: f = 700 * (10^(mel/2595) - 1)
        let hz_points: Vec<f32> = mel_points
            .iter()
            .map(|mel| 700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0))
            .collect();

        // Convert Hz to FFT bin numbers
        let bin_points: Vec<usize> = hz_points
            .iter()
            .map(|hz| {
                let bin_f32 = hz * spectrum.len() as f32 / (sample_rate as f32 / 2.0);
                if bin_f32 >= 0.0 {
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let bin = bin_f32 as usize;
                    bin.min(spectrum.len() - 1)
                } else {
                    0
                }
            })
            .collect();

        // Apply triangular filters
        for m in 0..num_mel_bins {
            let left = bin_points[m];
            let center = bin_points[m + 1];
            let right = bin_points[m + 2];

            for k in left..=right {
                if k < spectrum.len() {
                    let weight = if k <= center {
                        if center > left {
                            (k - left) as f32 / (center - left) as f32
                        } else {
                            1.0
                        }
                    } else if right > center {
                        (right - k) as f32 / (right - center) as f32
                    } else {
                        0.0
                    };
                    mel_spectrum[m] += spectrum[k] * weight;
                }
            }
        }

        mel_spectrum
    }

    /// Discrete Cosine Transform
    fn dct(input: &[f32]) -> Vec<f32> {
        let n = input.len();
        let mut output = vec![0.0; n];

        #[allow(clippy::needless_range_loop)]
        for k in 0..n {
            let mut sum = 0.0;
            for n_idx in 0..n {
                sum += input[n_idx]
                    * (std::f32::consts::PI * k as f32 * (2 * n_idx + 1) as f32 / (2.0 * n as f32))
                        .cos();
            }

            let norm = if k == 0 {
                (1.0 / n as f32).sqrt()
            } else {
                (2.0 / n as f32).sqrt()
            };

            output[k] = norm * sum;
        }

        output
    }

    /// Generate Hann window
    fn hann_window(size: usize) -> Vec<f32> {
        (0..size)
            .map(|i| {
                0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (size - 1) as f32).cos())
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use voirs_sdk::AudioBuffer;

    #[tokio::test]
    async fn test_quality_analyzer_creation() {
        let analyzer = QualityAnalyzer::new().await.unwrap();
        assert_eq!(analyzer._frame_size, 1024);
        assert_eq!(analyzer._hop_size, 512);
        assert_eq!(analyzer.window.len(), 1024);
    }

    #[tokio::test]
    async fn test_snr_calculation() {
        let analyzer = QualityAnalyzer::new().await.unwrap();

        // Test with clean sine wave
        let samples: Vec<f32> = (0..1000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin())
            .collect();
        let audio = AudioBuffer::new(samples, 16000, 1);

        let snr = analyzer.calculate_snr(&audio).await.unwrap();
        assert!(snr > 0.0); // Should have positive SNR
    }

    #[tokio::test]
    async fn test_thd_calculation() {
        let analyzer = QualityAnalyzer::new().await.unwrap();

        // Test with sine wave
        let samples: Vec<f32> = (0..1000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin())
            .collect();
        let audio = AudioBuffer::new(samples, 16000, 1);

        let thd = analyzer.calculate_thd(&audio).await.unwrap();
        assert!(thd >= 0.0); // THD should be non-negative
    }

    #[tokio::test]
    async fn test_spectral_centroid() {
        let analyzer = QualityAnalyzer::new().await.unwrap();

        let samples = vec![0.1; 1000];
        let audio = AudioBuffer::new(samples, 16000, 1);

        let centroid = analyzer.calculate_spectral_centroid(&audio).await.unwrap();
        assert!(centroid >= 0.0);
        assert!(centroid <= 8000.0); // Should be within Nyquist frequency
    }

    #[tokio::test]
    async fn test_zero_crossing_rate() {
        let analyzer = QualityAnalyzer::new().await.unwrap();

        // Test with alternating signal
        let samples: Vec<f32> = (0..1000)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let audio = AudioBuffer::new(samples, 16000, 1);

        let zcr = analyzer.calculate_zero_crossing_rate(&audio).await.unwrap();
        assert!(zcr > 0.9); // Should be close to 1.0 for alternating signal
    }

    #[tokio::test]
    async fn test_rms_calculation() {
        let analyzer = QualityAnalyzer::new().await.unwrap();

        // Test with known RMS
        let samples = vec![1.0; 1000]; // RMS should be 1.0
        let audio = AudioBuffer::new(samples, 16000, 1);

        let rms = analyzer.calculate_rms(&audio).await.unwrap();
        assert!((rms - 1.0).abs() < 0.001); // Should be very close to 1.0
    }

    #[tokio::test]
    async fn test_quality_analysis() {
        let analyzer = QualityAnalyzer::new().await.unwrap();

        let samples = vec![0.1; 1000];
        let audio = AudioBuffer::new(samples, 16000, 1);

        let metrics = vec![
            AudioMetric::SNR,
            AudioMetric::RootMeanSquare,
            AudioMetric::ZeroCrossingRate,
            AudioMetric::SpectralCentroid,
        ];

        let results = analyzer.analyze_quality(&audio, &metrics).await.unwrap();

        assert_eq!(results.len(), 4);
        assert!(results.contains_key("SNR"));
        assert!(results.contains_key("RootMeanSquare"));
        assert!(results.contains_key("ZeroCrossingRate"));
        assert!(results.contains_key("SpectralCentroid"));

        // All values should be valid numbers
        for value in results.values() {
            assert!(value.is_finite());
        }
    }

    #[tokio::test]
    async fn test_mfcc_calculation() {
        let analyzer = QualityAnalyzer::new().await.unwrap();

        let samples = vec![0.1; 1000];
        let audio = AudioBuffer::new(samples, 16000, 1);

        let mfcc = analyzer.calculate_mfcc(&audio).await.unwrap();
        assert_eq!(mfcc.len(), 13); // Should return 13 MFCC coefficients

        // All coefficients should be finite
        for coeff in &mfcc {
            assert!(coeff.is_finite());
        }
    }

    #[tokio::test]
    async fn test_chroma_calculation() {
        let analyzer = QualityAnalyzer::new().await.unwrap();

        let samples = vec![0.1; 1000];
        let audio = AudioBuffer::new(samples, 16000, 1);

        let chroma = analyzer.calculate_chroma(&audio).await.unwrap();
        assert_eq!(chroma.len(), 12); // Should return 12 chroma features

        // All features should be finite and non-negative
        for feature in &chroma {
            assert!(feature.is_finite());
            assert!(*feature >= 0.0);
        }

        // Should sum to approximately 1.0 (normalized)
        let sum: f32 = chroma.iter().sum();
        if sum > 0.0 {
            assert!((sum - 1.0).abs() < 0.001);
        }
    }
}

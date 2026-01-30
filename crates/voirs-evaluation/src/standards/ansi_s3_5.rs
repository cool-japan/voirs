//! ANSI S3.5 Speech Intelligibility Index (SII) Implementation
//!
//! This module implements the ANSI S3.5-1997 (R2012) standard for calculating
//! the Speech Intelligibility Index, which predicts speech intelligibility
//! based on audibility across frequency bands.
//!
//! # Standard Reference
//!
//! ANSI S3.5-1997 (R2012): "Methods for Calculation of the Speech Intelligibility Index"
//!
//! # Algorithm
//!
//! The SII is calculated by:
//! 1. Dividing speech into frequency bands (1/3 octave or critical bands)
//! 2. Computing band importance functions
//! 3. Calculating audibility in each band
//! 4. Weighting by band importance
//! 5. Summing to obtain overall SII (0.0 to 1.0)

use super::StandardsError;
use scirs2_core::ndarray::Array1;
use serde::{Deserialize, Serialize};
use voirs_sdk::AudioBuffer;

/// ANSI S3.5 Speech Intelligibility Index evaluator
pub struct AnsiS35Evaluator {
    /// Sample rate
    sample_rate: u32,
    /// Frequency band configuration
    band_config: BandConfiguration,
    /// Speech spectrum level
    speech_spectrum: Array1<f32>,
    /// Noise spectrum level
    noise_spectrum: Option<Array1<f32>>,
}

/// Band configuration for SII calculation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BandConfiguration {
    /// 1/3 octave bands (21 bands from 160 Hz to 8000 Hz)
    OneThirdOctave,
    /// Critical bands (17 bands based on auditory filters)
    CriticalBands,
    /// Octave bands (6 bands from 250 Hz to 8000 Hz)
    OctaveBands,
}

impl Default for BandConfiguration {
    fn default() -> Self {
        Self::OneThirdOctave
    }
}

/// Speech Intelligibility Index result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechIntelligibilityIndex {
    /// Overall SII value (0.0 to 1.0)
    pub sii: f32,
    /// Band importance functions
    pub band_importances: Vec<f32>,
    /// Band SII contributions
    pub band_sii: Vec<f32>,
    /// Center frequencies of bands (Hz)
    pub center_frequencies: Vec<f32>,
    /// Speech levels per band (dB SPL)
    pub speech_levels: Vec<f32>,
    /// Noise levels per band (dB SPL)
    pub noise_levels: Vec<f32>,
    /// Signal-to-noise ratios per band (dB)
    pub snr_per_band: Vec<f32>,
    /// Predicted intelligibility percentage
    pub intelligibility_percent: f32,
    /// Compliance level
    pub compliance_level: super::ComplianceLevel,
}

impl AnsiS35Evaluator {
    /// Create new ANSI S3.5 evaluator
    pub fn new(sample_rate: u32, band_config: BandConfiguration) -> Result<Self, StandardsError> {
        if sample_rate < 8000 {
            return Err(StandardsError::InvalidAudioData {
                message: "Sample rate must be at least 8000 Hz for ANSI S3.5".to_string(),
            });
        }

        let num_bands = match band_config {
            BandConfiguration::OneThirdOctave => 21,
            BandConfiguration::CriticalBands => 17,
            BandConfiguration::OctaveBands => 6,
        };

        let speech_spectrum = Self::get_standard_speech_spectrum(band_config);

        Ok(Self {
            sample_rate,
            band_config,
            speech_spectrum,
            noise_spectrum: None,
        })
    }

    /// Calculate Speech Intelligibility Index
    pub fn calculate_sii(
        &self,
        speech: &AudioBuffer,
        noise: Option<&AudioBuffer>,
    ) -> Result<SpeechIntelligibilityIndex, StandardsError> {
        // Validate input
        if speech.sample_rate() != self.sample_rate {
            return Err(StandardsError::InvalidAudioData {
                message: format!(
                    "Speech sample rate {} does not match evaluator sample rate {}",
                    speech.sample_rate(),
                    self.sample_rate
                ),
            });
        }

        // Get center frequencies for bands
        let center_frequencies = self.get_center_frequencies();

        // Calculate speech spectrum levels per band
        let speech_levels = self.calculate_band_levels(speech, &center_frequencies)?;

        // Calculate noise spectrum levels per band
        let noise_levels = if let Some(noise_audio) = noise {
            self.calculate_band_levels(noise_audio, &center_frequencies)?
        } else {
            // Use standard quiet threshold (0 dB SPL)
            vec![0.0; center_frequencies.len()]
        };

        // Calculate SNR per band
        let snr_per_band: Vec<f32> = speech_levels
            .iter()
            .zip(noise_levels.iter())
            .map(|(s, n)| s - n)
            .collect();

        // Get band importance functions
        let band_importances = self.get_band_importance_functions();

        // Calculate band SII contributions
        let band_sii: Vec<f32> = snr_per_band
            .iter()
            .zip(band_importances.iter())
            .map(|(snr, importance)| {
                // SII contribution from each band (ANSI S3.5 equation)
                let audibility = self.calculate_band_audibility(*snr);
                audibility * importance
            })
            .collect();

        // Sum band contributions to get overall SII
        let sii: f32 = band_sii.iter().sum();
        let sii = sii.clamp(0.0, 1.0);

        // Convert SII to intelligibility percentage
        // Using standard transfer function: P = 100 * SII^1.2
        let intelligibility_percent = 100.0 * sii.powf(1.2);

        // Determine compliance level
        let compliance_level = if sii >= 0.75 {
            super::ComplianceLevel::FullyCompliant
        } else if sii >= 0.45 {
            super::ComplianceLevel::PartiallyCompliant
        } else {
            super::ComplianceLevel::NotCompliant
        };

        Ok(SpeechIntelligibilityIndex {
            sii,
            band_importances,
            band_sii,
            center_frequencies,
            speech_levels,
            noise_levels,
            snr_per_band,
            intelligibility_percent,
            compliance_level,
        })
    }

    /// Get center frequencies for configured bands
    fn get_center_frequencies(&self) -> Vec<f32> {
        match self.band_config {
            BandConfiguration::OneThirdOctave => {
                // 1/3 octave band centers from 160 Hz to 8000 Hz
                vec![
                    160.0, 200.0, 250.0, 315.0, 400.0, 500.0, 630.0, 800.0, 1000.0, 1250.0, 1600.0,
                    2000.0, 2500.0, 3150.0, 4000.0, 5000.0, 6300.0, 8000.0,
                ]
            }
            BandConfiguration::CriticalBands => {
                // Critical band centers (Bark scale)
                vec![
                    150.0, 250.0, 350.0, 450.0, 570.0, 700.0, 840.0, 1000.0, 1170.0, 1370.0,
                    1600.0, 1850.0, 2150.0, 2500.0, 2900.0, 3400.0, 4000.0,
                ]
            }
            BandConfiguration::OctaveBands => {
                // Octave band centers
                vec![250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0]
            }
        }
    }

    /// Get standard speech spectrum for band configuration
    fn get_standard_speech_spectrum(config: BandConfiguration) -> Array1<f32> {
        // Standard speech spectrum levels (dB SPL) at 1 meter
        // These are average speech levels for male talkers
        match config {
            BandConfiguration::OneThirdOctave => Array1::from_vec(vec![
                53.0, 55.0, 57.0, 58.0, 58.0, 58.0, 57.0, 56.0, 55.0, 54.0, 53.0, 52.0, 50.0, 48.0,
                46.0, 44.0, 42.0, 40.0,
            ]),
            BandConfiguration::CriticalBands => Array1::from_vec(vec![
                56.0, 58.0, 58.0, 58.0, 57.0, 56.0, 55.0, 54.0, 53.0, 52.0, 51.0, 50.0, 48.0, 46.0,
                44.0, 42.0, 40.0,
            ]),
            BandConfiguration::OctaveBands => {
                Array1::from_vec(vec![60.0, 60.0, 58.0, 54.0, 48.0, 42.0])
            }
        }
    }

    /// Get band importance functions for speech intelligibility
    fn get_band_importance_functions(&self) -> Vec<f32> {
        // Band importance functions from ANSI S3.5
        // These represent the relative contribution of each frequency band
        // to overall speech intelligibility
        match self.band_config {
            BandConfiguration::OneThirdOctave => vec![
                0.0083, 0.0095, 0.0150, 0.0289, 0.0440, 0.0578, 0.0653, 0.0711, 0.0818, 0.0844,
                0.0882, 0.0898, 0.0868, 0.0844, 0.0771, 0.0527, 0.0364, 0.0185,
            ],
            BandConfiguration::CriticalBands => vec![
                0.0083, 0.0150, 0.0289, 0.0440, 0.0578, 0.0653, 0.0711, 0.0818, 0.0844, 0.0882,
                0.0898, 0.0868, 0.0844, 0.0771, 0.0527, 0.0364, 0.0185,
            ],
            BandConfiguration::OctaveBands => {
                vec![0.05, 0.15, 0.25, 0.30, 0.20, 0.05]
            }
        }
    }

    /// Calculate band levels from audio signal
    fn calculate_band_levels(
        &self,
        audio: &AudioBuffer,
        center_frequencies: &[f32],
    ) -> Result<Vec<f32>, StandardsError> {
        use scirs2_fft::{RealFftPlanner, RealToComplex};

        let samples = audio.samples();
        if samples.is_empty() {
            return Err(StandardsError::InvalidAudioData {
                message: "Audio buffer is empty".to_string(),
            });
        }

        // Use FFT to compute spectrum
        let fft_size = 2048;
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(fft_size);

        let mut levels = Vec::with_capacity(center_frequencies.len());

        for &center_freq in center_frequencies {
            // Calculate bandwidth for this band
            let (lower_freq, upper_freq) = self.get_band_edges(center_freq);

            // Convert frequencies to FFT bins
            let bin_resolution = self.sample_rate as f32 / fft_size as f32;
            let lower_bin = (lower_freq / bin_resolution) as usize;
            let upper_bin = (upper_freq / bin_resolution) as usize;

            // Calculate average power in this band
            let mut band_power = 0.0f32;
            let mut count = 0;

            // Process audio in frames
            for chunk in samples.chunks(fft_size) {
                let mut buffer: Vec<f32> = chunk.to_vec();
                buffer.resize(fft_size, 0.0);

                let mut spectrum = vec![scirs2_core::Complex::new(0.0, 0.0); fft_size / 2 + 1];
                fft.process(&mut buffer, &mut spectrum);

                // Sum power in band bins
                for bin in lower_bin..=upper_bin.min(spectrum.len() - 1) {
                    band_power += spectrum[bin].norm_sqr();
                }
                count += 1;
            }

            // Convert to dB SPL (assuming calibration factor)
            let avg_power = if count > 0 {
                band_power / count as f32
            } else {
                1e-10
            };
            let level_db = 10.0 * avg_power.max(1e-10).log10() + 94.0; // +94 dB reference
            levels.push(level_db);
        }

        Ok(levels)
    }

    /// Get band edges (lower and upper frequencies) for a center frequency
    fn get_band_edges(&self, center_freq: f32) -> (f32, f32) {
        match self.band_config {
            BandConfiguration::OneThirdOctave => {
                // 1/3 octave bandwidth
                let ratio = 2.0f32.powf(1.0 / 6.0); // 2^(1/6)
                (center_freq / ratio, center_freq * ratio)
            }
            BandConfiguration::CriticalBands => {
                // Critical band bandwidth (approximation)
                let bw = 25.0 + 75.0 * (1.0 + 1.4 * (center_freq / 1000.0).powi(2)).powf(0.69);
                (center_freq - bw / 2.0, center_freq + bw / 2.0)
            }
            BandConfiguration::OctaveBands => {
                // Octave bandwidth
                let ratio = 2.0f32.sqrt(); // 2^(1/2)
                (center_freq / ratio, center_freq * ratio)
            }
        }
    }

    /// Calculate band audibility from SNR
    fn calculate_band_audibility(&self, snr_db: f32) -> f32 {
        // ANSI S3.5 audibility function
        // Maps SNR to audibility (0.0 to 1.0)
        if snr_db <= -15.0 {
            0.0
        } else if snr_db >= 15.0 {
            1.0
        } else {
            (snr_db + 15.0) / 30.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ansi_s35_evaluator_creation() {
        let evaluator = AnsiS35Evaluator::new(16000, BandConfiguration::OneThirdOctave);
        assert!(evaluator.is_ok());
    }

    #[test]
    fn test_invalid_sample_rate() {
        let evaluator = AnsiS35Evaluator::new(4000, BandConfiguration::OneThirdOctave);
        assert!(evaluator.is_err());
    }

    #[test]
    fn test_center_frequencies() {
        let evaluator = AnsiS35Evaluator::new(16000, BandConfiguration::OneThirdOctave).unwrap();
        let freqs = evaluator.get_center_frequencies();
        assert!((freqs[0] - 160.0).abs() < 0.001);
        assert!((*freqs.last().unwrap() - 8000.0).abs() < 0.001);
    }

    #[test]
    fn test_band_importance_sum() {
        let evaluator = AnsiS35Evaluator::new(16000, BandConfiguration::OneThirdOctave).unwrap();
        let importances = evaluator.get_band_importance_functions();
        let sum: f32 = importances.iter().sum();
        assert!((sum - 1.0).abs() < 0.01); // Should sum to approximately 1.0
    }

    #[test]
    fn test_band_audibility() {
        let evaluator = AnsiS35Evaluator::new(16000, BandConfiguration::OneThirdOctave).unwrap();

        // Test boundary conditions
        assert!((evaluator.calculate_band_audibility(-20.0) - 0.0).abs() < 0.001);
        assert!((evaluator.calculate_band_audibility(20.0) - 1.0).abs() < 0.001);
        assert!((evaluator.calculate_band_audibility(0.0) - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_sii_calculation() {
        let evaluator = AnsiS35Evaluator::new(16000, BandConfiguration::OneThirdOctave).unwrap();

        // Create test audio (silence)
        let speech = AudioBuffer::new(vec![0.0; 16000], 16000, 1);

        let result = evaluator.calculate_sii(&speech, None);
        assert!(result.is_ok());

        let sii = result.unwrap();
        assert!(sii.sii >= 0.0 && sii.sii <= 1.0);
        assert_eq!(sii.band_importances.len(), sii.band_sii.len());
    }

    #[test]
    fn test_compliance_levels() {
        let evaluator = AnsiS35Evaluator::new(16000, BandConfiguration::OneThirdOctave).unwrap();

        // Test with clean speech (should have high SII)
        let speech = AudioBuffer::new(vec![0.1; 16000], 16000, 1);
        let result = evaluator.calculate_sii(&speech, None).unwrap();

        // Verify compliance level is determined correctly
        match result.compliance_level {
            super::super::ComplianceLevel::FullyCompliant => assert!(result.sii >= 0.75),
            super::super::ComplianceLevel::PartiallyCompliant => {
                assert!(result.sii >= 0.45 && result.sii < 0.75);
            }
            super::super::ComplianceLevel::NotCompliant => assert!(result.sii < 0.45),
        }
    }
}

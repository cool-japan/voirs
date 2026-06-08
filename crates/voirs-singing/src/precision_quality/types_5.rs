//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::{types::Expression, Error, Result};

use super::types_4::ExpressionFeatures;

/// Onset detection using spectral flux
pub struct OnsetDetector {
    prev_spectrum: Vec<f32>,
}
impl OnsetDetector {
    /// Creates a new onset detector.
    ///
    /// Initializes the onset detector for spectral flux-based onset detection
    /// with empty previous spectrum state.
    ///
    /// # Returns
    ///
    /// A new `OnsetDetector` instance
    pub fn new() -> Self {
        Self {
            prev_spectrum: Vec::new(),
        }
    }
    /// Detects note onsets in audio using spectral flux analysis.
    ///
    /// Computes the spectral flux between consecutive audio frames and
    /// performs peak picking to identify onset times. Uses an adaptive
    /// threshold based on flux statistics.
    ///
    /// # Arguments
    ///
    /// * `audio` - Audio samples to analyze
    /// * `sample_rate` - Sample rate of the audio in Hz
    ///
    /// # Returns
    ///
    /// A vector of onset times in seconds
    ///
    /// # Errors
    ///
    /// Returns an error if FFT processing fails
    pub fn detect_onsets(&mut self, audio: &[f32], sample_rate: f32) -> Result<Vec<f32>> {
        let frame_size = 1024;
        let hop_size = 512;
        let mut onsets = Vec::new();
        let mut spectral_flux = Vec::new();
        for i in (0..audio.len()).step_by(hop_size) {
            let end = (i + frame_size).min(audio.len());
            if end - i < frame_size {
                break;
            }
            let frame = &audio[i..end];
            let spectrum = self.calculate_spectrum(frame)?;
            if !self.prev_spectrum.is_empty() {
                let flux = self.calculate_spectral_flux(&spectrum, &self.prev_spectrum);
                spectral_flux.push(flux);
            }
            self.prev_spectrum = spectrum;
        }
        let threshold = self.calculate_adaptive_threshold(&spectral_flux);
        for (i, &flux) in spectral_flux.iter().enumerate() {
            if flux > threshold && self.is_local_maximum(&spectral_flux, i) {
                let time = (i * hop_size) as f32 / sample_rate;
                onsets.push(time);
            }
        }
        Ok(onsets)
    }
    fn calculate_spectrum(&self, frame: &[f32]) -> Result<Vec<f32>> {
        let input_f64: Vec<scirs2_core::Complex<f64>> = frame
            .iter()
            .map(|&x| scirs2_core::Complex::new(x as f64, 0.0))
            .collect();
        let fft_result = scirs2_fft::fft(&input_f64, None)
            .map_err(|e| Error::Processing(format!("FFT error: {e}")))?;
        Ok(fft_result
            .iter()
            .take(frame.len() / 2)
            .map(|c| c.norm() as f32)
            .collect())
    }
    fn calculate_spectral_flux(&self, current: &[f32], previous: &[f32]) -> f32 {
        current
            .iter()
            .zip(previous.iter())
            .map(|(&curr, &prev)| (curr - prev).max(0.0))
            .sum()
    }
    fn calculate_adaptive_threshold(&self, flux: &[f32]) -> f32 {
        if flux.is_empty() {
            return 0.0;
        }
        let mean = flux.iter().sum::<f32>() / flux.len() as f32;
        let std_dev =
            (flux.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / flux.len() as f32).sqrt();
        mean + 2.0 * std_dev
    }
    fn is_local_maximum(&self, values: &[f32], index: usize) -> bool {
        if index == 0 || index >= values.len() - 1 {
            return false;
        }
        values[index] > values[index - 1] && values[index] > values[index + 1]
    }
}
impl Default for OnsetDetector {
    fn default() -> Self {
        Self::new()
    }
}
/// Expression model for classification
#[derive(Clone)]
pub struct ExpressionModel {
    /// Expression name
    pub name: String,
    /// Typical feature values for this expression
    pub typical_features: ExpressionFeatures,
    /// Tolerance ranges for feature matching
    pub tolerance: ExpressionFeatures,
}
impl ExpressionModel {
    /// Creates a neutral expression model.
    ///
    /// Defines typical feature values for neutral expression including
    /// moderate attack, sustain, and decay characteristics.
    ///
    /// # Returns
    ///
    /// An `ExpressionModel` configured for neutral expression
    pub fn new_neutral() -> Self {
        Self {
            name: String::from("Neutral"),
            typical_features: ExpressionFeatures {
                attack_time: 0.02,
                sustain_level: 0.75,
                decay_time: 0.08,
                spectral_centroid: 1100.0,
                dynamic_range: 0.4,
            },
            tolerance: ExpressionFeatures {
                attack_time: 0.01,
                sustain_level: 0.1,
                decay_time: 0.03,
                spectral_centroid: 200.0,
                dynamic_range: 0.15,
            },
        }
    }
    /// Creates a happy expression model.
    ///
    /// Defines typical feature values for happy expression including
    /// quick attack, high sustain level, and bright spectral centroid.
    ///
    /// # Returns
    ///
    /// An `ExpressionModel` configured for happy expression
    pub fn new_happy() -> Self {
        Self {
            name: String::from("Happy"),
            typical_features: ExpressionFeatures {
                attack_time: 0.008,
                sustain_level: 0.9,
                decay_time: 0.12,
                spectral_centroid: 1800.0,
                dynamic_range: 0.8,
            },
            tolerance: ExpressionFeatures {
                attack_time: 0.005,
                sustain_level: 0.1,
                decay_time: 0.04,
                spectral_centroid: 400.0,
                dynamic_range: 0.2,
            },
        }
    }
    /// Creates a sad expression model.
    ///
    /// Defines typical feature values for sad expression including
    /// slow attack, high sustain, long decay, and low spectral centroid.
    ///
    /// # Returns
    ///
    /// An `ExpressionModel` configured for sad expression
    pub fn new_sad() -> Self {
        Self {
            name: String::from("Sad"),
            typical_features: ExpressionFeatures {
                attack_time: 0.05,
                sustain_level: 0.85,
                decay_time: 0.15,
                spectral_centroid: 1000.0,
                dynamic_range: 0.3,
            },
            tolerance: ExpressionFeatures {
                attack_time: 0.02,
                sustain_level: 0.1,
                decay_time: 0.05,
                spectral_centroid: 200.0,
                dynamic_range: 0.1,
            },
        }
    }
    /// Creates an excited expression model.
    ///
    /// Defines typical feature values for excited expression including
    /// very fast attack, short decay, and moderate dynamic range.
    ///
    /// # Returns
    ///
    /// An `ExpressionModel` configured for excited expression
    pub fn new_excited() -> Self {
        Self {
            name: String::from("Excited"),
            typical_features: ExpressionFeatures {
                attack_time: 0.005,
                sustain_level: 0.4,
                decay_time: 0.03,
                spectral_centroid: 1500.0,
                dynamic_range: 0.6,
            },
            tolerance: ExpressionFeatures {
                attack_time: 0.003,
                sustain_level: 0.15,
                decay_time: 0.01,
                spectral_centroid: 300.0,
                dynamic_range: 0.2,
            },
        }
    }
    /// Creates a calm expression model.
    ///
    /// Defines typical feature values for calm expression including
    /// slow attack, very high sustain, long decay, and low dynamic range.
    ///
    /// # Returns
    ///
    /// An `ExpressionModel` configured for calm expression
    pub fn new_calm() -> Self {
        Self {
            name: String::from("Calm"),
            typical_features: ExpressionFeatures {
                attack_time: 0.05,
                sustain_level: 0.8,
                decay_time: 0.2,
                spectral_centroid: 900.0,
                dynamic_range: 0.25,
            },
            tolerance: ExpressionFeatures {
                attack_time: 0.02,
                sustain_level: 0.1,
                decay_time: 0.05,
                spectral_centroid: 150.0,
                dynamic_range: 0.1,
            },
        }
    }
    /// Calculates similarity between detected features and this expression model.
    ///
    /// Computes a normalized similarity score by comparing each detected feature
    /// against the typical values for this expression, using the tolerance ranges
    /// to determine acceptable deviations.
    ///
    /// # Arguments
    ///
    /// * `detected_features` - Expression features extracted from audio
    ///
    /// # Returns
    ///
    /// Similarity score from 0.0 (no match) to 1.0 (perfect match)
    pub fn calculate_similarity(&self, detected_features: &ExpressionFeatures) -> f32 {
        let attack_similarity = 1.0
            - ((detected_features.attack_time - self.typical_features.attack_time).abs()
                / self.tolerance.attack_time)
                .min(1.0);
        let sustain_similarity = 1.0
            - ((detected_features.sustain_level - self.typical_features.sustain_level).abs()
                / self.tolerance.sustain_level)
                .min(1.0);
        let decay_similarity = 1.0
            - ((detected_features.decay_time - self.typical_features.decay_time).abs()
                / self.tolerance.decay_time)
                .min(1.0);
        let spectral_similarity = 1.0
            - ((detected_features.spectral_centroid - self.typical_features.spectral_centroid)
                .abs()
                / self.tolerance.spectral_centroid)
                .min(1.0);
        let dynamic_similarity = 1.0
            - ((detected_features.dynamic_range - self.typical_features.dynamic_range).abs()
                / self.tolerance.dynamic_range)
                .min(1.0);
        (attack_similarity
            + sustain_similarity
            + decay_similarity
            + spectral_similarity
            + dynamic_similarity)
            / 5.0
    }
}
/// Timing accuracy analysis report
#[derive(Debug, Clone)]
pub struct TimingAccuracyReport {
    /// Percentage of notes within 10ms of target timing
    pub accuracy_percentage: f32,
    /// Number of notes within 10ms of target
    pub notes_within_10ms: usize,
    /// Total number of notes analyzed
    pub total_notes: usize,
    /// Mean timing deviation in milliseconds
    pub mean_timing_deviation_ms: f32,
    /// Maximum timing deviation in milliseconds
    pub max_timing_deviation_ms: f32,
    /// Rhythm consistency score (0.0-1.0)
    pub rhythm_consistency: f32,
    /// Individual timing deviations in milliseconds
    pub timing_deviations_ms: Vec<f32>,
}
impl Default for TimingAccuracyReport {
    fn default() -> Self {
        Self {
            accuracy_percentage: 60.0,
            notes_within_10ms: 12,
            total_notes: 20,
            mean_timing_deviation_ms: 0.0,
            max_timing_deviation_ms: 0.0,
            rhythm_consistency: 1.0,
            timing_deviations_ms: Vec::new(),
        }
    }
}

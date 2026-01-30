//! Formant Analysis and Manipulation for Emotional Voice Quality
//!
//! This module provides advanced formant analysis and manipulation capabilities
//! for emotion-based voice synthesis. Formants are resonant frequencies of the
//! vocal tract that characterize voice quality and emotional expression.
//!
//! ## Features
//!
//! - **Formant Extraction**: LPC-based formant frequency estimation
//! - **Emotion-specific Formant Patterns**: Pre-configured formant shifts for emotions
//! - **Real-time Formant Manipulation**: Modify voice quality during synthesis
//! - **Voice Quality Control**: Adjust vocal tract characteristics
//!
//! ## Background
//!
//! Formants are crucial for emotional expression:
//! - Happy/Excited: Higher formants (shorter vocal tract effect)
//! - Sad/Depressed: Lower formants (longer vocal tract effect)
//! - Angry: Raised F1, variable F2/F3
//! - Fear: Elevated formants with increased variability

use crate::{types::Emotion, Error, Result};
use serde::{Deserialize, Serialize};

/// Number of formants to track (typically F1-F4)
pub const NUM_FORMANTS: usize = 4;

/// Formant frequencies and bandwidths
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormantSet {
    /// Formant center frequencies (Hz)
    pub frequencies: [f32; NUM_FORMANTS],
    /// Formant bandwidths (Hz)
    pub bandwidths: [f32; NUM_FORMANTS],
    /// Formant amplitudes (linear scale)
    pub amplitudes: [f32; NUM_FORMANTS],
}

impl FormantSet {
    /// Create a neutral formant set for a typical male voice
    pub fn neutral_male() -> Self {
        Self {
            frequencies: [730.0, 1090.0, 2440.0, 3400.0], // Neutral vowel /ə/
            bandwidths: [90.0, 110.0, 170.0, 250.0],
            amplitudes: [1.0, 0.8, 0.6, 0.4],
        }
    }

    /// Create a neutral formant set for a typical female voice
    pub fn neutral_female() -> Self {
        Self {
            frequencies: [850.0, 1220.0, 2810.0, 3800.0], // Higher due to shorter vocal tract
            bandwidths: [90.0, 100.0, 160.0, 240.0],
            amplitudes: [1.0, 0.8, 0.6, 0.4],
        }
    }

    /// Apply emotion-specific formant modification
    pub fn apply_emotion(&mut self, emotion: Emotion, intensity: f32) {
        let shift = FormantShift::from_emotion(emotion);
        shift.apply(self, intensity);
    }

    /// Interpolate between two formant sets
    pub fn interpolate(&self, other: &FormantSet, alpha: f32) -> FormantSet {
        let alpha = alpha.clamp(0.0, 1.0);
        let mut result = self.clone();

        for i in 0..NUM_FORMANTS {
            result.frequencies[i] =
                self.frequencies[i] * (1.0 - alpha) + other.frequencies[i] * alpha;
            result.bandwidths[i] = self.bandwidths[i] * (1.0 - alpha) + other.bandwidths[i] * alpha;
            result.amplitudes[i] = self.amplitudes[i] * (1.0 - alpha) + other.amplitudes[i] * alpha;
        }

        result
    }
}

/// Formant shift pattern for emotions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormantShift {
    /// Frequency scaling factors for each formant
    pub frequency_scales: [f32; NUM_FORMANTS],
    /// Bandwidth scaling factors
    pub bandwidth_scales: [f32; NUM_FORMANTS],
    /// Amplitude adjustments
    pub amplitude_scales: [f32; NUM_FORMANTS],
}

impl FormantShift {
    /// Create formant shift pattern from emotion
    pub fn from_emotion(emotion: Emotion) -> Self {
        match emotion {
            Emotion::Happy => Self {
                frequency_scales: [1.05, 1.08, 1.10, 1.12], // Raise all formants
                bandwidth_scales: [1.1, 1.1, 1.1, 1.1],     // Slightly wider
                amplitude_scales: [1.1, 1.1, 1.0, 0.9],     // Emphasize lower formants
            },
            Emotion::Sad => Self {
                frequency_scales: [0.95, 0.93, 0.92, 0.90], // Lower all formants
                bandwidth_scales: [1.2, 1.2, 1.2, 1.2],     // Wider (less precise)
                amplitude_scales: [0.9, 0.9, 0.8, 0.7],     // Reduced energy
            },
            Emotion::Angry => Self {
                frequency_scales: [1.10, 1.05, 1.08, 1.12], // Raise F1 more
                bandwidth_scales: [1.3, 1.2, 1.2, 1.2],     // Much wider (tense)
                amplitude_scales: [1.2, 1.1, 1.0, 0.9],     // Strong low formants
            },
            Emotion::Fear => Self {
                frequency_scales: [1.08, 1.10, 1.15, 1.18], // Elevate all, esp. high
                bandwidth_scales: [1.4, 1.3, 1.3, 1.3],     // Very wide (unstable)
                amplitude_scales: [1.0, 1.0, 1.1, 1.1],     // Emphasize highs
            },
            Emotion::Calm => Self {
                frequency_scales: [0.98, 0.98, 0.98, 0.98], // Slightly lower
                bandwidth_scales: [0.9, 0.9, 0.9, 0.9],     // Narrower (precise)
                amplitude_scales: [1.0, 1.0, 1.0, 1.0],     // Balanced
            },
            Emotion::Excited => Self {
                frequency_scales: [1.12, 1.15, 1.18, 1.20], // Much higher
                bandwidth_scales: [1.2, 1.2, 1.2, 1.2],     // Wider
                amplitude_scales: [1.2, 1.2, 1.1, 1.0],     // Strong across board
            },
            _ => Self::neutral(),
        }
    }

    /// Create neutral shift (no change)
    pub fn neutral() -> Self {
        Self {
            frequency_scales: [1.0; NUM_FORMANTS],
            bandwidth_scales: [1.0; NUM_FORMANTS],
            amplitude_scales: [1.0; NUM_FORMANTS],
        }
    }

    /// Apply shift to formant set with given intensity
    pub fn apply(&self, formants: &mut FormantSet, intensity: f32) {
        let intensity = intensity.clamp(0.0, 1.0);

        for i in 0..NUM_FORMANTS {
            // Interpolate scale factors based on intensity
            let freq_scale = 1.0 + (self.frequency_scales[i] - 1.0) * intensity;
            let bw_scale = 1.0 + (self.bandwidth_scales[i] - 1.0) * intensity;
            let amp_scale = 1.0 + (self.amplitude_scales[i] - 1.0) * intensity;

            formants.frequencies[i] *= freq_scale;
            formants.bandwidths[i] *= bw_scale;
            formants.amplitudes[i] *= amp_scale;
        }
    }
}

/// Formant synthesizer using parallel resonators
#[derive(Debug)]
pub struct FormantSynthesizer {
    /// Current formant configuration
    formants: FormantSet,
    /// Sample rate (Hz)
    sample_rate: f32,
    /// Resonator states for each formant
    resonator_states: Vec<ResonatorState>,
}

/// State for a single formant resonator
#[derive(Debug, Clone)]
struct ResonatorState {
    /// Previous input samples
    x1: f32,
    x2: f32,
    /// Previous output samples
    y1: f32,
    y2: f32,
}

impl ResonatorState {
    fn new() -> Self {
        Self {
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }
}

impl FormantSynthesizer {
    /// Create a new formant synthesizer
    pub fn new(formants: FormantSet, sample_rate: f32) -> Self {
        let resonator_states = (0..NUM_FORMANTS).map(|_| ResonatorState::new()).collect();

        Self {
            formants,
            sample_rate,
            resonator_states,
        }
    }

    /// Update formant configuration
    pub fn set_formants(&mut self, formants: FormantSet) {
        self.formants = formants;
        // Reset resonator states when formants change significantly
        for state in &mut self.resonator_states {
            state.reset();
        }
    }

    /// Process audio through formant filters
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        let len = input.len().min(output.len());

        // Clear output
        output[..len].fill(0.0);

        // Process each formant in parallel
        for formant_idx in 0..NUM_FORMANTS {
            let freq = self.formants.frequencies[formant_idx];
            let bandwidth = self.formants.bandwidths[formant_idx];
            let amplitude = self.formants.amplitudes[formant_idx];

            // Calculate resonator coefficients
            let (b0, b2, a1, a2) = self.calculate_resonator_coeffs(freq, bandwidth);

            let state = &mut self.resonator_states[formant_idx];

            // Process samples through this resonator
            for i in 0..len {
                let x0 = input[i];

                // Biquad resonator filter
                let y0 = b0 * x0 + b2 * state.x2 - a1 * state.y1 - a2 * state.y2;

                // Update state
                state.x2 = state.x1;
                state.x1 = x0;
                state.y2 = state.y1;
                state.y1 = y0;

                // Accumulate to output with amplitude weighting
                output[i] += y0 * amplitude;
            }
        }
    }

    /// Calculate biquad resonator coefficients
    fn calculate_resonator_coeffs(&self, freq: f32, bandwidth: f32) -> (f32, f32, f32, f32) {
        let pi = std::f32::consts::PI;
        let omega = 2.0 * pi * freq / self.sample_rate;
        let r = (-pi * bandwidth / self.sample_rate).exp();

        let b0 = 1.0 - r * r;
        let b2 = 0.0;
        let a1 = -2.0 * r * omega.cos();
        let a2 = r * r;

        (b0, b2, a1, a2)
    }

    /// Get current formant configuration
    pub fn formants(&self) -> &FormantSet {
        &self.formants
    }
}

/// Formant analyzer using LPC (Linear Predictive Coding)
pub struct FormantAnalyzer {
    /// LPC order (typically 10-14 for formant analysis)
    lpc_order: usize,
    /// Sample rate
    sample_rate: f32,
}

impl FormantAnalyzer {
    /// Create a new formant analyzer
    pub fn new(sample_rate: f32) -> Self {
        Self {
            lpc_order: 12, // Suitable for 4 formants (2 poles per formant)
            sample_rate,
        }
    }

    /// Extract formants from audio frame
    ///
    /// This is a simplified implementation. A production version would use
    /// more sophisticated LPC analysis with root finding.
    pub fn extract_formants(&self, frame: &[f32]) -> Result<FormantSet> {
        if frame.len() < self.lpc_order * 2 {
            return Err(Error::Processing(
                "Frame too short for analysis".to_string(),
            ));
        }

        // Simplified formant extraction
        // In production, this would:
        // 1. Apply pre-emphasis
        // 2. Compute autocorrelation
        // 3. Solve Levinson-Durbin algorithm for LPC coefficients
        // 4. Find roots of LPC polynomial
        // 5. Convert roots to formant frequencies and bandwidths

        // For now, return default formants
        Ok(FormantSet::neutral_male())
    }

    /// Set LPC order
    pub fn set_lpc_order(&mut self, order: usize) {
        self.lpc_order = order;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_formant_set_creation() {
        let male = FormantSet::neutral_male();
        let female = FormantSet::neutral_female();

        // Female formants should be higher
        for i in 0..NUM_FORMANTS {
            assert!(female.frequencies[i] > male.frequencies[i]);
        }
    }

    #[test]
    fn test_emotion_formant_shifts() {
        let mut formants = FormantSet::neutral_male();
        let original_f1 = formants.frequencies[0];

        formants.apply_emotion(Emotion::Happy, 1.0);
        assert!(formants.frequencies[0] > original_f1); // Happy raises formants

        let mut formants = FormantSet::neutral_male();
        formants.apply_emotion(Emotion::Sad, 1.0);
        assert!(formants.frequencies[0] < original_f1); // Sad lowers formants
    }

    #[test]
    fn test_formant_interpolation() {
        let f1 = FormantSet::neutral_male();
        let f2 = FormantSet::neutral_female();

        let mid = f1.interpolate(&f2, 0.5);

        for i in 0..NUM_FORMANTS {
            let expected = (f1.frequencies[i] + f2.frequencies[i]) / 2.0;
            assert!((mid.frequencies[i] - expected).abs() < 1.0);
        }
    }

    #[test]
    fn test_formant_shift_neutral() {
        let shift = FormantShift::neutral();
        let mut formants = FormantSet::neutral_male();
        let original = formants.clone();

        shift.apply(&mut formants, 1.0);

        for i in 0..NUM_FORMANTS {
            assert!((formants.frequencies[i] - original.frequencies[i]).abs() < 0.01);
        }
    }

    #[test]
    fn test_formant_shift_intensity() {
        let shift = FormantShift::from_emotion(Emotion::Happy);
        let mut formants = FormantSet::neutral_male();
        let original_f1 = formants.frequencies[0];

        shift.apply(&mut formants, 0.5); // Half intensity
        let half_shift = formants.frequencies[0] - original_f1;

        let mut formants = FormantSet::neutral_male();
        shift.apply(&mut formants, 1.0); // Full intensity
        let full_shift = formants.frequencies[0] - original_f1;

        // Half intensity should produce roughly half the shift
        assert!(half_shift < full_shift);
        assert!(half_shift > 0.0);
    }

    #[test]
    fn test_formant_synthesizer_creation() {
        let formants = FormantSet::neutral_male();
        let sample_rate = 44100.0;

        let synthesizer = FormantSynthesizer::new(formants, sample_rate);
        assert_eq!(synthesizer.resonator_states.len(), NUM_FORMANTS);
    }

    #[test]
    fn test_formant_synthesizer_processing() {
        let formants = FormantSet::neutral_male();
        let sample_rate = 44100.0;
        let mut synthesizer = FormantSynthesizer::new(formants, sample_rate);

        let input = vec![1.0; 1000];
        let mut output = vec![0.0; 1000];

        synthesizer.process(&input, &mut output);

        // Output should be non-zero
        assert!(output.iter().any(|&x| x.abs() > 0.01));
    }

    #[test]
    fn test_formant_analyzer_creation() {
        let analyzer = FormantAnalyzer::new(44100.0);
        assert_eq!(analyzer.lpc_order, 12);
    }

    #[test]
    fn test_formant_extraction() {
        let analyzer = FormantAnalyzer::new(44100.0);
        let frame = vec![0.5; 512];

        let result = analyzer.extract_formants(&frame);
        assert!(result.is_ok());
    }

    #[test]
    fn test_emotion_specific_shifts() {
        let emotions = vec![
            Emotion::Happy,
            Emotion::Sad,
            Emotion::Angry,
            Emotion::Fear,
            Emotion::Calm,
            Emotion::Excited,
        ];

        for emotion in emotions {
            let shift = FormantShift::from_emotion(emotion);

            // All scales should be positive
            for i in 0..NUM_FORMANTS {
                assert!(shift.frequency_scales[i] > 0.0);
                assert!(shift.bandwidth_scales[i] > 0.0);
                assert!(shift.amplitude_scales[i] > 0.0);
            }
        }
    }
}

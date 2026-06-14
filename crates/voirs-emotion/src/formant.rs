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
use scirs2_core::Complex;
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
    /// LPC analysis order. Defaults to roughly `2 + sample_rate / 1000`
    /// (about one pole pair per kHz plus a few), clamped to `[8, 50]`.
    lpc_order: usize,
    /// Sample rate (Hz)
    sample_rate: f32,
}

/// Number of frequency points used to sample the LPC spectral envelope.
const ENVELOPE_POINTS: usize = 512;

/// Standard search bands (Hz) for the first `NUM_FORMANTS` formants. Bands are
/// deliberately broad (and overlapping) to accommodate both male and female
/// vocal tracts; the strictly increasing selection logic disambiguates them.
const FORMANT_BANDS: [(f32, f32); NUM_FORMANTS] = [
    (200.0, 1100.0),  // F1
    (800.0, 2600.0),  // F2
    (1700.0, 3500.0), // F3
    (2800.0, 4800.0), // F4
];

/// LPC analysis order for a sample rate: roughly `2 + sample_rate / 1000`
/// (one pole pair per kHz plus a few), clamped to `[8, 50]`.
fn lpc_order_for_rate(sample_rate: f32) -> usize {
    if sample_rate <= 0.0 {
        return 8;
    }
    (2 + (sample_rate as usize / 1000)).clamp(8, 50)
}

impl FormantAnalyzer {
    /// Create a new formant analyzer
    pub fn new(sample_rate: f32) -> Self {
        Self {
            lpc_order: lpc_order_for_rate(sample_rate),
            sample_rate,
        }
    }

    /// Extract formants from an audio frame using LPC spectral analysis.
    ///
    /// Pipeline (autocorrelation method):
    /// 1. Hann-window the frame and compute the autocorrelation for lags
    ///    `0..=order`.
    /// 2. Solve the Levinson-Durbin recursion for the LPC coefficients of
    ///    `A(z) = 1 + Σ aₖ z⁻ᵏ`.
    /// 3. Evaluate the all-pole spectral envelope `1 / |A(e^{jω})|` over a
    ///    linear grid `[0, Nyquist]`.
    /// 4. Peak-pick the strongest spectral-envelope local maximum inside each
    ///    formant search band (strictly increasing in frequency), derive the
    ///    −3 dB bandwidth, and convert the peak height to a relative amplitude.
    ///
    /// Silence, unvoiced, or otherwise degenerate frames gracefully fall back
    /// to [`FormantSet::neutral_male`].
    pub fn extract_formants(&self, frame: &[f32]) -> Result<FormantSet> {
        if frame.len() < self.lpc_order * 2 {
            return Err(Error::Processing(
                "Frame too short for analysis".to_string(),
            ));
        }

        let fallback = FormantSet::neutral_male();

        // Order is bounded by the configured value and the frame length so the
        // Levinson-Durbin recursion always has enough lags to work with.
        let order = self.lpc_order.min(frame.len() / 2).max(2);

        // Steps 1-2: window + autocorrelation + Levinson-Durbin.
        let coeffs = match lpc_coefficients(frame, order) {
            Some(coeffs) => coeffs,
            None => return Ok(fallback), // silence / DC / degenerate frame
        };

        // Step 3: all-pole spectral envelope in dB over [0, Nyquist].
        let (frequencies, envelope_db) =
            lpc_envelope_db(&coeffs, self.sample_rate, ENVELOPE_POINTS);
        if frequencies.len() < 3 {
            return Ok(fallback);
        }

        // Step 4: peak-pick formants in their expected bands.
        let peaks = local_maxima(&envelope_db);
        if peaks.is_empty() {
            return Ok(fallback);
        }

        let nyquist = self.sample_rate / 2.0;
        let mut result = fallback.clone();
        let mut found = [false; NUM_FORMANTS];
        let mut peak_db = [0.0_f32; NUM_FORMANTS];
        let mut last_freq = 0.0_f32;

        for (i, &(lo, hi)) in FORMANT_BANDS.iter().enumerate() {
            if lo >= nyquist {
                continue; // band entirely above Nyquist: keep neutral fallback
            }
            let hi = hi.min(nyquist);

            // Strongest local maximum in the band that is above the previously
            // selected formant frequency (enforces F1 < F2 < F3 < F4).
            let candidate = peaks
                .iter()
                .copied()
                .filter(|&p| {
                    let f = frequencies[p];
                    f > last_freq && f >= lo && f <= hi
                })
                .max_by(|&a, &b| {
                    envelope_db[a]
                        .partial_cmp(&envelope_db[b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

            if let Some(p) = candidate {
                let freq = frequencies[p];
                let bandwidth = peak_bandwidth_hz(&frequencies, &envelope_db, p);
                result.frequencies[i] = freq;
                result.bandwidths[i] = if bandwidth.is_finite() && bandwidth > 0.0 {
                    bandwidth
                } else {
                    fallback.bandwidths[i]
                };
                peak_db[i] = envelope_db[p];
                found[i] = true;
                last_freq = freq;
            }
        }

        if !found.iter().any(|&f| f) {
            return Ok(fallback);
        }

        // Convert peak heights (dB) to linear amplitudes relative to the
        // strongest detected formant, so the loudest formant is 1.0.
        let reference_db = (0..NUM_FORMANTS)
            .filter(|&i| found[i])
            .map(|i| peak_db[i])
            .fold(f32::NEG_INFINITY, f32::max);
        for i in 0..NUM_FORMANTS {
            if found[i] {
                let amplitude = 10.0_f32.powf((peak_db[i] - reference_db) / 20.0);
                result.amplitudes[i] = amplitude.clamp(0.0, 1.0);
            }
        }

        Ok(result)
    }

    /// Set LPC order
    pub fn set_lpc_order(&mut self, order: usize) {
        self.lpc_order = order;
    }
}

/// Compute LPC coefficients via Hann-windowed autocorrelation + Levinson-Durbin.
///
/// Returns the polynomial coefficients `a[1..=order]` (length `order`) for
/// `A(z) = 1 + Σ aₖ z⁻ᵏ`, or `None` when the frame is too short or degenerate
/// (e.g. silence, where the zero-lag autocorrelation vanishes).
fn lpc_coefficients(samples: &[f32], order: usize) -> Option<Vec<f64>> {
    let n = samples.len();
    if order == 0 || n <= order + 1 {
        return None;
    }

    // Hann window to reduce edge effects before autocorrelation.
    let denom = (n as f64 - 1.0).max(1.0);
    let windowed: Vec<f64> = samples
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            let window = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / denom).cos();
            s as f64 * window
        })
        .collect();

    let mut autocorr = vec![0.0_f64; order + 1];
    for (lag, slot) in autocorr.iter_mut().enumerate() {
        let mut sum = 0.0;
        for i in lag..n {
            sum += windowed[i] * windowed[i - lag];
        }
        *slot = sum;
    }
    if autocorr[0] <= 0.0 {
        return None;
    }
    // Tiny white-noise floor for numerical stability of the recursion.
    autocorr[0] *= 1.0 + 1e-9;

    levinson_durbin(&autocorr, order)
}

/// Levinson-Durbin recursion over the autocorrelation `r[0..=order]`. Returns
/// LPC coefficients `a[1..=order]` (length `order`) or `None`.
fn levinson_durbin(r: &[f64], order: usize) -> Option<Vec<f64>> {
    if r.len() <= order || r[0] <= 0.0 {
        return None;
    }
    let mut a = vec![0.0_f64; order + 1];
    a[0] = 1.0;
    let mut error = r[0];

    for i in 1..=order {
        let mut acc = r[i];
        for j in 1..i {
            acc += a[j] * r[i - j];
        }
        let reflection = -acc / error;
        if !reflection.is_finite() {
            return None;
        }
        let previous = a.clone();
        for j in 1..i {
            a[j] = previous[j] + reflection * previous[i - j];
        }
        a[i] = reflection;
        error *= 1.0 - reflection * reflection;
        if error <= 0.0 {
            error = 1e-9;
        }
    }

    Some(a[1..=order].to_vec())
}

/// Evaluate the LPC all-pole spectral envelope (dB, ignoring the constant gain)
/// over a linear frequency grid `[0, Nyquist]`. Returns `(frequencies, dB)`.
fn lpc_envelope_db(coeffs: &[f64], sample_rate: f32, n_points: usize) -> (Vec<f32>, Vec<f32>) {
    let mut frequencies = Vec::with_capacity(n_points);
    let mut magnitudes_db = Vec::with_capacity(n_points);
    if n_points < 2 || sample_rate <= 0.0 {
        return (frequencies, magnitudes_db);
    }
    let nyquist = sample_rate as f64 / 2.0;
    let step = (n_points - 1) as f64;
    for m in 0..n_points {
        let ratio = m as f64 / step; // 0..1
        let omega = std::f64::consts::PI * ratio;
        // A(e^{jω}) = 1 + Σ aₖ e^{-jωk}.
        let mut a_eval = Complex::new(1.0_f64, 0.0_f64);
        for (k0, &ak) in coeffs.iter().enumerate() {
            let k = (k0 + 1) as f64;
            let phase = -omega * k;
            a_eval += Complex::new(ak, 0.0) * Complex::new(phase.cos(), phase.sin());
        }
        let mag_a = a_eval.norm().max(1e-12);
        // Envelope magnitude is 1/|A|, i.e. -20·log10|A| in dB.
        frequencies.push((nyquist * ratio) as f32);
        magnitudes_db.push((-20.0 * mag_a.log10()) as f32);
    }
    (frequencies, magnitudes_db)
}

/// Indices of local maxima in `values` (strict left, non-strict right to be
/// robust to plateaus).
fn local_maxima(values: &[f32]) -> Vec<usize> {
    let mut peaks = Vec::new();
    if values.len() < 3 {
        return peaks;
    }
    for i in 1..values.len() - 1 {
        if values[i] > values[i - 1] && values[i] >= values[i + 1] {
            peaks.push(i);
        }
    }
    peaks
}

/// Estimate the −3 dB bandwidth (Hz) of the spectral-envelope peak at `peak_idx`
/// with linear interpolation of the crossing frequencies.
fn peak_bandwidth_hz(frequencies: &[f32], magnitudes_db: &[f32], peak_idx: usize) -> f32 {
    let threshold = magnitudes_db[peak_idx] - 3.0;

    // Walk left to the first bin at or below the threshold.
    let mut left = peak_idx;
    while left > 0 && magnitudes_db[left] > threshold {
        left -= 1;
    }
    let left_freq = if magnitudes_db[left] <= threshold && left < peak_idx {
        interpolate_crossing(
            frequencies[left],
            magnitudes_db[left],
            frequencies[left + 1],
            magnitudes_db[left + 1],
            threshold,
        )
    } else {
        frequencies[left]
    };

    // Walk right to the first bin at or below the threshold.
    let mut right = peak_idx;
    while right + 1 < magnitudes_db.len() && magnitudes_db[right] > threshold {
        right += 1;
    }
    let right_freq = if magnitudes_db[right] <= threshold && right > peak_idx {
        interpolate_crossing(
            frequencies[right - 1],
            magnitudes_db[right - 1],
            frequencies[right],
            magnitudes_db[right],
            threshold,
        )
    } else {
        frequencies[right]
    };

    (right_freq - left_freq).max(0.0)
}

/// Linear interpolation of the frequency at which the line through `(f0, m0)`
/// and `(f1, m1)` reaches `target`.
fn interpolate_crossing(f0: f32, m0: f32, f1: f32, m1: f32, target: f32) -> f32 {
    let span = m1 - m0;
    if span.abs() < f32::EPSILON {
        return f0;
    }
    f0 + (target - m0) / span * (f1 - f0)
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
        // Order is derived from the sample rate: ~2 + sample_rate/1000, clamped.
        let analyzer = FormantAnalyzer::new(44100.0);
        assert_eq!(analyzer.lpc_order, lpc_order_for_rate(44100.0));
        assert_eq!(analyzer.lpc_order, 46);
    }

    #[test]
    fn test_lpc_order_derivation() {
        assert_eq!(lpc_order_for_rate(16000.0), 18); // 2 + 16
        assert_eq!(lpc_order_for_rate(8000.0), 10); // 2 + 8
        assert_eq!(lpc_order_for_rate(4000.0), 8); // 2 + 4 -> clamped up to 8
        assert_eq!(lpc_order_for_rate(96000.0), 50); // 2 + 96 -> clamped down to 50
        assert_eq!(lpc_order_for_rate(0.0), 8); // degenerate -> floor
    }

    #[test]
    fn test_formant_extraction() {
        let analyzer = FormantAnalyzer::new(44100.0);
        let frame = vec![0.5; 512];

        let result = analyzer.extract_formants(&frame);
        assert!(result.is_ok());
    }

    /// Single all-pole (two-pole) resonator: `y[n] = x[n] + a1·y[n-1] + a2·y[n-2]`
    /// with a pole pair at `freq` and −3 dB bandwidth `bw`.
    fn all_pole_resonator(input: &[f32], freq: f32, bw: f32, sample_rate: f32) -> Vec<f32> {
        let omega = 2.0 * std::f32::consts::PI * freq / sample_rate;
        let r = (-std::f32::consts::PI * bw / sample_rate).exp();
        let a1 = 2.0 * r * omega.cos();
        let a2 = -r * r;
        let mut output = vec![0.0_f32; input.len()];
        let mut y1 = 0.0_f32;
        let mut y2 = 0.0_f32;
        for (n, &x) in input.iter().enumerate() {
            let y0 = x + a1 * y1 + a2 * y2;
            output[n] = y0;
            y2 = y1;
            y1 = y0;
        }
        output
    }

    /// Synthesize a voiced, speech-like frame by exciting a cascade of all-pole
    /// resonators (the formants) with an impulse train at fundamental `f0`.
    fn synth_all_pole(
        freqs: &[f32],
        bws: &[f32],
        f0: f32,
        sample_rate: f32,
        num_samples: usize,
    ) -> Vec<f32> {
        // Glottal impulse train.
        let mut signal = vec![0.0_f32; num_samples];
        let period = (sample_rate / f0).round().max(1.0) as usize;
        for i in (0..num_samples).step_by(period) {
            signal[i] = 1.0;
        }
        // Cascade the formant resonators.
        for (&f, &bw) in freqs.iter().zip(bws.iter()) {
            signal = all_pole_resonator(&signal, f, bw, sample_rate);
        }
        // Normalize to unit peak.
        let peak = signal.iter().fold(0.0_f32, |m, &v| m.max(v.abs()));
        if peak > 0.0 {
            for v in &mut signal {
                *v /= peak;
            }
        }
        signal
    }

    #[test]
    fn test_extract_formants_recovers_resonances() {
        let sample_rate = 16000.0;
        let analyzer = FormantAnalyzer::new(sample_rate);

        // Known resonances, deliberately different from neutral_male()
        // (730/1090/2440) so a silent fallback could not pass this test.
        let freqs = [500.0_f32, 1500.0, 2700.0];
        let bws = [60.0_f32, 90.0, 120.0];
        let frame = synth_all_pole(&freqs, &bws, 120.0, sample_rate, 2048);

        let formants = analyzer
            .extract_formants(&frame)
            .expect("extraction should succeed for a voiced frame");

        // F1/F2/F3 recovered within LPC estimation tolerance.
        assert!(
            (formants.frequencies[0] - 500.0).abs() < 120.0,
            "F1 = {} (expected ~500)",
            formants.frequencies[0]
        );
        assert!(
            (formants.frequencies[1] - 1500.0).abs() < 150.0,
            "F2 = {} (expected ~1500)",
            formants.frequencies[1]
        );
        assert!(
            (formants.frequencies[2] - 2700.0).abs() < 200.0,
            "F3 = {} (expected ~2700)",
            formants.frequencies[2]
        );

        // Strictly increasing formant frequencies.
        assert!(formants.frequencies[0] < formants.frequencies[1]);
        assert!(formants.frequencies[1] < formants.frequencies[2]);

        // Real, sane bandwidths and normalized amplitudes for F1..F3.
        for i in 0..3 {
            assert!(
                formants.bandwidths[i].is_finite() && formants.bandwidths[i] > 0.0,
                "bandwidth[{i}] = {}",
                formants.bandwidths[i]
            );
            assert!(
                (0.0..=1.0).contains(&formants.amplitudes[i]),
                "amplitude[{i}] = {}",
                formants.amplitudes[i]
            );
        }

        // Confirm this is a real estimate, not the neutral_male() fallback.
        assert!(
            (formants.frequencies[1] - 1090.0).abs() > 100.0,
            "F2 should not match the neutral fallback"
        );
    }

    #[test]
    fn test_extract_formants_silence_returns_fallback() {
        let analyzer = FormantAnalyzer::new(16000.0);
        let silence = vec![0.0_f32; 1024];

        let formants = analyzer
            .extract_formants(&silence)
            .expect("silence must not error");

        // Graceful fallback to the neutral male formant set.
        let neutral = FormantSet::neutral_male();
        for i in 0..NUM_FORMANTS {
            assert!((formants.frequencies[i] - neutral.frequencies[i]).abs() < 1e-3);
        }
    }

    #[test]
    fn test_extract_formants_near_zero_does_not_panic() {
        let analyzer = FormantAnalyzer::new(16000.0);
        // Tiny, sub-audible values: numerically near-degenerate but non-zero.
        let frame: Vec<f32> = (0..1024)
            .map(|i| if i % 2 == 0 { 1e-9 } else { -1e-9 })
            .collect();

        // Must return a valid result without panicking; either a real estimate
        // or the neutral fallback is acceptable here.
        let formants = analyzer
            .extract_formants(&frame)
            .expect("near-zero frame must not error");
        for i in 0..NUM_FORMANTS {
            assert!(formants.frequencies[i].is_finite());
            assert!(formants.bandwidths[i].is_finite());
            assert!(formants.amplitudes[i].is_finite());
        }
    }

    #[test]
    fn test_extract_formants_frame_too_short_errors() {
        let analyzer = FormantAnalyzer::new(16000.0); // order 18 -> needs >= 36 samples
        let frame = vec![0.1_f32; 10];
        assert!(analyzer.extract_formants(&frame).is_err());
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

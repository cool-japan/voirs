//! Pitch processing and generation for singing synthesis

#![allow(dead_code, clippy::needless_range_loop)]

use crate::score::MusicalNote;
use crate::types::{Expression, VoiceCharacteristics};
use ndarray::{Array1, Array2};
use rustfft::{num_complex::Complex, FftPlanner};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Pitch contour representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PitchContour {
    /// Time points in seconds
    pub time_points: Vec<f32>,
    /// F0 values in Hz
    pub f0_values: Vec<f32>,
    /// Confidence values (0.0-1.0)
    pub confidence: Vec<f32>,
    /// Voicing decisions
    pub voicing: Vec<bool>,
    /// Pitch smoothness
    pub smoothness: f32,
    /// Interpolation method
    pub interpolation: InterpolationMethod,
}

/// Pitch generator for creating pitch contours
pub struct PitchGenerator {
    /// Voice characteristics
    voice_characteristics: VoiceCharacteristics,
    /// Pitch model parameters
    model_params: PitchModelParams,
    /// Vibrato parameters
    vibrato_params: VibratoParams,
    /// Portamento parameters
    portamento_params: PortamentoParams,
    /// Expression mappings
    expression_mappings: HashMap<Expression, ExpressionPitchParams>,
}

/// Pitch processor for real-time pitch modification
pub struct PitchProcessor {
    /// Sample rate
    sample_rate: f32,
    /// Frame size
    frame_size: usize,
    /// Hop size
    hop_size: usize,
    /// FFT planner
    fft_planner: FftPlanner<f32>,
    /// Window function
    window: Array1<f32>,
    /// Pitch shift factor
    pitch_shift: f32,
    /// Formant shift factor
    formant_shift: f32,
    /// Processing buffer
    buffer: Array2<f32>,
    /// Phase vocoder state
    phase_vocoder: PhaseVocoder,
}

/// Pitch model parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PitchModelParams {
    /// Base F0 in Hz
    pub base_f0: f32,
    /// F0 range (min, max) in Hz
    pub f0_range: (f32, f32),
    /// Pitch stability (0.0-1.0)
    pub stability: f32,
    /// Pitch variation (0.0-1.0)
    pub variation: f32,
    /// Intonation accuracy (0.0-1.0)
    pub intonation_accuracy: f32,
    /// Microtonal deviation (0.0-1.0)
    pub microtonal_deviation: f32,
    /// Glide time between notes (seconds)
    pub glide_time: f32,
    /// Overshoot amount (0.0-1.0)
    pub overshoot: f32,
}

/// Vibrato parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VibratoParams {
    /// Vibrato frequency in Hz
    pub frequency: f32,
    /// Vibrato depth in cents
    pub depth: f32,
    /// Vibrato onset time (seconds)
    pub onset_time: f32,
    /// Vibrato rate variation (0.0-1.0)
    pub rate_variation: f32,
    /// Vibrato depth variation (0.0-1.0)
    pub depth_variation: f32,
    /// Vibrato waveform
    pub waveform: VibratoWaveform,
    /// Vibrato phase
    pub phase: f32,
}

/// Portamento parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortamentoParams {
    /// Portamento time (seconds)
    pub time: f32,
    /// Portamento curve
    pub curve: PortamentoCurve,
    /// Portamento threshold (semitones)
    pub threshold: f32,
    /// Portamento strength (0.0-1.0)
    pub strength: f32,
    /// Enable automatic portamento
    pub auto_portamento: bool,
}

/// Expression pitch parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpressionPitchParams {
    /// Pitch modifier (1.0 = no change)
    pub pitch_modifier: f32,
    /// Vibrato modifier (1.0 = no change)
    pub vibrato_modifier: f32,
    /// Stability modifier (1.0 = no change)
    pub stability_modifier: f32,
    /// Variation modifier (1.0 = no change)
    pub variation_modifier: f32,
    /// Glide modifier (1.0 = no change)
    pub glide_modifier: f32,
}

/// Interpolation methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InterpolationMethod {
    /// Linear interpolation
    Linear,
    /// Cubic spline interpolation
    Cubic,
    /// Hermite interpolation
    Hermite,
    /// Bezier interpolation
    Bezier,
}

/// Vibrato waveform types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VibratoWaveform {
    /// Sine wave
    Sine,
    /// Triangle wave
    Triangle,
    /// Sawtooth wave
    Sawtooth,
    /// Square wave
    Square,
    /// Random
    Random,
}

/// Portamento curve types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PortamentoCurve {
    /// Linear curve
    Linear,
    /// Exponential curve
    Exponential,
    /// Logarithmic curve
    Logarithmic,
    /// Sigmoid curve
    Sigmoid,
    /// Bezier curve
    Bezier,
}

/// Phase vocoder for pitch shifting
struct PhaseVocoder {
    /// Analysis window
    analysis_window: Array1<f32>,
    /// Synthesis window
    synthesis_window: Array1<f32>,
    /// Previous phase
    previous_phase: Array1<f32>,
    /// Output phase
    output_phase: Array1<f32>,
    /// Overlap buffer
    overlap_buffer: Array1<f32>,
}

impl PitchContour {
    /// Create new pitch contour
    pub fn new(time_points: Vec<f32>, f0_values: Vec<f32>) -> Self {
        let len = time_points.len();
        Self {
            time_points,
            f0_values,
            confidence: vec![1.0; len],
            voicing: vec![true; len],
            smoothness: 0.5,
            interpolation: InterpolationMethod::Cubic,
        }
    }

    /// Get F0 value at specific time
    pub fn f0_at_time(&self, time: f32) -> f32 {
        if self.time_points.is_empty() {
            return 0.0;
        }

        // Find surrounding time points
        let mut left_idx = 0;
        let mut right_idx = self.time_points.len() - 1;

        for (i, &t) in self.time_points.iter().enumerate() {
            if t <= time {
                left_idx = i;
            } else {
                right_idx = i;
                break;
            }
        }

        if left_idx == right_idx {
            return self.f0_values[left_idx];
        }

        // Interpolate between points
        let t1 = self.time_points[left_idx];
        let t2 = self.time_points[right_idx];
        let f1 = self.f0_values[left_idx];
        let f2 = self.f0_values[right_idx];

        match self.interpolation {
            InterpolationMethod::Linear => {
                let alpha = (time - t1) / (t2 - t1);
                f1 * (1.0 - alpha) + f2 * alpha
            }
            InterpolationMethod::Cubic => self.cubic_interpolation(time, left_idx, right_idx),
            InterpolationMethod::Hermite => self.hermite_interpolation(time, left_idx, right_idx),
            InterpolationMethod::Bezier => self.bezier_interpolation(time, left_idx, right_idx),
        }
    }

    /// Cubic interpolation
    fn cubic_interpolation(&self, time: f32, left_idx: usize, right_idx: usize) -> f32 {
        let t1 = self.time_points[left_idx];
        let t2 = self.time_points[right_idx];
        let f1 = self.f0_values[left_idx];
        let f2 = self.f0_values[right_idx];

        let alpha = (time - t1) / (t2 - t1);
        let alpha2 = alpha * alpha;
        let alpha3 = alpha2 * alpha;

        // Simple cubic interpolation
        f1 * (2.0 * alpha3 - 3.0 * alpha2 + 1.0) + f2 * (-2.0 * alpha3 + 3.0 * alpha2)
    }

    /// Hermite interpolation
    fn hermite_interpolation(&self, time: f32, left_idx: usize, right_idx: usize) -> f32 {
        // Simplified Hermite interpolation
        self.cubic_interpolation(time, left_idx, right_idx)
    }

    /// Bezier interpolation
    fn bezier_interpolation(&self, time: f32, left_idx: usize, right_idx: usize) -> f32 {
        // Simplified Bezier interpolation
        self.cubic_interpolation(time, left_idx, right_idx)
    }

    /// Smooth pitch contour
    pub fn smooth(&mut self, factor: f32) {
        if self.f0_values.len() < 3 {
            return;
        }

        let smoothed: Vec<f32> = self
            .f0_values
            .windows(3)
            .enumerate()
            .map(|(i, window)| {
                let original = self.f0_values[i + 1];
                let smoothed = (window[0] + window[1] + window[2]) / 3.0;
                original * (1.0 - factor) + smoothed * factor
            })
            .collect();

        for (i, &value) in smoothed.iter().enumerate() {
            self.f0_values[i + 1] = value;
        }
    }

    /// Apply vibrato to pitch contour
    pub fn apply_vibrato(&mut self, params: &VibratoParams) {
        for (i, &time) in self.time_points.iter().enumerate() {
            if time < params.onset_time {
                continue;
            }

            let phase = params.phase
                + (time - params.onset_time) * params.frequency * 2.0 * std::f32::consts::PI;
            let vibrato_value = match params.waveform {
                VibratoWaveform::Sine => phase.sin(),
                VibratoWaveform::Triangle => {
                    let normalized = (phase / (2.0 * std::f32::consts::PI)) % 1.0;
                    if normalized < 0.5 {
                        4.0 * normalized - 1.0
                    } else {
                        3.0 - 4.0 * normalized
                    }
                }
                VibratoWaveform::Sawtooth => {
                    let normalized = (phase / (2.0 * std::f32::consts::PI)) % 1.0;
                    2.0 * normalized - 1.0
                }
                VibratoWaveform::Square => {
                    if phase.sin() > 0.0 {
                        1.0
                    } else {
                        -1.0
                    }
                }
                VibratoWaveform::Random => (rand::random::<f32>() - 0.5) * 2.0,
            };

            let depth_cents = params.depth * (1.0 + params.depth_variation * (phase * 0.1).sin());
            let pitch_multiplier = 2.0_f32.powf(depth_cents * vibrato_value / 1200.0);

            self.f0_values[i] *= pitch_multiplier;
        }
    }

    /// Detect pitch using autocorrelation
    pub fn detect_pitch(audio: &[f32], sample_rate: f32) -> Option<f32> {
        if audio.len() < 512 {
            return None;
        }

        let mut autocorr = vec![0.0; audio.len() / 2];

        // Compute autocorrelation
        for lag in 0..autocorr.len() {
            let mut sum = 0.0;
            for i in 0..audio.len() - lag {
                sum += audio[i] * audio[i + lag];
            }
            autocorr[lag] = sum;
        }

        // Find the first peak after the center
        let min_period = (sample_rate / 800.0) as usize; // 800 Hz max
        let max_period = (sample_rate / 80.0) as usize; // 80 Hz min

        let mut max_value = 0.0;
        let mut max_lag = 0;

        for lag in min_period..max_period.min(autocorr.len()) {
            if autocorr[lag] > max_value {
                max_value = autocorr[lag];
                max_lag = lag;
            }
        }

        if max_lag > 0 && max_value > 0.3 * autocorr[0] {
            Some(sample_rate / max_lag as f32)
        } else {
            None
        }
    }
}

impl PitchGenerator {
    /// Create new pitch generator
    pub fn new(voice_characteristics: VoiceCharacteristics) -> Self {
        Self {
            voice_characteristics,
            model_params: PitchModelParams::default(),
            vibrato_params: VibratoParams::default(),
            portamento_params: PortamentoParams::default(),
            expression_mappings: HashMap::new(),
        }
    }

    /// Set model parameters
    pub fn set_model_params(&mut self, params: PitchModelParams) {
        self.model_params = params;
    }

    /// Set vibrato parameters
    pub fn set_vibrato_params(&mut self, params: VibratoParams) {
        self.vibrato_params = params;
    }

    /// Set portamento parameters
    pub fn set_portamento_params(&mut self, params: PortamentoParams) {
        self.portamento_params = params;
    }

    /// Add expression mapping
    pub fn add_expression_mapping(
        &mut self,
        expression: Expression,
        params: ExpressionPitchParams,
    ) {
        self.expression_mappings.insert(expression, params);
    }

    /// Generate pitch contour from musical notes
    pub fn generate_contour(&self, notes: &[MusicalNote], duration: f32) -> PitchContour {
        let sample_rate = 100.0; // 100 Hz sampling for pitch contour
        let num_samples = (duration * sample_rate) as usize;
        let mut time_points = Vec::with_capacity(num_samples);
        let mut f0_values = Vec::with_capacity(num_samples);
        let mut confidence = Vec::with_capacity(num_samples);
        let mut voicing = Vec::with_capacity(num_samples);

        for i in 0..num_samples {
            let time = i as f32 / sample_rate;
            time_points.push(time);

            // Find active note at this time
            let active_note = notes.iter().find(|note| {
                let start_time = note.start_time * 60.0 / 120.0; // Convert from beats to seconds
                let end_time = start_time + note.duration * 60.0 / 120.0;
                time >= start_time && time < end_time
            });

            if let Some(note) = active_note {
                let mut target_f0 = note.event.frequency;

                // Apply expression modifications
                if let Some(expr_params) = self.expression_mappings.get(&note.event.expression) {
                    target_f0 *= expr_params.pitch_modifier;
                }

                // Apply pitch bend
                if let Some(pitch_bend) = &note.pitch_bend {
                    let bend_start =
                        note.start_time * 60.0 / 120.0 + pitch_bend.start_time * 60.0 / 120.0;
                    let bend_end = bend_start + pitch_bend.duration * 60.0 / 120.0;

                    if time >= bend_start && time < bend_end {
                        let bend_progress = (time - bend_start) / (bend_end - bend_start);
                        let bend_amount = self.apply_bend_curve(bend_progress, pitch_bend.curve);
                        let bend_multiplier = 2.0_f32.powf(pitch_bend.amount * bend_amount / 12.0);
                        target_f0 *= bend_multiplier;
                    }
                }

                // Apply portamento
                if let Some(prev_note) = self.get_previous_note(notes, note) {
                    let note_start = note.start_time * 60.0 / 120.0;
                    let portamento_duration = self.portamento_params.time;

                    if time >= note_start && time < note_start + portamento_duration {
                        let progress = (time - note_start) / portamento_duration;
                        let curve_progress = self.apply_portamento_curve(progress);
                        let prev_f0 = prev_note.event.frequency;
                        target_f0 = prev_f0 * (1.0 - curve_progress) + target_f0 * curve_progress;
                    }
                }

                // Apply model-based variations
                target_f0 *= self.apply_pitch_variations(time, note);

                f0_values.push(target_f0);
                confidence.push(0.9);
                voicing.push(true);
            } else {
                f0_values.push(0.0);
                confidence.push(0.0);
                voicing.push(false);
            }
        }

        let mut contour = PitchContour {
            time_points,
            f0_values,
            confidence,
            voicing,
            smoothness: 0.5,
            interpolation: InterpolationMethod::Cubic,
        };

        // Apply vibrato
        contour.apply_vibrato(&self.vibrato_params);

        // Smooth the contour
        contour.smooth(self.model_params.stability);

        contour
    }

    /// Get previous note in sequence
    fn get_previous_note<'a>(
        &self,
        notes: &'a [MusicalNote],
        current: &MusicalNote,
    ) -> Option<&'a MusicalNote> {
        let current_start = current.start_time;
        notes
            .iter()
            .filter(|note| note.start_time < current_start)
            .max_by(|a, b| a.start_time.partial_cmp(&b.start_time).unwrap())
    }

    /// Apply bend curve
    fn apply_bend_curve(&self, progress: f32, curve: crate::types::BendCurve) -> f32 {
        match curve {
            crate::types::BendCurve::Linear => progress,
            crate::types::BendCurve::Exponential => progress * progress,
            crate::types::BendCurve::Logarithmic => 1.0 - (1.0 - progress) * (1.0 - progress),
            crate::types::BendCurve::Sine => (progress * std::f32::consts::PI / 2.0).sin(),
            crate::types::BendCurve::Custom => progress, // Simplified
        }
    }

    /// Apply portamento curve
    fn apply_portamento_curve(&self, progress: f32) -> f32 {
        match self.portamento_params.curve {
            PortamentoCurve::Linear => progress,
            PortamentoCurve::Exponential => progress * progress,
            PortamentoCurve::Logarithmic => 1.0 - (1.0 - progress) * (1.0 - progress),
            PortamentoCurve::Sigmoid => 1.0 / (1.0 + (-6.0 * (progress - 0.5)).exp()),
            PortamentoCurve::Bezier => {
                // Simplified Bezier curve
                let t = progress;
                3.0 * t * t * (1.0 - t) + t * t * t
            }
        }
    }

    /// Apply pitch variations
    fn apply_pitch_variations(&self, time: f32, _note: &MusicalNote) -> f32 {
        let mut multiplier = 1.0;

        // Add intonation accuracy
        if self.model_params.intonation_accuracy < 1.0 {
            let deviation = (1.0 - self.model_params.intonation_accuracy) * 0.05;
            multiplier *= 1.0 + (rand::random::<f32>() - 0.5) * deviation;
        }

        // Add microtonal deviation
        if self.model_params.microtonal_deviation > 0.0 {
            let deviation = self.model_params.microtonal_deviation * 0.02;
            multiplier *= 1.0 + (rand::random::<f32>() - 0.5) * deviation;
        }

        // Add pitch variation
        if self.model_params.variation > 0.0 {
            let variation = self.model_params.variation * 0.01;
            multiplier *= 1.0 + (time * 0.5).sin() * variation;
        }

        multiplier
    }
}

impl PitchProcessor {
    /// Create new pitch processor
    pub fn new(sample_rate: f32, frame_size: usize) -> Self {
        let hop_size = frame_size / 4;
        let fft_planner = FftPlanner::new();
        let window = Self::create_window(frame_size);

        Self {
            sample_rate,
            frame_size,
            hop_size,
            fft_planner,
            window,
            pitch_shift: 1.0,
            formant_shift: 1.0,
            buffer: Array2::zeros((2, frame_size)),
            phase_vocoder: PhaseVocoder::new(frame_size),
        }
    }

    /// Create window function
    fn create_window(size: usize) -> Array1<f32> {
        Array1::from_iter(
            (0..size).map(|i| {
                0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (size - 1) as f32).cos()
            }),
        )
    }

    /// Set pitch shift factor
    pub fn set_pitch_shift(&mut self, factor: f32) {
        self.pitch_shift = factor;
    }

    /// Set formant shift factor
    pub fn set_formant_shift(&mut self, factor: f32) {
        self.formant_shift = factor;
    }

    /// Process audio with pitch modification
    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        let mut output = Vec::with_capacity(input.len());

        // Process in overlapping frames
        for chunk in input.chunks(self.hop_size) {
            let frame_output = self.process_frame(chunk);
            output.extend_from_slice(&frame_output);
        }

        output
    }

    /// Process single frame
    fn process_frame(&mut self, input: &[f32]) -> Vec<f32> {
        if input.len() < self.frame_size {
            return input.to_vec();
        }

        // Apply window
        let windowed: Vec<f32> = input
            .iter()
            .zip(self.window.iter())
            .map(|(x, w)| x * w)
            .collect();

        // Convert to complex for FFT
        let mut complex_input: Vec<Complex<f32>> =
            windowed.iter().map(|&x| Complex::new(x, 0.0)).collect();

        // Perform FFT
        let fft = self.fft_planner.plan_fft_forward(self.frame_size);
        fft.process(&mut complex_input);

        // Apply pitch shift using phase vocoder
        let mut shifted_spectrum = self
            .phase_vocoder
            .process_spectrum(&complex_input, self.pitch_shift);

        // Perform inverse FFT
        let ifft = self.fft_planner.plan_fft_inverse(self.frame_size);
        ifft.process(&mut shifted_spectrum);

        // Extract real part and apply window
        let output: Vec<f32> = shifted_spectrum
            .iter()
            .zip(self.window.iter())
            .map(|(x, w)| x.re * w / self.frame_size as f32)
            .collect();

        output
    }
}

impl PhaseVocoder {
    /// Create new phase vocoder
    fn new(frame_size: usize) -> Self {
        let analysis_window = Array1::from_iter((0..frame_size).map(|i| {
            0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (frame_size - 1) as f32).cos()
        }));

        let synthesis_window = analysis_window.clone();

        Self {
            analysis_window,
            synthesis_window,
            previous_phase: Array1::zeros(frame_size),
            output_phase: Array1::zeros(frame_size),
            overlap_buffer: Array1::zeros(frame_size),
        }
    }

    /// Process spectrum for pitch shifting
    fn process_spectrum(&mut self, input: &[Complex<f32>], shift_factor: f32) -> Vec<Complex<f32>> {
        let mut output = vec![Complex::new(0.0, 0.0); input.len()];

        for (i, &complex_val) in input.iter().enumerate() {
            let magnitude = complex_val.norm();
            let phase = complex_val.arg();

            // Calculate phase difference
            let phase_diff = phase - self.previous_phase[i];
            self.previous_phase[i] = phase;

            // Unwrap phase difference
            let unwrapped_diff = phase_diff
                - 2.0 * std::f32::consts::PI * (phase_diff / (2.0 * std::f32::consts::PI)).round();

            // Calculate true frequency
            let true_freq = i as f32 + unwrapped_diff / (2.0 * std::f32::consts::PI);

            // Calculate output frequency
            let output_freq = true_freq * shift_factor;
            let output_bin = output_freq.round() as usize;

            if output_bin < output.len() {
                // Calculate output phase
                self.output_phase[output_bin] += unwrapped_diff * shift_factor;

                // Create output complex value
                output[output_bin] = Complex::new(
                    magnitude * self.output_phase[output_bin].cos(),
                    magnitude * self.output_phase[output_bin].sin(),
                );
            }
        }

        output
    }
}

impl Default for PitchModelParams {
    fn default() -> Self {
        Self {
            base_f0: 220.0,
            f0_range: (80.0, 800.0),
            stability: 0.8,
            variation: 0.3,
            intonation_accuracy: 0.9,
            microtonal_deviation: 0.1,
            glide_time: 0.1,
            overshoot: 0.2,
        }
    }
}

impl Default for VibratoParams {
    fn default() -> Self {
        Self {
            frequency: 6.0,
            depth: 50.0, // cents
            onset_time: 0.5,
            rate_variation: 0.1,
            depth_variation: 0.1,
            waveform: VibratoWaveform::Sine,
            phase: 0.0,
        }
    }
}

impl Default for PortamentoParams {
    fn default() -> Self {
        Self {
            time: 0.1,
            curve: PortamentoCurve::Sigmoid,
            threshold: 3.0,
            strength: 0.5,
            auto_portamento: false,
        }
    }
}

impl Default for ExpressionPitchParams {
    fn default() -> Self {
        Self {
            pitch_modifier: 1.0,
            vibrato_modifier: 1.0,
            stability_modifier: 1.0,
            variation_modifier: 1.0,
            glide_modifier: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::NoteEvent;

    #[test]
    fn test_pitch_contour_creation() {
        let time_points = vec![0.0, 0.5, 1.0, 1.5, 2.0];
        let f0_values = vec![220.0, 240.0, 260.0, 280.0, 300.0];
        let contour = PitchContour::new(time_points, f0_values);

        assert_eq!(contour.time_points.len(), 5);
        assert_eq!(contour.f0_values.len(), 5);
        assert_eq!(contour.confidence.len(), 5);
        assert_eq!(contour.voicing.len(), 5);
    }

    #[test]
    fn test_pitch_interpolation() {
        let time_points = vec![0.0, 1.0, 2.0];
        let f0_values = vec![220.0, 440.0, 220.0];
        let contour = PitchContour::new(time_points, f0_values);

        let f0_mid = contour.f0_at_time(0.5);
        assert!(f0_mid > 220.0 && f0_mid < 440.0);

        let f0_end = contour.f0_at_time(1.5);
        assert!(f0_end > 220.0 && f0_end < 440.0);
    }

    #[test]
    fn test_pitch_detection() {
        // Create a simple sine wave at 440 Hz
        let sample_rate = 44100.0;
        let frequency = 440.0;
        let duration = 0.1; // 100ms
        let samples: Vec<f32> = (0..(sample_rate * duration) as usize)
            .map(|i| (2.0 * std::f32::consts::PI * frequency * i as f32 / sample_rate).sin())
            .collect();

        let detected_pitch = PitchContour::detect_pitch(&samples, sample_rate);
        assert!(detected_pitch.is_some());

        let pitch = detected_pitch.unwrap();
        assert!((pitch - frequency).abs() < 10.0); // Within 10 Hz tolerance
    }

    #[test]
    fn test_pitch_generator() {
        let voice_characteristics = VoiceCharacteristics::default();
        let generator = PitchGenerator::new(voice_characteristics);

        let event = NoteEvent::new("A".to_string(), 4, 1.0, 0.8);
        let note = crate::score::MusicalNote::new(event, 0.0, 1.0);
        let notes = vec![note];

        let contour = generator.generate_contour(&notes, 2.0);
        assert!(!contour.time_points.is_empty());
        assert!(!contour.f0_values.is_empty());
    }

    #[test]
    fn test_pitch_processor() {
        let sample_rate = 44100.0;
        let frame_size = 1024;
        let mut processor = PitchProcessor::new(sample_rate, frame_size);

        // Create test signal
        let test_signal: Vec<f32> = (0..frame_size)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate).sin())
            .collect();

        processor.set_pitch_shift(2.0); // Shift up an octave
        let output = processor.process(&test_signal);

        assert_eq!(output.len(), test_signal.len());
    }

    #[test]
    fn test_vibrato_application() {
        let time_points = vec![0.0, 0.5, 1.0, 1.5, 2.0];
        let f0_values = vec![440.0, 440.0, 440.0, 440.0, 440.0];
        let mut contour = PitchContour::new(time_points, f0_values);

        let vibrato_params = VibratoParams {
            frequency: 6.0,
            depth: 100.0,    // Larger depth to ensure visible change
            onset_time: 0.4, // Earlier onset to affect more points
            rate_variation: 0.1,
            depth_variation: 0.1,
            waveform: VibratoWaveform::Sine,
            phase: 0.0,
        };
        contour.apply_vibrato(&vibrato_params);

        // Check that vibrato was applied after onset time
        let mut found_changed = false;
        for (i, &time) in contour.time_points.iter().enumerate() {
            if time > vibrato_params.onset_time && (contour.f0_values[i] - 440.0).abs() > 0.1 {
                found_changed = true;
            }
        }
        assert!(
            found_changed,
            "Vibrato should change at least one f0 value after onset time"
        );
    }
}

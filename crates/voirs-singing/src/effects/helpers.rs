//! Helper structures for audio effects

use serde::{Deserialize, Serialize};

/// Low-frequency oscillator
#[derive(Debug, Clone)]
pub struct LFO {
    frequency: f32,
    amplitude: f32,
    phase: f32,
    waveform: LFOWaveform,
    sample_rate: f32,
}

impl LFO {
    pub fn new(frequency: f32, amplitude: f32, sample_rate: f32) -> Self {
        Self {
            frequency: frequency.max(0.1),
            amplitude: amplitude.clamp(0.0, 1.0),
            phase: 0.0,
            waveform: LFOWaveform::Sine,
            sample_rate: sample_rate.max(1.0),
        }
    }

    pub fn process(&mut self) -> f32 {
        let output = match self.waveform {
            LFOWaveform::Sine => (self.phase * 2.0 * std::f32::consts::PI).sin(),
            LFOWaveform::Triangle => {
                let normalized = self.phase.fract();
                if normalized < 0.5 {
                    4.0 * normalized - 1.0
                } else {
                    3.0 - 4.0 * normalized
                }
            }
            LFOWaveform::Sawtooth => 2.0 * self.phase.fract() - 1.0,
            LFOWaveform::Square => {
                if self.phase.fract() < 0.5 {
                    -1.0
                } else {
                    1.0
                }
            }
            LFOWaveform::Random => {
                // Simple pseudo-random using linear congruential generator
                let mut state = (self.phase * 1000.0) as u32;
                state = state.wrapping_mul(1103515245).wrapping_add(12345);
                (state as f32 / u32::MAX as f32) * 2.0 - 1.0
            }
        };

        self.phase += self.frequency / self.sample_rate;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }

        output * self.amplitude
    }

    pub fn set_frequency(&mut self, frequency: f32) {
        self.frequency = frequency.max(0.1);
    }

    pub fn set_amplitude(&mut self, amplitude: f32) {
        self.amplitude = amplitude.clamp(0.0, 1.0);
    }

    pub fn set_waveform(&mut self, waveform: LFOWaveform) {
        self.waveform = waveform;
    }

    pub fn reset(&mut self) {
        self.phase = 0.0;
    }
}

/// LFO waveform types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LFOWaveform {
    Sine,
    Triangle,
    Sawtooth,
    Square,
    Random,
}

/// Noise generator
#[derive(Debug, Clone)]
pub struct NoiseGenerator {
    noise_type: NoiseType,
    amplitude: f32,
    rng_state: u64,
}

impl NoiseGenerator {
    pub fn new(noise_type: NoiseType, amplitude: f32) -> Self {
        Self {
            noise_type,
            amplitude: amplitude.clamp(0.0, 1.0),
            rng_state: 1,
        }
    }

    pub fn process(&mut self) -> f32 {
        let white_noise = self.generate_white();

        let output = match self.noise_type {
            NoiseType::White => white_noise,
            NoiseType::Pink => white_noise * 0.7, // Simplified pink noise
            NoiseType::Brown => white_noise * 0.5, // Simplified brown noise
            NoiseType::Breath => {
                // Breath-like noise with low-frequency bias
                white_noise * 0.3 * (1.0 + 0.5 * (self.rng_state as f32 / u64::MAX as f32).sin())
            }
        };

        output * self.amplitude
    }

    fn generate_white(&mut self) -> f32 {
        // Linear congruential generator
        self.rng_state = self.rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        (self.rng_state as f32 / u64::MAX as f32) * 2.0 - 1.0
    }

    pub fn set_amplitude(&mut self, amplitude: f32) {
        self.amplitude = amplitude.clamp(0.0, 1.0);
    }

    pub fn set_type(&mut self, noise_type: NoiseType) {
        self.noise_type = noise_type;
    }
}

/// Noise types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NoiseType {
    White,
    Pink,
    Brown,
    Breath,
}

/// Envelope follower
#[derive(Debug, Clone)]
pub struct EnvelopeFollower {
    attack: f32,
    release: f32,
    envelope: f32,
    sample_rate: f32,
}

impl EnvelopeFollower {
    pub fn new(attack: f32, release: f32, sample_rate: f32) -> Self {
        Self {
            attack: attack.max(0.001),
            release: release.max(0.001),
            envelope: 0.0,
            sample_rate: sample_rate.max(1.0),
        }
    }

    pub fn process(&mut self, input: f32) -> f32 {
        let input_level = input.abs();

        if input_level > self.envelope {
            // Attack
            let coeff = (-1.0 / (self.attack * self.sample_rate)).exp();
            self.envelope = input_level + (self.envelope - input_level) * coeff;
        } else {
            // Release
            let coeff = (-1.0 / (self.release * self.sample_rate)).exp();
            self.envelope = input_level + (self.envelope - input_level) * coeff;
        }

        self.envelope
    }

    pub fn set_attack(&mut self, attack: f32) {
        self.attack = attack.max(0.001);
    }

    pub fn set_release(&mut self, release: f32) {
        self.release = release.max(0.001);
    }

    pub fn reset(&mut self) {
        self.envelope = 0.0;
    }
}

/// Formant filter using resonant bandpass filtering
#[derive(Debug, Clone)]
pub struct FormantFilter {
    freq: f32,
    bandwidth: f32,
    gain: f32,
    q: f32,
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}

impl FormantFilter {
    pub fn new(freq: f32, bandwidth: f32, gain: f32) -> Self {
        Self {
            freq: freq.max(20.0),
            bandwidth: bandwidth.max(1.0),
            gain,
            q: freq / bandwidth,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    pub fn process(&mut self, input: f32, sample_rate: f32) -> f32 {
        let omega = 2.0 * std::f32::consts::PI * self.freq / sample_rate;
        let alpha = omega.sin() / (2.0 * self.q);

        let a = 10.0_f32.powf(self.gain / 40.0);

        let b0 = alpha * a;
        let b1 = 0.0;
        let b2 = -alpha * a;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * omega.cos();
        let a2 = 1.0 - alpha;

        let output = (b0 * input + b1 * self.x1 + b2 * self.x2 - a1 * self.y1 - a2 * self.y2) / a0;

        self.x2 = self.x1;
        self.x1 = input;
        self.y2 = self.y1;
        self.y1 = output;

        output
    }

    pub fn set_parameters(&mut self, freq: f32, bandwidth: f32, gain: f32) {
        self.freq = freq.max(20.0);
        self.bandwidth = bandwidth.max(1.0);
        self.gain = gain;
        self.q = self.freq / self.bandwidth;
    }

    pub fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }
}

/// Anti-formant filter using resonant notch filtering
#[derive(Debug, Clone)]
pub struct AntiFormantFilter {
    freq: f32,
    bandwidth: f32,
    depth: f32,
    q: f32,
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}

impl AntiFormantFilter {
    pub fn new(freq: f32, bandwidth: f32, depth: f32) -> Self {
        Self {
            freq: freq.max(20.0),
            bandwidth: bandwidth.max(1.0),
            depth: depth.clamp(0.0, 1.0),
            q: freq / bandwidth,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    pub fn process(&mut self, input: f32, sample_rate: f32) -> f32 {
        let omega = 2.0 * std::f32::consts::PI * self.freq / sample_rate;
        let alpha = omega.sin() / (2.0 * self.q);

        // Notch filter coefficients
        let b0 = 1.0;
        let b1 = -2.0 * omega.cos();
        let b2 = 1.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * omega.cos();
        let a2 = 1.0 - alpha;

        let filtered =
            (b0 * input + b1 * self.x1 + b2 * self.x2 - a1 * self.y1 - a2 * self.y2) / a0;

        // Mix with dry signal based on depth
        let output = input * (1.0 - self.depth) + filtered * self.depth;

        self.x2 = self.x1;
        self.x1 = input;
        self.y2 = self.y1;
        self.y1 = filtered;

        output
    }

    pub fn set_parameters(&mut self, freq: f32, bandwidth: f32, depth: f32) {
        self.freq = freq.max(20.0);
        self.bandwidth = bandwidth.max(1.0);
        self.depth = depth.clamp(0.0, 1.0);
        self.q = self.freq / self.bandwidth;
    }

    pub fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }
}

/// Interpolation types for spectral envelope
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationType {
    Linear,
    Cubic,
    Spline,
}

/// Types of spectral morphing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MorphType {
    Linear,
    CrossFade,
    SpectralEnvelope,
    HarmonicMorph,
    FormantMorph,
    TimbreTransfer,
}

/// Phase alignment methods for morphing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhaseAlignment {
    None,
    Linear,
    CrossCorrelation,
    PhaseLock,
}

/// Interpolation modes for spectral data
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationMode {
    Linear,
    Logarithmic,
    Exponential,
    Cubic,
    Spectral,
}

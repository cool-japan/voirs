//! Helper structures for audio effects

use serde::{Deserialize, Serialize};

/// Low-frequency oscillator for modulation effects.
///
/// Generates periodic control signals for parameter modulation in effects like chorus and vibrato.
#[derive(Debug, Clone)]
pub struct LFO {
    /// LFO frequency in Hz
    frequency: f32,
    /// Modulation depth/amplitude (0.0-1.0)
    amplitude: f32,
    /// Current phase position (0.0-1.0)
    phase: f32,
    /// Waveform shape
    waveform: LFOWaveform,
    /// Sample rate in Hz
    sample_rate: f32,
}

impl LFO {
    /// Creates a new LFO with specified parameters.
    ///
    /// # Arguments
    ///
    /// * `frequency` - LFO frequency in Hz (minimum 0.1)
    /// * `amplitude` - Modulation depth (0.0-1.0)
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Returns
    ///
    /// A new `LFO` instance with sine waveform by default.
    pub fn new(frequency: f32, amplitude: f32, sample_rate: f32) -> Self {
        Self {
            frequency: frequency.max(0.1),
            amplitude: amplitude.clamp(0.0, 1.0),
            phase: 0.0,
            waveform: LFOWaveform::Sine,
            sample_rate: sample_rate.max(1.0),
        }
    }

    /// Generates the next LFO output sample.
    ///
    /// # Returns
    ///
    /// Modulation value in range -amplitude to +amplitude.
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

    /// Sets the LFO frequency.
    ///
    /// # Arguments
    ///
    /// * `frequency` - New frequency in Hz (minimum 0.1)
    pub fn set_frequency(&mut self, frequency: f32) {
        self.frequency = frequency.max(0.1);
    }

    /// Sets the modulation amplitude/depth.
    ///
    /// # Arguments
    ///
    /// * `amplitude` - New amplitude (clamped to 0.0-1.0)
    pub fn set_amplitude(&mut self, amplitude: f32) {
        self.amplitude = amplitude.clamp(0.0, 1.0);
    }

    /// Sets the waveform shape.
    ///
    /// # Arguments
    ///
    /// * `waveform` - New waveform type
    pub fn set_waveform(&mut self, waveform: LFOWaveform) {
        self.waveform = waveform;
    }

    /// Resets the LFO phase to zero.
    pub fn reset(&mut self) {
        self.phase = 0.0;
    }
}

/// LFO waveform types for modulation effects.
///
/// Defines the shape of low-frequency oscillator output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LFOWaveform {
    /// Smooth sinusoidal waveform
    Sine,
    /// Linear ramp up and down triangle waveform
    Triangle,
    /// Linear ramp sawtooth waveform
    Sawtooth,
    /// Instant-transition square waveform
    Square,
    /// Pseudo-random noise waveform
    Random,
}

/// Noise generator for synthesis effects.
///
/// Generates various types of noise with different spectral characteristics.
#[derive(Debug, Clone)]
pub struct NoiseGenerator {
    /// Type of noise to generate
    noise_type: NoiseType,
    /// Output amplitude (0.0-1.0)
    amplitude: f32,
    /// Internal random number generator state
    rng_state: u64,
}

impl NoiseGenerator {
    /// Creates a new noise generator.
    ///
    /// # Arguments
    ///
    /// * `noise_type` - Type of noise to generate
    /// * `amplitude` - Output amplitude (0.0-1.0)
    ///
    /// # Returns
    ///
    /// A new `NoiseGenerator` instance.
    pub fn new(noise_type: NoiseType, amplitude: f32) -> Self {
        Self {
            noise_type,
            amplitude: amplitude.clamp(0.0, 1.0),
            rng_state: 1,
        }
    }

    /// Generates the next noise sample.
    ///
    /// # Returns
    ///
    /// Noise sample value scaled by amplitude.
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

    /// Generates white noise using linear congruential generator.
    ///
    /// # Returns
    ///
    /// Random value in range -1.0 to 1.0.
    fn generate_white(&mut self) -> f32 {
        // Linear congruential generator
        self.rng_state = self.rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        (self.rng_state as f32 / u64::MAX as f32) * 2.0 - 1.0
    }

    /// Sets the output amplitude.
    ///
    /// # Arguments
    ///
    /// * `amplitude` - New amplitude (clamped to 0.0-1.0)
    pub fn set_amplitude(&mut self, amplitude: f32) {
        self.amplitude = amplitude.clamp(0.0, 1.0);
    }

    /// Sets the noise type.
    ///
    /// # Arguments
    ///
    /// * `noise_type` - New noise type
    pub fn set_type(&mut self, noise_type: NoiseType) {
        self.noise_type = noise_type;
    }
}

/// Noise types for synthesis effects.
///
/// Different spectral characteristics of generated noise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NoiseType {
    /// White noise with flat frequency spectrum
    White,
    /// Pink noise with 1/f power spectrum (more bass)
    Pink,
    /// Brown noise with 1/f² power spectrum (even more bass)
    Brown,
    /// Breath-like noise with low-frequency bias
    Breath,
}

/// Envelope follower for tracking signal amplitude.
///
/// Follows the amplitude envelope of an audio signal with configurable attack and release times.
#[derive(Debug, Clone)]
pub struct EnvelopeFollower {
    /// Attack time in seconds
    attack: f32,
    /// Release time in seconds
    release: f32,
    /// Current envelope value
    envelope: f32,
    /// Sample rate in Hz
    sample_rate: f32,
}

impl EnvelopeFollower {
    /// Creates a new envelope follower.
    ///
    /// # Arguments
    ///
    /// * `attack` - Attack time in seconds (minimum 0.001)
    /// * `release` - Release time in seconds (minimum 0.001)
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Returns
    ///
    /// A new `EnvelopeFollower` instance.
    pub fn new(attack: f32, release: f32, sample_rate: f32) -> Self {
        Self {
            attack: attack.max(0.001),
            release: release.max(0.001),
            envelope: 0.0,
            sample_rate: sample_rate.max(1.0),
        }
    }

    /// Processes a sample and returns the current envelope value.
    ///
    /// # Arguments
    ///
    /// * `input` - Input audio sample
    ///
    /// # Returns
    ///
    /// Current envelope amplitude.
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

    /// Sets the attack time.
    ///
    /// # Arguments
    ///
    /// * `attack` - Attack time in seconds (minimum 0.001)
    pub fn set_attack(&mut self, attack: f32) {
        self.attack = attack.max(0.001);
    }

    /// Sets the release time.
    ///
    /// # Arguments
    ///
    /// * `release` - Release time in seconds (minimum 0.001)
    pub fn set_release(&mut self, release: f32) {
        self.release = release.max(0.001);
    }

    /// Resets the envelope to zero.
    pub fn reset(&mut self) {
        self.envelope = 0.0;
    }
}

/// Formant filter using resonant bandpass filtering.
///
/// Creates vocal formants by emphasizing specific frequency regions with resonant peaks.
#[derive(Debug, Clone)]
pub struct FormantFilter {
    /// Center frequency in Hz
    freq: f32,
    /// Bandwidth in Hz
    bandwidth: f32,
    /// Gain in dB
    gain: f32,
    /// Q factor (derived from freq/bandwidth)
    q: f32,
    /// Previous input sample (x[n-1])
    x1: f32,
    /// Two-sample-delayed input (x[n-2])
    x2: f32,
    /// Previous output sample (y[n-1])
    y1: f32,
    /// Two-sample-delayed output (y[n-2])
    y2: f32,
}

impl FormantFilter {
    /// Creates a new formant filter.
    ///
    /// # Arguments
    ///
    /// * `freq` - Center frequency in Hz (minimum 20.0)
    /// * `bandwidth` - Bandwidth in Hz (minimum 1.0)
    /// * `gain` - Gain in dB
    ///
    /// # Returns
    ///
    /// A new `FormantFilter` instance.
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

    /// Processes a sample through the formant filter.
    ///
    /// # Arguments
    ///
    /// * `input` - Input audio sample
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Returns
    ///
    /// Filtered sample with formant emphasis.
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

    /// Sets all formant parameters at once.
    ///
    /// # Arguments
    ///
    /// * `freq` - Center frequency in Hz (minimum 20.0)
    /// * `bandwidth` - Bandwidth in Hz (minimum 1.0)
    /// * `gain` - Gain in dB
    pub fn set_parameters(&mut self, freq: f32, bandwidth: f32, gain: f32) {
        self.freq = freq.max(20.0);
        self.bandwidth = bandwidth.max(1.0);
        self.gain = gain;
        self.q = self.freq / self.bandwidth;
    }

    /// Resets the filter state by clearing all delay samples.
    pub fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }
}

/// Anti-formant filter using resonant notch filtering.
///
/// Creates anti-formants (notches) that suppress specific frequency regions in vocal spectra.
#[derive(Debug, Clone)]
pub struct AntiFormantFilter {
    /// Center frequency in Hz
    freq: f32,
    /// Bandwidth in Hz
    bandwidth: f32,
    /// Notch depth (0.0-1.0, 0=no effect, 1=full notch)
    depth: f32,
    /// Q factor (derived from freq/bandwidth)
    q: f32,
    /// Previous input sample (x[n-1])
    x1: f32,
    /// Two-sample-delayed input (x[n-2])
    x2: f32,
    /// Previous output sample (y[n-1])
    y1: f32,
    /// Two-sample-delayed output (y[n-2])
    y2: f32,
}

impl AntiFormantFilter {
    /// Creates a new anti-formant filter.
    ///
    /// # Arguments
    ///
    /// * `freq` - Center frequency in Hz (minimum 20.0)
    /// * `bandwidth` - Bandwidth in Hz (minimum 1.0)
    /// * `depth` - Notch depth (0.0-1.0)
    ///
    /// # Returns
    ///
    /// A new `AntiFormantFilter` instance.
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

    /// Processes a sample through the anti-formant filter.
    ///
    /// # Arguments
    ///
    /// * `input` - Input audio sample
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Returns
    ///
    /// Filtered sample with notch applied.
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

    /// Sets all anti-formant parameters at once.
    ///
    /// # Arguments
    ///
    /// * `freq` - Center frequency in Hz (minimum 20.0)
    /// * `bandwidth` - Bandwidth in Hz (minimum 1.0)
    /// * `depth` - Notch depth (0.0-1.0)
    pub fn set_parameters(&mut self, freq: f32, bandwidth: f32, depth: f32) {
        self.freq = freq.max(20.0);
        self.bandwidth = bandwidth.max(1.0);
        self.depth = depth.clamp(0.0, 1.0);
        self.q = self.freq / self.bandwidth;
    }

    /// Resets the filter state by clearing all delay samples.
    pub fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }
}

/// Interpolation types for spectral envelope processing.
///
/// Methods for interpolating between discrete spectral points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationType {
    /// Linear interpolation
    Linear,
    /// Cubic interpolation for smoother curves
    Cubic,
    /// Spline interpolation for natural curves
    Spline,
}

/// Types of spectral morphing algorithms.
///
/// Different approaches to blending or transforming vocal timbres.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MorphType {
    /// Simple linear blend between spectra
    Linear,
    /// Smooth cross-fade with energy preservation
    CrossFade,
    /// Spectral envelope-based morphing
    SpectralEnvelope,
    /// Harmonic structure morphing
    HarmonicMorph,
    /// Formant-preserving morphing
    FormantMorph,
    /// Timbre transfer between voices
    TimbreTransfer,
}

/// Phase alignment methods for morphing operations.
///
/// Techniques for aligning phase information when morphing between signals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhaseAlignment {
    /// No phase alignment
    None,
    /// Linear phase interpolation
    Linear,
    /// Cross-correlation-based alignment
    CrossCorrelation,
    /// Phase-locked morphing
    PhaseLock,
}

/// Interpolation modes for spectral data processing.
///
/// Different scaling methods for interpolating spectral values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationMode {
    /// Linear interpolation in linear space
    Linear,
    /// Logarithmic interpolation
    Logarithmic,
    /// Exponential interpolation
    Exponential,
    /// Cubic polynomial interpolation
    Cubic,
    /// Spectral-aware interpolation
    Spectral,
}

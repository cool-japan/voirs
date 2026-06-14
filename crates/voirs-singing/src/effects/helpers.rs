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
///
/// White noise is produced by an internal linear congruential generator. Pink and
/// brown noise are derived from that white source using IIR filters whose running
/// state is held in the fields below, so successive samples are spectrally correct
/// (true 1/f and 1/f² emphasis) rather than merely amplitude-scaled white noise.
#[derive(Debug, Clone)]
pub struct NoiseGenerator {
    /// Type of noise to generate
    noise_type: NoiseType,
    /// Output amplitude (0.0-1.0)
    amplitude: f32,
    /// Internal random number generator state
    rng_state: u64,
    /// Paul Kellet pink-noise filter state coefficients (b0..b6).
    ///
    /// Seven first-order low-pass sections summed together approximate a
    /// -3 dB/octave (1/f) spectrum. Each element is updated per sample.
    pink_state: [f32; 7],
    /// Brown-noise leaky-integrator state (running accumulator).
    ///
    /// A single-pole integrator of the white source yields a -6 dB/octave
    /// (1/f²) spectrum. The small leak keeps the value from drifting unbounded.
    brown_state: f32,
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
            pink_state: [0.0; 7],
            brown_state: 0.0,
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
            NoiseType::Pink => self.generate_pink(white_noise),
            NoiseType::Brown => self.generate_brown(white_noise),
            NoiseType::Breath => {
                // Breath-like noise with low-frequency bias
                white_noise * 0.3 * (1.0 + 0.5 * (self.rng_state as f32 / u64::MAX as f32).sin())
            }
        };

        output * self.amplitude
    }

    /// Generates one pink-noise sample from a white-noise input.
    ///
    /// Implements Paul Kellet's refined pink-noise filter: seven first-order
    /// low-pass sections (state `b0..b6`) whose coefficients are tuned so the
    /// summed output approximates a -3 dB/octave (1/f) power spectrum across the
    /// audio band. The running state is updated in place every call so the
    /// resulting sequence is genuinely low-frequency-emphasized (its lag-1
    /// autocorrelation is strongly positive), unlike scaled white noise.
    ///
    /// # Arguments
    ///
    /// * `white` - White-noise sample in range -1.0 to 1.0.
    ///
    /// # Returns
    ///
    /// Pink-noise sample, roughly normalized to the same amplitude range as the
    /// white input via the ~0.11 output scaling.
    fn generate_pink(&mut self, white: f32) -> f32 {
        let b = &mut self.pink_state;
        b[0] = 0.99886 * b[0] + white * 0.0555179;
        b[1] = 0.99332 * b[1] + white * 0.0750759;
        b[2] = 0.96900 * b[2] + white * 0.153_852;
        b[3] = 0.86650 * b[3] + white * 0.3104856;
        b[4] = 0.55000 * b[4] + white * 0.5329522;
        b[5] = -0.7616 * b[5] - white * 0.0168980;
        let pink = b[0] + b[1] + b[2] + b[3] + b[4] + b[5] + b[6] + white * 0.5362;
        b[6] = white * 0.115926;
        // Scale so the output sits in approximately the same range as the input.
        pink * 0.11
    }

    /// Generates one brown- (red-) noise sample from a white-noise input.
    ///
    /// Implements a leaky integrator: `running = (running + 0.02 * white) / 1.02`.
    /// Integrating white noise yields a -6 dB/octave (1/f²) spectrum, giving an
    /// even stronger low-frequency emphasis than pink noise. The `/ 1.02` leak
    /// prevents the accumulator from drifting unbounded (DC runaway). The result
    /// is scaled by 3.5 to restore usable amplitude and clamped to [-1.0, 1.0].
    ///
    /// # Arguments
    ///
    /// * `white` - White-noise sample in range -1.0 to 1.0.
    ///
    /// # Returns
    ///
    /// Brown-noise sample clamped to the range -1.0 to 1.0.
    fn generate_brown(&mut self, white: f32) -> f32 {
        self.brown_state = (self.brown_state + 0.02 * white) / 1.02;
        (self.brown_state * 3.5).clamp(-1.0, 1.0)
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Collect `n` raw (unit-amplitude) samples of a given noise type.
    ///
    /// Amplitude is set to 1.0 so the spectral shaping, not the gain, is measured.
    fn collect_noise(noise_type: NoiseType, n: usize) -> Vec<f32> {
        let mut gen = NoiseGenerator::new(noise_type, 1.0);
        (0..n).map(|_| gen.process()).collect()
    }

    /// Normalized lag-1 autocorrelation of a zero-mean version of `samples`.
    ///
    /// Higher values indicate stronger low-frequency emphasis (successive samples
    /// are more correlated). White noise sits near 0; pink is moderately positive;
    /// brown approaches 1.
    fn lag1_autocorrelation(samples: &[f32]) -> f32 {
        let n = samples.len() as f32;
        let mean = samples.iter().sum::<f32>() / n;
        let mut num = 0.0;
        let mut den = 0.0;
        for i in 0..samples.len() {
            let centered = samples[i] - mean;
            den += centered * centered;
            if i + 1 < samples.len() {
                num += centered * (samples[i + 1] - mean);
            }
        }
        if den.abs() < f32::EPSILON {
            0.0
        } else {
            num / den
        }
    }

    #[test]
    fn test_colored_noise_low_frequency_emphasis_ordering() {
        // Deterministic: the LCG seed is fixed, so these sequences are reproducible.
        let n = 16_384;
        let white = collect_noise(NoiseType::White, n);
        let pink = collect_noise(NoiseType::Pink, n);
        let brown = collect_noise(NoiseType::Brown, n);

        let r_white = lag1_autocorrelation(&white);
        let r_pink = lag1_autocorrelation(&pink);
        let r_brown = lag1_autocorrelation(&brown);

        // Expected low-frequency emphasis ordering: brown > pink > white.
        assert!(
            r_brown > r_pink,
            "brown lag-1 autocorr {r_brown} should exceed pink {r_pink}"
        );
        assert!(
            r_pink > r_white,
            "pink lag-1 autocorr {r_pink} should exceed white {r_white}"
        );
        // White should be close to uncorrelated.
        assert!(
            r_white.abs() < 0.2,
            "white lag-1 autocorr {r_white} should be near zero"
        );
        // Brown should be strongly correlated (near a random walk).
        assert!(
            r_brown > 0.8,
            "brown lag-1 autocorr {r_brown} should be strongly positive"
        );
    }

    #[test]
    fn test_pink_brown_not_scaled_white() {
        // Pink/brown must NOT equal white * constant. Compare the first few samples
        // of independent generators (same seed) sample-by-sample: a pure scaling
        // would keep the white/colored ratio constant, which the filters break.
        let n = 64;
        let white = collect_noise(NoiseType::White, n);
        let pink = collect_noise(NoiseType::Pink, n);
        let brown = collect_noise(NoiseType::Brown, n);

        // Ratio of pink[i]/white[i] must vary (not a single constant scale factor).
        let ratio0 = pink[1] / white[1];
        let ratio1 = pink[5] / white[5];
        assert!(
            (ratio0 - ratio1).abs() > 1e-4,
            "pink must be spectrally shaped, not a constant scale of white"
        );

        let bratio0 = brown[1] / white[1];
        let bratio1 = brown[5] / white[5];
        assert!(
            (bratio0 - bratio1).abs() > 1e-4,
            "brown must be spectrally shaped, not a constant scale of white"
        );
    }

    #[test]
    fn test_brown_noise_bounded() {
        // The leaky integrator + clamp keeps brown noise within [-1, 1].
        let samples = collect_noise(NoiseType::Brown, 8192);
        for &s in &samples {
            assert!(
                (-1.0..=1.0).contains(&s),
                "brown noise sample {s} out of [-1, 1]"
            );
        }
    }
}

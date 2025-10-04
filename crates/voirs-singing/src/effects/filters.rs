//! Audio filter components for effects

use std::collections::VecDeque;

/// Delay line for effects implementing variable delay with feedback.
///
/// Provides a circular buffer for delaying audio signals with support for
/// fractional sample delays using linear interpolation and adjustable feedback.
#[derive(Debug, Clone)]
pub struct DelayLine {
    /// Circular buffer storing delayed samples
    buffer: VecDeque<f32>,
    /// Maximum delay capacity in samples
    max_delay_samples: usize,
    /// Current delay amount in samples (supports fractional values)
    delay_samples: f32,
    /// Feedback amount (0.0-0.99)
    feedback: f32,
}

impl DelayLine {
    /// Creates a new delay line with specified parameters.
    ///
    /// # Arguments
    ///
    /// * `max_delay_samples` - Maximum delay capacity in samples
    /// * `delay_samples` - Initial delay amount in samples
    /// * `feedback` - Feedback amount (0.0-0.99)
    ///
    /// # Returns
    ///
    /// A new `DelayLine` instance initialized with zeros.
    pub fn new(max_delay_samples: usize, delay_samples: f32, feedback: f32) -> Self {
        let mut buffer = VecDeque::with_capacity(max_delay_samples);
        buffer.resize(max_delay_samples, 0.0);

        Self {
            buffer,
            max_delay_samples,
            delay_samples: delay_samples.clamp(0.0, max_delay_samples as f32),
            feedback: feedback.clamp(0.0, 0.99),
        }
    }

    /// Processes a single sample through the delay line.
    ///
    /// Uses linear interpolation for fractional delays and applies feedback.
    ///
    /// # Arguments
    ///
    /// * `input` - Input audio sample
    ///
    /// # Returns
    ///
    /// Delayed and possibly feedback-processed audio sample.
    pub fn process(&mut self, input: f32) -> f32 {
        // Simple linear interpolation for fractional delay
        let delay_int = self.delay_samples.floor() as usize;
        let delay_frac = self.delay_samples - delay_int as f32;

        let sample1 = self.buffer.get(delay_int).copied().unwrap_or(0.0);
        let sample2 = self.buffer.get(delay_int + 1).copied().unwrap_or(0.0);

        let delayed_sample = sample1 * (1.0 - delay_frac) + sample2 * delay_frac;

        // Add input with feedback
        let output = input + delayed_sample * self.feedback;

        // Push new sample to buffer
        self.buffer.pop_front();
        self.buffer.push_back(output);

        delayed_sample
    }

    /// Sets the delay amount in samples.
    ///
    /// # Arguments
    ///
    /// * `delay_samples` - New delay amount (clamped to 0.0 to max_delay_samples)
    pub fn set_delay(&mut self, delay_samples: f32) {
        self.delay_samples = delay_samples.clamp(0.0, self.max_delay_samples as f32);
    }

    /// Sets the feedback amount.
    ///
    /// # Arguments
    ///
    /// * `feedback` - New feedback amount (clamped to 0.0-0.99 for stability)
    pub fn set_feedback(&mut self, feedback: f32) {
        self.feedback = feedback.clamp(0.0, 0.99);
    }

    /// Clears the delay buffer by setting all samples to zero.
    pub fn clear(&mut self) {
        self.buffer.iter_mut().for_each(|x| *x = 0.0);
    }
}

/// All-pass filter for reverb and diffusion effects.
///
/// Passes all frequencies equally but introduces phase shifts,
/// useful for creating diffusion in reverb algorithms.
#[derive(Debug, Clone)]
pub struct AllPassFilter {
    /// Internal delay line for all-pass structure
    delay_line: DelayLine,
    /// All-pass gain coefficient (-0.99 to 0.99)
    gain: f32,
}

impl AllPassFilter {
    /// Creates a new all-pass filter.
    ///
    /// # Arguments
    ///
    /// * `delay_samples` - Delay length in samples
    /// * `gain` - All-pass gain coefficient (-0.99 to 0.99)
    ///
    /// # Returns
    ///
    /// A new `AllPassFilter` instance.
    pub fn new(delay_samples: usize, gain: f32) -> Self {
        Self {
            delay_line: DelayLine::new(delay_samples, delay_samples as f32, 0.0),
            gain: gain.clamp(-0.99, 0.99),
        }
    }

    /// Processes a single sample through the all-pass filter.
    ///
    /// # Arguments
    ///
    /// * `input` - Input audio sample
    ///
    /// # Returns
    ///
    /// Phase-shifted audio sample.
    pub fn process(&mut self, input: f32) -> f32 {
        let delayed = self.delay_line.process(input);
        -self.gain * input + delayed
    }

    /// Sets the all-pass gain coefficient.
    ///
    /// # Arguments
    ///
    /// * `gain` - New gain coefficient (clamped to -0.99 to 0.99)
    pub fn set_gain(&mut self, gain: f32) {
        self.gain = gain.clamp(-0.99, 0.99);
    }

    /// Clears the internal delay buffer.
    pub fn clear(&mut self) {
        self.delay_line.clear();
    }
}

/// Low-pass filter using 2-pole Butterworth design.
///
/// Attenuates frequencies above the cutoff frequency, allowing low frequencies to pass.
#[derive(Debug, Clone)]
pub struct LowPassFilter {
    /// Cutoff frequency in Hz
    cutoff: f32,
    /// Resonance/Q factor (0.1-10.0)
    resonance: f32,
    /// Previous input sample (x[n-1])
    x1: f32,
    /// Two-sample-delayed input (x[n-2])
    x2: f32,
    /// Previous output sample (y[n-1])
    y1: f32,
    /// Two-sample-delayed output (y[n-2])
    y2: f32,
}

impl LowPassFilter {
    /// Creates a new low-pass filter.
    ///
    /// # Arguments
    ///
    /// * `cutoff` - Cutoff frequency in Hz (minimum 1.0)
    /// * `resonance` - Q factor controlling resonance (0.1-10.0)
    ///
    /// # Returns
    ///
    /// A new `LowPassFilter` instance.
    pub fn new(cutoff: f32, resonance: f32) -> Self {
        Self {
            cutoff: cutoff.max(1.0),
            resonance: resonance.clamp(0.1, 10.0),
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    /// Processes a single sample through the low-pass filter.
    ///
    /// # Arguments
    ///
    /// * `input` - Input audio sample
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Returns
    ///
    /// Filtered audio sample.
    pub fn process(&mut self, input: f32, sample_rate: f32) -> f32 {
        // Simple 2-pole Butterworth filter
        let omega = 2.0 * std::f32::consts::PI * self.cutoff / sample_rate;
        let cos_omega = omega.cos();
        let sin_omega = omega.sin();
        let q = self.resonance;

        let alpha = sin_omega / (2.0 * q);

        let b0 = (1.0 - cos_omega) / 2.0;
        let b1 = 1.0 - cos_omega;
        let b2 = (1.0 - cos_omega) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_omega;
        let a2 = 1.0 - alpha;

        let output = (b0 * input + b1 * self.x1 + b2 * self.x2 - a1 * self.y1 - a2 * self.y2) / a0;

        self.x2 = self.x1;
        self.x1 = input;
        self.y2 = self.y1;
        self.y1 = output;

        output
    }

    /// Sets the cutoff frequency.
    ///
    /// # Arguments
    ///
    /// * `cutoff` - Cutoff frequency in Hz (minimum 1.0)
    pub fn set_cutoff(&mut self, cutoff: f32) {
        self.cutoff = cutoff.max(1.0);
    }

    /// Sets the resonance/Q factor.
    ///
    /// # Arguments
    ///
    /// * `resonance` - Q factor (clamped to 0.1-10.0)
    pub fn set_resonance(&mut self, resonance: f32) {
        self.resonance = resonance.clamp(0.1, 10.0);
    }

    /// Resets the filter state by clearing all delay samples.
    pub fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }
}

/// High-pass filter using 2-pole Butterworth design.
///
/// Attenuates frequencies below the cutoff frequency, allowing high frequencies to pass.
#[derive(Debug, Clone)]
pub struct HighPassFilter {
    /// Cutoff frequency in Hz
    cutoff: f32,
    /// Resonance/Q factor (0.1-10.0)
    resonance: f32,
    /// Previous input sample (x[n-1])
    x1: f32,
    /// Two-sample-delayed input (x[n-2])
    x2: f32,
    /// Previous output sample (y[n-1])
    y1: f32,
    /// Two-sample-delayed output (y[n-2])
    y2: f32,
}

impl HighPassFilter {
    /// Creates a new high-pass filter.
    ///
    /// # Arguments
    ///
    /// * `cutoff` - Cutoff frequency in Hz (minimum 1.0)
    /// * `resonance` - Q factor controlling resonance (0.1-10.0)
    ///
    /// # Returns
    ///
    /// A new `HighPassFilter` instance.
    pub fn new(cutoff: f32, resonance: f32) -> Self {
        Self {
            cutoff: cutoff.max(1.0),
            resonance: resonance.clamp(0.1, 10.0),
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    /// Processes a single sample through the high-pass filter.
    ///
    /// # Arguments
    ///
    /// * `input` - Input audio sample
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Returns
    ///
    /// Filtered audio sample.
    pub fn process(&mut self, input: f32, sample_rate: f32) -> f32 {
        // Simple 2-pole Butterworth high-pass filter
        let omega = 2.0 * std::f32::consts::PI * self.cutoff / sample_rate;
        let cos_omega = omega.cos();
        let sin_omega = omega.sin();
        let q = self.resonance;

        let alpha = sin_omega / (2.0 * q);

        let b0 = (1.0 + cos_omega) / 2.0;
        let b1 = -(1.0 + cos_omega);
        let b2 = (1.0 + cos_omega) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_omega;
        let a2 = 1.0 - alpha;

        let output = (b0 * input + b1 * self.x1 + b2 * self.x2 - a1 * self.y1 - a2 * self.y2) / a0;

        self.x2 = self.x1;
        self.x1 = input;
        self.y2 = self.y1;
        self.y1 = output;

        output
    }

    /// Sets the cutoff frequency.
    ///
    /// # Arguments
    ///
    /// * `cutoff` - Cutoff frequency in Hz (minimum 1.0)
    pub fn set_cutoff(&mut self, cutoff: f32) {
        self.cutoff = cutoff.max(1.0);
    }

    /// Sets the resonance/Q factor.
    ///
    /// # Arguments
    ///
    /// * `resonance` - Q factor (clamped to 0.1-10.0)
    pub fn set_resonance(&mut self, resonance: f32) {
        self.resonance = resonance.clamp(0.1, 10.0);
    }

    /// Resets the filter state by clearing all delay samples.
    pub fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }
}

/// Band-pass filter allowing a specific frequency band to pass.
///
/// Attenuates frequencies outside the specified band centered around a frequency.
#[derive(Debug, Clone)]
pub struct BandPassFilter {
    /// Center frequency in Hz
    center_freq: f32,
    /// Bandwidth in Hz
    bandwidth: f32,
    /// Previous input sample (x[n-1])
    x1: f32,
    /// Two-sample-delayed input (x[n-2])
    x2: f32,
    /// Previous output sample (y[n-1])
    y1: f32,
    /// Two-sample-delayed output (y[n-2])
    y2: f32,
}

impl BandPassFilter {
    /// Creates a new band-pass filter.
    ///
    /// # Arguments
    ///
    /// * `center_freq` - Center frequency in Hz (minimum 1.0)
    /// * `bandwidth` - Bandwidth in Hz (minimum 1.0)
    ///
    /// # Returns
    ///
    /// A new `BandPassFilter` instance.
    pub fn new(center_freq: f32, bandwidth: f32) -> Self {
        Self {
            center_freq: center_freq.max(1.0),
            bandwidth: bandwidth.max(1.0),
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    /// Processes a single sample through the band-pass filter.
    ///
    /// # Arguments
    ///
    /// * `input` - Input audio sample
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Returns
    ///
    /// Filtered audio sample.
    pub fn process(&mut self, input: f32, sample_rate: f32) -> f32 {
        let omega = 2.0 * std::f32::consts::PI * self.center_freq / sample_rate;
        let q = self.center_freq / self.bandwidth;
        let alpha = omega.sin() / (2.0 * q);

        let b0 = alpha;
        let b1 = 0.0;
        let b2 = -alpha;
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

    /// Sets the center frequency.
    ///
    /// # Arguments
    ///
    /// * `freq` - Center frequency in Hz (minimum 1.0)
    pub fn set_center_freq(&mut self, freq: f32) {
        self.center_freq = freq.max(1.0);
    }

    /// Sets the bandwidth.
    ///
    /// # Arguments
    ///
    /// * `bandwidth` - Bandwidth in Hz (minimum 1.0)
    pub fn set_bandwidth(&mut self, bandwidth: f32) {
        self.bandwidth = bandwidth.max(1.0);
    }

    /// Resets the filter state by clearing all delay samples.
    pub fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }
}

/// Peaking filter for parametric EQ with boost or cut at a specific frequency.
///
/// Allows precise control over frequency-specific gain adjustments.
#[derive(Debug, Clone)]
pub struct PeakingFilter {
    /// Center frequency in Hz
    freq: f32,
    /// Gain in dB (-24.0 to 24.0)
    gain: f32,
    /// Q factor controlling bandwidth (0.1-10.0)
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

impl PeakingFilter {
    /// Creates a new peaking filter.
    ///
    /// # Arguments
    ///
    /// * `freq` - Center frequency in Hz (minimum 1.0)
    /// * `gain` - Gain in dB (typically -24.0 to 24.0)
    /// * `q` - Q factor controlling bandwidth (0.1-10.0)
    ///
    /// # Returns
    ///
    /// A new `PeakingFilter` instance.
    pub fn new(freq: f32, gain: f32, q: f32) -> Self {
        Self {
            freq: freq.max(1.0),
            gain,
            q: q.clamp(0.1, 10.0),
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    /// Processes a single sample through the peaking filter.
    ///
    /// # Arguments
    ///
    /// * `input` - Input audio sample
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Returns
    ///
    /// Filtered audio sample with peak boost or cut applied.
    pub fn process(&mut self, input: f32, sample_rate: f32) -> f32 {
        let omega = 2.0 * std::f32::consts::PI * self.freq / sample_rate;
        let cos_omega = omega.cos();
        let sin_omega = omega.sin();
        let a = 10.0_f32.powf(self.gain / 40.0);
        let alpha = sin_omega / (2.0 * self.q);

        let b0 = 1.0 + (alpha * a);
        let b1 = -2.0 * cos_omega;
        let b2 = 1.0 - (alpha * a);
        let a0 = 1.0 + (alpha / a);
        let a1 = -2.0 * cos_omega;
        let a2 = 1.0 - (alpha / a);

        let output = (b0 * input + b1 * self.x1 + b2 * self.x2 - a1 * self.y1 - a2 * self.y2) / a0;

        self.x2 = self.x1;
        self.x1 = input;
        self.y2 = self.y1;
        self.y1 = output;

        output
    }

    /// Sets the center frequency.
    ///
    /// # Arguments
    ///
    /// * `freq` - Center frequency in Hz (minimum 1.0)
    pub fn set_freq(&mut self, freq: f32) {
        self.freq = freq.max(1.0);
    }

    /// Sets the gain.
    ///
    /// # Arguments
    ///
    /// * `gain` - Gain in dB (clamped to -24.0 to 24.0)
    pub fn set_gain(&mut self, gain: f32) {
        self.gain = gain.clamp(-24.0, 24.0);
    }

    /// Sets the Q factor.
    ///
    /// # Arguments
    ///
    /// * `q` - Q factor (clamped to 0.1-10.0)
    pub fn set_q(&mut self, q: f32) {
        self.q = q.clamp(0.1, 10.0);
    }

    /// Resets the filter state by clearing all delay samples.
    pub fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }
}

/// Low shelf filter for boosting or cutting low frequencies.
///
/// Applies gain to frequencies below the shelf frequency.
#[derive(Debug, Clone)]
pub struct LowShelfFilter {
    /// Shelf frequency in Hz
    freq: f32,
    /// Gain in dB (-24.0 to 24.0)
    gain: f32,
    /// Previous input sample (x[n-1])
    x1: f32,
    /// Previous output sample (y[n-1])
    y1: f32,
}

impl LowShelfFilter {
    /// Creates a new low shelf filter.
    ///
    /// # Arguments
    ///
    /// * `freq` - Shelf frequency in Hz (minimum 1.0)
    /// * `gain` - Gain in dB (typically -24.0 to 24.0)
    ///
    /// # Returns
    ///
    /// A new `LowShelfFilter` instance.
    pub fn new(freq: f32, gain: f32) -> Self {
        Self {
            freq: freq.max(1.0),
            gain,
            x1: 0.0,
            y1: 0.0,
        }
    }

    /// Processes a single sample through the low shelf filter.
    ///
    /// # Arguments
    ///
    /// * `input` - Input audio sample
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Returns
    ///
    /// Filtered audio sample with low frequency gain applied.
    pub fn process(&mut self, input: f32, sample_rate: f32) -> f32 {
        let omega = 2.0 * std::f32::consts::PI * self.freq / sample_rate;
        let s = omega.sin();
        let c = omega.cos();
        let a = 10.0_f32.powf(self.gain / 40.0);

        let beta = a.sqrt() / 1.0; // Q = 1

        let b0 = a * ((a + 1.0) - (a - 1.0) * c + beta * s);
        let b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * c);
        let b2 = a * ((a + 1.0) - (a - 1.0) * c - beta * s);
        let a0 = (a + 1.0) + (a - 1.0) * c + beta * s;
        let a1 = -2.0 * ((a - 1.0) + (a + 1.0) * c);
        let a2 = (a + 1.0) + (a - 1.0) * c - beta * s;

        let output = (b0 * input + b1 * self.x1 - a1 * self.y1 - a2 * 0.0) / a0;

        self.x1 = input;
        self.y1 = output;

        output
    }

    /// Sets the shelf frequency.
    ///
    /// # Arguments
    ///
    /// * `freq` - Shelf frequency in Hz (minimum 1.0)
    pub fn set_freq(&mut self, freq: f32) {
        self.freq = freq.max(1.0);
    }

    /// Sets the gain.
    ///
    /// # Arguments
    ///
    /// * `gain` - Gain in dB (clamped to -24.0 to 24.0)
    pub fn set_gain(&mut self, gain: f32) {
        self.gain = gain.clamp(-24.0, 24.0);
    }

    /// Resets the filter state by clearing all delay samples.
    pub fn reset(&mut self) {
        self.x1 = 0.0;
        self.y1 = 0.0;
    }
}

/// High shelf filter for boosting or cutting high frequencies.
///
/// Applies gain to frequencies above the shelf frequency.
#[derive(Debug, Clone)]
pub struct HighShelfFilter {
    /// Shelf frequency in Hz
    freq: f32,
    /// Gain in dB (-24.0 to 24.0)
    gain: f32,
    /// Previous input sample (x[n-1])
    x1: f32,
    /// Previous output sample (y[n-1])
    y1: f32,
}

impl HighShelfFilter {
    /// Creates a new high shelf filter.
    ///
    /// # Arguments
    ///
    /// * `freq` - Shelf frequency in Hz (minimum 1.0)
    /// * `gain` - Gain in dB (typically -24.0 to 24.0)
    ///
    /// # Returns
    ///
    /// A new `HighShelfFilter` instance.
    pub fn new(freq: f32, gain: f32) -> Self {
        Self {
            freq: freq.max(1.0),
            gain,
            x1: 0.0,
            y1: 0.0,
        }
    }

    /// Processes a single sample through the high shelf filter.
    ///
    /// # Arguments
    ///
    /// * `input` - Input audio sample
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Returns
    ///
    /// Filtered audio sample with high frequency gain applied.
    pub fn process(&mut self, input: f32, sample_rate: f32) -> f32 {
        let omega = 2.0 * std::f32::consts::PI * self.freq / sample_rate;
        let s = omega.sin();
        let c = omega.cos();
        let a = 10.0_f32.powf(self.gain / 40.0);

        let beta = a.sqrt() / 1.0; // Q = 1

        let b0 = a * ((a + 1.0) + (a - 1.0) * c + beta * s);
        let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * c);
        let b2 = a * ((a + 1.0) + (a - 1.0) * c - beta * s);
        let a0 = (a + 1.0) - (a - 1.0) * c + beta * s;
        let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * c);
        let a2 = (a + 1.0) - (a - 1.0) * c - beta * s;

        let output = (b0 * input + b1 * self.x1 - a1 * self.y1 - a2 * 0.0) / a0;

        self.x1 = input;
        self.y1 = output;

        output
    }

    /// Sets the shelf frequency.
    ///
    /// # Arguments
    ///
    /// * `freq` - Shelf frequency in Hz (minimum 1.0)
    pub fn set_freq(&mut self, freq: f32) {
        self.freq = freq.max(1.0);
    }

    /// Sets the gain.
    ///
    /// # Arguments
    ///
    /// * `gain` - Gain in dB (clamped to -24.0 to 24.0)
    pub fn set_gain(&mut self, gain: f32) {
        self.gain = gain.clamp(-24.0, 24.0);
    }

    /// Resets the filter state by clearing all delay samples.
    pub fn reset(&mut self) {
        self.x1 = 0.0;
        self.y1 = 0.0;
    }
}

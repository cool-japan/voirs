//! Audio filter components for effects

use std::collections::VecDeque;

/// Delay line for effects
#[derive(Debug, Clone)]
pub struct DelayLine {
    buffer: VecDeque<f32>,
    max_delay_samples: usize,
    delay_samples: f32,
    feedback: f32,
}

impl DelayLine {
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

    pub fn set_delay(&mut self, delay_samples: f32) {
        self.delay_samples = delay_samples.clamp(0.0, self.max_delay_samples as f32);
    }

    pub fn set_feedback(&mut self, feedback: f32) {
        self.feedback = feedback.clamp(0.0, 0.99);
    }

    pub fn clear(&mut self) {
        self.buffer.iter_mut().for_each(|x| *x = 0.0);
    }
}

/// All-pass filter for reverb
#[derive(Debug, Clone)]
pub struct AllPassFilter {
    delay_line: DelayLine,
    gain: f32,
}

impl AllPassFilter {
    pub fn new(delay_samples: usize, gain: f32) -> Self {
        Self {
            delay_line: DelayLine::new(delay_samples, delay_samples as f32, 0.0),
            gain: gain.clamp(-0.99, 0.99),
        }
    }

    pub fn process(&mut self, input: f32) -> f32 {
        let delayed = self.delay_line.process(input);
        let output = -self.gain * input + delayed;
        output
    }

    pub fn set_gain(&mut self, gain: f32) {
        self.gain = gain.clamp(-0.99, 0.99);
    }

    pub fn clear(&mut self) {
        self.delay_line.clear();
    }
}

/// Low-pass filter
#[derive(Debug, Clone)]
pub struct LowPassFilter {
    cutoff: f32,
    resonance: f32,
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}

impl LowPassFilter {
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

    pub fn set_cutoff(&mut self, cutoff: f32) {
        self.cutoff = cutoff.max(1.0);
    }

    pub fn set_resonance(&mut self, resonance: f32) {
        self.resonance = resonance.clamp(0.1, 10.0);
    }

    pub fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }
}

/// High-pass filter
#[derive(Debug, Clone)]
pub struct HighPassFilter {
    cutoff: f32,
    resonance: f32,
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}

impl HighPassFilter {
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

    pub fn set_cutoff(&mut self, cutoff: f32) {
        self.cutoff = cutoff.max(1.0);
    }

    pub fn set_resonance(&mut self, resonance: f32) {
        self.resonance = resonance.clamp(0.1, 10.0);
    }

    pub fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }
}

/// Band-pass filter
#[derive(Debug, Clone)]
pub struct BandPassFilter {
    center_freq: f32,
    bandwidth: f32,
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}

impl BandPassFilter {
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

    pub fn set_center_freq(&mut self, freq: f32) {
        self.center_freq = freq.max(1.0);
    }

    pub fn set_bandwidth(&mut self, bandwidth: f32) {
        self.bandwidth = bandwidth.max(1.0);
    }

    pub fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }
}

/// Peaking filter for EQ
#[derive(Debug, Clone)]
pub struct PeakingFilter {
    freq: f32,
    gain: f32,
    q: f32,
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}

impl PeakingFilter {
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

    pub fn set_freq(&mut self, freq: f32) {
        self.freq = freq.max(1.0);
    }

    pub fn set_gain(&mut self, gain: f32) {
        self.gain = gain.clamp(-24.0, 24.0);
    }

    pub fn set_q(&mut self, q: f32) {
        self.q = q.clamp(0.1, 10.0);
    }

    pub fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }
}

/// Low shelf filter
#[derive(Debug, Clone)]
pub struct LowShelfFilter {
    freq: f32,
    gain: f32,
    x1: f32,
    y1: f32,
}

impl LowShelfFilter {
    pub fn new(freq: f32, gain: f32) -> Self {
        Self {
            freq: freq.max(1.0),
            gain,
            x1: 0.0,
            y1: 0.0,
        }
    }

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

    pub fn set_freq(&mut self, freq: f32) {
        self.freq = freq.max(1.0);
    }

    pub fn set_gain(&mut self, gain: f32) {
        self.gain = gain.clamp(-24.0, 24.0);
    }

    pub fn reset(&mut self) {
        self.x1 = 0.0;
        self.y1 = 0.0;
    }
}

/// High shelf filter
#[derive(Debug, Clone)]
pub struct HighShelfFilter {
    freq: f32,
    gain: f32,
    x1: f32,
    y1: f32,
}

impl HighShelfFilter {
    pub fn new(freq: f32, gain: f32) -> Self {
        Self {
            freq: freq.max(1.0),
            gain,
            x1: 0.0,
            y1: 0.0,
        }
    }

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

    pub fn set_freq(&mut self, freq: f32) {
        self.freq = freq.max(1.0);
    }

    pub fn set_gain(&mut self, gain: f32) {
        self.gain = gain.clamp(-24.0, 24.0);
    }

    pub fn reset(&mut self) {
        self.x1 = 0.0;
        self.y1 = 0.0;
    }
}

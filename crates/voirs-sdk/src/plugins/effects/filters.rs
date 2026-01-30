//! Helper filters for audio effects.

/// Comb filter for reverb
pub(crate) struct CombFilter {
    pub(crate) buffer: Vec<f32>,
    pub(crate) index: usize,
    pub(crate) feedback: f32,
    pub(crate) filter_state: f32,
    pub(crate) damp1: f32,
    pub(crate) damp2: f32,
}

impl CombFilter {
    pub(crate) fn new(size: usize) -> Self {
        Self {
            buffer: vec![0.0; size],
            index: 0,
            feedback: 0.5,
            filter_state: 0.0,
            damp1: 0.5,
            damp2: 0.5,
        }
    }

    pub(crate) fn process(&mut self, input: f32) -> f32 {
        let output = self.buffer[self.index];
        self.filter_state = (output * self.damp2) + (self.filter_state * self.damp1);
        self.buffer[self.index] = input + (self.filter_state * self.feedback);

        self.index = (self.index + 1) % self.buffer.len();
        output
    }

    pub(crate) fn set_feedback(&mut self, feedback: f32) {
        self.feedback = feedback;
    }

    pub(crate) fn set_damp(&mut self, damp: f32) {
        self.damp1 = damp;
        self.damp2 = 1.0 - damp;
    }
}

/// All-pass filter for reverb
pub(crate) struct AllpassFilter {
    pub(crate) buffer: Vec<f32>,
    pub(crate) index: usize,
    pub(crate) feedback: f32,
}

impl AllpassFilter {
    pub(crate) fn new(size: usize) -> Self {
        Self {
            buffer: vec![0.0; size],
            index: 0,
            feedback: 0.5,
        }
    }

    pub(crate) fn process(&mut self, input: f32) -> f32 {
        let delayed = self.buffer[self.index];
        let output = -input + delayed;
        self.buffer[self.index] = input + (delayed * self.feedback);

        self.index = (self.index + 1) % self.buffer.len();
        output
    }
}

/// Biquad filter for EQ bands
#[derive(Clone)]
pub(crate) struct BiquadFilter {
    // Filter coefficients
    pub(crate) b0: f32,
    pub(crate) b1: f32,
    pub(crate) b2: f32,
    pub(crate) a1: f32,
    pub(crate) a2: f32,

    // Filter memory
    pub(crate) x1: f32,
    pub(crate) x2: f32,
    pub(crate) y1: f32,
    pub(crate) y2: f32,
}

impl BiquadFilter {
    pub(crate) fn new() -> Self {
        Self {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    pub(crate) fn process(&mut self, input: f32) -> f32 {
        let output = self.b0 * input + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;

        self.x2 = self.x1;
        self.x1 = input;
        self.y2 = self.y1;
        self.y1 = output;

        output
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn set_coefficients(
        &mut self,
        b0: f32,
        b1: f32,
        b2: f32,
        a0: f32,
        a1: f32,
        a2: f32,
    ) {
        // Normalize by a0
        self.b0 = b0 / a0;
        self.b1 = b1 / a0;
        self.b2 = b2 / a0;
        self.a1 = a1 / a0;
        self.a2 = a2 / a0;
    }

    pub(crate) fn configure_peaking(
        &mut self,
        sample_rate: f32,
        frequency: f32,
        q: f32,
        gain_db: f32,
    ) {
        use std::f32::consts::PI;

        let omega = 2.0 * PI * frequency / sample_rate;
        let sin_omega = omega.sin();
        let cos_omega = omega.cos();
        let alpha = sin_omega / (2.0 * q);
        let a = 10_f32.powf(gain_db / 40.0);

        let b0 = 1.0 + alpha * a;
        let b1 = -2.0 * cos_omega;
        let b2 = 1.0 - alpha * a;
        let a0 = 1.0 + alpha / a;
        let a1 = -2.0 * cos_omega;
        let a2 = 1.0 - alpha / a;

        self.set_coefficients(b0, b1, b2, a0, a1, a2);
    }

    pub(crate) fn configure_lowpass(&mut self, sample_rate: f32, frequency: f32, q: f32) {
        use std::f32::consts::PI;

        let omega = 2.0 * PI * frequency / sample_rate;
        let sin_omega = omega.sin();
        let cos_omega = omega.cos();
        let alpha = sin_omega / (2.0 * q);

        let b0 = (1.0 - cos_omega) / 2.0;
        let b1 = 1.0 - cos_omega;
        let b2 = (1.0 - cos_omega) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_omega;
        let a2 = 1.0 - alpha;

        self.set_coefficients(b0, b1, b2, a0, a1, a2);
    }

    pub(crate) fn configure_highpass(&mut self, sample_rate: f32, frequency: f32, q: f32) {
        use std::f32::consts::PI;

        let omega = 2.0 * PI * frequency / sample_rate;
        let sin_omega = omega.sin();
        let cos_omega = omega.cos();
        let alpha = sin_omega / (2.0 * q);

        let b0 = (1.0 + cos_omega) / 2.0;
        let b1 = -(1.0 + cos_omega);
        let b2 = (1.0 + cos_omega) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_omega;
        let a2 = 1.0 - alpha;

        self.set_coefficients(b0, b1, b2, a0, a1, a2);
    }

    pub(crate) fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }

    pub(crate) fn set_low_shelf(&mut self, frequency: f32, gain_db: f32, sample_rate: f32) {
        use std::f32::consts::{FRAC_1_SQRT_2, PI};

        let omega = 2.0 * PI * frequency / sample_rate;
        let sin_omega = omega.sin();
        let cos_omega = omega.cos();
        let a = 10_f32.powf(gain_db / 40.0);
        let beta = a.sqrt() * FRAC_1_SQRT_2; // Q = 1/sqrt(2)

        let b0 = a * ((a + 1.0) - (a - 1.0) * cos_omega + beta * sin_omega);
        let b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cos_omega);
        let b2 = a * ((a + 1.0) - (a - 1.0) * cos_omega - beta * sin_omega);
        let a0 = (a + 1.0) + (a - 1.0) * cos_omega + beta * sin_omega;
        let a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cos_omega);
        let a2 = (a + 1.0) + (a - 1.0) * cos_omega - beta * sin_omega;

        self.set_coefficients(b0, b1, b2, a0, a1, a2);
    }

    pub(crate) fn set_high_shelf(&mut self, frequency: f32, gain_db: f32, sample_rate: f32) {
        use std::f32::consts::{FRAC_1_SQRT_2, PI};

        let omega = 2.0 * PI * frequency / sample_rate;
        let sin_omega = omega.sin();
        let cos_omega = omega.cos();
        let a = 10_f32.powf(gain_db / 40.0);
        let beta = a.sqrt() * FRAC_1_SQRT_2; // Q = 1/sqrt(2)

        let b0 = a * ((a + 1.0) + (a - 1.0) * cos_omega + beta * sin_omega);
        let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_omega);
        let b2 = a * ((a + 1.0) + (a - 1.0) * cos_omega - beta * sin_omega);
        let a0 = (a + 1.0) - (a - 1.0) * cos_omega + beta * sin_omega;
        let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_omega);
        let a2 = (a + 1.0) - (a - 1.0) * cos_omega - beta * sin_omega;

        self.set_coefficients(b0, b1, b2, a0, a1, a2);
    }

    pub(crate) fn set_peaking(&mut self, frequency: f32, gain_db: f32, q: f32, sample_rate: f32) {
        self.configure_peaking(sample_rate, frequency, q, gain_db);
    }
}

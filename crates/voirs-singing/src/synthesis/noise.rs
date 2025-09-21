//! Noise processing for synthesis

use ndarray::Array1;

/// Noise types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoiseType {
    /// White noise - equal energy at all frequencies
    White,
    /// Pink noise - 1/f power spectrum
    Pink,
    /// Brown noise - 1/f² power spectrum
    Brown,
    /// Breath noise - filtered noise for breathing sounds
    Breath,
    /// Aspiration noise - high-frequency noise for vocal aspiration
    Aspiration,
}

/// Noise processor
pub struct NoiseProcessor {
    /// Noise level
    noise_level: f32,
    /// Noise type
    noise_type: NoiseType,
    /// Noise buffer
    noise_buffer: Array1<f32>,
    /// Random state
    rng_state: u64,
    /// Pink noise filter state
    pink_state: PinkNoiseState,
    /// Brown noise filter state
    brown_state: BrownNoiseState,
    /// Breath filter
    breath_filter: BreathFilter,
}

/// Pink noise generator state
struct PinkNoiseState {
    b0: f32,
    b1: f32,
    b2: f32,
    b3: f32,
    b4: f32,
    b5: f32,
    b6: f32,
}

/// Brown noise generator state
struct BrownNoiseState {
    last_output: f32,
}

/// Breath filter for breath noise generation
struct BreathFilter {
    /// Low-pass filter coefficients
    a1: f32,
    a2: f32,
    b0: f32,
    b1: f32,
    b2: f32,
    /// Filter state
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}

impl NoiseProcessor {
    /// Create a new noise processor
    pub fn new() -> Self {
        Self {
            noise_level: 0.1,
            noise_type: NoiseType::White,
            noise_buffer: Array1::zeros(1024),
            rng_state: 1,
            pink_state: PinkNoiseState::new(),
            brown_state: BrownNoiseState::new(),
            breath_filter: BreathFilter::new(),
        }
    }

    /// Process audio by adding noise
    pub fn process(&mut self, input: &Array1<f32>) -> crate::Result<Array1<f32>> {
        let mut output = input.clone();

        if self.noise_level > 0.0 {
            for sample in output.iter_mut() {
                let noise_sample = self.generate_noise_sample();
                *sample += noise_sample * self.noise_level;
            }
        }

        Ok(output)
    }

    /// Generate a single noise sample
    fn generate_noise_sample(&mut self) -> f32 {
        match self.noise_type {
            NoiseType::White => self.generate_white(),
            NoiseType::Pink => self.generate_pink(),
            NoiseType::Brown => self.generate_brown(),
            NoiseType::Breath => self.generate_breath(),
            NoiseType::Aspiration => self.generate_aspiration(),
        }
    }

    /// Generate white noise sample
    fn generate_white(&mut self) -> f32 {
        // Linear congruential generator
        self.rng_state = self.rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        (self.rng_state as f32 / u64::MAX as f32) * 2.0 - 1.0
    }

    /// Generate pink noise sample
    fn generate_pink(&mut self) -> f32 {
        let white = self.generate_white();

        // Paul Kellet's implementation
        self.pink_state.b0 = 0.99886 * self.pink_state.b0 + white * 0.0555179;
        self.pink_state.b1 = 0.99332 * self.pink_state.b1 + white * 0.0750759;
        self.pink_state.b2 = 0.96900 * self.pink_state.b2 + white * 0.1538520;
        self.pink_state.b3 = 0.86650 * self.pink_state.b3 + white * 0.3104856;
        self.pink_state.b4 = 0.55000 * self.pink_state.b4 + white * 0.5329522;
        self.pink_state.b5 = -0.7616 * self.pink_state.b5 - white * 0.0168980;

        let output = self.pink_state.b0
            + self.pink_state.b1
            + self.pink_state.b2
            + self.pink_state.b3
            + self.pink_state.b4
            + self.pink_state.b5
            + self.pink_state.b6
            + white * 0.5362;

        self.pink_state.b6 = white * 0.115926;

        output * 0.11 // Scale to reasonable level
    }

    /// Generate brown noise sample
    fn generate_brown(&mut self) -> f32 {
        let white = self.generate_white();

        // Simple integration for brown noise
        self.brown_state.last_output =
            (self.brown_state.last_output + white * 0.02).clamp(-1.0, 1.0);
        self.brown_state.last_output
    }

    /// Generate breath noise sample
    fn generate_breath(&mut self) -> f32 {
        let white = self.generate_white();
        self.breath_filter.process(white)
    }

    /// Generate aspiration noise sample
    fn generate_aspiration(&mut self) -> f32 {
        let white = self.generate_white();

        // High-pass filter for aspiration
        let cutoff = 2000.0; // Hz
        let sample_rate = 44100.0; // Assumed sample rate
        let omega = 2.0 * std::f32::consts::PI * cutoff / sample_rate;
        let alpha = omega.sin() / (2.0 * 0.707); // Q = 0.707

        // Simple high-pass filter
        white * (1.0 + alpha) / 2.0
    }

    /// Set noise level
    pub fn set_level(&mut self, level: f32) {
        self.noise_level = level.clamp(0.0, 1.0);
    }

    /// Set noise type
    pub fn set_type(&mut self, noise_type: NoiseType) {
        self.noise_type = noise_type;
    }

    /// Get noise level
    pub fn level(&self) -> f32 {
        self.noise_level
    }

    /// Get noise type
    pub fn noise_type(&self) -> NoiseType {
        self.noise_type
    }

    /// Reset processor state
    pub fn reset(&mut self) {
        self.noise_buffer.fill(0.0);
        self.pink_state = PinkNoiseState::new();
        self.brown_state = BrownNoiseState::new();
        self.breath_filter.reset();
    }

    /// Fill buffer with noise
    pub fn fill_buffer(&mut self, buffer: &mut [f32]) {
        for sample in buffer.iter_mut() {
            *sample = self.generate_noise_sample();
        }
    }

    /// Get noise energy for gating
    pub fn get_noise_energy(&self, window_size: usize) -> f32 {
        if window_size > self.noise_buffer.len() {
            return 0.0;
        }

        let energy: f32 = self
            .noise_buffer
            .slice(s![..window_size])
            .iter()
            .map(|x| x * x)
            .sum();

        energy / window_size as f32
    }
}

impl PinkNoiseState {
    fn new() -> Self {
        Self {
            b0: 0.0,
            b1: 0.0,
            b2: 0.0,
            b3: 0.0,
            b4: 0.0,
            b5: 0.0,
            b6: 0.0,
        }
    }
}

impl BrownNoiseState {
    fn new() -> Self {
        Self { last_output: 0.0 }
    }
}

impl BreathFilter {
    fn new() -> Self {
        // Low-pass filter coefficients for breath-like sound
        // Butterworth 2nd order, cutoff around 1000 Hz at 44.1 kHz
        let cutoff = 1000.0;
        let sample_rate = 44100.0;
        let omega = 2.0 * std::f32::consts::PI * cutoff / sample_rate;
        let cos_omega = omega.cos();
        let sin_omega = omega.sin();
        let q = 0.707; // Butterworth Q
        let alpha = sin_omega / (2.0 * q);

        let b0 = (1.0 - cos_omega) / 2.0;
        let b1 = 1.0 - cos_omega;
        let b2 = (1.0 - cos_omega) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_omega;
        let a2 = 1.0 - alpha;

        Self {
            a1: a1 / a0,
            a2: a2 / a0,
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    fn process(&mut self, input: f32) -> f32 {
        let output = self.b0 * input + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;

        self.x2 = self.x1;
        self.x1 = input;
        self.y2 = self.y1;
        self.y1 = output;

        output
    }

    fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }
}

impl Default for NoiseProcessor {
    fn default() -> Self {
        Self::new()
    }
}

use ndarray::s;

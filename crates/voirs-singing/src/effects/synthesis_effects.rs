//! Synthesis-based audio effects for singing voice

use super::core::SingingEffect;
use super::filters::BandPassFilter;
use super::helpers::{EnvelopeFollower, NoiseGenerator, NoiseType};
use std::collections::HashMap;

/// Breath noise effect
#[derive(Debug, Clone)]
pub struct BreathNoiseEffect {
    name: String,
    parameters: HashMap<String, f32>,
    noise_generator: NoiseGenerator,
    envelope_follower: EnvelopeFollower,
    filter: BandPassFilter,
}

impl BreathNoiseEffect {
    pub fn new(mut parameters: HashMap<String, f32>) -> Self {
        let mut effect = Self {
            name: "breath_noise".to_string(),
            parameters: HashMap::new(),
            noise_generator: NoiseGenerator::new(NoiseType::Breath, 0.1),
            envelope_follower: EnvelopeFollower::new(0.01, 0.1, 44100.0),
            filter: BandPassFilter::new(2000.0, 1000.0),
        };

        // Initialize default parameters
        effect.parameters.insert("level".to_string(), 0.1);
        effect.parameters.insert("frequency".to_string(), 2000.0);
        effect.parameters.insert("bandwidth".to_string(), 1000.0);
        effect.parameters.insert("sensitivity".to_string(), 0.5);

        // Override with provided parameters
        for (key, value) in parameters.drain() {
            effect.parameters.insert(key, value);
        }

        effect
    }
}

impl SingingEffect for BreathNoiseEffect {
    fn name(&self) -> &str {
        &self.name
    }

    fn process(&mut self, audio: &mut [f32], sample_rate: f32) -> crate::Result<()> {
        let level = *self.parameters.get("level").unwrap_or(&0.1);
        let sensitivity = *self.parameters.get("sensitivity").unwrap_or(&0.5);

        for sample in audio.iter_mut() {
            // Follow the envelope of the input signal
            let envelope = self.envelope_follower.process(*sample);

            // Generate breath noise proportional to the signal level
            let noise = self.noise_generator.process();
            let filtered_noise = self.filter.process(noise, sample_rate);

            // Mix breath noise with original signal
            let breath_amount = envelope * sensitivity * level;
            *sample += filtered_noise * breath_amount;
        }

        Ok(())
    }

    fn set_parameter(&mut self, name: &str, value: f32) -> crate::Result<()> {
        self.parameters.insert(name.to_string(), value);

        match name {
            "level" => {
                self.noise_generator.set_amplitude(value);
            }
            "frequency" => {
                self.filter.set_center_freq(value);
            }
            "bandwidth" => {
                self.filter.set_bandwidth(value);
            }
            _ => {}
        }

        Ok(())
    }

    fn get_parameter(&self, name: &str) -> Option<f32> {
        self.parameters.get(name).copied()
    }

    fn get_parameters(&self) -> HashMap<String, f32> {
        self.parameters.clone()
    }

    fn reset(&mut self) {
        self.envelope_follower.reset();
        self.filter.reset();
    }

    fn clone_effect(&self) -> Box<dyn SingingEffect> {
        Box::new(self.clone())
    }
}

/// Vocal fry effect for low-frequency vocal texture
#[derive(Debug, Clone)]
pub struct VocalFryEffect {
    name: String,
    parameters: HashMap<String, f32>,
    pulse_generator: PulseGenerator,
    envelope_follower: EnvelopeFollower,
    sample_rate: f32,
}

impl VocalFryEffect {
    pub fn new(mut parameters: HashMap<String, f32>) -> Self {
        let mut effect = Self {
            name: "vocal_fry".to_string(),
            parameters: HashMap::new(),
            pulse_generator: PulseGenerator::new(50.0, 0.1),
            envelope_follower: EnvelopeFollower::new(0.01, 0.05, 44100.0),
            sample_rate: 44100.0,
        };

        // Initialize default parameters
        effect.parameters.insert("frequency".to_string(), 50.0);
        effect.parameters.insert("intensity".to_string(), 0.1);
        effect.parameters.insert("irregularity".to_string(), 0.3);
        effect.parameters.insert("threshold".to_string(), 0.1);

        // Override with provided parameters
        for (key, value) in parameters.drain() {
            effect.parameters.insert(key, value);
        }

        effect
    }
}

impl SingingEffect for VocalFryEffect {
    fn name(&self) -> &str {
        &self.name
    }

    fn process(&mut self, audio: &mut [f32], sample_rate: f32) -> crate::Result<()> {
        self.sample_rate = sample_rate;

        let intensity = *self.parameters.get("intensity").unwrap_or(&0.1);
        let threshold = *self.parameters.get("threshold").unwrap_or(&0.1);

        for sample in audio.iter_mut() {
            let envelope = self.envelope_follower.process(*sample);

            // Only apply vocal fry at low signal levels
            if envelope < threshold {
                let pulse = self.pulse_generator.process(sample_rate);
                *sample += pulse * intensity * (threshold - envelope) / threshold;
            }
        }

        Ok(())
    }

    fn set_parameter(&mut self, name: &str, value: f32) -> crate::Result<()> {
        self.parameters.insert(name.to_string(), value);

        match name {
            "frequency" => {
                self.pulse_generator.set_frequency(value);
            }
            "intensity" => {
                self.pulse_generator.set_amplitude(value);
            }
            "irregularity" => {
                self.pulse_generator.set_irregularity(value);
            }
            _ => {}
        }

        Ok(())
    }

    fn get_parameter(&self, name: &str) -> Option<f32> {
        self.parameters.get(name).copied()
    }

    fn get_parameters(&self) -> HashMap<String, f32> {
        self.parameters.clone()
    }

    fn reset(&mut self) {
        self.pulse_generator.reset();
        self.envelope_follower.reset();
    }

    fn clone_effect(&self) -> Box<dyn SingingEffect> {
        Box::new(self.clone())
    }
}

/// Pulse generator for vocal fry effects
#[derive(Debug, Clone)]
struct PulseGenerator {
    frequency: f32,
    amplitude: f32,
    phase: f32,
    irregularity: f32,
    random_state: u32,
}

impl PulseGenerator {
    fn new(frequency: f32, amplitude: f32) -> Self {
        Self {
            frequency: frequency.max(1.0),
            amplitude: amplitude.clamp(0.0, 1.0),
            phase: 0.0,
            irregularity: 0.0,
            random_state: 1,
        }
    }

    fn process(&mut self, sample_rate: f32) -> f32 {
        // Generate irregular pulses
        let freq_variation = if self.irregularity > 0.0 {
            let random = self.generate_random();
            1.0 + (random - 0.5) * self.irregularity
        } else {
            1.0
        };

        let effective_freq = self.frequency * freq_variation;
        self.phase += effective_freq / sample_rate;

        let output = if self.phase >= 1.0 {
            self.phase -= 1.0;
            self.amplitude * (1.0 - 2.0 * self.phase).max(0.0) // Impulse-like pulse
        } else {
            0.0
        };

        output
    }

    fn generate_random(&mut self) -> f32 {
        self.random_state = self
            .random_state
            .wrapping_mul(1103515245)
            .wrapping_add(12345);
        self.random_state as f32 / u32::MAX as f32
    }

    fn set_frequency(&mut self, frequency: f32) {
        self.frequency = frequency.max(1.0);
    }

    fn set_amplitude(&mut self, amplitude: f32) {
        self.amplitude = amplitude.clamp(0.0, 1.0);
    }

    fn set_irregularity(&mut self, irregularity: f32) {
        self.irregularity = irregularity.clamp(0.0, 1.0);
    }

    fn reset(&mut self) {
        self.phase = 0.0;
    }
}

/// Harmonics generator for rich vocal textures
#[derive(Debug, Clone)]
pub struct HarmonicsEffect {
    name: String,
    parameters: HashMap<String, f32>,
    oscillators: Vec<Oscillator>,
    fundamental_tracker: FundamentalTracker,
}

impl HarmonicsEffect {
    pub fn new(mut parameters: HashMap<String, f32>) -> Self {
        let mut effect = Self {
            name: "harmonics".to_string(),
            parameters: HashMap::new(),
            oscillators: Vec::new(),
            fundamental_tracker: FundamentalTracker::new(),
        };

        // Initialize default parameters
        effect.parameters.insert("harmonics_count".to_string(), 5.0);
        effect.parameters.insert("intensity".to_string(), 0.2);
        effect.parameters.insert("decay_rate".to_string(), 0.8);

        // Override with provided parameters
        for (key, value) in parameters.drain() {
            effect.parameters.insert(key, value);
        }

        // Initialize oscillators
        effect.initialize_oscillators();
        effect
    }

    fn initialize_oscillators(&mut self) {
        let count = *self.parameters.get("harmonics_count").unwrap_or(&5.0) as usize;
        self.oscillators.clear();

        for i in 1..=count {
            self.oscillators
                .push(Oscillator::new(440.0 * i as f32, 0.1 / i as f32));
        }
    }
}

impl SingingEffect for HarmonicsEffect {
    fn name(&self) -> &str {
        &self.name
    }

    fn process(&mut self, audio: &mut [f32], sample_rate: f32) -> crate::Result<()> {
        let intensity = *self.parameters.get("intensity").unwrap_or(&0.2);
        let decay_rate = *self.parameters.get("decay_rate").unwrap_or(&0.8);

        for sample in audio.iter_mut() {
            let fundamental = self.fundamental_tracker.process(*sample, sample_rate);

            if fundamental > 20.0 && fundamental < 2000.0 {
                let mut harmonics_sum = 0.0;

                for (i, oscillator) in self.oscillators.iter_mut().enumerate() {
                    let harmonic_freq = fundamental * (i + 2) as f32; // Start from 2nd harmonic
                    let harmonic_amp = intensity * decay_rate.powi(i as i32);

                    oscillator.set_frequency(harmonic_freq);
                    oscillator.set_amplitude(harmonic_amp);
                    harmonics_sum += oscillator.process(sample_rate);
                }

                *sample += harmonics_sum;
            }
        }

        Ok(())
    }

    fn set_parameter(&mut self, name: &str, value: f32) -> crate::Result<()> {
        self.parameters.insert(name.to_string(), value);

        match name {
            "harmonics_count" => {
                self.initialize_oscillators();
            }
            _ => {}
        }

        Ok(())
    }

    fn get_parameter(&self, name: &str) -> Option<f32> {
        self.parameters.get(name).copied()
    }

    fn get_parameters(&self) -> HashMap<String, f32> {
        self.parameters.clone()
    }

    fn reset(&mut self) {
        for oscillator in &mut self.oscillators {
            oscillator.reset();
        }
        self.fundamental_tracker.reset();
    }

    fn clone_effect(&self) -> Box<dyn SingingEffect> {
        Box::new(self.clone())
    }
}

/// Simple oscillator for harmonic generation
#[derive(Debug, Clone)]
struct Oscillator {
    frequency: f32,
    amplitude: f32,
    phase: f32,
}

impl Oscillator {
    fn new(frequency: f32, amplitude: f32) -> Self {
        Self {
            frequency: frequency.max(1.0),
            amplitude: amplitude.clamp(0.0, 1.0),
            phase: 0.0,
        }
    }

    fn process(&mut self, sample_rate: f32) -> f32 {
        let output = (self.phase * 2.0 * std::f32::consts::PI).sin() * self.amplitude;
        self.phase += self.frequency / sample_rate;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }
        output
    }

    fn set_frequency(&mut self, frequency: f32) {
        self.frequency = frequency.max(1.0);
    }

    fn set_amplitude(&mut self, amplitude: f32) {
        self.amplitude = amplitude.clamp(0.0, 1.0);
    }

    fn reset(&mut self) {
        self.phase = 0.0;
    }
}

/// Simple fundamental frequency tracker
#[derive(Debug, Clone)]
struct FundamentalTracker {
    buffer: Vec<f32>,
    buffer_index: usize,
}

impl FundamentalTracker {
    fn new() -> Self {
        Self {
            buffer: vec![0.0; 1024],
            buffer_index: 0,
        }
    }

    fn process(&mut self, input: f32, sample_rate: f32) -> f32 {
        self.buffer[self.buffer_index] = input;
        self.buffer_index = (self.buffer_index + 1) % self.buffer.len();

        // Simple autocorrelation-based pitch detection
        let mut best_period = 0;
        let mut best_correlation = 0.0;

        let min_period = (sample_rate / 2000.0) as usize; // 2000 Hz max
        let max_period = (sample_rate / 50.0) as usize; // 50 Hz min

        for period in min_period..max_period.min(self.buffer.len() / 2) {
            let mut correlation = 0.0;
            for i in 0..self.buffer.len() - period {
                correlation += self.buffer[i] * self.buffer[i + period];
            }

            if correlation > best_correlation {
                best_correlation = correlation;
                best_period = period;
            }
        }

        if best_period > 0 && best_correlation > 0.1 {
            sample_rate / best_period as f32
        } else {
            0.0
        }
    }

    fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.buffer_index = 0;
    }
}

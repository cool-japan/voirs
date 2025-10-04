//! Dynamic audio effects for singing synthesis

use super::core::SingingEffect;
use super::helpers::EnvelopeFollower;
use std::collections::HashMap;

/// Dynamic range compressor effect for controlling vocal dynamics.
///
/// Reduces the dynamic range of audio by attenuating signals above a threshold,
/// making quiet parts louder and loud parts quieter relative to each other.
#[derive(Debug, Clone)]
pub struct CompressorEffect {
    /// Effect identifier name
    name: String,
    /// Effect parameters (threshold, ratio, attack, release, makeup_gain)
    parameters: HashMap<String, f32>,
    /// Envelope follower for tracking signal level
    envelope_follower: EnvelopeFollower,
    /// Current gain reduction amount in dB
    gain_reduction: f32,
    /// Current sample rate in Hz
    sample_rate: f32,
}

impl CompressorEffect {
    /// Creates a new compressor effect with specified parameters.
    ///
    /// # Arguments
    ///
    /// * `parameters` - Optional parameter overrides:
    ///   - `threshold`: Compression threshold in dB (default: -20.0)
    ///   - `ratio`: Compression ratio (default: 4.0)
    ///   - `attack`: Attack time in seconds (default: 0.003)
    ///   - `release`: Release time in seconds (default: 0.1)
    ///   - `makeup_gain`: Output gain compensation in dB (default: 0.0)
    ///
    /// # Returns
    ///
    /// A new `CompressorEffect` instance with default or custom parameters.
    pub fn new(mut parameters: HashMap<String, f32>) -> Self {
        let mut effect = Self {
            name: "compressor".to_string(),
            parameters: HashMap::new(),
            envelope_follower: EnvelopeFollower::new(0.003, 0.1, 44100.0),
            gain_reduction: 0.0,
            sample_rate: 44100.0,
        };

        // Initialize default parameters
        effect.parameters.insert("threshold".to_string(), -20.0);
        effect.parameters.insert("ratio".to_string(), 4.0);
        effect.parameters.insert("attack".to_string(), 0.003);
        effect.parameters.insert("release".to_string(), 0.1);
        effect.parameters.insert("makeup_gain".to_string(), 0.0);

        // Override with provided parameters
        for (key, value) in parameters.drain() {
            effect.parameters.insert(key, value);
        }

        effect
    }

    /// Converts decibels to linear amplitude.
    ///
    /// # Arguments
    ///
    /// * `db` - Decibel value
    ///
    /// # Returns
    ///
    /// Linear amplitude value
    fn db_to_linear(db: f32) -> f32 {
        10.0_f32.powf(db / 20.0)
    }

    /// Converts linear amplitude to decibels.
    ///
    /// # Arguments
    ///
    /// * `linear` - Linear amplitude value
    ///
    /// # Returns
    ///
    /// Decibel value
    fn linear_to_db(linear: f32) -> f32 {
        20.0 * linear.max(1e-10).log10()
    }
}

impl SingingEffect for CompressorEffect {
    fn name(&self) -> &str {
        &self.name
    }

    fn process(&mut self, audio: &mut [f32], sample_rate: f32) -> crate::Result<()> {
        self.sample_rate = sample_rate;

        let threshold = *self.parameters.get("threshold").unwrap_or(&-20.0);
        let ratio = *self.parameters.get("ratio").unwrap_or(&4.0);
        let makeup_gain = *self.parameters.get("makeup_gain").unwrap_or(&0.0);

        let threshold_linear = Self::db_to_linear(threshold);
        let makeup_gain_linear = Self::db_to_linear(makeup_gain);

        for sample in audio.iter_mut() {
            // Get the envelope of the input signal
            let envelope = self.envelope_follower.process(*sample);

            // Calculate compression
            if envelope > threshold_linear {
                let over_threshold_db = Self::linear_to_db(envelope) - threshold;
                let compressed_db = over_threshold_db / ratio;
                self.gain_reduction = over_threshold_db - compressed_db;
                let gain_linear = Self::db_to_linear(-self.gain_reduction);
                *sample *= gain_linear;
            }

            // Apply makeup gain
            *sample *= makeup_gain_linear;
        }

        Ok(())
    }

    fn set_parameter(&mut self, name: &str, value: f32) -> crate::Result<()> {
        self.parameters.insert(name.to_string(), value);

        match name {
            "attack" => {
                self.envelope_follower.set_attack(value);
            }
            "release" => {
                self.envelope_follower.set_release(value);
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
        self.gain_reduction = 0.0;
    }

    fn clone_effect(&self) -> Box<dyn SingingEffect> {
        Box::new(self.clone())
    }
}

/// Gate effect for noise reduction by attenuating signals below a threshold.
///
/// A noise gate that only allows audio above a certain threshold to pass,
/// effectively silencing background noise during quiet passages.
#[derive(Debug, Clone)]
pub struct GateEffect {
    /// Effect identifier name
    name: String,
    /// Effect parameters (threshold, ratio, attack, release)
    parameters: HashMap<String, f32>,
    /// Envelope follower for tracking signal level
    envelope_follower: EnvelopeFollower,
    /// Current gate state (true = open/passing signal)
    is_open: bool,
    /// Current sample rate in Hz
    sample_rate: f32,
}

impl GateEffect {
    /// Creates a new gate effect with specified parameters.
    ///
    /// # Arguments
    ///
    /// * `parameters` - Optional parameter overrides:
    ///   - `threshold`: Gate threshold in dB (default: -40.0)
    ///   - `ratio`: Attenuation ratio when gate is closed (default: 10.0)
    ///   - `attack`: Gate opening time in seconds (default: 0.001)
    ///   - `release`: Gate closing time in seconds (default: 0.05)
    ///
    /// # Returns
    ///
    /// A new `GateEffect` instance with default or custom parameters.
    pub fn new(mut parameters: HashMap<String, f32>) -> Self {
        let mut effect = Self {
            name: "gate".to_string(),
            parameters: HashMap::new(),
            envelope_follower: EnvelopeFollower::new(0.001, 0.05, 44100.0),
            is_open: false,
            sample_rate: 44100.0,
        };

        // Initialize default parameters
        effect.parameters.insert("threshold".to_string(), -40.0);
        effect.parameters.insert("ratio".to_string(), 10.0);
        effect.parameters.insert("attack".to_string(), 0.001);
        effect.parameters.insert("release".to_string(), 0.05);

        // Override with provided parameters
        for (key, value) in parameters.drain() {
            effect.parameters.insert(key, value);
        }

        effect
    }
}

impl SingingEffect for GateEffect {
    fn name(&self) -> &str {
        &self.name
    }

    fn process(&mut self, audio: &mut [f32], sample_rate: f32) -> crate::Result<()> {
        self.sample_rate = sample_rate;

        let threshold = *self.parameters.get("threshold").unwrap_or(&-40.0);
        let ratio = *self.parameters.get("ratio").unwrap_or(&10.0);

        let threshold_linear = 10.0_f32.powf(threshold / 20.0);

        for sample in audio.iter_mut() {
            let envelope = self.envelope_follower.process(*sample);

            if envelope > threshold_linear {
                self.is_open = true;
            } else {
                // Apply gating
                let reduction = (threshold_linear / envelope.max(1e-10)).min(ratio);
                *sample /= reduction;
                if envelope < threshold_linear * 0.1 {
                    self.is_open = false;
                }
            }

            if !self.is_open {
                *sample *= 0.1; // Heavy attenuation when gate is closed
            }
        }

        Ok(())
    }

    fn set_parameter(&mut self, name: &str, value: f32) -> crate::Result<()> {
        self.parameters.insert(name.to_string(), value);

        match name {
            "attack" => {
                self.envelope_follower.set_attack(value);
            }
            "release" => {
                self.envelope_follower.set_release(value);
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
        self.is_open = false;
    }

    fn clone_effect(&self) -> Box<dyn SingingEffect> {
        Box::new(self.clone())
    }
}

/// Limiter effect to prevent clipping by hard-limiting peaks.
///
/// A specialized compressor with a very high ratio that prevents audio
/// from exceeding a maximum threshold, protecting against distortion.
#[derive(Debug, Clone)]
pub struct LimiterEffect {
    /// Effect identifier name
    name: String,
    /// Effect parameters (threshold, release)
    parameters: HashMap<String, f32>,
    /// Envelope follower for fast peak detection
    envelope_follower: EnvelopeFollower,
    /// Current sample rate in Hz
    sample_rate: f32,
}

impl LimiterEffect {
    /// Creates a new limiter effect with specified parameters.
    ///
    /// # Arguments
    ///
    /// * `parameters` - Optional parameter overrides:
    ///   - `threshold`: Limiting threshold in dB (default: -3.0)
    ///   - `release`: Release time in seconds (default: 0.01)
    ///
    /// # Returns
    ///
    /// A new `LimiterEffect` instance with default or custom parameters.
    pub fn new(mut parameters: HashMap<String, f32>) -> Self {
        let mut effect = Self {
            name: "limiter".to_string(),
            parameters: HashMap::new(),
            envelope_follower: EnvelopeFollower::new(0.0001, 0.01, 44100.0),
            sample_rate: 44100.0,
        };

        // Initialize default parameters
        effect.parameters.insert("threshold".to_string(), -3.0);
        effect.parameters.insert("release".to_string(), 0.01);

        // Override with provided parameters
        for (key, value) in parameters.drain() {
            effect.parameters.insert(key, value);
        }

        effect
    }
}

impl SingingEffect for LimiterEffect {
    fn name(&self) -> &str {
        &self.name
    }

    fn process(&mut self, audio: &mut [f32], sample_rate: f32) -> crate::Result<()> {
        self.sample_rate = sample_rate;

        let threshold = *self.parameters.get("threshold").unwrap_or(&-3.0);
        let threshold_linear = 10.0_f32.powf(threshold / 20.0);

        for sample in audio.iter_mut() {
            let envelope = self.envelope_follower.process(*sample);

            if envelope > threshold_linear {
                let gain_reduction = threshold_linear / envelope;
                *sample *= gain_reduction;
            }
        }

        Ok(())
    }

    fn set_parameter(&mut self, name: &str, value: f32) -> crate::Result<()> {
        self.parameters.insert(name.to_string(), value);

        if name == "release" {
            self.envelope_follower.set_release(value);
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
    }

    fn clone_effect(&self) -> Box<dyn SingingEffect> {
        Box::new(self.clone())
    }
}

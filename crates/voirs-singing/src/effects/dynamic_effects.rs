//! Dynamic audio effects for singing synthesis

use super::core::SingingEffect;
use super::helpers::EnvelopeFollower;
use std::collections::HashMap;

/// Compressor effect
#[derive(Debug, Clone)]
pub struct CompressorEffect {
    name: String,
    parameters: HashMap<String, f32>,
    envelope_follower: EnvelopeFollower,
    gain_reduction: f32,
    sample_rate: f32,
}

impl CompressorEffect {
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

    fn db_to_linear(db: f32) -> f32 {
        10.0_f32.powf(db / 20.0)
    }

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

/// Gate effect for noise reduction
#[derive(Debug, Clone)]
pub struct GateEffect {
    name: String,
    parameters: HashMap<String, f32>,
    envelope_follower: EnvelopeFollower,
    is_open: bool,
    sample_rate: f32,
}

impl GateEffect {
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

/// Limiter effect to prevent clipping
#[derive(Debug, Clone)]
pub struct LimiterEffect {
    name: String,
    parameters: HashMap<String, f32>,
    envelope_follower: EnvelopeFollower,
    sample_rate: f32,
}

impl LimiterEffect {
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

        match name {
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
    }

    fn clone_effect(&self) -> Box<dyn SingingEffect> {
        Box::new(self.clone())
    }
}

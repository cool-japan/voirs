//! Dynamic range processing effects for audio enhancement.
//!
//! This module implements compressor, limiter, noise gate, and automatic gain control
//! for speech and audio post-processing.

use crate::{AudioBuffer, Result, VocoderError};
use super::{AudioEffect, EffectParameter};

/// Compressor for dynamic range control
pub struct Compressor {
    enabled: bool,
    threshold: EffectParameter,      // dB threshold
    ratio: EffectParameter,          // compression ratio
    attack: EffectParameter,         // attack time in ms
    release: EffectParameter,        // release time in ms
    makeup_gain: EffectParameter,    // makeup gain in dB
    knee_width: EffectParameter,     // knee width in dB
    
    // Internal state
    envelope: f32,
    sample_rate: u32,
    attack_coeff: f32,
    release_coeff: f32,
}

impl Compressor {
    pub fn new(sample_rate: u32) -> Self {
        let mut compressor = Self {
            enabled: true,
            threshold: EffectParameter::new("threshold", -20.0, -60.0, 0.0),
            ratio: EffectParameter::new("ratio", 4.0, 1.0, 20.0),
            attack: EffectParameter::new("attack", 5.0, 0.1, 100.0),
            release: EffectParameter::new("release", 50.0, 1.0, 1000.0),
            makeup_gain: EffectParameter::new("makeup_gain", 0.0, -20.0, 20.0),
            knee_width: EffectParameter::new("knee_width", 2.0, 0.0, 10.0),
            
            envelope: 0.0,
            sample_rate,
            attack_coeff: 0.0,
            release_coeff: 0.0,
        };
        
        compressor.update_coefficients();
        compressor
    }
    
    fn update_coefficients(&mut self) {
        let attack_time = self.attack.value / 1000.0; // Convert ms to seconds
        let release_time = self.release.value / 1000.0;
        
        self.attack_coeff = (-1.0 / (attack_time * self.sample_rate as f32)).exp();
        self.release_coeff = (-1.0 / (release_time * self.sample_rate as f32)).exp();
    }
    
    fn linear_to_db(linear: f32) -> f32 {
        20.0 * linear.abs().max(1e-10).log10()
    }
    
    fn db_to_linear(db: f32) -> f32 {
        10.0_f32.powf(db / 20.0)
    }
    
    fn compute_gain_reduction(&self, input_db: f32) -> f32 {
        let threshold = self.threshold.value;
        let ratio = self.ratio.value;
        let knee = self.knee_width.value;
        
        if input_db <= threshold - knee / 2.0 {
            // Below threshold - no compression
            0.0
        } else if input_db >= threshold + knee / 2.0 {
            // Above threshold - full compression
            let over_threshold = input_db - threshold;
            over_threshold * (1.0 - 1.0 / ratio)
        } else {
            // In knee region - soft compression
            let knee_input = input_db - threshold + knee / 2.0;
            let knee_ratio = knee_input / knee;
            let soft_ratio = 1.0 + (ratio - 1.0) * knee_ratio * knee_ratio;
            let over_threshold = input_db - threshold;
            over_threshold * (1.0 - 1.0 / soft_ratio)
        }
    }
    
    pub fn set_threshold(&mut self, threshold_db: f32) {
        self.threshold.set_value(threshold_db);
    }
    
    pub fn set_ratio(&mut self, ratio: f32) {
        self.ratio.set_value(ratio);
    }
    
    pub fn set_attack(&mut self, attack_ms: f32) {
        self.attack.set_value(attack_ms);
        self.update_coefficients();
    }
    
    pub fn set_release(&mut self, release_ms: f32) {
        self.release.set_value(release_ms);
        self.update_coefficients();
    }
}

impl AudioEffect for Compressor {
    fn process(&mut self, audio: &mut AudioBuffer) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }
        
        let samples = audio.samples_mut();
        let makeup_linear = Self::db_to_linear(self.makeup_gain.value);
        
        for sample in samples.iter_mut() {
            let input_db = Self::linear_to_db(*sample);
            
            // Compute target gain reduction
            let target_gain_reduction = self.compute_gain_reduction(input_db);
            let target_envelope = Self::db_to_linear(-target_gain_reduction);
            
            // Apply envelope following
            let coeff = if target_envelope < self.envelope {
                self.attack_coeff
            } else {
                self.release_coeff
            };
            
            self.envelope = target_envelope + (self.envelope - target_envelope) * coeff;
            
            // Apply compression and makeup gain
            *sample *= self.envelope * makeup_linear;
        }
        
        Ok(())
    }
    
    fn name(&self) -> &'static str {
        "Compressor"
    }
    
    fn reset(&mut self) {
        self.envelope = 1.0;
    }
    
    fn is_enabled(&self) -> bool {
        self.enabled
    }
    
    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}

/// Noise gate for background noise reduction
pub struct NoiseGate {
    enabled: bool,
    threshold: EffectParameter,    // Gate threshold in dB
    ratio: EffectParameter,        // Gate ratio (reduction amount)
    attack: EffectParameter,       // Attack time in ms
    release: EffectParameter,      // Release time in ms
    hold: EffectParameter,         // Hold time in ms
    
    // Internal state
    envelope: f32,
    hold_counter: usize,
    sample_rate: u32,
    attack_coeff: f32,
    release_coeff: f32,
}

impl NoiseGate {
    pub fn new(sample_rate: u32) -> Self {
        let mut gate = Self {
            enabled: true,
            threshold: EffectParameter::new("threshold", -40.0, -80.0, 0.0),
            ratio: EffectParameter::new("ratio", 10.0, 1.0, 100.0),
            attack: EffectParameter::new("attack", 1.0, 0.1, 100.0),
            release: EffectParameter::new("release", 100.0, 1.0, 1000.0),
            hold: EffectParameter::new("hold", 10.0, 0.0, 1000.0),
            
            envelope: 1.0,
            hold_counter: 0,
            sample_rate,
            attack_coeff: 0.0,
            release_coeff: 0.0,
        };
        
        gate.update_coefficients();
        gate
    }
    
    fn update_coefficients(&mut self) {
        let attack_time = self.attack.value / 1000.0;
        let release_time = self.release.value / 1000.0;
        
        self.attack_coeff = (-1.0 / (attack_time * self.sample_rate as f32)).exp();
        self.release_coeff = (-1.0 / (release_time * self.sample_rate as f32)).exp();
    }
    
    pub fn set_threshold(&mut self, threshold_db: f32) {
        self.threshold.set_value(threshold_db);
    }
    
    pub fn set_ratio(&mut self, ratio: f32) {
        self.ratio.set_value(ratio);
    }
}

impl AudioEffect for NoiseGate {
    fn process(&mut self, audio: &mut AudioBuffer) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }
        
        let samples = audio.samples_mut();
        let threshold_linear = Compressor::db_to_linear(self.threshold.value);
        let hold_samples = (self.hold.value / 1000.0 * self.sample_rate as f32) as usize;
        
        for sample in samples.iter_mut() {
            let input_level = sample.abs();
            
            // Determine if gate should be open or closed
            let gate_open = input_level > threshold_linear;
            
            let target_gain = if gate_open {
                self.hold_counter = hold_samples;
                1.0
            } else if self.hold_counter > 0 {
                self.hold_counter -= 1;
                1.0
            } else {
                1.0 / self.ratio.value
            };
            
            // Apply envelope following
            let coeff = if target_gain > self.envelope {
                self.attack_coeff
            } else {
                self.release_coeff
            };
            
            self.envelope = target_gain + (self.envelope - target_gain) * coeff;
            
            // Apply gating
            *sample *= self.envelope;
        }
        
        Ok(())
    }
    
    fn name(&self) -> &'static str {
        "NoiseGate"
    }
    
    fn reset(&mut self) {
        self.envelope = 1.0;
        self.hold_counter = 0;
    }
    
    fn is_enabled(&self) -> bool {
        self.enabled
    }
    
    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}

/// Limiter for peak protection
pub struct Limiter {
    enabled: bool,
    threshold: EffectParameter,    // Limiter threshold in dB
    release: EffectParameter,      // Release time in ms
    lookahead: EffectParameter,    // Lookahead time in ms
    
    // Internal state
    envelope: f32,
    delay_buffer: Vec<f32>,
    delay_index: usize,
    sample_rate: u32,
    release_coeff: f32,
}

impl Limiter {
    pub fn new(sample_rate: u32) -> Self {
        let mut limiter = Self {
            enabled: true,
            threshold: EffectParameter::new("threshold", -1.0, -20.0, 0.0),
            release: EffectParameter::new("release", 50.0, 1.0, 1000.0),
            lookahead: EffectParameter::new("lookahead", 5.0, 0.0, 20.0),
            
            envelope: 1.0,
            delay_buffer: Vec::new(),
            delay_index: 0,
            sample_rate,
            release_coeff: 0.0,
        };
        
        limiter.update_coefficients();
        limiter.update_delay_buffer();
        limiter
    }
    
    fn update_coefficients(&mut self) {
        let release_time = self.release.value / 1000.0;
        self.release_coeff = (-1.0 / (release_time * self.sample_rate as f32)).exp();
    }
    
    fn update_delay_buffer(&mut self) {
        let delay_samples = (self.lookahead.value / 1000.0 * self.sample_rate as f32) as usize;
        self.delay_buffer.resize(delay_samples.max(1), 0.0);
        self.delay_index = 0;
    }
    
    pub fn set_threshold(&mut self, threshold_db: f32) {
        self.threshold.set_value(threshold_db);
    }
    
    pub fn set_release(&mut self, release_ms: f32) {
        self.release.set_value(release_ms);
        self.update_coefficients();
    }
}

impl AudioEffect for Limiter {
    fn process(&mut self, audio: &mut AudioBuffer) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }
        
        let samples = audio.samples_mut();
        let threshold_linear = Compressor::db_to_linear(self.threshold.value);
        
        for sample in samples.iter_mut() {
            // Store current sample in delay buffer
            let delayed_sample = self.delay_buffer[self.delay_index];
            self.delay_buffer[self.delay_index] = *sample;
            self.delay_index = (self.delay_index + 1) % self.delay_buffer.len();
            
            // Calculate required gain reduction
            let input_level = sample.abs();
            let target_gain = if input_level > threshold_linear {
                threshold_linear / input_level
            } else {
                1.0
            };
            
            // Apply envelope following (attack is instantaneous)
            self.envelope = if target_gain < self.envelope {
                target_gain // Instant attack
            } else {
                target_gain + (self.envelope - target_gain) * self.release_coeff
            };
            
            // Apply limiting to delayed sample
            *sample = delayed_sample * self.envelope;
        }
        
        Ok(())
    }
    
    fn name(&self) -> &'static str {
        "Limiter"
    }
    
    fn reset(&mut self) {
        self.envelope = 1.0;
        self.delay_buffer.fill(0.0);
        self.delay_index = 0;
    }
    
    fn is_enabled(&self) -> bool {
        self.enabled
    }
    
    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}

/// Automatic Gain Control for maintaining consistent levels
pub struct AutomaticGainControl {
    enabled: bool,
    target_level: EffectParameter,   // Target RMS level in dB
    max_gain: EffectParameter,       // Maximum gain in dB
    attack: EffectParameter,         // Attack time in ms
    release: EffectParameter,        // Release time in ms
    
    // Internal state
    rms_accumulator: f32,
    rms_counter: usize,
    rms_window_size: usize,
    current_gain: f32,
    sample_rate: u32,
    attack_coeff: f32,
    release_coeff: f32,
}

impl AutomaticGainControl {
    pub fn new(sample_rate: u32) -> Self {
        let mut agc = Self {
            enabled: true,
            target_level: EffectParameter::new("target_level", -20.0, -60.0, 0.0),
            max_gain: EffectParameter::new("max_gain", 20.0, 0.0, 40.0),
            attack: EffectParameter::new("attack", 100.0, 10.0, 1000.0),
            release: EffectParameter::new("release", 1000.0, 100.0, 10000.0),
            
            rms_accumulator: 0.0,
            rms_counter: 0,
            rms_window_size: sample_rate as usize / 10, // 100ms window
            current_gain: 1.0,
            sample_rate,
            attack_coeff: 0.0,
            release_coeff: 0.0,
        };
        
        agc.update_coefficients();
        agc
    }
    
    fn update_coefficients(&mut self) {
        let attack_time = self.attack.value / 1000.0;
        let release_time = self.release.value / 1000.0;
        
        self.attack_coeff = (-1.0 / (attack_time * self.sample_rate as f32)).exp();
        self.release_coeff = (-1.0 / (release_time * self.sample_rate as f32)).exp();
    }
    
    pub fn set_target_level(&mut self, level_db: f32) {
        self.target_level.set_value(level_db);
    }
    
    pub fn set_max_gain(&mut self, gain_db: f32) {
        self.max_gain.set_value(gain_db);
    }
}

impl AudioEffect for AutomaticGainControl {
    fn process(&mut self, audio: &mut AudioBuffer) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }
        
        let samples = audio.samples_mut();
        let target_linear = Compressor::db_to_linear(self.target_level.value);
        let max_gain_linear = Compressor::db_to_linear(self.max_gain.value);
        
        for sample in samples.iter_mut() {
            // Accumulate RMS
            self.rms_accumulator += *sample * *sample;
            self.rms_counter += 1;
            
            if self.rms_counter >= self.rms_window_size {
                // Calculate RMS and required gain
                let rms = (self.rms_accumulator / self.rms_counter as f32).sqrt();
                let required_gain = if rms > 1e-10 {
                    (target_linear / rms).min(max_gain_linear)
                } else {
                    max_gain_linear
                };
                
                // Apply envelope following
                let coeff = if required_gain > self.current_gain {
                    self.attack_coeff
                } else {
                    self.release_coeff
                };
                
                self.current_gain = required_gain + (self.current_gain - required_gain) * coeff;
                
                // Reset RMS accumulator
                self.rms_accumulator = 0.0;
                self.rms_counter = 0;
            }
            
            // Apply gain
            *sample *= self.current_gain;
        }
        
        Ok(())
    }
    
    fn name(&self) -> &'static str {
        "AutomaticGainControl"
    }
    
    fn reset(&mut self) {
        self.rms_accumulator = 0.0;
        self.rms_counter = 0;
        self.current_gain = 1.0;
    }
    
    fn is_enabled(&self) -> bool {
        self.enabled
    }
    
    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}
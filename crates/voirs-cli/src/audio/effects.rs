//! Audio effects processing implementation.
//!
//! This module provides professional-grade audio effects for TTS post-processing:
//!
//! ## Basic Effects
//! - **VolumeEffect**: Simple gain adjustment
//! - **LowPassFilter**: Frequency filtering for bandwidth control
//! - **ReverbEffect**: Spatial audio processing with delay-based reverb
//!
//! ## Dynamic Processing
//! - **CompressorEffect**: Dynamic range compression with envelope follower
//!   - Reduces volume differences between loud and quiet parts
//!   - Essential for consistent podcast/broadcast audio
//!   - Configurable threshold, ratio, attack/release times, and makeup gain
//!
//! - **NoiseGateEffect**: Removes low-level noise and background sounds
//!   - Opens/closes based on signal threshold
//!   - Configurable attack, hold, and release times
//!   - Perfect for cleaning up TTS output
//!
//! ## Frequency Shaping
//! - **EqualizerEffect**: 3-band EQ for tonal shaping
//!   - Independent control of low, mid, and high frequencies
//!   - Adjustable crossover frequencies
//!   - ±12dB gain range per band
//!
//! ## Effect Chaining
//! Use `EffectChain` to combine multiple effects in series:
//! ```rust,ignore
//! let mut chain = EffectChain::new();
//! chain.add_effect(Box::new(NoiseGateEffect::new(-45.0, 2.0, 50.0, 100.0)));
//! chain.add_effect(Box::new(CompressorEffect::new(-18.0, 3.0, 5.0, 50.0)));
//! chain.add_effect(Box::new(EqualizerEffect::new(0.0, 2.0, 1.0)));
//! ```
//!
//! ## Presets
//! Available presets via `create_preset_effect_chain()`:
//! - `vocal_enhancement`: Professional vocal processing
//! - `podcast`: Optimized for speech with clarity
//! - `broadcast`: Broadcast-quality compression and EQ
//! - `audiobook`: Natural narration sound
//! - `radio`: Heavy compression for radio-style sound
//! - `warm`: Gentle processing with warm tone
//! - `clean`: Minimal processing, noise gate only
//! - `telephone`: Lo-fi telephone simulation

use super::AudioData;
use std::collections::HashMap;
use voirs_sdk::{Result, VoirsError};

/// Audio effect configuration
#[derive(Debug, Clone)]
pub struct EffectConfig {
    /// Effect type identifier
    pub effect_type: String,
    /// Effect parameters
    pub parameters: HashMap<String, f32>,
    /// Whether the effect is enabled
    pub enabled: bool,
    /// Effect processing order priority
    pub priority: i32,
}

impl EffectConfig {
    /// Create a new effect configuration
    pub fn new(effect_type: &str) -> Self {
        Self {
            effect_type: effect_type.to_string(),
            parameters: HashMap::new(),
            enabled: true,
            priority: 0,
        }
    }

    /// Set a parameter value
    pub fn with_parameter(mut self, name: &str, value: f32) -> Self {
        self.parameters.insert(name.to_string(), value);
        self
    }

    /// Set enabled state
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Set priority
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Get parameter value
    pub fn get_parameter(&self, name: &str) -> Option<f32> {
        self.parameters.get(name).copied()
    }
}

/// Audio effect trait
pub trait AudioEffect: Send + Sync {
    /// Get effect name
    fn name(&self) -> &str;

    /// Process audio samples
    fn process(&mut self, samples: &mut [f32], sample_rate: u32) -> Result<()>;

    /// Reset effect state
    fn reset(&mut self) -> Result<()>;

    /// Check if effect is enabled
    fn is_enabled(&self) -> bool;

    /// Set enabled state
    fn set_enabled(&mut self, enabled: bool);

    /// Get effect parameters
    fn get_parameters(&self) -> HashMap<String, f32>;

    /// Set effect parameter
    fn set_parameter(&mut self, name: &str, value: f32) -> Result<()>;
}

/// Volume effect for adjusting audio level
pub struct VolumeEffect {
    gain: f32,
    enabled: bool,
}

impl VolumeEffect {
    /// Create a new volume effect
    pub fn new(gain: f32) -> Self {
        Self {
            gain: gain.clamp(0.0, 2.0), // Limit to reasonable range
            enabled: true,
        }
    }
}

impl AudioEffect for VolumeEffect {
    fn name(&self) -> &str {
        "volume"
    }

    fn process(&mut self, samples: &mut [f32], _sample_rate: u32) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        for sample in samples.iter_mut() {
            *sample *= self.gain;
        }

        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        // Volume effect has no state to reset
        Ok(())
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    fn get_parameters(&self) -> HashMap<String, f32> {
        let mut params = HashMap::new();
        params.insert("gain".to_string(), self.gain);
        params
    }

    fn set_parameter(&mut self, name: &str, value: f32) -> Result<()> {
        match name {
            "gain" => {
                self.gain = value.clamp(0.0, 2.0);
                Ok(())
            }
            _ => Err(VoirsError::config_error(format!(
                "Unknown parameter: {name}"
            ))),
        }
    }
}

/// Low-pass filter effect
pub struct LowPassFilter {
    cutoff_freq: f32,
    resonance: f32,
    enabled: bool,
    // Filter state
    prev_input: f32,
    prev_output: f32,
}

impl LowPassFilter {
    /// Create a new low-pass filter
    pub fn new(cutoff_freq: f32, resonance: f32) -> Self {
        Self {
            cutoff_freq: cutoff_freq.clamp(20.0, 20000.0),
            resonance: resonance.clamp(0.1, 10.0),
            enabled: true,
            prev_input: 0.0,
            prev_output: 0.0,
        }
    }
}

impl AudioEffect for LowPassFilter {
    fn name(&self) -> &str {
        "lowpass"
    }

    fn process(&mut self, samples: &mut [f32], sample_rate: u32) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        // Simple first-order low-pass filter
        let dt = 1.0 / sample_rate as f32;
        let rc = 1.0 / (2.0 * std::f32::consts::PI * self.cutoff_freq);
        let alpha = dt / (rc + dt);

        for sample in samples.iter_mut() {
            let output = alpha * *sample + (1.0 - alpha) * self.prev_output;
            self.prev_output = output;
            *sample = output;
        }

        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.prev_input = 0.0;
        self.prev_output = 0.0;
        Ok(())
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    fn get_parameters(&self) -> HashMap<String, f32> {
        let mut params = HashMap::new();
        params.insert("cutoff_freq".to_string(), self.cutoff_freq);
        params.insert("resonance".to_string(), self.resonance);
        params
    }

    fn set_parameter(&mut self, name: &str, value: f32) -> Result<()> {
        match name {
            "cutoff_freq" => {
                self.cutoff_freq = value.clamp(20.0, 20000.0);
                Ok(())
            }
            "resonance" => {
                self.resonance = value.clamp(0.1, 10.0);
                Ok(())
            }
            _ => Err(VoirsError::config_error(format!(
                "Unknown parameter: {name}"
            ))),
        }
    }
}

/// Reverb effect for spatial audio processing
pub struct ReverbEffect {
    room_size: f32,
    damping: f32,
    wet_level: f32,
    dry_level: f32,
    enabled: bool,
    // Delay lines for reverb (simplified)
    delay_buffer: Vec<f32>,
    delay_index: usize,
}

impl ReverbEffect {
    /// Create a new reverb effect
    pub fn new(room_size: f32, damping: f32, wet_level: f32) -> Self {
        let delay_samples = (room_size * 22050.0) as usize; // Assume 22kHz for buffer size
        let clamped_wet = wet_level.clamp(0.0, 1.0);

        Self {
            room_size: room_size.clamp(0.0, 1.0),
            damping: damping.clamp(0.0, 1.0),
            wet_level: clamped_wet,
            dry_level: 1.0 - clamped_wet,
            enabled: true,
            delay_buffer: vec![0.0; delay_samples.max(1024)],
            delay_index: 0,
        }
    }
}

impl AudioEffect for ReverbEffect {
    fn name(&self) -> &str {
        "reverb"
    }

    fn process(&mut self, samples: &mut [f32], _sample_rate: u32) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        for sample in samples.iter_mut() {
            // Get delayed sample
            let delayed = self.delay_buffer[self.delay_index];

            // Apply damping to the delayed signal
            let reverb_sample = delayed * (1.0 - self.damping);

            // Mix dry and wet signals
            let output = (*sample * self.dry_level) + (reverb_sample * self.wet_level);

            // Write input + feedback to delay buffer
            self.delay_buffer[self.delay_index] = *sample + (reverb_sample * self.room_size * 0.5);

            // Update delay index
            self.delay_index = (self.delay_index + 1) % self.delay_buffer.len();

            *sample = output;
        }

        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.delay_buffer.fill(0.0);
        self.delay_index = 0;
        Ok(())
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    fn get_parameters(&self) -> HashMap<String, f32> {
        let mut params = HashMap::new();
        params.insert("room_size".to_string(), self.room_size);
        params.insert("damping".to_string(), self.damping);
        params.insert("wet_level".to_string(), self.wet_level);
        params.insert("dry_level".to_string(), self.dry_level);
        params
    }

    fn set_parameter(&mut self, name: &str, value: f32) -> Result<()> {
        match name {
            "room_size" => {
                self.room_size = value.clamp(0.0, 1.0);
                Ok(())
            }
            "damping" => {
                self.damping = value.clamp(0.0, 1.0);
                Ok(())
            }
            "wet_level" => {
                self.wet_level = value.clamp(0.0, 1.0);
                self.dry_level = 1.0 - self.wet_level;
                Ok(())
            }
            _ => Err(VoirsError::config_error(format!(
                "Unknown parameter: {name}"
            ))),
        }
    }
}

/// Compressor effect for dynamic range control
pub struct CompressorEffect {
    threshold: f32,    // dB threshold for compression
    ratio: f32,        // Compression ratio (e.g., 4.0 means 4:1)
    attack_time: f32,  // Attack time in ms
    release_time: f32, // Release time in ms
    makeup_gain: f32,  // Post-compression gain in dB
    enabled: bool,
    envelope: f32, // Current envelope follower value
}

impl CompressorEffect {
    /// Create a new compressor effect
    pub fn new(threshold: f32, ratio: f32, attack_ms: f32, release_ms: f32) -> Self {
        Self {
            threshold: threshold.clamp(-60.0, 0.0),
            ratio: ratio.clamp(1.0, 20.0),
            attack_time: attack_ms.clamp(0.1, 100.0),
            release_time: release_ms.clamp(10.0, 1000.0),
            makeup_gain: 0.0,
            enabled: true,
            envelope: 0.0,
        }
    }

    /// Set makeup gain
    pub fn with_makeup_gain(mut self, gain_db: f32) -> Self {
        self.makeup_gain = gain_db.clamp(0.0, 20.0);
        self
    }

    /// Convert dB to linear amplitude
    fn db_to_linear(db: f32) -> f32 {
        10.0_f32.powf(db / 20.0)
    }

    /// Convert linear amplitude to dB
    fn linear_to_db(linear: f32) -> f32 {
        20.0 * linear.abs().max(1e-10).log10()
    }
}

impl AudioEffect for CompressorEffect {
    fn name(&self) -> &str {
        "compressor"
    }

    fn process(&mut self, samples: &mut [f32], sample_rate: u32) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        let attack_coef = 1.0 - (-1.0 / (sample_rate as f32 * self.attack_time / 1000.0)).exp();
        let release_coef = 1.0 - (-1.0 / (sample_rate as f32 * self.release_time / 1000.0)).exp();

        for sample in samples.iter_mut() {
            let input_level_db = Self::linear_to_db(*sample);

            // Envelope follower
            let coef = if input_level_db > Self::linear_to_db(self.envelope) {
                attack_coef
            } else {
                release_coef
            };
            self.envelope = self.envelope + coef * (sample.abs() - self.envelope);

            // Calculate gain reduction
            let envelope_db = Self::linear_to_db(self.envelope);
            let gain_reduction = if envelope_db > self.threshold {
                (envelope_db - self.threshold) * (1.0 - 1.0 / self.ratio)
            } else {
                0.0
            };

            // Apply compression and makeup gain
            let total_gain_db = -gain_reduction + self.makeup_gain;
            let gain = Self::db_to_linear(total_gain_db);

            *sample *= gain;
        }

        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.envelope = 0.0;
        Ok(())
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    fn get_parameters(&self) -> HashMap<String, f32> {
        let mut params = HashMap::new();
        params.insert("threshold".to_string(), self.threshold);
        params.insert("ratio".to_string(), self.ratio);
        params.insert("attack_time".to_string(), self.attack_time);
        params.insert("release_time".to_string(), self.release_time);
        params.insert("makeup_gain".to_string(), self.makeup_gain);
        params
    }

    fn set_parameter(&mut self, name: &str, value: f32) -> Result<()> {
        match name {
            "threshold" => {
                self.threshold = value.clamp(-60.0, 0.0);
                Ok(())
            }
            "ratio" => {
                self.ratio = value.clamp(1.0, 20.0);
                Ok(())
            }
            "attack_time" => {
                self.attack_time = value.clamp(0.1, 100.0);
                Ok(())
            }
            "release_time" => {
                self.release_time = value.clamp(10.0, 1000.0);
                Ok(())
            }
            "makeup_gain" => {
                self.makeup_gain = value.clamp(0.0, 20.0);
                Ok(())
            }
            _ => Err(VoirsError::config_error(format!(
                "Unknown parameter: {name}"
            ))),
        }
    }
}

/// Noise gate effect for removing low-level noise
pub struct NoiseGateEffect {
    threshold: f32,    // dB threshold below which audio is muted
    attack_time: f32,  // Attack time in ms
    release_time: f32, // Release time in ms
    hold_time: f32,    // Hold time in ms
    enabled: bool,
    envelope: f32,       // Current envelope follower value
    hold_counter: usize, // Samples remaining in hold phase
}

impl NoiseGateEffect {
    /// Create a new noise gate effect
    pub fn new(threshold: f32, attack_ms: f32, hold_ms: f32, release_ms: f32) -> Self {
        Self {
            threshold: threshold.clamp(-80.0, -10.0),
            attack_time: attack_ms.clamp(0.1, 50.0),
            hold_time: hold_ms.clamp(0.0, 500.0),
            release_time: release_ms.clamp(10.0, 1000.0),
            enabled: true,
            envelope: 0.0,
            hold_counter: 0,
        }
    }

    /// Convert dB to linear amplitude
    fn db_to_linear(db: f32) -> f32 {
        10.0_f32.powf(db / 20.0)
    }

    /// Convert linear amplitude to dB
    fn linear_to_db(linear: f32) -> f32 {
        20.0 * linear.abs().max(1e-10).log10()
    }
}

impl AudioEffect for NoiseGateEffect {
    fn name(&self) -> &str {
        "noise_gate"
    }

    fn process(&mut self, samples: &mut [f32], sample_rate: u32) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        let threshold_linear = Self::db_to_linear(self.threshold);
        let attack_coef = 1.0 - (-1.0 / (sample_rate as f32 * self.attack_time / 1000.0)).exp();
        let release_coef = 1.0 - (-1.0 / (sample_rate as f32 * self.release_time / 1000.0)).exp();
        let hold_samples = (sample_rate as f32 * self.hold_time / 1000.0) as usize;

        for sample in samples.iter_mut() {
            let input_level = sample.abs();

            // Update envelope
            if input_level > threshold_linear {
                // Above threshold - open gate quickly
                self.envelope = self.envelope + attack_coef * (1.0 - self.envelope);
                self.hold_counter = hold_samples;
            } else if self.hold_counter > 0 {
                // In hold phase - keep gate open
                self.hold_counter -= 1;
                self.envelope = 1.0;
            } else {
                // Below threshold - close gate slowly
                self.envelope = self.envelope + release_coef * (0.0 - self.envelope);
            }

            // Apply gate
            *sample *= self.envelope;
        }

        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.envelope = 0.0;
        self.hold_counter = 0;
        Ok(())
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    fn get_parameters(&self) -> HashMap<String, f32> {
        let mut params = HashMap::new();
        params.insert("threshold".to_string(), self.threshold);
        params.insert("attack_time".to_string(), self.attack_time);
        params.insert("hold_time".to_string(), self.hold_time);
        params.insert("release_time".to_string(), self.release_time);
        params
    }

    fn set_parameter(&mut self, name: &str, value: f32) -> Result<()> {
        match name {
            "threshold" => {
                self.threshold = value.clamp(-80.0, -10.0);
                Ok(())
            }
            "attack_time" => {
                self.attack_time = value.clamp(0.1, 50.0);
                Ok(())
            }
            "hold_time" => {
                self.hold_time = value.clamp(0.0, 500.0);
                Ok(())
            }
            "release_time" => {
                self.release_time = value.clamp(10.0, 1000.0);
                Ok(())
            }
            _ => Err(VoirsError::config_error(format!(
                "Unknown parameter: {name}"
            ))),
        }
    }
}

/// Three-band equalizer effect for frequency shaping
pub struct EqualizerEffect {
    low_gain: f32,  // Low frequency gain in dB (-12 to +12)
    mid_gain: f32,  // Mid frequency gain in dB
    high_gain: f32, // High frequency gain in dB
    low_freq: f32,  // Low band center frequency (Hz)
    high_freq: f32, // High band center frequency (Hz)
    enabled: bool,
    // Filter state variables
    low_filter_state: [f32; 2],
    high_filter_state: [f32; 2],
}

impl EqualizerEffect {
    /// Create a new 3-band equalizer
    pub fn new(low_gain_db: f32, mid_gain_db: f32, high_gain_db: f32) -> Self {
        Self {
            low_gain: low_gain_db.clamp(-12.0, 12.0),
            mid_gain: mid_gain_db.clamp(-12.0, 12.0),
            high_gain: high_gain_db.clamp(-12.0, 12.0),
            low_freq: 250.0,   // Default low band at 250Hz
            high_freq: 4000.0, // Default high band at 4kHz
            enabled: true,
            low_filter_state: [0.0; 2],
            high_filter_state: [0.0; 2],
        }
    }

    /// Convert dB to linear gain
    fn db_to_linear(db: f32) -> f32 {
        10.0_f32.powf(db / 20.0)
    }
}

impl AudioEffect for EqualizerEffect {
    fn name(&self) -> &str {
        "equalizer"
    }

    fn process(&mut self, samples: &mut [f32], sample_rate: u32) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        let low_gain_linear = Self::db_to_linear(self.low_gain);
        let mid_gain_linear = Self::db_to_linear(self.mid_gain);
        let high_gain_linear = Self::db_to_linear(self.high_gain);

        // Simple first-order filter coefficients
        let low_omega = 2.0 * std::f32::consts::PI * self.low_freq / sample_rate as f32;
        let high_omega = 2.0 * std::f32::consts::PI * self.high_freq / sample_rate as f32;

        let low_alpha = low_omega / (1.0 + low_omega);
        let high_alpha = high_omega / (1.0 + high_omega);

        for sample in samples.iter_mut() {
            let input = *sample;

            // Low-pass filter for low band
            self.low_filter_state[1] =
                self.low_filter_state[1] + low_alpha * (input - self.low_filter_state[1]);
            let low_band = self.low_filter_state[1];

            // High-pass filter for high band
            self.high_filter_state[0] =
                low_alpha * (self.high_filter_state[0] + input - self.high_filter_state[1]);
            self.high_filter_state[1] = input;
            let high_band = self.high_filter_state[0];

            // Mid band is what's left
            let mid_band = input - low_band - high_band;

            // Apply gains and mix
            *sample = (low_band * low_gain_linear)
                + (mid_band * mid_gain_linear)
                + (high_band * high_gain_linear);
        }

        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.low_filter_state = [0.0; 2];
        self.high_filter_state = [0.0; 2];
        Ok(())
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    fn get_parameters(&self) -> HashMap<String, f32> {
        let mut params = HashMap::new();
        params.insert("low_gain".to_string(), self.low_gain);
        params.insert("mid_gain".to_string(), self.mid_gain);
        params.insert("high_gain".to_string(), self.high_gain);
        params.insert("low_freq".to_string(), self.low_freq);
        params.insert("high_freq".to_string(), self.high_freq);
        params
    }

    fn set_parameter(&mut self, name: &str, value: f32) -> Result<()> {
        match name {
            "low_gain" => {
                self.low_gain = value.clamp(-12.0, 12.0);
                Ok(())
            }
            "mid_gain" => {
                self.mid_gain = value.clamp(-12.0, 12.0);
                Ok(())
            }
            "high_gain" => {
                self.high_gain = value.clamp(-12.0, 12.0);
                Ok(())
            }
            "low_freq" => {
                self.low_freq = value.clamp(50.0, 500.0);
                Ok(())
            }
            "high_freq" => {
                self.high_freq = value.clamp(2000.0, 10000.0);
                Ok(())
            }
            _ => Err(VoirsError::config_error(format!(
                "Unknown parameter: {name}"
            ))),
        }
    }
}

/// Effect chain for processing multiple effects in sequence
pub struct EffectChain {
    effects: Vec<Box<dyn AudioEffect>>,
    enabled: bool,
}

impl EffectChain {
    /// Create a new empty effect chain
    pub fn new() -> Self {
        Self {
            effects: Vec::new(),
            enabled: true,
        }
    }

    /// Add an effect to the chain
    pub fn add_effect(&mut self, effect: Box<dyn AudioEffect>) {
        self.effects.push(effect);
    }

    /// Remove an effect by name
    pub fn remove_effect(&mut self, name: &str) -> Result<()> {
        let initial_len = self.effects.len();
        self.effects.retain(|effect| effect.name() != name);

        if self.effects.len() == initial_len {
            return Err(VoirsError::config_error(format!(
                "Effect '{}' not found",
                name
            )));
        }

        Ok(())
    }

    /// Get effect by name
    pub fn get_effect_mut(&mut self, name: &str) -> Option<&mut Box<dyn AudioEffect>> {
        self.effects.iter_mut().find(|effect| effect.name() == name)
    }

    /// Process audio through the entire effect chain
    pub fn process(&mut self, audio_data: &mut AudioData) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        let mut samples: Vec<f32> = audio_data
            .samples
            .iter()
            .map(|&s| s as f32 / i16::MAX as f32)
            .collect();

        // Process through each effect in order
        for effect in &mut self.effects {
            if effect.is_enabled() {
                effect.process(&mut samples, audio_data.sample_rate)?;
            }
        }

        // Convert back to i16
        audio_data.samples = samples
            .iter()
            .map(|&s| (s * i16::MAX as f32) as i16)
            .collect();

        Ok(())
    }

    /// Process raw f32 samples
    pub fn process_samples(&mut self, samples: &mut [f32], sample_rate: u32) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        for effect in &mut self.effects {
            if effect.is_enabled() {
                effect.process(samples, sample_rate)?;
            }
        }

        Ok(())
    }

    /// Reset all effects in the chain
    pub fn reset(&mut self) -> Result<()> {
        for effect in &mut self.effects {
            effect.reset()?;
        }
        Ok(())
    }

    /// Enable or disable the entire effect chain
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if the effect chain is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Get list of effect names in the chain
    pub fn get_effect_names(&self) -> Vec<String> {
        self.effects
            .iter()
            .map(|effect| effect.name().to_string())
            .collect()
    }

    /// Get number of effects in the chain
    pub fn len(&self) -> usize {
        self.effects.len()
    }

    /// Check if the chain is empty
    pub fn is_empty(&self) -> bool {
        self.effects.is_empty()
    }
}

impl Default for EffectChain {
    fn default() -> Self {
        Self::new()
    }
}

/// Create a preset effect chain for common use cases
pub fn create_preset_effect_chain(preset_name: &str) -> Result<EffectChain> {
    let mut chain = EffectChain::new();

    match preset_name {
        "vocal_enhancement" => {
            // Professional vocal enhancement with compression and EQ
            chain.add_effect(Box::new(NoiseGateEffect::new(-45.0, 2.0, 50.0, 100.0)));
            chain.add_effect(Box::new(EqualizerEffect::new(0.0, 2.0, 3.0))); // Boost mids and highs
            chain.add_effect(Box::new(
                CompressorEffect::new(-18.0, 3.0, 5.0, 50.0).with_makeup_gain(4.0),
            ));
            chain.add_effect(Box::new(LowPassFilter::new(8000.0, 0.7)));
            chain.add_effect(Box::new(ReverbEffect::new(0.3, 0.4, 0.2)));
        }
        "podcast" => {
            // Optimized for speech/podcast with professional dynamics
            chain.add_effect(Box::new(NoiseGateEffect::new(-50.0, 3.0, 100.0, 200.0)));
            chain.add_effect(Box::new(EqualizerEffect::new(-1.0, 3.0, 1.0))); // Boost mid-range clarity
            chain.add_effect(Box::new(
                CompressorEffect::new(-20.0, 4.0, 10.0, 100.0).with_makeup_gain(6.0),
            ));
            chain.add_effect(Box::new(LowPassFilter::new(7000.0, 0.8)));
        }
        "radio" => {
            // Radio-style processing with heavy compression
            chain.add_effect(Box::new(NoiseGateEffect::new(-40.0, 1.0, 20.0, 80.0)));
            chain.add_effect(Box::new(EqualizerEffect::new(3.0, 4.0, 2.0))); // Pumped sound
            chain.add_effect(Box::new(
                CompressorEffect::new(-15.0, 6.0, 3.0, 30.0).with_makeup_gain(8.0),
            ));
            chain.add_effect(Box::new(LowPassFilter::new(6000.0, 0.9)));
            chain.add_effect(Box::new(ReverbEffect::new(0.1, 0.8, 0.1)));
        }
        "warm" => {
            // Warm, pleasant sound with gentle processing
            chain.add_effect(Box::new(NoiseGateEffect::new(-55.0, 5.0, 150.0, 300.0)));
            chain.add_effect(Box::new(EqualizerEffect::new(2.0, 0.0, -2.0))); // Warm low-end
            chain.add_effect(Box::new(
                CompressorEffect::new(-24.0, 2.5, 15.0, 150.0).with_makeup_gain(3.0),
            ));
            chain.add_effect(Box::new(LowPassFilter::new(5000.0, 0.6)));
            chain.add_effect(Box::new(ReverbEffect::new(0.4, 0.3, 0.3)));
        }
        "clean" => {
            // Clean, minimal processing with only noise gate
            chain.add_effect(Box::new(NoiseGateEffect::new(-60.0, 5.0, 100.0, 250.0)));
            chain.add_effect(Box::new(VolumeEffect::new(1.0)));
        }
        "broadcast" => {
            // Professional broadcast-quality processing
            chain.add_effect(Box::new(NoiseGateEffect::new(-42.0, 2.0, 75.0, 150.0)));
            chain.add_effect(Box::new(EqualizerEffect::new(0.0, 4.0, 2.0))); // Clear and present
            chain.add_effect(Box::new(
                CompressorEffect::new(-16.0, 5.0, 5.0, 50.0).with_makeup_gain(7.0),
            ));
            chain.add_effect(Box::new(LowPassFilter::new(7500.0, 0.75)));
        }
        "audiobook" => {
            // Optimized for audiobook narration
            chain.add_effect(Box::new(NoiseGateEffect::new(-48.0, 4.0, 120.0, 250.0)));
            chain.add_effect(Box::new(EqualizerEffect::new(1.0, 2.0, 0.0))); // Natural warmth
            chain.add_effect(Box::new(
                CompressorEffect::new(-22.0, 3.5, 12.0, 120.0).with_makeup_gain(5.0),
            ));
            chain.add_effect(Box::new(LowPassFilter::new(6500.0, 0.7)));
        }
        "telephone" => {
            // Telephone/lo-fi sound simulation
            chain.add_effect(Box::new(EqualizerEffect::new(-6.0, 8.0, -10.0))); // Narrow mid-range
            chain.add_effect(Box::new(
                CompressorEffect::new(-12.0, 8.0, 2.0, 20.0).with_makeup_gain(10.0),
            ));
            chain.add_effect(Box::new(LowPassFilter::new(3400.0, 1.2)));
        }
        _ => {
            return Err(VoirsError::config_error(format!(
                "Unknown preset: {}. Available presets: vocal_enhancement, podcast, radio, warm, clean, broadcast, audiobook, telephone",
                preset_name
            )));
        }
    }

    Ok(chain)
}

#[cfg(test)]
mod tests {
    use super::AudioData;
    use super::*;

    #[test]
    fn test_effect_config() {
        let config = EffectConfig::new("volume")
            .with_parameter("gain", 1.5)
            .with_enabled(true)
            .with_priority(10);

        assert_eq!(config.effect_type, "volume");
        assert_eq!(config.get_parameter("gain"), Some(1.5));
        assert!(config.enabled);
        assert_eq!(config.priority, 10);
    }

    #[test]
    fn test_volume_effect() {
        let mut effect = VolumeEffect::new(2.0);
        assert_eq!(effect.name(), "volume");
        assert!(effect.is_enabled());

        let mut samples = vec![0.1, 0.2, 0.3, 0.4];
        effect.process(&mut samples, 22050).unwrap();

        // Volume should be doubled
        assert_eq!(samples, vec![0.2, 0.4, 0.6, 0.8]);

        // Test parameter setting
        effect.set_parameter("gain", 0.5).unwrap();
        let params = effect.get_parameters();
        assert_eq!(params.get("gain"), Some(&0.5));
    }

    #[test]
    fn test_lowpass_filter() {
        let mut filter = LowPassFilter::new(1000.0, 0.7);
        assert_eq!(filter.name(), "lowpass");

        let mut samples = vec![1.0, -1.0, 1.0, -1.0]; // High-frequency signal
        filter.process(&mut samples, 22050).unwrap();

        // High-frequency content should be attenuated
        assert!(samples.iter().all(|&s| s.abs() < 1.0));

        // Test reset
        filter.reset().unwrap();
    }

    #[test]
    fn test_reverb_effect() {
        let mut reverb = ReverbEffect::new(0.1, 0.2, 0.6); // Smaller room, more wet signal
        assert_eq!(reverb.name(), "reverb");

        // Process a longer sequence to allow reverb tail to develop
        let mut samples = vec![1.0; 10]; // Start with impulse followed by zeros
        samples.extend(vec![0.0; 100]); // Add many zeros for reverb tail

        let original_samples = samples.clone();
        reverb.process(&mut samples, 22050).unwrap();

        // After processing, samples should be different due to reverb processing
        assert_ne!(samples, original_samples);

        // The effect should mix dry and wet signal, so the first sample should change
        assert_ne!(samples[0], 1.0);
    }

    #[test]
    fn test_effect_chain() {
        let mut chain = EffectChain::new();
        assert!(chain.is_empty());

        // Add effects
        chain.add_effect(Box::new(VolumeEffect::new(2.0)));
        chain.add_effect(Box::new(LowPassFilter::new(5000.0, 0.7)));

        assert_eq!(chain.len(), 2);
        assert!(!chain.is_empty());

        let effect_names = chain.get_effect_names();
        assert!(effect_names.contains(&"volume".to_string()));
        assert!(effect_names.contains(&"lowpass".to_string()));

        // Test processing
        let mut audio_data = AudioData {
            samples: vec![1000, 2000, 3000, 4000],
            sample_rate: 22050,
            channels: 1,
        };

        chain.process(&mut audio_data).unwrap();

        // Samples should be modified by the effects
        assert_ne!(audio_data.samples, vec![1000, 2000, 3000, 4000]);

        // Test removal
        chain.remove_effect("volume").unwrap();
        assert_eq!(chain.len(), 1);
    }

    #[test]
    fn test_preset_effect_chains() {
        let presets = vec!["vocal_enhancement", "podcast", "radio", "warm", "clean"];

        for preset in presets {
            let chain = create_preset_effect_chain(preset).unwrap();
            assert!(!chain.is_empty());
        }

        // Test unknown preset
        assert!(create_preset_effect_chain("unknown").is_err());
    }

    #[test]
    fn test_effect_chain_enable_disable() {
        let mut chain = EffectChain::new();
        chain.add_effect(Box::new(VolumeEffect::new(2.0)));

        let mut samples = vec![0.1, 0.2, 0.3, 0.4];
        let original_samples = samples.clone();

        // Process with enabled chain
        chain.process_samples(&mut samples, 22050).unwrap();
        assert_ne!(samples, original_samples);

        // Disable chain and process again
        samples = original_samples.clone();
        chain.set_enabled(false);
        chain.process_samples(&mut samples, 22050).unwrap();
        assert_eq!(samples, original_samples); // Should be unchanged
    }
}

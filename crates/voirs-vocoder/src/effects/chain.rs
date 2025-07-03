//! Effect chain management for audio processing pipeline.
//!
//! This module provides a flexible effect chain system that allows
//! combining multiple audio effects in a configurable order.

use crate::{AudioBuffer, Result, VocoderError};
use super::{AudioEffect, EffectParameter};
use std::collections::HashMap;

/// Audio effect chain for processing pipeline
pub struct EffectChain {
    effects: Vec<Box<dyn AudioEffect>>,
    bypass: bool,
    wet_dry_mix: EffectParameter,
    output_gain: EffectParameter,
    
    // Performance monitoring
    processing_time_ms: f32,
    sample_rate: u32,
}

impl EffectChain {
    pub fn new(sample_rate: u32) -> Self {
        Self {
            effects: Vec::new(),
            bypass: false,
            wet_dry_mix: EffectParameter::new("wet_dry_mix", 1.0, 0.0, 1.0),
            output_gain: EffectParameter::new("output_gain", 1.0, 0.0, 2.0),
            processing_time_ms: 0.0,
            sample_rate,
        }
    }
    
    /// Add an effect to the end of the chain
    pub fn add_effect(&mut self, effect: Box<dyn AudioEffect>) {
        self.effects.push(effect);
    }
    
    /// Insert an effect at a specific position in the chain
    pub fn insert_effect(&mut self, index: usize, effect: Box<dyn AudioEffect>) {
        if index <= self.effects.len() {
            self.effects.insert(index, effect);
        }
    }
    
    /// Remove an effect from the chain
    pub fn remove_effect(&mut self, index: usize) -> Option<Box<dyn AudioEffect>> {
        if index < self.effects.len() {
            Some(self.effects.remove(index))
        } else {
            None
        }
    }
    
    /// Get the number of effects in the chain
    pub fn len(&self) -> usize {
        self.effects.len()
    }
    
    /// Check if the chain is empty
    pub fn is_empty(&self) -> bool {
        self.effects.is_empty()
    }
    
    /// Enable or disable a specific effect
    pub fn set_effect_enabled(&mut self, index: usize, enabled: bool) {
        if index < self.effects.len() {
            self.effects[index].set_enabled(enabled);
        }
    }
    
    /// Check if a specific effect is enabled
    pub fn is_effect_enabled(&self, index: usize) -> bool {
        self.effects.get(index).map_or(false, |effect| effect.is_enabled())
    }
    
    /// Get effect name by index
    pub fn get_effect_name(&self, index: usize) -> Option<&'static str> {
        self.effects.get(index).map(|effect| effect.name())
    }
    
    /// Set bypass for entire chain
    pub fn set_bypass(&mut self, bypass: bool) {
        self.bypass = bypass;
    }
    
    /// Get bypass status
    pub fn is_bypassed(&self) -> bool {
        self.bypass
    }
    
    /// Set wet/dry mix (0.0 = dry, 1.0 = wet)
    pub fn set_wet_dry_mix(&mut self, mix: f32) {
        self.wet_dry_mix.set_value(mix);
    }
    
    /// Get wet/dry mix value
    pub fn get_wet_dry_mix(&self) -> f32 {
        self.wet_dry_mix.value
    }
    
    /// Set output gain
    pub fn set_output_gain(&mut self, gain: f32) {
        self.output_gain.set_value(gain);
    }
    
    /// Get output gain
    pub fn get_output_gain(&self) -> f32 {
        self.output_gain.value
    }
    
    /// Get last processing time in milliseconds
    pub fn get_processing_time_ms(&self) -> f32 {
        self.processing_time_ms
    }
    
    /// Reset all effects in the chain
    pub fn reset_all(&mut self) {
        for effect in &mut self.effects {
            effect.reset();
        }
    }
    
    /// Process audio through the effect chain
    pub fn process(&mut self, audio: &mut AudioBuffer) -> Result<()> {
        if self.bypass {
            return Ok(());
        }
        
        let start_time = std::time::Instant::now();
        
        // Store original audio for wet/dry mixing
        let dry_audio = if self.wet_dry_mix.value < 1.0 {
            Some(audio.clone())
        } else {
            None
        };
        
        // Process through effect chain
        for effect in &mut self.effects {
            if effect.is_enabled() {
                effect.process(audio)?;
            }
        }
        
        // Apply wet/dry mix
        if let Some(dry) = dry_audio {
            let wet_level = self.wet_dry_mix.value;
            let dry_level = 1.0 - wet_level;
            
            let wet_samples = audio.samples();
            let dry_samples = dry.samples();
            let mixed_samples: Vec<f32> = wet_samples.iter()
                .zip(dry_samples.iter())
                .map(|(&wet, &dry)| wet * wet_level + dry * dry_level)
                .collect();
            
            *audio = AudioBuffer::new(mixed_samples, audio.sample_rate(), audio.channels());
        }
        
        // Apply output gain
        if (self.output_gain.value - 1.0).abs() > 0.001 {
            let samples = audio.samples_mut();
            for sample in samples {
                *sample *= self.output_gain.value;
            }
        }
        
        // Update processing time
        self.processing_time_ms = start_time.elapsed().as_secs_f32() * 1000.0;
        
        Ok(())
    }
}

/// Preset effect chains for common use cases
pub struct EffectPresets;

impl EffectPresets {
    /// Create a speech enhancement chain
    pub fn speech_enhancement(sample_rate: u32) -> EffectChain {
        let mut chain = EffectChain::new(sample_rate);
        
        // 1. Noise gate to remove background noise
        let mut noise_gate = super::dynamics::NoiseGate::new(sample_rate);
        noise_gate.set_threshold(-45.0);
        chain.add_effect(Box::new(noise_gate));
        
        // 2. De-esser to reduce sibilants
        let mut deesser = super::frequency::Deesser::new(sample_rate);
        deesser.set_threshold(-30.0);
        chain.add_effect(Box::new(deesser));
        
        // 3. Parametric EQ for voice clarity
        let mut eq = super::frequency::ParametricEQ::new(sample_rate);
        eq.set_band_gain(0, -2.0); // Reduce low-end muddiness
        eq.set_band_gain(2, 2.0);  // Boost presence
        chain.add_effect(Box::new(eq));
        
        // 4. Compressor for consistent levels
        let mut compressor = super::dynamics::Compressor::new(sample_rate);
        compressor.set_threshold(-20.0);
        compressor.set_ratio(3.0);
        chain.add_effect(Box::new(compressor));
        
        // 5. Limiter for peak protection
        let mut limiter = super::dynamics::Limiter::new(sample_rate);
        limiter.set_threshold(-3.0);
        chain.add_effect(Box::new(limiter));
        
        chain
    }
    
    /// Create a voice warmth enhancement chain
    pub fn voice_warmth(sample_rate: u32) -> EffectChain {
        let mut chain = EffectChain::new(sample_rate);
        
        // 1. Warmth and presence processor
        let mut warmth = super::frequency::WarmthPresence::new(sample_rate);
        warmth.set_warmth(2.0);
        warmth.set_presence(1.5);
        chain.add_effect(Box::new(warmth));
        
        // 2. Gentle compression
        let mut compressor = super::dynamics::Compressor::new(sample_rate);
        compressor.set_threshold(-25.0);
        compressor.set_ratio(2.5);
        compressor.set_attack(10.0);
        compressor.set_release(100.0);
        chain.add_effect(Box::new(compressor));
        
        // 3. Subtle reverb for ambience
        let mut reverb = super::spatial::Reverb::new(sample_rate);
        reverb.set_room_size(0.3);
        reverb.set_wet_level(0.15);
        reverb.set_dry_level(0.85);
        chain.add_effect(Box::new(reverb));
        
        chain
    }
    
    /// Create a broadcast quality chain
    pub fn broadcast_quality(sample_rate: u32) -> EffectChain {
        let mut chain = EffectChain::new(sample_rate);
        
        // 1. Aggressive noise gate
        let mut noise_gate = super::dynamics::NoiseGate::new(sample_rate);
        noise_gate.set_threshold(-50.0);
        noise_gate.set_ratio(20.0);
        chain.add_effect(Box::new(noise_gate));
        
        // 2. Multi-band processing with EQ
        let mut eq = super::frequency::ParametricEQ::new(sample_rate);
        eq.set_band_gain(0, -3.0); // Cut low-end
        eq.set_band_gain(1, 1.0);  // Slight mid boost
        eq.set_band_gain(2, 3.0);  // Presence boost
        eq.set_band_gain(3, 1.0);  // Slight high boost
        chain.add_effect(Box::new(eq));
        
        // 3. De-esser
        let mut deesser = super::frequency::Deesser::new(sample_rate);
        deesser.set_threshold(-25.0);
        deesser.set_ratio(6.0);
        chain.add_effect(Box::new(deesser));
        
        // 4. Compressor
        let mut compressor = super::dynamics::Compressor::new(sample_rate);
        compressor.set_threshold(-18.0);
        compressor.set_ratio(4.0);
        compressor.set_attack(3.0);
        compressor.set_release(80.0);
        chain.add_effect(Box::new(compressor));
        
        // 5. AGC for consistent levels
        let mut agc = super::dynamics::AutomaticGainControl::new(sample_rate);
        agc.set_target_level(-16.0);
        agc.set_max_gain(12.0);
        chain.add_effect(Box::new(agc));
        
        // 6. Final limiter
        let mut limiter = super::dynamics::Limiter::new(sample_rate);
        limiter.set_threshold(-1.0);
        chain.add_effect(Box::new(limiter));
        
        chain
    }
    
    /// Create a minimal enhancement chain
    pub fn minimal_enhancement(sample_rate: u32) -> EffectChain {
        let mut chain = EffectChain::new(sample_rate);
        
        // 1. Light compression
        let mut compressor = super::dynamics::Compressor::new(sample_rate);
        compressor.set_threshold(-24.0);
        compressor.set_ratio(2.0);
        compressor.set_attack(8.0);
        compressor.set_release(120.0);
        chain.add_effect(Box::new(compressor));
        
        // 2. Peak limiter
        let mut limiter = super::dynamics::Limiter::new(sample_rate);
        limiter.set_threshold(-2.0);
        chain.add_effect(Box::new(limiter));
        
        chain
    }
}

/// Effect chain configuration for serialization
#[derive(Debug, Clone)]
pub struct EffectChainConfig {
    pub effects: Vec<EffectConfig>,
    pub wet_dry_mix: f32,
    pub output_gain: f32,
    pub bypass: bool,
}

#[derive(Debug, Clone)]
pub struct EffectConfig {
    pub effect_type: String,
    pub enabled: bool,
    pub parameters: HashMap<String, f32>,
}

impl EffectChainConfig {
    pub fn new() -> Self {
        Self {
            effects: Vec::new(),
            wet_dry_mix: 1.0,
            output_gain: 1.0,
            bypass: false,
        }
    }
    
    pub fn add_effect(&mut self, effect_type: &str, enabled: bool, parameters: HashMap<String, f32>) {
        self.effects.push(EffectConfig {
            effect_type: effect_type.to_string(),
            enabled,
            parameters,
        });
    }
    
    /// Create an effect chain from this configuration
    pub fn build_chain(&self, sample_rate: u32) -> Result<EffectChain> {
        let mut chain = EffectChain::new(sample_rate);
        
        for effect_config in &self.effects {
            let effect: Box<dyn AudioEffect> = match effect_config.effect_type.as_str() {
                "Compressor" => {
                    let mut comp = super::dynamics::Compressor::new(sample_rate);
                    if let Some(&threshold) = effect_config.parameters.get("threshold") {
                        comp.set_threshold(threshold);
                    }
                    if let Some(&ratio) = effect_config.parameters.get("ratio") {
                        comp.set_ratio(ratio);
                    }
                    if let Some(&attack) = effect_config.parameters.get("attack") {
                        comp.set_attack(attack);
                    }
                    if let Some(&release) = effect_config.parameters.get("release") {
                        comp.set_release(release);
                    }
                    comp.set_enabled(effect_config.enabled);
                    Box::new(comp)
                }
                "NoiseGate" => {
                    let mut gate = super::dynamics::NoiseGate::new(sample_rate);
                    if let Some(&threshold) = effect_config.parameters.get("threshold") {
                        gate.set_threshold(threshold);
                    }
                    if let Some(&ratio) = effect_config.parameters.get("ratio") {
                        gate.set_ratio(ratio);
                    }
                    gate.set_enabled(effect_config.enabled);
                    Box::new(gate)
                }
                "Limiter" => {
                    let mut limiter = super::dynamics::Limiter::new(sample_rate);
                    if let Some(&threshold) = effect_config.parameters.get("threshold") {
                        limiter.set_threshold(threshold);
                    }
                    if let Some(&release) = effect_config.parameters.get("release") {
                        limiter.set_release(release);
                    }
                    limiter.set_enabled(effect_config.enabled);
                    Box::new(limiter)
                }
                "ParametricEQ" => {
                    let mut eq = super::frequency::ParametricEQ::new(sample_rate);
                    eq.set_enabled(effect_config.enabled);
                    Box::new(eq)
                }
                "Deesser" => {
                    let mut deesser = super::frequency::Deesser::new(sample_rate);
                    if let Some(&threshold) = effect_config.parameters.get("threshold") {
                        deesser.set_threshold(threshold);
                    }
                    if let Some(&frequency) = effect_config.parameters.get("frequency") {
                        deesser.set_frequency(frequency);
                    }
                    deesser.set_enabled(effect_config.enabled);
                    Box::new(deesser)
                }
                "Reverb" => {
                    let mut reverb = super::spatial::Reverb::new(sample_rate);
                    if let Some(&room_size) = effect_config.parameters.get("room_size") {
                        reverb.set_room_size(room_size);
                    }
                    if let Some(&wet_level) = effect_config.parameters.get("wet_level") {
                        reverb.set_wet_level(wet_level);
                    }
                    if let Some(&dry_level) = effect_config.parameters.get("dry_level") {
                        reverb.set_dry_level(dry_level);
                    }
                    reverb.set_enabled(effect_config.enabled);
                    Box::new(reverb)
                }
                _ => return Err(VocoderError::ConfigError(format!("Unknown effect type: {}", effect_config.effect_type))),
            };
            
            chain.add_effect(effect);
        }
        
        chain.set_wet_dry_mix(self.wet_dry_mix);
        chain.set_output_gain(self.output_gain);
        chain.set_bypass(self.bypass);
        
        Ok(chain)
    }
}

impl Default for EffectChainConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AudioBuffer;
    
    #[test]
    fn test_effect_chain_creation() {
        let mut chain = EffectChain::new(44100);
        assert_eq!(chain.len(), 0);
        assert!(chain.is_empty());
        
        let compressor = super::super::dynamics::Compressor::new(44100);
        chain.add_effect(Box::new(compressor));
        
        assert_eq!(chain.len(), 1);
        assert!(!chain.is_empty());
        assert_eq!(chain.get_effect_name(0), Some("Compressor"));
    }
    
    #[test]
    fn test_preset_chains() {
        let speech_chain = EffectPresets::speech_enhancement(44100);
        assert!(speech_chain.len() > 0);
        
        let warmth_chain = EffectPresets::voice_warmth(44100);
        assert!(warmth_chain.len() > 0);
        
        let broadcast_chain = EffectPresets::broadcast_quality(44100);
        assert!(broadcast_chain.len() > 0);
    }
    
    #[test]
    fn test_effect_chain_processing() {
        let mut chain = EffectPresets::minimal_enhancement(44100);
        
        // Create test audio
        let samples = vec![0.1, -0.1, 0.2, -0.2, 0.3, -0.3];
        let mut audio = AudioBuffer::new(samples, 44100, 1);
        
        // Process audio
        let result = chain.process(&mut audio);
        assert!(result.is_ok());
        
        // Check that processing time was recorded
        assert!(chain.get_processing_time_ms() >= 0.0);
    }
}
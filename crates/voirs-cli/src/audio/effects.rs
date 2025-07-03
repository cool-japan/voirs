//! Audio effects processing implementation.

use std::collections::HashMap;
use voirs::error::{Result, VoirsError};
use super::AudioData;

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
            gain: gain.max(0.0).min(2.0), // Limit to reasonable range
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
                self.gain = value.max(0.0).min(2.0);
                Ok(())
            }
            _ => Err(VoirsError::config_error(format!("Unknown parameter: {}", name))),
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
            cutoff_freq: cutoff_freq.max(20.0).min(20000.0),
            resonance: resonance.max(0.1).min(10.0),
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
                self.cutoff_freq = value.max(20.0).min(20000.0);
                Ok(())
            }
            "resonance" => {
                self.resonance = value.max(0.1).min(10.0);
                Ok(())
            }
            _ => Err(VoirsError::config_error(format!("Unknown parameter: {}", name))),
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
        
        Self {
            room_size: room_size.max(0.0).min(1.0),
            damping: damping.max(0.0).min(1.0),
            wet_level: wet_level.max(0.0).min(1.0),
            dry_level: 1.0 - wet_level.max(0.0).min(1.0),
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
                self.room_size = value.max(0.0).min(1.0);
                Ok(())
            }
            "damping" => {
                self.damping = value.max(0.0).min(1.0);
                Ok(())
            }
            "wet_level" => {
                self.wet_level = value.max(0.0).min(1.0);
                self.dry_level = 1.0 - self.wet_level;
                Ok(())
            }
            _ => Err(VoirsError::config_error(format!("Unknown parameter: {}", name))),
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
            return Err(VoirsError::config_error(format!("Effect '{}' not found", name)));
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
        
        let mut samples: Vec<f32> = audio_data.samples.iter()
            .map(|&s| s as f32 / i16::MAX as f32)
            .collect();
        
        // Process through each effect in order
        for effect in &mut self.effects {
            if effect.is_enabled() {
                effect.process(&mut samples, audio_data.sample_rate)?;
            }
        }
        
        // Convert back to i16
        audio_data.samples = samples.iter()
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
        self.effects.iter().map(|effect| effect.name().to_string()).collect()
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
            // Add effects for vocal enhancement
            chain.add_effect(Box::new(VolumeEffect::new(1.2)));
            chain.add_effect(Box::new(LowPassFilter::new(8000.0, 0.7)));
            chain.add_effect(Box::new(ReverbEffect::new(0.3, 0.4, 0.2)));
        }
        "podcast" => {
            // Optimized for speech/podcast
            chain.add_effect(Box::new(VolumeEffect::new(1.1)));
            chain.add_effect(Box::new(LowPassFilter::new(7000.0, 0.8)));
        }
        "radio" => {
            // Radio-style processing
            chain.add_effect(Box::new(VolumeEffect::new(1.3)));
            chain.add_effect(Box::new(LowPassFilter::new(6000.0, 0.9)));
            chain.add_effect(Box::new(ReverbEffect::new(0.1, 0.8, 0.1)));
        }
        "warm" => {
            // Warm, pleasant sound
            chain.add_effect(Box::new(VolumeEffect::new(1.0)));
            chain.add_effect(Box::new(LowPassFilter::new(5000.0, 0.6)));
            chain.add_effect(Box::new(ReverbEffect::new(0.4, 0.3, 0.3)));
        }
        "clean" => {
            // Clean, minimal processing
            chain.add_effect(Box::new(VolumeEffect::new(1.0)));
        }
        _ => {
            return Err(VoirsError::config_error(format!("Unknown preset: {}", preset_name)));
        }
    }
    
    Ok(chain)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::AudioData;
    
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
//! Core effect traits and infrastructure

use std::collections::HashMap;

/// Effect processor trait for singing synthesis audio effects.
///
/// This trait defines the interface for all audio effects that can be applied to singing voices.
/// Effects implementing this trait can be chained together and managed through the `EffectChain`.
pub trait SingingEffect: Send + Sync {
    /// Returns the name identifier of this effect.
    ///
    /// # Returns
    ///
    /// A string slice representing the effect's name (e.g., "reverb", "compressor").
    fn name(&self) -> &str;

    /// Processes audio samples in-place, applying the effect.
    ///
    /// # Arguments
    ///
    /// * `audio` - Mutable slice of audio samples to process (normalized to -1.0 to 1.0 range)
    /// * `sample_rate` - Sample rate in Hz (e.g., 44100.0, 48000.0)
    ///
    /// # Returns
    ///
    /// `Result<()>` indicating success or an error if processing fails.
    ///
    /// # Errors
    ///
    /// Returns an error if audio processing encounters invalid parameters or internal failures.
    fn process(&mut self, audio: &mut [f32], sample_rate: f32) -> crate::Result<()>;

    /// Sets a parameter value for this effect.
    ///
    /// # Arguments
    ///
    /// * `name` - Parameter name (e.g., "threshold", "attack", "frequency")
    /// * `value` - Parameter value (range and units depend on the specific parameter)
    ///
    /// # Returns
    ///
    /// `Result<()>` indicating success or an error if the parameter is invalid.
    ///
    /// # Errors
    ///
    /// Returns an error if the parameter name is unknown or the value is out of valid range.
    fn set_parameter(&mut self, name: &str, value: f32) -> crate::Result<()>;

    /// Gets the current value of a parameter.
    ///
    /// # Arguments
    ///
    /// * `name` - Parameter name to query
    ///
    /// # Returns
    ///
    /// `Some(f32)` with the parameter value if it exists, or `None` if the parameter is unknown.
    fn get_parameter(&self, name: &str) -> Option<f32>;

    /// Returns a map of all parameters and their current values.
    ///
    /// # Returns
    ///
    /// A `HashMap` containing all parameter names as keys and their values.
    fn get_parameters(&self) -> HashMap<String, f32>;

    /// Resets the effect's internal state to initial conditions.
    ///
    /// This clears any buffers, delays, or accumulated state while preserving parameter settings.
    fn reset(&mut self);

    /// Creates a boxed clone of this effect.
    ///
    /// # Returns
    ///
    /// A new boxed instance of this effect with the same configuration.
    fn clone_effect(&self) -> Box<dyn SingingEffect>;
}

/// Effect chain for processing audio through multiple effects in sequence.
///
/// The `EffectChain` manages a series of effects that are applied sequentially to audio,
/// with support for dry/wet mixing, bypass, and enable/disable functionality.
pub struct EffectChain {
    /// Ordered vector of effects to apply sequentially
    effects: Vec<Box<dyn SingingEffect>>,
    /// Whether the effect chain is enabled
    enabled: bool,
    /// Dry/wet mix ratio (0.0 = fully dry, 1.0 = fully wet)
    dry_wet_mix: f32,
    /// Whether to bypass all processing (pass audio through unchanged)
    bypass: bool,
}

/// Effect processor wrapper providing parameter management and processing control.
///
/// `EffectProcessor` wraps a single effect with additional control features like
/// parameter caching, enable/disable, and bypass functionality.
pub struct EffectProcessor {
    /// The wrapped effect instance
    effect: Box<dyn SingingEffect>,
    /// Cached parameter values for quick access
    parameters: HashMap<String, f32>,
    /// Whether the processor is enabled
    enabled: bool,
    /// Whether to bypass processing (pass audio through unchanged)
    bypass: bool,
}

impl EffectChain {
    /// Create a new effect chain
    pub fn new() -> Self {
        Self {
            effects: Vec::new(),
            enabled: true,
            dry_wet_mix: 1.0,
            bypass: false,
        }
    }

    /// Create effect chain with default singing effects
    pub fn with_defaults() -> Self {
        let mut chain = Self::new();

        // Add default effects
        let default_reverb = crate::effects::time_effects::ReverbEffect::new(HashMap::new());
        let default_compressor =
            crate::effects::dynamic_effects::CompressorEffect::new(HashMap::new());

        chain.effects.push(Box::new(default_reverb));
        chain.effects.push(Box::new(default_compressor));

        chain
    }

    /// Add an effect to the chain
    pub async fn add_effect(
        &mut self,
        name: &str,
        parameters: HashMap<String, f32>,
    ) -> crate::Result<()> {
        let effect = self.create_effect_by_name(name, parameters)?;
        self.effects.push(effect);
        Ok(())
    }

    /// Synchronous version of add_effect for convenience
    pub fn add_effect_blocking(
        &mut self,
        name: &str,
        parameters: HashMap<String, f32>,
    ) -> crate::Result<()> {
        let effect = self.create_effect_by_name(name, parameters)?;
        self.effects.push(effect);
        Ok(())
    }

    /// Process audio through the effect chain
    pub async fn process(
        &mut self,
        mut audio: Vec<f32>,
        sample_rate: f32,
    ) -> crate::Result<Vec<f32>> {
        if self.bypass || !self.enabled {
            return Ok(audio);
        }

        let dry_audio = if self.dry_wet_mix < 1.0 {
            audio.clone()
        } else {
            Vec::new()
        };

        for effect in &mut self.effects {
            effect.process(&mut audio, sample_rate)?;
        }

        // Apply dry/wet mix
        if !dry_audio.is_empty() {
            for (i, sample) in audio.iter_mut().enumerate() {
                if i < dry_audio.len() {
                    *sample = dry_audio[i] * (1.0 - self.dry_wet_mix) + *sample * self.dry_wet_mix;
                }
            }
        }

        Ok(audio)
    }

    /// Set parameter for specific effect
    pub fn set_effect_parameter(
        &mut self,
        effect_index: usize,
        name: &str,
        value: f32,
    ) -> crate::Result<()> {
        if effect_index < self.effects.len() {
            self.effects[effect_index].set_parameter(name, value)?;
        } else {
            return Err(crate::Error::Effect(format!(
                "Effect index {} out of range",
                effect_index
            )));
        }
        Ok(())
    }

    /// Get parameter from specific effect
    pub fn get_effect_parameter(&self, effect_index: usize, name: &str) -> Option<f32> {
        if effect_index < self.effects.len() {
            self.effects[effect_index].get_parameter(name)
        } else {
            None
        }
    }

    /// Set dry/wet mix for the entire chain
    pub fn set_dry_wet_mix(&mut self, mix: f32) {
        self.dry_wet_mix = mix.clamp(0.0, 1.0);
    }

    /// Enable or disable the effect chain
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Bypass the effect chain
    pub fn set_bypass(&mut self, bypass: bool) {
        self.bypass = bypass;
    }

    /// Get list of effect names
    pub fn effect_names(&self) -> Vec<String> {
        self.effects.iter().map(|e| e.name().to_string()).collect()
    }

    /// Remove an effect by name
    pub async fn remove_effect(&mut self, name: &str) -> crate::Result<()> {
        self.effects.retain(|effect| effect.name() != name);
        Ok(())
    }

    /// Synchronous version of remove_effect for convenience  
    pub fn remove_effect_blocking(&mut self, name: &str) -> crate::Result<()> {
        self.effects.retain(|effect| effect.name() != name);
        Ok(())
    }

    /// Clear all effects
    pub fn clear(&mut self) {
        self.effects.clear();
    }

    /// Create effect by name
    fn create_effect_by_name(
        &self,
        name: &str,
        parameters: HashMap<String, f32>,
    ) -> crate::Result<Box<dyn SingingEffect>> {
        match name {
            "reverb" => Ok(Box::new(crate::effects::time_effects::ReverbEffect::new(
                parameters,
            ))),
            "chorus" => Ok(Box::new(crate::effects::time_effects::ChorusEffect::new(
                parameters,
            ))),
            "vibrato" => Ok(Box::new(crate::effects::time_effects::VibratoEffect::new(
                parameters,
            ))),
            "compressor" => Ok(Box::new(
                crate::effects::dynamic_effects::CompressorEffect::new(parameters),
            )),
            "eq" | "equalizer" => Ok(Box::new(crate::effects::spectral_effects::EQEffect::new(
                parameters,
            ))),
            "formant" => Ok(Box::new(
                crate::effects::spectral_effects::FormantControlEffect::new(parameters),
            )),
            "spectral_morph" | "morph" => Ok(Box::new(
                crate::effects::spectral_effects::SpectralMorphingEffect::new(parameters),
            )),
            "breath_noise" => Ok(Box::new(
                crate::effects::synthesis_effects::BreathNoiseEffect::new(parameters),
            )),
            _ => Err(crate::Error::Effect(format!("Unknown effect: {name}"))),
        }
    }

    /// Reset all effects
    pub fn reset(&mut self) {
        for effect in &mut self.effects {
            effect.reset();
        }
    }
}

impl Default for EffectChain {
    fn default() -> Self {
        Self::new()
    }
}

impl EffectProcessor {
    /// Create a new effect processor
    pub fn new(effect: Box<dyn SingingEffect>) -> Self {
        Self {
            parameters: effect.get_parameters(),
            effect,
            enabled: true,
            bypass: false,
        }
    }

    /// Process audio
    pub fn process(&mut self, audio: &mut [f32], sample_rate: f32) -> crate::Result<()> {
        if !self.bypass && self.enabled {
            self.effect.process(audio, sample_rate)?;
        }
        Ok(())
    }

    /// Set parameter
    pub fn set_parameter(&mut self, name: &str, value: f32) -> crate::Result<()> {
        self.effect.set_parameter(name, value)?;
        self.parameters.insert(name.to_string(), value);
        Ok(())
    }

    /// Get parameter
    pub fn get_parameter(&self, name: &str) -> Option<f32> {
        self.parameters.get(name).copied()
    }

    /// Enable or disable the processor
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Bypass the processor
    pub fn set_bypass(&mut self, bypass: bool) {
        self.bypass = bypass;
    }

    /// Reset the effect
    pub fn reset(&mut self) {
        self.effect.reset();
    }
}

//! Prosody control for natural speech synthesis.

pub mod duration;
pub mod energy;
pub mod pitch;

pub use duration::{DurationConfig, DurationContext, PauseDurations, RhythmPattern};
pub use energy::{EnergyConfig, EnergyContext, EnergyContourPattern, VoiceQualityConfig};
pub use pitch::{IntonationPattern, PitchConfig, PitchContext, VibratoConfig};

use crate::speaker::emotion::{EmotionConfig, EmotionIntensity, EmotionType};
use crate::Result;
use serde::{Deserialize, Serialize};

/// Comprehensive prosody configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProsodyConfig {
    /// Duration control settings
    pub duration: DurationConfig,
    /// Pitch control settings
    pub pitch: PitchConfig,
    /// Energy control settings
    pub energy: EnergyConfig,
    /// Global prosody strength (0.0 - 1.0)
    pub strength: f32,
    /// Natural variation amount (0.0 - 1.0)
    pub variation: f32,
}

impl ProsodyConfig {
    /// Create new prosody config
    pub fn new() -> Self {
        Self {
            duration: DurationConfig::default(),
            pitch: PitchConfig::default(),
            energy: EnergyConfig::default(),
            strength: 1.0,
            variation: 0.1,
        }
    }

    /// Set duration config
    pub fn with_duration(mut self, duration: DurationConfig) -> Self {
        self.duration = duration;
        self
    }

    /// Set pitch config
    pub fn with_pitch(mut self, pitch: PitchConfig) -> Self {
        self.pitch = pitch;
        self
    }

    /// Set energy config
    pub fn with_energy(mut self, energy: EnergyConfig) -> Self {
        self.energy = energy;
        self
    }

    /// Set prosody strength
    pub fn with_strength(mut self, strength: f32) -> Self {
        self.strength = strength.clamp(0.0, 1.0);
        self
    }

    /// Set natural variation
    pub fn with_variation(mut self, variation: f32) -> Self {
        self.variation = variation.clamp(0.0, 1.0);
        self
    }

    /// Apply prosody adjustments to phoneme sequence
    pub fn apply_to_phonemes(&self, phonemes: &mut [crate::Phoneme]) -> Result<()> {
        // Apply duration adjustments
        self.duration.apply_to_phonemes(phonemes)?;

        // Note: Pitch and energy are typically applied during synthesis
        // as they affect the acoustic features rather than phoneme timing

        Ok(())
    }

    /// Validate prosody configuration
    pub fn validate(&self) -> Result<()> {
        self.duration.validate()?;
        self.pitch.validate()?;
        self.energy.validate()?;

        if !(0.0..=1.0).contains(&self.strength) {
            return Err(crate::AcousticError::ConfigError(format!(
                "Prosody strength must be between 0.0 and 1.0, got {}",
                self.strength
            )));
        }

        if !(0.0..=1.0).contains(&self.variation) {
            return Err(crate::AcousticError::ConfigError(format!(
                "Prosody variation must be between 0.0 and 1.0, got {}",
                self.variation
            )));
        }

        Ok(())
    }

    /// Create preset prosody configurations
    pub fn preset_natural() -> Self {
        Self::new()
            .with_duration(DurationConfig::natural())
            .with_pitch(PitchConfig::natural())
            .with_energy(EnergyConfig::natural())
            .with_variation(0.15)
    }

    pub fn preset_expressive() -> Self {
        Self::new()
            .with_duration(DurationConfig::expressive())
            .with_pitch(PitchConfig::expressive())
            .with_energy(EnergyConfig::expressive())
            .with_variation(0.25)
    }

    pub fn preset_monotone() -> Self {
        Self::new()
            .with_duration(DurationConfig::uniform())
            .with_pitch(PitchConfig::flat())
            .with_energy(EnergyConfig::uniform())
            .with_variation(0.05)
    }

    pub fn preset_fast() -> Self {
        Self::new()
            .with_duration(DurationConfig::fast())
            .with_pitch(PitchConfig::default())
            .with_energy(EnergyConfig::default())
    }

    pub fn preset_slow() -> Self {
        Self::new()
            .with_duration(DurationConfig::slow())
            .with_pitch(PitchConfig::default())
            .with_energy(EnergyConfig::default())
    }
}

impl Default for ProsodyConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Prosody controller for managing dynamic adjustments
#[derive(Debug, Clone)]
pub struct ProsodyController {
    /// Base prosody configuration
    base_config: ProsodyConfig,
    /// Current prosody state
    current_config: ProsodyConfig,
    /// Adjustment history
    adjustment_history: Vec<ProsodyConfig>,
    /// Maximum history length
    max_history: usize,
}

impl ProsodyController {
    /// Create new prosody controller
    pub fn new(base_config: ProsodyConfig) -> Self {
        Self {
            base_config: base_config.clone(),
            current_config: base_config,
            adjustment_history: Vec::new(),
            max_history: 20,
        }
    }

    /// Get current prosody configuration
    pub fn get_current_config(&self) -> &ProsodyConfig {
        &self.current_config
    }

    /// Set new prosody configuration
    pub fn set_config(&mut self, config: ProsodyConfig) {
        self.adjustment_history.push(self.current_config.clone());

        // Trim history if needed
        if self.adjustment_history.len() > self.max_history {
            self.adjustment_history.remove(0);
        }

        self.current_config = config;
    }

    /// Apply temporary adjustment
    pub fn apply_adjustment(&mut self, adjustment: ProsodyAdjustment) {
        let mut adjusted_config = self.current_config.clone();

        match adjustment {
            ProsodyAdjustment::Speed(factor) => {
                adjusted_config.duration.speed_factor *= factor;
            }
            ProsodyAdjustment::Pitch(shift) => {
                adjusted_config.pitch.base_frequency += shift;
            }
            ProsodyAdjustment::Energy(factor) => {
                adjusted_config.energy.base_energy *= factor;
            }
            ProsodyAdjustment::Variation(amount) => {
                adjusted_config.variation = (adjusted_config.variation + amount).clamp(0.0, 1.0);
            }
        }

        self.set_config(adjusted_config);
    }

    /// Reset to base configuration
    pub fn reset_to_base(&mut self) {
        self.set_config(self.base_config.clone());
    }

    /// Get adjustment history
    pub fn get_adjustment_history(&self) -> &[ProsodyConfig] {
        &self.adjustment_history
    }
}

/// Prosody adjustment types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProsodyAdjustment {
    /// Speed adjustment (factor)
    Speed(f32),
    /// Pitch adjustment (Hz shift)
    Pitch(f32),
    /// Energy adjustment (factor)
    Energy(f32),
    /// Variation adjustment (amount)
    Variation(f32),
}

impl Default for ProsodyController {
    fn default() -> Self {
        Self::new(ProsodyConfig::default())
    }
}

/// Emotion-aware prosody modifier
#[derive(Debug, Clone)]
pub struct EmotionProsodyModifier {
    /// Base prosody configuration
    base_config: ProsodyConfig,
    /// Current emotion configuration
    current_emotion: Option<EmotionConfig>,
    /// Emotion intensity scaling factor
    emotion_intensity_scale: f32,
    /// Whether to apply smooth transitions
    smooth_transitions: bool,
}

impl EmotionProsodyModifier {
    /// Create new emotion-aware prosody modifier
    pub fn new(base_config: ProsodyConfig) -> Self {
        Self {
            base_config,
            current_emotion: None,
            emotion_intensity_scale: 1.0,
            smooth_transitions: true,
        }
    }

    /// Set emotion intensity scaling factor
    pub fn with_intensity_scale(mut self, scale: f32) -> Self {
        self.emotion_intensity_scale = scale.clamp(0.0, 2.0);
        self
    }

    /// Enable or disable smooth transitions
    pub fn with_smooth_transitions(mut self, enabled: bool) -> Self {
        self.smooth_transitions = enabled;
        self
    }

    /// Apply emotion to prosody configuration
    pub fn apply_emotion(&mut self, emotion: EmotionConfig) -> Result<ProsodyConfig> {
        self.current_emotion = Some(emotion.clone());
        self.generate_emotion_prosody(&emotion)
    }

    /// Generate emotion-specific prosody configuration
    fn generate_emotion_prosody(&self, emotion: &EmotionConfig) -> Result<ProsodyConfig> {
        let mut config = self.base_config.clone();

        // Apply emotion-specific modifications
        self.apply_emotion_to_duration(&mut config, emotion)?;
        self.apply_emotion_to_pitch(&mut config, emotion)?;
        self.apply_emotion_to_energy(&mut config, emotion)?;
        self.apply_emotion_to_variation(&mut config, emotion)?;

        // Apply custom parameters if available
        self.apply_custom_parameters(&mut config, emotion)?;

        // Validate final configuration
        config.validate()?;

        Ok(config)
    }

    /// Apply emotion-specific duration modifications
    fn apply_emotion_to_duration(
        &self,
        config: &mut ProsodyConfig,
        emotion: &EmotionConfig,
    ) -> Result<()> {
        let intensity = emotion.intensity.as_f32() * self.emotion_intensity_scale;

        // Base speed adjustments by emotion type
        let speed_factor = match emotion.emotion_type {
            EmotionType::Happy => 1.0 + (0.15 * intensity),
            EmotionType::Excited => 1.0 + (0.25 * intensity),
            EmotionType::Angry => 1.0 + (0.20 * intensity),
            EmotionType::Fear => 1.0 + (0.30 * intensity),
            EmotionType::Surprise => 1.0 + (0.25 * intensity),
            EmotionType::Sad => 1.0 - (0.20 * intensity),
            EmotionType::Calm => 1.0 - (0.15 * intensity),
            EmotionType::Disgust => 1.0 - (0.10 * intensity),
            EmotionType::Love => 1.0 - (0.05 * intensity),
            EmotionType::Neutral => 1.0,
            EmotionType::Custom(_) => 1.0,
        };

        // Apply speed factor
        config.duration.speed_factor *= speed_factor;

        // Adjust pause durations based on emotion
        let pause_factor = match emotion.emotion_type {
            EmotionType::Happy | EmotionType::Excited => 1.0 - (0.1 * intensity), // Shorter pauses
            EmotionType::Angry => 1.0 - (0.15 * intensity),                       // Shorter pauses
            EmotionType::Sad | EmotionType::Calm => 1.0 + (0.2 * intensity),      // Longer pauses
            EmotionType::Fear => 1.0 + (0.1 * intensity), // Slightly longer pauses
            _ => 1.0,
        };

        // Apply pause adjustments
        config.duration.pause_durations.comma_ms *= pause_factor;
        config.duration.pause_durations.period_ms *= pause_factor;
        config.duration.pause_durations.sentence_ms *= pause_factor;
        config.duration.pause_durations.paragraph_ms *= pause_factor;

        Ok(())
    }

    /// Apply emotion-specific pitch modifications
    fn apply_emotion_to_pitch(
        &self,
        config: &mut ProsodyConfig,
        emotion: &EmotionConfig,
    ) -> Result<()> {
        let intensity = emotion.intensity.as_f32() * self.emotion_intensity_scale;

        // Base frequency adjustments by emotion type
        let frequency_shift = match emotion.emotion_type {
            EmotionType::Happy => 20.0 * intensity,
            EmotionType::Excited => 30.0 * intensity,
            EmotionType::Surprise => 40.0 * intensity,
            EmotionType::Fear => 25.0 * intensity,
            EmotionType::Angry => 15.0 * intensity,
            EmotionType::Sad => -15.0 * intensity,
            EmotionType::Calm => -10.0 * intensity,
            EmotionType::Love => 5.0 * intensity,
            EmotionType::Disgust => -5.0 * intensity,
            EmotionType::Neutral => 0.0,
            EmotionType::Custom(_) => 0.0,
        };

        // Apply frequency shift
        config.pitch.base_frequency += frequency_shift;

        // Adjust pitch range based on emotion
        let range_factor = match emotion.emotion_type {
            EmotionType::Happy | EmotionType::Excited => 1.0 + (0.3 * intensity),
            EmotionType::Surprise => 1.0 + (0.4 * intensity),
            EmotionType::Angry => 1.0 + (0.2 * intensity),
            EmotionType::Fear => 1.0 + (0.35 * intensity),
            EmotionType::Sad => 1.0 - (0.2 * intensity),
            EmotionType::Calm => 1.0 - (0.25 * intensity),
            EmotionType::Love => 1.0 + (0.1 * intensity),
            EmotionType::Disgust => 1.0 - (0.1 * intensity),
            EmotionType::Neutral => 1.0,
            EmotionType::Custom(_) => 1.0,
        };

        config.pitch.range_semitones *= range_factor;

        // Adjust intonation pattern intensity
        let _intonation_factor = match emotion.emotion_type {
            EmotionType::Happy | EmotionType::Excited => 1.0 + (0.2 * intensity),
            EmotionType::Surprise => 1.0 + (0.3 * intensity),
            EmotionType::Angry => 1.0 + (0.25 * intensity),
            EmotionType::Sad | EmotionType::Calm => 1.0 - (0.15 * intensity),
            _ => 1.0,
        };

        // Apply intonation adjustments
        // Note: IntonationPattern is an enum, so we can't directly modify intensity
        // This would need to be implemented differently based on the pattern type

        Ok(())
    }

    /// Apply emotion-specific energy modifications
    fn apply_emotion_to_energy(
        &self,
        config: &mut ProsodyConfig,
        emotion: &EmotionConfig,
    ) -> Result<()> {
        let intensity = emotion.intensity.as_f32() * self.emotion_intensity_scale;

        // Base energy adjustments by emotion type
        let energy_factor = match emotion.emotion_type {
            EmotionType::Happy => 1.0 + (0.2 * intensity),
            EmotionType::Excited => 1.0 + (0.4 * intensity),
            EmotionType::Angry => 1.0 + (0.5 * intensity),
            EmotionType::Fear => 1.0 + (0.3 * intensity),
            EmotionType::Surprise => 1.0 + (0.35 * intensity),
            EmotionType::Sad => 1.0 - (0.25 * intensity),
            EmotionType::Calm => 1.0 - (0.20 * intensity),
            EmotionType::Love => 1.0 + (0.1 * intensity),
            EmotionType::Disgust => 1.0 - (0.15 * intensity),
            EmotionType::Neutral => 1.0,
            EmotionType::Custom(_) => 1.0,
        };

        // Apply energy factor
        config.energy.base_energy *= energy_factor;

        // Adjust energy variation based on emotion
        let variation_factor = match emotion.emotion_type {
            EmotionType::Happy | EmotionType::Excited => 1.0 + (0.15 * intensity),
            EmotionType::Angry => 1.0 + (0.25 * intensity),
            EmotionType::Fear => 1.0 + (0.20 * intensity),
            EmotionType::Sad | EmotionType::Calm => 1.0 - (0.10 * intensity),
            _ => 1.0,
        };

        config.energy.dynamic_range *= variation_factor;

        Ok(())
    }

    /// Apply emotion-specific variation modifications
    fn apply_emotion_to_variation(
        &self,
        config: &mut ProsodyConfig,
        emotion: &EmotionConfig,
    ) -> Result<()> {
        let intensity = emotion.intensity.as_f32() * self.emotion_intensity_scale;

        // Overall variation adjustments by emotion type
        let variation_adjustment = match emotion.emotion_type {
            EmotionType::Happy | EmotionType::Excited => 0.1 * intensity,
            EmotionType::Angry => 0.15 * intensity,
            EmotionType::Fear => 0.2 * intensity,
            EmotionType::Surprise => 0.25 * intensity,
            EmotionType::Sad => -0.05 * intensity,
            EmotionType::Calm => -0.1 * intensity,
            EmotionType::Love => 0.05 * intensity,
            EmotionType::Disgust => -0.05 * intensity,
            EmotionType::Neutral => 0.0,
            EmotionType::Custom(_) => 0.0,
        };

        config.variation = (config.variation + variation_adjustment).clamp(0.0, 1.0);

        Ok(())
    }

    /// Apply custom emotion parameters to prosody
    fn apply_custom_parameters(
        &self,
        config: &mut ProsodyConfig,
        emotion: &EmotionConfig,
    ) -> Result<()> {
        // Apply speed factor from custom parameters
        if let Some(&speed_factor) = emotion.custom_params.get("speed_factor") {
            config.duration.speed_factor *= speed_factor;
        }

        // Apply pitch shift from custom parameters
        if let Some(&pitch_shift) = emotion.custom_params.get("pitch_shift") {
            config.pitch.base_frequency += pitch_shift * 12.0; // Convert semitones to Hz (approximation)
        }

        // Apply energy from custom parameters
        if let Some(&energy) = emotion.custom_params.get("energy") {
            config.energy.base_energy *= energy;
        }

        // Apply arousal and valence if available
        if let Some(&arousal) = emotion.custom_params.get("arousal") {
            // High arousal increases energy and variation
            config.energy.base_energy *= 1.0 + (arousal - 0.5) * 0.3;
            config.variation = (config.variation + (arousal - 0.5) * 0.2).clamp(0.0, 1.0);
        }

        if let Some(&valence) = emotion.custom_params.get("valence") {
            // Positive valence increases pitch and energy
            config.pitch.base_frequency += (valence - 0.5) * 20.0;
            config.energy.base_energy *= 1.0 + (valence - 0.5) * 0.2;
        }

        Ok(())
    }

    /// Create emotion-specific prosody presets
    pub fn create_emotion_preset(
        emotion_type: EmotionType,
        intensity: EmotionIntensity,
    ) -> Result<ProsodyConfig> {
        let base_config = ProsodyConfig::preset_natural();
        let mut modifier = EmotionProsodyModifier::new(base_config);

        let emotion_config = EmotionConfig::new(emotion_type).with_intensity(intensity);
        modifier.apply_emotion(emotion_config)
    }

    /// Transition between emotion prosody configurations
    pub fn transition_prosody(
        &self,
        from: &ProsodyConfig,
        to: &ProsodyConfig,
        alpha: f32,
    ) -> ProsodyConfig {
        let alpha = alpha.clamp(0.0, 1.0);

        // Interpolate between configurations
        let mut result = from.clone();

        // Interpolate duration
        result.duration.speed_factor =
            from.duration.speed_factor * (1.0 - alpha) + to.duration.speed_factor * alpha;

        // Interpolate pitch
        result.pitch.base_frequency =
            from.pitch.base_frequency * (1.0 - alpha) + to.pitch.base_frequency * alpha;
        result.pitch.range_semitones =
            from.pitch.range_semitones * (1.0 - alpha) + to.pitch.range_semitones * alpha;

        // Interpolate energy
        result.energy.base_energy =
            from.energy.base_energy * (1.0 - alpha) + to.energy.base_energy * alpha;
        result.energy.dynamic_range =
            from.energy.dynamic_range * (1.0 - alpha) + to.energy.dynamic_range * alpha;

        // Interpolate overall settings
        result.strength = from.strength * (1.0 - alpha) + to.strength * alpha;
        result.variation = from.variation * (1.0 - alpha) + to.variation * alpha;

        result
    }

    /// Get current emotion configuration
    pub fn get_current_emotion(&self) -> Option<&EmotionConfig> {
        self.current_emotion.as_ref()
    }

    /// Clear current emotion (reset to base)
    pub fn clear_emotion(&mut self) {
        self.current_emotion = None;
    }
}

impl Default for EmotionProsodyModifier {
    fn default() -> Self {
        Self::new(ProsodyConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prosody_config_creation() {
        let config = ProsodyConfig::new();
        assert_eq!(config.strength, 1.0);
        assert_eq!(config.variation, 0.1);
    }

    #[test]
    fn test_prosody_config_builder() {
        let config = ProsodyConfig::new().with_strength(0.8).with_variation(0.2);

        assert_eq!(config.strength, 0.8);
        assert_eq!(config.variation, 0.2);
    }

    #[test]
    fn test_prosody_config_validation() {
        let valid_config = ProsodyConfig::new().with_strength(0.5).with_variation(0.3);
        assert!(valid_config.validate().is_ok());

        let mut invalid_config = ProsodyConfig::new();
        invalid_config.strength = 1.5; // Invalid - bypass builder validation
        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn test_prosody_presets() {
        let natural = ProsodyConfig::preset_natural();
        assert_eq!(natural.variation, 0.15);

        let expressive = ProsodyConfig::preset_expressive();
        assert_eq!(expressive.variation, 0.25);

        let monotone = ProsodyConfig::preset_monotone();
        assert_eq!(monotone.variation, 0.05);
    }

    #[test]
    fn test_prosody_controller() {
        let base_config = ProsodyConfig::preset_natural();
        let mut controller = ProsodyController::new(base_config.clone());

        assert_eq!(controller.get_current_config().variation, 0.15);

        // Apply adjustment
        controller.apply_adjustment(ProsodyAdjustment::Speed(1.2));
        assert_eq!(controller.get_adjustment_history().len(), 1);

        // Reset
        controller.reset_to_base();
        assert_eq!(controller.get_current_config().variation, 0.15);
    }
}

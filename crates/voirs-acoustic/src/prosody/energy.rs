//! Energy control for loudness, dynamics, and spectral characteristics.

use super::duration::StressLevel;
use crate::{AcousticError, Phoneme, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Energy control configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyConfig {
    /// Base energy level (0.0 - 1.0)
    pub base_energy: f32,
    /// Dynamic range in dB
    pub dynamic_range: f32,
    /// Phoneme-specific energy adjustments (dB)
    pub phoneme_adjustments: HashMap<String, f32>,
    /// Stress-based energy adjustments (dB)
    pub stress_adjustments: HashMap<StressLevel, f32>,
    /// Energy contour pattern
    pub contour_pattern: EnergyContourPattern,
    /// Spectral tilt (dB/octave)
    pub spectral_tilt: f32,
    /// Breathiness level (0.0 - 1.0)
    pub breathiness: f32,
    /// Voice quality settings
    pub voice_quality: VoiceQualityConfig,
}

impl EnergyConfig {
    /// Create new energy config
    pub fn new() -> Self {
        Self {
            base_energy: 0.7,
            dynamic_range: 20.0,
            phoneme_adjustments: Self::default_phoneme_adjustments(),
            stress_adjustments: Self::default_stress_adjustments(),
            contour_pattern: EnergyContourPattern::Natural,
            spectral_tilt: -6.0, // Natural spectral slope
            breathiness: 0.1,
            voice_quality: VoiceQualityConfig::default(),
        }
    }

    /// Set base energy level
    pub fn with_base_energy(mut self, energy: f32) -> Self {
        self.base_energy = energy.clamp(0.0, 1.0);
        self
    }

    /// Set dynamic range
    pub fn with_dynamic_range(mut self, range_db: f32) -> Self {
        self.dynamic_range = range_db.clamp(1.0, 60.0);
        self
    }

    /// Set contour pattern
    pub fn with_contour_pattern(mut self, pattern: EnergyContourPattern) -> Self {
        self.contour_pattern = pattern;
        self
    }

    /// Set spectral tilt
    pub fn with_spectral_tilt(mut self, tilt: f32) -> Self {
        self.spectral_tilt = tilt.clamp(-20.0, 20.0);
        self
    }

    /// Set breathiness level
    pub fn with_breathiness(mut self, breathiness: f32) -> Self {
        self.breathiness = breathiness.clamp(0.0, 1.0);
        self
    }

    /// Set voice quality
    pub fn with_voice_quality(mut self, quality: VoiceQualityConfig) -> Self {
        self.voice_quality = quality;
        self
    }

    /// Get default phoneme energy adjustments
    fn default_phoneme_adjustments() -> HashMap<String, f32> {
        let mut adjustments = HashMap::new();

        // Vowels (typically higher energy)
        for vowel in ["a", "e", "i", "o", "u", "ɑ", "ɛ", "ɪ", "ɔ", "ʊ"] {
            adjustments.insert(vowel.to_string(), 3.0);
        }

        // Fricatives (high frequency energy)
        for fricative in ["f", "θ", "s", "ʃ", "v", "ð", "z", "ʒ", "h"] {
            adjustments.insert(fricative.to_string(), 1.0);
        }

        // Stops (burst energy)
        for stop in ["p", "t", "k", "b", "d", "g"] {
            adjustments.insert(stop.to_string(), 2.0);
        }

        // Nasals (moderate energy)
        for nasal in ["m", "n", "ŋ"] {
            adjustments.insert(nasal.to_string(), 0.0);
        }

        // Liquids (moderate energy)
        for liquid in ["l", "r"] {
            adjustments.insert(liquid.to_string(), 1.0);
        }

        adjustments
    }

    /// Get default stress energy adjustments
    fn default_stress_adjustments() -> HashMap<StressLevel, f32> {
        let mut adjustments = HashMap::new();
        adjustments.insert(StressLevel::Unstressed, -3.0);
        adjustments.insert(StressLevel::Secondary, 0.0);
        adjustments.insert(StressLevel::Primary, 6.0);
        adjustments
    }

    /// Calculate energy for a phoneme
    pub fn calculate_energy(&self, phoneme: &Phoneme, context: &EnergyContext) -> f32 {
        // Start with base energy
        let mut energy = self.base_energy;

        // Apply contour pattern
        energy *= self
            .contour_pattern
            .get_energy_factor(context.position_in_phrase);

        // Apply phoneme-specific adjustment
        if let Some(adjustment_db) = self.phoneme_adjustments.get(&phoneme.symbol) {
            energy *= Self::db_to_linear(*adjustment_db);
        }

        // Apply stress-based adjustment
        let stress_level = Self::extract_stress_level(phoneme);
        if let Some(adjustment_db) = self.stress_adjustments.get(&stress_level) {
            energy *= Self::db_to_linear(*adjustment_db);
        }

        // Apply position-based factors
        energy *= context.position_factor;

        // Apply phrase-level dynamics
        energy *= context.phrase_dynamics_factor;

        // Clamp to valid range
        energy.clamp(0.0, 1.0)
    }

    /// Generate energy contour for phoneme sequence
    pub fn generate_energy_contour(&self, phonemes: &[Phoneme], frame_rate: f32) -> Vec<f32> {
        let mut contour = Vec::new();
        let mut current_time = 0.0;
        let frame_duration = 1.0 / frame_rate;

        for (i, phoneme) in phonemes.iter().enumerate() {
            let phoneme_duration = phoneme.duration.unwrap_or(0.1);
            let position_in_phrase = i as f32 / phonemes.len() as f32;

            let frames_in_phoneme = (phoneme_duration * frame_rate) as usize;

            for frame in 0..frames_in_phoneme {
                let time_in_phoneme = frame as f32 * frame_duration;
                let context = EnergyContext {
                    time_in_utterance: current_time + time_in_phoneme,
                    time_in_phoneme,
                    position_in_phrase,
                    position_factor: self.calculate_position_factor(i, phonemes.len()),
                    phrase_dynamics_factor: 1.0, // Could be adjusted based on phrase context
                };

                let energy = self.calculate_energy(phoneme, &context);
                contour.push(energy);
            }

            current_time += phoneme_duration;
        }

        // Apply smoothing
        self.smooth_contour(&mut contour);

        contour
    }

    /// Calculate position-based energy factor
    fn calculate_position_factor(&self, position: usize, total: usize) -> f32 {
        if total <= 1 {
            return 1.0;
        }

        let normalized_pos = position as f32 / (total - 1) as f32;

        match self.contour_pattern {
            EnergyContourPattern::Natural => {
                // Slight increase at beginning, gradual decrease
                if normalized_pos < 0.1 {
                    1.1
                } else {
                    1.1 - (normalized_pos * 0.3)
                }
            }
            EnergyContourPattern::Uniform => 1.0,
            EnergyContourPattern::Crescendo => {
                // Gradual increase
                0.7 + (normalized_pos * 0.6)
            }
            EnergyContourPattern::Diminuendo => {
                // Gradual decrease
                1.3 - (normalized_pos * 0.6)
            }
            EnergyContourPattern::Dramatic => {
                // More dynamic variation
                1.0 + 0.5 * (2.0 * std::f32::consts::PI * normalized_pos * 3.0).sin()
            }
        }
    }

    /// Smooth energy contour
    fn smooth_contour(&self, contour: &mut [f32]) {
        if contour.len() < 3 {
            return;
        }

        // Simple moving average smoothing
        let window_size = 5;
        let mut smoothed = Vec::with_capacity(contour.len());

        for i in 0..contour.len() {
            let start = i.saturating_sub(window_size / 2);
            let end = (i + window_size / 2 + 1).min(contour.len());

            let sum: f32 = contour[start..end].iter().sum();
            let count = end - start;
            smoothed.push(sum / count as f32);
        }

        contour.copy_from_slice(&smoothed);
    }

    /// Extract stress level from phoneme
    fn extract_stress_level(phoneme: &Phoneme) -> StressLevel {
        if let Some(ref features) = phoneme.features {
            if let Some(stress) = features.get("stress") {
                match stress.as_str() {
                    "primary" | "1" => StressLevel::Primary,
                    "secondary" | "2" => StressLevel::Secondary,
                    _ => StressLevel::Unstressed,
                }
            } else {
                StressLevel::Unstressed
            }
        } else {
            StressLevel::Unstressed
        }
    }

    /// Convert dB to linear scale
    fn db_to_linear(db: f32) -> f32 {
        10.0_f32.powf(db / 20.0)
    }

    /// Convert linear to dB scale
    #[allow(dead_code)]
    fn linear_to_db(linear: f32) -> f32 {
        20.0 * linear.max(1e-6).log10()
    }

    /// Validate energy configuration
    pub fn validate(&self) -> Result<()> {
        if !(0.0..=1.0).contains(&self.base_energy) {
            return Err(AcousticError::ConfigError(format!(
                "Base energy {} must be between 0.0 and 1.0",
                self.base_energy
            )));
        }

        if self.dynamic_range < 1.0 || self.dynamic_range > 60.0 {
            return Err(AcousticError::ConfigError(format!(
                "Dynamic range {} dB is out of range (1-60 dB)",
                self.dynamic_range
            )));
        }

        if !(0.0..=1.0).contains(&self.breathiness) {
            return Err(AcousticError::ConfigError(format!(
                "Breathiness {} must be between 0.0 and 1.0",
                self.breathiness
            )));
        }

        self.voice_quality.validate()?;

        Ok(())
    }

    /// Create preset configurations
    pub fn natural() -> Self {
        Self::new()
            .with_contour_pattern(EnergyContourPattern::Natural)
            .with_breathiness(0.1)
    }

    pub fn expressive() -> Self {
        Self::new()
            .with_dynamic_range(30.0)
            .with_contour_pattern(EnergyContourPattern::Dramatic)
            .with_breathiness(0.05)
    }

    pub fn uniform() -> Self {
        Self::new()
            .with_dynamic_range(5.0)
            .with_contour_pattern(EnergyContourPattern::Uniform)
            .with_breathiness(0.0)
    }

    pub fn soft() -> Self {
        Self::new()
            .with_base_energy(0.4)
            .with_dynamic_range(15.0)
            .with_breathiness(0.3)
    }

    pub fn strong() -> Self {
        Self::new()
            .with_base_energy(0.9)
            .with_dynamic_range(25.0)
            .with_breathiness(0.0)
    }
}

impl Default for EnergyConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Energy contour patterns
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum EnergyContourPattern {
    /// Natural speech energy contour
    Natural,
    /// Uniform energy level
    Uniform,
    /// Gradually increasing energy
    Crescendo,
    /// Gradually decreasing energy
    Diminuendo,
    /// Dramatic energy variations
    Dramatic,
}

impl EnergyContourPattern {
    /// Get energy factor for position in phrase
    pub fn get_energy_factor(&self, position: f32) -> f32 {
        match self {
            EnergyContourPattern::Natural => {
                // Natural declination with slight variations
                1.0 - (position * 0.2) + 0.05 * (2.0 * std::f32::consts::PI * position * 8.0).sin()
            }
            EnergyContourPattern::Uniform => 1.0,
            EnergyContourPattern::Crescendo => 0.7 + (position * 0.6),
            EnergyContourPattern::Diminuendo => 1.3 - (position * 0.6),
            EnergyContourPattern::Dramatic => {
                1.0 + 0.4 * (2.0 * std::f32::consts::PI * position * 2.5).sin()
            }
        }
    }

    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            EnergyContourPattern::Natural => "natural",
            EnergyContourPattern::Uniform => "uniform",
            EnergyContourPattern::Crescendo => "crescendo",
            EnergyContourPattern::Diminuendo => "diminuendo",
            EnergyContourPattern::Dramatic => "dramatic",
        }
    }
}

/// Voice quality configuration for energy shaping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceQualityConfig {
    /// Vocal fry intensity (0.0 - 1.0)
    pub vocal_fry: f32,
    /// Creakiness (0.0 - 1.0)
    pub creakiness: f32,
    /// Harshness (0.0 - 1.0)
    pub harshness: f32,
    /// Nasality (0.0 - 1.0)
    pub nasality: f32,
    /// Formant shift (semitones)
    pub formant_shift: f32,
}

impl VoiceQualityConfig {
    /// Create new voice quality config
    pub fn new() -> Self {
        Self {
            vocal_fry: 0.0,
            creakiness: 0.0,
            harshness: 0.0,
            nasality: 0.0,
            formant_shift: 0.0,
        }
    }

    /// Validate voice quality configuration
    pub fn validate(&self) -> Result<()> {
        let fields = [
            ("vocal_fry", self.vocal_fry),
            ("creakiness", self.creakiness),
            ("harshness", self.harshness),
            ("nasality", self.nasality),
        ];

        for (name, value) in &fields {
            if !(0.0..=1.0).contains(value) {
                return Err(AcousticError::ConfigError(format!(
                    "{name} {value} must be between 0.0 and 1.0"
                )));
            }
        }

        if self.formant_shift < -12.0 || self.formant_shift > 12.0 {
            return Err(AcousticError::ConfigError(format!(
                "Formant shift {} semitones is out of range (-12 to 12)",
                self.formant_shift
            )));
        }

        Ok(())
    }

    /// Create preset voice qualities
    pub fn clear() -> Self {
        Self::new()
    }

    pub fn breathy() -> Self {
        Self {
            vocal_fry: 0.0,
            creakiness: 0.0,
            harshness: 0.0,
            nasality: 0.0,
            formant_shift: 0.0,
        }
    }

    pub fn creaky() -> Self {
        Self {
            vocal_fry: 0.6,
            creakiness: 0.8,
            harshness: 0.2,
            nasality: 0.0,
            formant_shift: -1.0,
        }
    }

    pub fn nasal() -> Self {
        Self {
            vocal_fry: 0.0,
            creakiness: 0.0,
            harshness: 0.0,
            nasality: 0.7,
            formant_shift: 0.0,
        }
    }
}

impl Default for VoiceQualityConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Context for energy calculation
#[derive(Debug, Clone)]
pub struct EnergyContext {
    /// Time since utterance start (seconds)
    pub time_in_utterance: f32,
    /// Time within current phoneme (seconds)
    pub time_in_phoneme: f32,
    /// Position in phrase (0.0 - 1.0)
    pub position_in_phrase: f32,
    /// Position-based factor
    pub position_factor: f32,
    /// Phrase-level dynamics factor
    pub phrase_dynamics_factor: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_energy_config_creation() {
        let config = EnergyConfig::new();
        assert_eq!(config.base_energy, 0.7);
        assert_eq!(config.dynamic_range, 20.0);
        assert_eq!(config.breathiness, 0.1);
    }

    #[test]
    fn test_energy_config_builder() {
        let config = EnergyConfig::new()
            .with_base_energy(0.8)
            .with_dynamic_range(30.0)
            .with_breathiness(0.2);

        assert_eq!(config.base_energy, 0.8);
        assert_eq!(config.dynamic_range, 30.0);
        assert_eq!(config.breathiness, 0.2);
    }

    #[test]
    fn test_energy_config_validation() {
        let valid_config = EnergyConfig::new();
        assert!(valid_config.validate().is_ok());

        let mut invalid_energy = EnergyConfig::new();
        invalid_energy.base_energy = 1.5; // Too high - bypass builder validation
        assert!(invalid_energy.validate().is_err());

        let mut invalid_range = EnergyConfig::new();
        invalid_range.dynamic_range = 100.0; // Too wide - bypass builder validation
        assert!(invalid_range.validate().is_err());
    }

    #[test]
    fn test_energy_calculation() {
        let config = EnergyConfig::new();
        let phoneme = Phoneme::new("a"); // Vowel should have positive adjustment
        let context = EnergyContext {
            time_in_utterance: 0.5,
            time_in_phoneme: 0.05,
            position_in_phrase: 0.3,
            position_factor: 1.0,
            phrase_dynamics_factor: 1.0,
        };

        let energy = config.calculate_energy(&phoneme, &context);
        assert!(energy >= 0.0);
        assert!(energy <= 1.0);
    }

    #[test]
    fn test_db_conversions() {
        assert!((EnergyConfig::db_to_linear(0.0) - 1.0).abs() < 1e-6);
        assert!((EnergyConfig::db_to_linear(20.0) - 10.0).abs() < 1e-6);
        assert!((EnergyConfig::db_to_linear(-20.0) - 0.1).abs() < 1e-6);

        assert!((EnergyConfig::linear_to_db(1.0) - 0.0).abs() < 1e-6);
        assert!((EnergyConfig::linear_to_db(10.0) - 20.0).abs() < 1e-6);
    }

    #[test]
    fn test_energy_contour_patterns() {
        assert_eq!(EnergyContourPattern::Uniform.get_energy_factor(0.5), 1.0);

        let crescendo_start = EnergyContourPattern::Crescendo.get_energy_factor(0.0);
        let crescendo_end = EnergyContourPattern::Crescendo.get_energy_factor(1.0);
        assert!(crescendo_end > crescendo_start);

        let diminuendo_start = EnergyContourPattern::Diminuendo.get_energy_factor(0.0);
        let diminuendo_end = EnergyContourPattern::Diminuendo.get_energy_factor(1.0);
        assert!(diminuendo_end < diminuendo_start);
    }

    #[test]
    fn test_voice_quality_config() {
        let quality = VoiceQualityConfig::new();
        assert!(quality.validate().is_ok());

        let creaky = VoiceQualityConfig::creaky();
        assert!(creaky.vocal_fry > 0.0);
        assert!(creaky.creakiness > 0.0);

        let nasal = VoiceQualityConfig::nasal();
        assert!(nasal.nasality > 0.0);
    }

    #[test]
    fn test_voice_quality_validation() {
        let invalid_quality = VoiceQualityConfig {
            vocal_fry: 1.5, // Invalid
            creakiness: 0.5,
            harshness: 0.3,
            nasality: 0.2,
            formant_shift: 0.0,
        };
        assert!(invalid_quality.validate().is_err());

        let invalid_shift = VoiceQualityConfig {
            vocal_fry: 0.0,
            creakiness: 0.0,
            harshness: 0.0,
            nasality: 0.0,
            formant_shift: 20.0, // Too large
        };
        assert!(invalid_shift.validate().is_err());
    }

    #[test]
    fn test_energy_contour_generation() {
        let config = EnergyConfig::new();
        let mut phonemes = vec![
            Phoneme::new("h"),
            Phoneme::new("e"),
            Phoneme::new("l"),
            Phoneme::new("o"),
        ];

        // Set durations
        for phoneme in &mut phonemes {
            phoneme.duration = Some(0.1);
        }

        let contour = config.generate_energy_contour(&phonemes, 100.0); // 100 Hz frame rate
        assert!(!contour.is_empty());

        // All energy values should be valid
        for &energy in &contour {
            assert!(energy >= 0.0);
            assert!(energy <= 1.0);
        }
    }

    #[test]
    fn test_preset_configurations() {
        let natural = EnergyConfig::natural();
        assert_eq!(natural.contour_pattern, EnergyContourPattern::Natural);

        let expressive = EnergyConfig::expressive();
        assert_eq!(expressive.dynamic_range, 30.0);
        assert_eq!(expressive.contour_pattern, EnergyContourPattern::Dramatic);

        let uniform = EnergyConfig::uniform();
        assert_eq!(uniform.contour_pattern, EnergyContourPattern::Uniform);

        let soft = EnergyConfig::soft();
        assert_eq!(soft.base_energy, 0.4);
        assert!(soft.breathiness > 0.0);

        let strong = EnergyConfig::strong();
        assert_eq!(strong.base_energy, 0.9);
        assert_eq!(strong.breathiness, 0.0);
    }
}

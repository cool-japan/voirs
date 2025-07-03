//! Pitch control for F0 contour and intonation.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::{Result, AcousticError, Phoneme};
use super::duration::StressLevel;

/// Pitch control configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PitchConfig {
    /// Base fundamental frequency in Hz
    pub base_frequency: f32,
    /// Pitch range in semitones (total range)
    pub range_semitones: f32,
    /// Intonation pattern
    pub intonation_pattern: IntonationPattern,
    /// Phoneme-specific pitch adjustments (semitones)
    pub phoneme_adjustments: HashMap<String, f32>,
    /// Stress-based pitch adjustments (semitones)
    pub stress_adjustments: HashMap<StressLevel, f32>,
    /// Declination rate (Hz per second)
    pub declination_rate: f32,
    /// Vibrato settings
    pub vibrato: VibratoConfig,
    /// F0 contour smoothing factor (0.0 - 1.0)
    pub smoothing: f32,
}

impl PitchConfig {
    /// Create new pitch config
    pub fn new() -> Self {
        Self {
            base_frequency: 150.0, // Neutral voice
            range_semitones: 12.0,  // One octave
            intonation_pattern: IntonationPattern::Natural,
            phoneme_adjustments: Self::default_phoneme_adjustments(),
            stress_adjustments: Self::default_stress_adjustments(),
            declination_rate: 2.0, // 2 Hz per second
            vibrato: VibratoConfig::default(),
            smoothing: 0.7,
        }
    }
    
    /// Set base frequency
    pub fn with_base_frequency(mut self, frequency: f32) -> Self {
        self.base_frequency = frequency.clamp(50.0, 500.0);
        self
    }
    
    /// Set pitch range
    pub fn with_range_semitones(mut self, range: f32) -> Self {
        self.range_semitones = range.clamp(1.0, 48.0);
        self
    }
    
    /// Set intonation pattern
    pub fn with_intonation_pattern(mut self, pattern: IntonationPattern) -> Self {
        self.intonation_pattern = pattern;
        self
    }
    
    /// Set declination rate
    pub fn with_declination_rate(mut self, rate: f32) -> Self {
        self.declination_rate = rate.clamp(0.0, 10.0);
        self
    }
    
    /// Set vibrato
    pub fn with_vibrato(mut self, vibrato: VibratoConfig) -> Self {
        self.vibrato = vibrato;
        self
    }
    
    /// Set smoothing factor
    pub fn with_smoothing(mut self, smoothing: f32) -> Self {
        self.smoothing = smoothing.clamp(0.0, 1.0);
        self
    }
    
    /// Get default phoneme adjustments
    fn default_phoneme_adjustments() -> HashMap<String, f32> {
        let mut adjustments = HashMap::new();
        
        // High vowels typically have higher F0
        for vowel in ["i", "ɪ", "u", "ʊ", "e"] {
            adjustments.insert(vowel.to_string(), 1.0);
        }
        
        // Low vowels typically have lower F0
        for vowel in ["a", "ɑ", "ɔ", "o"] {
            adjustments.insert(vowel.to_string(), -1.0);
        }
        
        // Voiced consonants slightly lower
        for consonant in ["b", "d", "g", "v", "z", "ʒ", "m", "n", "ŋ", "l", "r"] {
            adjustments.insert(consonant.to_string(), -0.5);
        }
        
        adjustments
    }
    
    /// Get default stress adjustments
    fn default_stress_adjustments() -> HashMap<StressLevel, f32> {
        let mut adjustments = HashMap::new();
        adjustments.insert(StressLevel::Unstressed, -1.0);
        adjustments.insert(StressLevel::Secondary, 0.0);
        adjustments.insert(StressLevel::Primary, 2.0);
        adjustments
    }
    
    /// Calculate F0 for a phoneme at a given time
    pub fn calculate_f0(&self, phoneme: &Phoneme, context: &PitchContext) -> f32 {
        // Start with base frequency
        let mut f0 = self.base_frequency;
        
        // Apply declination (gradual lowering over time)
        f0 -= self.declination_rate * context.time_in_utterance;
        
        // Apply intonation pattern
        f0 *= self.intonation_pattern.get_frequency_factor(context.position_in_phrase);
        
        // Apply phoneme-specific adjustment
        if let Some(adjustment) = self.phoneme_adjustments.get(&phoneme.symbol) {
            f0 *= Self::semitones_to_factor(*adjustment);
        }
        
        // Apply stress-based adjustment
        let stress_level = Self::extract_stress_level(phoneme);
        if let Some(adjustment) = self.stress_adjustments.get(&stress_level) {
            f0 *= Self::semitones_to_factor(*adjustment);
        }
        
        // Apply vibrato if enabled
        if self.vibrato.enabled {
            f0 += self.vibrato.calculate_modulation(context.time_in_phoneme);
        }
        
        // Clamp to reasonable range
        f0.clamp(50.0, 800.0)
    }
    
    /// Generate F0 contour for phoneme sequence
    pub fn generate_f0_contour(&self, phonemes: &[Phoneme], frame_rate: f32) -> Vec<f32> {
        let mut contour = Vec::new();
        let mut current_time = 0.0;
        let frame_duration = 1.0 / frame_rate;
        
        for (i, phoneme) in phonemes.iter().enumerate() {
            let phoneme_duration = phoneme.duration.unwrap_or(0.1); // 100ms default
            let position_in_phrase = i as f32 / phonemes.len() as f32;
            
            let frames_in_phoneme = (phoneme_duration * frame_rate) as usize;
            
            for frame in 0..frames_in_phoneme {
                let time_in_phoneme = frame as f32 * frame_duration;
                let context = PitchContext {
                    time_in_utterance: current_time + time_in_phoneme,
                    time_in_phoneme,
                    position_in_phrase,
                    is_voiced: Self::is_voiced_phoneme(phoneme),
                };
                
                let f0 = if context.is_voiced {
                    self.calculate_f0(phoneme, &context)
                } else {
                    0.0 // Unvoiced
                };
                
                contour.push(f0);
            }
            
            current_time += phoneme_duration;
        }
        
        // Apply smoothing
        if self.smoothing > 0.0 {
            self.smooth_contour(&mut contour);
        }
        
        contour
    }
    
    /// Smooth F0 contour
    fn smooth_contour(&self, contour: &mut [f32]) {
        if contour.len() < 3 {
            return;
        }
        
        let alpha = 1.0 - self.smoothing;
        let mut previous = contour[0];
        
        for i in 1..contour.len() {
            if contour[i] > 0.0 && previous > 0.0 {
                // Only smooth voiced frames
                contour[i] = alpha * contour[i] + (1.0 - alpha) * previous;
            }
            if contour[i] > 0.0 {
                previous = contour[i];
            }
        }
    }
    
    /// Check if phoneme is voiced
    fn is_voiced_phoneme(phoneme: &Phoneme) -> bool {
        // Simple heuristic based on phoneme symbol
        let voiced_phonemes = [
            // Vowels
            "a", "e", "i", "o", "u", "ɑ", "ɛ", "ɪ", "ɔ", "ʊ", "ə", "ɚ",
            // Voiced consonants
            "b", "d", "g", "v", "ð", "z", "ʒ", "m", "n", "ŋ", "l", "r", "w", "j",
        ];
        
        voiced_phonemes.contains(&phoneme.symbol.as_str())
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
    
    /// Convert semitones to frequency factor
    fn semitones_to_factor(semitones: f32) -> f32 {
        2.0_f32.powf(semitones / 12.0)
    }
    
    /// Validate pitch configuration
    pub fn validate(&self) -> Result<()> {
        if self.base_frequency < 50.0 || self.base_frequency > 500.0 {
            return Err(AcousticError::ConfigError(
                format!("Base frequency {} Hz is out of range (50-500 Hz)", self.base_frequency)
            ));
        }
        
        if self.range_semitones < 1.0 || self.range_semitones > 48.0 {
            return Err(AcousticError::ConfigError(
                format!("Pitch range {} semitones is out of range (1-48)", self.range_semitones)
            ));
        }
        
        if !(0.0..=1.0).contains(&self.smoothing) {
            return Err(AcousticError::ConfigError(
                format!("Smoothing factor {} must be between 0.0 and 1.0", self.smoothing)
            ));
        }
        
        self.vibrato.validate()?;
        
        Ok(())
    }
    
    /// Create preset configurations
    pub fn natural() -> Self {
        Self::new()
            .with_intonation_pattern(IntonationPattern::Natural)
            .with_vibrato(VibratoConfig::natural())
    }
    
    pub fn expressive() -> Self {
        Self::new()
            .with_range_semitones(18.0)
            .with_intonation_pattern(IntonationPattern::Expressive)
            .with_vibrato(VibratoConfig::expressive())
    }
    
    pub fn flat() -> Self {
        Self::new()
            .with_range_semitones(2.0)
            .with_intonation_pattern(IntonationPattern::Flat)
            .with_declination_rate(0.0)
            .with_vibrato(VibratoConfig::disabled())
    }
    
    pub fn male_voice() -> Self {
        Self::new()
            .with_base_frequency(120.0)
            .with_range_semitones(10.0)
    }
    
    pub fn female_voice() -> Self {
        Self::new()
            .with_base_frequency(220.0)
            .with_range_semitones(14.0)
    }
    
    pub fn child_voice() -> Self {
        Self::new()
            .with_base_frequency(280.0)
            .with_range_semitones(16.0)
            .with_vibrato(VibratoConfig::child())
    }
}

impl Default for PitchConfig {
    fn default() -> Self {
        Self::new()
    }
}


/// Intonation patterns
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum IntonationPattern {
    /// Natural speech intonation
    Natural,
    /// Flat/monotone intonation
    Flat,
    /// Rising intonation (question-like)
    Rising,
    /// Falling intonation (statement-like)
    Falling,
    /// Expressive intonation with more variation
    Expressive,
}

impl IntonationPattern {
    /// Get frequency factor for position in phrase
    pub fn get_frequency_factor(&self, position: f32) -> f32 {
        match self {
            IntonationPattern::Natural => {
                // Slight rise at beginning, fall at end
                if position < 0.2 {
                    1.0 + (position * 0.1)
                } else if position > 0.8 {
                    1.02 - ((position - 0.8) * 0.1)
                } else {
                    1.02
                }
            },
            IntonationPattern::Flat => 1.0,
            IntonationPattern::Rising => {
                // Gradual rise throughout
                1.0 + (position * 0.15)
            },
            IntonationPattern::Falling => {
                // Gradual fall throughout
                1.15 - (position * 0.15)
            },
            IntonationPattern::Expressive => {
                // More dynamic contour
                1.0 + 0.2 * (0.5 * (2.0 * std::f32::consts::PI * position * 2.0).sin())
            },
        }
    }
    
    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            IntonationPattern::Natural => "natural",
            IntonationPattern::Flat => "flat",
            IntonationPattern::Rising => "rising",
            IntonationPattern::Falling => "falling",
            IntonationPattern::Expressive => "expressive",
        }
    }
}

/// Vibrato configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VibratoConfig {
    /// Whether vibrato is enabled
    pub enabled: bool,
    /// Vibrato frequency in Hz
    pub frequency: f32,
    /// Vibrato extent in semitones
    pub extent: f32,
    /// Vibrato onset delay in seconds
    pub onset_delay: f32,
}

impl VibratoConfig {
    /// Create new vibrato config
    pub fn new() -> Self {
        Self {
            enabled: false,
            frequency: 5.0,
            extent: 0.5,
            onset_delay: 0.0,
        }
    }
    
    /// Calculate vibrato modulation
    pub fn calculate_modulation(&self, time_in_phoneme: f32) -> f32 {
        if !self.enabled || time_in_phoneme < self.onset_delay {
            return 0.0;
        }
        
        let adjusted_time = time_in_phoneme - self.onset_delay;
        let modulation = (2.0 * std::f32::consts::PI * self.frequency * adjusted_time).sin();
        let extent_hz = self.extent * 2.0; // Approximate conversion
        
        modulation * extent_hz
    }
    
    /// Validate vibrato configuration
    pub fn validate(&self) -> Result<()> {
        if self.frequency < 0.1 || self.frequency > 20.0 {
            return Err(AcousticError::ConfigError(
                format!("Vibrato frequency {} Hz is out of range (0.1-20.0 Hz)", self.frequency)
            ));
        }
        
        if self.extent < 0.0 || self.extent > 3.0 {
            return Err(AcousticError::ConfigError(
                format!("Vibrato extent {} semitones is out of range (0.0-3.0)", self.extent)
            ));
        }
        
        Ok(())
    }
    
    /// Create preset configurations
    pub fn disabled() -> Self {
        Self::new()
    }
    
    pub fn natural() -> Self {
        Self {
            enabled: true,
            frequency: 5.5,
            extent: 0.3,
            onset_delay: 0.1,
        }
    }
    
    pub fn expressive() -> Self {
        Self {
            enabled: true,
            frequency: 6.0,
            extent: 0.8,
            onset_delay: 0.05,
        }
    }
    
    pub fn child() -> Self {
        Self {
            enabled: true,
            frequency: 7.0,
            extent: 0.6,
            onset_delay: 0.0,
        }
    }
}

impl Default for VibratoConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Context for pitch calculation
#[derive(Debug, Clone)]
pub struct PitchContext {
    /// Time since utterance start (seconds)
    pub time_in_utterance: f32,
    /// Time within current phoneme (seconds)
    pub time_in_phoneme: f32,
    /// Position in phrase (0.0 - 1.0)
    pub position_in_phrase: f32,
    /// Whether the phoneme is voiced
    pub is_voiced: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pitch_config_creation() {
        let config = PitchConfig::new();
        assert_eq!(config.base_frequency, 150.0);
        assert_eq!(config.range_semitones, 12.0);
    }
    
    #[test]
    fn test_pitch_config_builder() {
        let config = PitchConfig::new()
            .with_base_frequency(200.0)
            .with_range_semitones(18.0)
            .with_smoothing(0.8);
        
        assert_eq!(config.base_frequency, 200.0);
        assert_eq!(config.range_semitones, 18.0);
        assert_eq!(config.smoothing, 0.8);
    }
    
    #[test]
    fn test_pitch_config_validation() {
        let valid_config = PitchConfig::new();
        assert!(valid_config.validate().is_ok());
        
        let mut invalid_frequency = PitchConfig::new();
        invalid_frequency.base_frequency = 1000.0; // Too high - bypass builder validation
        assert!(invalid_frequency.validate().is_err());
        
        let mut invalid_range = PitchConfig::new();
        invalid_range.range_semitones = 60.0; // Too wide - bypass builder validation
        assert!(invalid_range.validate().is_err());
    }
    
    #[test]
    fn test_f0_calculation() {
        let config = PitchConfig::new();
        let phoneme = Phoneme::new("a"); // Voiced vowel
        let context = PitchContext {
            time_in_utterance: 0.5,
            time_in_phoneme: 0.05,
            position_in_phrase: 0.3,
            is_voiced: true,
        };
        
        let f0 = config.calculate_f0(&phoneme, &context);
        assert!(f0 >= 50.0);
        assert!(f0 <= 800.0);
    }
    
    #[test]
    fn test_semitones_to_factor() {
        assert!((PitchConfig::semitones_to_factor(0.0) - 1.0).abs() < 1e-6);
        assert!((PitchConfig::semitones_to_factor(12.0) - 2.0).abs() < 1e-6);
        assert!((PitchConfig::semitones_to_factor(-12.0) - 0.5).abs() < 1e-6);
    }
    
    #[test]
    fn test_is_voiced_phoneme() {
        let voiced = Phoneme::new("a");
        assert!(PitchConfig::is_voiced_phoneme(&voiced));
        
        let unvoiced = Phoneme::new("p");
        assert!(!PitchConfig::is_voiced_phoneme(&unvoiced));
    }
    
    #[test]
    fn test_intonation_patterns() {
        assert_eq!(IntonationPattern::Flat.get_frequency_factor(0.5), 1.0);
        
        let rising_start = IntonationPattern::Rising.get_frequency_factor(0.0);
        let rising_end = IntonationPattern::Rising.get_frequency_factor(1.0);
        assert!(rising_end > rising_start);
        
        let falling_start = IntonationPattern::Falling.get_frequency_factor(0.0);
        let falling_end = IntonationPattern::Falling.get_frequency_factor(1.0);
        assert!(falling_end < falling_start);
    }
    
    #[test]
    fn test_vibrato_config() {
        let vibrato = VibratoConfig::natural();
        assert!(vibrato.enabled);
        assert_eq!(vibrato.frequency, 5.5);
        
        let disabled = VibratoConfig::disabled();
        assert!(!disabled.enabled);
        
        // Test modulation calculation
        let modulation = vibrato.calculate_modulation(0.2);
        assert!(modulation.abs() <= vibrato.extent * 2.0);
    }
    
    #[test]
    fn test_f0_contour_generation() {
        let config = PitchConfig::new();
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
        
        let contour = config.generate_f0_contour(&phonemes, 100.0); // 100 Hz frame rate
        assert!(!contour.is_empty());
        
        // Check that voiced frames have non-zero F0
        assert!(contour.iter().any(|&f0| f0 > 0.0));
    }
    
    #[test]
    fn test_preset_configurations() {
        let male = PitchConfig::male_voice();
        assert_eq!(male.base_frequency, 120.0);
        
        let female = PitchConfig::female_voice();
        assert_eq!(female.base_frequency, 220.0);
        
        let child = PitchConfig::child_voice();
        assert_eq!(child.base_frequency, 280.0);
        
        let flat = PitchConfig::flat();
        assert_eq!(flat.intonation_pattern, IntonationPattern::Flat);
        assert_eq!(flat.declination_rate, 0.0);
        assert!(!flat.vibrato.enabled);
    }
    
    #[test]
    fn test_vibrato_validation() {
        let valid_vibrato = VibratoConfig::natural();
        assert!(valid_vibrato.validate().is_ok());
        
        let invalid_frequency = VibratoConfig {
            enabled: true,
            frequency: 50.0, // Too high
            extent: 0.5,
            onset_delay: 0.0,
        };
        assert!(invalid_frequency.validate().is_err());
    }
}
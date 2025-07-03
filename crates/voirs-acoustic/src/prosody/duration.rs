//! Duration control for speech timing and rhythm.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::{Result, AcousticError, Phoneme};

/// Duration control configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DurationConfig {
    /// Global speaking rate multiplier (1.0 = normal)
    pub speed_factor: f32,
    /// Minimum phoneme duration in milliseconds
    pub min_duration_ms: f32,
    /// Maximum phoneme duration in milliseconds
    pub max_duration_ms: f32,
    /// Phoneme-specific duration multipliers
    pub phoneme_multipliers: HashMap<String, f32>,
    /// Stress-based duration adjustments
    pub stress_multipliers: HashMap<StressLevel, f32>,
    /// Pause durations for different contexts
    pub pause_durations: PauseDurations,
    /// Rhythm patterns
    pub rhythm_pattern: RhythmPattern,
}

impl DurationConfig {
    /// Create new duration config
    pub fn new() -> Self {
        Self {
            speed_factor: 1.0,
            min_duration_ms: 20.0,
            max_duration_ms: 500.0,
            phoneme_multipliers: Self::default_phoneme_multipliers(),
            stress_multipliers: Self::default_stress_multipliers(),
            pause_durations: PauseDurations::default(),
            rhythm_pattern: RhythmPattern::Natural,
        }
    }
    
    /// Set speed factor
    pub fn with_speed_factor(mut self, factor: f32) -> Self {
        self.speed_factor = factor.max(0.1); // Minimum 0.1x speed
        self
    }
    
    /// Set duration limits
    pub fn with_duration_limits(mut self, min_ms: f32, max_ms: f32) -> Self {
        self.min_duration_ms = min_ms.max(1.0);
        self.max_duration_ms = max_ms.max(self.min_duration_ms);
        self
    }
    
    /// Add phoneme-specific multiplier
    pub fn with_phoneme_multiplier(mut self, phoneme: String, multiplier: f32) -> Self {
        self.phoneme_multipliers.insert(phoneme, multiplier.max(0.1));
        self
    }
    
    /// Set rhythm pattern
    pub fn with_rhythm_pattern(mut self, pattern: RhythmPattern) -> Self {
        self.rhythm_pattern = pattern;
        self
    }
    
    /// Get default phoneme multipliers
    fn default_phoneme_multipliers() -> HashMap<String, f32> {
        let mut multipliers = HashMap::new();
        
        // Vowels (typically longer)
        for vowel in ["a", "e", "i", "o", "u", "ɑ", "ɛ", "ɪ", "ɔ", "ʊ"] {
            multipliers.insert(vowel.to_string(), 1.2);
        }
        
        // Consonants (typically shorter)
        for consonant in ["p", "t", "k", "b", "d", "g"] {
            multipliers.insert(consonant.to_string(), 0.8);
        }
        
        // Fricatives (medium duration)
        for fricative in ["f", "θ", "s", "ʃ", "v", "ð", "z", "ʒ"] {
            multipliers.insert(fricative.to_string(), 1.0);
        }
        
        // Nasals (medium-long duration)
        for nasal in ["m", "n", "ŋ"] {
            multipliers.insert(nasal.to_string(), 1.1);
        }
        
        multipliers
    }
    
    /// Get default stress multipliers
    fn default_stress_multipliers() -> HashMap<StressLevel, f32> {
        let mut multipliers = HashMap::new();
        multipliers.insert(StressLevel::Unstressed, 0.8);
        multipliers.insert(StressLevel::Secondary, 1.0);
        multipliers.insert(StressLevel::Primary, 1.3);
        multipliers
    }
    
    /// Calculate duration for a phoneme
    pub fn calculate_phoneme_duration(&self, phoneme: &Phoneme, context: &DurationContext) -> f32 {
        // Base duration (in milliseconds)
        let mut duration = context.base_duration_ms;
        
        // Apply speed factor
        duration /= self.speed_factor;
        
        // Apply phoneme-specific multiplier
        if let Some(multiplier) = self.phoneme_multipliers.get(&phoneme.symbol) {
            duration *= multiplier;
        }
        
        // Apply stress-based multiplier
        let stress_level = Self::extract_stress_level(phoneme);
        if let Some(multiplier) = self.stress_multipliers.get(&stress_level) {
            duration *= multiplier;
        }
        
        // Apply position-based adjustments
        duration *= context.position_factor;
        
        // Apply rhythm pattern
        duration *= self.rhythm_pattern.get_timing_factor(context.position_in_phrase);
        
        // Clamp to limits
        duration.clamp(self.min_duration_ms, self.max_duration_ms)
    }
    
    /// Extract stress level from phoneme features
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
    
    /// Apply duration adjustments to phoneme sequence
    pub fn apply_to_phonemes(&self, phonemes: &mut [Phoneme]) -> Result<()> {
        let phonemes_len = phonemes.len();
        
        for (i, phoneme) in phonemes.iter_mut().enumerate() {
            let context = DurationContext {
                base_duration_ms: 100.0, // Default base duration
                position_in_phrase: i as f32 / phonemes_len as f32,
                position_factor: self.calculate_position_factor(i, phonemes_len),
                is_at_boundary: i == 0 || i == phonemes_len - 1,
            };
            
            let calculated_duration = self.calculate_phoneme_duration(phoneme, &context);
            phoneme.duration = Some(calculated_duration / 1000.0); // Convert to seconds
        }
        
        Ok(())
    }
    
    /// Calculate position-based factor
    fn calculate_position_factor(&self, position: usize, total: usize) -> f32 {
        if total <= 1 {
            return 1.0;
        }
        
        let normalized_pos = position as f32 / (total - 1) as f32;
        
        match self.rhythm_pattern {
            RhythmPattern::Natural => {
                // Slight lengthening at phrase boundaries
                if normalized_pos < 0.1 || normalized_pos > 0.9 {
                    1.1
                } else {
                    1.0
                }
            },
            RhythmPattern::Uniform => 1.0,
            RhythmPattern::Accelerando => {
                // Gradually speed up
                1.2 - (normalized_pos * 0.4)
            },
            RhythmPattern::Ritardando => {
                // Gradually slow down
                0.8 + (normalized_pos * 0.4)
            },
        }
    }
    
    /// Validate duration configuration
    pub fn validate(&self) -> Result<()> {
        if self.speed_factor <= 0.0 {
            return Err(AcousticError::ConfigError(
                "Speed factor must be positive".to_string()
            ));
        }
        
        if self.min_duration_ms <= 0.0 {
            return Err(AcousticError::ConfigError(
                "Minimum duration must be positive".to_string()
            ));
        }
        
        if self.max_duration_ms <= self.min_duration_ms {
            return Err(AcousticError::ConfigError(
                "Maximum duration must be greater than minimum duration".to_string()
            ));
        }
        
        Ok(())
    }
    
    /// Create preset configurations
    pub fn natural() -> Self {
        Self::new()
            .with_rhythm_pattern(RhythmPattern::Natural)
    }
    
    pub fn fast() -> Self {
        Self::new()
            .with_speed_factor(1.5)
            .with_duration_limits(15.0, 300.0)
    }
    
    pub fn slow() -> Self {
        Self::new()
            .with_speed_factor(0.7)
            .with_duration_limits(30.0, 800.0)
    }
    
    pub fn uniform() -> Self {
        Self::new()
            .with_rhythm_pattern(RhythmPattern::Uniform)
    }
    
    pub fn expressive() -> Self {
        let mut config = Self::new()
            .with_rhythm_pattern(RhythmPattern::Natural);
        
        // More extreme stress multipliers for expressiveness
        config.stress_multipliers.insert(StressLevel::Unstressed, 0.6);
        config.stress_multipliers.insert(StressLevel::Primary, 1.6);
        
        config
    }
}

impl Default for DurationConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Stress levels for phonemes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StressLevel {
    /// No stress
    Unstressed,
    /// Secondary stress
    Secondary,
    /// Primary stress
    Primary,
}

/// Pause durations for different contexts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PauseDurations {
    /// Comma pause (milliseconds)
    pub comma_ms: f32,
    /// Period pause (milliseconds)
    pub period_ms: f32,
    /// Question mark pause (milliseconds)
    pub question_ms: f32,
    /// Sentence boundary pause (milliseconds)
    pub sentence_ms: f32,
    /// Paragraph boundary pause (milliseconds)
    pub paragraph_ms: f32,
}

impl Default for PauseDurations {
    fn default() -> Self {
        Self {
            comma_ms: 200.0,
            period_ms: 400.0,
            question_ms: 350.0,
            sentence_ms: 500.0,
            paragraph_ms: 800.0,
        }
    }
}

/// Rhythm patterns for speech timing
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum RhythmPattern {
    /// Natural speech rhythm with slight variations
    Natural,
    /// Uniform timing (robotic)
    Uniform,
    /// Gradually speeding up
    Accelerando,
    /// Gradually slowing down
    Ritardando,
}

impl RhythmPattern {
    /// Get timing factor for position in phrase
    pub fn get_timing_factor(&self, position: f32) -> f32 {
        match self {
            RhythmPattern::Natural => {
                // Slight random variation
                0.95 + (fastrand::f32() * 0.1)
            },
            RhythmPattern::Uniform => 1.0,
            RhythmPattern::Accelerando => {
                // Speed up over time
                1.2 - (position * 0.4)
            },
            RhythmPattern::Ritardando => {
                // Slow down over time
                0.8 + (position * 0.4)
            },
        }
    }
    
    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            RhythmPattern::Natural => "natural",
            RhythmPattern::Uniform => "uniform",
            RhythmPattern::Accelerando => "accelerando",
            RhythmPattern::Ritardando => "ritardando",
        }
    }
}

/// Context for duration calculation
#[derive(Debug, Clone)]
pub struct DurationContext {
    /// Base duration in milliseconds
    pub base_duration_ms: f32,
    /// Position in phrase (0.0 - 1.0)
    pub position_in_phrase: f32,
    /// Position-based timing factor
    pub position_factor: f32,
    /// Whether at phrase boundary
    pub is_at_boundary: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_duration_config_creation() {
        let config = DurationConfig::new();
        assert_eq!(config.speed_factor, 1.0);
        assert_eq!(config.min_duration_ms, 20.0);
        assert_eq!(config.max_duration_ms, 500.0);
    }
    
    #[test]
    fn test_duration_config_builder() {
        let config = DurationConfig::new()
            .with_speed_factor(1.5)
            .with_duration_limits(10.0, 300.0);
        
        assert_eq!(config.speed_factor, 1.5);
        assert_eq!(config.min_duration_ms, 10.0);
        assert_eq!(config.max_duration_ms, 300.0);
    }
    
    #[test]
    fn test_duration_config_validation() {
        let valid_config = DurationConfig::new();
        assert!(valid_config.validate().is_ok());
        
        let mut invalid_config = DurationConfig::new();
        invalid_config.speed_factor = 0.0; // Invalid - bypass builder validation
        assert!(invalid_config.validate().is_err());
        
        let invalid_limits = DurationConfig::new()
            .with_duration_limits(100.0, 50.0); // Max < Min
        assert!(invalid_limits.validate().is_err());
    }
    
    #[test]
    fn test_phoneme_duration_calculation() {
        let config = DurationConfig::new();
        let phoneme = Phoneme::new("a"); // Vowel, should have multiplier
        let context = DurationContext {
            base_duration_ms: 100.0,
            position_in_phrase: 0.5,
            position_factor: 1.0,
            is_at_boundary: false,
        };
        
        let duration = config.calculate_phoneme_duration(&phoneme, &context);
        assert!(duration >= config.min_duration_ms);
        assert!(duration <= config.max_duration_ms);
    }
    
    #[test]
    fn test_stress_level_extraction() {
        let mut phoneme = Phoneme::new("a");
        
        // No features - should be unstressed
        assert_eq!(DurationConfig::extract_stress_level(&phoneme), StressLevel::Unstressed);
        
        // Add stress feature
        let mut features = std::collections::HashMap::new();
        features.insert("stress".to_string(), "primary".to_string());
        phoneme.features = Some(features);
        
        assert_eq!(DurationConfig::extract_stress_level(&phoneme), StressLevel::Primary);
    }
    
    #[test]
    fn test_rhythm_patterns() {
        assert_eq!(RhythmPattern::Natural.as_str(), "natural");
        assert_eq!(RhythmPattern::Uniform.as_str(), "uniform");
        
        // Uniform should always return 1.0
        assert_eq!(RhythmPattern::Uniform.get_timing_factor(0.5), 1.0);
    }
    
    #[test]
    fn test_apply_to_phonemes() {
        let config = DurationConfig::new();
        let mut phonemes = vec![
            Phoneme::new("h"),
            Phoneme::new("e"),
            Phoneme::new("l"),
            Phoneme::new("o"),
        ];
        
        assert!(config.apply_to_phonemes(&mut phonemes).is_ok());
        
        // All phonemes should have durations set
        for phoneme in &phonemes {
            assert!(phoneme.duration.is_some());
        }
    }
    
    #[test]
    fn test_preset_configurations() {
        let fast = DurationConfig::fast();
        assert_eq!(fast.speed_factor, 1.5);
        
        let slow = DurationConfig::slow();
        assert_eq!(slow.speed_factor, 0.7);
        
        let uniform = DurationConfig::uniform();
        assert_eq!(uniform.rhythm_pattern, RhythmPattern::Uniform);
    }
    
    #[test]
    fn test_pause_durations() {
        let pauses = PauseDurations::default();
        assert_eq!(pauses.comma_ms, 200.0);
        assert_eq!(pauses.period_ms, 400.0);
        assert!(pauses.paragraph_ms > pauses.sentence_ms);
    }
    
    #[test]
    fn test_position_factor_calculation() {
        let config = DurationConfig::natural();
        
        // First position (boundary)
        let factor_start = config.calculate_position_factor(0, 10);
        assert!(factor_start >= 1.0);
        
        // Last position (boundary)
        let factor_end = config.calculate_position_factor(9, 10);
        assert!(factor_end >= 1.0);
        
        // Middle position
        let factor_middle = config.calculate_position_factor(5, 10);
        assert_eq!(factor_middle, 1.0);
    }
}
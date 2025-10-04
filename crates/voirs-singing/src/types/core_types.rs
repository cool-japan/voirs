//! Core type definitions for singing synthesis

use serde::{Deserialize, Serialize};

/// Voice types for singing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VoiceType {
    /// Soprano voice type
    Soprano,
    /// Mezzo-soprano voice type
    MezzoSoprano,
    /// Alto voice type
    Alto,
    /// Tenor voice type
    Tenor,
    /// Baritone voice type
    Baritone,
    /// Bass voice type
    Bass,
}

/// Expression types for singing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Expression {
    /// Neutral expression
    Neutral,
    /// Happy expression
    Happy,
    /// Sad expression
    Sad,
    /// Angry expression
    Angry,
    /// Excited expression
    Excited,
    /// Calm expression
    Calm,
    /// Passionate expression
    Passionate,
    /// Melancholic expression
    Melancholic,
    /// Playful expression
    Playful,
    /// Dramatic expression
    Dramatic,
}

/// Articulation types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Articulation {
    /// Normal articulation
    Normal,
    /// Staccato (short and detached)
    Staccato,
    /// Legato (smooth and connected)
    Legato,
    /// Tenuto (held full value)
    Tenuto,
    /// Accent (emphasized)
    Accent,
    /// Slur (connected notes)
    Slur,
    /// Glissando (sliding between notes)
    Glissando,
    /// Portamento (continuous pitch change)
    Portamento,
}

/// Dynamics levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Dynamics {
    /// Pianissimo (very soft)
    Pianissimo,
    /// Piano (soft)
    Piano,
    /// Mezzo-piano (moderately soft)
    MezzoPiano,
    /// Mezzo-forte (moderately loud)
    MezzoForte,
    /// Forte (loud)
    Forte,
    /// Fortissimo (very loud)
    Fortissimo,
    /// Crescendo (gradually louder)
    Crescendo,
    /// Diminuendo (gradually softer)
    Diminuendo,
}

/// Pitch bend curve types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BendCurve {
    /// Linear bend
    Linear,
    /// Exponential bend
    Exponential,
    /// Logarithmic bend
    Logarithmic,
    /// Sine wave bend
    Sine,
    /// Custom curve
    Custom,
}

/// Breath types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BreathType {
    /// Natural breath
    Natural,
    /// Quick breath
    Quick,
    /// Deep breath
    Deep,
    /// Gasp
    Gasp,
    /// Sigh
    Sigh,
}

impl VoiceType {
    /// Get typical frequency range for voice type
    ///
    /// # Returns
    ///
    /// A tuple containing the (min, max) frequency range in Hz for this voice type
    pub fn frequency_range(&self) -> (f32, f32) {
        match self {
            VoiceType::Soprano => (261.6, 1046.5),     // C4 to C6
            VoiceType::MezzoSoprano => (220.0, 880.0), // A3 to A5
            VoiceType::Alto => (196.0, 784.0),         // G3 to G5
            VoiceType::Tenor => (146.8, 587.3),        // D3 to D5
            VoiceType::Baritone => (110.0, 440.0),     // A2 to A4
            VoiceType::Bass => (87.3, 349.2),          // F2 to F4
        }
    }

    /// Get typical F0 mean for voice type
    ///
    /// # Returns
    ///
    /// The average fundamental frequency in Hz for this voice type
    pub fn f0_mean(&self) -> f32 {
        match self {
            VoiceType::Soprano => 523.3,      // C5
            VoiceType::MezzoSoprano => 440.0, // A4
            VoiceType::Alto => 349.2,         // F4
            VoiceType::Tenor => 293.7,        // D4
            VoiceType::Baritone => 220.0,     // A3
            VoiceType::Bass => 174.6,         // F3
        }
    }
}

impl Expression {
    /// Get intensity modifier for expression
    ///
    /// # Returns
    ///
    /// A multiplier for intensity/energy level (1.0 = neutral, >1.0 = more intense, <1.0 = less intense)
    pub fn intensity_modifier(&self) -> f32 {
        match self {
            Expression::Neutral => 1.0,
            Expression::Happy => 1.2,
            Expression::Sad => 0.7,
            Expression::Angry => 1.4,
            Expression::Excited => 1.3,
            Expression::Calm => 0.8,
            Expression::Passionate => 1.5,
            Expression::Melancholic => 0.6,
            Expression::Playful => 1.1,
            Expression::Dramatic => 1.6,
        }
    }

    /// Get pitch modifier for expression
    ///
    /// # Returns
    ///
    /// A multiplier for pitch modulation (1.0 = neutral, >1.0 = higher pitch, <1.0 = lower pitch)
    pub fn pitch_modifier(&self) -> f32 {
        match self {
            Expression::Neutral => 1.0,
            Expression::Happy => 1.05,
            Expression::Sad => 0.95,
            Expression::Angry => 1.1,
            Expression::Excited => 1.08,
            Expression::Calm => 0.98,
            Expression::Passionate => 1.12,
            Expression::Melancholic => 0.92,
            Expression::Playful => 1.03,
            Expression::Dramatic => 1.15,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voice_type_ranges() {
        let soprano_range = VoiceType::Soprano.frequency_range();
        let bass_range = VoiceType::Bass.frequency_range();

        assert!(soprano_range.0 > bass_range.0);
        assert!(soprano_range.1 > bass_range.1);
    }

    #[test]
    fn test_expression_modifiers() {
        assert!(Expression::Happy.intensity_modifier() > 1.0);
        assert!(Expression::Sad.intensity_modifier() < 1.0);
        assert!(Expression::Angry.pitch_modifier() > 1.0);
        assert!(Expression::Melancholic.pitch_modifier() < 1.0);
    }
}

//! Voice characteristics for detailed speaker modeling.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::LanguageCode;

/// Age group categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AgeGroup {
    /// Child (0-12 years)
    Child,
    /// Teenager (13-19 years)
    Teenager,
    /// Young adult (20-35 years)
    YoungAdult,
    /// Middle-aged (36-55 years)
    MiddleAged,
    /// Senior (56+ years)
    Senior,
}

impl AgeGroup {
    /// Get typical age range
    pub fn age_range(&self) -> (u8, u8) {
        match self {
            AgeGroup::Child => (0, 12),
            AgeGroup::Teenager => (13, 19),
            AgeGroup::YoungAdult => (20, 35),
            AgeGroup::MiddleAged => (36, 55),
            AgeGroup::Senior => (56, 100),
        }
    }
    
    /// Get from specific age
    pub fn from_age(age: u8) -> Self {
        match age {
            0..=12 => AgeGroup::Child,
            13..=19 => AgeGroup::Teenager,
            20..=35 => AgeGroup::YoungAdult,
            36..=55 => AgeGroup::MiddleAged,
            _ => AgeGroup::Senior,
        }
    }
    
    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            AgeGroup::Child => "child",
            AgeGroup::Teenager => "teenager",
            AgeGroup::YoungAdult => "young_adult",
            AgeGroup::MiddleAged => "middle_aged",
            AgeGroup::Senior => "senior",
        }
    }
}

/// Gender categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Gender {
    /// Male voice
    Male,
    /// Female voice
    Female,
    /// Non-binary/neutral voice
    NonBinary,
    /// Unspecified gender
    Unspecified,
}

impl Gender {
    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            Gender::Male => "male",
            Gender::Female => "female",
            Gender::NonBinary => "non_binary",
            Gender::Unspecified => "unspecified",
        }
    }
}

/// Accent/dialect categories
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Accent {
    /// Standard/neutral accent
    Standard,
    /// Regional accent
    Regional(String),
    /// International accent
    International(String),
    /// Custom accent
    Custom(String),
}

impl Accent {
    /// Get string representation
    pub fn as_str(&self) -> String {
        match self {
            Accent::Standard => "standard".to_string(),
            Accent::Regional(name) => format!("regional_{}", name),
            Accent::International(name) => format!("international_{}", name),
            Accent::Custom(name) => format!("custom_{}", name),
        }
    }
}

/// Voice quality characteristics
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum VoiceQuality {
    /// Clear, professional voice
    Clear,
    /// Warm, friendly voice
    Warm,
    /// Bright, energetic voice
    Bright,
    /// Deep, rich voice
    Deep,
    /// Soft, gentle voice
    Soft,
    /// Rough, textured voice
    Rough,
    /// Breathy voice
    Breathy,
    /// Nasal voice
    Nasal,
    /// Resonant voice
    Resonant,
}

impl VoiceQuality {
    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            VoiceQuality::Clear => "clear",
            VoiceQuality::Warm => "warm",
            VoiceQuality::Bright => "bright",
            VoiceQuality::Deep => "deep",
            VoiceQuality::Soft => "soft",
            VoiceQuality::Rough => "rough",
            VoiceQuality::Breathy => "breathy",
            VoiceQuality::Nasal => "nasal",
            VoiceQuality::Resonant => "resonant",
        }
    }
    
    /// Get all voice qualities
    pub fn all() -> Vec<VoiceQuality> {
        vec![
            VoiceQuality::Clear,
            VoiceQuality::Warm,
            VoiceQuality::Bright,
            VoiceQuality::Deep,
            VoiceQuality::Soft,
            VoiceQuality::Rough,
            VoiceQuality::Breathy,
            VoiceQuality::Nasal,
            VoiceQuality::Resonant,
        ]
    }
}

/// Personality trait categories
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PersonalityTrait {
    /// Extroverted personality
    Extroverted,
    /// Introverted personality
    Introverted,
    /// Confident personality
    Confident,
    /// Shy personality
    Shy,
    /// Energetic personality
    Energetic,
    /// Calm personality
    Calm,
    /// Authoritative personality
    Authoritative,
    /// Friendly personality
    Friendly,
    /// Serious personality
    Serious,
    /// Playful personality
    Playful,
}

impl PersonalityTrait {
    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            PersonalityTrait::Extroverted => "extroverted",
            PersonalityTrait::Introverted => "introverted",
            PersonalityTrait::Confident => "confident",
            PersonalityTrait::Shy => "shy",
            PersonalityTrait::Energetic => "energetic",
            PersonalityTrait::Calm => "calm",
            PersonalityTrait::Authoritative => "authoritative",
            PersonalityTrait::Friendly => "friendly",
            PersonalityTrait::Serious => "serious",
            PersonalityTrait::Playful => "playful",
        }
    }
}

/// Comprehensive voice characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceCharacteristics {
    /// Age group
    pub age_group: AgeGroup,
    /// Specific age (if known)
    pub age: Option<u8>,
    /// Gender
    pub gender: Gender,
    /// Primary accent/dialect
    pub accent: Accent,
    /// Voice quality
    pub voice_quality: VoiceQuality,
    /// Personality traits
    pub personality_traits: Vec<PersonalityTrait>,
    /// Supported languages
    pub supported_languages: Vec<LanguageCode>,
    /// Voice pitch range (Hz)
    pub pitch_range: (f32, f32),
    /// Speaking rate (words per minute)
    pub speaking_rate: f32,
    /// Voice brightness (0.0 - 1.0)
    pub brightness: f32,
    /// Voice warmth (0.0 - 1.0)
    pub warmth: f32,
    /// Voice roughness (0.0 - 1.0)
    pub roughness: f32,
    /// Voice breathiness (0.0 - 1.0)
    pub breathiness: f32,
    /// Custom characteristics
    pub custom_characteristics: HashMap<String, f32>,
}

impl VoiceCharacteristics {
    /// Create new voice characteristics
    pub fn new(age_group: AgeGroup, gender: Gender) -> Self {
        let pitch_range = Self::default_pitch_range(age_group, gender);
        let speaking_rate = Self::default_speaking_rate(age_group);
        
        Self {
            age_group,
            age: None,
            gender,
            accent: Accent::Standard,
            voice_quality: VoiceQuality::Clear,
            personality_traits: Vec::new(),
            supported_languages: vec![LanguageCode::EnUs],
            pitch_range,
            speaking_rate,
            brightness: 0.5,
            warmth: 0.5,
            roughness: 0.0,
            breathiness: 0.0,
            custom_characteristics: HashMap::new(),
        }
    }
    
    /// Set specific age
    pub fn with_age(mut self, age: u8) -> Self {
        self.age = Some(age);
        self.age_group = AgeGroup::from_age(age);
        self
    }
    
    /// Set accent
    pub fn with_accent(mut self, accent: Accent) -> Self {
        self.accent = accent;
        self
    }
    
    /// Set voice quality
    pub fn with_voice_quality(mut self, quality: VoiceQuality) -> Self {
        self.voice_quality = quality;
        self
    }
    
    /// Add personality trait
    pub fn with_personality_trait(mut self, trait_type: PersonalityTrait) -> Self {
        if !self.personality_traits.contains(&trait_type) {
            self.personality_traits.push(trait_type);
        }
        self
    }
    
    /// Set supported languages
    pub fn with_languages(mut self, languages: Vec<LanguageCode>) -> Self {
        self.supported_languages = languages;
        self
    }
    
    /// Set pitch range
    pub fn with_pitch_range(mut self, min_hz: f32, max_hz: f32) -> Self {
        self.pitch_range = (min_hz, max_hz);
        self
    }
    
    /// Set speaking rate
    pub fn with_speaking_rate(mut self, rate: f32) -> Self {
        self.speaking_rate = rate;
        self
    }
    
    /// Set voice brightness
    pub fn with_brightness(mut self, brightness: f32) -> Self {
        self.brightness = brightness.clamp(0.0, 1.0);
        self
    }
    
    /// Set voice warmth
    pub fn with_warmth(mut self, warmth: f32) -> Self {
        self.warmth = warmth.clamp(0.0, 1.0);
        self
    }
    
    /// Set voice roughness
    pub fn with_roughness(mut self, roughness: f32) -> Self {
        self.roughness = roughness.clamp(0.0, 1.0);
        self
    }
    
    /// Set voice breathiness
    pub fn with_breathiness(mut self, breathiness: f32) -> Self {
        self.breathiness = breathiness.clamp(0.0, 1.0);
        self
    }
    
    /// Add custom characteristic
    pub fn with_custom_characteristic(mut self, name: String, value: f32) -> Self {
        self.custom_characteristics.insert(name, value);
        self
    }
    
    /// Get default pitch range for age group and gender
    fn default_pitch_range(age_group: AgeGroup, gender: Gender) -> (f32, f32) {
        match (age_group, gender) {
            (AgeGroup::Child, _) => (250.0, 400.0),
            (AgeGroup::Teenager, Gender::Male) => (180.0, 300.0),
            (AgeGroup::Teenager, Gender::Female) => (200.0, 350.0),
            (AgeGroup::YoungAdult, Gender::Male) => (120.0, 200.0),
            (AgeGroup::YoungAdult, Gender::Female) => (180.0, 280.0),
            (AgeGroup::MiddleAged, Gender::Male) => (110.0, 180.0),
            (AgeGroup::MiddleAged, Gender::Female) => (170.0, 260.0),
            (AgeGroup::Senior, Gender::Male) => (100.0, 170.0),
            (AgeGroup::Senior, Gender::Female) => (160.0, 240.0),
            (_, Gender::NonBinary) => (150.0, 250.0),
            (_, Gender::Unspecified) => (120.0, 250.0),
        }
    }
    
    /// Get default speaking rate for age group
    fn default_speaking_rate(age_group: AgeGroup) -> f32 {
        match age_group {
            AgeGroup::Child => 120.0,
            AgeGroup::Teenager => 140.0,
            AgeGroup::YoungAdult => 150.0,
            AgeGroup::MiddleAged => 145.0,
            AgeGroup::Senior => 130.0,
        }
    }
    
    /// Check if characteristics match a filter
    pub fn matches(&self, filter: &VoiceCharacteristics) -> bool {
        // Age group must match
        if self.age_group != filter.age_group {
            return false;
        }
        
        // Gender must match
        if self.gender != filter.gender {
            return false;
        }
        
        // Voice quality must match
        if self.voice_quality != filter.voice_quality {
            return false;
        }
        
        // At least one personality trait must match (if filter has any)
        if !filter.personality_traits.is_empty() {
            let has_matching_trait = filter.personality_traits.iter()
                .any(|trait_type| self.personality_traits.contains(trait_type));
            if !has_matching_trait {
                return false;
            }
        }
        
        // At least one language must match
        let has_matching_language = filter.supported_languages.iter()
            .any(|lang| self.supported_languages.contains(lang));
        if !has_matching_language {
            return false;
        }
        
        true
    }
    
    /// Get voice characteristics as feature vector
    pub fn to_feature_vector(&self) -> Vec<f32> {
        let mut features = Vec::new();
        
        // Age group (one-hot encoding)
        let age_groups = [
            AgeGroup::Child,
            AgeGroup::Teenager,
            AgeGroup::YoungAdult,
            AgeGroup::MiddleAged,
            AgeGroup::Senior,
        ];
        for age_group in &age_groups {
            features.push(if self.age_group == *age_group { 1.0 } else { 0.0 });
        }
        
        // Gender (one-hot encoding)
        let genders = [Gender::Male, Gender::Female, Gender::NonBinary, Gender::Unspecified];
        for gender in &genders {
            features.push(if self.gender == *gender { 1.0 } else { 0.0 });
        }
        
        // Voice quality (one-hot encoding)
        let qualities = VoiceQuality::all();
        for quality in &qualities {
            features.push(if self.voice_quality == *quality { 1.0 } else { 0.0 });
        }
        
        // Personality traits (multi-hot encoding)
        let all_traits = [
            PersonalityTrait::Extroverted,
            PersonalityTrait::Introverted,
            PersonalityTrait::Confident,
            PersonalityTrait::Shy,
            PersonalityTrait::Energetic,
            PersonalityTrait::Calm,
            PersonalityTrait::Authoritative,
            PersonalityTrait::Friendly,
            PersonalityTrait::Serious,
            PersonalityTrait::Playful,
        ];
        for trait_type in &all_traits {
            features.push(if self.personality_traits.contains(trait_type) { 1.0 } else { 0.0 });
        }
        
        // Continuous features
        features.push(self.pitch_range.0 / 400.0); // Normalized min pitch
        features.push(self.pitch_range.1 / 400.0); // Normalized max pitch
        features.push(self.speaking_rate / 200.0); // Normalized speaking rate
        features.push(self.brightness);
        features.push(self.warmth);
        features.push(self.roughness);
        features.push(self.breathiness);
        
        features
    }
    
    /// Create preset characteristics
    pub fn preset_professional_male() -> Self {
        Self::new(AgeGroup::MiddleAged, Gender::Male)
            .with_voice_quality(VoiceQuality::Clear)
            .with_personality_trait(PersonalityTrait::Confident)
            .with_personality_trait(PersonalityTrait::Authoritative)
            .with_brightness(0.6)
            .with_warmth(0.4)
    }
    
    pub fn preset_friendly_female() -> Self {
        Self::new(AgeGroup::YoungAdult, Gender::Female)
            .with_voice_quality(VoiceQuality::Warm)
            .with_personality_trait(PersonalityTrait::Friendly)
            .with_personality_trait(PersonalityTrait::Energetic)
            .with_brightness(0.7)
            .with_warmth(0.8)
    }
    
    pub fn preset_child() -> Self {
        Self::new(AgeGroup::Child, Gender::Unspecified)
            .with_voice_quality(VoiceQuality::Bright)
            .with_personality_trait(PersonalityTrait::Playful)
            .with_personality_trait(PersonalityTrait::Energetic)
            .with_brightness(0.9)
            .with_warmth(0.6)
    }
    
    pub fn preset_elderly_wise() -> Self {
        Self::new(AgeGroup::Senior, Gender::Male)
            .with_voice_quality(VoiceQuality::Deep)
            .with_personality_trait(PersonalityTrait::Calm)
            .with_personality_trait(PersonalityTrait::Authoritative)
            .with_brightness(0.3)
            .with_warmth(0.7)
            .with_roughness(0.2)
    }
}

impl Default for VoiceCharacteristics {
    fn default() -> Self {
        Self::new(AgeGroup::YoungAdult, Gender::Unspecified)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_age_group_from_age() {
        assert_eq!(AgeGroup::from_age(8), AgeGroup::Child);
        assert_eq!(AgeGroup::from_age(16), AgeGroup::Teenager);
        assert_eq!(AgeGroup::from_age(25), AgeGroup::YoungAdult);
        assert_eq!(AgeGroup::from_age(45), AgeGroup::MiddleAged);
        assert_eq!(AgeGroup::from_age(65), AgeGroup::Senior);
    }
    
    #[test]
    fn test_age_group_range() {
        assert_eq!(AgeGroup::Child.age_range(), (0, 12));
        assert_eq!(AgeGroup::Senior.age_range(), (56, 100));
    }
    
    #[test]
    fn test_voice_characteristics_creation() {
        let characteristics = VoiceCharacteristics::new(AgeGroup::YoungAdult, Gender::Female);
        assert_eq!(characteristics.age_group, AgeGroup::YoungAdult);
        assert_eq!(characteristics.gender, Gender::Female);
        assert_eq!(characteristics.voice_quality, VoiceQuality::Clear);
        assert_eq!(characteristics.pitch_range, (180.0, 280.0));
    }
    
    #[test]
    fn test_voice_characteristics_builder() {
        let characteristics = VoiceCharacteristics::new(AgeGroup::MiddleAged, Gender::Male)
            .with_age(42)
            .with_voice_quality(VoiceQuality::Deep)
            .with_personality_trait(PersonalityTrait::Confident)
            .with_brightness(0.8)
            .with_warmth(0.6);
        
        assert_eq!(characteristics.age, Some(42));
        assert_eq!(characteristics.voice_quality, VoiceQuality::Deep);
        assert!(characteristics.personality_traits.contains(&PersonalityTrait::Confident));
        assert_eq!(characteristics.brightness, 0.8);
        assert_eq!(characteristics.warmth, 0.6);
    }
    
    #[test]
    fn test_voice_characteristics_matching() {
        let characteristics = VoiceCharacteristics::new(AgeGroup::YoungAdult, Gender::Female)
            .with_voice_quality(VoiceQuality::Warm)
            .with_personality_trait(PersonalityTrait::Friendly)
            .with_languages(vec![LanguageCode::EnUs, LanguageCode::Fr]);
        
        let filter = VoiceCharacteristics::new(AgeGroup::YoungAdult, Gender::Female)
            .with_voice_quality(VoiceQuality::Warm)
            .with_personality_trait(PersonalityTrait::Friendly)
            .with_languages(vec![LanguageCode::EnUs]);
        
        assert!(characteristics.matches(&filter));
        
        let non_matching_filter = VoiceCharacteristics::new(AgeGroup::YoungAdult, Gender::Male)
            .with_voice_quality(VoiceQuality::Warm)
            .with_languages(vec![LanguageCode::EnUs]);
        
        assert!(!characteristics.matches(&non_matching_filter));
    }
    
    #[test]
    fn test_voice_characteristics_feature_vector() {
        let characteristics = VoiceCharacteristics::new(AgeGroup::YoungAdult, Gender::Female)
            .with_voice_quality(VoiceQuality::Warm)
            .with_personality_trait(PersonalityTrait::Friendly);
        
        let features = characteristics.to_feature_vector();
        
        // Should have features for age groups, genders, voice qualities, personality traits, and continuous features
        assert!(features.len() > 20);
        
        // YoungAdult should be set (index 2)
        assert_eq!(features[2], 1.0);
        
        // Female should be set (index 6)
        assert_eq!(features[6], 1.0);
    }
    
    #[test]
    fn test_preset_characteristics() {
        let professional = VoiceCharacteristics::preset_professional_male();
        assert_eq!(professional.age_group, AgeGroup::MiddleAged);
        assert_eq!(professional.gender, Gender::Male);
        assert_eq!(professional.voice_quality, VoiceQuality::Clear);
        assert!(professional.personality_traits.contains(&PersonalityTrait::Confident));
        
        let friendly = VoiceCharacteristics::preset_friendly_female();
        assert_eq!(friendly.age_group, AgeGroup::YoungAdult);
        assert_eq!(friendly.gender, Gender::Female);
        assert_eq!(friendly.voice_quality, VoiceQuality::Warm);
        assert!(friendly.personality_traits.contains(&PersonalityTrait::Friendly));
        
        let child = VoiceCharacteristics::preset_child();
        assert_eq!(child.age_group, AgeGroup::Child);
        assert_eq!(child.voice_quality, VoiceQuality::Bright);
        assert!(child.personality_traits.contains(&PersonalityTrait::Playful));
        
        let elderly = VoiceCharacteristics::preset_elderly_wise();
        assert_eq!(elderly.age_group, AgeGroup::Senior);
        assert_eq!(elderly.voice_quality, VoiceQuality::Deep);
        assert!(elderly.personality_traits.contains(&PersonalityTrait::Calm));
    }
    
    #[test]
    fn test_accent_string_representation() {
        assert_eq!(Accent::Standard.as_str(), "standard");
        assert_eq!(Accent::Regional("southern".to_string()).as_str(), "regional_southern");
        assert_eq!(Accent::International("british".to_string()).as_str(), "international_british");
        assert_eq!(Accent::Custom("robot".to_string()).as_str(), "custom_robot");
    }
    
    #[test]
    fn test_voice_quality_all() {
        let qualities = VoiceQuality::all();
        assert_eq!(qualities.len(), 9);
        assert!(qualities.contains(&VoiceQuality::Clear));
        assert!(qualities.contains(&VoiceQuality::Warm));
        assert!(qualities.contains(&VoiceQuality::Deep));
    }
    
    #[test]
    fn test_custom_characteristics() {
        let mut characteristics = VoiceCharacteristics::default();
        characteristics = characteristics.with_custom_characteristic("naturalness".to_string(), 0.9);
        characteristics = characteristics.with_custom_characteristic("expressiveness".to_string(), 0.7);
        
        assert_eq!(characteristics.custom_characteristics.get("naturalness"), Some(&0.9));
        assert_eq!(characteristics.custom_characteristics.get("expressiveness"), Some(&0.7));
    }
}
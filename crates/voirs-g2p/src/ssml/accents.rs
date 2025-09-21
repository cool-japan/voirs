//! Regional accent modifications for SSML pronunciation control.

use crate::{G2pError, LanguageCode, Phoneme, Result, SyllablePosition};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Regional accent system for pronunciation modifications
pub struct AccentSystem {
    /// Accent profiles by region
    accents: HashMap<String, AccentProfile>,
    /// Active accent
    active_accent: Option<String>,
    /// Accent modification rules
    _rules: Vec<AccentRule>,
    /// Phoneme mappings cache
    mappings_cache: HashMap<String, Vec<PhonemeMapping>>,
}

/// Accent profile defining pronunciation characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccentProfile {
    /// Accent name
    pub name: String,
    /// Base language
    pub language: LanguageCode,
    /// Regional identifier
    pub region: String,
    /// Accent description
    pub description: String,
    /// Phoneme substitution rules
    pub substitutions: Vec<PhonemeSubstitution>,
    /// Prosodic modifications
    pub prosody: AccentProsody,
    /// Statistical data
    pub statistics: AccentStatistics,
    /// Metadata
    pub metadata: AccentMetadata,
}

/// Phoneme substitution rule for accents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhonemeSubstitution {
    /// Source phoneme pattern
    pub source: PhonemePattern,
    /// Target phoneme(s)
    pub target: Vec<Phoneme>,
    /// Substitution context
    pub context: SubstitutionContext,
    /// Application probability (0.0-1.0)
    pub probability: f32,
    /// Strength of accent feature (0.0-1.0)
    pub strength: f32,
}

/// Phoneme pattern for matching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhonemePattern {
    /// Target phoneme symbol
    pub phoneme: String,
    /// Preceding context
    pub preceding: Option<String>,
    /// Following context
    pub following: Option<String>,
    /// Syllable position requirement
    pub syllable_position: Option<SyllablePosition>,
    /// Stress requirement
    pub stress: Option<u8>,
    /// Word position requirement
    pub word_position: Option<WordPosition>,
}

/// Substitution context
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SubstitutionContext {
    /// Word types where this applies
    pub word_types: Option<Vec<WordType>>,
    /// Speaking styles where this applies
    pub speaking_styles: Option<Vec<SpeakingStyle>>,
    /// Formality levels where this applies
    pub formality: Option<Vec<FormalityLevel>>,
    /// Emotional contexts where this applies
    pub emotions: Option<Vec<EmotionContext>>,
}

/// Prosodic modifications for accents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccentProsody {
    /// Rhythm modifications
    pub rhythm: RhythmModification,
    /// Intonation patterns
    pub intonation: IntonationModification,
    /// Stress patterns
    pub stress: StressModification,
    /// Duration modifications
    pub duration: DurationModification,
    /// Pitch characteristics
    pub pitch: PitchModification,
}

/// Accent statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AccentStatistics {
    /// Number of phoneme rules
    pub phoneme_rules: usize,
    /// Most frequent substitutions
    pub frequent_substitutions: Vec<(String, String, f32)>,
    /// Accent strength distribution
    pub strength_distribution: Vec<f32>,
    /// Usage frequency by context
    pub context_usage: HashMap<String, f32>,
}

/// Accent metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccentMetadata {
    /// Source/authority for accent data
    pub source: Option<String>,
    /// Confidence in accent accuracy
    pub confidence: f32,
    /// Date created/updated
    pub created: String,
    /// Geographic coverage
    pub geographic_scope: GeographicScope,
    /// Social/demographic information
    pub demographics: Demographics,
}

/// Accent rule for systematic modifications
#[derive(Debug, Clone)]
pub struct AccentRule {
    /// Rule name
    pub name: String,
    /// Source accent
    pub source_accent: Option<String>,
    /// Target accent
    pub target_accent: String,
    /// Transformation function
    pub transformation: AccentTransformation,
    /// Rule priority
    pub priority: u32,
    /// Confidence level
    pub confidence: f32,
}

/// Accent transformation definition
#[derive(Debug, Clone)]
pub enum AccentTransformation {
    /// Simple phoneme mapping
    PhonemeMapping(HashMap<String, String>),
    /// Context-sensitive transformation
    ContextualMapping(Vec<ContextualTransform>),
    /// Prosodic transformation
    ProsodicTransform(ProsodicTransform),
    /// Complex rule-based transformation
    RuleBased(Vec<TransformRule>),
}

/// Contextual transformation
#[derive(Debug, Clone)]
pub struct ContextualTransform {
    /// Matching context
    pub context: TransformContext,
    /// Phoneme mappings for this context
    pub mappings: HashMap<String, String>,
    /// Confidence of this transformation
    pub confidence: f32,
}

/// Transformation context
#[derive(Debug, Clone)]
pub struct TransformContext {
    /// Phonetic environment
    pub phonetic: Option<PhoneticEnvironment>,
    /// Lexical context
    pub lexical: Option<LexicalContext>,
    /// Syntactic context
    pub syntactic: Option<SyntacticContext>,
    /// Pragmatic context
    pub pragmatic: Option<PragmaticContext>,
}

/// Phoneme mapping with context
#[derive(Debug, Clone)]
pub struct PhonemeMapping {
    /// Source phoneme
    pub source: String,
    /// Target phoneme
    pub target: String,
    /// Application context
    pub context: String,
    /// Confidence score
    pub confidence: f32,
}

// Supporting enums and structs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WordPosition {
    Initial,
    Medial,
    Final,
    Standalone,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WordType {
    ContentWord,
    FunctionWord,
    ProperNoun,
    Loanword,
    Colloquial,
    Technical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpeakingStyle {
    Casual,
    Formal,
    Emphatic,
    Rapid,
    Careful,
    Emotional,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FormalityLevel {
    VeryFormal,
    Formal,
    Neutral,
    Informal,
    VeryInformal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmotionContext {
    Neutral,
    Happy,
    Sad,
    Angry,
    Excited,
    Calm,
    Surprised,
    Disgusted,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RhythmModification {
    /// Tempo adjustment factor
    pub tempo_factor: f32,
    /// Stress-timed vs syllable-timed tendency
    pub timing_type: TimingType,
    /// Rhythm regularity
    pub regularity: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntonationModification {
    /// Overall pitch range
    pub pitch_range: f32,
    /// Question intonation pattern
    pub question_pattern: IntonationPattern,
    /// Statement intonation pattern
    pub statement_pattern: IntonationPattern,
    /// Emphasis pattern
    pub emphasis_pattern: IntonationPattern,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressModification {
    /// Primary stress strength
    pub primary_strength: f32,
    /// Secondary stress frequency
    pub secondary_frequency: f32,
    /// Stress shift tendencies
    pub stress_shifts: Vec<StressShift>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DurationModification {
    /// Overall speaking rate
    pub speaking_rate: f32,
    /// Vowel lengthening in stressed syllables
    pub vowel_lengthening: f32,
    /// Final syllable lengthening
    pub final_lengthening: f32,
    /// Pause insertion tendencies
    pub pause_patterns: Vec<PausePattern>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PitchModification {
    /// Base frequency adjustment
    pub base_frequency: f32,
    /// Pitch contour tendencies
    pub contour_patterns: Vec<PitchContour>,
    /// Vocal fry tendency
    pub vocal_fry: f32,
    /// Uptalk/high rising terminal
    pub uptalk: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeographicScope {
    /// Country/region
    pub region: String,
    /// More specific location
    pub subregion: Option<String>,
    /// Urban/rural classification
    pub area_type: Option<AreaType>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Demographics {
    /// Age group predominance
    pub age_groups: Vec<AgeGroup>,
    /// Social class associations
    pub social_class: Option<SocialClass>,
    /// Education level associations
    pub education: Option<EducationLevel>,
    /// Gender associations
    pub gender: Option<GenderAssociation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimingType {
    StressTimed,
    SyllableTimed,
    Mixed(f32), // Ratio of stress-timed vs syllable-timed
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IntonationPattern {
    Rising,
    Falling,
    RisingFalling,
    FallingRising,
    Flat,
    Complex(Vec<PitchMovement>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressShift {
    /// Source stress pattern
    pub source_pattern: String,
    /// Target stress pattern
    pub target_pattern: String,
    /// Probability of shift
    pub probability: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PausePattern {
    /// Location of pause
    pub location: PauseLocation,
    /// Duration modification
    pub duration_factor: f32,
    /// Frequency of occurrence
    pub frequency: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PitchContour {
    /// Contour type
    pub contour_type: ContourType,
    /// Frequency of use
    pub frequency: f32,
    /// Strength of the pattern
    pub strength: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AreaType {
    Urban,
    Suburban,
    Rural,
    Metropolitan,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgeGroup {
    Children,
    Teenagers,
    YoungAdults,
    MiddleAged,
    Elderly,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SocialClass {
    WorkingClass,
    LowerMiddle,
    MiddleClass,
    UpperMiddle,
    UpperClass,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EducationLevel {
    Primary,
    Secondary,
    Tertiary,
    Postgraduate,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GenderAssociation {
    Masculine,
    Feminine,
    Neutral,
    Mixed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PitchMovement {
    Rise(f32),    // Semitones
    Fall(f32),    // Semitones
    Plateau(f32), // Duration
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PauseLocation {
    WordBoundary,
    PhraseBoundary,
    ClauseBoundary,
    SentenceBoundary,
    ParagraphBoundary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContourType {
    Declarative,
    Interrogative,
    Imperative,
    Exclamatory,
    Parenthetical,
}

// Additional transformation types
#[derive(Debug, Clone)]
pub struct ProsodicTransform {
    /// Rhythm modifications
    pub rhythm: Option<RhythmTransform>,
    /// Pitch modifications
    pub pitch: Option<PitchTransform>,
    /// Duration modifications
    pub duration: Option<DurationTransform>,
}

#[derive(Debug, Clone)]
pub struct TransformRule {
    /// Rule condition
    pub condition: RuleCondition,
    /// Rule action
    pub action: RuleAction,
    /// Rule weight
    pub weight: f32,
}

#[derive(Debug, Clone)]
pub enum RuleCondition {
    PhonemeIs(String),
    PhonemeIn(Vec<String>),
    ContextMatches(String),
    PositionIs(WordPosition),
    StressIs(u8),
}

#[derive(Debug, Clone)]
pub enum RuleAction {
    Replace(String),
    Modify(PhonemeModification),
    Insert(String),
    Delete,
}

#[derive(Debug, Clone)]
pub struct PhonemeModification {
    /// Acoustic modifications
    pub acoustic: Option<AcousticModification>,
    /// Articulatory modifications
    pub articulatory: Option<ArticulatoryModification>,
}

#[derive(Debug, Clone)]
pub struct AcousticModification {
    /// Formant modifications
    pub formants: Option<Vec<FormantModification>>,
    /// Fundamental frequency modification
    pub f0_modification: Option<f32>,
    /// Duration modification
    pub duration_factor: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct ArticulatoryModification {
    /// Place of articulation shift
    pub place_shift: Option<PlaceShift>,
    /// Manner of articulation modification
    pub manner_modification: Option<MannerModification>,
    /// Voicing modification
    pub voicing_change: Option<bool>,
}

#[derive(Debug, Clone)]
pub struct FormantModification {
    /// Formant number (F1, F2, F3, etc.)
    pub formant: u32,
    /// Frequency modification (Hz)
    pub frequency_shift: f32,
    /// Bandwidth modification
    pub bandwidth_factor: f32,
}

#[derive(Debug, Clone)]
pub enum PlaceShift {
    Forward,
    Backward,
    Higher,
    Lower,
}

#[derive(Debug, Clone)]
pub enum MannerModification {
    MoreFrication,
    LessFriction,
    MoreAspiration,
    LessAspiration,
    Lengthen,
    Shorten,
}

// Context definitions
#[derive(Debug, Clone)]
pub struct PhoneticEnvironment {
    /// Preceding phonemes
    pub preceding: Option<Vec<String>>,
    /// Following phonemes
    pub following: Option<Vec<String>>,
    /// Syllable structure
    pub syllable_structure: Option<String>,
}

#[derive(Debug, Clone)]
pub struct LexicalContext {
    /// Word frequency
    pub frequency: Option<LexicalFrequency>,
    /// Word type
    pub word_type: Option<WordType>,
    /// Etymology
    pub etymology: Option<String>,
}

#[derive(Debug, Clone)]
pub struct SyntacticContext {
    /// Part of speech
    pub pos: Option<String>,
    /// Syntactic position
    pub position: Option<String>,
    /// Phrase type
    pub phrase_type: Option<String>,
}

#[derive(Debug, Clone)]
pub struct PragmaticContext {
    /// Speech act type
    pub speech_act: Option<String>,
    /// Formality level
    pub formality: Option<FormalityLevel>,
    /// Emotional content
    pub emotion: Option<EmotionContext>,
}

#[derive(Debug, Clone)]
pub enum LexicalFrequency {
    VeryHigh,
    High,
    Medium,
    Low,
    VeryLow,
}

// Transform types
#[derive(Debug, Clone)]
pub struct RhythmTransform {
    /// Tempo modification
    pub tempo_factor: f32,
    /// Stress pattern changes
    pub stress_patterns: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct PitchTransform {
    /// Base frequency shift
    pub base_shift: f32,
    /// Range modification
    pub range_factor: f32,
    /// Contour modifications
    pub contour_changes: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct DurationTransform {
    /// Overall duration factor
    pub duration_factor: f32,
    /// Specific phoneme duration changes
    pub phoneme_durations: HashMap<String, f32>,
}

impl AccentSystem {
    /// Create a new accent system
    pub fn new() -> Self {
        let mut system = Self {
            accents: HashMap::new(),
            active_accent: None,
            _rules: Vec::new(),
            mappings_cache: HashMap::new(),
        };

        system.load_default_accents();
        system
    }

    /// Load a regional accent profile
    pub fn load_accent(&mut self, accent: AccentProfile) {
        self.accents.insert(accent.name.clone(), accent);
    }

    /// Set active accent
    pub fn set_active_accent(&mut self, accent_name: &str) -> Result<()> {
        if self.accents.contains_key(accent_name) {
            self.active_accent = Some(accent_name.to_string());
            self.clear_cache();
            Ok(())
        } else {
            Err(G2pError::ConfigError(format!(
                "Accent '{accent_name}' not found"
            )))
        }
    }

    /// Apply accent modifications to phonemes
    pub fn apply_accent(
        &mut self,
        phonemes: &[Phoneme],
        context: Option<&str>,
    ) -> Result<Vec<Phoneme>> {
        if let Some(accent_name) = &self.active_accent.clone() {
            if let Some(accent) = self.accents.get(accent_name).cloned() {
                self.apply_accent_profile(phonemes, &accent, context)
            } else {
                Ok(phonemes.to_vec())
            }
        } else {
            Ok(phonemes.to_vec())
        }
    }

    /// Apply specific accent profile
    fn apply_accent_profile(
        &mut self,
        phonemes: &[Phoneme],
        accent: &AccentProfile,
        context: Option<&str>,
    ) -> Result<Vec<Phoneme>> {
        let mut result = Vec::new();

        for (i, phoneme) in phonemes.iter().enumerate() {
            let preceding = if i > 0 { Some(&phonemes[i - 1]) } else { None };
            let following = if i < phonemes.len() - 1 {
                Some(&phonemes[i + 1])
            } else {
                None
            };

            let modified = self.apply_substitutions(
                phoneme,
                preceding,
                following,
                &accent.substitutions,
                context,
            )?;
            result.push(modified);
        }

        Ok(result)
    }

    /// Apply phoneme substitutions
    fn apply_substitutions(
        &self,
        phoneme: &Phoneme,
        preceding: Option<&Phoneme>,
        following: Option<&Phoneme>,
        substitutions: &[PhonemeSubstitution],
        context: Option<&str>,
    ) -> Result<Phoneme> {
        for substitution in substitutions {
            if self.matches_pattern(&substitution.source, phoneme, preceding, following)?
                && self.matches_context(&substitution.context, context)?
            {
                // Apply substitution with probability
                if rand::random::<f32>() < substitution.probability {
                    if let Some(target_phoneme) = substitution.target.first() {
                        let mut modified = target_phoneme.clone();
                        // Preserve some original properties
                        modified.stress = phoneme.stress;
                        modified.syllable_position = phoneme.syllable_position.clone();
                        return Ok(modified);
                    }
                }
            }
        }

        Ok(phoneme.clone())
    }

    /// Check if phoneme matches pattern
    fn matches_pattern(
        &self,
        pattern: &PhonemePattern,
        phoneme: &Phoneme,
        preceding: Option<&Phoneme>,
        following: Option<&Phoneme>,
    ) -> Result<bool> {
        // Check main phoneme
        if phoneme.symbol != pattern.phoneme {
            return Ok(false);
        }

        // Check preceding context
        if let Some(prec_pattern) = &pattern.preceding {
            if let Some(prec_phoneme) = preceding {
                if prec_phoneme.symbol != *prec_pattern {
                    return Ok(false);
                }
            } else {
                return Ok(false);
            }
        }

        // Check following context
        if let Some(foll_pattern) = &pattern.following {
            if let Some(foll_phoneme) = following {
                if foll_phoneme.symbol != *foll_pattern {
                    return Ok(false);
                }
            } else {
                return Ok(false);
            }
        }

        // Check syllable position
        if let Some(req_pos) = &pattern.syllable_position {
            if phoneme.syllable_position != *req_pos {
                return Ok(false);
            }
        }

        // Check stress
        if let Some(req_stress) = pattern.stress {
            if phoneme.stress != req_stress {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Check if context matches substitution context
    fn matches_context(
        &self,
        subst_context: &SubstitutionContext,
        _context: Option<&str>,
    ) -> Result<bool> {
        // Simplified context matching - would be more sophisticated in practice
        // For now, just return true if no specific context requirements
        if subst_context.word_types.is_none()
            && subst_context.speaking_styles.is_none()
            && subst_context.formality.is_none()
            && subst_context.emotions.is_none()
        {
            return Ok(true);
        }

        // Would need more sophisticated context analysis
        Ok(true)
    }

    /// Get available accents
    pub fn get_available_accents(&self) -> Vec<String> {
        self.accents.keys().cloned().collect()
    }

    /// Get accent information
    pub fn get_accent_info(&self, accent_name: &str) -> Option<&AccentProfile> {
        self.accents.get(accent_name)
    }

    /// Clear mappings cache
    fn clear_cache(&mut self) {
        self.mappings_cache.clear();
    }

    /// Load default accent profiles
    fn load_default_accents(&mut self) {
        // American English accent
        self.load_accent(self.create_american_english_accent());

        // British English accent
        self.load_accent(self.create_british_english_accent());

        // Australian English accent
        self.load_accent(self.create_australian_english_accent());
    }

    /// Create American English accent profile
    fn create_american_english_accent(&self) -> AccentProfile {
        AccentProfile {
            name: "American English".to_string(),
            language: LanguageCode::EnUs,
            region: "United States".to_string(),
            description: "General American English pronunciation".to_string(),
            substitutions: vec![
                // R-colored vowels
                PhonemeSubstitution {
                    source: PhonemePattern {
                        phoneme: "ɑː".to_string(),
                        preceding: None,
                        following: Some("r".to_string()),
                        syllable_position: None,
                        stress: None,
                        word_position: None,
                    },
                    target: vec![self.create_test_phoneme("ɑr")],
                    context: SubstitutionContext::default(),
                    probability: 0.9,
                    strength: 0.8,
                },
                // Rhotic /r/
                PhonemeSubstitution {
                    source: PhonemePattern {
                        phoneme: "r".to_string(),
                        preceding: None,
                        following: None,
                        syllable_position: Some(SyllablePosition::Final),
                        stress: None,
                        word_position: None,
                    },
                    target: vec![self.create_test_phoneme("ɹ")],
                    context: SubstitutionContext::default(),
                    probability: 0.95,
                    strength: 0.9,
                },
            ],
            prosody: AccentProsody::default(),
            statistics: AccentStatistics::default(),
            metadata: AccentMetadata::default(),
        }
    }

    /// Create British English accent profile
    fn create_british_english_accent(&self) -> AccentProfile {
        AccentProfile {
            name: "British English".to_string(),
            language: LanguageCode::EnGb,
            region: "United Kingdom".to_string(),
            description: "Received Pronunciation (RP) British English".to_string(),
            substitutions: vec![
                // Non-rhotic /r/
                PhonemeSubstitution {
                    source: PhonemePattern {
                        phoneme: "r".to_string(),
                        preceding: Some("ɑː".to_string()),
                        following: None,
                        syllable_position: Some(SyllablePosition::Final),
                        stress: None,
                        word_position: None,
                    },
                    target: vec![], // Delete the /r/
                    context: SubstitutionContext::default(),
                    probability: 0.9,
                    strength: 0.8,
                },
                // TRAP-BATH split
                PhonemeSubstitution {
                    source: PhonemePattern {
                        phoneme: "æ".to_string(),
                        preceding: None,
                        following: Some("s".to_string()),
                        syllable_position: None,
                        stress: None,
                        word_position: None,
                    },
                    target: vec![self.create_test_phoneme("ɑː")],
                    context: SubstitutionContext::default(),
                    probability: 0.8,
                    strength: 0.7,
                },
            ],
            prosody: AccentProsody::default(),
            statistics: AccentStatistics::default(),
            metadata: AccentMetadata::default(),
        }
    }

    /// Create Australian English accent profile
    fn create_australian_english_accent(&self) -> AccentProfile {
        AccentProfile {
            name: "Australian English".to_string(),
            language: LanguageCode::EnUs, // Using EnUs as base
            region: "Australia".to_string(),
            description: "General Australian English pronunciation".to_string(),
            substitutions: vec![
                // FACE vowel fronting
                PhonemeSubstitution {
                    source: PhonemePattern {
                        phoneme: "eɪ".to_string(),
                        preceding: None,
                        following: None,
                        syllable_position: None,
                        stress: None,
                        word_position: None,
                    },
                    target: vec![self.create_test_phoneme("æɪ")],
                    context: SubstitutionContext::default(),
                    probability: 0.8,
                    strength: 0.7,
                },
                // PRICE vowel
                PhonemeSubstitution {
                    source: PhonemePattern {
                        phoneme: "aɪ".to_string(),
                        preceding: None,
                        following: None,
                        syllable_position: None,
                        stress: None,
                        word_position: None,
                    },
                    target: vec![self.create_test_phoneme("ɐɪ")],
                    context: SubstitutionContext::default(),
                    probability: 0.85,
                    strength: 0.8,
                },
            ],
            prosody: AccentProsody::default(),
            statistics: AccentStatistics::default(),
            metadata: AccentMetadata::default(),
        }
    }

    /// Helper method to create test phonemes
    fn create_test_phoneme(&self, symbol: &str) -> Phoneme {
        Phoneme {
            symbol: symbol.to_string(),
            ipa_symbol: Some(symbol.to_string()),
            language_notation: None,
            stress: 0,
            syllable_position: SyllablePosition::Standalone,
            duration_ms: None,
            confidence: 1.0,
            phonetic_features: None,
            custom_features: None,
            is_word_boundary: false,
            is_syllable_boundary: false,
        }
    }
}

impl Default for AccentSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for AccentProsody {
    fn default() -> Self {
        Self {
            rhythm: RhythmModification {
                tempo_factor: 1.0,
                timing_type: TimingType::Mixed(0.5),
                regularity: 0.5,
            },
            intonation: IntonationModification {
                pitch_range: 1.0,
                question_pattern: IntonationPattern::Rising,
                statement_pattern: IntonationPattern::Falling,
                emphasis_pattern: IntonationPattern::RisingFalling,
            },
            stress: StressModification {
                primary_strength: 1.0,
                secondary_frequency: 0.5,
                stress_shifts: Vec::new(),
            },
            duration: DurationModification {
                speaking_rate: 1.0,
                vowel_lengthening: 1.0,
                final_lengthening: 1.0,
                pause_patterns: Vec::new(),
            },
            pitch: PitchModification {
                base_frequency: 1.0,
                contour_patterns: Vec::new(),
                vocal_fry: 0.0,
                uptalk: 0.0,
            },
        }
    }
}

impl Default for AccentMetadata {
    fn default() -> Self {
        Self {
            source: None,
            confidence: 0.5,
            created: chrono::Utc::now().to_rfc3339(),
            geographic_scope: GeographicScope {
                region: "Unknown".to_string(),
                subregion: None,
                area_type: None,
            },
            demographics: Demographics {
                age_groups: Vec::new(),
                social_class: None,
                education: None,
                gender: None,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_phoneme(symbol: &str) -> Phoneme {
        Phoneme {
            symbol: symbol.to_string(),
            ipa_symbol: Some(symbol.to_string()),
            language_notation: None,
            stress: 0,
            syllable_position: SyllablePosition::Standalone,
            duration_ms: None,
            confidence: 1.0,
            phonetic_features: None,
            custom_features: None,
            is_word_boundary: false,
            is_syllable_boundary: false,
        }
    }

    #[test]
    fn test_accent_system_creation() {
        let system = AccentSystem::new();
        assert!(!system.accents.is_empty());
        assert!(system.get_available_accents().len() >= 3); // At least American, British, Australian
    }

    #[test]
    fn test_set_active_accent() {
        let mut system = AccentSystem::new();

        assert!(system.set_active_accent("American English").is_ok());
        assert_eq!(system.active_accent, Some("American English".to_string()));

        assert!(system.set_active_accent("Nonexistent Accent").is_err());
    }

    #[test]
    fn test_accent_application() {
        let mut system = AccentSystem::new();
        system.set_active_accent("American English").unwrap();

        let phonemes = vec![create_test_phoneme("ɑː"), create_test_phoneme("r")];

        let result = system.apply_accent(&phonemes, None).unwrap();
        // Should apply some accent modifications
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_phoneme_pattern_matching() {
        let system = AccentSystem::new();
        let pattern = PhonemePattern {
            phoneme: "r".to_string(),
            preceding: None,
            following: None,
            syllable_position: Some(SyllablePosition::Final),
            stress: None,
            word_position: None,
        };

        let phoneme = Phoneme {
            symbol: "r".to_string(),
            syllable_position: SyllablePosition::Final,
            ..create_test_phoneme("r")
        };

        assert!(system
            .matches_pattern(&pattern, &phoneme, None, None)
            .unwrap());
    }

    #[test]
    fn test_get_accent_info() {
        let system = AccentSystem::new();

        let info = system.get_accent_info("American English");
        assert!(info.is_some());
        assert_eq!(info.unwrap().language, LanguageCode::EnUs);

        let nonexistent = system.get_accent_info("Nonexistent");
        assert!(nonexistent.is_none());
    }
}

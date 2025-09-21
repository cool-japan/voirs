//! Universal phoneme mapping system for cross-linguistic evaluation
//!
//! This module provides comprehensive cross-linguistic phoneme mapping capabilities including:
//! - IPA-based universal phoneme representation
//! - Cross-language phoneme similarity matrices
//! - Phonetic feature-based distance calculations
//! - Language-specific phoneme inventory mapping
//! - Articulatory feature analysis for cross-linguistic comparison

use crate::traits::{EvaluationResult, QualityScore};
use crate::EvaluationError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use voirs_sdk::{LanguageCode, Phoneme};

/// Universal phoneme mapping configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalPhonemeMappingConfig {
    /// Enable articulatory feature analysis
    pub enable_articulatory_features: bool,
    /// Enable acoustic similarity modeling
    pub enable_acoustic_similarity: bool,
    /// Enable perceptual distance calculation
    pub enable_perceptual_distance: bool,
    /// Similarity threshold for phoneme matching
    pub similarity_threshold: f32,
    /// Weight for articulatory features in similarity calculation
    pub articulatory_weight: f32,
    /// Weight for acoustic features in similarity calculation
    pub acoustic_weight: f32,
    /// Weight for perceptual features in similarity calculation
    pub perceptual_weight: f32,
}

impl Default for UniversalPhonemeMappingConfig {
    fn default() -> Self {
        Self {
            enable_articulatory_features: true,
            enable_acoustic_similarity: true,
            enable_perceptual_distance: true,
            similarity_threshold: 0.7,
            articulatory_weight: 0.4,
            acoustic_weight: 0.3,
            perceptual_weight: 0.3,
        }
    }
}

/// Universal phoneme representation with comprehensive features
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UniversalPhoneme {
    /// IPA symbol
    pub ipa_symbol: String,
    /// Articulatory features
    pub articulatory_features: ArticulatoryFeatures,
    /// Acoustic features
    pub acoustic_features: AcousticFeatures,
    /// Perceptual features
    pub perceptual_features: PerceptualFeatures,
    /// Language-specific variations
    pub language_variations: HashMap<LanguageCode, String>,
    /// Phoneme frequency across languages
    pub language_frequency: HashMap<LanguageCode, f32>,
}

/// Articulatory features for phoneme classification
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArticulatoryFeatures {
    /// Manner of articulation
    pub manner: MannerOfArticulation,
    /// Place of articulation
    pub place: PlaceOfArticulation,
    /// Voicing
    pub voicing: Voicing,
    /// Vowel features (if applicable)
    pub vowel_features: Option<VowelFeatures>,
    /// Consonant features (if applicable)
    pub consonant_features: Option<ConsonantFeatures>,
    /// Airstream mechanism
    pub airstream: AirstreamMechanism,
}

/// Manner of articulation categories
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MannerOfArticulation {
    /// Stops/Plosives
    Stop,
    /// Fricatives
    Fricative,
    /// Affricates
    Affricate,
    /// Nasals
    Nasal,
    /// Liquids
    Liquid,
    /// Approximants
    Approximant,
    /// Vowels
    Vowel,
    /// Taps/Flaps
    Tap,
    /// Trills
    Trill,
    /// Lateral
    Lateral,
}

/// Place of articulation categories
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PlaceOfArticulation {
    /// Bilabial
    Bilabial,
    /// Labiodental
    Labiodental,
    /// Dental
    Dental,
    /// Alveolar
    Alveolar,
    /// Postalveolar
    Postalveolar,
    /// Retroflex
    Retroflex,
    /// Palatal
    Palatal,
    /// Velar
    Velar,
    /// Uvular
    Uvular,
    /// Pharyngeal
    Pharyngeal,
    /// Glottal
    Glottal,
    /// Labiovelar
    Labiovelar,
}

/// Voicing categories
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Voicing {
    /// Voiced
    Voiced,
    /// Voiceless
    Voiceless,
    /// Creaky
    Creaky,
    /// Breathy
    Breathy,
}

/// Vowel-specific features
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VowelFeatures {
    /// Vowel height
    pub height: VowelHeight,
    /// Vowel backness
    pub backness: VowelBackness,
    /// Lip rounding
    pub roundness: VowelRoundness,
    /// Tenseness
    pub tenseness: VowelTenseness,
    /// Nasalization
    pub nasalization: bool,
    /// Length
    pub length: VowelLength,
}

/// Vowel height categories
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum VowelHeight {
    /// High/Close
    High,
    /// Near-high/Near-close
    NearHigh,
    /// High-mid/Close-mid
    HighMid,
    /// Mid
    Mid,
    /// Low-mid/Open-mid
    LowMid,
    /// Near-low/Near-open
    NearLow,
    /// Low/Open
    Low,
}

/// Vowel backness categories
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum VowelBackness {
    /// Front
    Front,
    /// Near-front
    NearFront,
    /// Central
    Central,
    /// Near-back
    NearBack,
    /// Back
    Back,
}

/// Vowel roundness categories
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum VowelRoundness {
    /// Rounded
    Rounded,
    /// Unrounded
    Unrounded,
    /// Compressed
    Compressed,
    /// Protruded
    Protruded,
}

/// Vowel tenseness categories
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum VowelTenseness {
    /// Tense
    Tense,
    /// Lax
    Lax,
}

/// Vowel length categories
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum VowelLength {
    /// Short
    Short,
    /// Long
    Long,
    /// Half-long
    HalfLong,
}

/// Consonant-specific features
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConsonantFeatures {
    /// Aspiration
    pub aspiration: bool,
    /// Palatalization
    pub palatalization: bool,
    /// Velarization
    pub velarization: bool,
    /// Pharyngealization
    pub pharyngealization: bool,
    /// Labialization
    pub labialization: bool,
    /// Glottalization
    pub glottalization: bool,
    /// Gemination
    pub gemination: bool,
}

/// Airstream mechanism categories
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AirstreamMechanism {
    /// Pulmonic egressive
    Pulmonic,
    /// Ejective
    Ejective,
    /// Implosive
    Implosive,
    /// Click
    Click,
}

/// Acoustic features for phoneme analysis
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AcousticFeatures {
    /// Formant frequencies (F1, F2, F3) for vowels
    pub formant_frequencies: Option<Vec<f32>>,
    /// Voice onset time for consonants
    pub voice_onset_time: Option<f32>,
    /// Spectral centroid
    pub spectral_centroid: Option<f32>,
    /// Spectral tilt
    pub spectral_tilt: Option<f32>,
    /// Fundamental frequency characteristics
    pub f0_characteristics: Option<F0Characteristics>,
    /// Duration characteristics
    pub duration_characteristics: DurationCharacteristics,
}

/// F0 characteristics for phonemes
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct F0Characteristics {
    /// Typical F0 range
    pub f0_range: (f32, f32),
    /// F0 stability
    pub f0_stability: f32,
    /// F0 transitions
    pub f0_transitions: Vec<f32>,
}

/// Duration characteristics for phonemes
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DurationCharacteristics {
    /// Typical duration in milliseconds
    pub typical_duration: f32,
    /// Duration variability
    pub duration_variability: f32,
    /// Context-dependent duration factors
    pub context_factors: HashMap<String, f32>,
}

/// Perceptual features for phoneme similarity
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerceptualFeatures {
    /// Perceptual salience
    pub salience: f32,
    /// Discriminability from other phonemes
    pub discriminability: f32,
    /// Cross-linguistic confusability
    pub confusability: HashMap<String, f32>,
    /// Perceptual distance from prototype
    pub prototype_distance: f32,
    /// Categorical boundaries
    pub categorical_boundaries: Vec<f32>,
}

/// Phoneme similarity matrix entry
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhonemeSimilarity {
    /// Source phoneme
    pub source_phoneme: String,
    /// Target phoneme
    pub target_phoneme: String,
    /// Overall similarity score [0.0, 1.0]
    pub overall_similarity: f32,
    /// Articulatory similarity
    pub articulatory_similarity: f32,
    /// Acoustic similarity
    pub acoustic_similarity: f32,
    /// Perceptual similarity
    pub perceptual_similarity: f32,
    /// Language-specific adjustment
    pub language_adjustment: f32,
}

/// Cross-language phoneme mapping result
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CrossLanguageMapping {
    /// Source language
    pub source_language: LanguageCode,
    /// Target language
    pub target_language: LanguageCode,
    /// Phoneme mappings
    pub phoneme_mappings: Vec<PhonemeMapping>,
    /// Overall mapping quality
    pub mapping_quality: f32,
    /// Unmapped phonemes
    pub unmapped_phonemes: Vec<String>,
    /// Mapping confidence
    pub mapping_confidence: f32,
}

/// Individual phoneme mapping
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhonemeMapping {
    /// Source phoneme
    pub source_phoneme: String,
    /// Target phoneme candidates
    pub target_candidates: Vec<PhonemeCandidate>,
    /// Best mapping
    pub best_mapping: Option<PhonemeCandidate>,
    /// Mapping confidence
    pub mapping_confidence: f32,
}

/// Phoneme mapping candidate
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhonemeCandidate {
    /// Target phoneme
    pub target_phoneme: String,
    /// Similarity score
    pub similarity_score: f32,
    /// Confidence score
    pub confidence_score: f32,
    /// Mapping type
    pub mapping_type: MappingType,
}

/// Type of phoneme mapping
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MappingType {
    /// Direct mapping (identical or very similar)
    Direct,
    /// Approximate mapping (similar features)
    Approximate,
    /// Substitution mapping (best available alternative)
    Substitution,
    /// Deletion (no suitable mapping)
    Deletion,
    /// Insertion (target language requires additional phoneme)
    Insertion,
}

/// Universal phoneme mapping system
pub struct UniversalPhonemeMapper {
    /// Configuration
    config: UniversalPhonemeMappingConfig,
    /// Universal phoneme inventory
    universal_phonemes: HashMap<String, UniversalPhoneme>,
    /// Language-specific phoneme inventories
    language_inventories: HashMap<LanguageCode, Vec<String>>,
    /// Phoneme similarity matrices
    similarity_matrices: HashMap<(LanguageCode, LanguageCode), Vec<PhonemeSimilarity>>,
    /// Precomputed cross-language mappings
    cross_language_mappings: HashMap<(LanguageCode, LanguageCode), CrossLanguageMapping>,
}

impl UniversalPhonemeMapper {
    /// Create new universal phoneme mapper
    pub fn new(config: UniversalPhonemeMappingConfig) -> Self {
        let mut mapper = Self {
            config,
            universal_phonemes: HashMap::new(),
            language_inventories: HashMap::new(),
            similarity_matrices: HashMap::new(),
            cross_language_mappings: HashMap::new(),
        };

        mapper.initialize_universal_phonemes();
        mapper.initialize_language_inventories();
        mapper.compute_similarity_matrices();
        mapper.precompute_cross_language_mappings();

        mapper
    }

    /// Initialize universal phoneme inventory
    fn initialize_universal_phonemes(&mut self) {
        // Initialize vowels
        self.add_vowel(
            "i",
            VowelHeight::High,
            VowelBackness::Front,
            VowelRoundness::Unrounded,
            VowelTenseness::Tense,
        );
        self.add_vowel(
            "ɪ",
            VowelHeight::NearHigh,
            VowelBackness::NearFront,
            VowelRoundness::Unrounded,
            VowelTenseness::Lax,
        );
        self.add_vowel(
            "e",
            VowelHeight::HighMid,
            VowelBackness::Front,
            VowelRoundness::Unrounded,
            VowelTenseness::Tense,
        );
        self.add_vowel(
            "ɛ",
            VowelHeight::LowMid,
            VowelBackness::Front,
            VowelRoundness::Unrounded,
            VowelTenseness::Lax,
        );
        self.add_vowel(
            "æ",
            VowelHeight::NearLow,
            VowelBackness::Front,
            VowelRoundness::Unrounded,
            VowelTenseness::Lax,
        );
        self.add_vowel(
            "a",
            VowelHeight::Low,
            VowelBackness::Central,
            VowelRoundness::Unrounded,
            VowelTenseness::Lax,
        );
        self.add_vowel(
            "ɑ",
            VowelHeight::Low,
            VowelBackness::Back,
            VowelRoundness::Unrounded,
            VowelTenseness::Lax,
        );
        self.add_vowel(
            "ɔ",
            VowelHeight::LowMid,
            VowelBackness::Back,
            VowelRoundness::Rounded,
            VowelTenseness::Lax,
        );
        self.add_vowel(
            "o",
            VowelHeight::HighMid,
            VowelBackness::Back,
            VowelRoundness::Rounded,
            VowelTenseness::Tense,
        );
        self.add_vowel(
            "ʊ",
            VowelHeight::NearHigh,
            VowelBackness::NearBack,
            VowelRoundness::Rounded,
            VowelTenseness::Lax,
        );
        self.add_vowel(
            "u",
            VowelHeight::High,
            VowelBackness::Back,
            VowelRoundness::Rounded,
            VowelTenseness::Tense,
        );
        self.add_vowel(
            "ʌ",
            VowelHeight::LowMid,
            VowelBackness::Back,
            VowelRoundness::Unrounded,
            VowelTenseness::Lax,
        );
        self.add_vowel(
            "ə",
            VowelHeight::Mid,
            VowelBackness::Central,
            VowelRoundness::Unrounded,
            VowelTenseness::Lax,
        );

        // Initialize consonants
        self.add_consonant(
            "p",
            MannerOfArticulation::Stop,
            PlaceOfArticulation::Bilabial,
            Voicing::Voiceless,
        );
        self.add_consonant(
            "b",
            MannerOfArticulation::Stop,
            PlaceOfArticulation::Bilabial,
            Voicing::Voiced,
        );
        self.add_consonant(
            "t",
            MannerOfArticulation::Stop,
            PlaceOfArticulation::Alveolar,
            Voicing::Voiceless,
        );
        self.add_consonant(
            "d",
            MannerOfArticulation::Stop,
            PlaceOfArticulation::Alveolar,
            Voicing::Voiced,
        );
        self.add_consonant(
            "k",
            MannerOfArticulation::Stop,
            PlaceOfArticulation::Velar,
            Voicing::Voiceless,
        );
        self.add_consonant(
            "g",
            MannerOfArticulation::Stop,
            PlaceOfArticulation::Velar,
            Voicing::Voiced,
        );
        self.add_consonant(
            "f",
            MannerOfArticulation::Fricative,
            PlaceOfArticulation::Labiodental,
            Voicing::Voiceless,
        );
        self.add_consonant(
            "v",
            MannerOfArticulation::Fricative,
            PlaceOfArticulation::Labiodental,
            Voicing::Voiced,
        );
        self.add_consonant(
            "θ",
            MannerOfArticulation::Fricative,
            PlaceOfArticulation::Dental,
            Voicing::Voiceless,
        );
        self.add_consonant(
            "ð",
            MannerOfArticulation::Fricative,
            PlaceOfArticulation::Dental,
            Voicing::Voiced,
        );
        self.add_consonant(
            "s",
            MannerOfArticulation::Fricative,
            PlaceOfArticulation::Alveolar,
            Voicing::Voiceless,
        );
        self.add_consonant(
            "z",
            MannerOfArticulation::Fricative,
            PlaceOfArticulation::Alveolar,
            Voicing::Voiced,
        );
        self.add_consonant(
            "ʃ",
            MannerOfArticulation::Fricative,
            PlaceOfArticulation::Postalveolar,
            Voicing::Voiceless,
        );
        self.add_consonant(
            "ʒ",
            MannerOfArticulation::Fricative,
            PlaceOfArticulation::Postalveolar,
            Voicing::Voiced,
        );
        self.add_consonant(
            "h",
            MannerOfArticulation::Fricative,
            PlaceOfArticulation::Glottal,
            Voicing::Voiceless,
        );
        self.add_consonant(
            "m",
            MannerOfArticulation::Nasal,
            PlaceOfArticulation::Bilabial,
            Voicing::Voiced,
        );
        self.add_consonant(
            "n",
            MannerOfArticulation::Nasal,
            PlaceOfArticulation::Alveolar,
            Voicing::Voiced,
        );
        self.add_consonant(
            "ŋ",
            MannerOfArticulation::Nasal,
            PlaceOfArticulation::Velar,
            Voicing::Voiced,
        );
        self.add_consonant(
            "l",
            MannerOfArticulation::Lateral,
            PlaceOfArticulation::Alveolar,
            Voicing::Voiced,
        );
        self.add_consonant(
            "r",
            MannerOfArticulation::Liquid,
            PlaceOfArticulation::Alveolar,
            Voicing::Voiced,
        );
        self.add_consonant(
            "w",
            MannerOfArticulation::Approximant,
            PlaceOfArticulation::Labiovelar,
            Voicing::Voiced,
        );
        self.add_consonant(
            "j",
            MannerOfArticulation::Approximant,
            PlaceOfArticulation::Palatal,
            Voicing::Voiced,
        );
        self.add_consonant(
            "tʃ",
            MannerOfArticulation::Affricate,
            PlaceOfArticulation::Postalveolar,
            Voicing::Voiceless,
        );
        self.add_consonant(
            "dʒ",
            MannerOfArticulation::Affricate,
            PlaceOfArticulation::Postalveolar,
            Voicing::Voiced,
        );
    }

    /// Add vowel to universal phoneme inventory
    fn add_vowel(
        &mut self,
        symbol: &str,
        height: VowelHeight,
        backness: VowelBackness,
        roundness: VowelRoundness,
        tenseness: VowelTenseness,
    ) {
        let vowel_features = VowelFeatures {
            height,
            backness,
            roundness,
            tenseness,
            nasalization: false,
            length: VowelLength::Short,
        };

        let universal_phoneme = UniversalPhoneme {
            ipa_symbol: symbol.to_string(),
            articulatory_features: ArticulatoryFeatures {
                manner: MannerOfArticulation::Vowel,
                place: PlaceOfArticulation::Glottal, // Default for vowels
                voicing: Voicing::Voiced,
                vowel_features: Some(vowel_features),
                consonant_features: None,
                airstream: AirstreamMechanism::Pulmonic,
            },
            acoustic_features: self.generate_default_acoustic_features(symbol),
            perceptual_features: self.generate_default_perceptual_features(symbol),
            language_variations: HashMap::new(),
            language_frequency: HashMap::new(),
        };

        self.universal_phonemes
            .insert(symbol.to_string(), universal_phoneme);
    }

    /// Add consonant to universal phoneme inventory
    fn add_consonant(
        &mut self,
        symbol: &str,
        manner: MannerOfArticulation,
        place: PlaceOfArticulation,
        voicing: Voicing,
    ) {
        let consonant_features = ConsonantFeatures {
            aspiration: false,
            palatalization: false,
            velarization: false,
            pharyngealization: false,
            labialization: false,
            glottalization: false,
            gemination: false,
        };

        let universal_phoneme = UniversalPhoneme {
            ipa_symbol: symbol.to_string(),
            articulatory_features: ArticulatoryFeatures {
                manner,
                place,
                voicing,
                vowel_features: None,
                consonant_features: Some(consonant_features),
                airstream: AirstreamMechanism::Pulmonic,
            },
            acoustic_features: self.generate_default_acoustic_features(symbol),
            perceptual_features: self.generate_default_perceptual_features(symbol),
            language_variations: HashMap::new(),
            language_frequency: HashMap::new(),
        };

        self.universal_phonemes
            .insert(symbol.to_string(), universal_phoneme);
    }

    /// Generate default acoustic features for a phoneme
    fn generate_default_acoustic_features(&self, symbol: &str) -> AcousticFeatures {
        // Simplified acoustic feature generation
        let formant_frequencies = match symbol {
            "i" => Some(vec![270.0, 2290.0, 3010.0]),
            "ɪ" => Some(vec![400.0, 2000.0, 2550.0]),
            "e" => Some(vec![530.0, 1840.0, 2480.0]),
            "ɛ" => Some(vec![660.0, 1720.0, 2410.0]),
            "æ" => Some(vec![860.0, 1720.0, 2440.0]),
            "a" => Some(vec![850.0, 1610.0, 2410.0]),
            "ɑ" => Some(vec![750.0, 940.0, 2540.0]),
            "ɔ" => Some(vec![590.0, 880.0, 2540.0]),
            "o" => Some(vec![500.0, 700.0, 2240.0]),
            "ʊ" => Some(vec![470.0, 1160.0, 2680.0]),
            "u" => Some(vec![460.0, 1170.0, 2680.0]),
            "ʌ" => Some(vec![760.0, 1400.0, 2780.0]),
            "ə" => Some(vec![500.0, 1350.0, 1690.0]),
            _ => None, // Consonants don't have formant frequencies
        };

        AcousticFeatures {
            formant_frequencies,
            voice_onset_time: None,
            spectral_centroid: None,
            spectral_tilt: None,
            f0_characteristics: None,
            duration_characteristics: DurationCharacteristics {
                typical_duration: 100.0, // Default 100ms
                duration_variability: 20.0,
                context_factors: HashMap::new(),
            },
        }
    }

    /// Generate default perceptual features for a phoneme
    fn generate_default_perceptual_features(&self, _symbol: &str) -> PerceptualFeatures {
        PerceptualFeatures {
            salience: 0.7,
            discriminability: 0.8,
            confusability: HashMap::new(),
            prototype_distance: 0.0,
            categorical_boundaries: vec![],
        }
    }

    /// Initialize language-specific phoneme inventories
    fn initialize_language_inventories(&mut self) {
        // English inventory
        self.language_inventories.insert(
            LanguageCode::EnUs,
            vec![
                "i", "ɪ", "e", "ɛ", "æ", "ɑ", "ɔ", "o", "ʊ", "u", "ʌ", "ə", "p", "b", "t", "d",
                "k", "g", "f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ", "h", "m", "n", "ŋ", "l", "r",
                "w", "j", "tʃ", "dʒ",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        );

        // Spanish inventory
        self.language_inventories.insert(
            LanguageCode::EsEs,
            vec![
                "a", "e", "i", "o", "u", "p", "b", "t", "d", "k", "g", "f", "θ", "s", "x", "tʃ",
                "m", "n", "ɲ", "ŋ", "l", "ʎ", "r", "rr", "w", "j",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        );

        // French inventory
        self.language_inventories.insert(
            LanguageCode::FrFr,
            vec![
                "i", "e", "ɛ", "a", "ɑ", "ɔ", "o", "u", "y", "ø", "œ", "ə", "p", "b", "t", "d",
                "k", "g", "f", "v", "s", "z", "ʃ", "ʒ", "m", "n", "ɲ", "ŋ", "l", "r", "ʁ", "w",
                "ɥ", "j",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        );

        // German inventory
        self.language_inventories.insert(
            LanguageCode::DeDe,
            vec![
                "i", "ɪ", "e", "ɛ", "a", "ɑ", "ɔ", "o", "u", "ʊ", "y", "ʏ", "ø", "œ", "ə", "p",
                "b", "t", "d", "k", "g", "f", "v", "s", "z", "ʃ", "ʒ", "ç", "x", "h", "m", "n",
                "ŋ", "l", "r", "ʁ", "w", "j",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        );

        // Japanese inventory
        self.language_inventories.insert(
            LanguageCode::JaJp,
            vec![
                "a", "i", "u", "e", "o", "k", "g", "s", "z", "t", "d", "n", "h", "b", "p", "m",
                "y", "r", "w",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        );

        // Chinese inventory
        self.language_inventories.insert(
            LanguageCode::ZhCn,
            vec![
                "a", "o", "e", "i", "u", "ü", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k",
                "h", "j", "q", "x", "z", "c", "s", "zh", "ch", "sh", "r", "w", "y",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        );
    }

    /// Compute phoneme similarity matrices for all language pairs
    fn compute_similarity_matrices(&mut self) {
        let languages: Vec<LanguageCode> = self.language_inventories.keys().cloned().collect();

        for &lang1 in &languages {
            for &lang2 in &languages {
                if lang1 != lang2 {
                    let similarity_matrix = self.compute_similarity_matrix(lang1, lang2);
                    self.similarity_matrices
                        .insert((lang1, lang2), similarity_matrix);
                }
            }
        }
    }

    /// Compute similarity matrix for a specific language pair
    fn compute_similarity_matrix(
        &self,
        lang1: LanguageCode,
        lang2: LanguageCode,
    ) -> Vec<PhonemeSimilarity> {
        let mut similarities = Vec::new();

        if let (Some(inventory1), Some(inventory2)) = (
            self.language_inventories.get(&lang1),
            self.language_inventories.get(&lang2),
        ) {
            for phoneme1 in inventory1 {
                for phoneme2 in inventory2 {
                    if let (Some(univ_phoneme1), Some(univ_phoneme2)) = (
                        self.universal_phonemes.get(phoneme1),
                        self.universal_phonemes.get(phoneme2),
                    ) {
                        let similarity =
                            self.calculate_phoneme_similarity(univ_phoneme1, univ_phoneme2);
                        similarities.push(similarity);
                    }
                }
            }
        }

        similarities
    }

    /// Calculate similarity between two universal phonemes
    fn calculate_phoneme_similarity(
        &self,
        phoneme1: &UniversalPhoneme,
        phoneme2: &UniversalPhoneme,
    ) -> PhonemeSimilarity {
        let articulatory_similarity = if self.config.enable_articulatory_features {
            self.calculate_articulatory_similarity(
                &phoneme1.articulatory_features,
                &phoneme2.articulatory_features,
            )
        } else {
            0.0
        };

        let acoustic_similarity = if self.config.enable_acoustic_similarity {
            self.calculate_acoustic_similarity(
                &phoneme1.acoustic_features,
                &phoneme2.acoustic_features,
            )
        } else {
            0.0
        };

        let perceptual_similarity = if self.config.enable_perceptual_distance {
            self.calculate_perceptual_similarity(
                &phoneme1.perceptual_features,
                &phoneme2.perceptual_features,
            )
        } else {
            0.0
        };

        let overall_similarity = articulatory_similarity * self.config.articulatory_weight
            + acoustic_similarity * self.config.acoustic_weight
            + perceptual_similarity * self.config.perceptual_weight;

        PhonemeSimilarity {
            source_phoneme: phoneme1.ipa_symbol.clone(),
            target_phoneme: phoneme2.ipa_symbol.clone(),
            overall_similarity,
            articulatory_similarity,
            acoustic_similarity,
            perceptual_similarity,
            language_adjustment: 1.0,
        }
    }

    /// Calculate articulatory similarity between two phonemes
    fn calculate_articulatory_similarity(
        &self,
        features1: &ArticulatoryFeatures,
        features2: &ArticulatoryFeatures,
    ) -> f32 {
        let mut similarity = 0.0;
        let mut total_features = 0;

        // Manner of articulation
        if features1.manner == features2.manner {
            similarity += 1.0;
        }
        total_features += 1;

        // Place of articulation
        if features1.place == features2.place {
            similarity += 1.0;
        }
        total_features += 1;

        // Voicing
        if features1.voicing == features2.voicing {
            similarity += 1.0;
        }
        total_features += 1;

        // Vowel features (if both are vowels)
        if let (Some(vowel1), Some(vowel2)) = (&features1.vowel_features, &features2.vowel_features)
        {
            if vowel1.height == vowel2.height {
                similarity += 1.0;
            }
            total_features += 1;

            if vowel1.backness == vowel2.backness {
                similarity += 1.0;
            }
            total_features += 1;

            if vowel1.roundness == vowel2.roundness {
                similarity += 1.0;
            }
            total_features += 1;
        }

        if total_features > 0 {
            similarity / total_features as f32
        } else {
            0.0
        }
    }

    /// Calculate acoustic similarity between two phonemes
    fn calculate_acoustic_similarity(
        &self,
        features1: &AcousticFeatures,
        features2: &AcousticFeatures,
    ) -> f32 {
        // Simplified acoustic similarity calculation
        if let (Some(formants1), Some(formants2)) = (
            &features1.formant_frequencies,
            &features2.formant_frequencies,
        ) {
            let mut similarity = 0.0;
            let min_len = formants1.len().min(formants2.len());

            for i in 0..min_len {
                let diff = (formants1[i] - formants2[i]).abs();
                let max_diff = formants1[i].max(formants2[i]);
                if max_diff > 0.0 {
                    similarity += 1.0 - (diff / max_diff).min(1.0);
                }
            }

            if min_len > 0 {
                similarity / min_len as f32
            } else {
                0.5
            }
        } else {
            0.5 // Default similarity for non-vowel phonemes
        }
    }

    /// Calculate perceptual similarity between two phonemes
    fn calculate_perceptual_similarity(
        &self,
        features1: &PerceptualFeatures,
        features2: &PerceptualFeatures,
    ) -> f32 {
        // Simplified perceptual similarity calculation
        let salience_similarity = 1.0 - (features1.salience - features2.salience).abs();
        let discriminability_similarity =
            1.0 - (features1.discriminability - features2.discriminability).abs();

        (salience_similarity + discriminability_similarity) / 2.0
    }

    /// Precompute cross-language mappings for all language pairs
    fn precompute_cross_language_mappings(&mut self) {
        let languages: Vec<LanguageCode> = self.language_inventories.keys().cloned().collect();

        for &lang1 in &languages {
            for &lang2 in &languages {
                if lang1 != lang2 {
                    let mapping = self.compute_cross_language_mapping(lang1, lang2);
                    self.cross_language_mappings.insert((lang1, lang2), mapping);
                }
            }
        }
    }

    /// Compute cross-language mapping for a specific language pair
    fn compute_cross_language_mapping(
        &self,
        source_lang: LanguageCode,
        target_lang: LanguageCode,
    ) -> CrossLanguageMapping {
        let mut phoneme_mappings = Vec::new();
        let mut unmapped_phonemes = Vec::new();
        let mut total_quality = 0.0;
        let mut mapping_count = 0;

        if let Some(source_inventory) = self.language_inventories.get(&source_lang) {
            for source_phoneme in source_inventory {
                if let Some(similarity_matrix) =
                    self.similarity_matrices.get(&(source_lang, target_lang))
                {
                    let mut candidates = Vec::new();

                    for similarity in similarity_matrix {
                        if similarity.source_phoneme == *source_phoneme
                            && similarity.overall_similarity >= self.config.similarity_threshold
                        {
                            candidates.push(PhonemeCandidate {
                                target_phoneme: similarity.target_phoneme.clone(),
                                similarity_score: similarity.overall_similarity,
                                confidence_score: similarity.overall_similarity,
                                mapping_type: if similarity.overall_similarity > 0.9 {
                                    MappingType::Direct
                                } else if similarity.overall_similarity > 0.7 {
                                    MappingType::Approximate
                                } else {
                                    MappingType::Substitution
                                },
                            });
                        }
                    }

                    // Sort candidates by similarity score
                    candidates.sort_by(|a, b| {
                        b.similarity_score.partial_cmp(&a.similarity_score).unwrap()
                    });

                    let best_mapping = candidates.first().cloned();
                    let mapping_confidence =
                        best_mapping.as_ref().map_or(0.0, |m| m.confidence_score);

                    if best_mapping.is_some() {
                        total_quality += mapping_confidence;
                        mapping_count += 1;
                    } else {
                        unmapped_phonemes.push(source_phoneme.clone());
                    }

                    phoneme_mappings.push(PhonemeMapping {
                        source_phoneme: source_phoneme.clone(),
                        target_candidates: candidates,
                        best_mapping,
                        mapping_confidence,
                    });
                }
            }
        }

        let mapping_quality = if mapping_count > 0 {
            total_quality / mapping_count as f32
        } else {
            0.0
        };

        let mapping_confidence = if phoneme_mappings.len() > 0 {
            phoneme_mappings
                .iter()
                .map(|m| m.mapping_confidence)
                .sum::<f32>()
                / phoneme_mappings.len() as f32
        } else {
            0.0
        };

        CrossLanguageMapping {
            source_language: source_lang,
            target_language: target_lang,
            phoneme_mappings,
            mapping_quality,
            unmapped_phonemes,
            mapping_confidence,
        }
    }

    /// Get cross-language phoneme mapping
    pub fn get_cross_language_mapping(
        &self,
        source_lang: LanguageCode,
        target_lang: LanguageCode,
    ) -> Option<&CrossLanguageMapping> {
        self.cross_language_mappings
            .get(&(source_lang, target_lang))
    }

    /// Map a phoneme from source language to target language
    pub fn map_phoneme(
        &self,
        phoneme: &str,
        source_lang: LanguageCode,
        target_lang: LanguageCode,
    ) -> Option<PhonemeCandidate> {
        self.get_cross_language_mapping(source_lang, target_lang)?
            .phoneme_mappings
            .iter()
            .find(|mapping| mapping.source_phoneme == phoneme)?
            .best_mapping
            .clone()
    }

    /// Convert language-specific phoneme to universal phoneme
    pub fn to_universal_phoneme(
        &self,
        phoneme: &Phoneme,
        language: LanguageCode,
    ) -> Option<&UniversalPhoneme> {
        // Try to find the phoneme in the universal inventory
        self.universal_phonemes
            .get(&phoneme.ipa_symbol)
            .or_else(|| self.universal_phonemes.get(&phoneme.symbol))
    }

    /// Convert universal phoneme to language-specific representation
    pub fn from_universal_phoneme(
        &self,
        universal_phoneme: &UniversalPhoneme,
        target_language: LanguageCode,
    ) -> Option<Phoneme> {
        // Check if the universal phoneme has a specific variation for the target language
        if let Some(language_specific) = universal_phoneme.language_variations.get(&target_language)
        {
            return Some(Phoneme::new(language_specific.clone()));
        }

        // Otherwise, use the IPA symbol
        Some(Phoneme::new(universal_phoneme.ipa_symbol.clone()))
    }

    /// Calculate phoneme similarity score
    pub fn calculate_similarity_score(
        &self,
        phoneme1: &str,
        phoneme2: &str,
        lang1: LanguageCode,
        lang2: LanguageCode,
    ) -> f32 {
        // If phonemes are identical, return high similarity
        if phoneme1 == phoneme2 {
            return 1.0;
        }

        if let Some(similarity_matrix) = self.similarity_matrices.get(&(lang1, lang2)) {
            for similarity in similarity_matrix {
                if similarity.source_phoneme == phoneme1 && similarity.target_phoneme == phoneme2 {
                    return similarity.overall_similarity;
                }
            }
        }
        0.0
    }

    /// Get supported languages
    pub fn get_supported_languages(&self) -> Vec<LanguageCode> {
        self.language_inventories.keys().cloned().collect()
    }

    /// Get universal phoneme inventory
    pub fn get_universal_phonemes(&self) -> &HashMap<String, UniversalPhoneme> {
        &self.universal_phonemes
    }

    /// Get language-specific phoneme inventory
    pub fn get_language_inventory(&self, language: LanguageCode) -> Option<&Vec<String>> {
        self.language_inventories.get(&language)
    }

    /// Analyze phoneme coverage for a language pair
    pub fn analyze_phoneme_coverage(
        &self,
        source_lang: LanguageCode,
        target_lang: LanguageCode,
    ) -> EvaluationResult<PhonemeConverageAnalysis> {
        let mapping = self
            .get_cross_language_mapping(source_lang, target_lang)
            .ok_or_else(|| EvaluationError::ConfigurationError {
                message: format!(
                    "No mapping found for {:?} -> {:?}",
                    source_lang, target_lang
                ),
            })?;

        let source_inventory = self.get_language_inventory(source_lang).ok_or_else(|| {
            EvaluationError::ConfigurationError {
                message: format!("No inventory found for {:?}", source_lang),
            }
        })?;

        let target_inventory = self.get_language_inventory(target_lang).ok_or_else(|| {
            EvaluationError::ConfigurationError {
                message: format!("No inventory found for {:?}", target_lang),
            }
        })?;

        let total_phonemes = source_inventory.len();
        let mapped_phonemes = mapping
            .phoneme_mappings
            .iter()
            .filter(|m| m.best_mapping.is_some())
            .count();
        let unmapped_phonemes = total_phonemes - mapped_phonemes;

        let coverage_ratio = if total_phonemes > 0 {
            mapped_phonemes as f32 / total_phonemes as f32
        } else {
            0.0
        };

        let mapping_quality_distribution = mapping
            .phoneme_mappings
            .iter()
            .filter_map(|m| m.best_mapping.as_ref())
            .map(|m| m.similarity_score)
            .collect::<Vec<f32>>();

        let average_mapping_quality = if !mapping_quality_distribution.is_empty() {
            mapping_quality_distribution.iter().sum::<f32>()
                / mapping_quality_distribution.len() as f32
        } else {
            0.0
        };

        Ok(PhonemeConverageAnalysis {
            source_language: source_lang,
            target_language: target_lang,
            total_phonemes,
            mapped_phonemes,
            unmapped_phonemes,
            coverage_ratio,
            average_mapping_quality,
            mapping_quality_distribution,
            problematic_phonemes: mapping.unmapped_phonemes.clone(),
        })
    }
}

/// Phoneme coverage analysis result
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhonemeConverageAnalysis {
    /// Source language
    pub source_language: LanguageCode,
    /// Target language
    pub target_language: LanguageCode,
    /// Total number of phonemes in source language
    pub total_phonemes: usize,
    /// Number of successfully mapped phonemes
    pub mapped_phonemes: usize,
    /// Number of unmapped phonemes
    pub unmapped_phonemes: usize,
    /// Coverage ratio [0.0, 1.0]
    pub coverage_ratio: f32,
    /// Average mapping quality
    pub average_mapping_quality: f32,
    /// Distribution of mapping quality scores
    pub mapping_quality_distribution: Vec<f32>,
    /// Phonemes that couldn't be mapped
    pub problematic_phonemes: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_universal_phoneme_mapper_creation() {
        let config = UniversalPhonemeMappingConfig::default();
        let mapper = UniversalPhonemeMapper::new(config);

        assert!(!mapper.get_universal_phonemes().is_empty());
        assert!(!mapper.get_supported_languages().is_empty());
    }

    #[test]
    fn test_phoneme_mapping() {
        let config = UniversalPhonemeMappingConfig::default();
        let mapper = UniversalPhonemeMapper::new(config);

        // Test mapping from English to Spanish
        let mapping = mapper.map_phoneme("t", LanguageCode::EnUs, LanguageCode::EsEs);
        assert!(mapping.is_some());

        let candidate = mapping.unwrap();
        assert_eq!(candidate.target_phoneme, "t");
        assert!(candidate.similarity_score > 0.0);
    }

    #[test]
    fn test_cross_language_mapping() {
        let config = UniversalPhonemeMappingConfig::default();
        let mapper = UniversalPhonemeMapper::new(config);

        let mapping = mapper.get_cross_language_mapping(LanguageCode::EnUs, LanguageCode::EsEs);
        assert!(mapping.is_some());

        let mapping = mapping.unwrap();
        assert_eq!(mapping.source_language, LanguageCode::EnUs);
        assert_eq!(mapping.target_language, LanguageCode::EsEs);
        assert!(!mapping.phoneme_mappings.is_empty());
    }

    #[test]
    fn test_phoneme_similarity_calculation() {
        let config = UniversalPhonemeMappingConfig::default();
        let mapper = UniversalPhonemeMapper::new(config);

        // Test similarity between identical phonemes
        let similarity =
            mapper.calculate_similarity_score("t", "t", LanguageCode::EnUs, LanguageCode::EsEs);
        assert!(similarity > 0.9);

        // Test similarity between different phonemes
        let similarity =
            mapper.calculate_similarity_score("t", "k", LanguageCode::EnUs, LanguageCode::EsEs);
        assert!(similarity < 0.9);
    }

    #[test]
    fn test_phoneme_coverage_analysis() {
        let config = UniversalPhonemeMappingConfig::default();
        let mapper = UniversalPhonemeMapper::new(config);

        let analysis = mapper
            .analyze_phoneme_coverage(LanguageCode::EnUs, LanguageCode::EsEs)
            .unwrap();
        assert_eq!(analysis.source_language, LanguageCode::EnUs);
        assert_eq!(analysis.target_language, LanguageCode::EsEs);
        assert!(analysis.total_phonemes > 0);
        assert!(analysis.coverage_ratio >= 0.0 && analysis.coverage_ratio <= 1.0);
    }

    #[test]
    fn test_universal_phoneme_conversion() {
        let config = UniversalPhonemeMappingConfig::default();
        let mapper = UniversalPhonemeMapper::new(config);

        let phoneme = Phoneme::new("t");
        let universal = mapper.to_universal_phoneme(&phoneme, LanguageCode::EnUs);
        assert!(universal.is_some());

        let universal = universal.unwrap();
        assert_eq!(universal.ipa_symbol, "t");
        assert_eq!(
            universal.articulatory_features.manner,
            MannerOfArticulation::Stop
        );
        assert_eq!(
            universal.articulatory_features.place,
            PlaceOfArticulation::Alveolar
        );
        assert_eq!(universal.articulatory_features.voicing, Voicing::Voiceless);
    }

    #[test]
    fn test_language_inventory_access() {
        let config = UniversalPhonemeMappingConfig::default();
        let mapper = UniversalPhonemeMapper::new(config);

        let english_inventory = mapper.get_language_inventory(LanguageCode::EnUs);
        assert!(english_inventory.is_some());

        let inventory = english_inventory.unwrap();
        assert!(inventory.contains(&"t".to_string()));
        assert!(inventory.contains(&"i".to_string()));
        assert!(inventory.contains(&"æ".to_string()));
    }
}

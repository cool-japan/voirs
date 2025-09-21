//! Phoneme sets and language-specific phoneme inventories
//!
//! This module provides comprehensive phoneme sets for different languages
//! and utilities for working with phoneme inventories.

use crate::LanguageCode;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Phoneme set notation systems
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PhonemeNotation {
    /// International Phonetic Alphabet
    IPA,
    /// ARPABET notation used in CMU dictionary
    ARPABET,
    /// Speech Assessment Methods Phonetic Alphabet
    SAMPA,
    /// X-SAMPA (Extended SAMPA)
    XSAMPA,
    /// Custom notation system
    Custom(String),
}

/// Phoneme set specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhonemeSet {
    /// Name of the phoneme set
    pub name: String,
    /// Notation system used
    pub notation: PhonemeNotation,
    /// Phoneme mappings to IPA
    pub to_ipa_map: HashMap<String, String>,
    /// IPA to this notation mappings
    pub from_ipa_map: HashMap<String, String>,
    /// Language compatibility
    pub languages: Vec<LanguageCode>,
}

/// Phoneme inventory for a specific language
#[derive(Debug, Clone)]
pub struct PhonemeInventory {
    /// Language code
    pub language: LanguageCode,
    /// Set of vowel phonemes
    pub vowels: HashSet<String>,
    /// Set of consonant phonemes
    pub consonants: HashSet<String>,
    /// Phoneme feature mappings
    pub features: HashMap<String, PhonemeFeatures>,
    /// Supported phoneme notations
    pub notation_sets: Vec<PhonemeSet>,
}

/// Phonetic features for a phoneme
#[derive(Debug, Clone, PartialEq)]
pub struct PhonemeFeatures {
    /// Place of articulation (for consonants)
    pub place: Option<String>,
    /// Manner of articulation (for consonants)
    pub manner: Option<String>,
    /// Voicing
    pub voiced: Option<bool>,
    /// Height (for vowels)
    pub height: Option<String>,
    /// Backness (for vowels)
    pub backness: Option<String>,
    /// Roundedness (for vowels)
    pub rounded: Option<bool>,
}

impl PhonemeInventory {
    /// Create English phoneme inventory
    #[must_use]
    pub fn english() -> Self {
        let mut vowels = HashSet::new();
        vowels.insert("i".to_string()); // beat
        vowels.insert("ɪ".to_string()); // bit
        vowels.insert("e".to_string()); // bait
        vowels.insert("ɛ".to_string()); // bet
        vowels.insert("æ".to_string()); // bat
        vowels.insert("a".to_string()); // bot
        vowels.insert("ɑ".to_string()); // father
        vowels.insert("ɔ".to_string()); // bought
        vowels.insert("o".to_string()); // boat
        vowels.insert("ʊ".to_string()); // book
        vowels.insert("u".to_string()); // boot
        vowels.insert("ʌ".to_string()); // but
        vowels.insert("ə".to_string()); // about

        let mut consonants = HashSet::new();
        consonants.insert("p".to_string());
        consonants.insert("b".to_string());
        consonants.insert("t".to_string());
        consonants.insert("d".to_string());
        consonants.insert("k".to_string());
        consonants.insert("g".to_string());
        consonants.insert("f".to_string());
        consonants.insert("v".to_string());
        consonants.insert("θ".to_string());
        consonants.insert("ð".to_string());
        consonants.insert("s".to_string());
        consonants.insert("z".to_string());
        consonants.insert("ʃ".to_string());
        consonants.insert("ʒ".to_string());
        consonants.insert("h".to_string());
        consonants.insert("m".to_string());
        consonants.insert("n".to_string());
        consonants.insert("ŋ".to_string());
        consonants.insert("l".to_string());
        consonants.insert("r".to_string());
        consonants.insert("w".to_string());
        consonants.insert("j".to_string());
        consonants.insert("tʃ".to_string());
        consonants.insert("dʒ".to_string());

        let mut features = HashMap::new();

        // Add vowel features
        features.insert(
            "i".to_string(),
            PhonemeFeatures {
                place: None,
                manner: None,
                voiced: None,
                height: Some("high".to_string()),
                backness: Some("front".to_string()),
                rounded: Some(false),
            },
        );

        features.insert(
            "ɪ".to_string(),
            PhonemeFeatures {
                place: None,
                manner: None,
                voiced: None,
                height: Some("near-high".to_string()),
                backness: Some("front".to_string()),
                rounded: Some(false),
            },
        );

        // Add consonant features
        features.insert(
            "p".to_string(),
            PhonemeFeatures {
                place: Some("bilabial".to_string()),
                manner: Some("stop".to_string()),
                voiced: Some(false),
                height: None,
                backness: None,
                rounded: None,
            },
        );

        features.insert(
            "b".to_string(),
            PhonemeFeatures {
                place: Some("bilabial".to_string()),
                manner: Some("stop".to_string()),
                voiced: Some(true),
                height: None,
                backness: None,
                rounded: None,
            },
        );

        Self {
            language: LanguageCode::EnUs,
            vowels,
            consonants,
            features,
            notation_sets: vec![create_arpabet_set(), create_ipa_set(), create_sampa_set()],
        }
    }

    /// Create German phoneme inventory
    #[must_use]
    pub fn german() -> Self {
        let mut vowels = HashSet::new();
        vowels.insert("i".to_string());
        vowels.insert("ɪ".to_string());
        vowels.insert("e".to_string());
        vowels.insert("ɛ".to_string());
        vowels.insert("a".to_string());
        vowels.insert("ɑ".to_string());
        vowels.insert("ɔ".to_string());
        vowels.insert("o".to_string());
        vowels.insert("ʊ".to_string());
        vowels.insert("u".to_string());
        vowels.insert("y".to_string()); // German specific
        vowels.insert("ʏ".to_string()); // German specific
        vowels.insert("ø".to_string()); // German specific
        vowels.insert("œ".to_string()); // German specific
        vowels.insert("ə".to_string());

        let mut consonants = HashSet::new();
        consonants.insert("p".to_string());
        consonants.insert("b".to_string());
        consonants.insert("t".to_string());
        consonants.insert("d".to_string());
        consonants.insert("k".to_string());
        consonants.insert("g".to_string());
        consonants.insert("f".to_string());
        consonants.insert("v".to_string());
        consonants.insert("s".to_string());
        consonants.insert("z".to_string());
        consonants.insert("ʃ".to_string());
        consonants.insert("ʒ".to_string());
        consonants.insert("x".to_string()); // German specific
        consonants.insert("h".to_string());
        consonants.insert("m".to_string());
        consonants.insert("n".to_string());
        consonants.insert("ŋ".to_string());
        consonants.insert("l".to_string());
        consonants.insert("r".to_string());
        consonants.insert("ʁ".to_string()); // German r
        consonants.insert("j".to_string());
        consonants.insert("pf".to_string()); // German affricate
        consonants.insert("ts".to_string()); // German affricate

        Self {
            language: LanguageCode::DeDe,
            vowels,
            consonants,
            features: HashMap::new(), // Simplified for now
            notation_sets: vec![create_ipa_set(), create_sampa_set()],
        }
    }

    /// Create Japanese phoneme inventory
    #[must_use]
    pub fn japanese() -> Self {
        let mut vowels = HashSet::new();
        vowels.insert("a".to_string());
        vowels.insert("i".to_string());
        vowels.insert("u".to_string());
        vowels.insert("e".to_string());
        vowels.insert("o".to_string());

        let mut consonants = HashSet::new();
        consonants.insert("k".to_string());
        consonants.insert("g".to_string());
        consonants.insert("s".to_string());
        consonants.insert("z".to_string());
        consonants.insert("t".to_string());
        consonants.insert("d".to_string());
        consonants.insert("n".to_string());
        consonants.insert("h".to_string());
        consonants.insert("b".to_string());
        consonants.insert("p".to_string());
        consonants.insert("m".to_string());
        consonants.insert("j".to_string());
        consonants.insert("r".to_string());
        consonants.insert("w".to_string());
        consonants.insert("ʃ".to_string());
        consonants.insert("tʃ".to_string());
        consonants.insert("ts".to_string());
        consonants.insert("f".to_string());

        Self {
            language: LanguageCode::JaJp,
            vowels,
            consonants,
            features: HashMap::new(), // Simplified for now
            notation_sets: vec![create_ipa_set()],
        }
    }

    /// Create Mandarin Chinese phoneme inventory
    #[must_use]
    pub fn chinese_mandarin() -> Self {
        let mut vowels = HashSet::new();
        // Basic vowels
        vowels.insert("a".to_string());
        vowels.insert("o".to_string());
        vowels.insert("e".to_string());
        vowels.insert("i".to_string());
        vowels.insert("u".to_string());
        vowels.insert("ü".to_string()); // /y/
                                        // Diphthongs
        vowels.insert("ai".to_string());
        vowels.insert("ei".to_string());
        vowels.insert("ao".to_string());
        vowels.insert("ou".to_string());
        vowels.insert("ia".to_string());
        vowels.insert("ie".to_string());
        vowels.insert("ua".to_string());
        vowels.insert("uo".to_string());
        vowels.insert("üe".to_string());

        let mut consonants = HashSet::new();
        // Stops
        consonants.insert("p".to_string());
        consonants.insert("pʰ".to_string());
        consonants.insert("t".to_string());
        consonants.insert("tʰ".to_string());
        consonants.insert("k".to_string());
        consonants.insert("kʰ".to_string());
        // Affricates
        consonants.insert("ts".to_string());
        consonants.insert("tsʰ".to_string());
        consonants.insert("tʂ".to_string());
        consonants.insert("tʂʰ".to_string());
        consonants.insert("tɕ".to_string());
        consonants.insert("tɕʰ".to_string());
        // Fricatives
        consonants.insert("f".to_string());
        consonants.insert("s".to_string());
        consonants.insert("ʂ".to_string());
        consonants.insert("ɕ".to_string());
        consonants.insert("x".to_string());
        consonants.insert("h".to_string());
        // Nasals and liquids
        consonants.insert("m".to_string());
        consonants.insert("n".to_string());
        consonants.insert("ŋ".to_string());
        consonants.insert("l".to_string());
        consonants.insert("r".to_string());
        // Approximants
        consonants.insert("j".to_string());
        consonants.insert("w".to_string());

        Self {
            language: LanguageCode::ZhCn,
            vowels,
            consonants,
            features: HashMap::new(), // Simplified for now
            notation_sets: vec![create_ipa_set()],
        }
    }

    /// Create Spanish phoneme inventory
    #[must_use]
    pub fn spanish() -> Self {
        let mut vowels = HashSet::new();
        // Spanish has 5 cardinal vowels
        vowels.insert("a".to_string());
        vowels.insert("e".to_string());
        vowels.insert("i".to_string());
        vowels.insert("o".to_string());
        vowels.insert("u".to_string());

        let mut consonants = HashSet::new();
        // Stops
        consonants.insert("p".to_string());
        consonants.insert("b".to_string());
        consonants.insert("t".to_string());
        consonants.insert("d".to_string());
        consonants.insert("k".to_string());
        consonants.insert("g".to_string());
        // Fricatives
        consonants.insert("f".to_string());
        consonants.insert("β".to_string());
        consonants.insert("θ".to_string()); // Castilian 'th'
        consonants.insert("s".to_string());
        consonants.insert("ð".to_string());
        consonants.insert("ɣ".to_string());
        consonants.insert("x".to_string()); // 'j' sound
                                            // Affricates
        consonants.insert("tʃ".to_string());
        // Nasals
        consonants.insert("m".to_string());
        consonants.insert("n".to_string());
        consonants.insert("ɲ".to_string()); // 'ñ'
        consonants.insert("ŋ".to_string());
        // Liquids
        consonants.insert("l".to_string());
        consonants.insert("ʎ".to_string()); // 'll'
        consonants.insert("r".to_string());
        consonants.insert("rr".to_string()); // rolled 'rr'
                                             // Approximants
        consonants.insert("j".to_string());
        consonants.insert("w".to_string());

        Self {
            language: LanguageCode::EsEs,
            vowels,
            consonants,
            features: HashMap::new(), // Simplified for now
            notation_sets: vec![create_ipa_set()],
        }
    }

    /// Create French phoneme inventory
    #[must_use]
    pub fn french() -> Self {
        let mut vowels = HashSet::new();
        // Oral vowels
        vowels.insert("a".to_string());
        vowels.insert("e".to_string());
        vowels.insert("ɛ".to_string());
        vowels.insert("i".to_string());
        vowels.insert("o".to_string());
        vowels.insert("ɔ".to_string());
        vowels.insert("u".to_string());
        vowels.insert("y".to_string());
        vowels.insert("ø".to_string());
        vowels.insert("œ".to_string());
        vowels.insert("ə".to_string());
        // Nasal vowels
        vowels.insert("ã".to_string());
        vowels.insert("ɛ̃".to_string());
        vowels.insert("ɔ̃".to_string());
        vowels.insert("œ̃".to_string());

        let mut consonants = HashSet::new();
        // Stops
        consonants.insert("p".to_string());
        consonants.insert("b".to_string());
        consonants.insert("t".to_string());
        consonants.insert("d".to_string());
        consonants.insert("k".to_string());
        consonants.insert("g".to_string());
        // Fricatives
        consonants.insert("f".to_string());
        consonants.insert("v".to_string());
        consonants.insert("s".to_string());
        consonants.insert("z".to_string());
        consonants.insert("ʃ".to_string());
        consonants.insert("ʒ".to_string());
        // Nasals
        consonants.insert("m".to_string());
        consonants.insert("n".to_string());
        consonants.insert("ɲ".to_string());
        consonants.insert("ŋ".to_string());
        // Liquids
        consonants.insert("l".to_string());
        consonants.insert("r".to_string());
        consonants.insert("ʁ".to_string()); // French 'r'
                                            // Approximants
        consonants.insert("j".to_string());
        consonants.insert("w".to_string());
        consonants.insert("ɥ".to_string());

        Self {
            language: LanguageCode::FrFr,
            vowels,
            consonants,
            features: HashMap::new(), // Simplified for now
            notation_sets: vec![create_ipa_set()],
        }
    }

    /// Create Korean phoneme inventory
    #[must_use]
    pub fn korean() -> Self {
        let mut vowels = HashSet::new();
        // Simple vowels
        vowels.insert("a".to_string());
        vowels.insert("ə".to_string()); // eo
        vowels.insert("o".to_string());
        vowels.insert("u".to_string());
        vowels.insert("ɯ".to_string()); // eu
        vowels.insert("i".to_string());
        vowels.insert("ɛ".to_string()); // ae
        vowels.insert("e".to_string());
        // Diphthongs
        vowels.insert("ja".to_string()); // ya
        vowels.insert("jə".to_string()); // yeo
        vowels.insert("jo".to_string()); // yo
        vowels.insert("ju".to_string()); // yu
        vowels.insert("jɛ".to_string()); // yae
        vowels.insert("je".to_string()); // ye
        vowels.insert("wa".to_string());
        vowels.insert("wə".to_string()); // wo
        vowels.insert("wɛ".to_string()); // wae
        vowels.insert("we".to_string());
        vowels.insert("ɰi".to_string()); // ui

        let mut consonants = HashSet::new();
        // Stops (including tensed variants)
        consonants.insert("p".to_string());
        consonants.insert("pʰ".to_string());
        consonants.insert("p*".to_string()); // tensed p
        consonants.insert("t".to_string());
        consonants.insert("tʰ".to_string());
        consonants.insert("t*".to_string()); // tensed t
        consonants.insert("k".to_string());
        consonants.insert("kʰ".to_string());
        consonants.insert("k*".to_string()); // tensed k
                                             // Affricates
        consonants.insert("tʃ".to_string());
        consonants.insert("tʃʰ".to_string());
        consonants.insert("tʃ*".to_string()); // tensed ch
                                              // Fricatives
        consonants.insert("s".to_string());
        consonants.insert("s*".to_string()); // tensed s
        consonants.insert("h".to_string());
        // Nasals
        consonants.insert("m".to_string());
        consonants.insert("n".to_string());
        consonants.insert("ŋ".to_string());
        // Liquids
        consonants.insert("l".to_string());
        consonants.insert("r".to_string());

        Self {
            language: LanguageCode::KoKr,
            vowels,
            consonants,
            features: HashMap::new(), // Simplified for now
            notation_sets: vec![create_ipa_set()],
        }
    }

    /// Check if a phoneme is valid for this language
    #[must_use]
    pub fn is_valid_phoneme(&self, phoneme: &str) -> bool {
        self.vowels.contains(phoneme) || self.consonants.contains(phoneme)
    }

    /// Check if a phoneme is a vowel
    #[must_use]
    pub fn is_vowel(&self, phoneme: &str) -> bool {
        self.vowels.contains(phoneme)
    }

    /// Check if a phoneme is a consonant
    #[must_use]
    pub fn is_consonant(&self, phoneme: &str) -> bool {
        self.consonants.contains(phoneme)
    }

    /// Get phoneme features
    #[must_use]
    pub fn get_features(&self, phoneme: &str) -> Option<&PhonemeFeatures> {
        self.features.get(phoneme)
    }

    /// Get all phonemes for this language
    #[must_use]
    pub fn all_phonemes(&self) -> HashSet<String> {
        let mut all = self.vowels.clone();
        all.extend(self.consonants.clone());
        all
    }

    /// Convert phoneme from one notation to another
    #[must_use]
    pub fn convert_notation(
        &self,
        phoneme: &str,
        from: &PhonemeNotation,
        to: &PhonemeNotation,
    ) -> Option<String> {
        if from == to {
            return Some(phoneme.to_string());
        }

        // Find the from_notation set
        let from_set = self
            .notation_sets
            .iter()
            .find(|set| &set.notation == from)?;

        // Convert to IPA first
        let ipa_phoneme = if *from == PhonemeNotation::IPA {
            phoneme.to_string()
        } else {
            from_set.to_ipa_map.get(phoneme)?.clone()
        };

        // If target is IPA, return IPA
        if *to == PhonemeNotation::IPA {
            return Some(ipa_phoneme);
        }

        // Find the to_notation set and convert from IPA
        let to_set = self.notation_sets.iter().find(|set| &set.notation == to)?;
        to_set.from_ipa_map.get(&ipa_phoneme).cloned()
    }

    /// Get supported notations for this inventory
    #[must_use]
    pub fn supported_notations(&self) -> Vec<&PhonemeNotation> {
        self.notation_sets.iter().map(|set| &set.notation).collect()
    }
}

/// Get phoneme inventory for a language
#[must_use]
pub fn get_phoneme_inventory(language: LanguageCode) -> PhonemeInventory {
    match language {
        LanguageCode::EnUs | LanguageCode::EnGb => PhonemeInventory::english(),
        LanguageCode::DeDe => PhonemeInventory::german(),
        LanguageCode::JaJp => PhonemeInventory::japanese(),
        LanguageCode::ZhCn => PhonemeInventory::chinese_mandarin(),
        LanguageCode::EsEs => PhonemeInventory::spanish(),
        LanguageCode::FrFr => PhonemeInventory::french(),
        LanguageCode::KoKr => PhonemeInventory::korean(),
        _ => PhonemeInventory::english(), // Default fallback
    }
}

/// Calculate phoneme similarity based on features
#[must_use]
pub fn calculate_phoneme_similarity(p1: &str, p2: &str, language: LanguageCode) -> f32 {
    if p1 == p2 {
        return 1.0;
    }

    let inventory = get_phoneme_inventory(language);

    // If either phoneme is not in the inventory, return low similarity
    if !inventory.is_valid_phoneme(p1) || !inventory.is_valid_phoneme(p2) {
        return 0.1;
    }

    // Get features for both phonemes
    let features1 = inventory.get_features(p1);
    let features2 = inventory.get_features(p2);

    match (features1, features2) {
        (Some(f1), Some(f2)) => calculate_feature_similarity(f1, f2),
        _ => {
            // Simple fallback based on category
            if (inventory.is_vowel(p1) && inventory.is_vowel(p2))
                || (inventory.is_consonant(p1) && inventory.is_consonant(p2))
            {
                0.5
            } else {
                0.2
            }
        }
    }
}

/// Calculate similarity between phoneme features
fn calculate_feature_similarity(f1: &PhonemeFeatures, f2: &PhonemeFeatures) -> f32 {
    let mut similarity = 0.0;
    let mut total_features = 0;

    // Compare place of articulation
    if let (Some(p1), Some(p2)) = (&f1.place, &f2.place) {
        if p1 == p2 {
            similarity += 1.0;
        }
        total_features += 1;
    }

    // Compare manner of articulation
    if let (Some(m1), Some(m2)) = (&f1.manner, &f2.manner) {
        if m1 == m2 {
            similarity += 1.0;
        }
        total_features += 1;
    }

    // Compare voicing
    if let (Some(v1), Some(v2)) = (f1.voiced, f2.voiced) {
        if v1 == v2 {
            similarity += 1.0;
        }
        total_features += 1;
    }

    // Compare vowel features
    if let (Some(h1), Some(h2)) = (&f1.height, &f2.height) {
        if h1 == h2 {
            similarity += 1.0;
        }
        total_features += 1;
    }

    if let (Some(b1), Some(b2)) = (&f1.backness, &f2.backness) {
        if b1 == b2 {
            similarity += 1.0;
        }
        total_features += 1;
    }

    if let (Some(r1), Some(r2)) = (f1.rounded, f2.rounded) {
        if r1 == r2 {
            similarity += 1.0;
        }
        total_features += 1;
    }

    if total_features > 0 {
        similarity / total_features as f32
    } else {
        0.5 // Default similarity when no features are available
    }
}

/// Create ARPABET phoneme set (CMU Pronouncing Dictionary standard)
#[must_use]
pub fn create_arpabet_set() -> PhonemeSet {
    let mut to_ipa = HashMap::new();
    let mut from_ipa = HashMap::new();

    // ARPABET vowels to IPA
    to_ipa.insert("AA".to_string(), "ɑ".to_string());
    to_ipa.insert("AE".to_string(), "æ".to_string());
    to_ipa.insert("AH".to_string(), "ʌ".to_string());
    to_ipa.insert("AO".to_string(), "ɔ".to_string());
    to_ipa.insert("AW".to_string(), "aʊ".to_string());
    to_ipa.insert("AY".to_string(), "aɪ".to_string());
    to_ipa.insert("EH".to_string(), "ɛ".to_string());
    to_ipa.insert("ER".to_string(), "ɝ".to_string());
    to_ipa.insert("EY".to_string(), "eɪ".to_string());
    to_ipa.insert("IH".to_string(), "ɪ".to_string());
    to_ipa.insert("IY".to_string(), "i".to_string());
    to_ipa.insert("OW".to_string(), "oʊ".to_string());
    to_ipa.insert("OY".to_string(), "ɔɪ".to_string());
    to_ipa.insert("UH".to_string(), "ʊ".to_string());
    to_ipa.insert("UW".to_string(), "u".to_string());

    // ARPABET consonants to IPA
    to_ipa.insert("B".to_string(), "b".to_string());
    to_ipa.insert("CH".to_string(), "tʃ".to_string());
    to_ipa.insert("D".to_string(), "d".to_string());
    to_ipa.insert("DH".to_string(), "ð".to_string());
    to_ipa.insert("F".to_string(), "f".to_string());
    to_ipa.insert("G".to_string(), "g".to_string());
    to_ipa.insert("HH".to_string(), "h".to_string());
    to_ipa.insert("JH".to_string(), "dʒ".to_string());
    to_ipa.insert("K".to_string(), "k".to_string());
    to_ipa.insert("L".to_string(), "l".to_string());
    to_ipa.insert("M".to_string(), "m".to_string());
    to_ipa.insert("N".to_string(), "n".to_string());
    to_ipa.insert("NG".to_string(), "ŋ".to_string());
    to_ipa.insert("P".to_string(), "p".to_string());
    to_ipa.insert("R".to_string(), "r".to_string());
    to_ipa.insert("S".to_string(), "s".to_string());
    to_ipa.insert("SH".to_string(), "ʃ".to_string());
    to_ipa.insert("T".to_string(), "t".to_string());
    to_ipa.insert("TH".to_string(), "θ".to_string());
    to_ipa.insert("V".to_string(), "v".to_string());
    to_ipa.insert("W".to_string(), "w".to_string());
    to_ipa.insert("Y".to_string(), "j".to_string());
    to_ipa.insert("Z".to_string(), "z".to_string());
    to_ipa.insert("ZH".to_string(), "ʒ".to_string());

    // Create reverse mapping
    for (arpabet, ipa) in &to_ipa {
        from_ipa.insert(ipa.clone(), arpabet.clone());
    }

    PhonemeSet {
        name: "ARPABET".to_string(),
        notation: PhonemeNotation::ARPABET,
        to_ipa_map: to_ipa,
        from_ipa_map: from_ipa,
        languages: vec![LanguageCode::EnUs, LanguageCode::EnGb],
    }
}

/// Create IPA phoneme set (identity mapping)
#[must_use]
pub fn create_ipa_set() -> PhonemeSet {
    let to_ipa = HashMap::new(); // Identity mapping, so empty
    let from_ipa = HashMap::new();

    PhonemeSet {
        name: "IPA".to_string(),
        notation: PhonemeNotation::IPA,
        to_ipa_map: to_ipa,
        from_ipa_map: from_ipa,
        languages: vec![], // Universal
    }
}

/// Create SAMPA phoneme set
#[must_use]
pub fn create_sampa_set() -> PhonemeSet {
    let mut to_ipa = HashMap::new();
    let mut from_ipa = HashMap::new();

    // SAMPA to IPA mappings (English subset)
    to_ipa.insert("i:".to_string(), "i".to_string());
    to_ipa.insert("I".to_string(), "ɪ".to_string());
    to_ipa.insert("e".to_string(), "e".to_string());
    to_ipa.insert("E".to_string(), "ɛ".to_string());
    to_ipa.insert("{".to_string(), "æ".to_string());
    to_ipa.insert("A:".to_string(), "ɑ".to_string());
    to_ipa.insert("Q".to_string(), "ɒ".to_string());
    to_ipa.insert("O:".to_string(), "ɔ".to_string());
    to_ipa.insert("U".to_string(), "ʊ".to_string());
    to_ipa.insert("u:".to_string(), "u".to_string());
    to_ipa.insert("V".to_string(), "ʌ".to_string());
    to_ipa.insert("@".to_string(), "ə".to_string());

    // SAMPA consonants to IPA
    to_ipa.insert("p".to_string(), "p".to_string());
    to_ipa.insert("b".to_string(), "b".to_string());
    to_ipa.insert("t".to_string(), "t".to_string());
    to_ipa.insert("d".to_string(), "d".to_string());
    to_ipa.insert("k".to_string(), "k".to_string());
    to_ipa.insert("g".to_string(), "g".to_string());
    to_ipa.insert("f".to_string(), "f".to_string());
    to_ipa.insert("v".to_string(), "v".to_string());
    to_ipa.insert("T".to_string(), "θ".to_string());
    to_ipa.insert("D".to_string(), "ð".to_string());
    to_ipa.insert("s".to_string(), "s".to_string());
    to_ipa.insert("z".to_string(), "z".to_string());
    to_ipa.insert("S".to_string(), "ʃ".to_string());
    to_ipa.insert("Z".to_string(), "ʒ".to_string());
    to_ipa.insert("h".to_string(), "h".to_string());
    to_ipa.insert("m".to_string(), "m".to_string());
    to_ipa.insert("n".to_string(), "n".to_string());
    to_ipa.insert("N".to_string(), "ŋ".to_string());
    to_ipa.insert("l".to_string(), "l".to_string());
    to_ipa.insert("r".to_string(), "r".to_string());
    to_ipa.insert("w".to_string(), "w".to_string());
    to_ipa.insert("j".to_string(), "j".to_string());
    to_ipa.insert("tS".to_string(), "tʃ".to_string());
    to_ipa.insert("dZ".to_string(), "dʒ".to_string());

    // Create reverse mapping
    for (sampa, ipa) in &to_ipa {
        from_ipa.insert(ipa.clone(), sampa.clone());
    }

    PhonemeSet {
        name: "SAMPA".to_string(),
        notation: PhonemeNotation::SAMPA,
        to_ipa_map: to_ipa,
        from_ipa_map: from_ipa,
        languages: vec![LanguageCode::EnUs, LanguageCode::EnGb, LanguageCode::DeDe],
    }
}

/// Create a custom phoneme set
#[must_use]
pub fn create_custom_phoneme_set(
    name: String,
    mappings: HashMap<String, String>,
    languages: Vec<LanguageCode>,
) -> PhonemeSet {
    let mut from_ipa = HashMap::new();

    // Create reverse mapping from IPA
    for (custom, ipa) in &mappings {
        from_ipa.insert(ipa.clone(), custom.clone());
    }

    PhonemeSet {
        name: name.clone(),
        notation: PhonemeNotation::Custom(name),
        to_ipa_map: mappings,
        from_ipa_map: from_ipa,
        languages,
    }
}

/// Phoneme set utilities
pub mod notation_utils {
    use super::{calculate_phoneme_similarity, PhonemeInventory, PhonemeNotation};

    /// Convert a sequence of phonemes between notations
    #[must_use]
    pub fn convert_phoneme_sequence(
        phonemes: &[String],
        inventory: &PhonemeInventory,
        from: &PhonemeNotation,
        to: &PhonemeNotation,
    ) -> Vec<Option<String>> {
        phonemes
            .iter()
            .map(|p| inventory.convert_notation(p, from, to))
            .collect()
    }

    /// Validate phoneme against notation system
    #[must_use]
    pub fn validate_phoneme_notation(
        phoneme: &str,
        notation: &PhonemeNotation,
        inventory: &PhonemeInventory,
    ) -> bool {
        if let Some(set) = inventory
            .notation_sets
            .iter()
            .find(|s| &s.notation == notation)
        {
            match notation {
                PhonemeNotation::IPA => {
                    // For IPA, check if it's in the inventory
                    inventory.is_valid_phoneme(phoneme)
                }
                _ => {
                    // For other notations, check if mapping exists
                    set.to_ipa_map.contains_key(phoneme)
                }
            }
        } else {
            false
        }
    }

    /// Get all valid phonemes for a notation system
    #[must_use]
    pub fn get_valid_phonemes_for_notation(
        notation: &PhonemeNotation,
        inventory: &PhonemeInventory,
    ) -> Vec<String> {
        if let Some(set) = inventory
            .notation_sets
            .iter()
            .find(|s| &s.notation == notation)
        {
            match notation {
                PhonemeNotation::IPA => inventory.all_phonemes().into_iter().collect(),
                _ => set.to_ipa_map.keys().cloned().collect(),
            }
        } else {
            Vec::new()
        }
    }

    /// Find best notation match for a phoneme
    #[must_use]
    pub fn find_best_notation_match(
        phoneme: &str,
        target_notation: &PhonemeNotation,
        inventory: &PhonemeInventory,
    ) -> Option<String> {
        // Try direct conversion first
        if let Some(converted) =
            inventory.convert_notation(phoneme, &PhonemeNotation::IPA, target_notation)
        {
            return Some(converted);
        }

        // Try finding similar phonemes
        let valid_phonemes = get_valid_phonemes_for_notation(target_notation, inventory);
        let mut best_match: Option<(String, f32)> = None;

        for candidate in valid_phonemes {
            if let Some(candidate_ipa) =
                inventory.convert_notation(&candidate, target_notation, &PhonemeNotation::IPA)
            {
                let similarity =
                    calculate_phoneme_similarity(phoneme, &candidate_ipa, inventory.language);

                if let Some((_, best_sim)) = &best_match {
                    if similarity > *best_sim {
                        best_match = Some((candidate, similarity));
                    }
                } else {
                    best_match = Some((candidate, similarity));
                }
            }
        }

        best_match.map(|(phoneme, _)| phoneme)
    }
}

/// Cross-linguistic phoneme mapping for accent adaptation and multilingual processing
#[derive(Debug, Clone)]
pub struct CrossLinguisticMapper {
    /// Language mappings
    language_mappings: HashMap<(LanguageCode, LanguageCode), HashMap<String, String>>,
}

impl CrossLinguisticMapper {
    /// Create a new cross-linguistic mapper with predefined mappings
    #[must_use]
    pub fn new() -> Self {
        let mut mapper = Self {
            language_mappings: HashMap::new(),
        };

        // Initialize common cross-linguistic mappings
        mapper.initialize_common_mappings();
        mapper
    }

    /// Initialize common phoneme mappings between supported languages
    fn initialize_common_mappings(&mut self) {
        // English to Spanish mappings (common for Spanish speakers learning English)
        let mut en_to_es = HashMap::new();
        en_to_es.insert("θ".to_string(), "s".to_string()); // English 'th' -> Spanish 's'
        en_to_es.insert("ð".to_string(), "d".to_string()); // Voiced 'th' -> 'd'
        en_to_es.insert("ʃ".to_string(), "tʃ".to_string()); // 'sh' -> 'ch'
        en_to_es.insert("ʒ".to_string(), "j".to_string()); // 'zh' -> 'y'
        en_to_es.insert("v".to_string(), "b".to_string()); // 'v' -> 'b'
        en_to_es.insert("ɪ".to_string(), "i".to_string()); // Near-high front -> high front
        en_to_es.insert("ʊ".to_string(), "u".to_string()); // Near-high back -> high back
        self.language_mappings
            .insert((LanguageCode::EnUs, LanguageCode::EsEs), en_to_es);

        // Spanish to English mappings (reverse direction)
        let mut es_to_en = HashMap::new();
        es_to_en.insert("β".to_string(), "b".to_string()); // Spanish soft 'b' -> English 'b'
        es_to_en.insert("ð".to_string(), "ð".to_string()); // Keep voiced 'th'
        es_to_en.insert("ɣ".to_string(), "g".to_string()); // Spanish soft 'g' -> English 'g'
        es_to_en.insert("x".to_string(), "h".to_string()); // Spanish 'j' -> English 'h'
        es_to_en.insert("ɲ".to_string(), "nj".to_string()); // Spanish 'ñ' -> English 'ny'
        es_to_en.insert("rr".to_string(), "r".to_string()); // Rolled 'rr' -> English 'r'
        self.language_mappings
            .insert((LanguageCode::EsEs, LanguageCode::EnUs), es_to_en);

        // Chinese to English mappings (Mandarin speakers learning English)
        let mut zh_to_en = HashMap::new();
        zh_to_en.insert("pʰ".to_string(), "p".to_string()); // Aspirated p -> p
        zh_to_en.insert("tʰ".to_string(), "t".to_string()); // Aspirated t -> t
        zh_to_en.insert("kʰ".to_string(), "k".to_string()); // Aspirated k -> k
        zh_to_en.insert("tɕ".to_string(), "tʃ".to_string()); // Mandarin 'q' -> English 'ch'
        zh_to_en.insert("tɕʰ".to_string(), "tʃ".to_string()); // Mandarin 'ch' -> English 'ch'
        zh_to_en.insert("ɕ".to_string(), "ʃ".to_string()); // Mandarin 'x' -> English 'sh'
        zh_to_en.insert("ts".to_string(), "ts".to_string()); // Keep 'ts'
        zh_to_en.insert("ʂ".to_string(), "ʃ".to_string()); // Retroflex 'sh' -> post-alveolar 'sh'
        self.language_mappings
            .insert((LanguageCode::ZhCn, LanguageCode::EnUs), zh_to_en);

        // Japanese to English mappings (common substitutions)
        let mut ja_to_en = HashMap::new();
        ja_to_en.insert("ts".to_string(), "s".to_string()); // Japanese 'tsu' -> English 's'
        ja_to_en.insert("tʃ".to_string(), "tʃ".to_string()); // Keep 'ch'
        ja_to_en.insert("r".to_string(), "l".to_string()); // Japanese 'r' -> English 'l' (common confusion)
        ja_to_en.insert("f".to_string(), "h".to_string()); // Japanese 'fu' -> English 'h'
        self.language_mappings
            .insert((LanguageCode::JaJp, LanguageCode::EnUs), ja_to_en);

        // Korean to English mappings
        let mut ko_to_en = HashMap::new();
        ko_to_en.insert("pʰ".to_string(), "p".to_string()); // Aspirated p -> p
        ko_to_en.insert("tʰ".to_string(), "t".to_string()); // Aspirated t -> t
        ko_to_en.insert("kʰ".to_string(), "k".to_string()); // Aspirated k -> k
        ko_to_en.insert("p*".to_string(), "p".to_string()); // Tensed p -> p
        ko_to_en.insert("t*".to_string(), "t".to_string()); // Tensed t -> t
        ko_to_en.insert("k*".to_string(), "k".to_string()); // Tensed k -> k
        ko_to_en.insert("tʃ*".to_string(), "tʃ".to_string()); // Tensed ch -> ch
        ko_to_en.insert("s*".to_string(), "s".to_string()); // Tensed s -> s
        self.language_mappings
            .insert((LanguageCode::KoKr, LanguageCode::EnUs), ko_to_en);

        // French to English mappings
        let mut fr_to_en = HashMap::new();
        fr_to_en.insert("ʁ".to_string(), "r".to_string()); // French uvular 'r' -> English 'r'
        fr_to_en.insert("ɲ".to_string(), "nj".to_string()); // French 'gn' -> English 'ny'
        fr_to_en.insert("y".to_string(), "u".to_string()); // French 'u' -> English 'oo'
        fr_to_en.insert("ø".to_string(), "ə".to_string()); // French 'eu' -> English schwa
        fr_to_en.insert("œ".to_string(), "ʌ".to_string()); // French 'œ' -> English 'uh'
        fr_to_en.insert("ã".to_string(), "æ".to_string()); // French nasal 'an' -> English 'a'
        fr_to_en.insert("ɛ̃".to_string(), "ɛ".to_string()); // French nasal 'in' -> English 'e'
        fr_to_en.insert("ɔ̃".to_string(), "ɔ".to_string()); // French nasal 'on' -> English 'o'
        self.language_mappings
            .insert((LanguageCode::FrFr, LanguageCode::EnUs), fr_to_en);

        // German to English mappings
        let mut de_to_en = HashMap::new();
        de_to_en.insert("x".to_string(), "k".to_string()); // German 'ach' -> English 'k'
        de_to_en.insert("ʁ".to_string(), "r".to_string()); // German uvular 'r' -> English 'r'
        de_to_en.insert("pf".to_string(), "f".to_string()); // German 'pf' -> English 'f'
        de_to_en.insert("ts".to_string(), "s".to_string()); // German 'z' -> English 's'
        self.language_mappings
            .insert((LanguageCode::DeDe, LanguageCode::EnUs), de_to_en);
    }

    /// Map a phoneme from source language to target language
    #[must_use]
    pub fn map_phoneme(
        &self,
        phoneme: &str,
        from: LanguageCode,
        to: LanguageCode,
    ) -> Option<String> {
        if from == to {
            return Some(phoneme.to_string());
        }

        // Check for direct mapping
        if let Some(mapping) = self.language_mappings.get(&(from, to)) {
            if let Some(mapped) = mapping.get(phoneme) {
                return Some(mapped.clone());
            }
        }

        // Try feature-based mapping as fallback
        self.feature_based_mapping(phoneme, from, to)
    }

    /// Feature-based phoneme mapping when no direct mapping exists
    fn feature_based_mapping(
        &self,
        phoneme: &str,
        from: LanguageCode,
        to: LanguageCode,
    ) -> Option<String> {
        let _from_inventory = get_phoneme_inventory(from);
        let to_inventory = get_phoneme_inventory(to);

        // Find the most similar phoneme in target language
        let mut best_match = None;
        let mut best_similarity = 0.0;

        for target_phoneme in to_inventory.all_phonemes() {
            let similarity = calculate_phoneme_similarity(phoneme, &target_phoneme, from);
            if similarity > best_similarity {
                best_similarity = similarity;
                best_match = Some(target_phoneme);
            }
        }

        // Only return match if similarity is above threshold
        if best_similarity > 0.5 {
            best_match
        } else {
            None
        }
    }

    /// Map an entire phoneme sequence from source to target language
    #[must_use]
    pub fn map_phoneme_sequence(
        &self,
        phonemes: &[String],
        from: LanguageCode,
        to: LanguageCode,
    ) -> Vec<Option<String>> {
        phonemes
            .iter()
            .map(|p| self.map_phoneme(p, from, to))
            .collect()
    }

    /// Add a custom mapping between two languages
    pub fn add_custom_mapping(
        &mut self,
        from: LanguageCode,
        to: LanguageCode,
        mappings: HashMap<String, String>,
    ) {
        self.language_mappings.insert((from, to), mappings);
    }

    /// Get all supported language pairs for mapping
    #[must_use]
    pub fn supported_language_pairs(&self) -> Vec<(LanguageCode, LanguageCode)> {
        self.language_mappings.keys().copied().collect()
    }

    /// Check if mapping is available between two languages
    #[must_use]
    pub fn has_mapping(&self, from: LanguageCode, to: LanguageCode) -> bool {
        self.language_mappings.contains_key(&(from, to)) || from == to
    }
}

impl Default for CrossLinguisticMapper {
    fn default() -> Self {
        Self::new()
    }
}

/// Create a global cross-linguistic mapper instance
#[must_use]
pub fn create_cross_linguistic_mapper() -> CrossLinguisticMapper {
    CrossLinguisticMapper::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_english_inventory() {
        let inventory = PhonemeInventory::english();
        assert!(inventory.is_vowel("i"));
        assert!(inventory.is_consonant("p"));
        assert!(inventory.is_valid_phoneme("ə"));
        assert!(!inventory.is_valid_phoneme("x")); // Not in English
    }

    #[test]
    fn test_phoneme_similarity() {
        let similarity = calculate_phoneme_similarity("p", "b", LanguageCode::EnUs);
        assert!(similarity > 0.5); // Similar phonemes (both bilabial stops)

        let similarity = calculate_phoneme_similarity("p", "i", LanguageCode::EnUs);
        assert!(similarity < 0.7); // Different categories (adjust threshold)

        // Test identical phonemes
        let identity_similarity = calculate_phoneme_similarity("p", "p", LanguageCode::EnUs);
        assert_eq!(identity_similarity, 1.0);

        // Test very different phonemes
        let different_similarity = calculate_phoneme_similarity("k", "a", LanguageCode::EnUs);
        assert!(different_similarity < 0.5);
    }

    #[test]
    fn test_get_inventory() {
        let inventory = get_phoneme_inventory(LanguageCode::EnUs);
        assert_eq!(inventory.language, LanguageCode::EnUs);

        let inventory = get_phoneme_inventory(LanguageCode::DeDe);
        assert_eq!(inventory.language, LanguageCode::DeDe);
    }

    #[test]
    fn test_arpabet_notation() {
        let arpabet_set = create_arpabet_set();
        assert_eq!(arpabet_set.name, "ARPABET");
        assert_eq!(arpabet_set.notation, PhonemeNotation::ARPABET);

        // Test some mappings
        assert_eq!(arpabet_set.to_ipa_map.get("AA"), Some(&"ɑ".to_string()));
        assert_eq!(arpabet_set.to_ipa_map.get("IY"), Some(&"i".to_string()));
        assert_eq!(arpabet_set.to_ipa_map.get("P"), Some(&"p".to_string()));
        assert_eq!(arpabet_set.to_ipa_map.get("CH"), Some(&"tʃ".to_string()));
    }

    #[test]
    fn test_sampa_notation() {
        let sampa_set = create_sampa_set();
        assert_eq!(sampa_set.name, "SAMPA");
        assert_eq!(sampa_set.notation, PhonemeNotation::SAMPA);

        // Test some mappings
        assert_eq!(sampa_set.to_ipa_map.get("I"), Some(&"ɪ".to_string()));
        assert_eq!(sampa_set.to_ipa_map.get("{"), Some(&"æ".to_string()));
        assert_eq!(sampa_set.to_ipa_map.get("S"), Some(&"ʃ".to_string()));
        assert_eq!(sampa_set.to_ipa_map.get("tS"), Some(&"tʃ".to_string()));
    }

    #[test]
    fn test_notation_conversion() {
        let inventory = PhonemeInventory::english();

        // Test ARPABET to IPA conversion
        let result =
            inventory.convert_notation("IY", &PhonemeNotation::ARPABET, &PhonemeNotation::IPA);
        assert_eq!(result, Some("i".to_string()));

        // Test IPA to ARPABET conversion
        let result =
            inventory.convert_notation("i", &PhonemeNotation::IPA, &PhonemeNotation::ARPABET);
        assert_eq!(result, Some("IY".to_string()));

        // Test ARPABET to SAMPA conversion
        let result =
            inventory.convert_notation("IY", &PhonemeNotation::ARPABET, &PhonemeNotation::SAMPA);
        assert_eq!(result, Some("i:".to_string()));

        // Test identity conversion
        let result = inventory.convert_notation("p", &PhonemeNotation::IPA, &PhonemeNotation::IPA);
        assert_eq!(result, Some("p".to_string()));
    }

    #[test]
    fn test_custom_phoneme_set() {
        let mut mappings = HashMap::new();
        mappings.insert("ph1".to_string(), "p".to_string());
        mappings.insert("ph2".to_string(), "b".to_string());

        let custom_set =
            create_custom_phoneme_set("Custom".to_string(), mappings, vec![LanguageCode::EnUs]);

        assert_eq!(custom_set.name, "Custom");
        assert_eq!(
            custom_set.notation,
            PhonemeNotation::Custom("Custom".to_string())
        );
        assert_eq!(custom_set.to_ipa_map.get("ph1"), Some(&"p".to_string()));
        assert_eq!(custom_set.from_ipa_map.get("p"), Some(&"ph1".to_string()));
    }

    #[test]
    fn test_notation_validation() {
        let inventory = PhonemeInventory::english();

        // Test valid ARPABET phoneme
        assert!(notation_utils::validate_phoneme_notation(
            "IY",
            &PhonemeNotation::ARPABET,
            &inventory
        ));

        // Test invalid ARPABET phoneme
        assert!(!notation_utils::validate_phoneme_notation(
            "XX",
            &PhonemeNotation::ARPABET,
            &inventory
        ));

        // Test valid IPA phoneme
        assert!(notation_utils::validate_phoneme_notation(
            "i",
            &PhonemeNotation::IPA,
            &inventory
        ));

        // Test invalid IPA phoneme for English
        assert!(!notation_utils::validate_phoneme_notation(
            "x",
            &PhonemeNotation::IPA,
            &inventory
        ));
    }

    #[test]
    fn test_phoneme_sequence_conversion() {
        let inventory = PhonemeInventory::english();
        let arpabet_sequence = vec![
            "HH".to_string(),
            "EH".to_string(),
            "L".to_string(),
            "OW".to_string(),
        ];

        let ipa_sequence = notation_utils::convert_phoneme_sequence(
            &arpabet_sequence,
            &inventory,
            &PhonemeNotation::ARPABET,
            &PhonemeNotation::IPA,
        );

        assert_eq!(ipa_sequence.len(), 4);
        assert_eq!(ipa_sequence[0], Some("h".to_string()));
        assert_eq!(ipa_sequence[1], Some("ɛ".to_string()));
        assert_eq!(ipa_sequence[2], Some("l".to_string()));
        assert_eq!(ipa_sequence[3], Some("oʊ".to_string()));
    }

    #[test]
    fn test_get_valid_phonemes_for_notation() {
        let inventory = PhonemeInventory::english();

        let arpabet_phonemes =
            notation_utils::get_valid_phonemes_for_notation(&PhonemeNotation::ARPABET, &inventory);

        assert!(arpabet_phonemes.contains(&"IY".to_string()));
        assert!(arpabet_phonemes.contains(&"P".to_string()));
        assert!(arpabet_phonemes.contains(&"CH".to_string()));
        assert!(!arpabet_phonemes.is_empty());

        let ipa_phonemes =
            notation_utils::get_valid_phonemes_for_notation(&PhonemeNotation::IPA, &inventory);

        assert!(ipa_phonemes.contains(&"i".to_string()));
        assert!(ipa_phonemes.contains(&"p".to_string()));
        assert!(!ipa_phonemes.is_empty());
    }
}

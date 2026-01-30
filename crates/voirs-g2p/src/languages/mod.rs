//! Language-specific phoneme inventories and rules.
//!
//! This module provides comprehensive phoneme inventories for supported languages,
//! including detailed phonetic information and language-specific processing rules.

pub mod arabic;
pub mod chinese;
pub mod japanese;
pub mod korean;
pub mod russian;

use crate::{LanguageCode, Phoneme};
use std::collections::HashMap;

/// Phoneme inventory for a language
#[derive(Debug, Clone)]
pub struct PhonemeInventory {
    /// Language code
    pub language: LanguageCode,
    /// Consonants with IPA symbols
    pub consonants: Vec<PhonemeInfo>,
    /// Vowels with IPA symbols
    pub vowels: Vec<PhonemeInfo>,
    /// Additional sounds (clicks, ejectives, etc.)
    pub other_sounds: Vec<PhonemeInfo>,
    /// Grapheme to phoneme mappings
    pub grapheme_mappings: HashMap<String, Vec<String>>,
}

/// Detailed information about a phoneme
#[derive(Debug, Clone)]
pub struct PhonemeInfo {
    /// IPA symbol
    pub symbol: String,
    /// Phoneme type (consonant, vowel, etc.)
    pub phoneme_type: PhonemeType,
    /// Place of articulation (for consonants)
    pub place: Option<PlaceOfArticulation>,
    /// Manner of articulation (for consonants)
    pub manner: Option<MannerOfArticulation>,
    /// Voicing
    pub voiced: bool,
    /// Vowel height (for vowels)
    pub height: Option<VowelHeight>,
    /// Vowel frontness (for vowels)
    pub frontness: Option<VowelFrontness>,
    /// Roundedness (for vowels)
    pub rounded: Option<bool>,
    /// Example graphemes that produce this phoneme
    pub example_graphemes: Vec<String>,
}

/// Type of phoneme
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhonemeType {
    Consonant,
    Vowel,
    Diphthong,
    Other,
}

/// Place of articulation for consonants
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaceOfArticulation {
    Bilabial,
    Labiodental,
    Dental,
    Alveolar,
    Postalveolar,
    Retroflex,
    Palatal,
    Velar,
    Uvular,
    Pharyngeal,
    Glottal,
}

/// Manner of articulation for consonants
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MannerOfArticulation {
    Plosive,
    Nasal,
    Trill,
    TapFlap,
    Fricative,
    LateralFricative,
    Approximant,
    LateralApproximant,
    Affricate,
}

/// Vowel height
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VowelHeight {
    Close,     // i, u
    NearClose, // ɪ, ʊ
    CloseMid,  // e, o
    Mid,       // ə
    OpenMid,   // ɛ, ɔ
    NearOpen,  // æ
    Open,      // a, ɑ
}

/// Vowel frontness
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VowelFrontness {
    Front,
    Central,
    Back,
}

impl PhonemeInventory {
    /// Get all phonemes in this inventory
    pub fn all_phonemes(&self) -> Vec<Phoneme> {
        let mut phonemes = Vec::new();

        for consonant in &self.consonants {
            phonemes.push(Phoneme::new(consonant.symbol.clone()));
        }

        for vowel in &self.vowels {
            phonemes.push(Phoneme::new(vowel.symbol.clone()));
        }

        for other in &self.other_sounds {
            phonemes.push(Phoneme::new(other.symbol.clone()));
        }

        phonemes
    }

    /// Check if a phoneme symbol is valid for this language
    pub fn is_valid_phoneme(&self, symbol: &str) -> bool {
        self.consonants.iter().any(|p| p.symbol == symbol)
            || self.vowels.iter().any(|p| p.symbol == symbol)
            || self.other_sounds.iter().any(|p| p.symbol == symbol)
    }

    /// Get possible phonemes for a grapheme
    pub fn phonemes_for_grapheme(&self, grapheme: &str) -> Option<&Vec<String>> {
        self.grapheme_mappings.get(grapheme)
    }
}

/// Get the phoneme inventory for a language
pub fn get_inventory(language: LanguageCode) -> Option<PhonemeInventory> {
    match language {
        LanguageCode::Ru => Some(russian::get_russian_inventory()),
        LanguageCode::Ar => Some(arabic::get_arabic_inventory()),
        LanguageCode::ZhCn => Some(chinese::get_chinese_inventory()),
        LanguageCode::Ja => Some(japanese::get_japanese_inventory()),
        LanguageCode::Ko => Some(korean::get_korean_inventory()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_russian_inventory() {
        let inventory = get_inventory(LanguageCode::Ru).expect("Russian inventory should exist");
        assert_eq!(inventory.language, LanguageCode::Ru);
        assert!(
            !inventory.consonants.is_empty(),
            "Russian should have consonants"
        );
        assert!(!inventory.vowels.is_empty(), "Russian should have vowels");
    }

    #[test]
    fn test_arabic_inventory() {
        let inventory = get_inventory(LanguageCode::Ar).expect("Arabic inventory should exist");
        assert_eq!(inventory.language, LanguageCode::Ar);
        assert!(
            !inventory.consonants.is_empty(),
            "Arabic should have consonants"
        );
        assert!(!inventory.vowels.is_empty(), "Arabic should have vowels");
    }

    #[test]
    fn test_chinese_inventory() {
        let inventory = get_inventory(LanguageCode::ZhCn).expect("Chinese inventory should exist");
        assert_eq!(inventory.language, LanguageCode::ZhCn);
        assert!(
            !inventory.consonants.is_empty(),
            "Chinese should have consonants"
        );
        assert!(!inventory.vowels.is_empty(), "Chinese should have vowels");
    }

    #[test]
    fn test_japanese_inventory() {
        let inventory = get_inventory(LanguageCode::Ja).expect("Japanese inventory should exist");
        assert_eq!(inventory.language, LanguageCode::Ja);
        assert!(
            !inventory.consonants.is_empty(),
            "Japanese should have consonants"
        );
        assert!(!inventory.vowels.is_empty(), "Japanese should have vowels");
    }

    #[test]
    fn test_korean_inventory() {
        let inventory = get_inventory(LanguageCode::Ko).expect("Korean inventory should exist");
        assert_eq!(inventory.language, LanguageCode::Ko);
        assert!(
            !inventory.consonants.is_empty(),
            "Korean should have consonants"
        );
        assert!(!inventory.vowels.is_empty(), "Korean should have vowels");
    }

    #[test]
    fn test_all_phonemes() {
        let inventory = get_inventory(LanguageCode::Ru).unwrap();
        let phonemes = inventory.all_phonemes();
        assert!(!phonemes.is_empty(), "Should have phonemes");

        // All phonemes should have symbols
        for phoneme in &phonemes {
            assert!(!phoneme.symbol.is_empty(), "Phoneme should have a symbol");
        }
    }

    #[test]
    fn test_is_valid_phoneme() {
        let inventory = get_inventory(LanguageCode::Ru).unwrap();

        // Test a known Russian consonant
        assert!(
            inventory.is_valid_phoneme("p"),
            "p should be valid in Russian"
        );

        // Test a vowel
        assert!(
            inventory.is_valid_phoneme("a"),
            "a should be valid in Russian"
        );

        // Test an invalid phoneme (e.g., dental fricative θ not in Russian)
        assert!(
            !inventory.is_valid_phoneme("θ"),
            "θ should not be valid in Russian"
        );
    }
}

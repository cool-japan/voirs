//! Korean (Hangul) phoneme inventory and processing.
//!
//! This module provides comprehensive support for Korean phonology, including
//! the complete set of initial consonants (초성 choseong), medial vowels (중성 jungseong),
//! and final consonants (종성 jongseong) used in Hangul syllables.
//!
//! # Phonological Structure
//!
//! Korean syllables follow the structure: (Initial) + Medial + (Final)
//! - Initials (초성): 19 consonants including plain, aspirated, and tense variants
//! - Medials (중성): 21 vowels including monophthongs and diphthongs
//! - Finals (종성): 27 possible final consonants (including clusters)
//!
//! # Consonant System
//!
//! Korean has a three-way contrast in stops and affricates:
//! - Plain (lenis): ㄱ ㄷ ㅂ ㅈ
//! - Aspirated (fortis): ㅋ ㅌ ㅍ ㅊ
//! - Tense (fortis/glottalized): ㄲ ㄸ ㅃ ㅆ ㅉ
//!
//! # Vowel System
//!
//! Korean distinguishes:
//! - 7 monophthongs: ㅏ ㅓ ㅗ ㅜ ㅡ ㅣ ㅐ ㅔ
//! - 13 diphthongs: ㅑ ㅕ ㅛ ㅠ ㅒ ㅖ ㅘ ㅙ ㅚ ㅝ ㅞ ㅟ ㅢ
//!
//! # Example
//!
//! ```rust
//! use voirs_g2p::languages::korean::get_korean_inventory;
//! use voirs_g2p::LanguageCode;
//!
//! let inventory = get_korean_inventory();
//! assert_eq!(inventory.language, LanguageCode::Ko);
//!
//! // Check for specific phonemes
//! assert!(inventory.is_valid_phoneme("k"));   // ㄱ (plain velar)
//! assert!(inventory.is_valid_phoneme("kʰ"));  // ㅋ (aspirated velar)
//! assert!(inventory.is_valid_phoneme("k͈"));   // ㄲ (tense velar)
//! ```

use super::{
    MannerOfArticulation, PhonemeInfo, PhonemeInventory, PhonemeType, PlaceOfArticulation,
    VowelFrontness, VowelHeight,
};
use crate::LanguageCode;
use std::collections::HashMap;

/// Consonant type in Korean three-way contrast
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KoreanConsonantType {
    /// Plain (lenis) consonants: ㄱ ㄷ ㅂ ㅈ
    Plain,
    /// Aspirated consonants: ㅋ ㅌ ㅍ ㅊ
    Aspirated,
    /// Tense (fortis/glottalized) consonants: ㄲ ㄸ ㅃ ㅆ ㅉ
    Tense,
    /// Other consonants (nasals, liquids): ㄴ ㅁ ㅇ ㄹ ㅎ
    Other,
}

/// Position of consonant in syllable
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyllablePosition {
    /// Initial position (초성 choseong)
    Initial,
    /// Final position (종성 jongseong)
    Final,
    /// Can appear in both positions
    Both,
}

/// Get the complete Korean phoneme inventory
pub fn get_korean_inventory() -> PhonemeInventory {
    let consonants = get_korean_consonants();
    let vowels = get_korean_vowels();
    let other_sounds = Vec::new(); // Korean uses consonants and vowels
    let grapheme_mappings = get_korean_grapheme_mappings();

    PhonemeInventory {
        language: LanguageCode::Ko,
        consonants,
        vowels,
        other_sounds,
        grapheme_mappings,
    }
}

/// Korean consonants (자음 jaeum)
///
/// Includes all initial consonants (19) with detailed phonetic information.
/// Korean has a unique three-way contrast in obstruents.
fn get_korean_consonants() -> Vec<PhonemeInfo> {
    vec![
        // Plain stops (평음 pyeongeum)
        PhonemeInfo {
            symbol: "k".to_string(), // ㄱ (giyeok)
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Velar),
            manner: Some(MannerOfArticulation::Plosive),
            voiced: false,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ㄱ".to_string(), "g".to_string(), "k".to_string()],
        },
        PhonemeInfo {
            symbol: "t".to_string(), // ㄷ (digeut)
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Alveolar),
            manner: Some(MannerOfArticulation::Plosive),
            voiced: false,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ㄷ".to_string(), "d".to_string(), "t".to_string()],
        },
        PhonemeInfo {
            symbol: "p".to_string(), // ㅂ (bieup)
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Bilabial),
            manner: Some(MannerOfArticulation::Plosive),
            voiced: false,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ㅂ".to_string(), "b".to_string(), "p".to_string()],
        },
        PhonemeInfo {
            symbol: "tɕ".to_string(), // ㅈ (jieut)
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Palatal),
            manner: Some(MannerOfArticulation::Affricate),
            voiced: false,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ㅈ".to_string(), "j".to_string()],
        },
        // Aspirated stops (격음 gyeogeum)
        PhonemeInfo {
            symbol: "kʰ".to_string(), // ㅋ (kieuk)
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Velar),
            manner: Some(MannerOfArticulation::Plosive),
            voiced: false,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ㅋ".to_string(), "k".to_string()],
        },
        PhonemeInfo {
            symbol: "tʰ".to_string(), // ㅌ (tieut)
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Alveolar),
            manner: Some(MannerOfArticulation::Plosive),
            voiced: false,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ㅌ".to_string(), "t".to_string()],
        },
        PhonemeInfo {
            symbol: "pʰ".to_string(), // ㅍ (pieup)
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Bilabial),
            manner: Some(MannerOfArticulation::Plosive),
            voiced: false,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ㅍ".to_string(), "p".to_string()],
        },
        PhonemeInfo {
            symbol: "tɕʰ".to_string(), // ㅊ (chieut)
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Palatal),
            manner: Some(MannerOfArticulation::Affricate),
            voiced: false,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ㅊ".to_string(), "ch".to_string()],
        },
        // Tense stops (경음 gyeongeum) - using combining diacritic for tenseness
        PhonemeInfo {
            symbol: "k͈".to_string(), // ㄲ (ssangiyeok)
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Velar),
            manner: Some(MannerOfArticulation::Plosive),
            voiced: false,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ㄲ".to_string(), "kk".to_string(), "gg".to_string()],
        },
        PhonemeInfo {
            symbol: "t͈".to_string(), // ㄸ (ssangdigeut)
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Alveolar),
            manner: Some(MannerOfArticulation::Plosive),
            voiced: false,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ㄸ".to_string(), "tt".to_string(), "dd".to_string()],
        },
        PhonemeInfo {
            symbol: "p͈".to_string(), // ㅃ (ssangbieup)
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Bilabial),
            manner: Some(MannerOfArticulation::Plosive),
            voiced: false,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ㅃ".to_string(), "pp".to_string(), "bb".to_string()],
        },
        PhonemeInfo {
            symbol: "t͈ɕ".to_string(), // ㅉ (ssangjieut)
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Palatal),
            manner: Some(MannerOfArticulation::Affricate),
            voiced: false,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ㅉ".to_string(), "jj".to_string()],
        },
        PhonemeInfo {
            symbol: "s͈".to_string(), // ㅆ (ssangsiot)
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Alveolar),
            manner: Some(MannerOfArticulation::Fricative),
            voiced: false,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ㅆ".to_string(), "ss".to_string()],
        },
        // Fricatives and approximants
        PhonemeInfo {
            symbol: "s".to_string(), // ㅅ (siot)
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Alveolar),
            manner: Some(MannerOfArticulation::Fricative),
            voiced: false,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ㅅ".to_string(), "s".to_string()],
        },
        PhonemeInfo {
            symbol: "h".to_string(), // ㅎ (hieut)
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Glottal),
            manner: Some(MannerOfArticulation::Fricative),
            voiced: false,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ㅎ".to_string(), "h".to_string()],
        },
        // Nasals
        PhonemeInfo {
            symbol: "n".to_string(), // ㄴ (nieun)
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Alveolar),
            manner: Some(MannerOfArticulation::Nasal),
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ㄴ".to_string(), "n".to_string()],
        },
        PhonemeInfo {
            symbol: "m".to_string(), // ㅁ (mieum)
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Bilabial),
            manner: Some(MannerOfArticulation::Nasal),
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ㅁ".to_string(), "m".to_string()],
        },
        PhonemeInfo {
            symbol: "ŋ".to_string(), // ㅇ (ieung) in final position
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Velar),
            manner: Some(MannerOfArticulation::Nasal),
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ㅇ".to_string(), "ng".to_string()],
        },
        // Liquid
        PhonemeInfo {
            symbol: "l".to_string(), // ㄹ (rieul) - lateral in final position
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Alveolar),
            manner: Some(MannerOfArticulation::LateralApproximant),
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ㄹ".to_string(), "r".to_string(), "l".to_string()],
        },
    ]
}

/// Korean vowels (모음 moeum)
///
/// Includes all 21 vowels: 7 monophthongs and 13 diphthongs.
fn get_korean_vowels() -> Vec<PhonemeInfo> {
    vec![
        // Simple vowels (단모음 danmoeum)
        PhonemeInfo {
            symbol: "a".to_string(), // ㅏ (a)
            phoneme_type: PhonemeType::Vowel,
            place: None,
            manner: None,
            voiced: true,
            height: Some(VowelHeight::Open),
            frontness: Some(VowelFrontness::Central),
            rounded: Some(false),
            example_graphemes: vec!["ㅏ".to_string(), "a".to_string()],
        },
        PhonemeInfo {
            symbol: "ʌ".to_string(), // ㅓ (eo)
            phoneme_type: PhonemeType::Vowel,
            place: None,
            manner: None,
            voiced: true,
            height: Some(VowelHeight::OpenMid),
            frontness: Some(VowelFrontness::Back),
            rounded: Some(false),
            example_graphemes: vec!["ㅓ".to_string(), "eo".to_string()],
        },
        PhonemeInfo {
            symbol: "o".to_string(), // ㅗ (o)
            phoneme_type: PhonemeType::Vowel,
            place: None,
            manner: None,
            voiced: true,
            height: Some(VowelHeight::CloseMid),
            frontness: Some(VowelFrontness::Back),
            rounded: Some(true),
            example_graphemes: vec!["ㅗ".to_string(), "o".to_string()],
        },
        PhonemeInfo {
            symbol: "u".to_string(), // ㅜ (u)
            phoneme_type: PhonemeType::Vowel,
            place: None,
            manner: None,
            voiced: true,
            height: Some(VowelHeight::Close),
            frontness: Some(VowelFrontness::Back),
            rounded: Some(true),
            example_graphemes: vec!["ㅜ".to_string(), "u".to_string()],
        },
        PhonemeInfo {
            symbol: "ɯ".to_string(), // ㅡ (eu)
            phoneme_type: PhonemeType::Vowel,
            place: None,
            manner: None,
            voiced: true,
            height: Some(VowelHeight::Close),
            frontness: Some(VowelFrontness::Back),
            rounded: Some(false),
            example_graphemes: vec!["ㅡ".to_string(), "eu".to_string()],
        },
        PhonemeInfo {
            symbol: "i".to_string(), // ㅣ (i)
            phoneme_type: PhonemeType::Vowel,
            place: None,
            manner: None,
            voiced: true,
            height: Some(VowelHeight::Close),
            frontness: Some(VowelFrontness::Front),
            rounded: Some(false),
            example_graphemes: vec!["ㅣ".to_string(), "i".to_string()],
        },
        PhonemeInfo {
            symbol: "e".to_string(), // ㅔ (e)
            phoneme_type: PhonemeType::Vowel,
            place: None,
            manner: None,
            voiced: true,
            height: Some(VowelHeight::CloseMid),
            frontness: Some(VowelFrontness::Front),
            rounded: Some(false),
            example_graphemes: vec!["ㅔ".to_string(), "e".to_string()],
        },
        PhonemeInfo {
            symbol: "ɛ".to_string(), // ㅐ (ae)
            phoneme_type: PhonemeType::Vowel,
            place: None,
            manner: None,
            voiced: true,
            height: Some(VowelHeight::OpenMid),
            frontness: Some(VowelFrontness::Front),
            rounded: Some(false),
            example_graphemes: vec!["ㅐ".to_string(), "ae".to_string()],
        },
        // Y-diphthongs (이중모음 ijungmoeum)
        PhonemeInfo {
            symbol: "ja".to_string(), // ㅑ (ya)
            phoneme_type: PhonemeType::Diphthong,
            place: None,
            manner: None,
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ㅑ".to_string(), "ya".to_string()],
        },
        PhonemeInfo {
            symbol: "jʌ".to_string(), // ㅕ (yeo)
            phoneme_type: PhonemeType::Diphthong,
            place: None,
            manner: None,
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ㅕ".to_string(), "yeo".to_string()],
        },
        PhonemeInfo {
            symbol: "jo".to_string(), // ㅛ (yo)
            phoneme_type: PhonemeType::Diphthong,
            place: None,
            manner: None,
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ㅛ".to_string(), "yo".to_string()],
        },
        PhonemeInfo {
            symbol: "ju".to_string(), // ㅠ (yu)
            phoneme_type: PhonemeType::Diphthong,
            place: None,
            manner: None,
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ㅠ".to_string(), "yu".to_string()],
        },
        PhonemeInfo {
            symbol: "jɛ".to_string(), // ㅒ (yae)
            phoneme_type: PhonemeType::Diphthong,
            place: None,
            manner: None,
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ㅒ".to_string(), "yae".to_string()],
        },
        PhonemeInfo {
            symbol: "je".to_string(), // ㅖ (ye)
            phoneme_type: PhonemeType::Diphthong,
            place: None,
            manner: None,
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ㅖ".to_string(), "ye".to_string()],
        },
        // W-diphthongs
        PhonemeInfo {
            symbol: "wa".to_string(), // ㅘ (wa)
            phoneme_type: PhonemeType::Diphthong,
            place: None,
            manner: None,
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ㅘ".to_string(), "wa".to_string()],
        },
        PhonemeInfo {
            symbol: "wɛ".to_string(), // ㅙ (wae)
            phoneme_type: PhonemeType::Diphthong,
            place: None,
            manner: None,
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ㅙ".to_string(), "wae".to_string()],
        },
        PhonemeInfo {
            symbol: "ø".to_string(), // ㅚ (oe) - actually [we] in modern Seoul dialect
            phoneme_type: PhonemeType::Diphthong,
            place: None,
            manner: None,
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ㅚ".to_string(), "oe".to_string(), "we".to_string()],
        },
        PhonemeInfo {
            symbol: "wʌ".to_string(), // ㅝ (wo)
            phoneme_type: PhonemeType::Diphthong,
            place: None,
            manner: None,
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ㅝ".to_string(), "wo".to_string()],
        },
        PhonemeInfo {
            symbol: "we".to_string(), // ㅞ (we)
            phoneme_type: PhonemeType::Diphthong,
            place: None,
            manner: None,
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ㅞ".to_string(), "we".to_string()],
        },
        PhonemeInfo {
            symbol: "wi".to_string(), // ㅟ (wi)
            phoneme_type: PhonemeType::Diphthong,
            place: None,
            manner: None,
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ㅟ".to_string(), "wi".to_string()],
        },
        PhonemeInfo {
            symbol: "ɰi".to_string(), // ㅢ (ui)
            phoneme_type: PhonemeType::Diphthong,
            place: None,
            manner: None,
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ㅢ".to_string(), "ui".to_string(), "eui".to_string()],
        },
    ]
}

/// Grapheme to phoneme mappings for Korean romanization
fn get_korean_grapheme_mappings() -> HashMap<String, Vec<String>> {
    let mut mappings = HashMap::new();

    // Initial consonants (Revised Romanization)
    mappings.insert("ㄱ".to_string(), vec!["k".to_string(), "g".to_string()]);
    mappings.insert("ㄴ".to_string(), vec!["n".to_string()]);
    mappings.insert("ㄷ".to_string(), vec!["t".to_string(), "d".to_string()]);
    mappings.insert("ㄹ".to_string(), vec!["l".to_string(), "r".to_string()]);
    mappings.insert("ㅁ".to_string(), vec!["m".to_string()]);
    mappings.insert("ㅂ".to_string(), vec!["p".to_string(), "b".to_string()]);
    mappings.insert("ㅅ".to_string(), vec!["s".to_string()]);
    mappings.insert("ㅇ".to_string(), vec!["ŋ".to_string()]); // Final position
    mappings.insert("ㅈ".to_string(), vec!["tɕ".to_string(), "j".to_string()]);
    mappings.insert("ㅊ".to_string(), vec!["tɕʰ".to_string(), "ch".to_string()]);
    mappings.insert("ㅋ".to_string(), vec!["kʰ".to_string(), "k".to_string()]);
    mappings.insert("ㅌ".to_string(), vec!["tʰ".to_string(), "t".to_string()]);
    mappings.insert("ㅍ".to_string(), vec!["pʰ".to_string(), "p".to_string()]);
    mappings.insert("ㅎ".to_string(), vec!["h".to_string()]);
    mappings.insert("ㄲ".to_string(), vec!["k͈".to_string(), "kk".to_string()]);
    mappings.insert("ㄸ".to_string(), vec!["t͈".to_string(), "tt".to_string()]);
    mappings.insert("ㅃ".to_string(), vec!["p͈".to_string(), "pp".to_string()]);
    mappings.insert("ㅆ".to_string(), vec!["s͈".to_string(), "ss".to_string()]);
    mappings.insert("ㅉ".to_string(), vec!["t͈ɕ".to_string(), "jj".to_string()]);

    // Vowels
    mappings.insert("ㅏ".to_string(), vec!["a".to_string()]);
    mappings.insert("ㅐ".to_string(), vec!["ɛ".to_string(), "ae".to_string()]);
    mappings.insert("ㅑ".to_string(), vec!["ja".to_string(), "ya".to_string()]);
    mappings.insert("ㅒ".to_string(), vec!["jɛ".to_string(), "yae".to_string()]);
    mappings.insert("ㅓ".to_string(), vec!["ʌ".to_string(), "eo".to_string()]);
    mappings.insert("ㅔ".to_string(), vec!["e".to_string()]);
    mappings.insert("ㅕ".to_string(), vec!["jʌ".to_string(), "yeo".to_string()]);
    mappings.insert("ㅖ".to_string(), vec!["je".to_string(), "ye".to_string()]);
    mappings.insert("ㅗ".to_string(), vec!["o".to_string()]);
    mappings.insert("ㅘ".to_string(), vec!["wa".to_string()]);
    mappings.insert("ㅙ".to_string(), vec!["wɛ".to_string(), "wae".to_string()]);
    mappings.insert(
        "ㅚ".to_string(),
        vec!["ø".to_string(), "oe".to_string(), "we".to_string()],
    );
    mappings.insert("ㅛ".to_string(), vec!["jo".to_string(), "yo".to_string()]);
    mappings.insert("ㅜ".to_string(), vec!["u".to_string()]);
    mappings.insert("ㅝ".to_string(), vec!["wʌ".to_string(), "wo".to_string()]);
    mappings.insert("ㅞ".to_string(), vec!["we".to_string()]);
    mappings.insert("ㅟ".to_string(), vec!["wi".to_string()]);
    mappings.insert("ㅠ".to_string(), vec!["ju".to_string(), "yu".to_string()]);
    mappings.insert("ㅡ".to_string(), vec!["ɯ".to_string(), "eu".to_string()]);
    mappings.insert(
        "ㅢ".to_string(),
        vec!["ɰi".to_string(), "ui".to_string(), "eui".to_string()],
    );
    mappings.insert("ㅣ".to_string(), vec!["i".to_string()]);

    mappings
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_korean_inventory_structure() {
        let inventory = get_korean_inventory();
        assert_eq!(inventory.language, LanguageCode::Ko);

        assert!(!inventory.consonants.is_empty(), "Should have consonants");
        assert!(!inventory.vowels.is_empty(), "Should have vowels");

        // Check for Korean-specific features
        assert!(
            inventory.is_valid_phoneme("k͈"),
            "Should have tense velar stop"
        );
        assert!(
            inventory.is_valid_phoneme("kʰ"),
            "Should have aspirated velar stop"
        );
        assert!(
            inventory.is_valid_phoneme("ɯ"),
            "Should have close back unrounded vowel"
        );
    }

    #[test]
    fn test_korean_consonant_count() {
        let consonants = get_korean_consonants();
        // Korean has 19 initial consonants
        assert_eq!(consonants.len(), 19, "Korean should have 19 consonants");
    }

    #[test]
    fn test_korean_vowel_count() {
        let vowels = get_korean_vowels();
        // Korean has 21 vowels (8 monophthongs + 13 diphthongs)
        assert_eq!(vowels.len(), 21, "Korean should have 21 vowels");
    }

    #[test]
    fn test_three_way_contrast() {
        let consonants = get_korean_consonants();
        let symbols: Vec<String> = consonants.iter().map(|c| c.symbol.clone()).collect();

        // Check velar stops (plain, aspirated, tense)
        assert!(
            symbols.contains(&"k".to_string()),
            "Should have plain velar 'k'"
        );
        assert!(
            symbols.contains(&"kʰ".to_string()),
            "Should have aspirated velar 'kʰ'"
        );
        assert!(
            symbols.contains(&"k͈".to_string()),
            "Should have tense velar 'k͈'"
        );

        // Check alveolar stops
        assert!(symbols.contains(&"t".to_string()), "Should have plain 't'");
        assert!(
            symbols.contains(&"tʰ".to_string()),
            "Should have aspirated 'tʰ'"
        );
        assert!(symbols.contains(&"t͈".to_string()), "Should have tense 't͈'");
    }

    #[test]
    fn test_korean_nasals() {
        let consonants = get_korean_consonants();
        let symbols: Vec<String> = consonants.iter().map(|c| c.symbol.clone()).collect();

        // Check all three nasals
        assert!(symbols.contains(&"n".to_string()), "Should have 'n'");
        assert!(symbols.contains(&"m".to_string()), "Should have 'm'");
        assert!(symbols.contains(&"ŋ".to_string()), "Should have 'ŋ'");
    }

    #[test]
    fn test_korean_monophthongs() {
        let vowels = get_korean_vowels();
        let monophthongs: Vec<&PhonemeInfo> = vowels
            .iter()
            .filter(|v| v.phoneme_type == PhonemeType::Vowel)
            .collect();

        assert!(
            monophthongs.len() >= 7,
            "Should have at least 7 monophthongs"
        );

        let symbols: Vec<String> = monophthongs.iter().map(|v| v.symbol.clone()).collect();
        assert!(symbols.contains(&"a".to_string()), "Should have 'a'");
        assert!(symbols.contains(&"ʌ".to_string()), "Should have 'ʌ'");
        assert!(symbols.contains(&"o".to_string()), "Should have 'o'");
        assert!(symbols.contains(&"u".to_string()), "Should have 'u'");
        assert!(symbols.contains(&"ɯ".to_string()), "Should have 'ɯ'");
        assert!(symbols.contains(&"i".to_string()), "Should have 'i'");
    }

    #[test]
    fn test_korean_diphthongs() {
        let vowels = get_korean_vowels();
        let diphthongs: Vec<&PhonemeInfo> = vowels
            .iter()
            .filter(|v| v.phoneme_type == PhonemeType::Diphthong)
            .collect();

        assert!(diphthongs.len() >= 10, "Should have at least 10 diphthongs");

        let symbols: Vec<String> = diphthongs.iter().map(|v| v.symbol.clone()).collect();
        assert!(symbols.contains(&"ja".to_string()), "Should have 'ya'");
        assert!(symbols.contains(&"jo".to_string()), "Should have 'yo'");
        assert!(symbols.contains(&"ju".to_string()), "Should have 'yu'");
        assert!(symbols.contains(&"wa".to_string()), "Should have 'wa'");
        assert!(symbols.contains(&"wi".to_string()), "Should have 'wi'");
    }

    #[test]
    fn test_grapheme_mappings() {
        let mappings = get_korean_grapheme_mappings();

        // Test hangul consonants
        assert!(mappings.contains_key("ㄱ"), "Should have mapping for ㄱ");
        assert!(mappings.contains_key("ㄲ"), "Should have mapping for ㄲ");
        assert!(mappings.contains_key("ㅋ"), "Should have mapping for ㅋ");

        // Test hangul vowels
        assert!(mappings.contains_key("ㅏ"), "Should have mapping for ㅏ");
        assert!(mappings.contains_key("ㅑ"), "Should have mapping for ㅑ");
        assert!(mappings.contains_key("ㅢ"), "Should have mapping for ㅢ");
    }

    #[test]
    fn test_all_phonemes_have_symbols() {
        let inventory = get_korean_inventory();
        let all_phonemes = inventory.all_phonemes();

        for phoneme in all_phonemes {
            assert!(
                !phoneme.symbol.is_empty(),
                "All phonemes should have symbols"
            );
        }
    }

    #[test]
    fn test_consonant_features() {
        let consonants = get_korean_consonants();

        // Find plain velar stop 'k'
        let k = consonants
            .iter()
            .find(|c| c.symbol == "k")
            .expect("Should have 'k' consonant");

        assert_eq!(k.phoneme_type, PhonemeType::Consonant);
        assert_eq!(k.place, Some(PlaceOfArticulation::Velar));
        assert_eq!(k.manner, Some(MannerOfArticulation::Plosive));

        // Find aspirated bilabial 'pʰ'
        let ph = consonants
            .iter()
            .find(|c| c.symbol == "pʰ")
            .expect("Should have 'pʰ' consonant");

        assert_eq!(ph.place, Some(PlaceOfArticulation::Bilabial));
        assert_eq!(ph.manner, Some(MannerOfArticulation::Plosive));
    }

    #[test]
    fn test_vowel_features() {
        let vowels = get_korean_vowels();

        // Find 'i' vowel
        let i = vowels
            .iter()
            .find(|v| v.symbol == "i")
            .expect("Should have 'i' vowel");

        assert_eq!(i.phoneme_type, PhonemeType::Vowel);
        assert_eq!(i.height, Some(VowelHeight::Close));
        assert_eq!(i.frontness, Some(VowelFrontness::Front));
        assert_eq!(i.rounded, Some(false));

        // Find 'ɯ' vowel (close back unrounded)
        let eu = vowels
            .iter()
            .find(|v| v.symbol == "ɯ")
            .expect("Should have 'ɯ' vowel");

        assert_eq!(eu.height, Some(VowelHeight::Close));
        assert_eq!(eu.frontness, Some(VowelFrontness::Back));
        assert_eq!(eu.rounded, Some(false));
    }
}

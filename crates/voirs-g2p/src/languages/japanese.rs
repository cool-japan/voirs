//! Japanese phoneme inventory and processing.
//!
//! This module provides comprehensive support for Japanese phonology, including
//! the complete set of consonants, vowels, and special phonemes used in Standard
//! Tokyo Japanese. Japanese has a relatively simple phoneme inventory but complex
//! prosodic features including pitch accent and mora timing.
//!
//! # Phonological Structure
//!
//! Japanese syllables follow a strict (C)V(N) structure:
//! - Consonant (optional): One of 14 consonants
//! - Vowel (required): One of 5 vowels
//! - Nasal coda (optional): Moraic /N/
//!
//! # Mora Structure
//!
//! Japanese is mora-timed rather than syllable-timed:
//! - Each mora takes roughly equal time
//! - Long vowels = 2 morae (e.g., おう = o + u)
//! - Geminate consonants = 2 morae (e.g., がっこう = ga + Q + ko + u)
//! - Moraic nasal /N/ = 1 mora (e.g., さん = sa + N)
//!
//! # Pitch Accent
//!
//! Tokyo Japanese uses pitch accent (高低アクセント kōtei akusento):
//! - High (H) vs Low (L) pitch on morae
//! - Each word has fixed pitch pattern
//! - Downstep location is lexically determined
//!
//! # Special Features
//!
//! - **Palatalization**: /t/, /d/, /s/, /z/ palatalize before /i/ and /y/
//! - **Gemination**: Consonant doubling for emphasis (促音 sokuon)
//! - **Vowel Length**: Phonemic distinction between short and long vowels
//! - **Moraic Nasal**: Syllable-final /N/ (撥音 hatsuon)
//!
//! # Example
//!
//! ```rust
//! use voirs_g2p::languages::japanese::get_japanese_inventory;
//! use voirs_g2p::LanguageCode;
//!
//! let inventory = get_japanese_inventory();
//! assert_eq!(inventory.language, LanguageCode::Ja);
//!
//! // Check for specific phonemes
//! assert!(inventory.is_valid_phoneme("tɕ"));  // Palatalized 'chi'
//! assert!(inventory.is_valid_phoneme("ɴ"));   // Moraic nasal 'n'
//! assert!(inventory.is_valid_phoneme("o"));   // Vowel 'o'
//! ```

use super::{
    MannerOfArticulation, PhonemeInfo, PhonemeInventory, PhonemeType, PlaceOfArticulation,
    VowelFrontness, VowelHeight,
};
use crate::LanguageCode;
use std::collections::HashMap;

/// Mora type in Japanese phonology
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoraType {
    /// Regular CV mora (e.g., か ka)
    Regular,
    /// Long vowel mora (second mora of long vowel)
    LongVowel,
    /// Geminate consonant mora (促音 sokuon) - represented as Q
    Geminate,
    /// Moraic nasal (撥音 hatsuon) - syllable-final /N/
    MoraicNasal,
}

/// Pitch accent pattern for Japanese words
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PitchPattern {
    /// Flat (平板 heiban) - no downstep
    Flat,
    /// Head-high (頭高 atamadaka) - downstep after first mora
    HeadHigh,
    /// Mid-high (中高 nakadaka) - downstep in middle
    MidHigh,
    /// Tail-high (尾高 odaka) - downstep after final mora
    TailHigh,
}

/// Get the complete Japanese phoneme inventory
pub fn get_japanese_inventory() -> PhonemeInventory {
    let consonants = get_japanese_consonants();
    let vowels = get_japanese_vowels();
    let other_sounds = get_japanese_other_sounds();
    let grapheme_mappings = get_japanese_grapheme_mappings();

    PhonemeInventory {
        language: LanguageCode::Ja,
        consonants,
        vowels,
        other_sounds,
        grapheme_mappings,
    }
}

/// Japanese consonants (子音 shiin)
///
/// Standard Tokyo Japanese has 14 consonant phonemes. Note that palatalized
/// variants are considered allophones in some analyses but distinct phonemes in others.
fn get_japanese_consonants() -> Vec<PhonemeInfo> {
    vec![
        // Plosives (破裂音)
        PhonemeInfo {
            symbol: "k".to_string(), // か ka, き ki, く ku, け ke, こ ko
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Velar),
            manner: Some(MannerOfArticulation::Plosive),
            voiced: false,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["k".to_string(), "か".to_string(), "カ".to_string()],
        },
        PhonemeInfo {
            symbol: "ɡ".to_string(), // が ga, ぎ gi, ぐ gu, げ ge, ご go
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Velar),
            manner: Some(MannerOfArticulation::Plosive),
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["g".to_string(), "が".to_string(), "ガ".to_string()],
        },
        PhonemeInfo {
            symbol: "t".to_string(), // た ta, て te, と to (NOT before i)
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Alveolar),
            manner: Some(MannerOfArticulation::Plosive),
            voiced: false,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["t".to_string(), "た".to_string(), "タ".to_string()],
        },
        PhonemeInfo {
            symbol: "d".to_string(), // だ da, で de, ど do (NOT before i)
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Alveolar),
            manner: Some(MannerOfArticulation::Plosive),
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["d".to_string(), "だ".to_string(), "ダ".to_string()],
        },
        PhonemeInfo {
            symbol: "p".to_string(), // ぱ pa, ぴ pi, ぷ pu, ぺ pe, ぽ po
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Bilabial),
            manner: Some(MannerOfArticulation::Plosive),
            voiced: false,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["p".to_string(), "ぱ".to_string(), "パ".to_string()],
        },
        PhonemeInfo {
            symbol: "b".to_string(), // ば ba, び bi, ぶ bu, べ be, ぼ bo
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Bilabial),
            manner: Some(MannerOfArticulation::Plosive),
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["b".to_string(), "ば".to_string(), "バ".to_string()],
        },
        // Affricates (破擦音) - palatalized variants
        PhonemeInfo {
            symbol: "tɕ".to_string(), // ち chi (palatalized t before i)
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Palatal),
            manner: Some(MannerOfArticulation::Affricate),
            voiced: false,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ch".to_string(), "ち".to_string(), "チ".to_string()],
        },
        PhonemeInfo {
            symbol: "dʑ".to_string(), // じ ji, ぢ di (palatalized d before i)
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Palatal),
            manner: Some(MannerOfArticulation::Affricate),
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["j".to_string(), "じ".to_string(), "ジ".to_string()],
        },
        PhonemeInfo {
            symbol: "ts".to_string(), // つ tsu
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Alveolar),
            manner: Some(MannerOfArticulation::Affricate),
            voiced: false,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ts".to_string(), "つ".to_string(), "ツ".to_string()],
        },
        // Fricatives (摩擦音)
        PhonemeInfo {
            symbol: "s".to_string(), // さ sa, せ se, そ so (NOT before i)
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Alveolar),
            manner: Some(MannerOfArticulation::Fricative),
            voiced: false,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["s".to_string(), "さ".to_string(), "サ".to_string()],
        },
        PhonemeInfo {
            symbol: "z".to_string(), // ざ za, ぜ ze, ぞ zo (NOT before i)
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Alveolar),
            manner: Some(MannerOfArticulation::Fricative),
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["z".to_string(), "ざ".to_string(), "ザ".to_string()],
        },
        PhonemeInfo {
            symbol: "ɕ".to_string(), // し shi (palatalized s before i)
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Palatal),
            manner: Some(MannerOfArticulation::Fricative),
            voiced: false,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["sh".to_string(), "し".to_string(), "シ".to_string()],
        },
        PhonemeInfo {
            symbol: "h".to_string(), // は ha, へ he, ほ ho
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Glottal),
            manner: Some(MannerOfArticulation::Fricative),
            voiced: false,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["h".to_string(), "は".to_string(), "ハ".to_string()],
        },
        PhonemeInfo {
            symbol: "ɸ".to_string(), // ふ fu (bilabial fricative, not 'hu')
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Bilabial),
            manner: Some(MannerOfArticulation::Fricative),
            voiced: false,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["f".to_string(), "ふ".to_string(), "フ".to_string()],
        },
        // Nasals (鼻音)
        PhonemeInfo {
            symbol: "m".to_string(), // ま ma, み mi, む mu, め me, も mo
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Bilabial),
            manner: Some(MannerOfArticulation::Nasal),
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["m".to_string(), "ま".to_string(), "マ".to_string()],
        },
        PhonemeInfo {
            symbol: "n".to_string(), // な na, に ni, ぬ nu, ね ne, の no
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Alveolar),
            manner: Some(MannerOfArticulation::Nasal),
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["n".to_string(), "な".to_string(), "ナ".to_string()],
        },
        // Approximants (接近音)
        PhonemeInfo {
            symbol: "ɾ".to_string(), // ら ra, り ri, る ru, れ re, ろ ro (tap/flap)
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Alveolar),
            manner: Some(MannerOfArticulation::TapFlap),
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["r".to_string(), "ら".to_string(), "ラ".to_string()],
        },
        PhonemeInfo {
            symbol: "w".to_string(), // わ wa, を wo
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Velar),
            manner: Some(MannerOfArticulation::Approximant),
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["w".to_string(), "わ".to_string(), "ワ".to_string()],
        },
        PhonemeInfo {
            symbol: "j".to_string(), // や ya, ゆ yu, よ yo (palatal approximant)
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Palatal),
            manner: Some(MannerOfArticulation::Approximant),
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["y".to_string(), "や".to_string(), "ヤ".to_string()],
        },
    ]
}

/// Japanese vowels (母音 boin)
///
/// Japanese has 5 pure vowels with phonemic length distinction.
fn get_japanese_vowels() -> Vec<PhonemeInfo> {
    vec![
        PhonemeInfo {
            symbol: "a".to_string(), // あ a, か ka, etc.
            phoneme_type: PhonemeType::Vowel,
            place: None,
            manner: None,
            voiced: true,
            height: Some(VowelHeight::Open),
            frontness: Some(VowelFrontness::Central),
            rounded: Some(false),
            example_graphemes: vec!["a".to_string(), "あ".to_string(), "ア".to_string()],
        },
        PhonemeInfo {
            symbol: "i".to_string(), // い i, き ki, etc.
            phoneme_type: PhonemeType::Vowel,
            place: None,
            manner: None,
            voiced: true,
            height: Some(VowelHeight::Close),
            frontness: Some(VowelFrontness::Front),
            rounded: Some(false),
            example_graphemes: vec!["i".to_string(), "い".to_string(), "イ".to_string()],
        },
        PhonemeInfo {
            symbol: "ɯ".to_string(), // う u (close back unrounded, not 'oo')
            phoneme_type: PhonemeType::Vowel,
            place: None,
            manner: None,
            voiced: true,
            height: Some(VowelHeight::Close),
            frontness: Some(VowelFrontness::Back),
            rounded: Some(false),
            example_graphemes: vec!["u".to_string(), "う".to_string(), "ウ".to_string()],
        },
        PhonemeInfo {
            symbol: "e".to_string(), // え e, け ke, etc.
            phoneme_type: PhonemeType::Vowel,
            place: None,
            manner: None,
            voiced: true,
            height: Some(VowelHeight::CloseMid),
            frontness: Some(VowelFrontness::Front),
            rounded: Some(false),
            example_graphemes: vec!["e".to_string(), "え".to_string(), "エ".to_string()],
        },
        PhonemeInfo {
            symbol: "o".to_string(), // お o, こ ko, etc.
            phoneme_type: PhonemeType::Vowel,
            place: None,
            manner: None,
            voiced: true,
            height: Some(VowelHeight::CloseMid),
            frontness: Some(VowelFrontness::Back),
            rounded: Some(true),
            example_graphemes: vec!["o".to_string(), "お".to_string(), "オ".to_string()],
        },
    ]
}

/// Special sounds in Japanese
fn get_japanese_other_sounds() -> Vec<PhonemeInfo> {
    vec![
        // Moraic nasal (撥音 hatsuon) - syllable-final /N/
        PhonemeInfo {
            symbol: "ɴ".to_string(), // ん n (moraic nasal)
            phoneme_type: PhonemeType::Other,
            place: Some(PlaceOfArticulation::Uvular),
            manner: Some(MannerOfArticulation::Nasal),
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["n".to_string(), "ん".to_string(), "ン".to_string()],
        },
        // Geminate consonant placeholder (促音 sokuon)
        PhonemeInfo {
            symbol: "Q".to_string(), // っ (small tsu) - geminate marker
            phoneme_type: PhonemeType::Other,
            place: None,
            manner: None,
            voiced: false,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["Q".to_string(), "っ".to_string(), "ッ".to_string()],
        },
    ]
}

/// Grapheme to phoneme mappings for Japanese romanization
fn get_japanese_grapheme_mappings() -> HashMap<String, Vec<String>> {
    let mut mappings = HashMap::new();

    // Consonants
    mappings.insert("k".to_string(), vec!["k".to_string()]);
    mappings.insert("g".to_string(), vec!["ɡ".to_string()]);
    mappings.insert("t".to_string(), vec!["t".to_string()]);
    mappings.insert("d".to_string(), vec!["d".to_string()]);
    mappings.insert("p".to_string(), vec!["p".to_string()]);
    mappings.insert("b".to_string(), vec!["b".to_string()]);
    mappings.insert("ch".to_string(), vec!["tɕ".to_string()]);
    mappings.insert("j".to_string(), vec!["dʑ".to_string()]);
    mappings.insert("ts".to_string(), vec!["ts".to_string()]);
    mappings.insert("s".to_string(), vec!["s".to_string()]);
    mappings.insert("z".to_string(), vec!["z".to_string()]);
    mappings.insert("sh".to_string(), vec!["ɕ".to_string()]);
    mappings.insert("h".to_string(), vec!["h".to_string()]);
    mappings.insert("f".to_string(), vec!["ɸ".to_string()]);
    mappings.insert("m".to_string(), vec!["m".to_string()]);
    mappings.insert("n".to_string(), vec!["n".to_string()]);
    mappings.insert("r".to_string(), vec!["ɾ".to_string()]);
    mappings.insert("w".to_string(), vec!["w".to_string()]);
    mappings.insert("y".to_string(), vec!["j".to_string()]);

    // Vowels
    mappings.insert("a".to_string(), vec!["a".to_string()]);
    mappings.insert("i".to_string(), vec!["i".to_string()]);
    mappings.insert("u".to_string(), vec!["ɯ".to_string()]);
    mappings.insert("e".to_string(), vec!["e".to_string()]);
    mappings.insert("o".to_string(), vec!["o".to_string()]);

    // Hiragana mappings (basic)
    mappings.insert("あ".to_string(), vec!["a".to_string()]);
    mappings.insert("い".to_string(), vec!["i".to_string()]);
    mappings.insert("う".to_string(), vec!["ɯ".to_string()]);
    mappings.insert("え".to_string(), vec!["e".to_string()]);
    mappings.insert("お".to_string(), vec!["o".to_string()]);
    mappings.insert("ん".to_string(), vec!["ɴ".to_string()]);
    mappings.insert("っ".to_string(), vec!["Q".to_string()]);

    // Katakana mappings (basic)
    mappings.insert("ア".to_string(), vec!["a".to_string()]);
    mappings.insert("イ".to_string(), vec!["i".to_string()]);
    mappings.insert("ウ".to_string(), vec!["ɯ".to_string()]);
    mappings.insert("エ".to_string(), vec!["e".to_string()]);
    mappings.insert("オ".to_string(), vec!["o".to_string()]);
    mappings.insert("ン".to_string(), vec!["ɴ".to_string()]);
    mappings.insert("ッ".to_string(), vec!["Q".to_string()]);

    mappings
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_japanese_inventory_structure() {
        let inventory = get_japanese_inventory();
        assert_eq!(inventory.language, LanguageCode::Ja);

        assert!(!inventory.consonants.is_empty(), "Should have consonants");
        assert!(!inventory.vowels.is_empty(), "Should have vowels");
        assert!(
            !inventory.other_sounds.is_empty(),
            "Should have special sounds"
        );

        // Check for Japanese-specific features
        assert!(
            inventory.is_valid_phoneme("tɕ"),
            "Should have palatalized affricate 'chi'"
        );
        assert!(inventory.is_valid_phoneme("ɴ"), "Should have moraic nasal");
        assert!(
            inventory.is_valid_phoneme("Q"),
            "Should have geminate marker"
        );
    }

    #[test]
    fn test_japanese_consonant_count() {
        let consonants = get_japanese_consonants();
        // Japanese has ~19 consonant phonemes (including palatalized variants)
        assert!(
            consonants.len() >= 14,
            "Should have at least 14 consonant phonemes"
        );
    }

    #[test]
    fn test_japanese_vowel_count() {
        let vowels = get_japanese_vowels();
        // Japanese has exactly 5 vowel phonemes
        assert_eq!(vowels.len(), 5, "Japanese should have exactly 5 vowels");
    }

    #[test]
    fn test_japanese_five_vowel_system() {
        let vowels = get_japanese_vowels();
        let symbols: Vec<String> = vowels.iter().map(|v| v.symbol.clone()).collect();

        // Check all 5 vowels present
        assert!(symbols.contains(&"a".to_string()), "Should have 'a'");
        assert!(symbols.contains(&"i".to_string()), "Should have 'i'");
        assert!(symbols.contains(&"ɯ".to_string()), "Should have 'ɯ' (u)");
        assert!(symbols.contains(&"e".to_string()), "Should have 'e'");
        assert!(symbols.contains(&"o".to_string()), "Should have 'o'");
    }

    #[test]
    fn test_palatalized_consonants() {
        let consonants = get_japanese_consonants();
        let symbols: Vec<String> = consonants.iter().map(|c| c.symbol.clone()).collect();

        // Check palatalized consonants
        assert!(
            symbols.contains(&"tɕ".to_string()),
            "Should have 'tɕ' (chi)"
        );
        assert!(symbols.contains(&"dʑ".to_string()), "Should have 'dʑ' (ji)");
        assert!(symbols.contains(&"ɕ".to_string()), "Should have 'ɕ' (shi)");
    }

    #[test]
    fn test_special_japanese_consonants() {
        let consonants = get_japanese_consonants();
        let symbols: Vec<String> = consonants.iter().map(|c| c.symbol.clone()).collect();

        // Bilabial fricative (ふ fu)
        assert!(
            symbols.contains(&"ɸ".to_string()),
            "Should have 'ɸ' (bilabial fricative)"
        );

        // Tap/flap (ら ra)
        assert!(symbols.contains(&"ɾ".to_string()), "Should have 'ɾ' (tap)");

        // Back unrounded u is in vowels
        let vowels = get_japanese_vowels();
        let vowel_symbols: Vec<String> = vowels.iter().map(|v| v.symbol.clone()).collect();
        assert!(
            vowel_symbols.contains(&"ɯ".to_string()),
            "Should have 'ɯ' (back unrounded)"
        );
    }

    #[test]
    fn test_moraic_nasal() {
        let other = get_japanese_other_sounds();
        let symbols: Vec<String> = other.iter().map(|s| s.symbol.clone()).collect();

        assert!(
            symbols.contains(&"ɴ".to_string()),
            "Should have moraic nasal 'ɴ'"
        );
    }

    #[test]
    fn test_geminate_marker() {
        let other = get_japanese_other_sounds();
        let symbols: Vec<String> = other.iter().map(|s| s.symbol.clone()).collect();

        assert!(
            symbols.contains(&"Q".to_string()),
            "Should have geminate marker 'Q'"
        );
    }

    #[test]
    fn test_grapheme_mappings() {
        let mappings = get_japanese_grapheme_mappings();

        // Test romanization
        assert_eq!(mappings.get("ch"), Some(&vec!["tɕ".to_string()]));
        assert_eq!(mappings.get("sh"), Some(&vec!["ɕ".to_string()]));
        assert_eq!(mappings.get("ts"), Some(&vec!["ts".to_string()]));

        // Test hiragana
        assert!(mappings.contains_key("あ"), "Should have hiragana 'あ'");
        assert!(mappings.contains_key("ん"), "Should have hiragana 'ん'");
        assert!(mappings.contains_key("っ"), "Should have hiragana 'っ'");

        // Test katakana
        assert!(mappings.contains_key("ア"), "Should have katakana 'ア'");
        assert!(mappings.contains_key("ン"), "Should have katakana 'ン'");
    }

    #[test]
    fn test_all_phonemes_have_symbols() {
        let inventory = get_japanese_inventory();
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
        let consonants = get_japanese_consonants();

        // Find bilabial plosive 'p'
        let p = consonants
            .iter()
            .find(|c| c.symbol == "p")
            .expect("Should have 'p' consonant");

        assert_eq!(p.phoneme_type, PhonemeType::Consonant);
        assert_eq!(p.place, Some(PlaceOfArticulation::Bilabial));
        assert_eq!(p.manner, Some(MannerOfArticulation::Plosive));

        // Find palatal affricate 'tɕ'
        let chi = consonants
            .iter()
            .find(|c| c.symbol == "tɕ")
            .expect("Should have 'tɕ' consonant");

        assert_eq!(chi.place, Some(PlaceOfArticulation::Palatal));
        assert_eq!(chi.manner, Some(MannerOfArticulation::Affricate));
    }

    #[test]
    fn test_vowel_features() {
        let vowels = get_japanese_vowels();

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
        let u = vowels
            .iter()
            .find(|v| v.symbol == "ɯ")
            .expect("Should have 'ɯ' vowel");

        assert_eq!(u.height, Some(VowelHeight::Close));
        assert_eq!(u.frontness, Some(VowelFrontness::Back));
        assert_eq!(u.rounded, Some(false)); // Unrounded!
    }

    #[test]
    fn test_phoneme_type_distribution() {
        let inventory = get_japanese_inventory();
        let all_phonemes = inventory.all_phonemes();

        let consonant_count = all_phonemes
            .iter()
            .filter(|p| {
                inventory
                    .consonants
                    .iter()
                    .any(|c| c.symbol == p.symbol.as_str())
            })
            .count();

        let vowel_count = all_phonemes
            .iter()
            .filter(|p| {
                inventory
                    .vowels
                    .iter()
                    .any(|v| v.symbol == p.symbol.as_str())
            })
            .count();

        let other_count = all_phonemes
            .iter()
            .filter(|p| {
                inventory
                    .other_sounds
                    .iter()
                    .any(|o| o.symbol == p.symbol.as_str())
            })
            .count();

        assert!(consonant_count >= 14, "Should have at least 14 consonants");
        assert_eq!(vowel_count, 5, "Should have exactly 5 vowels");
        assert_eq!(other_count, 2, "Should have 2 special sounds (ɴ, Q)");
    }
}

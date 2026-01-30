//! Chinese (Mandarin) phoneme inventory and processing.
//!
//! This module provides comprehensive support for Mandarin Chinese phonology,
//! including the complete set of initials (声母 shēngmǔ), finals (韵母 yùnmǔ),
//! and tones (声调 shēngdiào). Mandarin Chinese has a unique phonological system
//! with approximately 400 distinct syllables (before tones) and 5 tones.
//!
//! # Phonological Structure
//!
//! Mandarin syllables follow the structure: (Initial) + Final + Tone
//! - Initials: 21 consonants (声母)
//! - Finals: 35+ vowel combinations (韵母)
//! - Tones: 5 tones (4 lexical + 1 neutral)
//!
//! # Tone System
//!
//! - Tone 1 (阴平 yīnpíng): High level (55) - ā
//! - Tone 2 (阳平 yángpíng): Rising (35) - á
//! - Tone 3 (上声 shǎngshēng): Falling-rising (214) - ǎ
//! - Tone 4 (去声 qùshēng): Falling (51) - à
//! - Tone 5 (轻声 qīngshēng): Neutral (no mark) - a
//!
//! # Example
//!
//! ```rust
//! use voirs_g2p::languages::chinese::get_chinese_inventory;
//! use voirs_g2p::LanguageCode;
//!
//! let inventory = get_chinese_inventory();
//! assert_eq!(inventory.language, LanguageCode::ZhCn);
//!
//! // Check for specific phonemes
//! assert!(inventory.is_valid_phoneme("ʈʂ"));  // Retroflex affricate (zh)
//! assert!(inventory.is_valid_phoneme("ü"));   // High front rounded vowel
//! ```

use super::{
    MannerOfArticulation, PhonemeInfo, PhonemeInventory, PhonemeType, PlaceOfArticulation,
    VowelFrontness, VowelHeight,
};
use crate::LanguageCode;
use std::collections::HashMap;

/// Tone information for Chinese phonemes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MandarinTone {
    /// First tone (阴平): High level (55) - ā
    Tone1,
    /// Second tone (阳平): Rising (35) - á
    Tone2,
    /// Third tone (上声): Falling-rising (214) - ǎ
    Tone3,
    /// Fourth tone (去声): Falling (51) - à
    Tone4,
    /// Neutral tone (轻声): Unstressed - a
    Neutral,
}

impl MandarinTone {
    /// Get the tone number (1-5, where 5 is neutral)
    pub fn number(&self) -> u8 {
        match self {
            Self::Tone1 => 1,
            Self::Tone2 => 2,
            Self::Tone3 => 3,
            Self::Tone4 => 4,
            Self::Neutral => 5,
        }
    }

    /// Get the tone contour in Chao tone letters
    pub fn contour(&self) -> &'static str {
        match self {
            Self::Tone1 => "55",  // High level
            Self::Tone2 => "35",  // Rising
            Self::Tone3 => "214", // Falling-rising
            Self::Tone4 => "51",  // Falling
            Self::Neutral => "",  // Unstressed/neutral
        }
    }

    /// Get the tone diacritic example (using 'a')
    pub fn diacritic_example(&self) -> &'static str {
        match self {
            Self::Tone1 => "ā",
            Self::Tone2 => "á",
            Self::Tone3 => "ǎ",
            Self::Tone4 => "à",
            Self::Neutral => "a",
        }
    }
}

/// Get the complete Chinese (Mandarin) phoneme inventory
pub fn get_chinese_inventory() -> PhonemeInventory {
    let consonants = get_chinese_consonants();
    let vowels = get_chinese_vowels();
    let other_sounds = get_chinese_other_sounds();
    let grapheme_mappings = get_chinese_grapheme_mappings();

    PhonemeInventory {
        language: LanguageCode::ZhCn,
        consonants,
        vowels,
        other_sounds,
        grapheme_mappings,
    }
}

/// Chinese initial consonants (声母 shēngmǔ)
///
/// Mandarin has 21 initial consonants, organized by place and manner of articulation.
fn get_chinese_consonants() -> Vec<PhonemeInfo> {
    vec![
        // Bilabial consonants
        PhonemeInfo {
            symbol: "b".to_string(),
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Bilabial),
            manner: Some(MannerOfArticulation::Plosive),
            voiced: false, // Voiceless unaspirated in Mandarin
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["b".to_string(), "巴".to_string()],
        },
        PhonemeInfo {
            symbol: "p".to_string(),
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Bilabial),
            manner: Some(MannerOfArticulation::Plosive),
            voiced: false, // Voiceless aspirated
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["p".to_string(), "怕".to_string()],
        },
        PhonemeInfo {
            symbol: "m".to_string(),
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Bilabial),
            manner: Some(MannerOfArticulation::Nasal),
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["m".to_string(), "妈".to_string()],
        },
        PhonemeInfo {
            symbol: "f".to_string(),
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Labiodental),
            manner: Some(MannerOfArticulation::Fricative),
            voiced: false,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["f".to_string(), "发".to_string()],
        },
        // Alveolar consonants
        PhonemeInfo {
            symbol: "d".to_string(),
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Alveolar),
            manner: Some(MannerOfArticulation::Plosive),
            voiced: false, // Voiceless unaspirated
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["d".to_string(), "大".to_string()],
        },
        PhonemeInfo {
            symbol: "t".to_string(),
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Alveolar),
            manner: Some(MannerOfArticulation::Plosive),
            voiced: false, // Voiceless aspirated
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["t".to_string(), "他".to_string()],
        },
        PhonemeInfo {
            symbol: "n".to_string(),
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Alveolar),
            manner: Some(MannerOfArticulation::Nasal),
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["n".to_string(), "那".to_string()],
        },
        PhonemeInfo {
            symbol: "l".to_string(),
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Alveolar),
            manner: Some(MannerOfArticulation::LateralApproximant),
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["l".to_string(), "啦".to_string()],
        },
        // Velar consonants
        PhonemeInfo {
            symbol: "g".to_string(),
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Velar),
            manner: Some(MannerOfArticulation::Plosive),
            voiced: false, // Voiceless unaspirated
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["g".to_string(), "哥".to_string()],
        },
        PhonemeInfo {
            symbol: "k".to_string(),
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Velar),
            manner: Some(MannerOfArticulation::Plosive),
            voiced: false, // Voiceless aspirated
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["k".to_string(), "卡".to_string()],
        },
        PhonemeInfo {
            symbol: "h".to_string(),
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Glottal),
            manner: Some(MannerOfArticulation::Fricative),
            voiced: false,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["h".to_string(), "哈".to_string()],
        },
        // Palatal consonants (Pinyin j, q, x)
        PhonemeInfo {
            symbol: "ɕ".to_string(), // IPA for Pinyin 'x'
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Palatal),
            manner: Some(MannerOfArticulation::Fricative),
            voiced: false,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["x".to_string(), "西".to_string()],
        },
        PhonemeInfo {
            symbol: "tɕ".to_string(), // IPA for Pinyin 'j'
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Palatal),
            manner: Some(MannerOfArticulation::Affricate),
            voiced: false, // Voiceless unaspirated
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["j".to_string(), "家".to_string()],
        },
        PhonemeInfo {
            symbol: "tɕʰ".to_string(), // IPA for Pinyin 'q'
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Palatal),
            manner: Some(MannerOfArticulation::Affricate),
            voiced: false, // Voiceless aspirated
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["q".to_string(), "七".to_string()],
        },
        // Retroflex consonants (Pinyin zh, ch, sh, r)
        PhonemeInfo {
            symbol: "ʈʂ".to_string(), // IPA for Pinyin 'zh'
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Retroflex),
            manner: Some(MannerOfArticulation::Affricate),
            voiced: false, // Voiceless unaspirated
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["zh".to_string(), "知".to_string()],
        },
        PhonemeInfo {
            symbol: "ʈʂʰ".to_string(), // IPA for Pinyin 'ch'
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Retroflex),
            manner: Some(MannerOfArticulation::Affricate),
            voiced: false, // Voiceless aspirated
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ch".to_string(), "吃".to_string()],
        },
        PhonemeInfo {
            symbol: "ʂ".to_string(), // IPA for Pinyin 'sh'
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Retroflex),
            manner: Some(MannerOfArticulation::Fricative),
            voiced: false,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["sh".to_string(), "是".to_string()],
        },
        PhonemeInfo {
            symbol: "ʐ".to_string(), // IPA for Pinyin 'r'
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Retroflex),
            manner: Some(MannerOfArticulation::Approximant),
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["r".to_string(), "日".to_string()],
        },
        // Dental sibilants (Pinyin z, c, s)
        PhonemeInfo {
            symbol: "ts".to_string(), // IPA for Pinyin 'z'
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Alveolar),
            manner: Some(MannerOfArticulation::Affricate),
            voiced: false, // Voiceless unaspirated
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["z".to_string(), "字".to_string()],
        },
        PhonemeInfo {
            symbol: "tsʰ".to_string(), // IPA for Pinyin 'c'
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Alveolar),
            manner: Some(MannerOfArticulation::Affricate),
            voiced: false, // Voiceless aspirated
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["c".to_string(), "次".to_string()],
        },
        PhonemeInfo {
            symbol: "s".to_string(), // IPA for Pinyin 's'
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Alveolar),
            manner: Some(MannerOfArticulation::Fricative),
            voiced: false,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["s".to_string(), "四".to_string()],
        },
    ]
}

/// Chinese vowels and finals (韵母 yùnmǔ)
///
/// Mandarin has a complex final system with monophthongs, diphthongs, and triphthongs.
fn get_chinese_vowels() -> Vec<PhonemeInfo> {
    vec![
        // Simple vowels (单韵母)
        PhonemeInfo {
            symbol: "a".to_string(),
            phoneme_type: PhonemeType::Vowel,
            place: None,
            manner: None,
            voiced: true,
            height: Some(VowelHeight::Open),
            frontness: Some(VowelFrontness::Central),
            rounded: Some(false),
            example_graphemes: vec!["a".to_string(), "啊".to_string()],
        },
        PhonemeInfo {
            symbol: "o".to_string(),
            phoneme_type: PhonemeType::Vowel,
            place: None,
            manner: None,
            voiced: true,
            height: Some(VowelHeight::CloseMid),
            frontness: Some(VowelFrontness::Back),
            rounded: Some(true),
            example_graphemes: vec!["o".to_string(), "哦".to_string()],
        },
        PhonemeInfo {
            symbol: "e".to_string(),
            phoneme_type: PhonemeType::Vowel,
            place: None,
            manner: None,
            voiced: true,
            height: Some(VowelHeight::Mid),
            frontness: Some(VowelFrontness::Central),
            rounded: Some(false),
            example_graphemes: vec!["e".to_string(), "鹅".to_string()],
        },
        PhonemeInfo {
            symbol: "i".to_string(),
            phoneme_type: PhonemeType::Vowel,
            place: None,
            manner: None,
            voiced: true,
            height: Some(VowelHeight::Close),
            frontness: Some(VowelFrontness::Front),
            rounded: Some(false),
            example_graphemes: vec!["i".to_string(), "衣".to_string()],
        },
        PhonemeInfo {
            symbol: "u".to_string(),
            phoneme_type: PhonemeType::Vowel,
            place: None,
            manner: None,
            voiced: true,
            height: Some(VowelHeight::Close),
            frontness: Some(VowelFrontness::Back),
            rounded: Some(true),
            example_graphemes: vec!["u".to_string(), "乌".to_string()],
        },
        PhonemeInfo {
            symbol: "ü".to_string(), // ü or 'yu' in Pinyin
            phoneme_type: PhonemeType::Vowel,
            place: None,
            manner: None,
            voiced: true,
            height: Some(VowelHeight::Close),
            frontness: Some(VowelFrontness::Front),
            rounded: Some(true),
            example_graphemes: vec!["ü".to_string(), "v".to_string(), "鱼".to_string()],
        },
        // Compound finals (复韵母) - Common diphthongs
        PhonemeInfo {
            symbol: "ai".to_string(),
            phoneme_type: PhonemeType::Diphthong,
            place: None,
            manner: None,
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ai".to_string(), "爱".to_string()],
        },
        PhonemeInfo {
            symbol: "ei".to_string(),
            phoneme_type: PhonemeType::Diphthong,
            place: None,
            manner: None,
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ei".to_string(), "诶".to_string()],
        },
        PhonemeInfo {
            symbol: "ui".to_string(),
            phoneme_type: PhonemeType::Diphthong,
            place: None,
            manner: None,
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ui".to_string(), "wei".to_string(), "威".to_string()],
        },
        PhonemeInfo {
            symbol: "ao".to_string(),
            phoneme_type: PhonemeType::Diphthong,
            place: None,
            manner: None,
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ao".to_string(), "熬".to_string()],
        },
        PhonemeInfo {
            symbol: "ou".to_string(),
            phoneme_type: PhonemeType::Diphthong,
            place: None,
            manner: None,
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ou".to_string(), "欧".to_string()],
        },
        PhonemeInfo {
            symbol: "iu".to_string(),
            phoneme_type: PhonemeType::Diphthong,
            place: None,
            manner: None,
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["iu".to_string(), "you".to_string(), "优".to_string()],
        },
        PhonemeInfo {
            symbol: "ie".to_string(),
            phoneme_type: PhonemeType::Diphthong,
            place: None,
            manner: None,
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ie".to_string(), "耶".to_string()],
        },
        PhonemeInfo {
            symbol: "üe".to_string(),
            phoneme_type: PhonemeType::Diphthong,
            place: None,
            manner: None,
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["üe".to_string(), "yue".to_string(), "月".to_string()],
        },
        // Nasal finals (鼻韵母)
        PhonemeInfo {
            symbol: "an".to_string(),
            phoneme_type: PhonemeType::Vowel,
            place: None,
            manner: None,
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["an".to_string(), "安".to_string()],
        },
        PhonemeInfo {
            symbol: "en".to_string(),
            phoneme_type: PhonemeType::Vowel,
            place: None,
            manner: None,
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["en".to_string(), "恩".to_string()],
        },
        PhonemeInfo {
            symbol: "in".to_string(),
            phoneme_type: PhonemeType::Vowel,
            place: None,
            manner: None,
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["in".to_string(), "yin".to_string(), "音".to_string()],
        },
        PhonemeInfo {
            symbol: "un".to_string(),
            phoneme_type: PhonemeType::Vowel,
            place: None,
            manner: None,
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["un".to_string(), "wen".to_string(), "温".to_string()],
        },
        PhonemeInfo {
            symbol: "ün".to_string(),
            phoneme_type: PhonemeType::Vowel,
            place: None,
            manner: None,
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ün".to_string(), "yun".to_string(), "云".to_string()],
        },
        PhonemeInfo {
            symbol: "ang".to_string(),
            phoneme_type: PhonemeType::Vowel,
            place: None,
            manner: None,
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ang".to_string(), "昂".to_string()],
        },
        PhonemeInfo {
            symbol: "eng".to_string(),
            phoneme_type: PhonemeType::Vowel,
            place: None,
            manner: None,
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["eng".to_string(), "eng".to_string()],
        },
        PhonemeInfo {
            symbol: "ing".to_string(),
            phoneme_type: PhonemeType::Vowel,
            place: None,
            manner: None,
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ing".to_string(), "ying".to_string(), "英".to_string()],
        },
        PhonemeInfo {
            symbol: "ong".to_string(),
            phoneme_type: PhonemeType::Vowel,
            place: None,
            manner: None,
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ong".to_string(), "翁".to_string()],
        },
    ]
}

/// Special sounds in Mandarin Chinese
fn get_chinese_other_sounds() -> Vec<PhonemeInfo> {
    vec![
        // Retroflex vowel 'er' (儿化音)
        PhonemeInfo {
            symbol: "er".to_string(),
            phoneme_type: PhonemeType::Vowel,
            place: None,
            manner: None,
            voiced: true,
            height: Some(VowelHeight::Mid),
            frontness: Some(VowelFrontness::Central),
            rounded: Some(false),
            example_graphemes: vec!["er".to_string(), "儿".to_string()],
        },
        // Syllabic consonants (after zh, ch, sh, r, z, c, s)
        PhonemeInfo {
            symbol: "ɹ̩".to_string(), // Syllabic approximant
            phoneme_type: PhonemeType::Other,
            place: None,
            manner: None,
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["i".to_string(), "知".to_string(), "吃".to_string()],
        },
    ]
}

/// Grapheme to phoneme mappings for Pinyin romanization
fn get_chinese_grapheme_mappings() -> HashMap<String, Vec<String>> {
    let mut mappings = HashMap::new();

    // Initials (consonants)
    mappings.insert("b".to_string(), vec!["b".to_string()]);
    mappings.insert("p".to_string(), vec!["p".to_string()]);
    mappings.insert("m".to_string(), vec!["m".to_string()]);
    mappings.insert("f".to_string(), vec!["f".to_string()]);
    mappings.insert("d".to_string(), vec!["d".to_string()]);
    mappings.insert("t".to_string(), vec!["t".to_string()]);
    mappings.insert("n".to_string(), vec!["n".to_string()]);
    mappings.insert("l".to_string(), vec!["l".to_string()]);
    mappings.insert("g".to_string(), vec!["g".to_string()]);
    mappings.insert("k".to_string(), vec!["k".to_string()]);
    mappings.insert("h".to_string(), vec!["h".to_string()]);
    mappings.insert("j".to_string(), vec!["tɕ".to_string()]);
    mappings.insert("q".to_string(), vec!["tɕʰ".to_string()]);
    mappings.insert("x".to_string(), vec!["ɕ".to_string()]);
    mappings.insert("zh".to_string(), vec!["ʈʂ".to_string()]);
    mappings.insert("ch".to_string(), vec!["ʈʂʰ".to_string()]);
    mappings.insert("sh".to_string(), vec!["ʂ".to_string()]);
    mappings.insert("r".to_string(), vec!["ʐ".to_string()]);
    mappings.insert("z".to_string(), vec!["ts".to_string()]);
    mappings.insert("c".to_string(), vec!["tsʰ".to_string()]);
    mappings.insert("s".to_string(), vec!["s".to_string()]);

    // Finals (vowels and compounds)
    mappings.insert("a".to_string(), vec!["a".to_string()]);
    mappings.insert("o".to_string(), vec!["o".to_string()]);
    mappings.insert("e".to_string(), vec!["e".to_string()]);
    mappings.insert("i".to_string(), vec!["i".to_string()]);
    mappings.insert("u".to_string(), vec!["u".to_string()]);
    mappings.insert("ü".to_string(), vec!["ü".to_string()]);
    mappings.insert("v".to_string(), vec!["ü".to_string()]); // Alternative for ü

    // Compound finals
    mappings.insert("ai".to_string(), vec!["ai".to_string()]);
    mappings.insert("ei".to_string(), vec!["ei".to_string()]);
    mappings.insert("ui".to_string(), vec!["ui".to_string()]);
    mappings.insert("ao".to_string(), vec!["ao".to_string()]);
    mappings.insert("ou".to_string(), vec!["ou".to_string()]);
    mappings.insert("iu".to_string(), vec!["iu".to_string()]);
    mappings.insert("ie".to_string(), vec!["ie".to_string()]);
    mappings.insert("üe".to_string(), vec!["üe".to_string()]);
    mappings.insert("er".to_string(), vec!["er".to_string()]);

    // Nasal finals
    mappings.insert("an".to_string(), vec!["an".to_string()]);
    mappings.insert("en".to_string(), vec!["en".to_string()]);
    mappings.insert("in".to_string(), vec!["in".to_string()]);
    mappings.insert("un".to_string(), vec!["un".to_string()]);
    mappings.insert("ün".to_string(), vec!["ün".to_string()]);
    mappings.insert("ang".to_string(), vec!["ang".to_string()]);
    mappings.insert("eng".to_string(), vec!["eng".to_string()]);
    mappings.insert("ing".to_string(), vec!["ing".to_string()]);
    mappings.insert("ong".to_string(), vec!["ong".to_string()]);

    mappings
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chinese_inventory_structure() {
        let inventory = get_chinese_inventory();
        assert_eq!(inventory.language, LanguageCode::ZhCn);

        // Verify we have all phoneme categories
        assert!(!inventory.consonants.is_empty(), "Should have consonants");
        assert!(!inventory.vowels.is_empty(), "Should have vowels");

        // Check for specific Mandarin features
        assert!(
            inventory.is_valid_phoneme("ʈʂ"),
            "Should have retroflex affricate 'zh'"
        );
        assert!(
            inventory.is_valid_phoneme("ɕ"),
            "Should have palatal fricative 'x'"
        );
        assert!(
            inventory.is_valid_phoneme("ü"),
            "Should have fronted rounded vowel 'ü'"
        );
    }

    #[test]
    fn test_mandarin_consonant_count() {
        let consonants = get_chinese_consonants();
        // Mandarin has 21 initial consonants
        assert_eq!(consonants.len(), 21, "Mandarin should have 21 initials");
    }

    #[test]
    fn test_mandarin_vowel_finals() {
        let vowels = get_chinese_vowels();
        // Should have basic vowels plus compounds
        assert!(
            vowels.len() >= 20,
            "Should have at least 20 finals (vowels + compounds)"
        );

        // Check for specific finals
        let symbols: Vec<String> = vowels.iter().map(|v| v.symbol.clone()).collect();
        assert!(symbols.contains(&"a".to_string()), "Should have 'a'");
        assert!(symbols.contains(&"ai".to_string()), "Should have 'ai'");
        assert!(symbols.contains(&"ang".to_string()), "Should have 'ang'");
    }

    #[test]
    fn test_tone_system() {
        // Test all 5 tones
        let tone1 = MandarinTone::Tone1;
        assert_eq!(tone1.number(), 1);
        assert_eq!(tone1.contour(), "55");
        assert_eq!(tone1.diacritic_example(), "ā");

        let tone2 = MandarinTone::Tone2;
        assert_eq!(tone2.number(), 2);
        assert_eq!(tone2.contour(), "35");

        let tone3 = MandarinTone::Tone3;
        assert_eq!(tone3.number(), 3);
        assert_eq!(tone3.contour(), "214");

        let tone4 = MandarinTone::Tone4;
        assert_eq!(tone4.number(), 4);
        assert_eq!(tone4.contour(), "51");

        let neutral = MandarinTone::Neutral;
        assert_eq!(neutral.number(), 5);
        assert_eq!(neutral.contour(), "");
    }

    #[test]
    fn test_retroflex_consonants() {
        let consonants = get_chinese_consonants();
        let symbols: Vec<String> = consonants.iter().map(|c| c.symbol.clone()).collect();

        // Check for all retroflex consonants (zh, ch, sh, r)
        assert!(symbols.contains(&"ʈʂ".to_string()), "Should have 'zh' (ʈʂ)");
        assert!(
            symbols.contains(&"ʈʂʰ".to_string()),
            "Should have 'ch' (ʈʂʰ)"
        );
        assert!(symbols.contains(&"ʂ".to_string()), "Should have 'sh' (ʂ)");
        assert!(symbols.contains(&"ʐ".to_string()), "Should have 'r' (ʐ)");
    }

    #[test]
    fn test_palatal_consonants() {
        let consonants = get_chinese_consonants();
        let symbols: Vec<String> = consonants.iter().map(|c| c.symbol.clone()).collect();

        // Check for palatal consonants (j, q, x)
        assert!(symbols.contains(&"tɕ".to_string()), "Should have 'j' (tɕ)");
        assert!(
            symbols.contains(&"tɕʰ".to_string()),
            "Should have 'q' (tɕʰ)"
        );
        assert!(symbols.contains(&"ɕ".to_string()), "Should have 'x' (ɕ)");
    }

    #[test]
    fn test_grapheme_mappings() {
        let mappings = get_chinese_grapheme_mappings();

        // Test some basic mappings
        assert_eq!(mappings.get("b"), Some(&vec!["b".to_string()]));
        assert_eq!(mappings.get("zh"), Some(&vec!["ʈʂ".to_string()]));
        assert_eq!(mappings.get("x"), Some(&vec!["ɕ".to_string()]));
        assert_eq!(mappings.get("ai"), Some(&vec!["ai".to_string()]));
        assert_eq!(mappings.get("ang"), Some(&vec!["ang".to_string()]));
    }

    #[test]
    fn test_special_sounds() {
        let other = get_chinese_other_sounds();
        let symbols: Vec<String> = other.iter().map(|s| s.symbol.clone()).collect();

        // Check for 'er' sound (儿化音)
        assert!(
            symbols.contains(&"er".to_string()),
            "Should have 'er' retroflex vowel"
        );
    }

    #[test]
    fn test_all_phonemes_have_symbols() {
        let inventory = get_chinese_inventory();
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
        let consonants = get_chinese_consonants();

        // Find bilabial plosive 'b'
        let b = consonants
            .iter()
            .find(|c| c.symbol == "b")
            .expect("Should have 'b' consonant");

        assert_eq!(b.phoneme_type, PhonemeType::Consonant);
        assert_eq!(b.place, Some(PlaceOfArticulation::Bilabial));
        assert_eq!(b.manner, Some(MannerOfArticulation::Plosive));

        // Find palatal fricative 'x'
        let x = consonants
            .iter()
            .find(|c| c.symbol == "ɕ")
            .expect("Should have 'ɕ' (x) consonant");

        assert_eq!(x.place, Some(PlaceOfArticulation::Palatal));
        assert_eq!(x.manner, Some(MannerOfArticulation::Fricative));
    }

    #[test]
    fn test_vowel_features() {
        let vowels = get_chinese_vowels();

        // Find 'i' vowel
        let i = vowels
            .iter()
            .find(|v| v.symbol == "i")
            .expect("Should have 'i' vowel");

        assert_eq!(i.phoneme_type, PhonemeType::Vowel);
        assert_eq!(i.height, Some(VowelHeight::Close));
        assert_eq!(i.frontness, Some(VowelFrontness::Front));
        assert_eq!(i.rounded, Some(false));

        // Find 'ü' vowel (front rounded)
        let u_umlaut = vowels
            .iter()
            .find(|v| v.symbol == "ü")
            .expect("Should have 'ü' vowel");

        assert_eq!(u_umlaut.height, Some(VowelHeight::Close));
        assert_eq!(u_umlaut.frontness, Some(VowelFrontness::Front));
        assert_eq!(u_umlaut.rounded, Some(true));
    }
}

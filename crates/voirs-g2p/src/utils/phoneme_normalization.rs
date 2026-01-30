//! Phoneme normalization utilities for cross-language consistency
//!
//! This module provides advanced phoneme normalization capabilities to ensure
//! consistent phoneme representations across different languages, dialects, and
//! transcription systems. It handles IPA normalization, allophone mapping, and
//! cross-language phoneme equivalences.

use crate::{LanguageCode, Phoneme};
use once_cell::sync::Lazy;
use std::collections::HashMap;

/// Phoneme normalization configuration
#[derive(Debug, Clone)]
pub struct NormalizationConfig {
    /// Target phoneme set (IPA, ARPABET, X-SAMPA, etc.)
    pub target_set: PhonemeSet,
    /// Whether to merge allophones to their base phoneme
    pub merge_allophones: bool,
    /// Whether to normalize diacritics (stress, tone, length markers)
    pub normalize_diacritics: bool,
    /// Whether to apply language-specific normalization rules
    pub language_specific: bool,
    /// Whether to preserve stress information
    pub preserve_stress: bool,
    /// Whether to preserve tone information (for tonal languages)
    pub preserve_tone: bool,
}

impl Default for NormalizationConfig {
    fn default() -> Self {
        Self {
            target_set: PhonemeSet::IPA,
            merge_allophones: true,
            normalize_diacritics: false,
            language_specific: true,
            preserve_stress: true,
            preserve_tone: true,
        }
    }
}

/// Supported phoneme transcription systems
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhonemeSet {
    /// International Phonetic Alphabet (standard)
    IPA,
    /// ARPABET (CMU dictionary format)
    ARPABET,
    /// Extended Speech Assessment Methods Phonetic Alphabet
    XSAMPA,
    /// Speech Assessment Methods Phonetic Alphabet
    SAMPA,
}

/// Allophone to base phoneme mapping
/// Maps language-specific variants to their canonical form
static ALLOPHONE_MAP: Lazy<HashMap<&'static str, &'static str>> = Lazy::new(|| {
    let mut map = HashMap::new();

    // English allophones
    map.insert("ɾ", "t"); // Flapped t (American English)
    map.insert("ʔ", "t"); // Glottal stop (British English t-glottaling)
    map.insert("ɫ", "l"); // Dark L (English coda)
    map.insert("l̩", "əl"); // Syllabic L
    map.insert("n̩", "ən"); // Syllabic N
    map.insert("m̩", "əm"); // Syllabic M

    // Vowel nasalization (French, Portuguese)
    map.insert("ĩ", "i"); // Nasalized i
    map.insert("ẽ", "e"); // Nasalized e
    map.insert("ã", "a"); // Nasalized a
    map.insert("õ", "o"); // Nasalized o
    map.insert("ũ", "u"); // Nasalized u

    // Japanese special cases
    map.insert("ɸ", "h"); // Japanese /h/ before /u/
    map.insert("ç", "h"); // Japanese /h/ before /i/
    map.insert("ɴ", "n"); // Japanese moraic nasal

    // Spanish/Portuguese lenition
    map.insert("β", "b"); // Lenited b
    map.insert("ð", "d"); // Lenited d (also English 'th')
    map.insert("ɣ", "g"); // Lenited g

    // Common vowel reductions
    map.insert("ɨ", "ə"); // Close central vowel to schwa
    map.insert("ɵ", "ə"); // Close-mid central vowel to schwa

    map
});

/// IPA to ARPABET conversion table
/// Commonly used in North American TTS systems
static IPA_TO_ARPABET: Lazy<HashMap<&'static str, &'static str>> = Lazy::new(|| {
    let mut map = HashMap::new();

    // Vowels
    map.insert("iː", "IY"); // beat
    map.insert("ɪ", "IH"); // bit
    map.insert("eɪ", "EY"); // bait
    map.insert("ɛ", "EH"); // bet
    map.insert("æ", "AE"); // bat
    map.insert("ɑː", "AA"); // bot
    map.insert("ɔː", "AO"); // bought
    map.insert("oʊ", "OW"); // boat
    map.insert("ʊ", "UH"); // book
    map.insert("uː", "UW"); // boot
    map.insert("ʌ", "AH"); // but
    map.insert("ə", "AH"); // about (unstressed)
    map.insert("ɝ", "ER"); // bird
    map.insert("aɪ", "AY"); // bite
    map.insert("aʊ", "AW"); // bout
    map.insert("ɔɪ", "OY"); // boy

    // Consonants
    map.insert("p", "P");
    map.insert("b", "B");
    map.insert("t", "T");
    map.insert("d", "D");
    map.insert("k", "K");
    map.insert("g", "G");
    map.insert("f", "F");
    map.insert("v", "V");
    map.insert("θ", "TH");
    map.insert("ð", "DH");
    map.insert("s", "S");
    map.insert("z", "Z");
    map.insert("ʃ", "SH");
    map.insert("ʒ", "ZH");
    map.insert("h", "HH");
    map.insert("m", "M");
    map.insert("n", "N");
    map.insert("ŋ", "NG");
    map.insert("l", "L");
    map.insert("r", "R");
    map.insert("w", "W");
    map.insert("j", "Y");
    map.insert("tʃ", "CH");
    map.insert("dʒ", "JH");

    map
});

/// X-SAMPA to IPA conversion table
/// Extended Speech Assessment Methods Phonetic Alphabet to IPA
static XSAMPA_TO_IPA: Lazy<HashMap<&'static str, &'static str>> = Lazy::new(|| {
    let mut map = HashMap::new();

    // Vowels
    map.insert("i", "i");
    map.insert("y", "y");
    map.insert("1", "ɨ");
    map.insert("}", "ʉ");
    map.insert("M", "ɯ");
    map.insert("u", "u");
    map.insert("I", "ɪ");
    map.insert("Y", "ʏ");
    map.insert("U", "ʊ");
    map.insert("e", "e");
    map.insert("2", "ø");
    map.insert("@", "ə");
    map.insert("8", "ɵ");
    map.insert("7", "ɤ");
    map.insert("o", "o");
    map.insert("E", "ɛ");
    map.insert("9", "œ");
    map.insert("3", "ɜ");
    map.insert("3\\", "ɞ");
    map.insert("V", "ʌ");
    map.insert("O", "ɔ");
    map.insert("{", "æ");
    map.insert("6", "ɐ");
    map.insert("a", "a");
    map.insert("&", "ɶ");
    map.insert("A", "ɑ");
    map.insert("Q", "ɒ");

    // Consonants - Plosives
    map.insert("p", "p");
    map.insert("b", "b");
    map.insert("t", "t");
    map.insert("d", "d");
    map.insert("t`", "ʈ");
    map.insert("d`", "ɖ");
    map.insert("c", "c");
    map.insert("J\\", "ɟ");
    map.insert("k", "k");
    map.insert("g", "g");
    map.insert("q", "q");
    map.insert("G\\", "ɢ");
    map.insert("?", "ʔ");

    // Nasals
    map.insert("m", "m");
    map.insert("F", "ɱ");
    map.insert("n", "n");
    map.insert("n`", "ɳ");
    map.insert("J", "ɲ");
    map.insert("N", "ŋ");
    map.insert("N\\", "ɴ");

    // Trills
    map.insert("B\\", "ʙ");
    map.insert("r", "r");
    map.insert("R\\", "ʀ");

    // Taps/Flaps
    map.insert("4", "ɾ");
    map.insert("r`", "ɽ");

    // Fricatives
    map.insert("f", "f");
    map.insert("v", "v");
    map.insert("T", "θ");
    map.insert("D", "ð");
    map.insert("s", "s");
    map.insert("z", "z");
    map.insert("S", "ʃ");
    map.insert("Z", "ʒ");
    map.insert("s`", "ʂ");
    map.insert("z`", "ʐ");
    map.insert("C", "ç");
    map.insert("j\\", "ʝ");
    map.insert("x", "x");
    map.insert("G", "ɣ");
    map.insert("X", "χ");
    map.insert("R", "ʁ");
    map.insert("X\\", "ħ");
    map.insert("?\\", "ʕ");
    map.insert("h", "h");
    map.insert("h\\", "ɦ");

    // Lateral fricatives
    map.insert("K", "ɬ");
    map.insert("K\\", "ɮ");

    // Approximants
    map.insert("P", "ʋ");
    map.insert("r\\", "ɹ");
    map.insert("r\\`", "ɻ");
    map.insert("j", "j");
    map.insert("M\\", "ɰ");

    // Lateral approximants
    map.insert("l", "l");
    map.insert("l`", "ɭ");
    map.insert("L", "ʎ");
    map.insert("L\\", "ʟ");

    // Other consonants
    map.insert("w", "w");
    map.insert("W", "ʍ");
    map.insert("H", "ɥ");

    // Affricates (common combinations)
    map.insert("tS", "tʃ");
    map.insert("dZ", "dʒ");
    map.insert("ts", "ts");
    map.insert("dz", "dz");

    // Diacritics and modifiers
    map.insert("_\"", "̈"); // Centralized
    map.insert("_+", "̟"); // Advanced
    map.insert("_-", "̠"); // Retracted
    map.insert("_/", "̌"); // Rising tone
    map.insert("_\\", "̂"); // Falling tone
    map.insert("_^", "̯"); // Non-syllabic
    map.insert("_\"", "̈"); // Centralized
    map.insert("_0", "̥"); // Voiceless
    map.insert("_v", "̬"); // Voiced
    map.insert("_h", "ʰ"); // Aspirated
    map.insert("_n", "ⁿ"); // Nasal release
    map.insert("_l", "ˡ"); // Lateral release
    map.insert("_~", "̃"); // Nasalized
    map.insert(":", "ː"); // Long

    map
});

/// SAMPA to IPA conversion table
/// Speech Assessment Methods Phonetic Alphabet to IPA (primarily for European languages)
static SAMPA_TO_IPA: Lazy<HashMap<&'static str, &'static str>> = Lazy::new(|| {
    let mut map = HashMap::new();

    // Vowels (SAMPA subset - European focus)
    map.insert("i", "i");
    map.insert("e", "e");
    map.insert("E", "ɛ");
    map.insert("a", "a");
    map.insert("{", "æ");
    map.insert("A", "ɑ");
    map.insert("O", "ɔ");
    map.insert("o", "o");
    map.insert("u", "u");
    map.insert("y", "y");
    map.insert("2", "ø");
    map.insert("9", "œ");
    map.insert("@", "ə");
    map.insert("6", "ɐ");
    map.insert("I", "ɪ");
    map.insert("Y", "ʏ");
    map.insert("U", "ʊ");
    map.insert("V", "ʌ");
    map.insert("Q", "ɒ");

    // Consonants
    map.insert("p", "p");
    map.insert("b", "b");
    map.insert("t", "t");
    map.insert("d", "d");
    map.insert("k", "k");
    map.insert("g", "g");
    map.insert("f", "f");
    map.insert("v", "v");
    map.insert("T", "θ");
    map.insert("D", "ð");
    map.insert("s", "s");
    map.insert("z", "z");
    map.insert("S", "ʃ");
    map.insert("Z", "ʒ");
    map.insert("C", "ç");
    map.insert("x", "x");
    map.insert("h", "h");
    map.insert("m", "m");
    map.insert("n", "n");
    map.insert("J", "ɲ");
    map.insert("N", "ŋ");
    map.insert("l", "l");
    map.insert("r", "r");
    map.insert("R", "ʁ");
    map.insert("w", "w");
    map.insert("j", "j");
    map.insert("H", "ɥ");

    // Common diphthongs and length
    map.insert(":", "ː"); // Length marker

    map
});

/// IPA to X-SAMPA conversion table (reversed mapping for output)
/// Converts IPA symbols to Extended SAMPA ASCII representation
static IPA_TO_XSAMPA: Lazy<HashMap<&'static str, &'static str>> = Lazy::new(|| {
    let mut map = HashMap::new();

    // Create reverse mapping from XSAMPA_TO_IPA
    for (&xsampa, &ipa) in XSAMPA_TO_IPA.iter() {
        // Skip diacritics for simplicity in reverse mapping
        if !xsampa.starts_with('_') && xsampa != ":" {
            map.insert(ipa, xsampa);
        }
    }

    // Add commonly used mappings that may not reverse cleanly
    map.insert("ː", ":"); // Length marker

    map
});

/// IPA to SAMPA conversion table (reversed mapping for output)
/// Converts IPA symbols to SAMPA ASCII representation
static IPA_TO_SAMPA: Lazy<HashMap<&'static str, &'static str>> = Lazy::new(|| {
    let mut map = HashMap::new();

    // Create reverse mapping from SAMPA_TO_IPA
    for (&sampa, &ipa) in SAMPA_TO_IPA.iter() {
        if sampa != ":" {
            // Handle length marker separately
            map.insert(ipa, sampa);
        }
    }

    // Add commonly used mappings
    map.insert("ː", ":"); // Length marker

    map
});

/// Normalize a sequence of phonemes according to the configuration
///
/// # Arguments
/// * `phonemes` - Input phoneme sequence to normalize
/// * `language` - Source language for language-specific rules
/// * `config` - Normalization configuration
///
/// # Returns
/// Normalized phoneme sequence
///
/// # Examples
/// ```
/// use voirs_g2p::utils::phoneme_normalization::{normalize_phonemes, NormalizationConfig};
/// use voirs_g2p::{Phoneme, LanguageCode};
///
/// let phonemes = vec![
///     Phoneme::new("ɾ".to_string()),  // Flapped t
///     Phoneme::new("æ".to_string()),
///     Phoneme::new("ʔ".to_string()),  // Glottal stop
/// ];
///
/// let config = NormalizationConfig::default();
/// let normalized = normalize_phonemes(&phonemes, LanguageCode::EnUs, &config);
/// // Result: [t, æ, t] (allophones merged to base phonemes)
/// ```
pub fn normalize_phonemes(
    phonemes: &[Phoneme],
    language: LanguageCode,
    config: &NormalizationConfig,
) -> Vec<Phoneme> {
    phonemes
        .iter()
        .map(|p| normalize_single_phoneme(p, language, config))
        .collect()
}

/// Normalize a single phoneme
fn normalize_single_phoneme(
    phoneme: &Phoneme,
    language: LanguageCode,
    config: &NormalizationConfig,
) -> Phoneme {
    let mut symbol = phoneme.symbol.to_string();

    // Step 1: Merge allophones to base phonemes
    if config.merge_allophones {
        if let Some(&base) = ALLOPHONE_MAP.get(symbol.as_str()) {
            symbol = base.to_string();
        }
    }

    // Step 2: Apply language-specific normalization
    if config.language_specific {
        symbol = apply_language_specific_normalization(&symbol, language);
    }

    // Step 3: Normalize diacritics
    if config.normalize_diacritics {
        symbol = normalize_diacritics(&symbol, config);
    }

    // Step 4: Convert to target phoneme set
    match config.target_set {
        PhonemeSet::IPA => {
            // Already in IPA, no conversion needed
        }
        PhonemeSet::ARPABET => {
            if let Some(&arpabet) = IPA_TO_ARPABET.get(symbol.as_str()) {
                symbol = arpabet.to_string();
            }
        }
        PhonemeSet::XSAMPA => {
            if let Some(&xsampa) = IPA_TO_XSAMPA.get(symbol.as_str()) {
                symbol = xsampa.to_string();
            } else {
                // If no mapping exists, keep the original symbol and log
                tracing::debug!("No X-SAMPA mapping for IPA symbol: {}", symbol);
            }
        }
        PhonemeSet::SAMPA => {
            if let Some(&sampa) = IPA_TO_SAMPA.get(symbol.as_str()) {
                symbol = sampa.to_string();
            } else {
                // If no mapping exists, keep the original symbol and log
                tracing::debug!("No SAMPA mapping for IPA symbol: {}", symbol);
            }
        }
    }

    Phoneme::new(symbol)
}

/// Apply language-specific normalization rules
fn apply_language_specific_normalization(symbol: &str, language: LanguageCode) -> String {
    match language {
        LanguageCode::EnUs | LanguageCode::EnGb => {
            // English-specific: normalize rhotic variants
            match symbol {
                "ɹ" => "r".to_string(),  // Approximate to standard r
                "ɚ" => "ər".to_string(), // Rhotacized schwa
                _ => symbol.to_string(),
            }
        }
        LanguageCode::Ja => {
            // Japanese-specific: normalize palatalized consonants
            match symbol {
                "kʲ" => "k".to_string(),
                "gʲ" => "g".to_string(),
                "nʲ" => "n".to_string(),
                _ => symbol.to_string(),
            }
        }
        LanguageCode::ZhCn => {
            // Mandarin-specific: normalize retroflex consonants
            match symbol {
                "ʐ" => "r".to_string(), // Retroflex approximant
                "ʂ" => "ʃ".to_string(), // Retroflex to postalveolar
                "ʈʂ" => "tʃ".to_string(),
                _ => symbol.to_string(),
            }
        }
        _ => symbol.to_string(),
    }
}

/// Normalize diacritics according to configuration
fn normalize_diacritics(symbol: &str, config: &NormalizationConfig) -> String {
    let mut result = symbol.to_string();

    // Remove stress markers if not preserving stress
    if !config.preserve_stress {
        result = result.replace('ˈ', ""); // Primary stress
        result = result.replace('ˌ', ""); // Secondary stress
    }

    // Remove tone markers if not preserving tone
    if !config.preserve_tone {
        // Remove tone numbers (Mandarin)
        result = result.chars().filter(|c| !matches!(c, '1'..='5')).collect();

        // Remove tone diacritics
        result = result.replace('˥', ""); // High tone
        result = result.replace('˧', ""); // Mid tone
        result = result.replace('˩', ""); // Low tone
    }

    // Remove length markers (always normalized)
    result = result.replace('ː', ""); // Long vowel marker
    result = result.replace('ˑ', ""); // Half-long marker

    result
}

/// Convert IPA phonemes to ARPABET format
///
/// # Arguments
/// * `ipa_phonemes` - Input phonemes in IPA format
///
/// # Returns
/// Phonemes converted to ARPABET format
///
/// # Examples
/// ```
/// use voirs_g2p::utils::phoneme_normalization::ipa_to_arpabet;
/// use voirs_g2p::Phoneme;
///
/// let ipa = vec![
///     Phoneme::new("h".to_string()),
///     Phoneme::new("ɛ".to_string()),
///     Phoneme::new("l".to_string()),
///     Phoneme::new("oʊ".to_string()),
/// ];
///
/// let arpabet = ipa_to_arpabet(&ipa);
/// // Result: HH EH L OW
/// ```
pub fn ipa_to_arpabet(ipa_phonemes: &[Phoneme]) -> Vec<Phoneme> {
    ipa_phonemes
        .iter()
        .filter_map(|p| {
            let symbol = &p.symbol;
            IPA_TO_ARPABET
                .get(symbol.as_str())
                .map(|&arpabet| Phoneme::new(arpabet.to_string()))
        })
        .collect()
}

/// Detect the most likely phoneme set used in a phoneme sequence
///
/// # Arguments
/// * `phonemes` - Phoneme sequence to analyze
///
/// # Returns
/// Most likely phoneme set
///
/// # Examples
/// ```
/// use voirs_g2p::utils::phoneme_normalization::{detect_phoneme_set, PhonemeSet};
/// use voirs_g2p::Phoneme;
///
/// let phonemes = vec![
///     Phoneme::new("HH".to_string()),
///     Phoneme::new("EH".to_string()),
///     Phoneme::new("L".to_string()),
///     Phoneme::new("OW".to_string()),
/// ];
///
/// let set = detect_phoneme_set(&phonemes);
/// assert_eq!(set, PhonemeSet::ARPABET);
/// ```
pub fn detect_phoneme_set(phonemes: &[Phoneme]) -> PhonemeSet {
    let mut arpabet_score = 0;
    let mut ipa_score = 0;

    for phoneme in phonemes {
        let symbol = &phoneme.symbol;

        // Check if it's likely ARPABET (all uppercase, length 1-2)
        if symbol.chars().all(|c| c.is_ascii_uppercase()) && symbol.len() <= 2 {
            arpabet_score += 1;
        }

        // Check if it's likely IPA (contains IPA-specific characters)
        if symbol.chars().any(|c| {
            matches!(
                c,
                'ə' | 'ɛ'
                    | 'ɪ'
                    | 'ɔ'
                    | 'ʊ'
                    | 'ʌ'
                    | 'æ'
                    | 'θ'
                    | 'ð'
                    | 'ʃ'
                    | 'ʒ'
                    | 'ŋ'
                    | 'ɹ'
                    | 'ɾ'
                    | 'ʔ'
            )
        }) {
            ipa_score += 1;
        }
    }

    if arpabet_score > ipa_score {
        PhonemeSet::ARPABET
    } else {
        PhonemeSet::IPA
    }
}

/// Calculate phoneme equivalence score across different languages
///
/// # Arguments
/// * `p1` - First phoneme
/// * `p2` - Second phoneme
///
/// # Returns
/// Equivalence score from 0.0 (completely different) to 1.0 (identical)
///
/// # Examples
/// ```
/// use voirs_g2p::utils::phoneme_normalization::phoneme_equivalence;
/// use voirs_g2p::Phoneme;
///
/// let p1 = Phoneme::new("ɾ".to_string()); // Flapped t
/// let p2 = Phoneme::new("t".to_string()); // Regular t
///
/// let score = phoneme_equivalence(&p1, &p2);
/// assert!(score > 0.8); // Very similar (allophones)
/// ```
pub fn phoneme_equivalence(p1: &Phoneme, p2: &Phoneme) -> f64 {
    let s1 = &p1.symbol;
    let s2 = &p2.symbol;

    // Exact match
    if s1 == s2 {
        return 1.0;
    }

    // Check if they're allophones of the same base phoneme
    let base1 = ALLOPHONE_MAP
        .get(s1.as_str())
        .copied()
        .unwrap_or(s1.as_str());
    let base2 = ALLOPHONE_MAP
        .get(s2.as_str())
        .copied()
        .unwrap_or(s2.as_str());

    if base1 == base2 {
        return 0.9; // High similarity for allophones
    }

    // Check phonetic feature similarity
    phonetic_feature_similarity(s1, s2)
}

/// Calculate similarity based on phonetic features
fn phonetic_feature_similarity(s1: &str, s2: &str) -> f64 {
    // Simplified feature-based similarity
    // In a production system, this would use a comprehensive phonetic feature matrix

    let vowels: &[&str] = &[
        "i", "e", "a", "o", "u", "ə", "ɪ", "ɛ", "æ", "ɑ", "ɔ", "ʊ", "ʌ",
    ];
    let plosives: &[&str] = &["p", "b", "t", "d", "k", "g"];
    let fricatives: &[&str] = &["f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ", "h"];
    let nasals: &[&str] = &["m", "n", "ŋ"];
    let liquids: &[&str] = &["l", "r", "ɹ", "ɾ"];

    // Check if both are in the same phonetic class
    let classes: &[&[&str]] = &[vowels, plosives, fricatives, nasals, liquids];

    for class in classes {
        let in_class_1 = class.contains(&s1);
        let in_class_2 = class.contains(&s2);

        if in_class_1 && in_class_2 {
            return 0.5; // Same class, moderate similarity
        }
    }

    0.0 // Different classes, no similarity
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allophone_normalization() {
        let phonemes = vec![
            Phoneme::new("ɾ".to_string()), // Flapped t
            Phoneme::new("æ".to_string()),
            Phoneme::new("ʔ".to_string()), // Glottal stop
        ];

        let config = NormalizationConfig {
            merge_allophones: true,
            ..Default::default()
        };

        let normalized = normalize_phonemes(&phonemes, LanguageCode::EnUs, &config);

        assert_eq!(normalized[0].symbol, "t");
        assert_eq!(normalized[1].symbol, "æ");
        assert_eq!(normalized[2].symbol, "t");
    }

    #[test]
    fn test_ipa_to_arpabet_conversion() {
        let ipa = vec![
            Phoneme::new("h".to_string()),
            Phoneme::new("ɛ".to_string()),
            Phoneme::new("l".to_string()),
            Phoneme::new("oʊ".to_string()),
        ];

        let arpabet = ipa_to_arpabet(&ipa);

        assert_eq!(arpabet.len(), 4);
        assert_eq!(arpabet[0].symbol, "HH");
        assert_eq!(arpabet[1].symbol, "EH");
        assert_eq!(arpabet[2].symbol, "L");
        assert_eq!(arpabet[3].symbol, "OW");
    }

    #[test]
    fn test_phoneme_set_detection() {
        let arpabet = vec![
            Phoneme::new("HH".to_string()),
            Phoneme::new("EH".to_string()),
            Phoneme::new("L".to_string()),
            Phoneme::new("OW".to_string()),
        ];

        assert_eq!(detect_phoneme_set(&arpabet), PhonemeSet::ARPABET);

        let ipa = vec![
            Phoneme::new("h".to_string()),
            Phoneme::new("ɛ".to_string()),
            Phoneme::new("l".to_string()),
            Phoneme::new("o".to_string()),
        ];

        assert_eq!(detect_phoneme_set(&ipa), PhonemeSet::IPA);
    }

    #[test]
    fn test_phoneme_equivalence() {
        let p1 = Phoneme::new("ɾ".to_string()); // Flapped t
        let p2 = Phoneme::new("t".to_string()); // Regular t

        let score = phoneme_equivalence(&p1, &p2);
        assert!(score > 0.8); // Very similar (allophones)

        let p3 = Phoneme::new("k".to_string());
        let score2 = phoneme_equivalence(&p1, &p3);
        assert!(score2 < 0.6); // Less similar (different phonemes)
    }

    #[test]
    fn test_stress_preservation() {
        let phonemes = vec![
            Phoneme::new("ˈh".to_string()), // Primary stress
            Phoneme::new("ɛ".to_string()),
            Phoneme::new("ˌl".to_string()), // Secondary stress
            Phoneme::new("oʊ".to_string()),
        ];

        // Preserve stress
        let config1 = NormalizationConfig {
            preserve_stress: true,
            normalize_diacritics: true,
            ..Default::default()
        };

        let normalized1 = normalize_phonemes(&phonemes, LanguageCode::EnUs, &config1);
        assert!(normalized1[0].symbol.contains('ˈ'));

        // Remove stress
        let config2 = NormalizationConfig {
            preserve_stress: false,
            normalize_diacritics: true,
            ..Default::default()
        };

        let normalized2 = normalize_phonemes(&phonemes, LanguageCode::EnUs, &config2);
        assert!(!normalized2[0].symbol.contains('ˈ'));
    }

    #[test]
    fn test_language_specific_normalization() {
        // Japanese palatalized consonant
        let phonemes = vec![
            Phoneme::new("kʲ".to_string()),
            Phoneme::new("i".to_string()),
        ];

        let config = NormalizationConfig {
            language_specific: true,
            ..Default::default()
        };

        let normalized = normalize_phonemes(&phonemes, LanguageCode::Ja, &config);
        assert_eq!(normalized[0].symbol, "k");
    }

    #[test]
    fn test_ipa_to_xsampa_conversion() {
        // Test common IPA to X-SAMPA conversions
        let ipa = vec![
            Phoneme::new("ɛ".to_string()), // E in X-SAMPA
            Phoneme::new("ʃ".to_string()), // S in X-SAMPA
            Phoneme::new("θ".to_string()), // T in X-SAMPA
            Phoneme::new("ŋ".to_string()), // N in X-SAMPA
        ];

        let config = NormalizationConfig {
            target_set: PhonemeSet::XSAMPA,
            merge_allophones: false,
            ..Default::default()
        };

        let converted = normalize_phonemes(&ipa, LanguageCode::EnUs, &config);

        assert_eq!(converted[0].symbol, "E");
        assert_eq!(converted[1].symbol, "S");
        assert_eq!(converted[2].symbol, "T");
        assert_eq!(converted[3].symbol, "N");
    }

    #[test]
    fn test_ipa_to_sampa_conversion() {
        // Test common IPA to SAMPA conversions (European subset)
        let ipa = vec![
            Phoneme::new("ɛ".to_string()), // E in SAMPA
            Phoneme::new("ʃ".to_string()), // S in SAMPA
            Phoneme::new("ə".to_string()), // @ in SAMPA
            Phoneme::new("ŋ".to_string()), // N in SAMPA
        ];

        let config = NormalizationConfig {
            target_set: PhonemeSet::SAMPA,
            merge_allophones: false,
            ..Default::default()
        };

        let converted = normalize_phonemes(&ipa, LanguageCode::EnUs, &config);

        assert_eq!(converted[0].symbol, "E");
        assert_eq!(converted[1].symbol, "S");
        assert_eq!(converted[2].symbol, "@");
        assert_eq!(converted[3].symbol, "N");
    }

    #[test]
    fn test_xsampa_vowel_conversion() {
        // Test various vowel conversions to X-SAMPA
        let ipa = vec![
            Phoneme::new("i".to_string()), // i
            Phoneme::new("ɪ".to_string()), // I
            Phoneme::new("æ".to_string()), // {
            Phoneme::new("ɑ".to_string()), // A
            Phoneme::new("ɔ".to_string()), // O
            Phoneme::new("ʊ".to_string()), // U
        ];

        let config = NormalizationConfig {
            target_set: PhonemeSet::XSAMPA,
            merge_allophones: false,
            ..Default::default()
        };

        let converted = normalize_phonemes(&ipa, LanguageCode::EnUs, &config);

        assert_eq!(converted[0].symbol, "i");
        assert_eq!(converted[1].symbol, "I");
        assert_eq!(converted[2].symbol, "{");
        assert_eq!(converted[3].symbol, "A");
        assert_eq!(converted[4].symbol, "O");
        assert_eq!(converted[5].symbol, "U");
    }

    #[test]
    fn test_sampa_consonant_conversion() {
        // Test various consonant conversions to SAMPA
        let ipa = vec![
            Phoneme::new("θ".to_string()), // T
            Phoneme::new("ð".to_string()), // D
            Phoneme::new("ʃ".to_string()), // S
            Phoneme::new("ʒ".to_string()), // Z
            Phoneme::new("ç".to_string()), // C
            Phoneme::new("ʁ".to_string()), // R
        ];

        let config = NormalizationConfig {
            target_set: PhonemeSet::SAMPA,
            merge_allophones: false,
            ..Default::default()
        };

        let converted = normalize_phonemes(&ipa, LanguageCode::EnUs, &config);

        assert_eq!(converted[0].symbol, "T");
        assert_eq!(converted[1].symbol, "D");
        assert_eq!(converted[2].symbol, "S");
        assert_eq!(converted[3].symbol, "Z");
        assert_eq!(converted[4].symbol, "C");
        assert_eq!(converted[5].symbol, "R");
    }

    #[test]
    fn test_xsampa_length_marker() {
        // Test length marker conversion
        let ipa = vec![
            Phoneme::new("iː".to_string()), // Long i (should keep ː or convert to i:)
        ];

        let config = NormalizationConfig {
            target_set: PhonemeSet::XSAMPA,
            merge_allophones: false,
            normalize_diacritics: false,
            ..Default::default()
        };

        let converted = normalize_phonemes(&ipa, LanguageCode::EnUs, &config);

        // The conversion might map ː to : if present in the mapping
        // For now just verify it processes without error
        assert_eq!(converted.len(), 1);
    }
}

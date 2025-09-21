//! Phoneme analysis and validation utilities for G2P conversion.

use crate::{LanguageCode, Phoneme};

/// Validate phoneme sequence
pub fn validate_phonemes(phonemes: &[Phoneme], language: LanguageCode) -> bool {
    if phonemes.is_empty() {
        return true; // Empty sequence is valid
    }

    // Check valid phoneme inventory for the language
    if !validate_phoneme_inventory(phonemes, language) {
        return false;
    }

    // Check basic phonotactic constraints
    if !validate_phonotactic_constraints(phonemes, language) {
        return false;
    }

    // Check syllable structure validity
    if !validate_syllable_structure(phonemes, language) {
        return false;
    }

    true
}

/// Enhanced phoneme analysis with detailed statistics
pub fn analyze_phoneme_sequence(phonemes: &[Phoneme]) -> PhonemeAnalysis {
    let mut analysis = PhonemeAnalysis::default();

    for phoneme in phonemes {
        let symbol = phoneme.effective_symbol();

        if is_vowel(symbol) {
            analysis.vowel_count += 1;
            if phoneme.stress > 0 {
                analysis.stressed_vowels += 1;
            }
        } else if is_consonant(symbol) {
            analysis.consonant_count += 1;
        }

        if phoneme.is_syllable_boundary {
            analysis.syllable_count += 1;
        }

        if phoneme.is_word_boundary {
            analysis.word_count += 1;
        }

        analysis.total_duration_ms += phoneme.duration_ms.unwrap_or(0.0);
        analysis.average_confidence += phoneme.confidence;
    }

    if !phonemes.is_empty() {
        analysis.average_confidence /= phonemes.len() as f32;
        analysis.vowel_consonant_ratio = if analysis.consonant_count > 0 {
            analysis.vowel_count as f32 / analysis.consonant_count as f32
        } else {
            analysis.vowel_count as f32
        };
    }

    analysis
}

/// Phoneme sequence analysis results
#[derive(Debug, Clone, Default)]
pub struct PhonemeAnalysis {
    pub vowel_count: u32,
    pub consonant_count: u32,
    pub stressed_vowels: u32,
    pub syllable_count: u32,
    pub word_count: u32,
    pub vowel_consonant_ratio: f32,
    pub total_duration_ms: f32,
    pub average_confidence: f32,
}

/// Get valid phoneme inventory for a language as a slice (optimized version)
pub fn get_valid_phoneme_inventory_slice(language: LanguageCode) -> &'static [&'static str] {
    match language {
        LanguageCode::EnUs | LanguageCode::EnGb => &[
            // English consonants
            "p", "b", "t", "d", "k", "g", "f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ", "h", "m", "n",
            "ŋ", "l", "ɹ", "j", "w", "tʃ", "dʒ", // English vowels
            "i", "ɪ", "e", "ɛ", "æ", "ə", "ɚ", "ɜ", "ʌ", "a", "ɑ", "ɔ", "o", "ʊ", "u",
            // English diphthongs
            "aɪ", "aʊ", "ɔɪ", "eɪ", "oʊ", "iː", "uː", "ɜːr", "ɑːr", "ɔːr",
            // Common combinations
            "ks", "kw", "sp", "st", "sk", "spl", "str", "skr",
        ],

        LanguageCode::De => &[
            // German consonants
            "p", "b", "t", "d", "k", "g", "f", "v", "s", "z", "ʃ", "ʒ", "ç", "x", "h", "m", "n",
            "ŋ", "l", "ʁ", "j", "pf", "ts", "tʃ", "dʒ", // German vowels
            "i", "ɪ", "e", "ɛ", "ø", "œ", "y", "ʏ", "ə", "a", "ɑ", "o", "ɔ", "u", "ʊ",
            // German umlauts
            "aɪ", "aʊ", "ɔɪ",
        ],

        LanguageCode::Fr => &[
            // French consonants
            "p", "b", "t", "d", "k", "g", "f", "v", "s", "z", "ʃ", "ʒ", "m", "n", "ɲ", "ŋ", "l",
            "ʁ", "j", "w", "ɥ", // French vowels
            "i", "e", "ɛ", "a", "ɑ", "ɔ", "o", "u", "y", "ø", "œ", "ə", "ɛ̃", "ɑ̃", "ɔ̃", "œ̃",
        ],

        LanguageCode::Es => &[
            // Spanish consonants
            "p", "b", "t", "d", "k", "g", "f", "θ", "s", "x", "m", "n", "ɲ", "l", "ʎ", "ɾ", "r",
            "j", "w", "tʃ", // Spanish vowels
            "i", "e", "a", "o", "u",
        ],

        LanguageCode::It => &[
            // Italian consonants
            "p", "b", "t", "d", "k", "g", "f", "v", "s", "z", "ʃ", "m", "n", "ɲ", "l", "ʎ", "ɾ",
            "r", "j", "w", "tʃ", "dʒ", "ts", "dz", // Italian vowels
            "i", "e", "ɛ", "a", "ɔ", "o", "u",
        ],

        LanguageCode::Pt => &[
            // Portuguese consonants
            "p", "b", "t", "d", "k", "g", "f", "v", "s", "z", "ʃ", "ʒ", "m", "n", "ɲ", "l", "ʎ",
            "ɾ", "r", "j", "w", "tʃ", "dʒ", // Portuguese vowels
            "i", "ĩ", "e", "ẽ", "ɛ", "a", "ã", "ɔ", "õ", "o", "u", "ũ",
        ],

        LanguageCode::Ja => &[
            // Japanese consonants (romanized)
            "k", "g", "s", "z", "t", "d", "n", "h", "b", "p", "m", "y", "r", "w", "ʔ",
            // Japanese vowels
            "a", "i", "u", "e", "o", // Special Japanese sounds
            "ʃi", "ʃu", "tʃi", "tʃu", "dʒi", "dʒu", "ts", "ɲ",
        ],

        LanguageCode::ZhCn => &[
            // Mandarin initials
            "p", "pʰ", "m", "f", "t", "tʰ", "n", "l", "ts", "tsʰ", "s", "tʃ", "tʃʰ", "ʃ", "tʂ",
            "tʂʰ", "ʂ", "ʐ", "k", "kʰ", "x", "tɕ", "tɕʰ", "ɕ", // Mandarin finals
            "a", "o", "e", "ai", "ei", "ao", "ou", "an", "en", "ang", "eng", "er", "i", "ia", "ie",
            "iao", "iou", "ian", "in", "iang", "ing", "iong", "u", "ua", "uo", "uai", "uei", "uan",
            "uen", "uang", "ueng", "ü", "üe", "üan", "ün",
        ],

        _ => &[
            // Default/fallback phoneme set
            "a", "e", "i", "o", "u", "p", "t", "k", "s", "m", "n", "l", "r",
        ],
    }
}

/// Get valid phoneme inventory for a language (legacy function for compatibility)
#[allow(dead_code)]
pub fn get_valid_phoneme_inventory(language: LanguageCode) -> Vec<String> {
    get_valid_phoneme_inventory_slice(language)
        .iter()
        .map(|s| s.to_string())
        .collect()
}

/// Validate that all phonemes are in the valid inventory for the language
fn validate_phoneme_inventory(phonemes: &[Phoneme], language: LanguageCode) -> bool {
    let valid_phonemes = get_valid_phoneme_inventory_slice(language);

    for phoneme in phonemes {
        let symbol = phoneme.effective_symbol();
        if symbol != " " && symbol != "." && !valid_phonemes.contains(&symbol) {
            // Allow word boundaries and syllable boundaries, check others
            if !phoneme.is_word_boundary && !phoneme.is_syllable_boundary {
                return false;
            }
        }
    }

    true
}

/// Validate basic phonotactic constraints
fn validate_phonotactic_constraints(phonemes: &[Phoneme], language: LanguageCode) -> bool {
    match language {
        LanguageCode::EnUs | LanguageCode::EnGb => validate_english_phonotactics(phonemes),
        LanguageCode::De => validate_german_phonotactics(phonemes),
        LanguageCode::Ja => validate_japanese_phonotactics(phonemes),
        _ => true, // No specific constraints for other languages yet
    }
}

/// Validate English phonotactic constraints
fn validate_english_phonotactics(phonemes: &[Phoneme]) -> bool {
    for i in 0..phonemes.len() {
        let current = &phonemes[i].effective_symbol();

        // Check for invalid consonant clusters
        if i > 0 {
            let prev = &phonemes[i - 1].effective_symbol();

            // No triple consonant clusters in English (except across syllable boundaries)
            if i > 1 && is_consonant(prev) && is_consonant(current) {
                let prev_prev = &phonemes[i - 2].effective_symbol();
                if is_consonant(prev_prev) && !phonemes[i - 1].is_syllable_boundary {
                    return false;
                }
            }

            // Invalid consonant combinations
            if ((*current == "p" || *current == "b") && *prev == "ŋ")
                || (*prev == "ʃ" && *current == "z")
            {
                return false;
            }
        }
    }

    true
}

/// Validate German phonotactic constraints
fn validate_german_phonotactics(phonemes: &[Phoneme]) -> bool {
    for i in 0..phonemes.len() {
        let current = &phonemes[i].effective_symbol();

        // German-specific constraints
        if i > 0 {
            let prev = &phonemes[i - 1].effective_symbol();

            // ç can't follow back vowels
            if *current == "ç" && (*prev == "a" || *prev == "o" || *prev == "u") {
                return false;
            }
        }
    }

    true
}

/// Validate Japanese phonotactic constraints  
fn validate_japanese_phonotactics(phonemes: &[Phoneme]) -> bool {
    for i in 0..phonemes.len() {
        let current = &phonemes[i].effective_symbol();

        // Japanese syllable structure: (C)V(N)
        if i > 0 {
            let prev = &phonemes[i - 1].effective_symbol();

            // Only /n/ can close syllables
            if is_consonant(prev) && is_consonant(current) && *prev != "n" && *prev != "ɴ" {
                return false;
            }
        }
    }

    true
}

/// Validate syllable structure
fn validate_syllable_structure(phonemes: &[Phoneme], _language: LanguageCode) -> bool {
    // Check that syllables have at least one vowel
    let mut current_syllable_has_vowel = false;

    for phoneme in phonemes {
        if phoneme.is_syllable_boundary {
            if !current_syllable_has_vowel {
                return false; // Syllable without vowel
            }
            current_syllable_has_vowel = false;
        } else if is_vowel(phoneme.effective_symbol()) {
            current_syllable_has_vowel = true;
        }
    }

    true
}

/// Check if a phoneme is a consonant
pub fn is_consonant(phoneme: &str) -> bool {
    !is_vowel(phoneme) && phoneme != " " && phoneme != "."
}

/// Check if a phoneme is a vowel
pub fn is_vowel(phoneme: &str) -> bool {
    let vowels = [
        "a", "e", "i", "o", "u", "ə", "ɛ", "ɪ", "ɔ", "ʊ", "ʌ", "æ", "ɜ", "ɑ", "iː", "uː", "eɪ",
        "aɪ", "ɔɪ", "aʊ", "oʊ", "ɜːr", "ɑːr", "ɔːr",
    ];
    vowels.contains(&phoneme)
}

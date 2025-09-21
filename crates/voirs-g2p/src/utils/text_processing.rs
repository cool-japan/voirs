//! Text preprocessing and postprocessing utilities for G2P conversion.

use crate::preprocessing::{PreprocessingConfig, TextPreprocessor};
use crate::{LanguageCode, Phoneme, Result};

/// Text preprocessing utilities
pub fn preprocess_text(text: &str, language: LanguageCode) -> Result<String> {
    let preprocessor = TextPreprocessor::new(language);
    preprocessor.preprocess(text)
}

/// Text preprocessing with custom configuration
pub fn preprocess_text_with_config(
    text: &str,
    language: LanguageCode,
    config: PreprocessingConfig,
) -> Result<String> {
    let preprocessor = TextPreprocessor::with_config(language, config);
    preprocessor.preprocess(text)
}

/// Simple text preprocessing (legacy function for backward compatibility)
pub fn preprocess_text_simple(text: &str, language: LanguageCode) -> String {
    match language {
        LanguageCode::EnUs | LanguageCode::EnGb => {
            // Basic English preprocessing
            text.to_lowercase()
                .chars()
                .filter(|c| c.is_alphabetic() || c.is_whitespace())
                .collect()
        }
        _ => {
            // Generic preprocessing
            text.to_lowercase()
        }
    }
}

/// Phoneme post-processing utilities
pub fn postprocess_phonemes(mut phonemes: Vec<Phoneme>, language: LanguageCode) -> Vec<Phoneme> {
    if phonemes.is_empty() {
        return phonemes;
    }

    // Language-specific post-processing
    match language {
        LanguageCode::EnUs | LanguageCode::EnGb => postprocess_english(&mut phonemes),
        LanguageCode::De => postprocess_german(&mut phonemes),
        LanguageCode::Fr => postprocess_french(&mut phonemes),
        LanguageCode::Es => postprocess_spanish(&mut phonemes),
        LanguageCode::Ja => postprocess_japanese(&mut phonemes),
        _ => {} // No specific post-processing for other languages
    }

    phonemes
}

/// Assign stress patterns based on language-specific rules
fn assign_stress_patterns(phonemes: &mut [Phoneme], language: LanguageCode) {
    match language {
        LanguageCode::EnUs | LanguageCode::EnGb => assign_english_stress(phonemes),
        LanguageCode::De => assign_german_stress(phonemes),
        LanguageCode::Fr => assign_french_stress(phonemes),
        LanguageCode::Es => assign_spanish_stress(phonemes),
        LanguageCode::Ja => assign_japanese_stress(phonemes),
        _ => {} // No stress assignment for other languages
    }
}

/// Assign English stress patterns
fn assign_english_stress(phonemes: &mut [Phoneme]) {
    if phonemes.is_empty() {
        return;
    }

    // Detect syllable boundaries first
    let syllable_boundaries = detect_syllable_boundaries(phonemes, LanguageCode::EnUs);

    // Process each syllable for stress assignment
    let mut current_syllable_start = 0;
    for &boundary in &syllable_boundaries {
        assign_stress_english_word(&mut phonemes[current_syllable_start..boundary]);
        current_syllable_start = boundary;
    }

    // Handle the last syllable
    if current_syllable_start < phonemes.len() {
        assign_stress_english_word(&mut phonemes[current_syllable_start..]);
    }
}

/// Assign stress for English words
fn assign_stress_english_word(phonemes: &mut [Phoneme]) {
    if phonemes.is_empty() {
        return;
    }

    // Simple English stress rules
    let syllable_count = count_syllables(phonemes);

    if syllable_count == 1 {
        // Monosyllabic words: primary stress on the only syllable
        if let Some(vowel_idx) = find_first_vowel(phonemes) {
            phonemes[vowel_idx].stress = 1; // Primary stress
        }
    } else if syllable_count == 2 {
        // Disyllabic words: stress on first syllable for most nouns/adjectives
        if let Some(vowel_idx) = find_first_vowel(phonemes) {
            phonemes[vowel_idx].stress = 1; // Primary stress
        }
        // Secondary stress on second syllable if it exists
        if let Some(second_vowel_idx) = find_nth_vowel(phonemes, 2) {
            phonemes[second_vowel_idx].stress = 2; // Secondary stress
        }
    } else {
        // Multisyllabic words: antepenultimate stress rule
        let vowel_positions: Vec<usize> = phonemes
            .iter()
            .enumerate()
            .filter(|(_, p)| is_vowel(&p.symbol))
            .map(|(i, _)| i)
            .collect();

        if vowel_positions.len() >= 3 {
            // Primary stress on antepenultimate (third from end)
            let antepenult_idx = vowel_positions[vowel_positions.len() - 3];
            phonemes[antepenult_idx].stress = 1;

            // Secondary stress on first syllable if long enough
            if vowel_positions.len() >= 4 {
                phonemes[vowel_positions[0]].stress = 2;
            }
        } else if vowel_positions.len() == 2 {
            // Two syllables: primary on first
            phonemes[vowel_positions[0]].stress = 1;
        }
    }
}

/// Assign German stress patterns
fn assign_german_stress(phonemes: &mut [Phoneme]) {
    // German typically stresses the first syllable
    if let Some(vowel_idx) = find_first_vowel(phonemes) {
        phonemes[vowel_idx].stress = 1; // Primary stress
    }
}

/// Assign French stress patterns
fn assign_french_stress(phonemes: &mut [Phoneme]) {
    // French typically stresses the final syllable
    if let Some(vowel_idx) = find_last_vowel(phonemes) {
        phonemes[vowel_idx].stress = 1; // Primary stress
    }
}

/// Assign Spanish stress patterns
fn assign_spanish_stress(phonemes: &mut [Phoneme]) {
    // Spanish stress rules based on word endings
    let vowel_positions: Vec<usize> = phonemes
        .iter()
        .enumerate()
        .filter(|(_, p)| is_vowel(&p.symbol))
        .map(|(i, _)| i)
        .collect();

    if vowel_positions.len() >= 2 {
        // Default: penultimate stress
        let penult_idx = vowel_positions[vowel_positions.len() - 2];
        phonemes[penult_idx].stress = 1;
    } else if vowel_positions.len() == 1 {
        // Single syllable: stressed
        phonemes[vowel_positions[0]].stress = 1;
    }
}

/// Assign Japanese stress patterns (minimal stress language)
fn assign_japanese_stress(_phonemes: &mut [Phoneme]) {
    // Japanese has minimal stress distinctions, mostly pitch accent
    // For simplicity, we don't assign stress levels
}

/// Detect syllable boundaries
pub fn detect_syllable_boundaries(phonemes: &[Phoneme], language: LanguageCode) -> Vec<usize> {
    match language {
        LanguageCode::EnUs | LanguageCode::EnGb => detect_english_syllables(phonemes),
        LanguageCode::De => detect_german_syllables(phonemes),
        LanguageCode::Ja => detect_japanese_syllables(phonemes),
        _ => {
            // Generic syllable detection
            detect_english_syllables(phonemes)
        }
    }
}

/// Detect English syllable boundaries
fn detect_english_syllables(phonemes: &[Phoneme]) -> Vec<usize> {
    let mut boundaries = Vec::new();
    let mut i = 0;

    while i < phonemes.len() {
        // Find next vowel (syllable nucleus)
        while i < phonemes.len() && !is_vowel(&phonemes[i].symbol) {
            i += 1;
        }

        if i < phonemes.len() {
            // Found vowel, look for syllable boundary
            i += 1;
            while i < phonemes.len() && !is_vowel(&phonemes[i].symbol) {
                i += 1;
            }
            if i < phonemes.len() {
                boundaries.push(i);
            }
        }
    }

    boundaries
}

/// Detect German syllable boundaries
fn detect_german_syllables(phonemes: &[Phoneme]) -> Vec<usize> {
    // German syllable structure similar to English
    detect_english_syllables(phonemes)
}

/// Detect Japanese syllable boundaries (mora-based)
fn detect_japanese_syllables(phonemes: &[Phoneme]) -> Vec<usize> {
    let mut boundaries = Vec::new();

    // Japanese uses mora-based timing
    for (i, phoneme) in phonemes.iter().enumerate() {
        if is_vowel(&phoneme.symbol) && i > 0 {
            boundaries.push(i);
        }
    }

    boundaries
}

// Helper functions

/// Count syllables in phoneme sequence
fn count_syllables(phonemes: &[Phoneme]) -> usize {
    phonemes.iter().filter(|p| is_vowel(&p.symbol)).count()
}

/// Find first vowel index
fn find_first_vowel(phonemes: &[Phoneme]) -> Option<usize> {
    phonemes.iter().position(|p| is_vowel(&p.symbol))
}

/// Find last vowel index
fn find_last_vowel(phonemes: &[Phoneme]) -> Option<usize> {
    phonemes.iter().rposition(|p| is_vowel(&p.symbol))
}

/// Find nth vowel (1-indexed)
fn find_nth_vowel(phonemes: &[Phoneme], n: usize) -> Option<usize> {
    let mut count = 0;
    for (i, phoneme) in phonemes.iter().enumerate() {
        if is_vowel(&phoneme.symbol) {
            count += 1;
            if count == n {
                return Some(i);
            }
        }
    }
    None
}

/// Check if a phoneme symbol represents a vowel
fn is_vowel(symbol: &str) -> bool {
    matches!(
        symbol,
        "a" | "e"
            | "i"
            | "o"
            | "u"
            | "A"
            | "E"
            | "I"
            | "O"
            | "U"
            | "æ"
            | "ɛ"
            | "ɪ"
            | "ɔ"
            | "ʊ"
            | "ə"
            | "ʌ"
            | "ɝ"
            | "ɚ"
            | "ɑ"
            | "ɒ"
            | "y"
            | "ø"
            | "œ"
            | "ɶ"
            | "ɤ"
            | "ɯ"
            | "ɨ"
            | "ɘ"
            | "ɵ"
            | "ɞ"
            | "ɐ"
            | "ɜ"
            | "ɢ"
            | "ɾ"
            | "ɣ"
            | "ɭ"
            | "ɮ"
            | "ɰ"
            | "ɱ"
            | "ɲ"
            | "ɳ"
            | "ɴ"
            | "ɸ"
            | "ɹ"
            | "ɺ"
            | "ɻ"
            | "ɽ"
            | "ɿ"
            | "ʀ"
            | "ʁ"
            | "ʂ"
            | "ʃ"
            | "ʈ"
            | "ʉ"
            | "ʋ"
            | "ʍ"
            | "ʎ"
            | "ʏ"
            | "ʐ"
            | "ʑ"
            | "ʒ"
            | "ʔ"
            | "ʕ"
            | "ʖ"
            | "ʗ"
            | "ʘ"
            | "ʙ"
            | "ʚ"
            | "ʛ"
            | "ʜ"
            | "ʝ"
            | "ʞ"
            | "ʟ"
            | "ʠ"
            | "ʡ"
            | "ʢ"
            | "ʣ"
            | "ʤ"
            | "ʥ"
            | "ʦ"
            | "ʧ"
            | "ʨ"
            | "ʩ"
            | "ʪ"
            | "ʫ"
            | "ʬ"
            | "ʭ"
            | "ʮ"
            | "ʯ"
            | "a:"
            | "e:"
            | "i:"
            | "o:"
            | "u:"
    )
}

// Language-specific post-processing functions

/// Post-process English phonemes
fn postprocess_english(phonemes: &mut [Phoneme]) {
    assign_stress_patterns(phonemes, LanguageCode::EnUs);
    predict_phoneme_durations(phonemes, LanguageCode::EnUs);
    assign_syllable_positions(phonemes);
}

/// Post-process German phonemes
fn postprocess_german(phonemes: &mut [Phoneme]) {
    assign_stress_patterns(phonemes, LanguageCode::De);
    predict_phoneme_durations(phonemes, LanguageCode::De);
    assign_syllable_positions(phonemes);
}

/// Post-process French phonemes
fn postprocess_french(phonemes: &mut [Phoneme]) {
    assign_stress_patterns(phonemes, LanguageCode::Fr);
    predict_phoneme_durations(phonemes, LanguageCode::Fr);
    assign_syllable_positions(phonemes);
}

/// Post-process Spanish phonemes
fn postprocess_spanish(phonemes: &mut [Phoneme]) {
    assign_stress_patterns(phonemes, LanguageCode::Es);
    predict_phoneme_durations(phonemes, LanguageCode::Es);
    assign_syllable_positions(phonemes);
}

/// Post-process Japanese phonemes
fn postprocess_japanese(phonemes: &mut [Phoneme]) {
    assign_stress_patterns(phonemes, LanguageCode::Ja);
    predict_phoneme_durations(phonemes, LanguageCode::Ja);
    assign_syllable_positions(phonemes);
}

/// Predict phoneme durations based on language and phoneme type
fn predict_phoneme_durations(phonemes: &mut [Phoneme], language: LanguageCode) {
    for phoneme in phonemes.iter_mut() {
        phoneme.duration_ms = Some(predict_duration_for_phoneme(&phoneme.symbol, language));
    }
}

/// Predict duration for a phoneme
fn predict_duration_for_phoneme(symbol: &str, language: LanguageCode) -> f32 {
    let base_duration = if is_vowel(symbol) {
        // Vowels are generally longer
        match language {
            LanguageCode::EnUs | LanguageCode::EnGb => 0.12, // English vowels
            LanguageCode::De => 0.10,                        // German vowels
            LanguageCode::Fr => 0.08,                        // French vowels
            LanguageCode::Es => 0.09,                        // Spanish vowels
            LanguageCode::Ja => 0.07,                        // Japanese vowels (shorter)
            _ => 0.10,                                       // Default vowel duration
        }
    } else {
        // Consonants are generally shorter
        match symbol {
            // Fricatives
            "s" | "z" | "f" | "v" | "θ" | "ð" | "ʃ" | "ʒ" | "h" => 0.08,
            // Stops
            "p" | "b" | "t" | "d" | "k" | "g" | "ʔ" => 0.06,
            // Nasals
            "m" | "n" | "ŋ" | "ɲ" | "ɳ" => 0.07,
            // Liquids
            "l" | "r" | "ɹ" | "ɾ" | "ɭ" => 0.05,
            // Approximants
            "w" | "j" | "ɥ" => 0.04,
            // Default consonant
            _ => 0.06,
        }
    };

    // Language-specific adjustments
    match language {
        LanguageCode::Ja => base_duration * 0.8, // Japanese tends to be faster
        LanguageCode::De => base_duration * 1.1, // German tends to be more deliberate
        LanguageCode::Fr => base_duration * 0.9, // French tends to be faster
        _ => base_duration,
    }
}

/// Assign syllable positions based on syllable boundaries
fn assign_syllable_positions(phonemes: &mut [Phoneme]) {
    if phonemes.is_empty() {
        return;
    }

    let boundaries = detect_syllable_boundaries(phonemes, LanguageCode::EnUs); // Use English as default
    let mut current_syllable_start = 0;

    for &boundary in &boundaries {
        assign_positions_within_syllable(&mut phonemes[current_syllable_start..boundary]);
        current_syllable_start = boundary;
    }

    // Handle the last syllable
    if current_syllable_start < phonemes.len() {
        assign_positions_within_syllable(&mut phonemes[current_syllable_start..]);
    }
}

/// Assign positions within a syllable
fn assign_positions_within_syllable(phonemes: &mut [Phoneme]) {
    use crate::SyllablePosition;

    if phonemes.is_empty() {
        return;
    }

    // Find the nucleus (vowel)
    let nucleus_idx = phonemes.iter().position(|p| is_vowel(&p.symbol));

    match nucleus_idx {
        Some(idx) => {
            // Assign onset
            for phoneme in phonemes.iter_mut().take(idx) {
                phoneme.syllable_position = SyllablePosition::Onset;
            }

            // Assign nucleus
            phonemes[idx].syllable_position = SyllablePosition::Nucleus;

            // Assign coda
            for phoneme in phonemes.iter_mut().skip(idx + 1) {
                phoneme.syllable_position = SyllablePosition::Coda;
            }
        }
        None => {
            // No nucleus found, treat all as onset
            for phoneme in phonemes.iter_mut() {
                phoneme.syllable_position = SyllablePosition::Onset;
            }
        }
    }
}

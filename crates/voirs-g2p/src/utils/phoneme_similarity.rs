//! Phoneme similarity and distance metrics for G2P quality assessment.
//!
//! This module provides algorithms for measuring phonetic similarity between
//! phoneme sequences, which is useful for:
//! - Pronunciation variant comparison
//! - G2P model evaluation
//! - Error analysis and correction suggestions
//! - Phonetic search and fuzzy matching

use crate::Phoneme;
use std::cmp::{max, min};

/// Calculate Levenshtein distance between two phoneme sequences
///
/// The Levenshtein distance measures the minimum number of single-phoneme edits
/// (insertions, deletions, or substitutions) required to change one sequence into another.
///
/// # Examples
///
/// ```
/// use voirs_g2p::{Phoneme, utils::phoneme_similarity::phoneme_levenshtein_distance};
///
/// let seq1 = vec![Phoneme::new("k"), Phoneme::new("æ"), Phoneme::new("t")];
/// let seq2 = vec![Phoneme::new("k"), Phoneme::new("ɑ"), Phoneme::new("t")];
///
/// // One substitution: æ → ɑ
/// let distance = phoneme_levenshtein_distance(&seq1, &seq2);
/// assert_eq!(distance, 1);
/// ```
pub fn phoneme_levenshtein_distance(seq1: &[Phoneme], seq2: &[Phoneme]) -> usize {
    let len1 = seq1.len();
    let len2 = seq2.len();

    if len1 == 0 {
        return len2;
    }
    if len2 == 0 {
        return len1;
    }

    // Create distance matrix
    let mut matrix = vec![vec![0usize; len2 + 1]; len1 + 1];

    // Initialize first row and column
    for (i, row) in matrix.iter_mut().enumerate().take(len1 + 1) {
        row[0] = i;
    }
    for (j, cell) in matrix[0].iter_mut().enumerate().take(len2 + 1) {
        *cell = j;
    }

    // Fill matrix using dynamic programming
    for i in 1..=len1 {
        for j in 1..=len2 {
            let cost = if seq1[i - 1].symbol == seq2[j - 1].symbol {
                0
            } else {
                1
            };

            matrix[i][j] = min(
                min(
                    matrix[i - 1][j] + 1, // deletion
                    matrix[i][j - 1] + 1, // insertion
                ),
                matrix[i - 1][j - 1] + cost, // substitution
            );
        }
    }

    matrix[len1][len2]
}

/// Calculate phonetic feature-based similarity between two phonemes
///
/// This function compares phonemes based on their distinctive features
/// (voicing, place of articulation, manner of articulation) rather than
/// treating them as arbitrary symbols.
///
/// Returns a similarity score between 0.0 (completely different) and 1.0 (identical).
///
/// # Examples
///
/// ```
/// use voirs_g2p::{Phoneme, utils::phoneme_similarity::phoneme_feature_similarity};
///
/// let p1 = Phoneme::new("p");  // voiceless bilabial plosive
/// let p2 = Phoneme::new("b");  // voiced bilabial plosive
/// let p3 = Phoneme::new("k");  // voiceless velar plosive
///
/// let sim_pb = phoneme_feature_similarity(&p1, &p2);  // Same place, manner; diff voicing
/// let sim_pk = phoneme_feature_similarity(&p1, &p3);  // Same voicing, manner; diff place
///
/// // p and b are more similar than p and k
/// assert!(sim_pb > sim_pk);
/// ```
pub fn phoneme_feature_similarity(p1: &Phoneme, p2: &Phoneme) -> f32 {
    if p1.symbol == p2.symbol {
        return 1.0;
    }

    let features1 = extract_phonetic_features(&p1.symbol);
    let features2 = extract_phonetic_features(&p2.symbol);

    let mut matching_features = 0;
    let mut total_features = 0;

    // Compare voicing
    if features1.voicing == features2.voicing {
        matching_features += 1;
    }
    total_features += 1;

    // Compare place of articulation
    if features1.place == features2.place {
        matching_features += 2; // Place is more important
    }
    total_features += 2;

    // Compare manner of articulation
    if features1.manner == features2.manner {
        matching_features += 2; // Manner is more important
    }
    total_features += 2;

    // Compare vowel/consonant class
    if features1.is_vowel == features2.is_vowel {
        matching_features += 1;
    }
    total_features += 1;

    matching_features as f32 / total_features as f32
}

/// Calculate weighted phoneme sequence similarity
///
/// This is a more sophisticated version of Levenshtein distance that uses
/// feature-based similarity scores instead of binary equal/not-equal comparisons.
///
/// Returns a normalized similarity score between 0.0 (completely different)
/// and 1.0 (identical).
///
/// # Examples
///
/// ```
/// use voirs_g2p::{Phoneme, utils::phoneme_similarity::phoneme_sequence_similarity};
///
/// let seq1 = vec![Phoneme::new("p"), Phoneme::new("ɑ"), Phoneme::new("t")];
/// let seq2 = vec![Phoneme::new("b"), Phoneme::new("ɑ"), Phoneme::new("t")];
///
/// let similarity = phoneme_sequence_similarity(&seq1, &seq2);
/// // Should be high because only p→b differs, and they're similar sounds
/// assert!(similarity > 0.8);
/// ```
pub fn phoneme_sequence_similarity(seq1: &[Phoneme], seq2: &[Phoneme]) -> f32 {
    let len1 = seq1.len();
    let len2 = seq2.len();

    if len1 == 0 && len2 == 0 {
        return 1.0;
    }
    if len1 == 0 || len2 == 0 {
        return 0.0;
    }

    // Use feature-weighted edit distance
    let mut matrix = vec![vec![0.0f32; len2 + 1]; len1 + 1];

    // Initialize first row and column
    for (i, row) in matrix.iter_mut().enumerate().take(len1 + 1) {
        row[0] = i as f32;
    }
    for (j, cell) in matrix[0].iter_mut().enumerate().take(len2 + 1) {
        *cell = j as f32;
    }

    // Fill matrix with feature-weighted costs
    for i in 1..=len1 {
        for j in 1..=len2 {
            let similarity = phoneme_feature_similarity(&seq1[i - 1], &seq2[j - 1]);
            let substitution_cost = 1.0 - similarity;

            matrix[i][j] = matrix[i - 1][j - 1] + substitution_cost; // substitution
            matrix[i][j] = matrix[i][j].min(matrix[i - 1][j] + 1.0); // deletion
            matrix[i][j] = matrix[i][j].min(matrix[i][j - 1] + 1.0); // insertion
        }
    }

    let max_len = max(len1, len2) as f32;
    let distance = matrix[len1][len2];

    // Convert distance to similarity (0.0 = different, 1.0 = identical)
    1.0 - (distance / max_len).min(1.0)
}

/// Phonetic features for a phoneme
#[derive(Debug, Clone, PartialEq)]
struct PhoneticFeatures {
    is_vowel: bool,
    voicing: Voicing,
    place: PlaceOfArticulation,
    manner: MannerOfArticulation,
}

/// Voicing feature
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Voicing {
    Voiced,
    Voiceless,
    Unknown,
}

/// Place of articulation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlaceOfArticulation {
    Bilabial,
    Labiodental,
    Dental,
    Alveolar,
    Postalveolar,
    Palatal,
    Velar,
    Glottal,
    Front,   // vowels
    Central, // vowels
    Back,    // vowels
    Unknown,
}

/// Manner of articulation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MannerOfArticulation {
    Plosive,
    Fricative,
    Affricate,
    Nasal,
    Approximant,
    Lateral,
    Trill,
    Tap,
    Close,    // vowels
    CloseMid, // vowels
    OpenMid,  // vowels
    Open,     // vowels
    Unknown,
}

/// Extract phonetic features from IPA symbol
fn extract_phonetic_features(symbol: &str) -> PhoneticFeatures {
    match symbol {
        // Voiceless plosives
        "p" => PhoneticFeatures {
            is_vowel: false,
            voicing: Voicing::Voiceless,
            place: PlaceOfArticulation::Bilabial,
            manner: MannerOfArticulation::Plosive,
        },
        "t" => PhoneticFeatures {
            is_vowel: false,
            voicing: Voicing::Voiceless,
            place: PlaceOfArticulation::Alveolar,
            manner: MannerOfArticulation::Plosive,
        },
        "k" => PhoneticFeatures {
            is_vowel: false,
            voicing: Voicing::Voiceless,
            place: PlaceOfArticulation::Velar,
            manner: MannerOfArticulation::Plosive,
        },

        // Voiced plosives
        "b" => PhoneticFeatures {
            is_vowel: false,
            voicing: Voicing::Voiced,
            place: PlaceOfArticulation::Bilabial,
            manner: MannerOfArticulation::Plosive,
        },
        "d" => PhoneticFeatures {
            is_vowel: false,
            voicing: Voicing::Voiced,
            place: PlaceOfArticulation::Alveolar,
            manner: MannerOfArticulation::Plosive,
        },
        "g" => PhoneticFeatures {
            is_vowel: false,
            voicing: Voicing::Voiced,
            place: PlaceOfArticulation::Velar,
            manner: MannerOfArticulation::Plosive,
        },

        // Voiceless fricatives
        "f" => PhoneticFeatures {
            is_vowel: false,
            voicing: Voicing::Voiceless,
            place: PlaceOfArticulation::Labiodental,
            manner: MannerOfArticulation::Fricative,
        },
        "θ" => PhoneticFeatures {
            is_vowel: false,
            voicing: Voicing::Voiceless,
            place: PlaceOfArticulation::Dental,
            manner: MannerOfArticulation::Fricative,
        },
        "s" => PhoneticFeatures {
            is_vowel: false,
            voicing: Voicing::Voiceless,
            place: PlaceOfArticulation::Alveolar,
            manner: MannerOfArticulation::Fricative,
        },
        "ʃ" => PhoneticFeatures {
            is_vowel: false,
            voicing: Voicing::Voiceless,
            place: PlaceOfArticulation::Postalveolar,
            manner: MannerOfArticulation::Fricative,
        },
        "h" => PhoneticFeatures {
            is_vowel: false,
            voicing: Voicing::Voiceless,
            place: PlaceOfArticulation::Glottal,
            manner: MannerOfArticulation::Fricative,
        },

        // Voiced fricatives
        "v" => PhoneticFeatures {
            is_vowel: false,
            voicing: Voicing::Voiced,
            place: PlaceOfArticulation::Labiodental,
            manner: MannerOfArticulation::Fricative,
        },
        "ð" => PhoneticFeatures {
            is_vowel: false,
            voicing: Voicing::Voiced,
            place: PlaceOfArticulation::Dental,
            manner: MannerOfArticulation::Fricative,
        },
        "z" => PhoneticFeatures {
            is_vowel: false,
            voicing: Voicing::Voiced,
            place: PlaceOfArticulation::Alveolar,
            manner: MannerOfArticulation::Fricative,
        },
        "ʒ" => PhoneticFeatures {
            is_vowel: false,
            voicing: Voicing::Voiced,
            place: PlaceOfArticulation::Postalveolar,
            manner: MannerOfArticulation::Fricative,
        },

        // Nasals
        "m" => PhoneticFeatures {
            is_vowel: false,
            voicing: Voicing::Voiced,
            place: PlaceOfArticulation::Bilabial,
            manner: MannerOfArticulation::Nasal,
        },
        "n" => PhoneticFeatures {
            is_vowel: false,
            voicing: Voicing::Voiced,
            place: PlaceOfArticulation::Alveolar,
            manner: MannerOfArticulation::Nasal,
        },
        "ŋ" => PhoneticFeatures {
            is_vowel: false,
            voicing: Voicing::Voiced,
            place: PlaceOfArticulation::Velar,
            manner: MannerOfArticulation::Nasal,
        },

        // Approximants
        "l" => PhoneticFeatures {
            is_vowel: false,
            voicing: Voicing::Voiced,
            place: PlaceOfArticulation::Alveolar,
            manner: MannerOfArticulation::Lateral,
        },
        "ɹ" | "r" => PhoneticFeatures {
            is_vowel: false,
            voicing: Voicing::Voiced,
            place: PlaceOfArticulation::Alveolar,
            manner: MannerOfArticulation::Approximant,
        },
        "j" => PhoneticFeatures {
            is_vowel: false,
            voicing: Voicing::Voiced,
            place: PlaceOfArticulation::Palatal,
            manner: MannerOfArticulation::Approximant,
        },
        "w" => PhoneticFeatures {
            is_vowel: false,
            voicing: Voicing::Voiced,
            place: PlaceOfArticulation::Velar,
            manner: MannerOfArticulation::Approximant,
        },

        // Affricates
        "tʃ" => PhoneticFeatures {
            is_vowel: false,
            voicing: Voicing::Voiceless,
            place: PlaceOfArticulation::Postalveolar,
            manner: MannerOfArticulation::Affricate,
        },
        "dʒ" => PhoneticFeatures {
            is_vowel: false,
            voicing: Voicing::Voiced,
            place: PlaceOfArticulation::Postalveolar,
            manner: MannerOfArticulation::Affricate,
        },

        // Close vowels
        "i" | "iː" => PhoneticFeatures {
            is_vowel: true,
            voicing: Voicing::Voiced,
            place: PlaceOfArticulation::Front,
            manner: MannerOfArticulation::Close,
        },
        "u" | "uː" => PhoneticFeatures {
            is_vowel: true,
            voicing: Voicing::Voiced,
            place: PlaceOfArticulation::Back,
            manner: MannerOfArticulation::Close,
        },

        // Mid vowels
        "e" | "eɪ" => PhoneticFeatures {
            is_vowel: true,
            voicing: Voicing::Voiced,
            place: PlaceOfArticulation::Front,
            manner: MannerOfArticulation::CloseMid,
        },
        "ɛ" => PhoneticFeatures {
            is_vowel: true,
            voicing: Voicing::Voiced,
            place: PlaceOfArticulation::Front,
            manner: MannerOfArticulation::OpenMid,
        },
        "ə" | "ɚ" => PhoneticFeatures {
            is_vowel: true,
            voicing: Voicing::Voiced,
            place: PlaceOfArticulation::Central,
            manner: MannerOfArticulation::OpenMid,
        },
        "o" | "oʊ" => PhoneticFeatures {
            is_vowel: true,
            voicing: Voicing::Voiced,
            place: PlaceOfArticulation::Back,
            manner: MannerOfArticulation::CloseMid,
        },
        "ɔ" => PhoneticFeatures {
            is_vowel: true,
            voicing: Voicing::Voiced,
            place: PlaceOfArticulation::Back,
            manner: MannerOfArticulation::OpenMid,
        },

        // Open vowels
        "æ" => PhoneticFeatures {
            is_vowel: true,
            voicing: Voicing::Voiced,
            place: PlaceOfArticulation::Front,
            manner: MannerOfArticulation::Open,
        },
        "a" | "ɑ" | "ɑː" => PhoneticFeatures {
            is_vowel: true,
            voicing: Voicing::Voiced,
            place: PlaceOfArticulation::Back,
            manner: MannerOfArticulation::Open,
        },

        // Other vowels
        "ɪ" => PhoneticFeatures {
            is_vowel: true,
            voicing: Voicing::Voiced,
            place: PlaceOfArticulation::Front,
            manner: MannerOfArticulation::Close,
        },
        "ʊ" => PhoneticFeatures {
            is_vowel: true,
            voicing: Voicing::Voiced,
            place: PlaceOfArticulation::Back,
            manner: MannerOfArticulation::Close,
        },
        "ʌ" | "ɜ" => PhoneticFeatures {
            is_vowel: true,
            voicing: Voicing::Voiced,
            place: PlaceOfArticulation::Central,
            manner: MannerOfArticulation::OpenMid,
        },

        // Default for unknown phonemes
        _ => PhoneticFeatures {
            is_vowel: false,
            voicing: Voicing::Unknown,
            place: PlaceOfArticulation::Unknown,
            manner: MannerOfArticulation::Unknown,
        },
    }
}

/// Calculate phoneme error rate (PER) between reference and hypothesis sequences
///
/// PER is defined as: (insertions + deletions + substitutions) / reference_length
///
/// This is a standard metric for evaluating G2P systems.
///
/// # Examples
///
/// ```
/// use voirs_g2p::{Phoneme, utils::phoneme_similarity::calculate_phoneme_error_rate};
///
/// let reference = vec![Phoneme::new("k"), Phoneme::new("æ"), Phoneme::new("t")];
/// let hypothesis = vec![Phoneme::new("k"), Phoneme::new("ɑ"), Phoneme::new("t")];
///
/// let per = calculate_phoneme_error_rate(&reference, &hypothesis);
/// assert!((per - 0.333).abs() < 0.01); // 1 error / 3 phonemes ≈ 33.3%
/// ```
pub fn calculate_phoneme_error_rate(reference: &[Phoneme], hypothesis: &[Phoneme]) -> f32 {
    if reference.is_empty() {
        return if hypothesis.is_empty() { 0.0 } else { 1.0 };
    }

    let distance = phoneme_levenshtein_distance(reference, hypothesis);
    distance as f32 / reference.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_levenshtein_distance_identical() {
        let seq1 = vec![Phoneme::new("k"), Phoneme::new("æ"), Phoneme::new("t")];
        let seq2 = vec![Phoneme::new("k"), Phoneme::new("æ"), Phoneme::new("t")];
        assert_eq!(phoneme_levenshtein_distance(&seq1, &seq2), 0);
    }

    #[test]
    fn test_levenshtein_distance_one_substitution() {
        let seq1 = vec![Phoneme::new("k"), Phoneme::new("æ"), Phoneme::new("t")];
        let seq2 = vec![Phoneme::new("k"), Phoneme::new("ɑ"), Phoneme::new("t")];
        assert_eq!(phoneme_levenshtein_distance(&seq1, &seq2), 1);
    }

    #[test]
    fn test_levenshtein_distance_insertion() {
        let seq1 = vec![Phoneme::new("k"), Phoneme::new("æ"), Phoneme::new("t")];
        let seq2 = vec![
            Phoneme::new("k"),
            Phoneme::new("æ"),
            Phoneme::new("t"),
            Phoneme::new("s"),
        ];
        assert_eq!(phoneme_levenshtein_distance(&seq1, &seq2), 1);
    }

    #[test]
    fn test_levenshtein_distance_deletion() {
        let seq1 = vec![Phoneme::new("k"), Phoneme::new("æ"), Phoneme::new("t")];
        let seq2 = vec![Phoneme::new("k"), Phoneme::new("t")];
        assert_eq!(phoneme_levenshtein_distance(&seq1, &seq2), 1);
    }

    #[test]
    fn test_levenshtein_distance_empty() {
        let seq1: Vec<Phoneme> = vec![];
        let seq2 = vec![Phoneme::new("k"), Phoneme::new("æ"), Phoneme::new("t")];
        assert_eq!(phoneme_levenshtein_distance(&seq1, &seq2), 3);
        assert_eq!(phoneme_levenshtein_distance(&seq2, &seq1), 3);
    }

    #[test]
    fn test_feature_similarity_identical() {
        let p1 = Phoneme::new("p");
        let p2 = Phoneme::new("p");
        assert_eq!(phoneme_feature_similarity(&p1, &p2), 1.0);
    }

    #[test]
    fn test_feature_similarity_voicing_difference() {
        let p1 = Phoneme::new("p"); // voiceless
        let p2 = Phoneme::new("b"); // voiced
        let similarity = phoneme_feature_similarity(&p1, &p2);
        // Same place and manner, different voicing
        assert!(similarity > 0.5); // Should be fairly similar
        assert!(similarity < 1.0);
    }

    #[test]
    fn test_feature_similarity_different_phones() {
        let p1 = Phoneme::new("p"); // voiceless bilabial plosive
        let p2 = Phoneme::new("i"); // close front vowel
        let similarity = phoneme_feature_similarity(&p1, &p2);
        // Consonant vs vowel - very different
        assert!(similarity < 0.3);
    }

    #[test]
    fn test_sequence_similarity_identical() {
        let seq1 = vec![Phoneme::new("k"), Phoneme::new("æ"), Phoneme::new("t")];
        let seq2 = vec![Phoneme::new("k"), Phoneme::new("æ"), Phoneme::new("t")];
        let similarity = phoneme_sequence_similarity(&seq1, &seq2);
        assert!((similarity - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_sequence_similarity_similar_phones() {
        let seq1 = vec![Phoneme::new("p"), Phoneme::new("ɑ"), Phoneme::new("t")];
        let seq2 = vec![Phoneme::new("b"), Phoneme::new("ɑ"), Phoneme::new("t")];
        let similarity = phoneme_sequence_similarity(&seq1, &seq2);
        // p and b are similar, so should have high similarity
        assert!(similarity > 0.8);
    }

    #[test]
    fn test_sequence_similarity_empty() {
        let seq1: Vec<Phoneme> = vec![];
        let seq2: Vec<Phoneme> = vec![];
        assert_eq!(phoneme_sequence_similarity(&seq1, &seq2), 1.0);

        let seq3 = vec![Phoneme::new("k")];
        assert!(phoneme_sequence_similarity(&seq1, &seq3) < 0.1);
    }

    #[test]
    fn test_phoneme_error_rate() {
        let reference = vec![Phoneme::new("k"), Phoneme::new("æ"), Phoneme::new("t")];
        let hypothesis = vec![Phoneme::new("k"), Phoneme::new("ɑ"), Phoneme::new("t")];

        let per = calculate_phoneme_error_rate(&reference, &hypothesis);
        assert!((per - 0.333).abs() < 0.01);
    }

    #[test]
    fn test_phoneme_error_rate_perfect() {
        let reference = vec![Phoneme::new("k"), Phoneme::new("æ"), Phoneme::new("t")];
        let hypothesis = vec![Phoneme::new("k"), Phoneme::new("æ"), Phoneme::new("t")];

        let per = calculate_phoneme_error_rate(&reference, &hypothesis);
        assert_eq!(per, 0.0);
    }

    #[test]
    fn test_extract_features() {
        let features_p = extract_phonetic_features("p");
        assert_eq!(features_p.voicing, Voicing::Voiceless);
        assert_eq!(features_p.place, PlaceOfArticulation::Bilabial);
        assert_eq!(features_p.manner, MannerOfArticulation::Plosive);
        assert!(!features_p.is_vowel);

        let features_i = extract_phonetic_features("i");
        assert_eq!(features_i.voicing, Voicing::Voiced);
        assert_eq!(features_i.place, PlaceOfArticulation::Front);
        assert_eq!(features_i.manner, MannerOfArticulation::Close);
        assert!(features_i.is_vowel);
    }
}

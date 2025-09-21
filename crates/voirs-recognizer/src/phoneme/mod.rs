//! Phoneme recognition and forced alignment implementations
//!
//! This module provides phoneme recognition capabilities including:
//! - Forced alignment between audio and text/phonemes
//! - Montreal Forced Alignment (MFA) integration
//! - Custom alignment algorithms
//!
//! These tools are essential for speech synthesis evaluation and pronunciation assessment.

use crate::traits::{PhonemeRecognizer, RecognitionResult};
#[allow(unused_imports)] // Used in conditional compilation
use crate::{LanguageCode, RecognitionError};
use std::sync::Arc;

// Core phoneme sets and utilities
pub mod analysis;
pub mod confidence;
pub mod phoneme_sets;
pub use analysis::*;
pub use confidence::*;
pub use phoneme_sets::*;

// Alignment implementations
#[cfg(feature = "forced-align")]
pub mod forced_align;
#[cfg(feature = "forced-align")]
pub use forced_align::ForcedAlignModel;

#[cfg(feature = "mfa")]
pub mod mfa;
#[cfg(feature = "mfa")]
pub use mfa::MFAModel;

/// Phoneme recognizer backend enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum PhonemeRecognizerBackend {
    /// Basic forced alignment
    ForcedAlign {
        /// Model path
        model_path: String,
        /// Dictionary path
        dictionary_path: Option<String>,
    },
    /// Montreal Forced Alignment
    MFA {
        /// Model name or path
        model: String,
        /// Dictionary name or path
        dictionary: String,
        /// Acoustic model path
        acoustic_model_path: Option<String>,
    },
}

/// Phoneme recognition quality metrics
#[derive(Debug, Clone, PartialEq)]
pub struct PhonemeQualityMetrics {
    /// Phoneme accuracy (percentage of correctly aligned phonemes)
    pub phoneme_accuracy: f32,
    /// Word accuracy (percentage of correctly aligned words)
    pub word_accuracy: f32,
    /// Average alignment confidence
    pub average_confidence: f32,
    /// Alignment precision (how precise the boundaries are)
    pub boundary_precision: f32,
    /// Temporal consistency (how consistent timing is)
    pub temporal_consistency: f32,
}

/// Pronunciation assessment result
#[derive(Debug, Clone, PartialEq)]
pub struct PronunciationAssessment {
    /// Overall pronunciation score [0.0, 1.0]
    pub overall_score: f32,
    /// Per-phoneme scores
    pub phoneme_scores: Vec<PhonemeScore>,
    /// Word-level pronunciation scores
    pub word_scores: Vec<WordPronunciationScore>,
    /// Identified mispronunciations
    pub mispronunciations: Vec<Mispronunciation>,
    /// Fluency score
    pub fluency_score: f32,
    /// Rhythm score
    pub rhythm_score: f32,
}

/// Individual phoneme pronunciation score
#[derive(Debug, Clone, PartialEq)]
pub struct PhonemeScore {
    /// Expected phoneme
    pub expected: String,
    /// Actual/detected phoneme
    pub actual: String,
    /// Accuracy score [0.0, 1.0]
    pub accuracy: f32,
    /// Duration accuracy
    pub duration_accuracy: f32,
    /// Position in word
    pub position: usize,
}

/// Word-level pronunciation score
#[derive(Debug, Clone, PartialEq)]
pub struct WordPronunciationScore {
    /// Word text
    pub word: String,
    /// Expected pronunciation
    pub expected_phonemes: Vec<String>,
    /// Actual pronunciation
    pub actual_phonemes: Vec<String>,
    /// Word-level accuracy [0.0, 1.0]
    pub accuracy: f32,
    /// Stress pattern accuracy
    pub stress_accuracy: f32,
    /// Syllable count accuracy
    pub syllable_accuracy: f32,
}

/// Mispronunciation detection
#[derive(Debug, Clone, PartialEq)]
pub struct Mispronunciation {
    /// Position in text
    pub position: usize,
    /// Expected phoneme/word
    pub expected: String,
    /// Actual phoneme/word
    pub actual: String,
    /// Type of error
    pub error_type: MispronunciationType,
    /// Severity [0.0, 1.0]
    pub severity: f32,
    /// Suggested correction
    pub suggestion: Option<String>,
}

/// Types of pronunciation errors
#[derive(Debug, Clone, PartialEq)]
pub enum MispronunciationType {
    /// Phoneme substitution
    Substitution,
    /// Phoneme insertion
    Insertion,
    /// Phoneme deletion
    Deletion,
    /// Stress pattern error
    StressError,
    /// Duration error (too long/short)
    DurationError,
    /// Voice quality error
    VoiceQualityError,
}

/// Factory function to create phoneme recognizers
pub async fn create_phoneme_recognizer(
    backend: PhonemeRecognizerBackend,
) -> RecognitionResult<Arc<dyn PhonemeRecognizer>> {
    match backend {
        #[cfg(feature = "forced-align")]
        PhonemeRecognizerBackend::ForcedAlign {
            model_path,
            dictionary_path,
        } => {
            let model = ForcedAlignModel::new(model_path, dictionary_path).await?;
            Ok(Arc::new(model))
        }
        #[cfg(not(feature = "forced-align"))]
        PhonemeRecognizerBackend::ForcedAlign { .. } => {
            Err(RecognitionError::FeatureNotSupported {
                feature: "forced-align".to_string(),
            }
            .into())
        }

        #[cfg(feature = "mfa")]
        PhonemeRecognizerBackend::MFA {
            model,
            dictionary,
            acoustic_model_path,
        } => {
            let mfa_model = MFAModel::new(model, dictionary, acoustic_model_path).await?;
            Ok(Arc::new(mfa_model))
        }
        #[cfg(not(feature = "mfa"))]
        PhonemeRecognizerBackend::MFA { .. } => Err(RecognitionError::FeatureNotSupported {
            feature: "mfa".to_string(),
        }
        .into()),
    }
}

/// Get recommended phoneme recognizer for a language
#[must_use]
pub fn recommended_backend_for_language(language: LanguageCode) -> PhonemeRecognizerBackend {
    match language {
        LanguageCode::EnUs | LanguageCode::EnGb => PhonemeRecognizerBackend::MFA {
            model: "english_us_arpa".to_string(),
            dictionary: "english_us_arpa".to_string(),
            acoustic_model_path: None,
        },
        LanguageCode::DeDe => PhonemeRecognizerBackend::MFA {
            model: "german_mfa".to_string(),
            dictionary: "german_mfa".to_string(),
            acoustic_model_path: None,
        },
        LanguageCode::FrFr => PhonemeRecognizerBackend::MFA {
            model: "french_mfa".to_string(),
            dictionary: "french_mfa".to_string(),
            acoustic_model_path: None,
        },
        _ => {
            // Default to basic forced alignment
            PhonemeRecognizerBackend::ForcedAlign {
                model_path: "default_acoustic_model.bin".to_string(),
                dictionary_path: None,
            }
        }
    }
}

/// Utility functions for phoneme processing
pub mod utils {
    use super::{
        AlignedPhoneme, LanguageCode, Mispronunciation, MispronunciationType, PhonemeAlignment,
        PhonemeScore, PronunciationAssessment, WordPronunciationScore,
    };

    use voirs_sdk::Phoneme;

    /// Calculate phoneme accuracy between expected and actual phonemes
    #[must_use]
    pub fn calculate_phoneme_accuracy(expected: &[Phoneme], actual: &[AlignedPhoneme]) -> f32 {
        if expected.is_empty() || actual.is_empty() {
            return 0.0;
        }

        let min_len = expected.len().min(actual.len());
        let mut correct = 0;

        for i in 0..min_len {
            if expected[i].symbol == actual[i].phoneme.symbol {
                correct += 1;
            }
        }

        correct as f32 / expected.len() as f32
    }

    /// Calculate word accuracy from phoneme alignment
    #[must_use]
    pub fn calculate_word_accuracy(alignment: &PhonemeAlignment) -> f32 {
        if alignment.word_alignments.is_empty() {
            return 0.0;
        }

        let total_words = alignment.word_alignments.len();
        let accurate_words = alignment
            .word_alignments
            .iter()
            .filter(|w| w.confidence > 0.7) // Threshold for "accurate"
            .count();

        accurate_words as f32 / total_words as f32
    }

    /// Calculate boundary precision (how precise the timing boundaries are)
    #[must_use]
    pub fn calculate_boundary_precision(alignment: &PhonemeAlignment) -> f32 {
        if alignment.phonemes.len() < 2 {
            return 1.0;
        }

        let mut precision_sum = 0.0;
        let mut count = 0;

        for window in alignment.phonemes.windows(2) {
            let gap = window[1].start_time - window[0].end_time;
            let ideal_gap = 0.01; // 10ms ideal gap
            let precision = 1.0 - (gap - ideal_gap).abs().min(0.1) / 0.1;
            precision_sum += precision;
            count += 1;
        }

        if count > 0 {
            precision_sum / count as f32
        } else {
            1.0
        }
    }

    /// Assess pronunciation quality
    #[must_use]
    pub fn assess_pronunciation(
        expected: &[Phoneme],
        alignment: &PhonemeAlignment,
        language: LanguageCode,
    ) -> PronunciationAssessment {
        let phoneme_accuracy = calculate_phoneme_accuracy(expected, &alignment.phonemes);
        let word_accuracy = calculate_word_accuracy(alignment);

        // Calculate per-phoneme scores
        let mut phoneme_scores = Vec::new();
        let min_len = expected.len().min(alignment.phonemes.len());

        for i in 0..min_len {
            let expected_phoneme = &expected[i];
            let actual_phoneme = &alignment.phonemes[i];

            let accuracy = if expected_phoneme.symbol == actual_phoneme.phoneme.symbol {
                1.0
            } else {
                calculate_phoneme_similarity(
                    &expected_phoneme.symbol,
                    &actual_phoneme.phoneme.symbol,
                    language,
                )
            };

            let duration_accuracy = calculate_duration_accuracy(expected_phoneme, actual_phoneme);

            phoneme_scores.push(PhonemeScore {
                expected: expected_phoneme.symbol.clone(),
                actual: actual_phoneme.phoneme.symbol.clone(),
                accuracy,
                duration_accuracy,
                position: i,
            });
        }

        // Calculate word scores
        let word_scores = alignment
            .word_alignments
            .iter()
            .map(|word_alignment| WordPronunciationScore {
                word: word_alignment.word.clone(),
                expected_phonemes: word_alignment
                    .phonemes
                    .iter()
                    .map(|p| p.phoneme.symbol.clone())
                    .collect(),
                actual_phonemes: word_alignment
                    .phonemes
                    .iter()
                    .map(|p| p.phoneme.symbol.clone())
                    .collect(),
                accuracy: word_alignment.confidence,
                stress_accuracy: calculate_stress_accuracy(&word_alignment.phonemes),
                syllable_accuracy: calculate_syllable_accuracy(&word_alignment.phonemes),
            })
            .collect();

        // Detect mispronunciations
        let mispronunciations = detect_mispronunciations(expected, &alignment.phonemes);

        // Calculate overall scores
        let overall_score = (phoneme_accuracy + word_accuracy) / 2.0;
        let fluency_score = calculate_fluency_score(alignment);
        let rhythm_score = calculate_rhythm_score(alignment);

        PronunciationAssessment {
            overall_score,
            phoneme_scores,
            word_scores,
            mispronunciations,
            fluency_score,
            rhythm_score,
        }
    }

    /// Calculate phoneme similarity for partial credit
    fn calculate_phoneme_similarity(expected: &str, actual: &str, _language: LanguageCode) -> f32 {
        // Simple similarity based on phonetic features
        // In a real implementation, this would use phonetic feature matrices
        if expected == actual {
            1.0
        } else if are_similar_phonemes(expected, actual) {
            0.7
        } else if are_same_category(expected, actual) {
            0.4
        } else {
            0.0
        }
    }

    /// Check if phonemes are similar (e.g., /p/ and /b/)
    fn are_similar_phonemes(p1: &str, p2: &str) -> bool {
        // Simplified similarity checking
        let similar_pairs = [
            ("p", "b"),
            ("t", "d"),
            ("k", "g"),
            ("f", "v"),
            ("s", "z"),
            ("θ", "ð"),
            ("ʃ", "ʒ"),
            ("tʃ", "dʒ"),
        ];

        similar_pairs
            .iter()
            .any(|(a, b)| (p1 == *a && p2 == *b) || (p1 == *b && p2 == *a))
    }

    /// Check if phonemes are in the same category
    fn are_same_category(p1: &str, p2: &str) -> bool {
        let vowels = ["a", "e", "i", "o", "u", "ə", "ɛ", "ɪ", "ɔ", "ʊ", "æ"];
        let consonants = [
            "p", "b", "t", "d", "k", "g", "f", "v", "s", "z", "m", "n", "l", "r",
        ];

        (vowels.contains(&p1) && vowels.contains(&p2))
            || (consonants.contains(&p1) && consonants.contains(&p2))
    }

    /// Calculate duration accuracy
    fn calculate_duration_accuracy(expected: &Phoneme, actual: &AlignedPhoneme) -> f32 {
        if let Some(expected_duration) = expected.duration_ms {
            let actual_duration = (actual.end_time - actual.start_time) * 1000.0;
            let ratio =
                (actual_duration / expected_duration).min(expected_duration / actual_duration);
            ratio.max(0.0)
        } else {
            1.0 // No expected duration to compare
        }
    }

    /// Calculate stress accuracy for a word based on phoneme stress patterns
    fn calculate_stress_accuracy(phonemes: &[AlignedPhoneme]) -> f32 {
        if phonemes.is_empty() {
            return 1.0;
        }

        // Get stress patterns from phonemes
        let stress_levels: Vec<i32> = phonemes.iter().map(|p| p.phoneme.stress as i32).collect();

        // Find primary stress positions (stress level > 0)
        let primary_stress_positions: Vec<usize> = stress_levels
            .iter()
            .enumerate()
            .filter(|(_, &stress)| stress > 0)
            .map(|(i, _)| i)
            .collect();

        // For most English words, there should be exactly one primary stress
        // Calculate accuracy based on stress pattern plausibility
        let accuracy = if primary_stress_positions.is_empty() {
            // No stress marked - could be correct for function words
            if phonemes.len() <= 2 {
                1.0 // Short words often have no marked stress
            } else {
                0.7 // Longer words should typically have stress
            }
        } else if primary_stress_positions.len() == 1 {
            // One primary stress - typically correct
            let stress_pos = primary_stress_positions[0];
            let word_length = phonemes.len();

            // Stress position should typically be in first 2/3 of the word for English
            if stress_pos < (word_length * 2) / 3 {
                1.0
            } else {
                0.8 // Less common but possible stress pattern
            }
        } else {
            // Multiple primary stresses - less common but possible for compounds
            0.6
        };

        accuracy
    }

    /// Calculate syllable accuracy for a word based on phoneme syllable positions
    fn calculate_syllable_accuracy(phonemes: &[AlignedPhoneme]) -> f32 {
        if phonemes.is_empty() {
            return 1.0;
        }

        use voirs_sdk::types::SyllablePosition;

        // Count syllables by counting nuclei (vowels/syllabic consonants)
        let nuclei_count = phonemes
            .iter()
            .filter(|p| p.phoneme.syllable_position == SyllablePosition::Nucleus)
            .count();

        // Estimate expected syllable count based on phoneme pattern
        let vowel_phonemes = [
            "a", "e", "i", "o", "u", "ɑ", "ɛ", "ɪ", "ɔ", "ʌ", "æ", "ə", "ɨ", "ʊ",
        ];
        let estimated_syllables = phonemes
            .iter()
            .filter(|p| {
                vowel_phonemes
                    .iter()
                    .any(|vowel| p.phoneme.symbol.contains(vowel))
            })
            .count();

        if estimated_syllables == 0 {
            return 1.0;
        }

        // Calculate accuracy based on how close the detected syllable count is to expected
        let syllable_accuracy = if nuclei_count == estimated_syllables {
            1.0
        } else if nuclei_count == 0 {
            // No nuclei detected - significant issue
            0.3
        } else {
            // Partial match - calculate proportional accuracy
            let diff = (nuclei_count as i32 - estimated_syllables as i32).abs() as f32;
            let max_syllables = estimated_syllables.max(nuclei_count) as f32;
            1.0 - (diff / max_syllables).min(1.0)
        };

        syllable_accuracy
    }

    /// Detect mispronunciations
    #[must_use]
    pub fn detect_mispronunciations(
        expected: &[Phoneme],
        actual: &[AlignedPhoneme],
    ) -> Vec<Mispronunciation> {
        let mut mispronunciations = Vec::new();
        let min_len = expected.len().min(actual.len());

        for i in 0..min_len {
            let exp = &expected[i];
            let act = &actual[i];

            if exp.symbol != act.phoneme.symbol {
                let severity = 1.0
                    - calculate_phoneme_similarity(
                        &exp.symbol,
                        &act.phoneme.symbol,
                        LanguageCode::EnUs,
                    );

                mispronunciations.push(Mispronunciation {
                    position: i,
                    expected: exp.symbol.clone(),
                    actual: act.phoneme.symbol.clone(),
                    error_type: MispronunciationType::Substitution,
                    severity,
                    suggestion: Some(format!(
                        "Try pronouncing '{}' instead of '{}'",
                        exp.symbol, act.phoneme.symbol
                    )),
                });
            }
        }

        // Detect insertions
        if actual.len() > expected.len() {
            for i in expected.len()..actual.len() {
                mispronunciations.push(Mispronunciation {
                    position: i,
                    expected: String::new(),
                    actual: actual[i].phoneme.symbol.clone(),
                    error_type: MispronunciationType::Insertion,
                    severity: 0.7,
                    suggestion: Some(format!("Remove extra '{}' sound", actual[i].phoneme.symbol)),
                });
            }
        }

        // Detect deletions
        if expected.len() > actual.len() {
            for i in actual.len()..expected.len() {
                mispronunciations.push(Mispronunciation {
                    position: i,
                    expected: expected[i].symbol.clone(),
                    actual: String::new(),
                    error_type: MispronunciationType::Deletion,
                    severity: 0.8,
                    suggestion: Some(format!("Add missing '{}' sound", expected[i].symbol)),
                });
            }
        }

        mispronunciations
    }

    /// Calculate fluency score based on timing and rhythm
    fn calculate_fluency_score(alignment: &PhonemeAlignment) -> f32 {
        if alignment.phonemes.len() < 2 {
            return 1.0;
        }

        // Calculate speech rate
        let total_duration = alignment.total_duration;
        let phoneme_count = alignment.phonemes.len();
        let speech_rate = phoneme_count as f32 / total_duration;

        // Ideal speech rate is around 10-15 phonemes per second
        let ideal_rate = 12.0;
        let rate_score = 1.0 - (speech_rate - ideal_rate).abs().min(5.0) / 5.0;

        // Calculate rhythm regularity
        let rhythm_score = calculate_rhythm_regularity(alignment);

        (rate_score + rhythm_score) / 2.0
    }

    /// Calculate rhythm score
    fn calculate_rhythm_score(alignment: &PhonemeAlignment) -> f32 {
        calculate_rhythm_regularity(alignment)
    }

    /// Calculate rhythm regularity
    fn calculate_rhythm_regularity(alignment: &PhonemeAlignment) -> f32 {
        if alignment.phonemes.len() < 3 {
            return 1.0;
        }

        // Calculate inter-phoneme intervals
        let mut intervals = Vec::new();
        for window in alignment.phonemes.windows(2) {
            let interval = window[1].start_time - window[0].start_time;
            intervals.push(interval);
        }

        // Calculate coefficient of variation
        let mean_interval: f32 = intervals.iter().sum::<f32>() / intervals.len() as f32;
        let variance: f32 = intervals
            .iter()
            .map(|x| (x - mean_interval).powi(2))
            .sum::<f32>()
            / intervals.len() as f32;
        let std_dev = variance.sqrt();

        if mean_interval > 0.0 {
            let cv = std_dev / mean_interval;
            // Lower coefficient of variation indicates more regular rhythm
            1.0 - cv.min(1.0)
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use voirs_sdk::Phoneme;

    #[test]
    fn test_phoneme_accuracy_calculation() {
        let expected = vec![
            Phoneme {
                symbol: "h".to_string(),
                ipa_symbol: "h".to_string(),
                stress: 0,
                syllable_position: voirs_sdk::types::SyllablePosition::Onset,
                duration_ms: Some(100.0),
                confidence: 1.0,
            },
            Phoneme {
                symbol: "ɛ".to_string(),
                ipa_symbol: "ɛ".to_string(),
                stress: 0,
                syllable_position: voirs_sdk::types::SyllablePosition::Nucleus,
                duration_ms: Some(200.0),
                confidence: 1.0,
            },
        ];

        let actual = vec![
            AlignedPhoneme {
                phoneme: Phoneme {
                    symbol: "h".to_string(),
                    ipa_symbol: "h".to_string(),
                    stress: 0,
                    syllable_position: voirs_sdk::types::SyllablePosition::Onset,
                    duration_ms: Some(100.0),
                    confidence: 1.0,
                },
                start_time: 0.0,
                end_time: 0.1,
                confidence: 0.9,
            },
            AlignedPhoneme {
                phoneme: Phoneme {
                    symbol: "e".to_string(), // Different from expected
                    ipa_symbol: "e".to_string(),
                    stress: 0,
                    syllable_position: voirs_sdk::types::SyllablePosition::Nucleus,
                    duration_ms: Some(200.0),
                    confidence: 1.0,
                },
                start_time: 0.1,
                end_time: 0.3,
                confidence: 0.8,
            },
        ];

        let accuracy = utils::calculate_phoneme_accuracy(&expected, &actual);
        assert!((accuracy - 0.5).abs() < f32::EPSILON); // 1 out of 2 correct
    }

    #[test]
    fn test_recommended_backend() {
        let backend = recommended_backend_for_language(LanguageCode::EnUs);
        match backend {
            PhonemeRecognizerBackend::MFA {
                model, dictionary, ..
            } => {
                assert!(model.contains("english"));
                assert!(dictionary.contains("english"));
            }
            PhonemeRecognizerBackend::ForcedAlign { .. } => {
                assert!(false, "Expected MFA backend for English")
            }
        }
    }

    #[test]
    fn test_mispronunciation_detection() {
        let expected = vec![Phoneme {
            symbol: "p".to_string(),
            ipa_symbol: "p".to_string(),
            stress: 0,
            syllable_position: voirs_sdk::types::SyllablePosition::Onset,
            duration_ms: Some(50.0),
            confidence: 1.0,
        }];

        let actual = vec![AlignedPhoneme {
            phoneme: Phoneme {
                symbol: "b".to_string(), // Substitution error
                ipa_symbol: "b".to_string(),
                stress: 0,
                syllable_position: voirs_sdk::types::SyllablePosition::Onset,
                duration_ms: Some(50.0),
                confidence: 1.0,
            },
            start_time: 0.0,
            end_time: 0.05,
            confidence: 0.9,
        }];

        let mispronunciations = utils::detect_mispronunciations(&expected, &actual);
        assert_eq!(mispronunciations.len(), 1);
        assert_eq!(
            mispronunciations[0].error_type,
            MispronunciationType::Substitution
        );
        assert_eq!(mispronunciations[0].expected, "p");
        assert_eq!(mispronunciations[0].actual, "b");
    }
}

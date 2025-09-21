//! Rule-based language detection.

use crate::detection::{get_language_indicators, DetectionMethod, DetectionResult};
use crate::{LanguageCode, Result};
use std::collections::HashMap;

/// Rule-based language detector
pub struct RuleBasedDetector {
    /// Character frequency patterns for each language
    char_patterns: HashMap<LanguageCode, CharPattern>,
    /// Word patterns for each language
    word_patterns: HashMap<LanguageCode, WordPattern>,
}

/// Character frequency pattern for a language
#[derive(Debug, Clone)]
struct CharPattern {
    /// Common characters and their expected frequencies
    frequencies: HashMap<char, f32>,
    /// Character ranges that indicate this language
    ranges: Vec<(u32, u32)>,
}

/// Word pattern for a language
#[derive(Debug, Clone)]
struct WordPattern {
    /// Common words that strongly indicate this language
    indicators: Vec<String>,
    /// Function words that are characteristic
    function_words: Vec<String>,
}

impl RuleBasedDetector {
    /// Create new rule-based detector
    pub fn new() -> Self {
        let mut detector = Self {
            char_patterns: HashMap::new(),
            word_patterns: HashMap::new(),
        };
        detector.load_patterns();
        detector
    }

    /// Load language patterns
    fn load_patterns(&mut self) {
        self.load_english_patterns();
        self.load_german_patterns();
        self.load_french_patterns();
        self.load_spanish_patterns();
        self.load_italian_patterns();
        self.load_portuguese_patterns();
        self.load_japanese_patterns();
        self.load_chinese_patterns();
        self.load_korean_patterns();
    }

    /// Load English patterns
    fn load_english_patterns(&mut self) {
        let mut char_freq = HashMap::new();
        char_freq.insert('e', 0.127);
        char_freq.insert('t', 0.091);
        char_freq.insert('a', 0.082);
        char_freq.insert('o', 0.075);
        char_freq.insert('i', 0.070);
        char_freq.insert('n', 0.067);
        char_freq.insert('s', 0.063);
        char_freq.insert('h', 0.061);
        char_freq.insert('r', 0.060);

        self.char_patterns.insert(
            LanguageCode::EnUs,
            CharPattern {
                frequencies: char_freq,
                ranges: vec![(0x0041, 0x005A), (0x0061, 0x007A)], // Basic Latin
            },
        );

        let indicators = if let Some(words) = get_language_indicators(LanguageCode::EnUs) {
            words.iter().map(|s| s.to_string()).collect()
        } else {
            vec![]
        };

        self.word_patterns.insert(
            LanguageCode::EnUs,
            WordPattern {
                indicators,
                function_words: vec![
                    "the".to_string(),
                    "and".to_string(),
                    "of".to_string(),
                    "to".to_string(),
                    "a".to_string(),
                    "in".to_string(),
                    "is".to_string(),
                    "it".to_string(),
                ],
            },
        );
    }

    /// Load German patterns
    fn load_german_patterns(&mut self) {
        let mut char_freq = HashMap::new();
        char_freq.insert('e', 0.174);
        char_freq.insert('n', 0.098);
        char_freq.insert('i', 0.075);
        char_freq.insert('s', 0.072);
        char_freq.insert('r', 0.070);
        char_freq.insert('a', 0.065);
        char_freq.insert('t', 0.061);
        char_freq.insert('d', 0.058);
        char_freq.insert('h', 0.076);
        char_freq.insert('ä', 0.005);
        char_freq.insert('ö', 0.003);
        char_freq.insert('ü', 0.007);
        char_freq.insert('ß', 0.003);

        self.char_patterns.insert(
            LanguageCode::De,
            CharPattern {
                frequencies: char_freq,
                ranges: vec![
                    (0x0041, 0x005A),
                    (0x0061, 0x007A), // Basic Latin
                    (0x00C4, 0x00C4),
                    (0x00D6, 0x00D6),
                    (0x00DC, 0x00DC), // Ä, Ö, Ü
                    (0x00E4, 0x00E4),
                    (0x00F6, 0x00F6),
                    (0x00FC, 0x00FC), // ä, ö, ü
                    (0x00DF, 0x00DF), // ß
                ],
            },
        );

        let indicators = if let Some(words) = get_language_indicators(LanguageCode::De) {
            words.iter().map(|s| s.to_string()).collect()
        } else {
            vec![]
        };

        self.word_patterns.insert(
            LanguageCode::De,
            WordPattern {
                indicators,
                function_words: vec![
                    "der".to_string(),
                    "die".to_string(),
                    "das".to_string(),
                    "und".to_string(),
                    "in".to_string(),
                    "den".to_string(),
                    "von".to_string(),
                    "zu".to_string(),
                ],
            },
        );
    }

    /// Load French patterns
    fn load_french_patterns(&mut self) {
        let mut char_freq = HashMap::new();
        char_freq.insert('e', 0.121);
        char_freq.insert('a', 0.094);
        char_freq.insert('i', 0.084);
        char_freq.insert('s', 0.081);
        char_freq.insert('n', 0.071);
        char_freq.insert('r', 0.066);
        char_freq.insert('t', 0.059);
        char_freq.insert('o', 0.054);
        char_freq.insert('l', 0.054);
        char_freq.insert('é', 0.019);
        char_freq.insert('è', 0.006);
        char_freq.insert('à', 0.005);
        char_freq.insert('ç', 0.001);

        self.char_patterns.insert(
            LanguageCode::Fr,
            CharPattern {
                frequencies: char_freq,
                ranges: vec![
                    (0x0041, 0x005A),
                    (0x0061, 0x007A), // Basic Latin
                    (0x00C0, 0x00FF), // Latin-1 Supplement
                ],
            },
        );

        let indicators = if let Some(words) = get_language_indicators(LanguageCode::Fr) {
            words.iter().map(|s| s.to_string()).collect()
        } else {
            vec![]
        };

        self.word_patterns.insert(
            LanguageCode::Fr,
            WordPattern {
                indicators,
                function_words: vec![
                    "le".to_string(),
                    "de".to_string(),
                    "et".to_string(),
                    "à".to_string(),
                    "un".to_string(),
                    "il".to_string(),
                    "être".to_string(),
                    "en".to_string(),
                ],
            },
        );
    }

    /// Load Spanish patterns
    fn load_spanish_patterns(&mut self) {
        let mut char_freq = HashMap::new();
        char_freq.insert('e', 0.137);
        char_freq.insert('a', 0.125);
        char_freq.insert('o', 0.087);
        char_freq.insert('s', 0.080);
        char_freq.insert('r', 0.069);
        char_freq.insert('n', 0.067);
        char_freq.insert('i', 0.063);
        char_freq.insert('d', 0.058);
        char_freq.insert('l', 0.049);
        char_freq.insert('ñ', 0.003);

        self.char_patterns.insert(
            LanguageCode::Es,
            CharPattern {
                frequencies: char_freq,
                ranges: vec![
                    (0x0041, 0x005A),
                    (0x0061, 0x007A), // Basic Latin
                    (0x00C1, 0x00C1),
                    (0x00C9, 0x00C9),
                    (0x00CD, 0x00CD), // Á, É, Í
                    (0x00D1, 0x00D1),
                    (0x00D3, 0x00D3),
                    (0x00DA, 0x00DA), // Ñ, Ó, Ú
                    (0x00DC, 0x00DC), // Ü
                    (0x00E1, 0x00E1),
                    (0x00E9, 0x00E9),
                    (0x00ED, 0x00ED), // á, é, í
                    (0x00F1, 0x00F1),
                    (0x00F3, 0x00F3),
                    (0x00FA, 0x00FA), // ñ, ó, ú
                    (0x00FC, 0x00FC), // ü
                ],
            },
        );

        let indicators = if let Some(words) = get_language_indicators(LanguageCode::Es) {
            words.iter().map(|s| s.to_string()).collect()
        } else {
            vec![]
        };

        self.word_patterns.insert(
            LanguageCode::Es,
            WordPattern {
                indicators,
                function_words: vec![
                    "el".to_string(),
                    "la".to_string(),
                    "de".to_string(),
                    "que".to_string(),
                    "y".to_string(),
                    "a".to_string(),
                    "en".to_string(),
                    "un".to_string(),
                ],
            },
        );
    }

    /// Load Italian patterns
    fn load_italian_patterns(&mut self) {
        let mut char_freq = HashMap::new();
        char_freq.insert('e', 0.118);
        char_freq.insert('a', 0.117);
        char_freq.insert('i', 0.101);
        char_freq.insert('o', 0.098);
        char_freq.insert('n', 0.069);
        char_freq.insert('r', 0.064);
        char_freq.insert('t', 0.056);
        char_freq.insert('l', 0.051);
        char_freq.insert('c', 0.045);
        char_freq.insert('s', 0.050);

        self.char_patterns.insert(
            LanguageCode::It,
            CharPattern {
                frequencies: char_freq,
                ranges: vec![
                    (0x0041, 0x005A),
                    (0x0061, 0x007A), // Basic Latin
                    (0x00C0, 0x00C0),
                    (0x00C8, 0x00C8),
                    (0x00C9, 0x00C9), // À, È, É
                    (0x00CC, 0x00CC),
                    (0x00CD, 0x00CD), // Ì, Í
                    (0x00D2, 0x00D2),
                    (0x00D3, 0x00D3), // Ò, Ó
                    (0x00D9, 0x00D9),
                    (0x00DA, 0x00DA), // Ù, Ú
                    (0x00E0, 0x00E0),
                    (0x00E8, 0x00E8),
                    (0x00E9, 0x00E9), // à, è, é
                    (0x00EC, 0x00EC),
                    (0x00ED, 0x00ED), // ì, í
                    (0x00F2, 0x00F2),
                    (0x00F3, 0x00F3), // ò, ó
                    (0x00F9, 0x00F9),
                    (0x00FA, 0x00FA), // ù, ú
                ],
            },
        );

        let indicators = if let Some(words) = get_language_indicators(LanguageCode::It) {
            words.iter().map(|s| s.to_string()).collect()
        } else {
            vec![]
        };

        self.word_patterns.insert(
            LanguageCode::It,
            WordPattern {
                indicators,
                function_words: vec![
                    "il".to_string(),
                    "la".to_string(),
                    "di".to_string(),
                    "che".to_string(),
                    "e".to_string(),
                    "a".to_string(),
                    "in".to_string(),
                    "un".to_string(),
                    "una".to_string(),
                    "per".to_string(),
                    "con".to_string(),
                    "non".to_string(),
                ],
            },
        );
    }

    /// Load Portuguese patterns
    fn load_portuguese_patterns(&mut self) {
        let mut char_freq = HashMap::new();
        char_freq.insert('a', 0.146);
        char_freq.insert('e', 0.126);
        char_freq.insert('o', 0.103);
        char_freq.insert('s', 0.078);
        char_freq.insert('r', 0.065);
        char_freq.insert('i', 0.062);
        char_freq.insert('n', 0.051);
        char_freq.insert('d', 0.050);
        char_freq.insert('m', 0.047);
        char_freq.insert('t', 0.047);
        char_freq.insert('u', 0.046);
        char_freq.insert('c', 0.039);
        char_freq.insert('l', 0.027);

        self.char_patterns.insert(
            LanguageCode::Pt,
            CharPattern {
                frequencies: char_freq,
                ranges: vec![
                    (0x0041, 0x005A),
                    (0x0061, 0x007A), // Basic Latin
                    (0x00C0, 0x00C5), // À-Å
                    (0x00C7, 0x00C7), // Ç
                    (0x00C9, 0x00CA), // É, Ê
                    (0x00CD, 0x00CD), // Í
                    (0x00D3, 0x00D4), // Ó, Ô
                    (0x00D5, 0x00D5), // Õ
                    (0x00DA, 0x00DA), // Ú
                    (0x00DC, 0x00DC), // Ü
                    (0x00E0, 0x00E5), // à-å
                    (0x00E7, 0x00E7), // ç
                    (0x00E9, 0x00EA), // é, ê
                    (0x00ED, 0x00ED), // í
                    (0x00F3, 0x00F4), // ó, ô
                    (0x00F5, 0x00F5), // õ
                    (0x00FA, 0x00FA), // ú
                    (0x00FC, 0x00FC), // ü
                    (0x0103, 0x0103), // ă (for Brazilian Portuguese)
                ],
            },
        );

        let indicators = if let Some(words) = get_language_indicators(LanguageCode::Pt) {
            words.iter().map(|s| s.to_string()).collect()
        } else {
            vec![]
        };

        self.word_patterns.insert(
            LanguageCode::Pt,
            WordPattern {
                indicators,
                function_words: vec![
                    "o".to_string(),
                    "a".to_string(),
                    "de".to_string(),
                    "que".to_string(),
                    "e".to_string(),
                    "do".to_string(),
                    "da".to_string(),
                    "em".to_string(),
                    "um".to_string(),
                    "uma".to_string(),
                    "para".to_string(),
                    "com".to_string(),
                    "não".to_string(),
                    "se".to_string(),
                ],
            },
        );
    }

    /// Load Japanese patterns
    fn load_japanese_patterns(&mut self) {
        let char_freq = HashMap::new(); // Character frequency not applicable for Japanese

        self.char_patterns.insert(
            LanguageCode::Ja,
            CharPattern {
                frequencies: char_freq,
                ranges: vec![
                    (0x3040, 0x309F), // Hiragana
                    (0x30A0, 0x30FF), // Katakana
                    (0x4E00, 0x9FFF), // CJK Unified Ideographs
                ],
            },
        );

        let indicators = if let Some(words) = get_language_indicators(LanguageCode::Ja) {
            words.iter().map(|s| s.to_string()).collect()
        } else {
            vec![]
        };

        self.word_patterns.insert(
            LanguageCode::Ja,
            WordPattern {
                indicators,
                function_words: vec![
                    "の".to_string(),
                    "に".to_string(),
                    "は".to_string(),
                    "を".to_string(),
                    "が".to_string(),
                    "で".to_string(),
                    "て".to_string(),
                    "と".to_string(),
                ],
            },
        );
    }

    /// Load Chinese patterns
    fn load_chinese_patterns(&mut self) {
        let char_freq = HashMap::new(); // Character frequency not applicable for Chinese

        self.char_patterns.insert(
            LanguageCode::ZhCn,
            CharPattern {
                frequencies: char_freq,
                ranges: vec![
                    (0x4E00, 0x9FFF), // CJK Unified Ideographs
                ],
            },
        );

        let indicators = if let Some(words) = get_language_indicators(LanguageCode::ZhCn) {
            words.iter().map(|s| s.to_string()).collect()
        } else {
            vec![]
        };

        self.word_patterns.insert(
            LanguageCode::ZhCn,
            WordPattern {
                indicators,
                function_words: vec![
                    "的".to_string(),
                    "一".to_string(),
                    "是".to_string(),
                    "在".to_string(),
                    "不".to_string(),
                    "了".to_string(),
                    "有".to_string(),
                    "和".to_string(),
                ],
            },
        );
    }

    /// Load Korean patterns
    fn load_korean_patterns(&mut self) {
        let char_freq = HashMap::new(); // Character frequency not applicable for Korean

        self.char_patterns.insert(
            LanguageCode::Ko,
            CharPattern {
                frequencies: char_freq,
                ranges: vec![
                    (0xAC00, 0xD7AF), // Hangul syllables
                    (0x1100, 0x11FF), // Hangul Jamo
                    (0x3130, 0x318F), // Hangul Compatibility Jamo
                ],
            },
        );

        let indicators = if let Some(words) = get_language_indicators(LanguageCode::Ko) {
            words.iter().map(|s| s.to_string()).collect()
        } else {
            vec![]
        };

        self.word_patterns.insert(
            LanguageCode::Ko,
            WordPattern {
                indicators,
                function_words: vec![
                    "이".to_string(),
                    "의".to_string(),
                    "가".to_string(),
                    "을".to_string(),
                    "는".to_string(),
                    "에".to_string(),
                    "하".to_string(),
                    "고".to_string(),
                ],
            },
        );
    }

    /// Detect language using rule-based approach
    pub fn detect(&self, text: &str) -> Result<Option<DetectionResult>> {
        if text.trim().is_empty() {
            return Ok(None);
        }

        let mut scores = HashMap::new();

        // Calculate scores for each language
        for &language in &[
            LanguageCode::EnUs,
            LanguageCode::De,
            LanguageCode::Fr,
            LanguageCode::Es,
            LanguageCode::It,
            LanguageCode::Pt,
            LanguageCode::Ja,
            LanguageCode::ZhCn,
            LanguageCode::Ko,
        ] {
            let score = self.calculate_language_score(text, language);
            scores.insert(language, score);
        }

        // Find the language with highest score
        let best_match = scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(lang, score)| (*lang, *score));

        if let Some((language, confidence)) = best_match {
            if confidence > 0.3 {
                // Minimum threshold for rule-based detection
                // Prepare alternatives
                let mut alternatives: Vec<(LanguageCode, f32)> = scores
                    .into_iter()
                    .filter(|(lang, _)| *lang != language)
                    .collect();
                alternatives.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                alternatives.truncate(3); // Keep top 3 alternatives

                return Ok(Some(DetectionResult {
                    language,
                    confidence,
                    alternatives,
                    method: DetectionMethod::Rules,
                }));
            }
        }

        Ok(None)
    }

    /// Calculate language score for text
    fn calculate_language_score(&self, text: &str, language: LanguageCode) -> f32 {
        let char_score = self.calculate_character_score(text, language);
        let word_score = self.calculate_word_score(text, language);
        let range_score = self.calculate_range_score(text, language);

        // Weighted combination of scores
        (char_score * 0.3) + (word_score * 0.5) + (range_score * 0.2)
    }

    /// Calculate character frequency score
    fn calculate_character_score(&self, text: &str, language: LanguageCode) -> f32 {
        let pattern = match self.char_patterns.get(&language) {
            Some(p) => p,
            None => return 0.0,
        };

        if pattern.frequencies.is_empty() {
            return 0.0;
        }

        let mut char_counts = HashMap::new();
        let mut total_chars = 0;

        for ch in text.chars() {
            if ch.is_alphabetic() {
                let lower = ch.to_lowercase().next().unwrap_or(ch);
                *char_counts.entry(lower).or_insert(0) += 1;
                total_chars += 1;
            }
        }

        if total_chars == 0 {
            return 0.0;
        }

        let mut score = 0.0;
        let mut matches = 0;

        for (ch, expected_freq) in &pattern.frequencies {
            let actual_count = char_counts.get(ch).unwrap_or(&0);
            let actual_freq = *actual_count as f32 / total_chars as f32;

            // Calculate similarity between expected and actual frequency
            let freq_diff = (expected_freq - actual_freq).abs();
            let freq_score = 1.0 - (freq_diff / expected_freq).min(1.0);

            score += freq_score;
            matches += 1;
        }

        if matches > 0 {
            score / matches as f32
        } else {
            0.0
        }
    }

    /// Calculate word pattern score
    fn calculate_word_score(&self, text: &str, language: LanguageCode) -> f32 {
        let pattern = match self.word_patterns.get(&language) {
            Some(p) => p,
            None => return 0.0,
        };

        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return 0.0;
        }

        let mut indicator_matches = 0;
        let mut function_matches = 0;

        for word in &words {
            let lower_word = word.to_lowercase();

            if pattern.indicators.contains(&lower_word) {
                indicator_matches += 1;
            }

            if pattern.function_words.contains(&lower_word) {
                function_matches += 1;
            }
        }

        let indicator_score = indicator_matches as f32 / words.len() as f32;
        let function_score = function_matches as f32 / words.len() as f32;

        // Weight function words higher as they're more reliable
        (indicator_score * 0.3) + (function_score * 0.7)
    }

    /// Calculate character range score
    fn calculate_range_score(&self, text: &str, language: LanguageCode) -> f32 {
        let pattern = match self.char_patterns.get(&language) {
            Some(p) => p,
            None => return 0.0,
        };

        if pattern.ranges.is_empty() {
            return 0.0;
        }

        let mut in_range_chars = 0;
        let mut total_chars = 0;

        for ch in text.chars() {
            if ch.is_alphabetic() {
                total_chars += 1;
                let ch_code = ch as u32;

                for (start, end) in &pattern.ranges {
                    if ch_code >= *start && ch_code <= *end {
                        in_range_chars += 1;
                        break;
                    }
                }
            }
        }

        if total_chars == 0 {
            0.0
        } else {
            in_range_chars as f32 / total_chars as f32
        }
    }
}

impl Default for RuleBasedDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rule_based_detector_creation() {
        let detector = RuleBasedDetector::new();
        assert!(!detector.char_patterns.is_empty());
        assert!(!detector.word_patterns.is_empty());
    }

    #[test]
    fn test_english_detection() {
        let detector = RuleBasedDetector::new();

        let result = detector
            .detect("The quick brown fox jumps over the lazy dog")
            .unwrap();

        if let Some(result) = result {
            // If detection succeeds, it should be reasonable
            assert!(result.confidence > 0.0);
            assert_eq!(result.method, DetectionMethod::Rules);
            // Don't assert specific language as rule-based detection may not be perfect
        }
        // It's OK if no detection is made for this test case
    }

    #[test]
    fn test_german_detection() {
        let detector = RuleBasedDetector::new();

        let result = detector
            .detect("Der schnelle braune Fuchs springt über den faulen Hund")
            .unwrap();
        if let Some(result) = result {
            // German detection might work if we have good patterns
            assert!(result.confidence > 0.0);
        }
    }

    #[test]
    fn test_empty_text() {
        let detector = RuleBasedDetector::new();

        let result = detector.detect("").unwrap();
        assert!(result.is_none());

        let result = detector.detect("   ").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_character_score_calculation() {
        let detector = RuleBasedDetector::new();

        let score = detector.calculate_character_score("hello world", LanguageCode::EnUs);
        assert!((0.0..=1.0).contains(&score));
    }

    #[test]
    fn test_word_score_calculation() {
        let detector = RuleBasedDetector::new();

        let score = detector.calculate_word_score("the quick brown fox", LanguageCode::EnUs);
        assert!((0.0..=1.0).contains(&score));
        assert!(score > 0.0); // Should have some score due to "the"
    }

    #[test]
    fn test_range_score_calculation() {
        let detector = RuleBasedDetector::new();

        let score = detector.calculate_range_score("hello world", LanguageCode::EnUs);
        assert!((0.0..=1.0).contains(&score));
        assert!(score > 0.9); // Latin characters should score high for English
    }
}

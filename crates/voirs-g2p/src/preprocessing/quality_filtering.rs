//! Quality filtering for G2P preprocessing.
//!
//! This module provides text quality filtering, validation, and noise removal
//! for improved G2P conversion accuracy.

use crate::preprocessing::context_aware::PhoneticContext;
use crate::preprocessing::pos_tagging::PosTag;
use crate::{G2pError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Trait for quality filtering
pub trait QualityFiltering: Send + Sync {
    /// Validate and sanitize input text
    fn validate_input(&self, text: &str) -> Result<String>;

    /// Detect and remove noise
    fn denoise_text(&self, text: &str) -> Result<String>;

    /// Detect encoding issues
    fn detect_encoding_issues(&self, text: &str) -> Result<Vec<String>>;

    /// Handle malformed input
    fn handle_malformed_input(&self, text: &str) -> Result<String>;

    /// Analyze morphological features of a word
    fn analyze_morphology(&self, word: &str, pos_tag: &PosTag) -> Result<HashMap<String, String>>;

    /// Analyze phonetic context for a word at given position
    fn analyze_phonetic_context(
        &self,
        word: &str,
        position: usize,
        tokens: &[String],
    ) -> Result<Option<PhoneticContext>>;
}

/// Basic quality filter implementation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasicQualityFilter {
    /// Minimum text length
    pub min_length: usize,
    /// Maximum text length
    pub max_length: usize,
    /// Allow non-alphabetic characters
    pub allow_non_alphabetic: bool,
    /// Enable encoding validation
    pub validate_encoding: bool,
}

impl Default for BasicQualityFilter {
    fn default() -> Self {
        Self {
            min_length: 1,
            max_length: 10000,
            allow_non_alphabetic: true,
            validate_encoding: true,
        }
    }
}

impl QualityFiltering for BasicQualityFilter {
    fn validate_input(&self, text: &str) -> Result<String> {
        if text.len() < self.min_length {
            return Err(G2pError::InvalidInput(format!(
                "Text too short: {} characters, minimum {}",
                text.len(),
                self.min_length
            )));
        }

        if text.len() > self.max_length {
            return Err(G2pError::InvalidInput(format!(
                "Text too long: {} characters, maximum {}",
                text.len(),
                self.max_length
            )));
        }

        if self.validate_encoding {
            self.detect_encoding_issues(text)?;
        }

        Ok(text.to_string())
    }

    fn denoise_text(&self, text: &str) -> Result<String> {
        // Basic noise removal: trim whitespace and normalize spaces
        let cleaned = text
            .chars()
            .map(|c| if c.is_whitespace() { ' ' } else { c })
            .collect::<String>();

        // Collapse multiple spaces
        let mut result = String::new();
        let mut last_was_space = false;

        for c in cleaned.chars() {
            if c == ' ' {
                if !last_was_space {
                    result.push(c);
                }
                last_was_space = true;
            } else {
                result.push(c);
                last_was_space = false;
            }
        }

        Ok(result.trim().to_string())
    }

    fn detect_encoding_issues(&self, text: &str) -> Result<Vec<String>> {
        let mut issues = Vec::new();

        // Check for common encoding issues
        if text.contains('\u{FFFD}') {
            issues.push("Contains replacement character (U+FFFD)".to_string());
        }

        // Check for control characters
        if text
            .chars()
            .any(|c| c.is_control() && c != '\n' && c != '\r' && c != '\t')
        {
            issues.push("Contains control characters".to_string());
        }

        // Check for mixed scripts (simplified check)
        let has_latin = text.chars().any(|c| c.is_ascii_alphabetic());
        let has_cyrillic = text.chars().any(|c| matches!(c, '\u{0400}'..='\u{04FF}'));
        let has_greek = text.chars().any(|c| matches!(c, '\u{0370}'..='\u{03FF}'));

        if [has_latin, has_cyrillic, has_greek]
            .iter()
            .filter(|&&x| x)
            .count()
            > 1
        {
            issues.push("Mixed scripts detected".to_string());
        }

        Ok(issues)
    }

    fn handle_malformed_input(&self, text: &str) -> Result<String> {
        // Handle common malformed input cases
        let mut result = text.to_string();

        // Remove replacement characters
        result = result.replace('\u{FFFD}', "");

        // Remove control characters except common whitespace
        result = result
            .chars()
            .filter(|&c| !c.is_control() || c == '\n' || c == '\r' || c == '\t')
            .collect();

        // Normalize line endings
        result = result.replace("\r\n", "\n").replace('\r', "\n");

        Ok(result)
    }

    fn analyze_morphology(&self, word: &str, pos_tag: &PosTag) -> Result<HashMap<String, String>> {
        let mut morphology = HashMap::new();

        // Basic morphological analysis based on POS tag
        match pos_tag {
            PosTag::Noun => {
                morphology.insert("category".to_string(), "noun".to_string());
                // Simple plural detection
                if word.ends_with('s') && word.len() > 1 {
                    morphology.insert("number".to_string(), "plural".to_string());
                } else {
                    morphology.insert("number".to_string(), "singular".to_string());
                }
            }
            PosTag::Verb => {
                morphology.insert("category".to_string(), "verb".to_string());
                // Simple tense detection
                if word.ends_with("ed") {
                    morphology.insert("tense".to_string(), "past".to_string());
                } else if word.ends_with("ing") {
                    morphology.insert("tense".to_string(), "present_participle".to_string());
                } else {
                    morphology.insert("tense".to_string(), "present".to_string());
                }
            }
            PosTag::Adjective => {
                morphology.insert("category".to_string(), "adjective".to_string());
                // Simple comparative detection
                if word.ends_with("er") {
                    morphology.insert("degree".to_string(), "comparative".to_string());
                } else if word.ends_with("est") {
                    morphology.insert("degree".to_string(), "superlative".to_string());
                } else {
                    morphology.insert("degree".to_string(), "positive".to_string());
                }
            }
            PosTag::Adverb => {
                morphology.insert("category".to_string(), "adverb".to_string());
                if word.ends_with("ly") {
                    morphology.insert("formation".to_string(), "derived".to_string());
                }
            }
            _ => {
                morphology.insert("category".to_string(), "other".to_string());
            }
        }

        // Basic syllable count
        let syllable_count = self.count_syllables(word);
        morphology.insert("syllables".to_string(), syllable_count.to_string());

        Ok(morphology)
    }

    fn analyze_phonetic_context(
        &self,
        word: &str,
        position: usize,
        tokens: &[String],
    ) -> Result<Option<PhoneticContext>> {
        if tokens.is_empty() {
            return Ok(None);
        }

        // Get preceding and following words
        let preceding: Vec<String> = if position > 0 {
            tokens[..position].iter().rev().take(2).cloned().collect()
        } else {
            Vec::new()
        };

        let following: Vec<String> = if position + 1 < tokens.len() {
            tokens[position + 1..].iter().take(2).cloned().collect()
        } else {
            Vec::new()
        };

        // Simple syllable structure analysis
        let syllable_structure = self.analyze_syllable_structure(word)?;

        // Simple stress pattern (default to penultimate stress for multi-syllable words)
        let syllable_count = self.count_syllables(word);
        let stress_pattern = if syllable_count > 1 {
            let mut pattern = vec![0; syllable_count];
            pattern[syllable_count.saturating_sub(2)] = 1; // Penultimate stress
            pattern
        } else {
            vec![1] // Single syllable gets primary stress
        };

        // Basic vowel harmony analysis
        let vowel_harmony = self.analyze_vowel_harmony(word)?;

        Ok(Some(PhoneticContext {
            preceding,
            following,
            syllable_structure,
            stress_pattern,
            vowel_harmony,
        }))
    }
}

impl BasicQualityFilter {
    /// Create new quality filter
    pub fn new() -> Self {
        Self::default()
    }

    /// Count syllables in a word
    fn count_syllables(&self, word: &str) -> usize {
        let word = word.to_lowercase();
        let vowels = ['a', 'e', 'i', 'o', 'u', 'y'];

        let mut count = 0;
        let mut previous_was_vowel = false;

        for ch in word.chars() {
            let is_vowel = vowels.contains(&ch);
            if is_vowel && !previous_was_vowel {
                count += 1;
            }
            previous_was_vowel = is_vowel;
        }

        // Handle silent 'e'
        if word.ends_with('e') && count > 1 {
            count -= 1;
        }

        // Special cases for common patterns that create additional syllables
        if word.contains("tion") || word.contains("sion") {
            // "tion" and "sion" usually add a syllable
            // For "pronunciation": pro-nun-ci-a-tion
            // The 'ia' creates an extra syllable before 'tion'
            if word.contains("iation") || word.contains("uation") {
                count += 1;
            }
        }

        // Every word has at least one syllable
        count.max(1)
    }

    /// Analyze syllable structure
    fn analyze_syllable_structure(&self, word: &str) -> Result<String> {
        let mut structure = String::new();
        let vowels = ['a', 'e', 'i', 'o', 'u', 'y'];

        for ch in word.to_lowercase().chars() {
            if vowels.contains(&ch) {
                structure.push('V');
            } else if ch.is_alphabetic() {
                structure.push('C');
            }
        }

        Ok(structure)
    }

    /// Analyze vowel harmony
    fn analyze_vowel_harmony(&self, word: &str) -> Result<Option<String>> {
        let word = word.to_lowercase();
        let front_vowels = ['e', 'i'];
        let back_vowels = ['a', 'o', 'u'];

        let mut front_count = 0;
        let mut back_count = 0;

        for ch in word.chars() {
            if front_vowels.contains(&ch) {
                front_count += 1;
            } else if back_vowels.contains(&ch) {
                back_count += 1;
            }
        }

        let harmony = if front_count > 0 && back_count == 0 {
            Some("front".to_string())
        } else if back_count > 0 && front_count == 0 {
            Some("back".to_string())
        } else if front_count > 0 && back_count > 0 {
            Some("mixed".to_string())
        } else {
            None
        };

        Ok(harmony)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_quality_filter() {
        let filter = BasicQualityFilter::default();

        // Test valid input
        assert!(filter.validate_input("hello world").is_ok());

        // Test too short
        let short_filter = BasicQualityFilter {
            min_length: 10,
            ..Default::default()
        };
        assert!(short_filter.validate_input("hi").is_err());
    }

    #[test]
    fn test_denoise_text() {
        let filter = BasicQualityFilter::default();

        let result = filter.denoise_text("  hello    world  \n\n  ").unwrap();
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_syllable_counting() {
        let filter = BasicQualityFilter::default();

        assert_eq!(filter.count_syllables("cat"), 1);
        assert_eq!(filter.count_syllables("hello"), 2);
        assert_eq!(filter.count_syllables("beautiful"), 3);
        assert_eq!(filter.count_syllables("pronunciation"), 5);
    }

    #[test]
    fn test_morphology_analysis() {
        let filter = BasicQualityFilter::default();

        let noun_morph = filter.analyze_morphology("cats", &PosTag::Noun).unwrap();
        assert_eq!(noun_morph.get("category"), Some(&"noun".to_string()));
        assert_eq!(noun_morph.get("number"), Some(&"plural".to_string()));

        let verb_morph = filter.analyze_morphology("walked", &PosTag::Verb).unwrap();
        assert_eq!(verb_morph.get("category"), Some(&"verb".to_string()));
        assert_eq!(verb_morph.get("tense"), Some(&"past".to_string()));
    }
}

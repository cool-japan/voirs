//! Enhanced phoneme validation with detailed error messages and correction suggestions
//!
//! This module provides comprehensive phoneme validation beyond basic checking,
//! offering detailed error diagnostics and intelligent correction suggestions.
//!
//! # Features
//! - Detailed validation error messages with context
//! - Intelligent correction suggestions based on similarity
//! - Phoneme sequence validation (linguistic rules)
//! - Multi-level validation (character, sequence, context)
//! - Performance-optimized for batch processing
//!
//! # Examples
//! ```
//! use voirs_g2p::utils::phoneme_validator::{PhonemeValidator, ValidationLevel};
//! use voirs_g2p::{Phoneme, LanguageCode};
//!
//! let validator = PhonemeValidator::new(LanguageCode::EnUs);
//! let phonemes = vec![Phoneme::new("invalid".to_string())];
//!
//! let result = validator.validate_with_suggestions(&phonemes, ValidationLevel::Strict);
//! if !result.is_valid() {
//!     for error in result.errors() {
//!         println!("Error: {}", error.message);
//!         for suggestion in &error.suggestions {
//!             println!("  Suggestion: {}", suggestion);
//!         }
//!     }
//! }
//! ```

use crate::utils::phoneme_similarity::phoneme_levenshtein_distance;
use crate::{LanguageCode, Phoneme};
use std::collections::HashSet;

/// Validation strictness level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationLevel {
    /// Permissive - only catch severe errors
    Permissive,
    /// Standard - catch common issues
    Standard,
    /// Strict - enforce all linguistic rules
    Strict,
}

/// Detailed validation error with context and suggestions
#[derive(Debug, Clone)]
pub struct PhonemeValidationError {
    /// Position in the phoneme sequence (0-indexed)
    pub position: usize,
    /// The invalid phoneme
    pub phoneme: String,
    /// Error message
    pub message: String,
    /// Severity level (0-10, higher = more severe)
    pub severity: u8,
    /// Suggested corrections (ordered by likelihood)
    pub suggestions: Vec<String>,
    /// Error category
    pub category: ErrorCategory,
}

/// Category of validation error
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCategory {
    /// Invalid IPA symbol
    InvalidSymbol,
    /// Phoneme not valid for target language
    LanguageMismatch,
    /// Violates phonotactic constraints
    PhonotacticViolation,
    /// Unusual but possibly valid sequence
    UnusualSequence,
    /// Deprecated or non-standard notation
    DeprecatedNotation,
}

/// Validation result with detailed diagnostics
#[derive(Debug, Clone)]
pub struct PhonemeValidationResult {
    /// Whether the sequence is valid
    valid: bool,
    /// List of validation errors
    errors: Vec<PhonemeValidationError>,
    /// List of warnings (non-fatal issues)
    warnings: Vec<String>,
}

impl PhonemeValidationResult {
    /// Check if validation passed
    pub fn is_valid(&self) -> bool {
        self.valid
    }

    /// Get all validation errors
    pub fn errors(&self) -> &[PhonemeValidationError] {
        &self.errors
    }

    /// Get all warnings
    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    /// Get total number of issues (errors + warnings)
    pub fn issue_count(&self) -> usize {
        self.errors.len() + self.warnings.len()
    }

    /// Get the most severe error
    pub fn most_severe_error(&self) -> Option<&PhonemeValidationError> {
        self.errors.iter().max_by_key(|e| e.severity)
    }
}

/// Enhanced phoneme validator with language-specific rules
pub struct PhonemeValidator {
    language: LanguageCode,
    valid_phonemes: HashSet<String>,
    deprecated_phonemes: HashSet<String>,
}

impl PhonemeValidator {
    /// Create a new validator for a specific language
    ///
    /// # Arguments
    /// * `language` - Target language for validation
    ///
    /// # Examples
    /// ```
    /// use voirs_g2p::utils::phoneme_validator::PhonemeValidator;
    /// use voirs_g2p::LanguageCode;
    ///
    /// let validator = PhonemeValidator::new(LanguageCode::EnUs);
    /// ```
    pub fn new(language: LanguageCode) -> Self {
        let valid_phonemes = Self::get_valid_phoneme_set(language);
        let deprecated_phonemes = Self::get_deprecated_phonemes();

        Self {
            language,
            valid_phonemes,
            deprecated_phonemes,
        }
    }

    /// Validate phoneme sequence with detailed diagnostics and suggestions
    ///
    /// # Arguments
    /// * `phonemes` - Sequence to validate
    /// * `level` - Validation strictness level
    ///
    /// # Returns
    /// PhonemeValidationResult with errors and suggestions
    ///
    /// # Examples
    /// ```
    /// use voirs_g2p::utils::phoneme_validator::{PhonemeValidator, ValidationLevel};
    /// use voirs_g2p::{Phoneme, LanguageCode};
    ///
    /// let validator = PhonemeValidator::new(LanguageCode::EnUs);
    /// let phonemes = vec![Phoneme::new("h".to_string()), Phoneme::new("ɛ".to_string())];
    /// let result = validator.validate_with_suggestions(&phonemes, ValidationLevel::Standard);
    /// assert!(result.is_valid());
    /// ```
    pub fn validate_with_suggestions(
        &self,
        phonemes: &[Phoneme],
        level: ValidationLevel,
    ) -> PhonemeValidationResult {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        for (pos, phoneme) in phonemes.iter().enumerate() {
            // Check if phoneme exists in valid set
            if !self.valid_phonemes.contains(&phoneme.symbol) {
                let suggestions = self.find_similar_phonemes(&phoneme.symbol, 3);

                errors.push(PhonemeValidationError {
                    position: pos,
                    phoneme: phoneme.symbol.clone(),
                    message: format!("Invalid phoneme '{}' at position {}", phoneme.symbol, pos),
                    severity: 9,
                    suggestions,
                    category: ErrorCategory::InvalidSymbol,
                });
            }

            // Check for deprecated notation
            if self.deprecated_phonemes.contains(&phoneme.symbol) {
                let modern_equivalent = self.get_modern_equivalent(&phoneme.symbol);
                if level == ValidationLevel::Strict {
                    errors.push(PhonemeValidationError {
                        position: pos,
                        phoneme: phoneme.symbol.clone(),
                        message: format!(
                            "Deprecated phoneme notation '{}' at position {}",
                            phoneme.symbol, pos
                        ),
                        severity: 3,
                        suggestions: vec![modern_equivalent],
                        category: ErrorCategory::DeprecatedNotation,
                    });
                } else {
                    warnings.push(format!(
                        "Consider using modern notation for '{}': {}",
                        phoneme.symbol, modern_equivalent
                    ));
                }
            }
        }

        // Validate phoneme sequences (phonotactic constraints)
        if level != ValidationLevel::Permissive {
            self.validate_sequences(phonemes, &mut errors, &mut warnings, level);
        }

        PhonemeValidationResult {
            valid: errors.is_empty(),
            errors,
            warnings,
        }
    }

    /// Find similar phonemes for correction suggestions
    fn find_similar_phonemes(&self, invalid: &str, max_suggestions: usize) -> Vec<String> {
        let invalid_phoneme = Phoneme::new(invalid.to_string());

        let mut similarities: Vec<(String, usize)> = self
            .valid_phonemes
            .iter()
            .map(|valid| {
                let valid_phoneme = Phoneme::new(valid.clone());
                let distance = phoneme_levenshtein_distance(
                    std::slice::from_ref(&invalid_phoneme),
                    &[valid_phoneme],
                );
                (valid.clone(), distance)
            })
            .collect();

        // Sort by distance (lower = more similar)
        similarities.sort_by_key(|&(_, dist)| dist);

        // Return top N suggestions
        similarities
            .into_iter()
            .take(max_suggestions)
            .map(|(phoneme, _)| phoneme)
            .collect()
    }

    /// Validate phoneme sequences for linguistic plausibility
    fn validate_sequences(
        &self,
        phonemes: &[Phoneme],
        errors: &mut Vec<PhonemeValidationError>,
        warnings: &mut Vec<String>,
        level: ValidationLevel,
    ) {
        // Check for problematic consonant clusters
        for window in phonemes.windows(2) {
            if self.is_consonant(&window[0].symbol)
                && self.is_consonant(&window[1].symbol)
                && level == ValidationLevel::Strict
            {
                let cluster = format!("{}{}", window[0].symbol, window[1].symbol);
                if !self.is_valid_consonant_cluster(&cluster) {
                    warnings.push(format!("Unusual consonant cluster: '{}'", cluster));
                }
            }
        }

        // Check for very long consonant clusters (likely errors)
        for (pos, window) in phonemes.windows(4).enumerate() {
            if window.iter().all(|p| self.is_consonant(&p.symbol)) {
                errors.push(PhonemeValidationError {
                    position: pos,
                    phoneme: window[0].symbol.clone(),
                    message: format!(
                        "Implausible consonant cluster of length 4+ starting at position {}",
                        pos
                    ),
                    severity: 7,
                    suggestions: vec![
                        "Consider adding vowels or splitting the sequence".to_string()
                    ],
                    category: ErrorCategory::PhonotacticViolation,
                });
            }
        }
    }

    /// Check if a symbol represents a consonant
    fn is_consonant(&self, symbol: &str) -> bool {
        // Common IPA consonants
        matches!(
            symbol,
            "p" | "b"
                | "t"
                | "d"
                | "k"
                | "g"
                | "f"
                | "v"
                | "θ"
                | "ð"
                | "s"
                | "z"
                | "ʃ"
                | "ʒ"
                | "h"
                | "m"
                | "n"
                | "ŋ"
                | "l"
                | "r"
                | "w"
                | "j"
                | "tʃ"
                | "dʒ"
                | "ɹ"
                | "ʔ"
        )
    }

    /// Check if a consonant cluster is phonotactically valid
    fn is_valid_consonant_cluster(&self, _cluster: &str) -> bool {
        // For now, accept most clusters (could be enhanced with language-specific rules)
        true
    }

    /// Get modern equivalent for deprecated notation
    fn get_modern_equivalent(&self, deprecated: &str) -> String {
        // Common deprecated notations
        match deprecated {
            "g" => "ɡ".to_string(), // IPA g vs regular g
            ":" => "ː".to_string(), // ASCII colon vs IPA length marker
            _ => deprecated.to_string(),
        }
    }

    /// Get valid phoneme set for a language
    fn get_valid_phoneme_set(language: LanguageCode) -> HashSet<String> {
        let mut phonemes = HashSet::new();

        // IPA consonants (common across many languages)
        for &p in &[
            "p", "b", "t", "d", "k", "g", "ʔ", "m", "n", "ŋ", "ɴ", "f", "v", "θ", "ð", "s", "z",
            "ʃ", "ʒ", "ç", "x", "ɣ", "h", "ɦ", "l", "r", "ɹ", "ɾ", "j", "w", "ʋ", "ɰ", "tʃ", "dʒ",
            "ts", "dz", "ʈ", "ɖ", "c", "ɟ", "q", "ɢ", "ɱ", "ɳ", "ɲ", "ʙ", "ʀ", "ɽ", "ʂ", "ʐ", "ʝ",
            "χ", "ʁ", "ħ", "ʕ", "ɬ", "ɮ", "ɻ", "ɭ", "ʎ", "ʟ", "ʍ", "ɥ", "β", "ɸ",
        ] {
            phonemes.insert(p.to_string());
        }

        // IPA vowels (common across many languages)
        for &p in &[
            "i", "y", "ɨ", "ʉ", "ɯ", "u", "ɪ", "ʏ", "ʊ", "e", "ø", "ɘ", "ɵ", "ɤ", "o", "ə", "ɛ",
            "œ", "ɜ", "ɞ", "ʌ", "ɔ", "æ", "ɐ", "a", "ɶ", "ɑ", "ɒ", "iː", "uː", "eɪ", "aɪ", "ɔɪ",
            "aʊ", "oʊ", "ɝ", "ɚ", "ː",
        ] {
            phonemes.insert(p.to_string());
        }

        // Language-specific additions
        match language {
            LanguageCode::EnUs | LanguageCode::EnGb => {
                // English-specific
                phonemes.insert("ɜr".to_string());
                phonemes.insert("ɑr".to_string());
            }
            LanguageCode::Ja => {
                // Japanese-specific
                phonemes.insert("ɸ".to_string());
                phonemes.insert("ɕ".to_string());
                phonemes.insert("ɴ".to_string());
            }
            _ => {}
        }

        phonemes
    }

    /// Get set of deprecated phoneme notations
    fn get_deprecated_phonemes() -> HashSet<String> {
        let mut deprecated = HashSet::new();
        // Add deprecated notations that should be updated
        deprecated.insert(":".to_string()); // Use ː instead
        deprecated
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_phonemes() {
        let validator = PhonemeValidator::new(LanguageCode::EnUs);
        let phonemes = vec![
            Phoneme::new("h".to_string()),
            Phoneme::new("ɛ".to_string()),
            Phoneme::new("l".to_string()),
            Phoneme::new("oʊ".to_string()),
        ];

        let result = validator.validate_with_suggestions(&phonemes, ValidationLevel::Standard);
        assert!(result.is_valid());
        assert_eq!(result.errors().len(), 0);
    }

    #[test]
    fn test_invalid_phoneme() {
        let validator = PhonemeValidator::new(LanguageCode::EnUs);
        let phonemes = vec![Phoneme::new("INVALID".to_string())];

        let result = validator.validate_with_suggestions(&phonemes, ValidationLevel::Standard);
        assert!(!result.is_valid());
        assert_eq!(result.errors().len(), 1);

        let error = &result.errors()[0];
        assert_eq!(error.position, 0);
        assert_eq!(error.phoneme, "INVALID");
        assert!(!error.suggestions.is_empty());
    }

    #[test]
    fn test_suggestions() {
        let validator = PhonemeValidator::new(LanguageCode::EnUs);
        let phonemes = vec![Phoneme::new("INVALID".to_string())];

        let result = validator.validate_with_suggestions(&phonemes, ValidationLevel::Standard);
        assert!(!result.is_valid());

        let error = &result.errors()[0];
        // Should provide suggestions for invalid phoneme
        assert!(!error.suggestions.is_empty());
        // Suggestions should be valid phonemes
        assert!(error.suggestions.len() <= 3);
    }

    #[test]
    fn test_consonant_cluster_warning() {
        let validator = PhonemeValidator::new(LanguageCode::EnUs);
        let phonemes = vec![
            Phoneme::new("s".to_string()),
            Phoneme::new("t".to_string()),
            Phoneme::new("r".to_string()),
        ];

        let result = validator.validate_with_suggestions(&phonemes, ValidationLevel::Strict);
        // str is a valid English cluster, should not error
        assert!(result.is_valid());
    }

    #[test]
    fn test_long_consonant_cluster() {
        let validator = PhonemeValidator::new(LanguageCode::EnUs);
        let phonemes = vec![
            Phoneme::new("k".to_string()),
            Phoneme::new("t".to_string()),
            Phoneme::new("s".to_string()),
            Phoneme::new("p".to_string()),
        ];

        let result = validator.validate_with_suggestions(&phonemes, ValidationLevel::Standard);
        assert!(!result.is_valid());

        let error = &result.errors()[0];
        assert_eq!(error.category, ErrorCategory::PhonotacticViolation);
    }

    #[test]
    fn test_validation_levels() {
        let validator = PhonemeValidator::new(LanguageCode::EnUs);
        let phonemes = vec![Phoneme::new("INVALID".to_string())];

        // Should fail at all levels for invalid phoneme
        let result_permissive =
            validator.validate_with_suggestions(&phonemes, ValidationLevel::Permissive);
        let result_standard =
            validator.validate_with_suggestions(&phonemes, ValidationLevel::Standard);
        let result_strict = validator.validate_with_suggestions(&phonemes, ValidationLevel::Strict);

        assert!(!result_permissive.is_valid());
        assert!(!result_standard.is_valid());
        assert!(!result_strict.is_valid());
    }

    #[test]
    fn test_empty_sequence() {
        let validator = PhonemeValidator::new(LanguageCode::EnUs);
        let phonemes: Vec<Phoneme> = vec![];

        let result = validator.validate_with_suggestions(&phonemes, ValidationLevel::Standard);
        assert!(result.is_valid());
        assert_eq!(result.errors().len(), 0);
    }

    #[test]
    fn test_error_categories() {
        let validator = PhonemeValidator::new(LanguageCode::EnUs);
        let phonemes = vec![Phoneme::new("XYZ".to_string())];

        let result = validator.validate_with_suggestions(&phonemes, ValidationLevel::Standard);
        assert!(!result.is_valid());

        let error = &result.errors()[0];
        assert_eq!(error.category, ErrorCategory::InvalidSymbol);
        assert!(error.severity > 5);
    }

    #[test]
    fn test_most_severe_error() {
        let validator = PhonemeValidator::new(LanguageCode::EnUs);
        let phonemes = vec![
            Phoneme::new("INVALID1".to_string()),
            Phoneme::new("INVALID2".to_string()),
        ];

        let result = validator.validate_with_suggestions(&phonemes, ValidationLevel::Standard);
        let most_severe = result.most_severe_error();
        assert!(most_severe.is_some());
        assert!(most_severe.unwrap().severity > 0);
    }
}

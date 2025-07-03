//! Utility functions for G2P processing.

use crate::{LanguageCode, Phoneme, Result};
use crate::preprocessing::{TextPreprocessor, PreprocessingConfig};

/// Text preprocessing utilities
pub fn preprocess_text(text: &str, language: LanguageCode) -> Result<String> {
    let preprocessor = TextPreprocessor::new(language);
    preprocessor.preprocess(text)
}

/// Text preprocessing with custom configuration
pub fn preprocess_text_with_config(text: &str, language: LanguageCode, config: PreprocessingConfig) -> Result<String> {
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
pub fn postprocess_phonemes(phonemes: Vec<Phoneme>, language: LanguageCode) -> Vec<Phoneme> {
    // TODO: Implement language-specific phoneme post-processing
    // - Stress assignment
    // - Syllable boundary detection
    // - Duration prediction
    
    let _ = language; // Suppress unused warning
    phonemes
}

/// Validate phoneme sequence
pub fn validate_phonemes(phonemes: &[Phoneme], language: LanguageCode) -> bool {
    // TODO: Implement phoneme sequence validation
    // - Valid phoneme inventory check
    // - Phonotactic constraints
    // - Syllable structure validation
    
    let _ = (phonemes, language); // Suppress unused warnings
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preprocess_text() {
        let result = preprocess_text("Hello, World! 123", LanguageCode::EnUs).unwrap();
        assert!(!result.is_empty());
        // The new preprocessing should expand numbers
        assert!(result.contains("one hundred twenty three"));
    }

    #[test]
    fn test_preprocess_text_simple() {
        let result = preprocess_text_simple("Hello, World! 123", LanguageCode::EnUs);
        assert_eq!(result, "hello world ");
    }

    #[test]
    fn test_validate_phonemes() {
        let phonemes = vec![
            Phoneme::new("h"),
            Phoneme::new("ɛ"),
            Phoneme::new("l"),
            Phoneme::new("oʊ"),
        ];
        
        assert!(validate_phonemes(&phonemes, LanguageCode::EnUs));
    }
}
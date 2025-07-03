//! Text preprocessing for G2P conversion.

use crate::{LanguageCode, Result};
use std::collections::HashMap;
use std::sync::LazyLock;

pub mod unicode;
pub mod numbers;
pub mod text;

/// Main text preprocessor
pub struct TextPreprocessor {
    language: LanguageCode,
    config: PreprocessingConfig,
}

/// Configuration for text preprocessing
#[derive(Debug, Clone)]
pub struct PreprocessingConfig {
    /// Enable Unicode normalization
    pub unicode_normalization: bool,
    /// Enable number expansion
    pub expand_numbers: bool,
    /// Enable abbreviation expansion
    pub expand_abbreviations: bool,
    /// Enable currency expansion
    pub expand_currency: bool,
    /// Enable date/time expansion
    pub expand_datetime: bool,
    /// Enable URL handling
    pub handle_urls: bool,
    /// Enable punctuation removal
    pub remove_punctuation: bool,
}

impl Default for PreprocessingConfig {
    fn default() -> Self {
        Self {
            unicode_normalization: true,
            expand_numbers: true,
            expand_abbreviations: true,
            expand_currency: true,
            expand_datetime: true,
            handle_urls: true,
            remove_punctuation: false,
        }
    }
}

impl TextPreprocessor {
    /// Create new text preprocessor for language
    pub fn new(language: LanguageCode) -> Self {
        Self {
            language,
            config: PreprocessingConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(language: LanguageCode, config: PreprocessingConfig) -> Self {
        Self {
            language,
            config,
        }
    }

    /// Preprocess text for G2P conversion
    pub fn preprocess(&self, text: &str) -> Result<String> {
        let mut result = text.to_string();
        
        // Unicode normalization
        if self.config.unicode_normalization {
            result = unicode::normalize_text(&result)?;
        }
        
        // Number expansion
        if self.config.expand_numbers {
            result = numbers::expand_numbers(&result, self.language)?;
        }
        
        // Abbreviation expansion
        if self.config.expand_abbreviations {
            result = text::expand_abbreviations(&result, self.language)?;
        }
        
        // Currency expansion
        if self.config.expand_currency {
            result = text::expand_currency(&result, self.language)?;
        }
        
        // Date/time expansion
        if self.config.expand_datetime {
            result = text::expand_datetime(&result, self.language)?;
        }
        
        // URL handling
        if self.config.handle_urls {
            result = text::handle_urls(&result, self.language)?;
        }
        
        // Punctuation removal
        if self.config.remove_punctuation {
            result = text::remove_punctuation(&result);
        }
        
        Ok(result)
    }
}

/// Common abbreviations for different languages
static ABBREVIATIONS: LazyLock<HashMap<LanguageCode, HashMap<&'static str, &'static str>>> = LazyLock::new(|| {
    let mut map = HashMap::new();
    
    // English abbreviations
    let mut en_abbrevs = HashMap::new();
    en_abbrevs.insert("Dr.", "Doctor");
    en_abbrevs.insert("Mr.", "Mister");
    en_abbrevs.insert("Mrs.", "Missus");
    en_abbrevs.insert("Ms.", "Miss");
    en_abbrevs.insert("Prof.", "Professor");
    en_abbrevs.insert("U.S.A.", "United States of America");
    en_abbrevs.insert("U.K.", "United Kingdom");
    en_abbrevs.insert("etc.", "etcetera");
    en_abbrevs.insert("vs.", "versus");
    en_abbrevs.insert("Ave.", "Avenue");
    en_abbrevs.insert("St.", "Street");
    en_abbrevs.insert("Blvd.", "Boulevard");
    en_abbrevs.insert("Rd.", "Road");
    en_abbrevs.insert("Corp.", "Corporation");
    en_abbrevs.insert("Inc.", "Incorporated");
    en_abbrevs.insert("Ltd.", "Limited");
    en_abbrevs.insert("Co.", "Company");
    
    map.insert(LanguageCode::EnUs, en_abbrevs.clone());
    map.insert(LanguageCode::EnGb, en_abbrevs);
    
    // German abbreviations
    let mut de_abbrevs = HashMap::new();
    de_abbrevs.insert("Dr.", "Doktor");
    de_abbrevs.insert("Prof.", "Professor");
    de_abbrevs.insert("z.B.", "zum Beispiel");
    de_abbrevs.insert("usw.", "und so weiter");
    de_abbrevs.insert("bzw.", "beziehungsweise");
    
    map.insert(LanguageCode::De, de_abbrevs);
    
    // French abbreviations
    let mut fr_abbrevs = HashMap::new();
    fr_abbrevs.insert("Dr.", "Docteur");
    fr_abbrevs.insert("M.", "Monsieur");
    fr_abbrevs.insert("Mme", "Madame");
    fr_abbrevs.insert("Mlle", "Mademoiselle");
    fr_abbrevs.insert("Prof.", "Professeur");
    fr_abbrevs.insert("etc.", "et cetera");
    
    map.insert(LanguageCode::Fr, fr_abbrevs);
    
    // Spanish abbreviations
    let mut es_abbrevs = HashMap::new();
    es_abbrevs.insert("Dr.", "Doctor");
    es_abbrevs.insert("Dra.", "Doctora");
    es_abbrevs.insert("Prof.", "Profesor");
    es_abbrevs.insert("Sr.", "Señor");
    es_abbrevs.insert("Sra.", "Señora");
    es_abbrevs.insert("Srta.", "Señorita");
    es_abbrevs.insert("etc.", "etcétera");
    
    map.insert(LanguageCode::Es, es_abbrevs);
    
    map
});

/// Get abbreviations for language
pub fn get_abbreviations(language: LanguageCode) -> Option<&'static HashMap<&'static str, &'static str>> {
    ABBREVIATIONS.get(&language)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preprocessor_creation() {
        let preprocessor = TextPreprocessor::new(LanguageCode::EnUs);
        assert!(preprocessor.config.unicode_normalization);
        assert!(preprocessor.config.expand_numbers);
    }

    #[test]
    fn test_abbreviations() {
        let abbrevs = get_abbreviations(LanguageCode::EnUs).unwrap();
        assert_eq!(abbrevs.get("Dr."), Some(&"Doctor"));
        assert_eq!(abbrevs.get("U.S.A."), Some(&"United States of America"));
    }

    #[test]
    fn test_custom_config() {
        let mut config = PreprocessingConfig::default();
        config.expand_numbers = false;
        config.remove_punctuation = true;
        
        let preprocessor = TextPreprocessor::with_config(LanguageCode::EnUs, config);
        assert!(!preprocessor.config.expand_numbers);
        assert!(preprocessor.config.remove_punctuation);
    }
}
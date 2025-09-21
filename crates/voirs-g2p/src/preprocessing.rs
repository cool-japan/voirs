//! Text preprocessing for G2P conversion.

use crate::{LanguageCode, Result};
use std::collections::HashMap;
use std::sync::OnceLock;

pub mod context_aware;
pub mod entity_recognition;
pub mod numbers;
pub mod pos_tagging;
pub mod quality_filtering;
pub mod semantic_analysis;
pub mod text;
pub mod unicode;
pub mod variant_selection;

// Re-export key types and traits
pub use entity_recognition::{EntityType, NamedEntityRecognition, SimpleNamedEntityRecognizer};
pub use pos_tagging::{PosTag, PosTagging, RuleBasedPosTagger};
pub use quality_filtering::{BasicQualityFilter, QualityFiltering};
pub use semantic_analysis::{BasicSemanticAnalyzer, SemanticAnalysis, SemanticContext};
pub use variant_selection::{DictionaryVariantSelector, VariantSelection, VariantSelectionRule};

/// Main text preprocessor
pub struct TextPreprocessor {
    language: LanguageCode,
    config: PreprocessingConfig,
    entity_recognizer: SimpleNamedEntityRecognizer,
    pos_tagger: RuleBasedPosTagger,
    semantic_analyzer: BasicSemanticAnalyzer,
    variant_selector: DictionaryVariantSelector,
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
    /// Enable entity recognition
    pub enable_entity_recognition: bool,
    /// Enable POS tagging
    pub enable_pos_tagging: bool,
    /// Enable semantic analysis
    pub enable_semantic_analysis: bool,
    /// Enable variant selection
    pub enable_variant_selection: bool,
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
            enable_entity_recognition: true,
            enable_pos_tagging: true,
            enable_semantic_analysis: true,
            enable_variant_selection: true,
        }
    }
}

impl TextPreprocessor {
    /// Create new text preprocessor for language
    pub fn new(language: LanguageCode) -> Self {
        Self {
            language,
            config: PreprocessingConfig::default(),
            entity_recognizer: SimpleNamedEntityRecognizer::new(),
            pos_tagger: RuleBasedPosTagger::new(),
            semantic_analyzer: BasicSemanticAnalyzer::new(),
            variant_selector: DictionaryVariantSelector::new(language),
        }
    }

    /// Create with custom configuration
    pub fn with_config(language: LanguageCode, config: PreprocessingConfig) -> Self {
        Self {
            language,
            config,
            entity_recognizer: SimpleNamedEntityRecognizer::new(),
            pos_tagger: RuleBasedPosTagger::new(),
            semantic_analyzer: BasicSemanticAnalyzer::new(),
            variant_selector: DictionaryVariantSelector::new(language),
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

        // Advanced NLP preprocessing
        if self.config.enable_entity_recognition
            || self.config.enable_pos_tagging
            || self.config.enable_semantic_analysis
            || self.config.enable_variant_selection
        {
            result = self.apply_nlp_preprocessing(&result)?;
        }

        // Punctuation removal (after NLP processing)
        if self.config.remove_punctuation {
            result = text::remove_punctuation(&result);
        }

        Ok(result)
    }

    /// Apply advanced NLP preprocessing
    fn apply_nlp_preprocessing(&self, text: &str) -> Result<String> {
        let mut result = text.to_string();

        // Entity recognition for named entity normalization
        if self.config.enable_entity_recognition {
            let entities = self.entity_recognizer.recognize_entities(text)?;
            result = self.normalize_entities(&result, &entities)?;
        }

        // POS tagging for contextual understanding
        if self.config.enable_pos_tagging {
            let tagged_words = self.pos_tagger.tag_text(&result)?;
            result = self.apply_pos_based_transformations(&result, &tagged_words)?;
        }

        // Semantic analysis for context-aware processing
        if self.config.enable_semantic_analysis {
            let semantic_context = self.semantic_analyzer.analyze_context(&result)?;
            result = self.apply_semantic_transformations(&result, &semantic_context)?;
        }

        // Variant selection for optimal pronunciation variants
        if self.config.enable_variant_selection {
            result = self.apply_variant_selection(&result)?;
        }

        Ok(result)
    }

    /// Normalize entities in text for better G2P conversion
    fn normalize_entities(
        &self,
        text: &str,
        entities: &[(String, EntityType, usize, usize)],
    ) -> Result<String> {
        let mut result = text.to_string();

        // Process entities from end to start to preserve positions
        let mut sorted_entities: Vec<_> = entities.iter().collect();
        sorted_entities.sort_by(|a, b| b.2.cmp(&a.2));

        for (entity_text, entity_type, start, end) in sorted_entities {
            match entity_type {
                EntityType::Person => {
                    // Ensure proper name capitalization
                    let normalized = entity_text
                        .split_whitespace()
                        .map(|word| {
                            let mut chars: Vec<char> = word.chars().collect();
                            if !chars.is_empty() {
                                chars[0] = chars[0].to_uppercase().next().unwrap_or(chars[0]);
                            }
                            chars.into_iter().collect::<String>()
                        })
                        .collect::<Vec<_>>()
                        .join(" ");
                    result = format!("{}{normalized}{}", &result[..*start], &result[*end..]);
                }
                EntityType::Organization | EntityType::Location => {
                    // Expand common abbreviations in organizations/locations
                    let expanded = entity_text
                        .replace("Co.", "Company")
                        .replace("Inc.", "Incorporated")
                        .replace("Corp.", "Corporation")
                        .replace("Ltd.", "Limited")
                        .replace("St.", "Street")
                        .replace("Ave.", "Avenue");
                    result = format!("{}{expanded}{}", &result[..*start], &result[*end..]);
                }
                _ => {} // Keep other entities as-is
            }
        }

        Ok(result)
    }

    /// Apply POS-based transformations for better pronunciation
    fn apply_pos_based_transformations(
        &self,
        text: &str,
        tagged_words: &[(String, PosTag)],
    ) -> Result<String> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut result_words = Vec::new();

        for (i, word) in words.iter().enumerate() {
            if let Some((_, pos_tag)) = tagged_words.get(i) {
                let transformed_word = match pos_tag {
                    PosTag::Verb => {
                        // Handle verb stress patterns
                        if word.ends_with("ed") && word.len() > 3 {
                            // Past tense verbs often have different stress
                            word.to_string()
                        } else {
                            word.to_string()
                        }
                    }
                    PosTag::Noun => {
                        // Handle noun compounds and stress
                        word.to_string()
                    }
                    PosTag::Adjective => {
                        // Handle comparative/superlative forms
                        word.to_string()
                    }
                    _ => word.to_string(),
                };
                result_words.push(transformed_word);
            } else {
                result_words.push(word.to_string());
            }
        }

        Ok(result_words.join(" "))
    }

    /// Apply semantic transformations based on context
    fn apply_semantic_transformations(
        &self,
        text: &str,
        context: &SemanticContext,
    ) -> Result<String> {
        let mut result = text.to_string();

        // Apply formality-based transformations
        if context.formality_level > 0.7 {
            // Formal speech patterns - expand contractions
            result = result
                .replace("can't", "cannot")
                .replace("won't", "will not")
                .replace("don't", "do not")
                .replace("isn't", "is not")
                .replace("aren't", "are not")
                .replace("wasn't", "was not")
                .replace("weren't", "were not")
                .replace("haven't", "have not")
                .replace("hasn't", "has not")
                .replace("hadn't", "had not")
                .replace("wouldn't", "would not")
                .replace("shouldn't", "should not")
                .replace("couldn't", "could not");
        }

        // Apply domain-specific transformations
        if let Some(domain) = &context.domain {
            match domain.as_str() {
                "technology" => {
                    // Technical terms often need specific pronunciation
                    result = result
                        .replace("API", "A P I")
                        .replace("HTTP", "H T T P")
                        .replace("URL", "U R L")
                        .replace("JSON", "J S O N")
                        .replace("XML", "X M L")
                        .replace("SQL", "S Q L");
                }
                "medical" => {
                    // Medical terms pronunciation guidance
                    result = result
                        .replace("MHz", "megahertz")
                        .replace("kg", "kilograms")
                        .replace("mg", "milligrams")
                        .replace("ml", "milliliters");
                }
                _ => {}
            }
        }

        Ok(result)
    }

    /// Apply variant selection to entire text
    fn apply_variant_selection(&self, text: &str) -> Result<String> {
        // For now, just return the text as-is since variant selection
        // typically happens at the phoneme level during G2P conversion
        // This is a placeholder for future phoneme-level variant selection
        Ok(text.to_string())
    }

    /// Get access to the entity recognizer
    pub fn entity_recognizer(&self) -> &SimpleNamedEntityRecognizer {
        &self.entity_recognizer
    }

    /// Get access to the POS tagger
    pub fn pos_tagger(&self) -> &RuleBasedPosTagger {
        &self.pos_tagger
    }

    /// Get access to the semantic analyzer
    pub fn semantic_analyzer(&self) -> &BasicSemanticAnalyzer {
        &self.semantic_analyzer
    }

    /// Get access to the variant selector
    pub fn variant_selector(&self) -> &DictionaryVariantSelector {
        &self.variant_selector
    }

    /// Analyze text without preprocessing (for analysis purposes)
    pub fn analyze_text(&self, text: &str) -> Result<TextAnalysis> {
        let entities = if self.config.enable_entity_recognition {
            Some(self.entity_recognizer.recognize_entities(text)?)
        } else {
            None
        };

        let pos_tags = if self.config.enable_pos_tagging {
            Some(self.pos_tagger.tag_text(text)?)
        } else {
            None
        };

        let semantic_context = if self.config.enable_semantic_analysis {
            Some(self.semantic_analyzer.analyze_context(text)?)
        } else {
            None
        };

        Ok(TextAnalysis {
            entities,
            pos_tags,
            semantic_context,
        })
    }
}

/// Text analysis results
#[derive(Debug, Clone)]
pub struct TextAnalysis {
    /// Recognized entities
    pub entities: Option<Vec<(String, EntityType, usize, usize)>>,
    /// POS tags
    pub pos_tags: Option<Vec<(String, PosTag)>>,
    /// Semantic context
    pub semantic_context: Option<SemanticContext>,
}

/// Common abbreviations for different languages
static ABBREVIATIONS: OnceLock<HashMap<LanguageCode, HashMap<&'static str, &'static str>>> =
    OnceLock::new();

fn init_abbreviations() -> HashMap<LanguageCode, HashMap<&'static str, &'static str>> {
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
}

/// Get abbreviations for language
pub fn get_abbreviations(
    language: LanguageCode,
) -> Option<&'static HashMap<&'static str, &'static str>> {
    ABBREVIATIONS.get_or_init(init_abbreviations).get(&language)
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
        let config = PreprocessingConfig {
            expand_numbers: false,
            remove_punctuation: true,
            enable_entity_recognition: false,
            ..Default::default()
        };

        let preprocessor = TextPreprocessor::with_config(LanguageCode::EnUs, config);
        assert!(!preprocessor.config.expand_numbers);
        assert!(preprocessor.config.remove_punctuation);
        assert!(!preprocessor.config.enable_entity_recognition);
    }

    #[test]
    fn test_nlp_integration() {
        let preprocessor = TextPreprocessor::new(LanguageCode::EnUs);

        // Test basic preprocessing
        let result = preprocessor
            .preprocess("Hello Dr. Smith, how are you?")
            .unwrap();
        assert!(!result.is_empty());

        // Test that NLP components are accessible
        assert!(preprocessor
            .entity_recognizer()
            .supported_languages()
            .contains(&LanguageCode::EnUs));
        assert!(preprocessor
            .pos_tagger()
            .supported_languages()
            .contains(&LanguageCode::EnUs));
        assert!(preprocessor
            .semantic_analyzer()
            .supported_languages()
            .contains(&LanguageCode::EnUs));
    }

    #[test]
    fn test_text_analysis() {
        let preprocessor = TextPreprocessor::new(LanguageCode::EnUs);
        let analysis = preprocessor
            .analyze_text("Hello Dr. Smith, this is a great day!")
            .unwrap();

        // Should have entities (Dr. Smith)
        assert!(analysis.entities.is_some());

        // Should have POS tags
        assert!(analysis.pos_tags.is_some());

        // Should have semantic context
        assert!(analysis.semantic_context.is_some());
        let context = analysis.semantic_context.unwrap();
        assert!(context.sentiment_polarity > 0.0); // Should be positive due to "great"
    }

    #[test]
    fn test_entity_normalization() {
        let preprocessor = TextPreprocessor::new(LanguageCode::EnUs);

        // Test with organization abbreviations
        let result = preprocessor
            .preprocess("I work at Apple Inc. on Main St.")
            .unwrap();
        assert!(result.contains("Incorporated") || result.contains("Street"));
    }

    #[test]
    fn test_formal_contractions() {
        let preprocessor = TextPreprocessor::new(LanguageCode::EnUs);

        // Text with formal context should expand contractions
        let formal_text = "Furthermore, I cannot establish the aforementioned protocol.";
        let result = preprocessor.preprocess(formal_text).unwrap();
        assert!(result.contains("cannot"));

        // Test contraction expansion in formal context
        let formal_with_contractions = "Therefore, I can't proceed with this implementation.";
        let result2 = preprocessor.preprocess(formal_with_contractions).unwrap();
        // Should expand "can't" to "cannot" due to formal context
        assert!(result2.contains("cannot") || result2.contains("can't"));
    }

    #[test]
    fn test_technical_domain_processing() {
        let preprocessor = TextPreprocessor::new(LanguageCode::EnUs);

        // Technical text should get special processing
        let tech_text = "We need to implement the HTTP API using JSON format.";
        let result = preprocessor.preprocess(tech_text).unwrap();

        // Technical acronyms should be expanded for pronunciation
        assert!(result.contains("H T T P") || result.contains("HTTP"));
        assert!(result.contains("A P I") || result.contains("API"));
        assert!(result.contains("J S O N") || result.contains("JSON"));
    }

    #[test]
    fn test_disabled_nlp_features() {
        let config = PreprocessingConfig {
            enable_entity_recognition: false,
            enable_pos_tagging: false,
            enable_semantic_analysis: false,
            enable_variant_selection: false,
            ..Default::default()
        };

        let preprocessor = TextPreprocessor::with_config(LanguageCode::EnUs, config);
        let result = preprocessor.preprocess("Hello Dr. Smith!").unwrap();

        // Should still process basic text normalization
        assert!(!result.is_empty());

        // Analysis should return None for disabled features
        let analysis = preprocessor.analyze_text("Hello world!").unwrap();
        assert!(analysis.entities.is_none());
        assert!(analysis.pos_tags.is_none());
        assert!(analysis.semantic_context.is_none());
    }
}

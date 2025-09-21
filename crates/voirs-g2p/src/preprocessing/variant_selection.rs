//! Pronunciation variant selection functionality for context-aware preprocessing.

use crate::{LanguageCode, Phoneme, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Trait for pronunciation variant selection
pub trait VariantSelection: Send + Sync {
    /// Select best pronunciation variant for a word in context
    fn select_variant(
        &self,
        word: &str,
        context: &[String],
        available_variants: &[Vec<Phoneme>],
    ) -> Result<Option<Vec<Phoneme>>>;

    /// Get all available variants for a word
    fn get_variants(&self, word: &str) -> Result<Vec<Vec<Phoneme>>>;

    /// Check if word has multiple pronunciation variants
    fn has_variants(&self, word: &str) -> bool;

    /// Get supported languages
    fn supported_languages(&self) -> Vec<LanguageCode>;
}

/// Dictionary-based variant selector
#[derive(Debug, Clone)]
pub struct DictionaryVariantSelector {
    /// Pronunciation variants dictionary
    pub variants_dict: HashMap<String, Vec<Vec<Phoneme>>>,
    /// Context-based selection rules
    pub selection_rules: Vec<VariantSelectionRule>,
    /// Regional preferences
    pub regional_preferences: HashMap<String, HashMap<String, f32>>,
    /// Language-specific settings
    pub language: LanguageCode,
}

/// Variant selection rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantSelectionRule {
    /// Word pattern (regex or exact match)
    pub word_pattern: String,
    /// Context conditions that must be met
    pub context_conditions: Vec<String>,
    /// Preferred variant index
    pub preferred_variant: usize,
    /// Selection confidence (0.0 to 1.0)
    pub confidence: f32,
    /// Rule description
    pub description: Option<String>,
}

/// Context information for variant selection
#[derive(Debug, Clone)]
pub struct VariantSelectionContext {
    /// Preceding words
    pub preceding_words: Vec<String>,
    /// Following words
    pub following_words: Vec<String>,
    /// Part of speech context
    pub pos_context: Vec<String>,
    /// Semantic domain
    pub semantic_domain: Option<String>,
    /// Formality level
    pub formality_level: f32,
}

impl VariantSelection for DictionaryVariantSelector {
    fn select_variant(
        &self,
        word: &str,
        context: &[String],
        available_variants: &[Vec<Phoneme>],
    ) -> Result<Option<Vec<Phoneme>>> {
        if available_variants.is_empty() {
            return Ok(None);
        }

        // If only one variant, return it
        if available_variants.len() == 1 {
            return Ok(Some(available_variants[0].clone()));
        }

        // Apply selection rules
        for rule in &self.selection_rules {
            if self.rule_matches(word, context, rule)
                && rule.preferred_variant < available_variants.len()
            {
                return Ok(Some(available_variants[rule.preferred_variant].clone()));
            }
        }

        // Apply regional preferences
        if let Some(regional_prefs) = self.regional_preferences.get(word) {
            let mut best_score = 0.0;
            let mut best_variant_idx = 0;

            for (i, variant) in available_variants.iter().enumerate() {
                let variant_key = self.phonemes_to_key(variant);
                if let Some(score) = regional_prefs.get(&variant_key) {
                    if *score > best_score {
                        best_score = *score;
                        best_variant_idx = i;
                    }
                }
            }

            if best_score > 0.0 {
                return Ok(Some(available_variants[best_variant_idx].clone()));
            }
        }

        // Default to first variant
        Ok(Some(available_variants[0].clone()))
    }

    fn get_variants(&self, word: &str) -> Result<Vec<Vec<Phoneme>>> {
        Ok(self.variants_dict.get(word).cloned().unwrap_or_default())
    }

    fn has_variants(&self, word: &str) -> bool {
        self.variants_dict
            .get(word)
            .map(|variants| variants.len() > 1)
            .unwrap_or(false)
    }

    fn supported_languages(&self) -> Vec<LanguageCode> {
        vec![self.language]
    }
}

impl DictionaryVariantSelector {
    /// Create a new dictionary variant selector
    pub fn new(language: LanguageCode) -> Self {
        Self {
            variants_dict: HashMap::new(),
            selection_rules: Vec::new(),
            regional_preferences: HashMap::new(),
            language,
        }
    }

    /// Add pronunciation variants for a word
    pub fn add_variants(&mut self, word: String, variants: Vec<Vec<Phoneme>>) {
        self.variants_dict.insert(word, variants);
    }

    /// Add a single variant for a word
    pub fn add_variant(&mut self, word: String, variant: Vec<Phoneme>) {
        self.variants_dict.entry(word).or_default().push(variant);
    }

    /// Add a selection rule
    pub fn add_selection_rule(&mut self, rule: VariantSelectionRule) {
        self.selection_rules.push(rule);
    }

    /// Add regional preference for a word variant
    pub fn add_regional_preference(&mut self, word: String, variant_key: String, score: f32) {
        self.regional_preferences
            .entry(word)
            .or_default()
            .insert(variant_key, score.clamp(0.0, 1.0));
    }

    /// Load variants from a dictionary file or resource
    pub fn load_variants_dictionary(&mut self, variants: HashMap<String, Vec<Vec<Phoneme>>>) {
        self.variants_dict.extend(variants);
    }

    /// Check if a rule matches the current context
    fn rule_matches(&self, word: &str, context: &[String], rule: &VariantSelectionRule) -> bool {
        // Check if word matches the pattern
        if !self.word_matches_pattern(word, &rule.word_pattern) {
            return false;
        }

        // Check context conditions
        for condition in &rule.context_conditions {
            if !self.context_matches_condition(context, condition) {
                return false;
            }
        }

        true
    }

    /// Check if word matches a pattern
    fn word_matches_pattern(&self, word: &str, pattern: &str) -> bool {
        // Simple exact match for now, could be extended to regex
        word.eq_ignore_ascii_case(pattern)
    }

    /// Check if context matches a condition
    fn context_matches_condition(&self, context: &[String], condition: &str) -> bool {
        // Simple contains check for now
        context
            .iter()
            .any(|ctx_word| ctx_word.eq_ignore_ascii_case(condition))
    }

    /// Convert phonemes to a string key for regional preferences
    fn phonemes_to_key(&self, phonemes: &[Phoneme]) -> String {
        phonemes
            .iter()
            .map(|p| format!("{p:?}"))
            .collect::<Vec<_>>()
            .join("-")
    }

    /// Get statistics about the variant dictionary
    pub fn get_statistics(&self) -> VariantStatistics {
        let total_words = self.variants_dict.len();
        let words_with_multiple_variants = self
            .variants_dict
            .values()
            .filter(|variants| variants.len() > 1)
            .count();

        let max_variants = self
            .variants_dict
            .values()
            .map(|variants| variants.len())
            .max()
            .unwrap_or(0);

        let total_rules = self.selection_rules.len();
        let total_regional_prefs = self.regional_preferences.len();

        VariantStatistics {
            total_words,
            words_with_multiple_variants,
            max_variants,
            total_rules,
            total_regional_preferences: total_regional_prefs,
        }
    }

    /// Remove variants for a word
    pub fn remove_variants(&mut self, word: &str) {
        self.variants_dict.remove(word);
    }

    /// Clear all variants
    pub fn clear_variants(&mut self) {
        self.variants_dict.clear();
    }

    /// Get words that have multiple variants
    pub fn get_ambiguous_words(&self) -> Vec<String> {
        self.variants_dict
            .iter()
            .filter(|(_, variants)| variants.len() > 1)
            .map(|(word, _)| word.clone())
            .collect()
    }
}

impl Default for DictionaryVariantSelector {
    fn default() -> Self {
        Self::new(LanguageCode::EnUs)
    }
}

/// Statistics about variant selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantStatistics {
    /// Total number of words in dictionary
    pub total_words: usize,
    /// Number of words with multiple variants
    pub words_with_multiple_variants: usize,
    /// Maximum number of variants for any word
    pub max_variants: usize,
    /// Total number of selection rules
    pub total_rules: usize,
    /// Total number of regional preferences
    pub total_regional_preferences: usize,
}

impl VariantSelectionRule {
    /// Create a new variant selection rule
    pub fn new(
        word_pattern: String,
        context_conditions: Vec<String>,
        preferred_variant: usize,
        confidence: f32,
    ) -> Self {
        Self {
            word_pattern,
            context_conditions,
            preferred_variant,
            confidence: confidence.clamp(0.0, 1.0),
            description: None,
        }
    }

    /// Create a rule with description
    pub fn with_description(mut self, description: String) -> Self {
        self.description = Some(description);
        self
    }

    /// Update the confidence score
    pub fn set_confidence(&mut self, confidence: f32) {
        self.confidence = confidence.clamp(0.0, 1.0);
    }
}

impl VariantSelectionContext {
    /// Create a new variant selection context
    pub fn new() -> Self {
        Self {
            preceding_words: Vec::new(),
            following_words: Vec::new(),
            pos_context: Vec::new(),
            semantic_domain: None,
            formality_level: 0.5,
        }
    }

    /// Set preceding words
    pub fn with_preceding_words(mut self, words: Vec<String>) -> Self {
        self.preceding_words = words;
        self
    }

    /// Set following words
    pub fn with_following_words(mut self, words: Vec<String>) -> Self {
        self.following_words = words;
        self
    }

    /// Set POS context
    pub fn with_pos_context(mut self, pos: Vec<String>) -> Self {
        self.pos_context = pos;
        self
    }

    /// Set semantic domain
    pub fn with_semantic_domain(mut self, domain: String) -> Self {
        self.semantic_domain = Some(domain);
        self
    }

    /// Set formality level
    pub fn with_formality_level(mut self, level: f32) -> Self {
        self.formality_level = level.clamp(0.0, 1.0);
        self
    }
}

impl Default for VariantSelectionContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dictionary_variant_selector_creation() {
        let selector = DictionaryVariantSelector::new(LanguageCode::EnUs);
        assert_eq!(selector.language, LanguageCode::EnUs);
        assert!(selector.variants_dict.is_empty());
        assert!(selector.selection_rules.is_empty());
    }

    #[test]
    fn test_add_variants() {
        let mut selector = DictionaryVariantSelector::new(LanguageCode::EnUs);
        let variants = vec![vec![Phoneme::new("A")], vec![Phoneme::new("E")]];

        selector.add_variants("test".to_string(), variants.clone());
        assert_eq!(selector.get_variants("test").unwrap(), variants);
        assert!(selector.has_variants("test"));
    }

    #[test]
    fn test_variant_selection_rule() {
        let rule =
            VariantSelectionRule::new("test".to_string(), vec!["context".to_string()], 0, 0.8)
                .with_description("Test rule".to_string());

        assert_eq!(rule.word_pattern, "test");
        assert_eq!(rule.confidence, 0.8);
        assert_eq!(rule.description, Some("Test rule".to_string()));
    }

    #[test]
    fn test_select_variant_single() {
        let selector = DictionaryVariantSelector::new(LanguageCode::EnUs);
        let variants = vec![vec![Phoneme::new("A")]];

        let result = selector.select_variant("test", &[], &variants).unwrap();
        assert_eq!(result, Some(vec![Phoneme::new("A")]));
    }

    #[test]
    fn test_select_variant_empty() {
        let selector = DictionaryVariantSelector::new(LanguageCode::EnUs);
        let variants = vec![];

        let result = selector.select_variant("test", &[], &variants).unwrap();
        assert_eq!(result, None);
    }

    #[test]
    fn test_variant_statistics() {
        let mut selector = DictionaryVariantSelector::new(LanguageCode::EnUs);
        selector.add_variants("word1".to_string(), vec![vec![Phoneme::new("A")]]);
        selector.add_variants(
            "word2".to_string(),
            vec![vec![Phoneme::new("A")], vec![Phoneme::new("E")]],
        );

        let stats = selector.get_statistics();
        assert_eq!(stats.total_words, 2);
        assert_eq!(stats.words_with_multiple_variants, 1);
        assert_eq!(stats.max_variants, 2);
    }

    #[test]
    fn test_ambiguous_words() {
        let mut selector = DictionaryVariantSelector::new(LanguageCode::EnUs);
        selector.add_variants("single".to_string(), vec![vec![Phoneme::new("A")]]);
        selector.add_variants(
            "multiple".to_string(),
            vec![vec![Phoneme::new("A")], vec![Phoneme::new("E")]],
        );

        let ambiguous = selector.get_ambiguous_words();
        assert_eq!(ambiguous.len(), 1);
        assert!(ambiguous.contains(&"multiple".to_string()));
    }

    #[test]
    fn test_variant_selection_context() {
        let context = VariantSelectionContext::new()
            .with_preceding_words(vec!["the".to_string()])
            .with_following_words(vec!["is".to_string()])
            .with_formality_level(0.8);

        assert_eq!(context.preceding_words, vec!["the".to_string()]);
        assert_eq!(context.following_words, vec!["is".to_string()]);
        assert_eq!(context.formality_level, 0.8);
    }

    #[test]
    fn test_regional_preferences() {
        let mut selector = DictionaryVariantSelector::new(LanguageCode::EnUs);
        selector.add_regional_preference("test".to_string(), "variant1".to_string(), 0.8);

        assert!(selector.regional_preferences.contains_key("test"));
    }
}

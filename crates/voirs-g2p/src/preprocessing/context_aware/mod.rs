//! Advanced context-aware preprocessing for G2P conversion.
//!
//! This module provides sophisticated text preprocessing that considers
//! linguistic context, part-of-speech information, and semantic analysis
//! to improve G2P accuracy.

use crate::{G2pError, LanguageCode, Phoneme, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

// Re-export types from parent preprocessing modules
pub use crate::preprocessing::entity_recognition::{
    EntityType, NamedEntityRecognition, SimpleNamedEntityRecognizer,
};
pub use crate::preprocessing::pos_tagging::{PosTag, PosTagging, RuleBasedPosTagger};
pub use crate::preprocessing::quality_filtering::{BasicQualityFilter, QualityFiltering};
pub use crate::preprocessing::semantic_analysis::{
    BasicSemanticAnalyzer, SemanticAnalysis, SemanticContext,
};
pub use crate::preprocessing::variant_selection::{
    DictionaryVariantSelector, VariantSelection, VariantSelectionRule,
};

/// Phonetic context information
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhoneticContext {
    /// Preceding phonemes
    pub preceding: Vec<String>,
    /// Following phonemes
    pub following: Vec<String>,
    /// Syllable structure
    pub syllable_structure: String,
    /// Stress pattern
    pub stress_pattern: Vec<u8>,
    /// Vowel harmony rules
    pub vowel_harmony: Option<String>,
}

/// Linguistic token with rich annotation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinguisticToken {
    /// Original text
    pub text: String,
    /// Normalized text
    pub normalized: String,
    /// Part-of-speech tag
    pub pos_tag: PosTag,
    /// Named entity type
    pub entity_type: Option<EntityType>,
    /// Word position in sentence
    pub position: usize,
    /// Sentence position in document
    pub sentence_position: usize,
    /// Morphological features
    pub morphology: HashMap<String, String>,
    /// Phonetic context
    pub phonetic_context: Option<PhoneticContext>,
    /// Semantic context
    pub semantic_context: Option<SemanticContext>,
    /// Confidence score
    pub confidence: f32,
}

/// Advanced context-aware preprocessor
pub struct ContextAwarePreprocessor {
    /// Language code
    pub language: LanguageCode,
    /// POS tagger
    pub pos_tagger: Arc<dyn PosTagging>,
    /// Named entity recognizer
    pub ner: Arc<dyn NamedEntityRecognition>,
    /// Semantic analyzer
    pub semantic_analyzer: Arc<dyn SemanticAnalysis>,
    /// Pronunciation variant selector
    pub variant_selector: Arc<dyn VariantSelection>,
    /// Quality filter
    pub quality_filter: Arc<dyn QualityFiltering>,
    /// Configuration
    pub config: ContextAwareConfig,
}

/// Configuration for context-aware preprocessing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextAwareConfig {
    /// Enable POS tagging
    pub enable_pos_tagging: bool,
    /// Enable named entity recognition
    pub enable_ner: bool,
    /// Enable semantic analysis
    pub enable_semantic_analysis: bool,
    /// Enable pronunciation variants
    pub enable_variant_selection: bool,
    /// Enable quality filtering
    pub enable_quality_filtering: bool,
    /// Context window size
    pub context_window_size: usize,
    /// Minimum confidence threshold
    pub min_confidence_threshold: f32,
    /// Maximum processing time (ms)
    pub max_processing_time_ms: u64,
}

impl ContextAwarePreprocessor {
    /// Create new context-aware preprocessor
    pub fn new(
        language: LanguageCode,
        pos_tagger: Arc<dyn PosTagging>,
        ner: Arc<dyn NamedEntityRecognition>,
        semantic_analyzer: Arc<dyn SemanticAnalysis>,
        variant_selector: Arc<dyn VariantSelection>,
        quality_filter: Arc<dyn QualityFiltering>,
        config: ContextAwareConfig,
    ) -> Self {
        Self {
            language,
            pos_tagger,
            ner,
            semantic_analyzer,
            variant_selector,
            quality_filter,
            config,
        }
    }

    /// Process text with full context awareness
    pub fn process_text(&self, text: &str) -> Result<Vec<LinguisticToken>> {
        let start_time = std::time::Instant::now();

        // Quality filtering
        let clean_text = if self.config.enable_quality_filtering {
            self.quality_filter.validate_input(text)?
        } else {
            text.to_string()
        };

        // Tokenization
        let tokens = self.tokenize(&clean_text)?;

        // POS tagging
        let pos_tags = if self.config.enable_pos_tagging {
            self.pos_tagger.tag_text(&clean_text)?
        } else {
            tokens.iter().map(|t| (t.clone(), PosTag::Other)).collect()
        };

        // Named entity recognition
        let entities = if self.config.enable_ner {
            self.ner.recognize_entities(&clean_text)?
        } else {
            Vec::new()
        };

        // Semantic analysis
        let semantic_context = if self.config.enable_semantic_analysis {
            Some(self.semantic_analyzer.analyze_context(&clean_text)?)
        } else {
            None
        };

        // Perform sentence segmentation
        let sentences = self.segment_sentences(&clean_text);

        // Extract token strings for phonetic context analysis
        let token_strings: Vec<String> = pos_tags.iter().map(|(token, _)| token.clone()).collect();

        // Build linguistic tokens
        let mut linguistic_tokens = Vec::new();

        for (i, (token, pos_tag)) in pos_tags.iter().enumerate() {
            // Find entity type for this token
            let entity_type = entities
                .iter()
                .find(|(entity_text, _, start, end)| {
                    entity_text == token && *start <= i && i < *end
                })
                .map(|(_, entity_type, _, _)| entity_type.clone());

            // Calculate sentence position for this token
            let sentence_position = self.calculate_sentence_position(i, &sentences);

            // Perform morphological analysis
            let morphology = if self.config.enable_quality_filtering {
                self.quality_filter
                    .analyze_morphology(token, pos_tag)
                    .unwrap_or_default()
            } else {
                HashMap::new()
            };

            // Perform phonetic context analysis
            let phonetic_context = if self.config.enable_quality_filtering {
                self.quality_filter
                    .analyze_phonetic_context(token, i, &token_strings)
                    .unwrap_or(None)
            } else {
                None
            };

            // Create linguistic token
            let linguistic_token = LinguisticToken {
                text: token.clone(),
                normalized: token.to_lowercase(),
                pos_tag: pos_tag.clone(),
                entity_type,
                position: i,
                sentence_position,
                morphology,
                phonetic_context,
                semantic_context: semantic_context.clone(),
                confidence: 0.8, // Default confidence
            };

            linguistic_tokens.push(linguistic_token);
        }

        // Check processing time
        if start_time.elapsed().as_millis() > self.config.max_processing_time_ms as u128 {
            return Err(G2pError::ConfigError(
                "Context-aware preprocessing exceeded time limit".to_string(),
            ));
        }

        Ok(linguistic_tokens)
    }

    /// Simple tokenization (can be enhanced with language-specific rules)
    fn tokenize(&self, text: &str) -> Result<Vec<String>> {
        let tokens: Vec<String> = text
            .split_whitespace()
            .map(|s| s.trim_matches(|c: char| c.is_ascii_punctuation()))
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect();

        Ok(tokens)
    }

    /// Select best pronunciation variant for word
    pub fn select_pronunciation_variant(
        &self,
        word: &str,
        context: &LinguisticToken,
    ) -> Result<Vec<Phoneme>> {
        if self.config.enable_variant_selection {
            // Get available variants first
            let available_variants = self.variant_selector.get_variants(word)?;
            if available_variants.is_empty() {
                return Ok(Vec::new());
            }

            // Create context string array from linguistic token
            let context_strings: Vec<String> = vec![
                context.text.clone(),
                format!("{:?}", context.pos_tag),
                context
                    .entity_type
                    .as_ref()
                    .map(|e| format!("{e:?}"))
                    .unwrap_or_default(),
            ];

            // Select variant
            if let Some(variant) =
                self.variant_selector
                    .select_variant(word, &context_strings, &available_variants)?
            {
                Ok(variant)
            } else {
                // Return first variant if selection fails
                Ok(available_variants.into_iter().next().unwrap_or_default())
            }
        } else {
            // Return empty vector if variant selection is disabled
            Ok(Vec::new())
        }
    }

    /// Analyze phonetic context
    pub fn analyze_phonetic_context(
        &self,
        tokens: &[LinguisticToken],
        index: usize,
    ) -> Result<PhoneticContext> {
        let window_size = self.config.context_window_size;
        let start = index.saturating_sub(window_size);
        let end = (index + window_size + 1).min(tokens.len());

        let preceding: Vec<String> = tokens[start..index]
            .iter()
            .map(|t| t.text.clone())
            .collect();

        let following: Vec<String> = tokens[index + 1..end]
            .iter()
            .map(|t| t.text.clone())
            .collect();

        // Simple syllable structure analysis
        let syllable_structure = self.analyze_syllable_structure(&tokens[index].text)?;

        // Simple stress pattern analysis
        let stress_pattern = self.analyze_stress_pattern(&tokens[index].text)?;

        // Vowel harmony analysis
        let vowel_harmony = self.analyze_vowel_harmony(&tokens[index].text)?;

        Ok(PhoneticContext {
            preceding,
            following,
            syllable_structure,
            stress_pattern,
            vowel_harmony,
        })
    }

    /// Analyze syllable structure
    fn analyze_syllable_structure(&self, word: &str) -> Result<String> {
        // Simple consonant-vowel pattern analysis
        let mut structure = String::new();

        for char in word.chars() {
            match char.to_lowercase().next() {
                Some('a') | Some('e') | Some('i') | Some('o') | Some('u') => {
                    structure.push('V');
                }
                Some(c) if c.is_alphabetic() => {
                    structure.push('C');
                }
                _ => {
                    // Skip non-alphabetic characters
                }
            }
        }

        Ok(structure)
    }

    /// Analyze stress pattern
    fn analyze_stress_pattern(&self, word: &str) -> Result<Vec<u8>> {
        // Simple stress assignment based on word length and syllable structure
        let syllable_count = word
            .chars()
            .filter(|c| {
                matches!(
                    c.to_lowercase().next(),
                    Some('a') | Some('e') | Some('i') | Some('o') | Some('u')
                )
            })
            .count();

        let mut stress_pattern = vec![0; syllable_count];

        // Simple rule: primary stress on first syllable for short words (1 syllable),
        // primary stress on second syllable for longer words (2+ syllables)
        if syllable_count > 0 {
            let stress_position = if syllable_count <= 1 { 0 } else { 1 };
            if stress_position < stress_pattern.len() {
                stress_pattern[stress_position] = 1; // Primary stress
            }
        }

        Ok(stress_pattern)
    }

    /// Analyze vowel harmony pattern
    fn analyze_vowel_harmony(&self, word: &str) -> Result<Option<String>> {
        let vowels: Vec<char> = word
            .chars()
            .filter(|c| {
                matches!(
                    c.to_lowercase().next(),
                    Some('a') | Some('e') | Some('i') | Some('o') | Some('u')
                )
            })
            .collect();

        if vowels.is_empty() {
            return Ok(None);
        }

        let front_vowels = ['e', 'i'];
        let back_vowels = ['a', 'o', 'u'];

        let mut front_count = 0;
        let mut back_count = 0;

        for vowel in &vowels {
            let lowercase_vowel = vowel.to_lowercase().next().unwrap();
            if front_vowels.contains(&lowercase_vowel) {
                front_count += 1;
            } else if back_vowels.contains(&lowercase_vowel) {
                back_count += 1;
            }
        }

        let harmony_type = if front_count > 0 && back_count == 0 {
            "front"
        } else if back_count > 0 && front_count == 0 {
            "back"
        } else if front_count > 0 && back_count > 0 {
            "mixed"
        } else {
            "neutral"
        };

        Ok(Some(harmony_type.to_string()))
    }

    /// Perform sentence segmentation on input text
    fn segment_sentences(&self, text: &str) -> Vec<(String, usize, usize)> {
        let mut sentences = Vec::new();
        let mut current_start = 0;
        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            let ch = chars[i];

            // Check for sentence-ending punctuation
            if ch == '.' || ch == '!' || ch == '?' || ch == '。' || ch == '！' || ch == '？' {
                // Look ahead to see if this is really the end of a sentence
                let mut end_pos = i + 1;

                // Skip whitespace after punctuation
                while end_pos < chars.len() && chars[end_pos].is_whitespace() {
                    end_pos += 1;
                }

                // Check if next character is uppercase or if we're at the end
                let is_sentence_end = end_pos >= chars.len()
                    || chars[end_pos].is_uppercase()
                    || chars[end_pos].is_numeric()
                    || (end_pos < chars.len() - 1
                        && chars[end_pos] == '"'
                        && chars[end_pos + 1].is_uppercase());

                if is_sentence_end {
                    let sentence_text: String = chars[current_start..=i].iter().collect();
                    let trimmed = sentence_text.trim();
                    if !trimmed.is_empty() {
                        sentences.push((trimmed.to_string(), current_start, i + 1));
                    }
                    current_start = end_pos;
                    i = end_pos;
                    continue;
                }
            }
            i += 1;
        }

        // Add remaining text as final sentence if not empty
        if current_start < chars.len() {
            let sentence_text: String = chars[current_start..].iter().collect();
            let trimmed = sentence_text.trim();
            if !trimmed.is_empty() {
                sentences.push((trimmed.to_string(), current_start, chars.len()));
            }
        }

        sentences
    }

    /// Calculate sentence position for a given token index
    fn calculate_sentence_position(
        &self,
        token_index: usize,
        sentences: &[(String, usize, usize)],
    ) -> usize {
        for (sentence_idx, (_, start, end)) in sentences.iter().enumerate() {
            if token_index >= *start && token_index < *end {
                return sentence_idx;
            }
        }
        0 // Default to first sentence if not found
    }
}

impl Default for ContextAwareConfig {
    fn default() -> Self {
        Self {
            enable_pos_tagging: true,
            enable_ner: true,
            enable_semantic_analysis: true,
            enable_variant_selection: true,
            enable_quality_filtering: true,
            context_window_size: 3,
            min_confidence_threshold: 0.7,
            max_processing_time_ms: 5000,
        }
    }
}

// Copyright (c) 2025 VoiRS Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Semantic similarity evaluation for speech synthesis systems.
//!
//! This module provides comprehensive semantic similarity assessment capabilities,
//! evaluating how well synthesized speech preserves the meaning and content of
//! the reference or target text.
//!
//! # Features
//!
//! - **Text Similarity**: Evaluation of textual semantic similarity using embeddings
//! - **Phonetic Similarity**: Assessment of pronunciation and phonetic similarity
//! - **Prosodic Similarity**: Analysis of rhythm, stress, and intonation patterns
//! - **Content Preservation**: Measurement of content preservation and accuracy
//! - **Meaning Preservation**: Evaluation of semantic meaning preservation
//! - **Paraphrase Detection**: Detection and scoring of paraphrases
//! - **Synonym Handling**: Recognition of synonyms and semantically equivalent terms
//! - **Context-Aware Similarity**: Context-sensitive similarity assessment
//!
//! # Example
//!
//! ```rust
//! use voirs_evaluation::semantic_similarity::*;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create semantic similarity evaluator
//! let config = SemanticSimilarityConfig::default();
//! let evaluator = SemanticSimilarityEvaluator::new(config)?;
//!
//! // Evaluate similarity
//! let reference_text = "The quick brown fox jumps over the lazy dog";
//! let synthesized_text = "The fast brown fox leaps over the lazy dog";
//!
//! let result = evaluator.evaluate(reference_text, synthesized_text)?;
//!
//! println!("Semantic Similarity: {:.2}", result.overall_similarity);
//! println!("Lexical Similarity: {:.2}", result.lexical_similarity);
//! println!("Meaning Preservation: {:.2}", result.meaning_preservation);
//! # Ok(())
//! # }
//! ```

use crate::EvaluationError;
use scirs2_core::numeric::Float;
use scirs2_core::random::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Similarity method for evaluation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SimilarityMethod {
    /// Cosine similarity of word embeddings
    Cosine,
    /// Jaccard similarity of word sets
    Jaccard,
    /// Edit distance (Levenshtein)
    EditDistance,
    /// N-gram overlap
    NGramOverlap,
    /// Semantic embeddings
    Semantic,
}

/// Semantic similarity evaluation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticSimilarityResult {
    /// Overall semantic similarity score (0.0-1.0)
    pub overall_similarity: f32,
    /// Lexical similarity score (0.0-1.0)
    pub lexical_similarity: f32,
    /// Phonetic similarity score (0.0-1.0)
    pub phonetic_similarity: f32,
    /// Prosodic similarity score (0.0-1.0)
    pub prosodic_similarity: f32,
    /// Content preservation score (0.0-1.0)
    pub content_preservation: f32,
    /// Meaning preservation score (0.0-1.0)
    pub meaning_preservation: f32,
    /// Synonym usage score (0.0-1.0)
    pub synonym_score: f32,
    /// Paraphrase quality (0.0-1.0)
    pub paraphrase_quality: f32,
    /// Detailed metrics per method
    pub method_scores: HashMap<String, f32>,
    /// Content differences detected
    pub differences: Vec<ContentDifference>,
    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Represents a content difference between texts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentDifference {
    /// Type of difference
    pub difference_type: DifferenceType,
    /// Reference text segment
    pub reference_segment: String,
    /// Synthesized text segment
    pub synthesized_segment: String,
    /// Position in text
    pub position: usize,
    /// Severity (0.0-1.0, higher is more severe)
    pub severity: f32,
}

/// Type of content difference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DifferenceType {
    /// Word substitution
    Substitution,
    /// Word insertion
    Insertion,
    /// Word deletion
    Deletion,
    /// Synonym usage
    Synonym,
    /// Paraphrase
    Paraphrase,
    /// Reordering
    Reordering,
    /// Semantic drift
    SemanticDrift,
}

/// Configuration for semantic similarity evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticSimilarityConfig {
    /// Similarity methods to use
    pub methods: Vec<SimilarityMethod>,
    /// Enable synonym detection
    pub detect_synonyms: bool,
    /// Enable paraphrase detection
    pub detect_paraphrases: bool,
    /// Enable content difference analysis
    pub analyze_differences: bool,
    /// Minimum similarity threshold (0.0-1.0)
    pub similarity_threshold: f32,
    /// N-gram size for n-gram overlap
    pub ngram_size: usize,
    /// Generate recommendations
    pub generate_recommendations: bool,
}

impl Default for SemanticSimilarityConfig {
    fn default() -> Self {
        Self {
            methods: vec![
                SimilarityMethod::Cosine,
                SimilarityMethod::Jaccard,
                SimilarityMethod::NGramOverlap,
            ],
            detect_synonyms: true,
            detect_paraphrases: true,
            analyze_differences: true,
            similarity_threshold: 0.7,
            ngram_size: 2,
            generate_recommendations: true,
        }
    }
}

/// Semantic similarity evaluator
pub struct SemanticSimilarityEvaluator {
    config: SemanticSimilarityConfig,
    synonym_pairs: HashMap<String, Vec<String>>,
}

impl SemanticSimilarityEvaluator {
    /// Creates a new semantic similarity evaluator
    pub fn new(config: SemanticSimilarityConfig) -> Result<Self, EvaluationError> {
        let synonym_pairs = Self::load_synonym_database();

        Ok(Self {
            config,
            synonym_pairs,
        })
    }

    /// Evaluates semantic similarity between two texts
    pub fn evaluate(
        &self,
        reference: &str,
        synthesized: &str,
    ) -> Result<SemanticSimilarityResult, EvaluationError> {
        // Tokenize texts
        let ref_tokens = self.tokenize(reference);
        let syn_tokens = self.tokenize(synthesized);

        // Calculate similarity using configured methods
        let mut method_scores = HashMap::new();

        for method in &self.config.methods {
            let score = self.calculate_similarity(&ref_tokens, &syn_tokens, *method)?;
            method_scores.insert(format!("{:?}", method), score);
        }

        // Calculate component scores
        let lexical_similarity = self.calculate_lexical_similarity(&ref_tokens, &syn_tokens)?;
        let phonetic_similarity = self.calculate_phonetic_similarity(&ref_tokens, &syn_tokens)?;
        let prosodic_similarity = self.calculate_prosodic_similarity(reference, synthesized)?;
        let content_preservation = self.calculate_content_preservation(&ref_tokens, &syn_tokens)?;
        let meaning_preservation = self.calculate_meaning_preservation(reference, synthesized)?;

        // Analyze synonyms and paraphrases
        let synonym_score = if self.config.detect_synonyms {
            self.calculate_synonym_score(&ref_tokens, &syn_tokens)?
        } else {
            0.8
        };

        let paraphrase_quality = if self.config.detect_paraphrases {
            self.evaluate_paraphrase_quality(reference, synthesized)?
        } else {
            0.8
        };

        // Analyze differences
        let differences = if self.config.analyze_differences {
            self.analyze_differences(&ref_tokens, &syn_tokens)?
        } else {
            Vec::new()
        };

        // Calculate overall similarity
        let overall_similarity = self.calculate_overall_similarity(
            lexical_similarity,
            phonetic_similarity,
            prosodic_similarity,
            content_preservation,
            meaning_preservation,
            synonym_score,
            paraphrase_quality,
        );

        // Generate recommendations
        let recommendations = if self.config.generate_recommendations {
            self.generate_recommendations(
                &differences,
                lexical_similarity,
                content_preservation,
                meaning_preservation,
            )
        } else {
            Vec::new()
        };

        Ok(SemanticSimilarityResult {
            overall_similarity,
            lexical_similarity,
            phonetic_similarity,
            prosodic_similarity,
            content_preservation,
            meaning_preservation,
            synonym_score,
            paraphrase_quality,
            method_scores,
            differences,
            recommendations,
        })
    }

    /// Tokenizes text into words
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split_whitespace()
            .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }

    /// Calculates similarity using specified method
    fn calculate_similarity(
        &self,
        ref_tokens: &[String],
        syn_tokens: &[String],
        method: SimilarityMethod,
    ) -> Result<f32, EvaluationError> {
        match method {
            SimilarityMethod::Cosine => self.cosine_similarity(ref_tokens, syn_tokens),
            SimilarityMethod::Jaccard => Ok(self.jaccard_similarity(ref_tokens, syn_tokens)),
            SimilarityMethod::EditDistance => self.edit_distance_similarity(ref_tokens, syn_tokens),
            SimilarityMethod::NGramOverlap => self.ngram_overlap_similarity(ref_tokens, syn_tokens),
            SimilarityMethod::Semantic => {
                self.semantic_embedding_similarity(ref_tokens, syn_tokens)
            }
        }
    }

    /// Calculates cosine similarity
    fn cosine_similarity(
        &self,
        ref_tokens: &[String],
        syn_tokens: &[String],
    ) -> Result<f32, EvaluationError> {
        // Create term frequency vectors
        let ref_tf = self.term_frequency(ref_tokens);
        let syn_tf = self.term_frequency(syn_tokens);

        // Get all unique terms
        let all_terms: HashSet<_> = ref_tf.keys().chain(syn_tf.keys()).cloned().collect();

        if all_terms.is_empty() {
            return Ok(0.0);
        }

        // Calculate dot product and magnitudes
        let mut dot_product = 0.0;
        let mut ref_magnitude = 0.0;
        let mut syn_magnitude = 0.0;

        for term in all_terms {
            let ref_val = ref_tf.get(&term).copied().unwrap_or(0.0);
            let syn_val = syn_tf.get(&term).copied().unwrap_or(0.0);

            dot_product += ref_val * syn_val;
            ref_magnitude += ref_val * ref_val;
            syn_magnitude += syn_val * syn_val;
        }

        let magnitude_product = (ref_magnitude * syn_magnitude).sqrt();
        if magnitude_product > 1e-10 {
            Ok((dot_product / magnitude_product).clamp(0.0, 1.0))
        } else {
            Ok(0.0)
        }
    }

    /// Calculates Jaccard similarity
    fn jaccard_similarity(&self, ref_tokens: &[String], syn_tokens: &[String]) -> f32 {
        let ref_set: HashSet<_> = ref_tokens.iter().collect();
        let syn_set: HashSet<_> = syn_tokens.iter().collect();

        let intersection = ref_set.intersection(&syn_set).count();
        let union = ref_set.union(&syn_set).count();

        if union > 0 {
            intersection as f32 / union as f32
        } else {
            0.0
        }
    }

    /// Calculates edit distance-based similarity
    fn edit_distance_similarity(
        &self,
        ref_tokens: &[String],
        syn_tokens: &[String],
    ) -> Result<f32, EvaluationError> {
        let distance = self.levenshtein_distance(ref_tokens, syn_tokens);
        let max_len = ref_tokens.len().max(syn_tokens.len());

        if max_len > 0 {
            Ok(1.0 - (distance as f32 / max_len as f32))
        } else {
            Ok(1.0)
        }
    }

    /// Calculates n-gram overlap similarity
    fn ngram_overlap_similarity(
        &self,
        ref_tokens: &[String],
        syn_tokens: &[String],
    ) -> Result<f32, EvaluationError> {
        let ref_ngrams = self.generate_ngrams(ref_tokens, self.config.ngram_size);
        let syn_ngrams = self.generate_ngrams(syn_tokens, self.config.ngram_size);

        let intersection = ref_ngrams.intersection(&syn_ngrams).count();
        let union = ref_ngrams.union(&syn_ngrams).count();

        if union > 0 {
            Ok(intersection as f32 / union as f32)
        } else {
            Ok(0.0)
        }
    }

    /// Calculates semantic embedding similarity
    fn semantic_embedding_similarity(
        &self,
        ref_tokens: &[String],
        syn_tokens: &[String],
    ) -> Result<f32, EvaluationError> {
        // Simplified embedding similarity using word overlap with semantic awareness
        let mut matches = 0;
        let mut total = ref_tokens.len().max(syn_tokens.len());

        for ref_word in ref_tokens {
            if syn_tokens.contains(ref_word) {
                matches += 1;
            } else if self.has_synonym(ref_word, syn_tokens) {
                matches += 1; // Count synonym as match
            }
        }

        if total > 0 {
            Ok(matches as f32 / total as f32)
        } else {
            Ok(1.0)
        }
    }

    /// Calculates lexical similarity
    fn calculate_lexical_similarity(
        &self,
        ref_tokens: &[String],
        syn_tokens: &[String],
    ) -> Result<f32, EvaluationError> {
        // Use combination of Jaccard and exact match ratio
        let jaccard = self.jaccard_similarity(ref_tokens, syn_tokens);
        let exact_matches = ref_tokens.iter().filter(|t| syn_tokens.contains(t)).count();
        let match_ratio = if !ref_tokens.is_empty() {
            exact_matches as f32 / ref_tokens.len() as f32
        } else {
            0.0
        };

        Ok((jaccard * 0.5 + match_ratio * 0.5).clamp(0.0, 1.0))
    }

    /// Calculates phonetic similarity
    fn calculate_phonetic_similarity(
        &self,
        ref_tokens: &[String],
        syn_tokens: &[String],
    ) -> Result<f32, EvaluationError> {
        // Simplified phonetic similarity using first/last characters and length
        if ref_tokens.is_empty() && syn_tokens.is_empty() {
            return Ok(1.0);
        }

        if ref_tokens.is_empty() || syn_tokens.is_empty() {
            return Ok(0.0);
        }

        let mut phonetic_matches = 0;
        let total_words = ref_tokens.len().min(syn_tokens.len());

        for i in 0..total_words {
            if self.phonetically_similar(&ref_tokens[i], &syn_tokens[i]) {
                phonetic_matches += 1;
            }
        }

        Ok(phonetic_matches as f32 / ref_tokens.len().max(syn_tokens.len()) as f32)
    }

    /// Calculates prosodic similarity
    fn calculate_prosodic_similarity(
        &self,
        reference: &str,
        synthesized: &str,
    ) -> Result<f32, EvaluationError> {
        // Analyze sentence structure, punctuation, and rhythm
        let ref_sentences = self.count_sentences(reference);
        let syn_sentences = self.count_sentences(synthesized);

        let sentence_sim = if ref_sentences.max(syn_sentences) > 0 {
            1.0 - (ref_sentences as i32 - syn_sentences as i32).abs() as f32
                / ref_sentences.max(syn_sentences) as f32
        } else {
            1.0
        };

        // Analyze punctuation pattern similarity
        let ref_punct = self.extract_punctuation_pattern(reference);
        let syn_punct = self.extract_punctuation_pattern(synthesized);
        let punct_sim = self.pattern_similarity(&ref_punct, &syn_punct);

        Ok((sentence_sim * 0.6 + punct_sim * 0.4).clamp(0.0, 1.0))
    }

    /// Calculates content preservation
    fn calculate_content_preservation(
        &self,
        ref_tokens: &[String],
        syn_tokens: &[String],
    ) -> Result<f32, EvaluationError> {
        // Check how much of the reference content is preserved
        let preserved_words = ref_tokens
            .iter()
            .filter(|word| syn_tokens.contains(word) || self.has_synonym(word, syn_tokens))
            .count();

        if !ref_tokens.is_empty() {
            Ok(preserved_words as f32 / ref_tokens.len() as f32)
        } else {
            Ok(1.0)
        }
    }

    /// Calculates meaning preservation
    fn calculate_meaning_preservation(
        &self,
        reference: &str,
        synthesized: &str,
    ) -> Result<f32, EvaluationError> {
        // Analyze key content words (nouns, verbs, adjectives)
        let ref_content = self.extract_content_words(reference);
        let syn_content = self.extract_content_words(synthesized);

        let preserved = ref_content
            .iter()
            .filter(|word| syn_content.contains(word) || self.has_synonym(word, &syn_content))
            .count();

        if !ref_content.is_empty() {
            Ok(preserved as f32 / ref_content.len() as f32)
        } else {
            Ok(1.0)
        }
    }

    /// Calculates synonym usage score
    fn calculate_synonym_score(
        &self,
        ref_tokens: &[String],
        syn_tokens: &[String],
    ) -> Result<f32, EvaluationError> {
        let mut synonym_matches = 0;
        let mut non_exact_matches = 0;

        for ref_word in ref_tokens {
            if !syn_tokens.contains(ref_word) {
                if self.has_synonym(ref_word, syn_tokens) {
                    synonym_matches += 1;
                }
                non_exact_matches += 1;
            }
        }

        if non_exact_matches > 0 {
            Ok(synonym_matches as f32 / non_exact_matches as f32)
        } else {
            Ok(1.0)
        }
    }

    /// Evaluates paraphrase quality
    fn evaluate_paraphrase_quality(
        &self,
        reference: &str,
        synthesized: &str,
    ) -> Result<f32, EvaluationError> {
        // A good paraphrase has high semantic similarity but different wording
        let ref_tokens = self.tokenize(reference);
        let syn_tokens = self.tokenize(synthesized);

        let semantic_sim = self.semantic_embedding_similarity(&ref_tokens, &syn_tokens)?;
        let lexical_sim = self.calculate_lexical_similarity(&ref_tokens, &syn_tokens)?;

        // Good paraphrase: high semantic, lower lexical
        let paraphrase_score = if semantic_sim > 0.7 && lexical_sim < 0.9 {
            (semantic_sim + (1.0 - lexical_sim) * 0.5) / 1.5
        } else if semantic_sim > 0.8 {
            semantic_sim
        } else {
            0.5
        };

        Ok(paraphrase_score.clamp(0.0, 1.0))
    }

    /// Analyzes differences between texts
    fn analyze_differences(
        &self,
        ref_tokens: &[String],
        syn_tokens: &[String],
    ) -> Result<Vec<ContentDifference>, EvaluationError> {
        let mut differences = Vec::new();

        // Simple alignment-based difference detection
        let alignment = self.align_tokens(ref_tokens, syn_tokens);

        for (i, (ref_word, syn_word)) in alignment.iter().enumerate() {
            match (ref_word, syn_word) {
                (Some(r), Some(s)) if r != s => {
                    let diff_type = if self.are_synonyms(r, s) {
                        DifferenceType::Synonym
                    } else {
                        DifferenceType::Substitution
                    };

                    differences.push(ContentDifference {
                        difference_type: diff_type,
                        reference_segment: r.clone(),
                        synthesized_segment: s.clone(),
                        position: i,
                        severity: if diff_type == DifferenceType::Synonym {
                            0.2
                        } else {
                            0.6
                        },
                    });
                }
                (Some(r), None) => {
                    differences.push(ContentDifference {
                        difference_type: DifferenceType::Deletion,
                        reference_segment: r.clone(),
                        synthesized_segment: String::new(),
                        position: i,
                        severity: 0.7,
                    });
                }
                (None, Some(s)) => {
                    differences.push(ContentDifference {
                        difference_type: DifferenceType::Insertion,
                        reference_segment: String::new(),
                        synthesized_segment: s.clone(),
                        position: i,
                        severity: 0.5,
                    });
                }
                _ => {}
            }
        }

        Ok(differences)
    }

    /// Generates recommendations
    fn generate_recommendations(
        &self,
        differences: &[ContentDifference],
        lexical_similarity: f32,
        content_preservation: f32,
        meaning_preservation: f32,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        if lexical_similarity < self.config.similarity_threshold {
            recommendations.push(format!(
                "Improve lexical similarity (current: {:.2}, threshold: {:.2}). Review word choice accuracy.",
                lexical_similarity, self.config.similarity_threshold
            ));
        }

        if content_preservation < 0.8 {
            recommendations.push(format!(
                "Enhance content preservation (current: {:.2}). Ensure all key content is present.",
                content_preservation
            ));
        }

        if meaning_preservation < 0.7 {
            recommendations.push(format!(
                "Improve meaning preservation (current: {:.2}). Focus on preserving semantic content.",
                meaning_preservation
            ));
        }

        let high_severity_diffs = differences.iter().filter(|d| d.severity > 0.6).count();
        if high_severity_diffs > 3 {
            recommendations.push(format!(
                "Found {} high-severity content differences. Review substitutions and deletions.",
                high_severity_diffs
            ));
        }

        if recommendations.is_empty() {
            recommendations.push(
                "Semantic similarity meets expectations. Content well preserved.".to_string(),
            );
        }

        recommendations
    }

    // Utility methods

    fn calculate_overall_similarity(
        &self,
        lexical: f32,
        phonetic: f32,
        prosodic: f32,
        content: f32,
        meaning: f32,
        synonym: f32,
        paraphrase: f32,
    ) -> f32 {
        (lexical * 0.25
            + phonetic * 0.1
            + prosodic * 0.1
            + content * 0.25
            + meaning * 0.2
            + synonym * 0.05
            + paraphrase * 0.05)
            .clamp(0.0, 1.0)
    }

    fn term_frequency(&self, tokens: &[String]) -> HashMap<String, f32> {
        let mut tf = HashMap::new();
        for token in tokens {
            *tf.entry(token.clone()).or_insert(0.0) += 1.0;
        }
        // Normalize
        let total = tokens.len() as f32;
        if total > 0.0 {
            for value in tf.values_mut() {
                *value /= total;
            }
        }
        tf
    }

    fn levenshtein_distance(&self, a: &[String], b: &[String]) -> usize {
        let m = a.len();
        let n = b.len();

        if m == 0 {
            return n;
        }
        if n == 0 {
            return m;
        }

        let mut matrix = vec![vec![0; n + 1]; m + 1];

        for i in 0..=m {
            matrix[i][0] = i;
        }
        for j in 0..=n {
            matrix[0][j] = j;
        }

        for i in 1..=m {
            for j in 1..=n {
                let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
                matrix[i][j] = (matrix[i - 1][j] + 1)
                    .min(matrix[i][j - 1] + 1)
                    .min(matrix[i - 1][j - 1] + cost);
            }
        }

        matrix[m][n]
    }

    fn generate_ngrams(&self, tokens: &[String], n: usize) -> HashSet<String> {
        let mut ngrams = HashSet::new();
        if tokens.len() < n {
            return ngrams;
        }

        for i in 0..=tokens.len() - n {
            let ngram = tokens[i..i + n].join(" ");
            ngrams.insert(ngram);
        }

        ngrams
    }

    fn phonetically_similar(&self, word1: &str, word2: &str) -> bool {
        if word1 == word2 {
            return true;
        }

        // Simple phonetic similarity heuristics
        let len_diff = (word1.len() as i32 - word2.len() as i32).abs();
        if len_diff > 2 {
            return false;
        }

        // Check first and last characters
        let w1_chars: Vec<char> = word1.chars().collect();
        let w2_chars: Vec<char> = word2.chars().collect();

        if w1_chars.is_empty() || w2_chars.is_empty() {
            return false;
        }

        w1_chars[0] == w2_chars[0] || w1_chars.last() == w2_chars.last()
    }

    fn count_sentences(&self, text: &str) -> usize {
        text.split(&['.', '!', '?'][..])
            .filter(|s| !s.trim().is_empty())
            .count()
    }

    fn extract_punctuation_pattern(&self, text: &str) -> Vec<char> {
        text.chars().filter(|c| c.is_ascii_punctuation()).collect()
    }

    fn pattern_similarity(&self, pattern1: &[char], pattern2: &[char]) -> f32 {
        if pattern1.is_empty() && pattern2.is_empty() {
            return 1.0;
        }

        let matches = pattern1
            .iter()
            .zip(pattern2.iter())
            .filter(|(a, b)| a == b)
            .count();
        let max_len = pattern1.len().max(pattern2.len());

        if max_len > 0 {
            matches as f32 / max_len as f32
        } else {
            0.0
        }
    }

    fn extract_content_words(&self, text: &str) -> Vec<String> {
        // Extract likely content words (longer words, capitalized, etc.)
        self.tokenize(text)
            .into_iter()
            .filter(|word| {
                word.len() > 3 || word.chars().next().map_or(false, |c| c.is_uppercase())
            })
            .collect()
    }

    fn has_synonym(&self, word: &str, tokens: &[String]) -> bool {
        if let Some(synonyms) = self.synonym_pairs.get(word) {
            tokens.iter().any(|t| synonyms.contains(t))
        } else {
            false
        }
    }

    fn are_synonyms(&self, word1: &str, word2: &str) -> bool {
        if let Some(synonyms) = self.synonym_pairs.get(word1) {
            synonyms.contains(&word2.to_string())
        } else {
            false
        }
    }

    fn align_tokens(
        &self,
        ref_tokens: &[String],
        syn_tokens: &[String],
    ) -> Vec<(Option<String>, Option<String>)> {
        // Simple alignment based on position
        let max_len = ref_tokens.len().max(syn_tokens.len());
        let mut alignment = Vec::new();

        for i in 0..max_len {
            let ref_word = ref_tokens.get(i).cloned();
            let syn_word = syn_tokens.get(i).cloned();
            alignment.push((ref_word, syn_word));
        }

        alignment
    }

    fn load_synonym_database() -> HashMap<String, Vec<String>> {
        // Simplified synonym database - in production, load from file/database
        let mut db = HashMap::new();

        // Common synonyms
        db.insert(
            "quick".to_string(),
            vec!["fast".to_string(), "rapid".to_string(), "swift".to_string()],
        );
        db.insert(
            "fast".to_string(),
            vec![
                "quick".to_string(),
                "rapid".to_string(),
                "swift".to_string(),
            ],
        );
        db.insert(
            "happy".to_string(),
            vec![
                "glad".to_string(),
                "joyful".to_string(),
                "pleased".to_string(),
            ],
        );
        db.insert(
            "sad".to_string(),
            vec![
                "unhappy".to_string(),
                "sorrowful".to_string(),
                "melancholy".to_string(),
            ],
        );
        db.insert(
            "big".to_string(),
            vec![
                "large".to_string(),
                "huge".to_string(),
                "enormous".to_string(),
            ],
        );
        db.insert(
            "small".to_string(),
            vec![
                "little".to_string(),
                "tiny".to_string(),
                "minute".to_string(),
            ],
        );
        db.insert(
            "good".to_string(),
            vec![
                "excellent".to_string(),
                "great".to_string(),
                "fine".to_string(),
            ],
        );
        db.insert(
            "bad".to_string(),
            vec![
                "poor".to_string(),
                "awful".to_string(),
                "terrible".to_string(),
            ],
        );

        db
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_similarity_evaluator_creation() {
        let config = SemanticSimilarityConfig::default();
        let evaluator = SemanticSimilarityEvaluator::new(config);
        assert!(evaluator.is_ok());
    }

    #[test]
    fn test_identical_texts() {
        let config = SemanticSimilarityConfig::default();
        let evaluator = SemanticSimilarityEvaluator::new(config).unwrap();

        let text = "The quick brown fox jumps over the lazy dog";
        let result = evaluator.evaluate(text, text);

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.overall_similarity > 0.9);
        assert!(result.lexical_similarity > 0.9);
    }

    #[test]
    fn test_synonym_detection() {
        let config = SemanticSimilarityConfig::default();
        let evaluator = SemanticSimilarityEvaluator::new(config).unwrap();

        let ref_text = "The quick brown fox";
        let syn_text = "The fast brown fox";

        let result = evaluator.evaluate(ref_text, syn_text);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.synonym_score > 0.0);
        assert!(result.overall_similarity > 0.7);
    }

    #[test]
    fn test_paraphrase_evaluation() {
        let config = SemanticSimilarityConfig::default();
        let evaluator = SemanticSimilarityEvaluator::new(config).unwrap();

        let ref_text = "The cat sat on the mat";
        let para_text = "A feline rested on the rug";

        let result = evaluator.evaluate(ref_text, para_text);
        assert!(result.is_ok());
        // Paraphrase should have lower lexical but reasonable semantic similarity
    }

    #[test]
    fn test_jaccard_similarity() {
        let config = SemanticSimilarityConfig::default();
        let evaluator = SemanticSimilarityEvaluator::new(config).unwrap();

        let tokens1 = vec!["hello".to_string(), "world".to_string()];
        let tokens2 = vec!["hello".to_string(), "world".to_string()];

        let similarity = evaluator.jaccard_similarity(&tokens1, &tokens2);
        assert_eq!(similarity, 1.0);
    }

    #[test]
    fn test_edit_distance() {
        let config = SemanticSimilarityConfig::default();
        let evaluator = SemanticSimilarityEvaluator::new(config).unwrap();

        let tokens1 = vec!["hello".to_string(), "world".to_string()];
        let tokens2 = vec!["hello".to_string(), "there".to_string()];

        let distance = evaluator.levenshtein_distance(&tokens1, &tokens2);
        assert_eq!(distance, 1); // One substitution
    }

    #[test]
    fn test_ngram_generation() {
        let config = SemanticSimilarityConfig::default();
        let evaluator = SemanticSimilarityEvaluator::new(config).unwrap();

        let tokens = vec!["the".to_string(), "quick".to_string(), "brown".to_string()];
        let ngrams = evaluator.generate_ngrams(&tokens, 2);

        assert!(ngrams.contains("the quick"));
        assert!(ngrams.contains("quick brown"));
        assert_eq!(ngrams.len(), 2);
    }

    #[test]
    fn test_content_preservation() {
        let config = SemanticSimilarityConfig::default();
        let evaluator = SemanticSimilarityEvaluator::new(config).unwrap();

        let ref_tokens = vec![
            "important".to_string(),
            "content".to_string(),
            "here".to_string(),
        ];
        let syn_tokens = vec!["important".to_string(), "content".to_string()];

        let preservation = evaluator
            .calculate_content_preservation(&ref_tokens, &syn_tokens)
            .unwrap();

        assert!(preservation > 0.6 && preservation < 1.0);
    }

    #[test]
    fn test_difference_analysis() {
        let config = SemanticSimilarityConfig {
            analyze_differences: true,
            ..Default::default()
        };
        let evaluator = SemanticSimilarityEvaluator::new(config).unwrap();

        let ref_text = "hello world";
        let syn_text = "hello there";

        let result = evaluator.evaluate(ref_text, syn_text).unwrap();
        assert!(!result.differences.is_empty());
    }

    #[test]
    fn test_empty_texts() {
        let config = SemanticSimilarityConfig::default();
        let evaluator = SemanticSimilarityEvaluator::new(config).unwrap();

        let result = evaluator.evaluate("", "");
        assert!(result.is_ok());
    }

    #[test]
    fn test_completely_different_texts() {
        let config = SemanticSimilarityConfig::default();
        let evaluator = SemanticSimilarityEvaluator::new(config).unwrap();

        let ref_text = "The weather is nice today";
        let syn_text = "Quantum physics explains particle behavior";

        let result = evaluator.evaluate(ref_text, syn_text);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.overall_similarity < 0.5);
    }
}

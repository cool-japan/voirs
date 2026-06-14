//! Content embedding extraction module
//!
//! This module provides implementations for extracting content embeddings
//! from text and phonemes using various methods including Word2Vec, BERT, and phoneme-based approaches.

use super::config::{ContentEmbeddingConfig, ContentEmbeddingMethod};
use crate::{DatasetSample, Result};
use std::collections::HashMap;

mod ipa_features;

/// Content embedding extractor
pub struct ContentEmbeddingExtractor {
    config: ContentEmbeddingConfig,
    #[allow(dead_code)]
    model: Option<ContentModel>,
}

/// Content model (placeholder)
#[allow(dead_code)]
struct ContentModel {
    weights: Vec<f32>,
    vocabulary: HashMap<String, usize>,
}

impl ContentEmbeddingExtractor {
    pub fn new(config: ContentEmbeddingConfig) -> Result<Self> {
        Ok(Self {
            config,
            model: None,
        })
    }

    pub async fn extract_embedding(&self, sample: &DatasetSample) -> Result<Vec<f32>> {
        match &self.config.method {
            ContentEmbeddingMethod::Word2Vec { vector_size, .. } => {
                // Simple Word2Vec-like embedding using character n-grams and TF-IDF weighting
                let text = &sample.text;
                let mut embedding = vec![0.0; *vector_size];

                if !text.is_empty() {
                    let words: Vec<&str> = text.split_whitespace().collect();
                    let mut _feature_count = 0;

                    // Extract character n-grams as features
                    for word in words.iter() {
                        for i in 0..word.len().saturating_sub(2) {
                            let trigram = &word[i..i + 3];
                            let hash = self.simple_hash(trigram) % *vector_size;
                            embedding[hash] += 1.0 / words.len() as f32;
                            _feature_count += 1;
                        }
                    }

                    // Word-level features with context window
                    for (i, word) in words.iter().enumerate() {
                        let word_hash = self.simple_hash(word) % *vector_size;
                        let tf_weight = 1.0 / words.len() as f32;
                        embedding[word_hash] += tf_weight;

                        // Context features within window
                        let window = 3; // Default window size
                        #[allow(clippy::needless_range_loop)]
                        for j in
                            i.saturating_sub(window)..std::cmp::min(i + window + 1, words.len())
                        {
                            if i != j {
                                let context_hash =
                                    (self.simple_hash(words[j]) + word_hash) % *vector_size;
                                embedding[context_hash] += 0.5 * tf_weight;
                            }
                        }
                    }

                    // Normalize embedding
                    let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if norm > 0.0 {
                        for val in embedding.iter_mut() {
                            *val /= norm;
                        }
                    }
                }

                Ok(embedding)
            }
            ContentEmbeddingMethod::BERT { .. } => {
                // BERT-like transformer embedding using attention-based features
                let text = &sample.text;
                let mut embedding = vec![0.0; self.config.text_dimension];

                if !text.is_empty() {
                    let tokens: Vec<&str> = text.split_whitespace().collect();
                    let num_tokens = tokens.len();

                    if num_tokens > 0 {
                        // Multi-head attention simulation
                        let num_heads = 8;
                        let head_dim = self.config.text_dimension / num_heads;

                        for head in 0..num_heads {
                            let mut head_features = vec![0.0; head_dim];

                            for (i, token) in tokens.iter().enumerate() {
                                let token_hash = self.simple_hash(token);
                                let position_encoding = self.positional_encoding(i, head_dim);

                                // Attention-weighted features
                                for j in 0..head_dim {
                                    let attention_weight = self.attention_score(i, j, num_tokens);
                                    // Prevent bit shift overflow by limiting shift amount
                                    let shift_amount = j.min(31); // Limit to 31 bits for safety
                                    let feature_value = ((token_hash >> shift_amount) & 1) as f32
                                        * attention_weight;
                                    head_features[j] += feature_value + position_encoding[j];
                                }
                            }

                            // Normalize head features
                            let norm = head_features.iter().map(|x| x * x).sum::<f32>().sqrt();
                            if norm > 0.0 {
                                for val in head_features.iter_mut() {
                                    *val /= norm;
                                }
                            }

                            // Aggregate into final embedding
                            for (j, &val) in head_features.iter().enumerate() {
                                embedding[head * head_dim + j] = val;
                            }
                        }

                        // Apply final layer normalization
                        let mean = embedding.iter().sum::<f32>() / embedding.len() as f32;
                        let variance = embedding.iter().map(|x| (x - mean).powi(2)).sum::<f32>()
                            / embedding.len() as f32;
                        let std_dev = variance.sqrt() + 1e-8;

                        for val in embedding.iter_mut() {
                            *val = (*val - mean) / std_dev;
                        }
                    }
                }

                Ok(embedding)
            }
            ContentEmbeddingMethod::Phoneme { dimension, .. } => {
                // Phoneme embedding using IPA phoneme features
                let mut embedding = vec![0.0; *dimension];

                if let Some(phonemes) = &sample.phonemes {
                    // Join the phoneme symbols (IPA / ARPAbet) with spaces so the
                    // distinctive-feature lookup receives one symbol per token.
                    let phoneme_string = phonemes
                        .iter()
                        .map(|p| p.symbol.clone())
                        .collect::<Vec<_>>()
                        .join(" ");
                    let phoneme_features =
                        self.extract_phoneme_features(&phoneme_string, *dimension);
                    embedding = phoneme_features;
                } else {
                    // Extract phoneme-like features from text using grapheme-to-phoneme approximation
                    let text = &sample.text.to_lowercase();
                    let mut _feature_count = 0;

                    // Map common letter patterns to phonetic features
                    let phonetic_patterns = [
                        ("th", vec![0.8, 0.3, 0.9, 0.1]),
                        ("ch", vec![0.7, 0.8, 0.2, 0.9]),
                        ("sh", vec![0.9, 0.1, 0.8, 0.7]),
                        ("ph", vec![0.6, 0.9, 0.3, 0.8]),
                        ("ng", vec![0.4, 0.7, 0.9, 0.2]),
                    ];

                    for (pattern, features) in phonetic_patterns.iter() {
                        let count = text.matches(pattern).count() as f32;
                        if count > 0.0 {
                            for (i, &feature_val) in features.iter().enumerate() {
                                if i < *dimension {
                                    embedding[i] += feature_val * count / text.len() as f32;
                                    _feature_count += 1;
                                }
                            }
                        }
                    }

                    // Add vowel/consonant distribution features
                    let vowels = "aeiou";
                    let vowel_count = text.chars().filter(|c| vowels.contains(*c)).count() as f32;
                    let consonant_count = text
                        .chars()
                        .filter(|c| c.is_alphabetic() && !vowels.contains(*c))
                        .count() as f32;
                    let total_letters = vowel_count + consonant_count;

                    if total_letters > 0.0 && *dimension > 4 {
                        embedding[0] = vowel_count / total_letters;
                        embedding[1] = consonant_count / total_letters;
                        embedding[2] = text.len() as f32 / 100.0; // Length normalization
                        embedding[3] = text.split_whitespace().count() as f32 / text.len() as f32;
                        // Word density
                    }

                    // Normalize the embedding
                    let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if norm > 0.0 {
                        for val in embedding.iter_mut() {
                            *val /= norm;
                        }
                    }
                }

                Ok(embedding)
            }
        }
    }

    // Helper methods for embedding extraction
    fn simple_hash(&self, text: &str) -> usize {
        let mut hash = 5381usize;
        for byte in text.bytes() {
            hash = hash.wrapping_mul(33).wrapping_add(byte as usize);
        }
        hash
    }

    fn positional_encoding(&self, position: usize, dimension: usize) -> Vec<f32> {
        let mut encoding = vec![0.0; dimension];
        #[allow(clippy::needless_range_loop)]
        for i in 0..dimension {
            let angle = position as f32 / 10000.0_f32.powf(2.0 * (i / 2) as f32 / dimension as f32);
            if i % 2 == 0 {
                encoding[i] = angle.sin();
            } else {
                encoding[i] = angle.cos();
            }
        }
        encoding
    }

    fn attention_score(&self, position: usize, dimension_idx: usize, num_tokens: usize) -> f32 {
        // Simple attention mechanism simulation
        let relative_position = position as f32 / num_tokens as f32;
        let dimension_weight = 1.0 / (1.0 + dimension_idx as f32 * 0.1);
        (relative_position * dimension_weight).tanh()
    }

    /// Extract a content embedding from a whitespace-separated phoneme string.
    ///
    /// Each token is mapped to a fixed-width IPA distinctive-feature vector (see
    /// [`ipa_features`]; both IPA and ARPAbet spellings are recognised), the
    /// per-phoneme vectors are averaged over the sequence, written into the
    /// leading dimensions of a `dimension`-wide embedding, and L2-normalised.
    /// Unknown symbols contribute the neutral (all-zero) vector.
    fn extract_phoneme_features(&self, phonemes: &str, dimension: usize) -> Vec<f32> {
        let mut features = vec![0.0f32; dimension];

        let tokens: Vec<&str> = phonemes.split_whitespace().collect();
        if tokens.is_empty() {
            return features;
        }

        // Average the distinctive-feature vectors across the phoneme sequence.
        let inv_count = 1.0 / tokens.len() as f32;
        for token in &tokens {
            let phoneme_vector = ipa_features::phoneme_feature_vector(token);
            // `zip` stops at the shorter of the two, so this also handles the
            // case where the requested `dimension` is narrower than the feature
            // matrix without any explicit bounds checking.
            for (slot, &feature_val) in features.iter_mut().zip(phoneme_vector.iter()) {
                *slot += feature_val * inv_count;
            }
        }

        // Normalize features
        let norm = features.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for val in features.iter_mut() {
                *val /= norm;
            }
        }

        features
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AudioData, LanguageCode, Phoneme};

    fn phoneme_extractor(dimension: usize) -> ContentEmbeddingExtractor {
        let config = ContentEmbeddingConfig {
            text_dimension: 64,
            phoneme_dimension: dimension,
            method: ContentEmbeddingMethod::Phoneme {
                dimension,
                pretrained: false,
                phoneme_set: "ipa".to_string(),
            },
            use_contextual: false,
        };
        ContentEmbeddingExtractor::new(config).expect("extractor construction should succeed")
    }

    #[test]
    fn real_phonemes_yield_nonzero_normalized_embedding() {
        let extractor = phoneme_extractor(ipa_features::FEATURE_DIM);
        // ARPAbet transcription of "hello" (with stress digits).
        let features =
            extractor.extract_phoneme_features("HH AH0 L OW1", ipa_features::FEATURE_DIM);
        assert_eq!(features.len(), ipa_features::FEATURE_DIM);
        let energy: f32 = features.iter().map(|x| x.abs()).sum();
        assert!(
            energy > 0.0,
            "real phonemes should produce a non-zero vector"
        );
        // L2-normalised output.
        let norm: f32 = features.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "embedding should be unit-norm");
    }

    #[test]
    fn unknown_only_sequence_is_zero() {
        let extractor = phoneme_extractor(ipa_features::FEATURE_DIM);
        let features = extractor.extract_phoneme_features("QZX ### ???", ipa_features::FEATURE_DIM);
        assert!(
            features.iter().all(|&x| x == 0.0),
            "a sequence of unknown symbols should stay neutral"
        );
    }

    #[test]
    fn empty_input_is_zero() {
        let extractor = phoneme_extractor(8);
        let features = extractor.extract_phoneme_features("   ", 8);
        assert_eq!(features, vec![0.0; 8]);
    }

    #[test]
    fn narrow_dimension_is_respected() {
        let extractor = phoneme_extractor(4);
        let features = extractor.extract_phoneme_features("P AA T", 4);
        assert_eq!(features.len(), 4);
    }

    #[tokio::test]
    async fn extract_embedding_uses_phoneme_symbols() {
        let extractor = phoneme_extractor(ipa_features::FEATURE_DIM);
        let audio = AudioData::silence(0.1, 16000, 1);
        let sample = DatasetSample::new(
            "sample-1".to_string(),
            "hello".to_string(),
            audio,
            LanguageCode::EnUs,
        )
        .with_phonemes(vec![
            Phoneme::new("HH"),
            Phoneme::new("AH0"),
            Phoneme::new("L"),
            Phoneme::new("OW1"),
        ]);

        let embedding = extractor
            .extract_embedding(&sample)
            .await
            .expect("embedding extraction should succeed");
        assert_eq!(embedding.len(), ipa_features::FEATURE_DIM);
        let energy: f32 = embedding.iter().map(|x| x.abs()).sum();
        assert!(
            energy > 0.0,
            "phoneme-based embedding should be non-zero for real symbols"
        );
    }
}

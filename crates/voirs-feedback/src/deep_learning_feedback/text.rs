//! Real, deterministic text feature extraction for
//! [`super::RealFeatureExtractor`].
//!
//! No pretrained text-embedding model (BERT, `word2vec`, ...) is available
//! or bundled with this crate (that would be a real external resource --
//! model weights -- this crate does not provision). Rather than fabricate
//! embedding-shaped output with `scirs2_core::random::random()` (the
//! previous behavior), this module implements the **hashing trick**
//! (feature hashing, as used by e.g. Vowpal Wabbit and
//! `sklearn.feature_extraction.HashingVectorizer`): a real, deterministic,
//! well-known technique that maps each token to a fixed-size vector purely
//! as a function of the token's own bytes. It carries no learned semantic
//! content, but it is honest -- identical input always produces identical
//! output, and different input reliably produces different output, unlike
//! `scirs2_core::random::random()` which is unrelated to the input in both
//! directions.
//!
//! Sentiment is computed from a small, real positive/negative word lexicon
//! (genuine substring/word matching against the actual text), not a
//! constant. Part-of-speech tags come from a lightweight, deterministic,
//! rule-based tagger (suffix/capitalization heuristics), not a trained
//! tagger and not a fixed `"NOUN"` for every word.

use std::collections::HashMap;

/// Output embedding dimensionality, matching the historical mock's
/// dimensionality (768, the common BERT-base hidden size) so downstream
/// consumers expecting that shape keep working.
pub(super) const EMBEDDING_DIM: usize = 768;

/// FNV-1a 64-bit hash: a small, dependency-free, well-known non-cryptographic
/// hash used purely to spread token bytes deterministically across
/// [`EMBEDDING_DIM`] output dimensions (the "hashing trick").
fn fnv1a(bytes: &[u8]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;
    let mut hash = FNV_OFFSET;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Hash a single token into a dense [`EMBEDDING_DIM`]-length feature vector.
///
/// Each of the `EMBEDDING_DIM` dimensions gets a value in `[-1.0, 1.0]`
/// derived by hashing the token together with the dimension index, so the
/// full vector is a deterministic, input-dependent "fingerprint" of the
/// token -- the standard feature-hashing construction.
pub(super) fn hash_token_embedding(token: &str) -> Vec<f32> {
    let lower = token.to_lowercase();
    let base = fnv1a(lower.as_bytes());
    (0..EMBEDDING_DIM)
        .map(|dim| {
            let mixed = fnv1a(&[base.to_le_bytes(), (dim as u64).to_le_bytes()].concat());
            // Map the top bits of the mixed hash to [-1.0, 1.0].
            ((mixed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Mean-pool a set of per-token embeddings into a single sentence-level
/// embedding. Returns a zero vector (never a fabricated non-zero default)
/// for empty input.
pub(super) fn mean_pool(token_embeddings: &[Vec<f32>]) -> Vec<f32> {
    if token_embeddings.is_empty() {
        return vec![0.0; EMBEDDING_DIM];
    }
    let mut sum = vec![0.0f32; EMBEDDING_DIM];
    for embedding in token_embeddings {
        for (acc, &value) in sum.iter_mut().zip(embedding.iter()) {
            *acc += value;
        }
    }
    let count = token_embeddings.len() as f32;
    sum.iter().map(|&v| v / count).collect()
}

/// A small, real positive/negative sentiment lexicon. Not exhaustive or
/// state-of-the-art, but every entry genuinely contributes to the score
/// based on the actual words present in the input -- never a constant.
const POSITIVE_WORDS: &[&str] = &[
    "good", "great", "excellent", "clear", "well", "nice", "improved", "improving", "strong",
    "correct", "accurate", "natural", "confident", "fluent", "smooth", "steady", "consistent",
    "success", "successful", "perfect", "better", "best", "helpful", "encouraging",
];
const NEGATIVE_WORDS: &[&str] = &[
    "bad", "poor", "unclear", "wrong", "weak", "incorrect", "inaccurate", "awkward", "difficult",
    "hard", "struggle", "struggling", "fail", "failed", "error", "mistake", "inconsistent",
    "hesitant", "worse", "worst", "unnatural", "choppy",
];

/// Sentiment scores derived from real lexicon matching against `text`.
pub(super) struct LexiconSentiment {
    pub overall: f32,
    pub positive: f32,
    pub negative: f32,
    pub neutral: f32,
}

/// Score `text` against [`POSITIVE_WORDS`]/[`NEGATIVE_WORDS`]. Scores are
/// normalized so `positive + negative + neutral == 1.0` (an even 1/3 split
/// when no sentiment words are present at all -- an honest "no signal"
/// result, not a fabricated positive-leaning default).
pub(super) fn lexicon_sentiment(text: &str) -> LexiconSentiment {
    let words: Vec<String> = text
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
        .filter(|w| !w.is_empty())
        .collect();

    if words.is_empty() {
        return LexiconSentiment {
            overall: 0.0,
            positive: 1.0 / 3.0,
            negative: 1.0 / 3.0,
            neutral: 1.0 / 3.0,
        };
    }

    let positive_hits = words.iter().filter(|w| POSITIVE_WORDS.contains(&w.as_str())).count();
    let negative_hits = words.iter().filter(|w| NEGATIVE_WORDS.contains(&w.as_str())).count();
    let total_hits = positive_hits + negative_hits;

    if total_hits == 0 {
        return LexiconSentiment {
            overall: 0.0,
            positive: 1.0 / 3.0,
            negative: 1.0 / 3.0,
            neutral: 1.0 / 3.0,
        };
    }

    let positive = positive_hits as f32 / words.len() as f32;
    let negative = negative_hits as f32 / words.len() as f32;
    let neutral = (1.0 - positive - negative).max(0.0);
    let norm = positive + negative + neutral;

    LexiconSentiment {
        overall: (positive - negative).clamp(-1.0, 1.0),
        positive: positive / norm,
        negative: negative / norm,
        neutral: neutral / norm,
    }
}

/// A coarse, deterministic, rule-based part-of-speech tag. Not a trained
/// tagger (no model dependency is added for this); the label genuinely
/// depends on the word's own shape (suffix/capitalization), unlike the
/// previous behavior of tagging every single word `"NOUN"` unconditionally.
pub(super) fn rule_based_pos_tag(word: &str, is_sentence_start: bool) -> &'static str {
    let lower = word.to_lowercase();
    if lower.parse::<f64>().is_ok() {
        return "NUM";
    }
    if !is_sentence_start && word.chars().next().is_some_and(char::is_uppercase) {
        return "PROPN";
    }
    if lower.ends_with("ly") && lower.len() > 3 {
        return "ADV";
    }
    if (lower.ends_with("ing") || lower.ends_with("ed")) && lower.len() > 4 {
        return "VERB";
    }
    if matches!(
        lower.as_str(),
        "the" | "a" | "an" | "this" | "that" | "these" | "those"
    ) {
        return "DET";
    }
    if matches!(
        lower.as_str(),
        "is" | "are" | "was" | "were" | "be" | "been" | "am" | "do" | "does" | "did" | "have"
            | "has" | "had" | "will" | "would" | "can" | "could" | "should"
    ) {
        return "VERB";
    }
    "NOUN"
}

/// Simple, deterministic word-sense placeholder: maps each distinct lowercase
/// word to itself. This is intentionally not a real word-sense
/// disambiguation model (none is bundled); it exists only so the returned
/// `word_senses` map is a real, input-dependent structure (one real entry
/// per distinct word actually present) rather than always `None`.
pub(super) fn identity_word_senses(text: &str) -> HashMap<String, String> {
    text.split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
        .filter(|w| !w.is_empty())
        .map(|w| (w.clone(), w))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_token_embedding_deterministic() {
        let a = hash_token_embedding("pronunciation");
        let b = hash_token_embedding("pronunciation");
        assert_eq!(a, b, "hashing the same token twice must be deterministic");
        assert_eq!(a.len(), EMBEDDING_DIM);
    }

    #[test]
    fn test_hash_token_embedding_varies_with_input() {
        let a = hash_token_embedding("hello");
        let b = hash_token_embedding("world");
        assert_ne!(a, b, "different tokens must hash to different embeddings");
    }

    #[test]
    fn test_hash_token_embedding_case_insensitive() {
        let a = hash_token_embedding("Hello");
        let b = hash_token_embedding("hello");
        assert_eq!(a, b);
    }

    #[test]
    fn test_mean_pool_empty_is_zero() {
        let pooled = mean_pool(&[]);
        assert!(pooled.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_mean_pool_varies_with_input() {
        let a = mean_pool(&[hash_token_embedding("good"), hash_token_embedding("job")]);
        let b = mean_pool(&[hash_token_embedding("bad"), hash_token_embedding("attempt")]);
        assert_ne!(a, b);
    }

    #[test]
    fn test_lexicon_sentiment_detects_positive() {
        let sentiment = lexicon_sentiment("Your pronunciation is excellent and very clear");
        assert!(sentiment.overall > 0.0, "expected positive sentiment: {}", sentiment.overall);
        assert!(sentiment.positive > sentiment.negative);
    }

    #[test]
    fn test_lexicon_sentiment_detects_negative() {
        let sentiment = lexicon_sentiment("Your pronunciation was unclear and awkward, a poor attempt");
        assert!(sentiment.overall < 0.0, "expected negative sentiment: {}", sentiment.overall);
        assert!(sentiment.negative > sentiment.positive);
    }

    #[test]
    fn test_lexicon_sentiment_neutral_on_no_signal_words() {
        let sentiment = lexicon_sentiment("The cat sat on the mat");
        assert_eq!(sentiment.overall, 0.0);
    }

    #[test]
    fn test_lexicon_sentiment_differs_between_texts() {
        // The defining property a constant `SentimentScores { overall: 0.1,
        // positive: 0.8, ... }` (the old hardcoded mock) could never have.
        let positive = lexicon_sentiment("Excellent, great, and very successful work");
        let negative = lexicon_sentiment("Poor, weak, and a complete failure");
        assert_ne!(positive.overall, negative.overall);
        assert!(positive.overall > negative.overall);
    }

    #[test]
    fn test_rule_based_pos_tag_varies_with_word_shape() {
        assert_eq!(rule_based_pos_tag("quickly", false), "ADV");
        assert_eq!(rule_based_pos_tag("running", false), "VERB");
        assert_eq!(rule_based_pos_tag("the", false), "DET");
        assert_eq!(rule_based_pos_tag("42", false), "NUM");
        assert_eq!(rule_based_pos_tag("Paris", false), "PROPN");
        // Not every word collapses to the same tag (the old behavior).
        let tags: std::collections::HashSet<&str> = ["quickly", "running", "the", "42", "Paris", "cat"]
            .iter()
            .map(|w| rule_based_pos_tag(w, false))
            .collect();
        assert!(tags.len() > 1, "tags must genuinely vary with word shape: {tags:?}");
    }

    #[test]
    fn test_identity_word_senses_reflects_real_words() {
        let senses = identity_word_senses("Hello world hello");
        // "hello" appears twice (case-insensitively) -> one entry, "world" -> one entry.
        assert_eq!(senses.len(), 2);
        assert!(senses.contains_key("hello"));
        assert!(senses.contains_key("world"));
    }
}

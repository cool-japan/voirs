//! Phoneme statistics and analysis utilities
//!
//! This module provides comprehensive statistical analysis tools for phoneme sequences,
//! useful for model training, quality assurance, debugging, and linguistic research.
//!
//! # Features
//! - Frequency analysis (unigrams, bigrams, trigrams)
//! - Diversity metrics (Shannon entropy, type-token ratio, Simpson's index)
//! - Pattern detection (common sequences, anomalies)
//! - Co-occurrence analysis
//! - Phonological feature statistics
//!
//! # Examples
//! ```
//! use voirs_g2p::utils::phoneme_stats::PhonemeStatistics;
//! use voirs_g2p::Phoneme;
//!
//! let phonemes = vec![
//!     Phoneme::new("h".to_string()),
//!     Phoneme::new("ɛ".to_string()),
//!     Phoneme::new("l".to_string()),
//!     Phoneme::new("oʊ".to_string()),
//! ];
//!
//! let stats = PhonemeStatistics::from_sequence(&phonemes);
//! let entropy = stats.shannon_entropy();
//! let diversity = stats.type_token_ratio();
//! ```

use crate::Phoneme;
use std::collections::HashMap;

/// Comprehensive phoneme statistics analyzer
#[derive(Debug, Clone)]
pub struct PhonemeStatistics {
    /// Total number of phonemes
    pub total_phonemes: usize,
    /// Unique phoneme count
    pub unique_phonemes: usize,
    /// Unigram frequencies (single phoneme counts)
    pub unigram_freq: HashMap<String, usize>,
    /// Bigram frequencies (two-phoneme sequence counts)
    pub bigram_freq: HashMap<(String, String), usize>,
    /// Trigram frequencies (three-phoneme sequence counts)
    pub trigram_freq: HashMap<(String, String, String), usize>,
}

impl PhonemeStatistics {
    /// Create statistics from a phoneme sequence
    ///
    /// # Arguments
    /// * `phonemes` - Sequence of phonemes to analyze
    ///
    /// # Returns
    /// PhonemeStatistics with computed frequency distributions
    ///
    /// # Examples
    /// ```
    /// use voirs_g2p::utils::phoneme_stats::PhonemeStatistics;
    /// use voirs_g2p::Phoneme;
    ///
    /// let phonemes = vec![
    ///     Phoneme::new("k".to_string()),
    ///     Phoneme::new("æ".to_string()),
    ///     Phoneme::new("t".to_string()),
    /// ];
    ///
    /// let stats = PhonemeStatistics::from_sequence(&phonemes);
    /// assert_eq!(stats.total_phonemes, 3);
    /// assert_eq!(stats.unique_phonemes, 3);
    /// ```
    pub fn from_sequence(phonemes: &[Phoneme]) -> Self {
        let mut unigram_freq = HashMap::new();
        let mut bigram_freq = HashMap::new();
        let mut trigram_freq = HashMap::new();

        // Count unigrams
        for phoneme in phonemes {
            *unigram_freq.entry(phoneme.symbol.clone()).or_insert(0) += 1;
        }

        // Count bigrams
        for window in phonemes.windows(2) {
            let bigram = (window[0].symbol.clone(), window[1].symbol.clone());
            *bigram_freq.entry(bigram).or_insert(0) += 1;
        }

        // Count trigrams
        for window in phonemes.windows(3) {
            let trigram = (
                window[0].symbol.clone(),
                window[1].symbol.clone(),
                window[2].symbol.clone(),
            );
            *trigram_freq.entry(trigram).or_insert(0) += 1;
        }

        Self {
            total_phonemes: phonemes.len(),
            unique_phonemes: unigram_freq.len(),
            unigram_freq,
            bigram_freq,
            trigram_freq,
        }
    }

    /// Create statistics from multiple phoneme sequences
    ///
    /// # Arguments
    /// * `sequences` - Multiple phoneme sequences to analyze
    ///
    /// # Returns
    /// Aggregate PhonemeStatistics across all sequences
    ///
    /// # Examples
    /// ```
    /// use voirs_g2p::utils::phoneme_stats::PhonemeStatistics;
    /// use voirs_g2p::Phoneme;
    ///
    /// let sequences = vec![
    ///     vec![Phoneme::new("h".to_string()), Phoneme::new("i".to_string())],
    ///     vec![Phoneme::new("h".to_string()), Phoneme::new("æ".to_string())],
    /// ];
    ///
    /// let stats = PhonemeStatistics::from_sequences(&sequences);
    /// assert_eq!(stats.total_phonemes, 4);
    /// ```
    pub fn from_sequences(sequences: &[Vec<Phoneme>]) -> Self {
        let mut combined_stats = Self {
            total_phonemes: 0,
            unique_phonemes: 0,
            unigram_freq: HashMap::new(),
            bigram_freq: HashMap::new(),
            trigram_freq: HashMap::new(),
        };

        for sequence in sequences {
            let seq_stats = Self::from_sequence(sequence);
            combined_stats.merge(seq_stats);
        }

        combined_stats.unique_phonemes = combined_stats.unigram_freq.len();
        combined_stats
    }

    /// Merge another PhonemeStatistics into this one
    fn merge(&mut self, other: Self) {
        self.total_phonemes += other.total_phonemes;

        // Merge unigrams
        for (phoneme, count) in other.unigram_freq {
            *self.unigram_freq.entry(phoneme).or_insert(0) += count;
        }

        // Merge bigrams
        for (bigram, count) in other.bigram_freq {
            *self.bigram_freq.entry(bigram).or_insert(0) += count;
        }

        // Merge trigrams
        for (trigram, count) in other.trigram_freq {
            *self.trigram_freq.entry(trigram).or_insert(0) += count;
        }
    }

    /// Calculate Shannon entropy (information content)
    ///
    /// Higher entropy indicates more diverse/unpredictable phoneme distribution
    ///
    /// # Returns
    /// Shannon entropy in bits
    ///
    /// # Examples
    /// ```
    /// use voirs_g2p::utils::phoneme_stats::PhonemeStatistics;
    /// use voirs_g2p::Phoneme;
    ///
    /// let phonemes = vec![
    ///     Phoneme::new("a".to_string()),
    ///     Phoneme::new("b".to_string()),
    ///     Phoneme::new("c".to_string()),
    /// ];
    ///
    /// let stats = PhonemeStatistics::from_sequence(&phonemes);
    /// let entropy = stats.shannon_entropy();
    /// assert!(entropy > 0.0);
    /// ```
    pub fn shannon_entropy(&self) -> f64 {
        if self.total_phonemes == 0 {
            return 0.0;
        }

        let total = self.total_phonemes as f64;
        let mut entropy = 0.0;

        for &count in self.unigram_freq.values() {
            let prob = count as f64 / total;
            if prob > 0.0 {
                entropy -= prob * prob.log2();
            }
        }

        entropy
    }

    /// Calculate type-token ratio (lexical diversity)
    ///
    /// Ratio of unique phonemes to total phonemes. Range: [0.0, 1.0]
    /// Higher values indicate more diverse phoneme usage.
    ///
    /// # Returns
    /// Type-token ratio (unique / total)
    ///
    /// # Examples
    /// ```
    /// use voirs_g2p::utils::phoneme_stats::PhonemeStatistics;
    /// use voirs_g2p::Phoneme;
    ///
    /// let phonemes = vec![
    ///     Phoneme::new("a".to_string()),
    ///     Phoneme::new("b".to_string()),
    ///     Phoneme::new("a".to_string()),
    /// ];
    ///
    /// let stats = PhonemeStatistics::from_sequence(&phonemes);
    /// let ttr = stats.type_token_ratio();
    /// assert!((ttr - 0.666).abs() < 0.01);
    /// ```
    pub fn type_token_ratio(&self) -> f64 {
        if self.total_phonemes == 0 {
            return 0.0;
        }
        self.unique_phonemes as f64 / self.total_phonemes as f64
    }

    /// Calculate Simpson's diversity index
    ///
    /// Probability that two randomly selected phonemes are different.
    /// Range: [0.0, 1.0]. Higher values indicate more diversity.
    ///
    /// # Returns
    /// Simpson's diversity index (1 - sum(p_i^2))
    ///
    /// # Examples
    /// ```
    /// use voirs_g2p::utils::phoneme_stats::PhonemeStatistics;
    /// use voirs_g2p::Phoneme;
    ///
    /// let phonemes = vec![
    ///     Phoneme::new("a".to_string()),
    ///     Phoneme::new("b".to_string()),
    ///     Phoneme::new("c".to_string()),
    /// ];
    ///
    /// let stats = PhonemeStatistics::from_sequence(&phonemes);
    /// let diversity = stats.simpsons_diversity();
    /// assert!(diversity > 0.5);
    /// ```
    pub fn simpsons_diversity(&self) -> f64 {
        if self.total_phonemes == 0 {
            return 0.0;
        }

        let total = self.total_phonemes as f64;
        let sum_squares: f64 = self
            .unigram_freq
            .values()
            .map(|&count| {
                let prob = count as f64 / total;
                prob * prob
            })
            .sum();

        1.0 - sum_squares
    }

    /// Get the most frequent phonemes
    ///
    /// # Arguments
    /// * `n` - Number of top phonemes to return
    ///
    /// # Returns
    /// Vector of (phoneme, count) tuples sorted by frequency (descending)
    ///
    /// # Examples
    /// ```
    /// use voirs_g2p::utils::phoneme_stats::PhonemeStatistics;
    /// use voirs_g2p::Phoneme;
    ///
    /// let phonemes = vec![
    ///     Phoneme::new("a".to_string()),
    ///     Phoneme::new("a".to_string()),
    ///     Phoneme::new("b".to_string()),
    /// ];
    ///
    /// let stats = PhonemeStatistics::from_sequence(&phonemes);
    /// let top = stats.most_frequent_phonemes(1);
    /// assert_eq!(top[0].0, "a");
    /// assert_eq!(top[0].1, 2);
    /// ```
    pub fn most_frequent_phonemes(&self, n: usize) -> Vec<(String, usize)> {
        let mut freq_vec: Vec<_> = self
            .unigram_freq
            .iter()
            .map(|(k, &v)| (k.clone(), v))
            .collect();
        freq_vec.sort_by(|a, b| b.1.cmp(&a.1));
        freq_vec.into_iter().take(n).collect()
    }

    /// Get the most frequent bigrams
    ///
    /// # Arguments
    /// * `n` - Number of top bigrams to return
    ///
    /// # Returns
    /// Vector of ((phoneme1, phoneme2), count) tuples sorted by frequency
    ///
    /// # Examples
    /// ```
    /// use voirs_g2p::utils::phoneme_stats::PhonemeStatistics;
    /// use voirs_g2p::Phoneme;
    ///
    /// let phonemes = vec![
    ///     Phoneme::new("a".to_string()),
    ///     Phoneme::new("b".to_string()),
    ///     Phoneme::new("a".to_string()),
    ///     Phoneme::new("b".to_string()),
    /// ];
    ///
    /// let stats = PhonemeStatistics::from_sequence(&phonemes);
    /// let top_bigrams = stats.most_frequent_bigrams(1);
    /// assert_eq!(top_bigrams[0].1, 2); // "a,b" appears twice
    /// ```
    pub fn most_frequent_bigrams(&self, n: usize) -> Vec<((String, String), usize)> {
        let mut freq_vec: Vec<_> = self
            .bigram_freq
            .iter()
            .map(|(k, &v)| (k.clone(), v))
            .collect();
        freq_vec.sort_by(|a, b| b.1.cmp(&a.1));
        freq_vec.into_iter().take(n).collect()
    }

    /// Get the most frequent trigrams
    ///
    /// # Arguments
    /// * `n` - Number of top trigrams to return
    ///
    /// # Returns
    /// Vector of ((phoneme1, phoneme2, phoneme3), count) tuples sorted by frequency
    pub fn most_frequent_trigrams(&self, n: usize) -> Vec<((String, String, String), usize)> {
        let mut freq_vec: Vec<_> = self
            .trigram_freq
            .iter()
            .map(|(k, &v)| (k.clone(), v))
            .collect();
        freq_vec.sort_by(|a, b| b.1.cmp(&a.1));
        freq_vec.into_iter().take(n).collect()
    }

    /// Calculate phoneme probability distribution
    ///
    /// # Returns
    /// HashMap of phoneme to probability [0.0, 1.0]
    pub fn phoneme_probabilities(&self) -> HashMap<String, f64> {
        let total = self.total_phonemes as f64;
        self.unigram_freq
            .iter()
            .map(|(phoneme, &count)| (phoneme.clone(), count as f64 / total))
            .collect()
    }

    /// Detect rare phonemes (below threshold frequency)
    ///
    /// # Arguments
    /// * `threshold` - Minimum frequency threshold (e.g., 0.01 for 1%)
    ///
    /// # Returns
    /// Vector of rare phonemes with their frequencies
    ///
    /// # Examples
    /// ```
    /// use voirs_g2p::utils::phoneme_stats::PhonemeStatistics;
    /// use voirs_g2p::Phoneme;
    ///
    /// let mut phonemes = vec![
    ///     Phoneme::new("a".to_string()),
    ///     Phoneme::new("a".to_string()),
    ///     Phoneme::new("a".to_string()),
    ///     Phoneme::new("b".to_string()), // Rare phoneme (25%)
    /// ];
    ///
    /// let stats = PhonemeStatistics::from_sequence(&phonemes);
    /// let rare = stats.rare_phonemes(0.5); // 50% threshold
    /// assert_eq!(rare.len(), 1);
    /// ```
    pub fn rare_phonemes(&self, threshold: f64) -> Vec<(String, f64)> {
        let probabilities = self.phoneme_probabilities();
        probabilities
            .into_iter()
            .filter(|(_, prob)| *prob < threshold)
            .collect()
    }

    /// Calculate coverage (cumulative probability of top N phonemes)
    ///
    /// # Arguments
    /// * `n` - Number of top phonemes to include
    ///
    /// # Returns
    /// Cumulative probability [0.0, 1.0] of top N phonemes
    ///
    /// # Examples
    /// ```
    /// use voirs_g2p::utils::phoneme_stats::PhonemeStatistics;
    /// use voirs_g2p::Phoneme;
    ///
    /// let phonemes = vec![
    ///     Phoneme::new("a".to_string()),
    ///     Phoneme::new("a".to_string()),
    ///     Phoneme::new("b".to_string()),
    /// ];
    ///
    /// let stats = PhonemeStatistics::from_sequence(&phonemes);
    /// let coverage = stats.coverage(1);
    /// assert!((coverage - 0.666).abs() < 0.01);
    /// ```
    pub fn coverage(&self, n: usize) -> f64 {
        let top_phonemes = self.most_frequent_phonemes(n);
        let total = self.total_phonemes as f64;
        top_phonemes
            .iter()
            .map(|(_, count)| *count as f64 / total)
            .sum()
    }

    /// Generate a summary report of phoneme statistics
    ///
    /// # Returns
    /// String with formatted statistical summary
    pub fn summary_report(&self) -> String {
        format!(
            "Phoneme Statistics Report:\n\
             Total Phonemes: {}\n\
             Unique Phonemes: {}\n\
             Type-Token Ratio: {:.4}\n\
             Shannon Entropy: {:.4} bits\n\
             Simpson's Diversity: {:.4}\n\
             Top 5 Phonemes: {:?}\n\
             Top 5 Bigrams: {:?}",
            self.total_phonemes,
            self.unique_phonemes,
            self.type_token_ratio(),
            self.shannon_entropy(),
            self.simpsons_diversity(),
            self.most_frequent_phonemes(5)
                .iter()
                .map(|(p, c)| format!("{}:{}", p, c))
                .collect::<Vec<_>>(),
            self.most_frequent_bigrams(5)
                .iter()
                .map(|((p1, p2), c)| format!("{}-{}:{}", p1, p2, c))
                .collect::<Vec<_>>()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_statistics() {
        let phonemes = vec![
            Phoneme::new("a".to_string()),
            Phoneme::new("b".to_string()),
            Phoneme::new("a".to_string()),
        ];

        let stats = PhonemeStatistics::from_sequence(&phonemes);
        assert_eq!(stats.total_phonemes, 3);
        assert_eq!(stats.unique_phonemes, 2);
        assert_eq!(*stats.unigram_freq.get("a").unwrap_or(&0), 2);
        assert_eq!(*stats.unigram_freq.get("b").unwrap_or(&0), 1);
    }

    #[test]
    fn test_bigram_counting() {
        let phonemes = vec![
            Phoneme::new("a".to_string()),
            Phoneme::new("b".to_string()),
            Phoneme::new("a".to_string()),
        ];

        let stats = PhonemeStatistics::from_sequence(&phonemes);
        assert_eq!(stats.bigram_freq.len(), 2);
        let ab_count = stats
            .bigram_freq
            .get(&("a".to_string(), "b".to_string()))
            .unwrap_or(&0);
        assert_eq!(*ab_count, 1);
    }

    #[test]
    fn test_trigram_counting() {
        let phonemes = vec![
            Phoneme::new("a".to_string()),
            Phoneme::new("b".to_string()),
            Phoneme::new("c".to_string()),
        ];

        let stats = PhonemeStatistics::from_sequence(&phonemes);
        assert_eq!(stats.trigram_freq.len(), 1);
    }

    #[test]
    fn test_shannon_entropy() {
        // Uniform distribution should have maximum entropy
        let phonemes = vec![
            Phoneme::new("a".to_string()),
            Phoneme::new("b".to_string()),
            Phoneme::new("c".to_string()),
        ];

        let stats = PhonemeStatistics::from_sequence(&phonemes);
        let entropy = stats.shannon_entropy();

        // For 3 equally likely outcomes, max entropy = log2(3) ≈ 1.585
        assert!((entropy - 1.585).abs() < 0.01);
    }

    #[test]
    fn test_type_token_ratio() {
        let phonemes = vec![
            Phoneme::new("a".to_string()),
            Phoneme::new("b".to_string()),
            Phoneme::new("a".to_string()),
        ];

        let stats = PhonemeStatistics::from_sequence(&phonemes);
        let ttr = stats.type_token_ratio();
        assert!((ttr - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_simpsons_diversity() {
        let phonemes = vec![
            Phoneme::new("a".to_string()),
            Phoneme::new("b".to_string()),
            Phoneme::new("c".to_string()),
        ];

        let stats = PhonemeStatistics::from_sequence(&phonemes);
        let diversity = stats.simpsons_diversity();

        // For equal distribution, diversity = 1 - 3*(1/3)^2 = 2/3
        assert!((diversity - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_most_frequent_phonemes() {
        let phonemes = vec![
            Phoneme::new("a".to_string()),
            Phoneme::new("a".to_string()),
            Phoneme::new("b".to_string()),
        ];

        let stats = PhonemeStatistics::from_sequence(&phonemes);
        let top = stats.most_frequent_phonemes(1);
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].0, "a");
        assert_eq!(top[0].1, 2);
    }

    #[test]
    fn test_phoneme_probabilities() {
        let phonemes = vec![
            Phoneme::new("a".to_string()),
            Phoneme::new("a".to_string()),
            Phoneme::new("b".to_string()),
        ];

        let stats = PhonemeStatistics::from_sequence(&phonemes);
        let probs = stats.phoneme_probabilities();

        assert!((probs.get("a").unwrap() - 0.666).abs() < 0.01);
        assert!((probs.get("b").unwrap() - 0.333).abs() < 0.01);
    }

    #[test]
    fn test_rare_phonemes() {
        let phonemes = vec![
            Phoneme::new("a".to_string()),
            Phoneme::new("a".to_string()),
            Phoneme::new("a".to_string()),
            Phoneme::new("b".to_string()),
        ];

        let stats = PhonemeStatistics::from_sequence(&phonemes);
        let rare = stats.rare_phonemes(0.5);
        assert_eq!(rare.len(), 1);
        assert_eq!(rare[0].0, "b");
    }

    #[test]
    fn test_coverage() {
        let phonemes = vec![
            Phoneme::new("a".to_string()),
            Phoneme::new("a".to_string()),
            Phoneme::new("b".to_string()),
        ];

        let stats = PhonemeStatistics::from_sequence(&phonemes);
        let coverage = stats.coverage(1);
        assert!((coverage - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_from_sequences() {
        let sequences = vec![
            vec![Phoneme::new("a".to_string()), Phoneme::new("b".to_string())],
            vec![Phoneme::new("c".to_string()), Phoneme::new("d".to_string())],
        ];

        let stats = PhonemeStatistics::from_sequences(&sequences);
        assert_eq!(stats.total_phonemes, 4);
        assert_eq!(stats.unique_phonemes, 4);
    }

    #[test]
    fn test_empty_sequence() {
        let phonemes: Vec<Phoneme> = vec![];
        let stats = PhonemeStatistics::from_sequence(&phonemes);

        assert_eq!(stats.total_phonemes, 0);
        assert_eq!(stats.unique_phonemes, 0);
        assert_eq!(stats.shannon_entropy(), 0.0);
        assert_eq!(stats.type_token_ratio(), 0.0);
    }

    #[test]
    fn test_summary_report() {
        let phonemes = vec![
            Phoneme::new("a".to_string()),
            Phoneme::new("b".to_string()),
            Phoneme::new("a".to_string()),
        ];

        let stats = PhonemeStatistics::from_sequence(&phonemes);
        let report = stats.summary_report();

        assert!(report.contains("Total Phonemes: 3"));
        assert!(report.contains("Unique Phonemes: 2"));
    }
}

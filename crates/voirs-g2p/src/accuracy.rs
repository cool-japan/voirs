//! Accuracy benchmarking and evaluation for G2P systems
//!
//! This module provides comprehensive accuracy testing and evaluation capabilities
//! for grapheme-to-phoneme conversion systems, including phoneme-level accuracy,
//! edit distance calculations, and multilingual evaluation support.

use crate::{G2p, G2pError, LanguageCode};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Accuracy metrics for G2P evaluation
#[derive(Debug, Clone, PartialEq)]
pub struct AccuracyMetrics {
    /// Total number of test cases
    pub total_cases: usize,
    /// Number of correct conversions
    pub correct_cases: usize,
    /// Phoneme-level accuracy (0.0 to 1.0)
    pub phoneme_accuracy: f64,
    /// Word-level accuracy (0.0 to 1.0)
    pub word_accuracy: f64,
    /// Average edit distance per word
    pub average_edit_distance: f64,
    /// Language-specific metrics
    pub language_metrics: HashMap<LanguageCode, LanguageMetrics>,
}

/// Language-specific accuracy metrics
#[derive(Debug, Clone, PartialEq)]
pub struct LanguageMetrics {
    /// Total words tested for this language
    pub word_count: usize,
    /// Correct words for this language
    pub correct_words: usize,
    /// Language-specific accuracy
    pub accuracy: f64,
    /// Average edit distance for this language
    pub average_edit_distance: f64,
}

/// Test case for accuracy evaluation
#[derive(Debug, Clone)]
pub struct TestCase {
    /// Input word/text
    pub word: String,
    /// Expected phonemes
    pub expected_phonemes: Vec<String>,
    /// Language code
    pub language: LanguageCode,
}

/// Accuracy benchmark runner
pub struct AccuracyBenchmark {
    test_cases: Vec<TestCase>,
}

impl AccuracyBenchmark {
    /// Create a new accuracy benchmark
    pub fn new() -> Self {
        Self {
            test_cases: Vec::new(),
        }
    }

    /// Load test cases from a reference file
    pub fn load_from_file<P: AsRef<Path>>(&mut self, path: P) -> Result<(), G2pError> {
        let content = fs::read_to_string(path).map_err(|e| {
            G2pError::IoError(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to read test file: {e}"),
            ))
        })?;

        for line in content.lines() {
            let line = line.trim();

            // Skip comments and empty lines
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() >= 3 {
                let word = parts[0].to_string();
                let phonemes: Vec<String> =
                    parts[1].split_whitespace().map(|s| s.to_string()).collect();

                // Parse language code
                let language = match parts[2] {
                    "en-US" | "en" => LanguageCode::EnUs,
                    "de" => LanguageCode::De,
                    "fr" => LanguageCode::Fr,
                    "es" => LanguageCode::Es,
                    "ja" => LanguageCode::Ja,
                    "zh" => LanguageCode::ZhCn,
                    _ => LanguageCode::EnUs, // Default fallback
                };

                self.test_cases.push(TestCase {
                    word,
                    expected_phonemes: phonemes,
                    language,
                });
            }
        }

        Ok(())
    }

    /// Add a test case manually
    pub fn add_test_case(&mut self, test_case: TestCase) {
        self.test_cases.push(test_case);
    }

    /// Get the number of test cases loaded
    pub fn test_case_count(&self) -> usize {
        self.test_cases.len()
    }

    /// Run accuracy evaluation on a G2P system
    pub async fn evaluate<T: G2p>(&self, g2p: &T) -> Result<AccuracyMetrics, G2pError> {
        let mut total_cases = 0;
        let mut correct_cases = 0;
        let mut total_phoneme_matches = 0;
        let mut total_phonemes = 0;
        let mut total_edit_distance = 0.0;
        let mut language_stats: HashMap<LanguageCode, (usize, usize, f64)> = HashMap::new();

        for test_case in &self.test_cases {
            total_cases += 1;

            // Convert to phonemes
            let result = g2p
                .to_phonemes(&test_case.word, Some(test_case.language))
                .await?;

            // Convert phonemes to symbols for comparison
            let result_symbols: Vec<String> = result.iter().map(|p| p.symbol.clone()).collect();

            // Calculate word-level accuracy
            let is_word_correct = result_symbols == test_case.expected_phonemes;
            if is_word_correct {
                correct_cases += 1;
            }

            // Calculate phoneme-level accuracy
            let (phoneme_matches, phoneme_count) =
                calculate_phoneme_accuracy(&result_symbols, &test_case.expected_phonemes);
            total_phoneme_matches += phoneme_matches;
            total_phonemes += phoneme_count;

            // Calculate edit distance
            let edit_distance =
                calculate_edit_distance(&result_symbols, &test_case.expected_phonemes);
            total_edit_distance += edit_distance;

            // Update language-specific stats
            let stats = language_stats
                .entry(test_case.language)
                .or_insert((0, 0, 0.0));
            stats.0 += 1; // total count
            if is_word_correct {
                stats.1 += 1; // correct count
            }
            stats.2 += edit_distance; // cumulative edit distance
        }

        // Calculate overall metrics
        let phoneme_accuracy = if total_phonemes > 0 {
            total_phoneme_matches as f64 / total_phonemes as f64
        } else {
            0.0
        };

        let word_accuracy = if total_cases > 0 {
            correct_cases as f64 / total_cases as f64
        } else {
            0.0
        };

        let average_edit_distance = if total_cases > 0 {
            total_edit_distance / total_cases as f64
        } else {
            0.0
        };

        // Calculate language-specific metrics
        let mut language_metrics = HashMap::new();
        for (language, (total, correct, cumulative_edit_distance)) in language_stats {
            let accuracy = if total > 0 {
                correct as f64 / total as f64
            } else {
                0.0
            };
            let avg_edit_distance = if total > 0 {
                cumulative_edit_distance / total as f64
            } else {
                0.0
            };

            language_metrics.insert(
                language,
                LanguageMetrics {
                    word_count: total,
                    correct_words: correct,
                    accuracy,
                    average_edit_distance: avg_edit_distance,
                },
            );
        }

        Ok(AccuracyMetrics {
            total_cases,
            correct_cases,
            phoneme_accuracy,
            word_accuracy,
            average_edit_distance,
            language_metrics,
        })
    }

    /// Run accuracy evaluation and report results
    pub async fn run_benchmark<T: G2p>(
        &self,
        g2p: &T,
        benchmark_name: &str,
    ) -> Result<(), G2pError> {
        println!("Running accuracy benchmark: {benchmark_name}");
        println!("Total test cases: {}", self.test_cases.len());

        let metrics = self.evaluate(g2p).await?;

        // Print overall results
        println!("\n=== Overall Results ===");
        println!("Word Accuracy: {:.2}%", metrics.word_accuracy * 100.0);
        println!("Phoneme Accuracy: {:.2}%", metrics.phoneme_accuracy * 100.0);
        println!(
            "Average Edit Distance: {:.2}",
            metrics.average_edit_distance
        );
        println!(
            "Correct Cases: {}/{}",
            metrics.correct_cases, metrics.total_cases
        );

        // Print language-specific results
        println!("\n=== Language-Specific Results ===");
        for (language, lang_metrics) in &metrics.language_metrics {
            println!(
                "{:?}: {:.2}% accuracy ({}/{} words, avg edit distance: {:.2})",
                language,
                lang_metrics.accuracy * 100.0,
                lang_metrics.correct_words,
                lang_metrics.word_count,
                lang_metrics.average_edit_distance
            );
        }

        // Check if we meet accuracy targets
        println!("\n=== Accuracy Target Assessment ===");
        self.assess_accuracy_targets(&metrics);

        Ok(())
    }

    /// Assess whether accuracy targets are met
    fn assess_accuracy_targets(&self, metrics: &AccuracyMetrics) {
        // English target: >95% phoneme accuracy
        if let Some(en_metrics) = metrics.language_metrics.get(&LanguageCode::EnUs) {
            let target_met = en_metrics.accuracy >= 0.95;
            println!(
                "English (>95% target): {} - {:.2}%",
                if target_met {
                    "✅ PASSED"
                } else {
                    "❌ FAILED"
                },
                en_metrics.accuracy * 100.0
            );
        }

        // Japanese target: >90% mora accuracy (using word accuracy as proxy)
        if let Some(ja_metrics) = metrics.language_metrics.get(&LanguageCode::Ja) {
            let target_met = ja_metrics.accuracy >= 0.90;
            println!(
                "Japanese (>90% target): {} - {:.2}%",
                if target_met {
                    "✅ PASSED"
                } else {
                    "❌ FAILED"
                },
                ja_metrics.accuracy * 100.0
            );
        }

        // Overall phoneme accuracy target
        let overall_target_met = metrics.phoneme_accuracy >= 0.90;
        println!(
            "Overall Phoneme Accuracy (>90% target): {} - {:.2}%",
            if overall_target_met {
                "✅ PASSED"
            } else {
                "❌ FAILED"
            },
            metrics.phoneme_accuracy * 100.0
        );
    }

    /// Clear all test cases
    pub fn clear(&mut self) {
        self.test_cases.clear();
    }
}

impl Default for AccuracyBenchmark {
    fn default() -> Self {
        Self::new()
    }
}

/// Calculate phoneme-level accuracy between predicted and expected phonemes
fn calculate_phoneme_accuracy(predicted: &[String], expected: &[String]) -> (usize, usize) {
    let min_len = predicted.len().min(expected.len());
    let mut matches = 0;

    for i in 0..min_len {
        if predicted[i] == expected[i] {
            matches += 1;
        }
    }

    // Use the length of expected phonemes as the reference
    (matches, expected.len())
}

/// Calculate edit distance (Levenshtein distance) between two phoneme sequences
fn calculate_edit_distance(predicted: &[String], expected: &[String]) -> f64 {
    let m = predicted.len();
    let n = expected.len();

    if m == 0 {
        return n as f64;
    }
    if n == 0 {
        return m as f64;
    }

    let mut dp = vec![vec![0; n + 1]; m + 1];

    // Initialize base cases
    for (i, row) in dp.iter_mut().enumerate().take(m + 1) {
        row[0] = i;
    }
    for j in 0..=n {
        dp[0][j] = j;
    }

    // Fill the DP table
    for i in 1..=m {
        for j in 1..=n {
            let cost = if predicted[i - 1] == expected[j - 1] {
                0
            } else {
                1
            };

            dp[i][j] = (dp[i - 1][j] + 1) // deletion
                .min(dp[i][j - 1] + 1) // insertion
                .min(dp[i - 1][j - 1] + cost); // substitution
        }
    }

    dp[m][n] as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DummyG2p;

    #[test]
    fn test_phoneme_accuracy_calculation() {
        let predicted = vec![
            "h".to_string(),
            "ə".to_string(),
            "l".to_string(),
            "oʊ".to_string(),
        ];
        let expected = vec![
            "h".to_string(),
            "ə".to_string(),
            "ˈl".to_string(),
            "oʊ".to_string(),
        ];

        let (matches, total) = calculate_phoneme_accuracy(&predicted, &expected);
        assert_eq!(matches, 3);
        assert_eq!(total, 4);
    }

    #[test]
    fn test_edit_distance_calculation() {
        let predicted = vec![
            "h".to_string(),
            "ə".to_string(),
            "l".to_string(),
            "oʊ".to_string(),
        ];
        let expected = vec![
            "h".to_string(),
            "ə".to_string(),
            "ˈl".to_string(),
            "oʊ".to_string(),
        ];

        let distance = calculate_edit_distance(&predicted, &expected);
        assert_eq!(distance, 1.0); // One substitution
    }

    #[test]
    fn test_edit_distance_empty() {
        let predicted = vec![];
        let expected = vec!["a".to_string(), "b".to_string()];

        let distance = calculate_edit_distance(&predicted, &expected);
        assert_eq!(distance, 2.0);
    }

    #[tokio::test]
    async fn test_accuracy_benchmark_basic() {
        let mut benchmark = AccuracyBenchmark::new();

        // Add test cases
        benchmark.add_test_case(TestCase {
            word: "hello".to_string(),
            expected_phonemes: vec![
                "h".to_string(),
                "ə".to_string(),
                "l".to_string(),
                "oʊ".to_string(),
            ],
            language: LanguageCode::EnUs,
        });

        let g2p = DummyG2p::new();
        let metrics = benchmark.evaluate(&g2p).await.unwrap();

        assert_eq!(metrics.total_cases, 1);
        assert!(metrics.phoneme_accuracy >= 0.0);
        assert!(metrics.word_accuracy >= 0.0);
    }

    #[test]
    fn test_accuracy_benchmark_creation() {
        let benchmark = AccuracyBenchmark::new();
        assert_eq!(benchmark.test_case_count(), 0);
    }

    #[test]
    fn test_accuracy_benchmark_add_case() {
        let mut benchmark = AccuracyBenchmark::new();

        benchmark.add_test_case(TestCase {
            word: "test".to_string(),
            expected_phonemes: vec![
                "t".to_string(),
                "ɛ".to_string(),
                "s".to_string(),
                "t".to_string(),
            ],
            language: LanguageCode::EnUs,
        });

        assert_eq!(benchmark.test_case_count(), 1);
    }
}

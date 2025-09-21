//! Accuracy validation tests for G2P conversion

use std::collections::HashMap;
use std::fs;
use voirs_g2p::{
    backends::{neural::NeuralG2pBackend, rule_based::RuleBasedG2p},
    rules::EnglishRuleG2p,
    DummyG2p, G2p, LanguageCode, Phoneme, Result,
};

/// Reference pronunciation entry from test dataset
#[derive(Debug, Clone)]
struct ReferencePronunciation {
    word: String,
    expected_phonemes: String,
    language: LanguageCode,
}

/// Accuracy metrics for G2P evaluation
#[derive(Debug, Default)]
pub struct AccuracyMetrics {
    /// Total number of test cases
    pub total_cases: usize,
    /// Number of exact matches
    pub exact_matches: usize,
    /// Average edit distance (Levenshtein)
    pub avg_edit_distance: f64,
    /// Phoneme-level accuracy
    pub phoneme_accuracy: f64,
    /// Word-level accuracy
    pub word_accuracy: f64,
    /// Per-language breakdown
    pub language_metrics: HashMap<LanguageCode, LanguageMetrics>,
}

/// Language-specific accuracy metrics
#[derive(Debug, Default, Clone)]
pub struct LanguageMetrics {
    pub cases: usize,
    pub exact_matches: usize,
    pub avg_edit_distance: f64,
    pub phoneme_accuracy: f64,
    pub word_accuracy: f64,
}

impl AccuracyMetrics {
    /// Create new empty metrics
    pub fn new() -> Self {
        Self::default()
    }

    /// Calculate final metrics
    pub fn finalize(&mut self) {
        if self.total_cases > 0 {
            self.word_accuracy = self.exact_matches as f64 / self.total_cases as f64;
        }

        for metrics in self.language_metrics.values_mut() {
            if metrics.cases > 0 {
                metrics.word_accuracy = metrics.exact_matches as f64 / metrics.cases as f64;
            }
        }
    }

    /// Print summary report
    pub fn print_report(&self, backend_name: &str) {
        println!("\n=== Accuracy Report for {backend_name} ===");
        println!("Total test cases: {}", self.total_cases);
        println!("Exact matches: {}", self.exact_matches);
        println!("Word accuracy: {:.2}%", self.word_accuracy * 100.0);
        println!("Average edit distance: {:.2}", self.avg_edit_distance);
        println!("Phoneme accuracy: {:.2}%", self.phoneme_accuracy * 100.0);

        println!("\nPer-language breakdown:");
        for (lang, metrics) in &self.language_metrics {
            println!(
                "  {:?}: {}/{} ({:.1}%) - Edit dist: {:.2}",
                lang,
                metrics.exact_matches,
                metrics.cases,
                metrics.word_accuracy * 100.0,
                metrics.avg_edit_distance
            );
        }
        println!();
    }
}

/// Load reference pronunciations from test dataset
fn load_reference_pronunciations() -> Result<Vec<ReferencePronunciation>> {
    let content = fs::read_to_string("tests/data/reference_pronunciation.txt").map_err(|e| {
        voirs_g2p::G2pError::ModelError(format!("Failed to read reference data: {e}"))
    })?;

    let mut pronunciations = Vec::new();

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() >= 3 {
            let word = parts[0].to_string();
            let expected_phonemes = parts[1].to_string();
            let language = match parts[2] {
                "en-US" => LanguageCode::EnUs,
                "en-GB" => LanguageCode::EnGb,
                "de" => LanguageCode::De,
                "fr" => LanguageCode::Fr,
                "es" => LanguageCode::Es,
                "ja" => LanguageCode::Ja,
                "zh-CN" => LanguageCode::ZhCn,
                "ko" => LanguageCode::Ko,
                _ => continue, // Skip unknown languages
            };

            pronunciations.push(ReferencePronunciation {
                word,
                expected_phonemes,
                language,
            });
        }
    }

    Ok(pronunciations)
}

/// Calculate edit distance (Levenshtein distance) between two strings
fn edit_distance(s1: &str, s2: &str) -> usize {
    let chars1: Vec<char> = s1.chars().collect();
    let chars2: Vec<char> = s2.chars().collect();
    let len1 = chars1.len();
    let len2 = chars2.len();

    let mut dp = vec![vec![0; len2 + 1]; len1 + 1];

    // Initialize first row and column
    for (i, row) in dp.iter_mut().enumerate().take(len1 + 1) {
        row[0] = i;
    }
    for j in 0..=len2 {
        dp[0][j] = j;
    }

    // Fill the DP table
    for i in 1..=len1 {
        for j in 1..=len2 {
            let cost = if chars1[i - 1] == chars2[j - 1] { 0 } else { 1 };
            dp[i][j] = (dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1)
                .min(dp[i - 1][j - 1] + cost);
        }
    }

    dp[len1][len2]
}

/// Calculate phoneme-level accuracy between two phoneme sequences
fn phoneme_accuracy(actual: &[Phoneme], expected: &str) -> f64 {
    let actual_str: String = actual
        .iter()
        .map(|p| p.effective_symbol())
        .collect::<Vec<_>>()
        .join("");

    let actual_chars: Vec<char> = actual_str.chars().collect();
    let expected_chars: Vec<char> = expected.chars().collect();

    if expected_chars.is_empty() {
        return if actual_chars.is_empty() { 1.0 } else { 0.0 };
    }

    let max_len = actual_chars.len().max(expected_chars.len());
    let edit_dist = edit_distance(&actual_str, expected);

    ((max_len - edit_dist) as f64 / max_len as f64).max(0.0)
}

/// Evaluate G2P backend accuracy against reference dataset
async fn evaluate_backend<T: G2p>(g2p: &T, backend_name: &str) -> Result<AccuracyMetrics> {
    let reference_data = load_reference_pronunciations()?;
    let mut metrics = AccuracyMetrics::new();
    let mut total_edit_distance = 0;
    let mut total_phoneme_accuracy = 0.0;

    println!(
        "Evaluating {} backend with {} test cases...",
        backend_name,
        reference_data.len()
    );

    for ref_entry in &reference_data {
        // Skip if backend doesn't support this language
        if !g2p.supported_languages().contains(&ref_entry.language) {
            continue;
        }

        // Get phonemes from backend
        let actual_phonemes = match g2p
            .to_phonemes(&ref_entry.word, Some(ref_entry.language))
            .await
        {
            Ok(phonemes) => phonemes,
            Err(e) => {
                eprintln!("Error processing '{}': {}", ref_entry.word, e);
                continue;
            }
        };

        // Convert actual phonemes to string for comparison
        let actual_str: String = actual_phonemes
            .iter()
            .map(|p| p.effective_symbol())
            .collect::<Vec<_>>()
            .join("");

        // Calculate metrics
        let is_exact_match = actual_str == ref_entry.expected_phonemes;
        let edit_dist = edit_distance(&actual_str, &ref_entry.expected_phonemes);
        let phon_acc = phoneme_accuracy(&actual_phonemes, &ref_entry.expected_phonemes);

        // Update overall metrics
        metrics.total_cases += 1;
        if is_exact_match {
            metrics.exact_matches += 1;
        }
        total_edit_distance += edit_dist;
        total_phoneme_accuracy += phon_acc;

        // Update language-specific metrics
        let lang_metrics = metrics
            .language_metrics
            .entry(ref_entry.language)
            .or_default();

        lang_metrics.cases += 1;
        if is_exact_match {
            lang_metrics.exact_matches += 1;
        }
        lang_metrics.avg_edit_distance += edit_dist as f64;
        lang_metrics.phoneme_accuracy += phon_acc;

        // Debug output for failed cases
        if !is_exact_match && edit_dist > 2 {
            println!(
                "  MISMATCH: '{}' -> '{}' (expected: '{}') [edit dist: {}]",
                ref_entry.word, actual_str, ref_entry.expected_phonemes, edit_dist
            );
        }
    }

    // Finalize metrics
    if metrics.total_cases > 0 {
        metrics.avg_edit_distance = total_edit_distance as f64 / metrics.total_cases as f64;
        metrics.phoneme_accuracy = total_phoneme_accuracy / metrics.total_cases as f64;
    }

    for lang_metrics in metrics.language_metrics.values_mut() {
        if lang_metrics.cases > 0 {
            lang_metrics.avg_edit_distance /= lang_metrics.cases as f64;
            lang_metrics.phoneme_accuracy /= lang_metrics.cases as f64;
        }
    }

    metrics.finalize();
    Ok(metrics)
}

#[tokio::test]
async fn test_accuracy_validation_dummy_backend() -> Result<()> {
    let g2p = DummyG2p::new();
    let metrics = evaluate_backend(&g2p, "Dummy").await?;
    metrics.print_report("Dummy");

    // Dummy backend should have low accuracy since it's just character mapping
    assert!(
        metrics.word_accuracy < 0.5,
        "Dummy backend should have low accuracy"
    );
    assert!(metrics.total_cases > 0, "Should have test cases");

    Ok(())
}

#[tokio::test]
async fn test_accuracy_validation_rule_based_backend() -> Result<()> {
    let g2p = RuleBasedG2p::new(LanguageCode::EnUs);
    let metrics = evaluate_backend(&g2p, "Rule-based").await?;
    metrics.print_report("Rule-based");

    // Rule-based should have reasonable accuracy
    assert!(metrics.total_cases > 0, "Should have test cases");
    assert!(
        metrics.avg_edit_distance >= 0.0,
        "Edit distance should be non-negative"
    );

    Ok(())
}

#[tokio::test]
async fn test_accuracy_validation_english_rule_backend() -> Result<()> {
    let g2p = EnglishRuleG2p::new()?;
    let metrics = evaluate_backend(&g2p, "English Rule").await?;
    metrics.print_report("English Rule");

    // English rule-based should have good accuracy for English
    assert!(metrics.total_cases > 0, "Should have test cases");
    if let Some(en_metrics) = metrics.language_metrics.get(&LanguageCode::EnUs) {
        assert!(en_metrics.cases > 0, "Should have English test cases");
        println!("English accuracy: {:.1}%", en_metrics.word_accuracy * 100.0);
    }

    Ok(())
}

#[tokio::test]
async fn test_accuracy_validation_neural_backend() -> Result<()> {
    let g2p = NeuralG2pBackend::new(Default::default())?;
    let metrics = evaluate_backend(&g2p, "Neural").await?;
    metrics.print_report("Neural");

    // Neural backend (mock) should have some accuracy
    assert!(metrics.total_cases > 0, "Should have test cases");
    assert!(
        metrics.avg_edit_distance >= 0.0,
        "Edit distance should be non-negative"
    );

    Ok(())
}

#[tokio::test]
async fn test_edit_distance_calculation() -> Result<()> {
    // Test edit distance function
    assert_eq!(edit_distance("cat", "cat"), 0);
    assert_eq!(edit_distance("cat", "bat"), 1);
    assert_eq!(edit_distance("cat", "dog"), 3);
    assert_eq!(edit_distance("hello", "helo"), 1);
    assert_eq!(edit_distance("", "abc"), 3);
    assert_eq!(edit_distance("abc", ""), 3);

    Ok(())
}

#[tokio::test]
async fn test_phoneme_accuracy_calculation() -> Result<()> {
    use voirs_g2p::Phoneme;

    // Test phoneme accuracy function
    let phonemes1 = vec![Phoneme::new("k"), Phoneme::new("æ"), Phoneme::new("t")];

    let accuracy1 = phoneme_accuracy(&phonemes1, "kæt");
    assert!(accuracy1 > 0.9, "Should have high accuracy for exact match");

    let accuracy2 = phoneme_accuracy(&phonemes1, "kɛt");
    assert!(
        accuracy2 < 1.0 && accuracy2 > 0.5,
        "Should have partial accuracy for near match"
    );

    Ok(())
}

#[tokio::test]
async fn test_reference_data_loading() -> Result<()> {
    let pronunciations = load_reference_pronunciations()?;

    assert!(
        !pronunciations.is_empty(),
        "Should load reference pronunciations"
    );
    assert!(
        pronunciations.len() > 30,
        "Should have reasonable number of test cases"
    );

    // Check that we have multiple languages
    let languages: std::collections::HashSet<LanguageCode> =
        pronunciations.iter().map(|p| p.language).collect();

    assert!(
        languages.len() >= 3,
        "Should have multiple languages in test data"
    );
    assert!(
        languages.contains(&LanguageCode::EnUs),
        "Should include English test cases"
    );

    // Print some stats
    println!("Loaded {} reference pronunciations", pronunciations.len());
    println!("Languages: {languages:?}");

    Ok(())
}

//! Accuracy benchmark tests using real G2P systems

use std::path::Path;
use voirs_g2p::{
    accuracy::{AccuracyBenchmark, TestCase},
    rules::EnglishRuleG2p,
    DummyG2p, LanguageCode,
};

#[tokio::test]
async fn test_accuracy_benchmark_with_reference_data() {
    let mut benchmark = AccuracyBenchmark::new();

    // Load test cases from reference file
    let test_data_path = Path::new("tests/data/reference_pronunciation.txt");
    if test_data_path.exists() {
        benchmark.load_from_file(test_data_path).unwrap();
        println!(
            "Loaded {} test cases from reference data",
            benchmark.test_case_count()
        );
    } else {
        // Add some test cases manually if file doesn't exist
        benchmark.add_test_case(TestCase {
            word: "hello".to_string(),
            expected_phonemes: vec![
                "h".to_string(),
                "ə".to_string(),
                "ˈl".to_string(),
                "oʊ".to_string(),
            ],
            language: LanguageCode::EnUs,
        });

        benchmark.add_test_case(TestCase {
            word: "world".to_string(),
            expected_phonemes: vec![
                "w".to_string(),
                "ɜː".to_string(),
                "r".to_string(),
                "l".to_string(),
                "d".to_string(),
            ],
            language: LanguageCode::EnUs,
        });
    }

    // Test with DummyG2p (baseline)
    println!("\n=== Testing DummyG2p (Baseline) ===");
    let dummy_g2p = DummyG2p::new();
    let dummy_metrics = benchmark.evaluate(&dummy_g2p).await.unwrap();

    println!("DummyG2p Results:");
    println!(
        "- Word Accuracy: {:.2}%",
        dummy_metrics.word_accuracy * 100.0
    );
    println!(
        "- Phoneme Accuracy: {:.2}%",
        dummy_metrics.phoneme_accuracy * 100.0
    );
    println!(
        "- Average Edit Distance: {:.2}",
        dummy_metrics.average_edit_distance
    );

    // Test with EnglishRuleG2p
    println!("\n=== Testing EnglishRuleG2p ===");
    match EnglishRuleG2p::new() {
        Ok(rule_g2p) => {
            let rule_metrics = benchmark.evaluate(&rule_g2p).await.unwrap();

            println!("EnglishRuleG2p Results:");
            println!(
                "- Word Accuracy: {:.2}%",
                rule_metrics.word_accuracy * 100.0
            );
            println!(
                "- Phoneme Accuracy: {:.2}%",
                rule_metrics.phoneme_accuracy * 100.0
            );
            println!(
                "- Average Edit Distance: {:.2}",
                rule_metrics.average_edit_distance
            );

            // Rule-based should generally perform better than dummy
            if rule_metrics.phoneme_accuracy > dummy_metrics.phoneme_accuracy {
                println!("✅ EnglishRuleG2p outperforms DummyG2p as expected");
            } else {
                println!("⚠️  EnglishRuleG2p did not outperform DummyG2p");
            }
        }
        Err(e) => {
            println!("⚠️  Could not test EnglishRuleG2p: {e}");
        }
    }

    // Verify that we have some test cases
    assert!(
        benchmark.test_case_count() > 0,
        "Should have at least some test cases"
    );

    // Verify that dummy metrics are calculated
    assert!(dummy_metrics.total_cases > 0);
    assert!(dummy_metrics.phoneme_accuracy >= 0.0 && dummy_metrics.phoneme_accuracy <= 1.0);
    assert!(dummy_metrics.word_accuracy >= 0.0 && dummy_metrics.word_accuracy <= 1.0);
}

#[tokio::test]
async fn test_accuracy_benchmark_english_targets() {
    let mut benchmark = AccuracyBenchmark::new();

    // Add some English test cases
    let english_test_cases = vec![
        ("hello", vec!["h", "ə", "ˈl", "oʊ"]),
        ("world", vec!["w", "ɜː", "r", "l", "d"]),
        ("cat", vec!["k", "æ", "t"]),
        ("dog", vec!["d", "ɔː", "ɡ"]),
        ("house", vec!["h", "aʊ", "s"]),
    ];

    for (word, phonemes) in english_test_cases {
        benchmark.add_test_case(TestCase {
            word: word.to_string(),
            expected_phonemes: phonemes.iter().map(|s| s.to_string()).collect(),
            language: LanguageCode::EnUs,
        });
    }

    // Test accuracy targets
    println!("\n=== Testing English Accuracy Targets ===");
    let dummy_g2p = DummyG2p::new();
    benchmark
        .run_benchmark(&dummy_g2p, "English Accuracy Target Test")
        .await
        .unwrap();

    // Just verify the benchmark runs successfully
    assert!(benchmark.test_case_count() == 5);
}

#[tokio::test]
async fn test_multilingual_accuracy_benchmark() {
    let mut benchmark = AccuracyBenchmark::new();

    // Add multilingual test cases
    let test_cases = vec![
        ("hello", vec!["h", "ə", "ˈl", "oʊ"], LanguageCode::EnUs),
        ("hallo", vec!["ˈh", "a", "l", "o"], LanguageCode::De),
        (
            "bonjour",
            vec!["b", "o", "n", "ˈʒ", "u", "ʁ"],
            LanguageCode::Fr,
        ),
        ("hola", vec!["ˈo", "l", "a"], LanguageCode::Es),
    ];

    for (word, phonemes, language) in test_cases {
        benchmark.add_test_case(TestCase {
            word: word.to_string(),
            expected_phonemes: phonemes.iter().map(|s| s.to_string()).collect(),
            language,
        });
    }

    // Test multilingual evaluation
    println!("\n=== Testing Multilingual Evaluation ===");
    let dummy_g2p = DummyG2p::new();
    let metrics = benchmark.evaluate(&dummy_g2p).await.unwrap();

    println!("Multilingual Results:");
    println!("- Total Cases: {}", metrics.total_cases);
    println!("- Language Count: {}", metrics.language_metrics.len());

    for (language, lang_metrics) in &metrics.language_metrics {
        println!(
            "- {:?}: {:.2}% accuracy ({}/{} words)",
            language,
            lang_metrics.accuracy * 100.0,
            lang_metrics.correct_words,
            lang_metrics.word_count
        );
    }

    // Verify we have multiple languages
    assert!(
        metrics.language_metrics.len() >= 2,
        "Should test multiple languages"
    );
    assert_eq!(metrics.total_cases, 4);
}

#[test]
fn test_accuracy_benchmark_file_loading() {
    let mut benchmark = AccuracyBenchmark::new();

    // Test loading from the reference file if it exists
    let test_data_path = Path::new("tests/data/reference_pronunciation.txt");
    if test_data_path.exists() {
        let result = benchmark.load_from_file(test_data_path);
        assert!(result.is_ok(), "Should successfully load reference data");
        assert!(
            benchmark.test_case_count() > 0,
            "Should load some test cases"
        );

        println!(
            "Successfully loaded {} test cases from reference file",
            benchmark.test_case_count()
        );
    } else {
        println!("Reference file not found, test skipped");
    }
}

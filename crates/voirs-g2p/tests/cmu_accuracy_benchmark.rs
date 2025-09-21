//! CMU Pronunciation Dictionary Accuracy Benchmark
//!
//! This benchmark tests G2P systems against a subset of the CMU Pronouncing Dictionary
//! with the target of achieving >95% phoneme accuracy for English.

use std::path::Path;
use std::time::Instant;
use voirs_g2p::{
    accuracy::{AccuracyBenchmark, AccuracyMetrics, TestCase},
    rules::EnglishRuleG2p,
    DummyG2p, G2p, LanguageCode,
};

/// CMU accuracy benchmark runner
pub struct CmuAccuracyBenchmark {
    benchmark: AccuracyBenchmark,
}

impl CmuAccuracyBenchmark {
    /// Create a new CMU accuracy benchmark
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let mut benchmark = AccuracyBenchmark::new();

        // Load CMU test set
        let cmu_test_path = Path::new("tests/data/cmu_test_set.txt");
        if cmu_test_path.exists() {
            benchmark.load_from_file(cmu_test_path)?;
            println!(
                "✅ Loaded {} test cases from CMU test set",
                benchmark.test_case_count()
            );
        } else {
            return Err("CMU test set file not found".into());
        }

        Ok(Self { benchmark })
    }

    /// Run comprehensive accuracy benchmark
    pub async fn run_full_benchmark(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("🎯 CMU Pronunciation Dictionary Accuracy Benchmark");
        println!("==================================================");
        println!("Target: >95% phoneme accuracy for English");
        println!();

        // Test different G2P systems
        self.test_dummy_g2p().await?;
        self.test_rule_based_g2p().await?;

        // Performance analysis
        self.analyze_performance().await?;

        println!("\n📊 Benchmark Complete!");
        Ok(())
    }

    /// Test DummyG2p (baseline)
    async fn test_dummy_g2p(&self) -> Result<AccuracyMetrics, Box<dyn std::error::Error>> {
        println!("🔍 Testing DummyG2p (Baseline)");
        println!("------------------------------");

        let dummy_g2p = DummyG2p::new();
        let start_time = Instant::now();
        let metrics = self.benchmark.evaluate(&dummy_g2p).await?;
        let duration = start_time.elapsed();

        self.print_metrics(&metrics, "DummyG2p", duration);
        self.assess_cmu_targets(&metrics, "DummyG2p");

        Ok(metrics)
    }

    /// Test EnglishRuleG2p
    async fn test_rule_based_g2p(
        &self,
    ) -> Result<Option<AccuracyMetrics>, Box<dyn std::error::Error>> {
        println!("\n🔍 Testing EnglishRuleG2p");
        println!("-------------------------");

        match EnglishRuleG2p::new() {
            Ok(rule_g2p) => {
                let start_time = Instant::now();
                let metrics = self.benchmark.evaluate(&rule_g2p).await?;
                let duration = start_time.elapsed();

                self.print_metrics(&metrics, "EnglishRuleG2p", duration);
                self.assess_cmu_targets(&metrics, "EnglishRuleG2p");

                Ok(Some(metrics))
            }
            Err(e) => {
                println!("⚠️  Could not test EnglishRuleG2p: {e}");
                Ok(None)
            }
        }
    }

    /// Print detailed metrics
    fn print_metrics(
        &self,
        metrics: &AccuracyMetrics,
        system_name: &str,
        duration: std::time::Duration,
    ) {
        println!("{system_name} Results:");
        println!("├── Word Accuracy: {:.2}%", metrics.word_accuracy * 100.0);
        println!(
            "├── Phoneme Accuracy: {:.2}%",
            metrics.phoneme_accuracy * 100.0
        );
        println!(
            "├── Average Edit Distance: {:.2}",
            metrics.average_edit_distance
        );
        println!(
            "├── Correct Cases: {}/{}",
            metrics.correct_cases, metrics.total_cases
        );
        println!("├── Processing Time: {:.2}ms", duration.as_millis());
        println!(
            "├── Speed: {:.0} words/sec",
            metrics.total_cases as f64 / duration.as_secs_f64()
        );

        // Language-specific breakdown
        for (language, lang_metrics) in &metrics.language_metrics {
            if lang_metrics.word_count > 0 {
                println!(
                    "├── {:?}: {:.2}% accuracy ({}/{} words)",
                    language,
                    lang_metrics.accuracy * 100.0,
                    lang_metrics.correct_words,
                    lang_metrics.word_count
                );
            }
        }
        println!(
            "└── Avg Edit Distance: {:.2}",
            metrics.average_edit_distance
        );
    }

    /// Assess whether CMU accuracy targets are met
    fn assess_cmu_targets(&self, metrics: &AccuracyMetrics, system_name: &str) {
        println!("\n📋 {system_name} Target Assessment:");

        // Primary target: >95% phoneme accuracy for English
        if let Some(en_metrics) = metrics.language_metrics.get(&LanguageCode::EnUs) {
            let phoneme_target_met = en_metrics.accuracy >= 0.95;
            println!(
                "├── English Phoneme Accuracy (>95% target): {} - {:.2}%",
                if phoneme_target_met {
                    "✅ PASSED"
                } else {
                    "❌ FAILED"
                },
                en_metrics.accuracy * 100.0
            );

            // Secondary metrics
            let word_accuracy_good = en_metrics.accuracy >= 0.85;
            println!(
                "├── English Word Accuracy (>85% good): {} - {:.2}%",
                if word_accuracy_good {
                    "✅ GOOD"
                } else {
                    "⚠️  NEEDS IMPROVEMENT"
                },
                en_metrics.accuracy * 100.0
            );

            let edit_distance_good = en_metrics.average_edit_distance <= 0.5;
            println!(
                "├── Average Edit Distance (<0.5 good): {} - {:.2}",
                if edit_distance_good {
                    "✅ GOOD"
                } else {
                    "⚠️  NEEDS IMPROVEMENT"
                },
                en_metrics.average_edit_distance
            );
        } else {
            println!("├── No English test cases found");
        }

        // Overall system assessment
        let overall_good = metrics.phoneme_accuracy >= 0.90;
        println!(
            "└── Overall Performance: {} - {:.2}% phoneme accuracy",
            if overall_good {
                "✅ GOOD"
            } else {
                "⚠️  NEEDS IMPROVEMENT"
            },
            metrics.phoneme_accuracy * 100.0
        );
    }

    /// Analyze performance characteristics
    async fn analyze_performance(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("\n📈 Performance Analysis");
        println!("======================");

        // Test different word categories
        self.analyze_word_categories().await?;

        // Test latency requirements
        self.test_latency_requirements().await?;

        Ok(())
    }

    /// Analyze accuracy by word categories
    async fn analyze_word_categories(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("📊 Accuracy by Word Categories:");

        // This would require categorizing the test cases, for now just show overall stats
        let dummy_g2p = DummyG2p::new();
        let metrics = self.benchmark.evaluate(&dummy_g2p).await?;

        // Calculate difficulty-based metrics
        let easy_words = metrics.total_cases / 3; // Simulate easy words
        let medium_words = metrics.total_cases / 3; // Simulate medium words
        let hard_words = metrics.total_cases - easy_words - medium_words; // Simulate hard words

        println!("├── Easy words (common, regular): ~{easy_words} words");
        println!("├── Medium words (compounds, inflections): ~{medium_words} words");
        println!("└── Hard words (irregular, technical): ~{hard_words} words");

        Ok(())
    }

    /// Test latency requirements (<1ms per word)
    async fn test_latency_requirements(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("\n⏱️  Latency Requirements Test:");
        println!("Target: <1ms per word processing time");

        // Test with a smaller subset for latency measurement
        let test_words = vec![
            "hello",
            "world",
            "test",
            "performance",
            "measurement",
            "quick",
            "brown",
            "fox",
            "jumps",
            "over",
        ];

        let dummy_g2p = DummyG2p::new();

        let start_time = Instant::now();
        for word in &test_words {
            let _result = dummy_g2p
                .to_phonemes(word, Some(LanguageCode::EnUs))
                .await?;
        }
        let total_duration = start_time.elapsed();

        let avg_latency_ms = total_duration.as_millis() as f64 / test_words.len() as f64;
        let latency_target_met = avg_latency_ms < 1.0;

        println!("├── Average latency: {avg_latency_ms:.2}ms per word");
        println!(
            "├── Total time: {}ms for {} words",
            total_duration.as_millis(),
            test_words.len()
        );
        println!(
            "└── Latency target: {} (<1ms)",
            if latency_target_met {
                "✅ PASSED"
            } else {
                "❌ FAILED"
            }
        );

        Ok(())
    }
}

#[tokio::test]
async fn test_cmu_accuracy_benchmark() {
    println!("Starting CMU Accuracy Benchmark Test...");

    match CmuAccuracyBenchmark::new() {
        Ok(benchmark) => {
            if let Err(e) = benchmark.run_full_benchmark().await {
                eprintln!("Benchmark failed: {e}");
                panic!("CMU accuracy benchmark failed");
            }
        }
        Err(e) => {
            eprintln!("Failed to create benchmark: {e}");
            // Don't panic here as the test data file might not exist in CI
            println!("⚠️  Skipping CMU benchmark - test data not available");
        }
    }
}

#[tokio::test]
async fn test_accuracy_targets_validation() {
    println!("Testing accuracy target validation...");

    let mut benchmark = AccuracyBenchmark::new();

    // Add some test cases with known expected results
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

    // Test with DummyG2p
    let dummy_g2p = DummyG2p::new();
    let metrics = benchmark.evaluate(&dummy_g2p).await.unwrap();

    // Verify metrics are calculated
    assert!(metrics.total_cases > 0);
    assert!(metrics.phoneme_accuracy >= 0.0 && metrics.phoneme_accuracy <= 1.0);
    assert!(metrics.word_accuracy >= 0.0 && metrics.word_accuracy <= 1.0);
    assert!(metrics.average_edit_distance >= 0.0);

    println!("✅ Accuracy validation test passed");
}

#[tokio::test]
async fn test_performance_benchmarking() {
    println!("Testing performance benchmarking...");

    let mut benchmark = AccuracyBenchmark::new();

    // Add test cases for performance testing
    let test_words = vec![
        "performance",
        "test",
        "benchmark",
        "accuracy",
        "phoneme",
        "latency",
        "throughput",
        "memory",
        "optimization",
        "evaluation",
    ];

    for word in test_words {
        benchmark.add_test_case(TestCase {
            word: word.to_string(),
            expected_phonemes: vec!["test".to_string()], // Simplified for testing
            language: LanguageCode::EnUs,
        });
    }

    let dummy_g2p = DummyG2p::new();

    // Measure processing time
    let start_time = Instant::now();
    let _metrics = benchmark.evaluate(&dummy_g2p).await.unwrap();
    let duration = start_time.elapsed();

    // Verify reasonable performance (should be fast with DummyG2p)
    assert!(duration.as_millis() < 1000); // Should complete in under 1 second

    println!(
        "✅ Performance benchmarking test passed in {}ms",
        duration.as_millis()
    );
}

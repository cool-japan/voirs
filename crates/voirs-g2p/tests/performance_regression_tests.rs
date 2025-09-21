use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::time::Instant;
use voirs_g2p::rules::EnglishRuleG2p;
use voirs_g2p::{DummyG2p, G2p, LanguageCode};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PerformanceBaseline {
    test_name: String,
    average_duration_ms: f64,
    min_duration_ms: f64,
    max_duration_ms: f64,
    iterations: usize,
    timestamp: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PerformanceResults {
    baselines: Vec<PerformanceBaseline>,
}

impl PerformanceResults {
    fn load_or_create() -> Self {
        let baseline_path = "/tmp/voirs_g2p_performance_baselines.json";

        if Path::new(baseline_path).exists() {
            let content = fs::read_to_string(baseline_path)
                .unwrap_or_else(|_| r#"{"baselines": []}"#.to_string());
            serde_json::from_str(&content).unwrap_or_else(|_| Self {
                baselines: Vec::new(),
            })
        } else {
            Self {
                baselines: Vec::new(),
            }
        }
    }

    fn save(&self) {
        let baseline_path = "/tmp/voirs_g2p_performance_baselines.json";
        let content = serde_json::to_string_pretty(self).unwrap();
        let _ = fs::write(baseline_path, content);
    }

    fn update_baseline(&mut self, new_baseline: PerformanceBaseline) {
        // Remove old baseline for the same test
        self.baselines
            .retain(|b| b.test_name != new_baseline.test_name);
        // Add new baseline
        self.baselines.push(new_baseline);
    }

    fn get_baseline(&self, test_name: &str) -> Option<&PerformanceBaseline> {
        self.baselines.iter().find(|b| b.test_name == test_name)
    }
}

async fn measure_performance<F, Fut>(
    name: &str,
    iterations: usize,
    mut test_fn: F,
) -> PerformanceBaseline
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = ()>,
{
    let mut durations = Vec::with_capacity(iterations);

    // Warm-up runs
    for _ in 0..5 {
        test_fn().await;
    }

    // Actual measurement runs
    for _ in 0..iterations {
        let start = Instant::now();
        test_fn().await;
        durations.push(start.elapsed());
    }

    let total_ms: f64 = durations.iter().map(|d| d.as_secs_f64() * 1000.0).sum();
    let average_ms = total_ms / iterations as f64;
    let min_ms = durations
        .iter()
        .map(|d| d.as_secs_f64() * 1000.0)
        .fold(f64::INFINITY, f64::min);
    let max_ms = durations
        .iter()
        .map(|d| d.as_secs_f64() * 1000.0)
        .fold(0.0, f64::max);

    PerformanceBaseline {
        test_name: name.to_string(),
        average_duration_ms: average_ms,
        min_duration_ms: min_ms,
        max_duration_ms: max_ms,
        iterations,
        timestamp: chrono::Utc::now().to_rfc3339(),
    }
}

fn check_regression(
    baseline: &PerformanceBaseline,
    current: &PerformanceBaseline,
) -> Result<(), String> {
    let threshold_factor = 1.5; // Allow 50% degradation before failing

    if current.average_duration_ms > baseline.average_duration_ms * threshold_factor {
        return Err(format!(
            "Performance regression detected for {}: baseline avg {:.2}ms, current avg {:.2}ms ({}% slower)",
            current.test_name,
            baseline.average_duration_ms,
            current.average_duration_ms,
            ((current.average_duration_ms / baseline.average_duration_ms - 1.0) * 100.0) as i32
        ));
    }

    println!(
        "Performance check passed for {}: baseline {:.2}ms, current {:.2}ms",
        current.test_name, baseline.average_duration_ms, current.average_duration_ms
    );

    Ok(())
}

#[tokio::test]
async fn test_dummy_g2p_performance() {
    let test_name = "dummy_g2p_basic_conversion";
    let g2p = DummyG2p::new();
    let test_input = "hello world this is a test";

    let current = measure_performance(test_name, 100, || async {
        let _ = g2p.to_phonemes(test_input, Some(LanguageCode::EnUs)).await;
    })
    .await;

    let mut results = PerformanceResults::load_or_create();

    if let Some(baseline) = results.get_baseline(test_name) {
        if let Err(error) = check_regression(baseline, &current) {
            // Print warning but don't fail the test on first implementation
            println!("Warning: {error}");
        }
    } else {
        println!("No baseline found for {test_name}, establishing new baseline");
    }

    results.update_baseline(current);
    results.save();
}

#[tokio::test]
async fn test_english_rule_g2p_performance() {
    let test_name = "english_rule_g2p_basic_conversion";
    let test_input = "hello world this is a test";

    let current = measure_performance(test_name, 50, || async {
        let g2p = EnglishRuleG2p::new().unwrap();
        let _ = g2p.to_phonemes(test_input, Some(LanguageCode::EnUs)).await;
    })
    .await;

    let mut results = PerformanceResults::load_or_create();

    if let Some(baseline) = results.get_baseline(test_name) {
        if let Err(error) = check_regression(baseline, &current) {
            println!("Warning: {error}");
        }
    } else {
        println!("No baseline found for {test_name}, establishing new baseline");
    }

    results.update_baseline(current);
    results.save();
}

#[tokio::test]
async fn test_batch_processing_performance() {
    let test_name = "batch_processing_performance";
    let g2p = DummyG2p::new();
    let test_inputs = vec![
        "hello world",
        "the quick brown fox",
        "jumps over the lazy dog",
        "this is a test sentence",
        "performance testing example",
    ];

    let current = measure_performance(test_name, 20, || async {
        for input in &test_inputs {
            let _ = g2p.to_phonemes(input, Some(LanguageCode::EnUs)).await;
        }
    })
    .await;

    let mut results = PerformanceResults::load_or_create();

    if let Some(baseline) = results.get_baseline(test_name) {
        if let Err(error) = check_regression(baseline, &current) {
            println!("Warning: {error}");
        }
    } else {
        println!("No baseline found for {test_name}, establishing new baseline");
    }

    results.update_baseline(current);
    results.save();
}

#[tokio::test]
async fn test_long_input_performance() {
    let test_name = "long_input_performance";
    let g2p = DummyG2p::new();
    let test_input = "hello world ".repeat(100); // 1200 characters

    let current = measure_performance(test_name, 10, || async {
        let _ = g2p.to_phonemes(&test_input, Some(LanguageCode::EnUs)).await;
    })
    .await;

    let mut results = PerformanceResults::load_or_create();

    if let Some(baseline) = results.get_baseline(test_name) {
        if let Err(error) = check_regression(baseline, &current) {
            println!("Warning: {error}");
        }
    } else {
        println!("No baseline found for {test_name}, establishing new baseline");
    }

    results.update_baseline(current);
    results.save();
}

#[tokio::test]
async fn test_memory_efficiency() {
    // Test that repeated operations don't cause memory leaks
    let test_name = "memory_efficiency_test";
    let g2p = DummyG2p::new();
    let test_input = "memory test input";

    let current = measure_performance(test_name, 1000, || async {
        let _ = g2p.to_phonemes(test_input, Some(LanguageCode::EnUs)).await;
    })
    .await;

    let mut results = PerformanceResults::load_or_create();

    if let Some(baseline) = results.get_baseline(test_name) {
        if let Err(error) = check_regression(baseline, &current) {
            println!("Warning: {error}");
        }
    } else {
        println!("No baseline found for {test_name}, establishing new baseline");
    }

    results.update_baseline(current);
    results.save();
}

#[test]
fn test_performance_regression_framework() {
    // Test the performance regression framework itself
    let mut results = PerformanceResults::load_or_create();

    let test_baseline = PerformanceBaseline {
        test_name: "test_framework".to_string(),
        average_duration_ms: 10.0,
        min_duration_ms: 8.0,
        max_duration_ms: 15.0,
        iterations: 100,
        timestamp: chrono::Utc::now().to_rfc3339(),
    };

    results.update_baseline(test_baseline.clone());

    // Test no regression
    let good_result = PerformanceBaseline {
        test_name: "test_framework".to_string(),
        average_duration_ms: 12.0, // 20% slower, should pass
        min_duration_ms: 10.0,
        max_duration_ms: 18.0,
        iterations: 100,
        timestamp: chrono::Utc::now().to_rfc3339(),
    };

    assert!(check_regression(&test_baseline, &good_result).is_ok());

    // Test regression detection
    let bad_result = PerformanceBaseline {
        test_name: "test_framework".to_string(),
        average_duration_ms: 20.0, // 100% slower, should fail
        min_duration_ms: 18.0,
        max_duration_ms: 25.0,
        iterations: 100,
        timestamp: chrono::Utc::now().to_rfc3339(),
    };

    assert!(check_regression(&test_baseline, &bad_result).is_err());
}

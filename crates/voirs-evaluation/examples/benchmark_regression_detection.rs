//! Example demonstrating automated benchmark runner with regression detection.
//!
//! This example shows how to use the VoiRS evaluation framework to automatically
//! run benchmarks, detect performance regressions, and generate reports.

use std::time::Duration;
use tokio;
use voirs_evaluation::{
    benchmark_runner::{BenchmarkConfig, BenchmarkRunner},
    regression_detector::{RegressionConfig, RegressionDetector, RegressionSeverity},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 VoiRS Benchmark Regression Detection Example");
    println!("===============================================\n");

    // Create a custom regression detection configuration
    let regression_config = RegressionConfig {
        minor_threshold: 0.05,    // 5% performance degradation
        major_threshold: 0.15,    // 15% performance degradation
        critical_threshold: 0.30, // 30% performance degradation
        baseline_window: 5,       // Consider last 5 measurements for baseline
        min_samples: 3,           // Need at least 3 samples for detection
    };

    // Create benchmark configuration
    let benchmark_config = BenchmarkConfig {
        benchmark_names: vec![
            "evaluation_metrics".to_string(),
            "memory_benchmark".to_string(),
        ],
        timeout: Duration::from_secs(300),
        warmup_iterations: 1,      // Reduced for demo
        measurement_iterations: 2, // Reduced for demo
        baseline_file: "/tmp/voirs_benchmark_baseline_example.json".to_string(),
        regression_config,
    };

    // Create benchmark runner
    let mut runner = BenchmarkRunner::with_config(benchmark_config);

    println!("📊 Running benchmarks...");

    // Simulate some historical measurements for demonstration
    simulate_historical_data(&mut runner).await?;

    // Run the benchmark suite
    let results = runner.run_all_benchmarks().await?;

    println!("\n📈 Benchmark Results:");
    for result in &results {
        println!(
            "  ✅ {} - {} measurements in {:?}",
            result.benchmark_name,
            result.measurements.len(),
            result.duration
        );
    }

    // Detect regressions
    println!("\n🔍 Detecting performance regressions...");
    let regressions = runner.detect_regressions();

    if regressions.is_empty() {
        println!("✅ No performance regressions detected!");
    } else {
        println!("⚠️  Performance regressions detected:");
        for regression in &regressions {
            let emoji = match regression.severity {
                RegressionSeverity::Minor => "🟡",
                RegressionSeverity::Major => "🟠",
                RegressionSeverity::Critical => "🔴",
            };
            println!(
                "  {} {}: {:.1}% slower ({:.2}ms -> {:.2}ms)",
                emoji,
                regression.measurement_name,
                regression.change_percentage * 100.0,
                regression.baseline_value,
                regression.current_value
            );
        }
    }

    // Generate comprehensive report
    println!("\n📋 Comprehensive Regression Report:");
    println!("{}", runner.generate_regression_report());

    // Generate trend analysis
    println!("📊 Performance Trend Analysis:");
    println!("{}", runner.generate_trend_analysis());

    // Demonstrate CI integration
    println!("🔧 Running CI checks...");
    let ci_passed = runner.run_ci_checks().await?;

    if ci_passed {
        println!("✅ All CI checks passed - ready for deployment!");
    } else {
        println!("❌ CI checks failed - review performance issues before deployment");
    }

    Ok(())
}

/// Simulate some historical benchmark data for demonstration purposes
async fn simulate_historical_data(
    runner: &mut BenchmarkRunner,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("📝 Simulating historical benchmark data...");

    // Simulate some baseline measurements
    let measurements = vec![
        ("evaluation_metrics::total_time", 145.0),
        ("evaluation_metrics::total_time", 148.0),
        ("evaluation_metrics::total_time", 142.0),
        ("evaluation_metrics::total_time", 150.0),
        ("memory_benchmark::total_time", 195.0),
        ("memory_benchmark::total_time", 202.0),
        ("memory_benchmark::total_time", 198.0),
        ("memory_benchmark::total_time", 205.0),
    ];

    for (name, value) in measurements {
        let measurement = RegressionDetector::create_measurement(
            name.to_string(),
            value,
            "ms".to_string(),
            Some("historical_commit".to_string()),
            "1.0.0".to_string(),
        );
        runner.add_measurement(measurement);

        // Small delay to ensure different timestamps
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    println!("✅ Historical data simulation complete");
    Ok(())
}

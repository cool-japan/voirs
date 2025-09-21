//! Automated performance regression test suite
//!
//! This module provides automated regression testing that can be run in CI/CD pipelines
//! to detect performance regressions across different model configurations and scenarios.

use std::env;
use std::time::Instant;
use voirs_recognizer::prelude::*;

// Import the regression testing infrastructure
mod performance_regression_tests;
use performance_regression_tests::*;

/// Comprehensive automated regression test suite
#[tokio::test]
async fn test_automated_performance_regression_suite() {
    let start_time = Instant::now();

    println!("🚀 Starting Automated Performance Regression Test Suite");
    println!("================================================");

    let tester = RegressionTester::new();

    // Define comprehensive test configurations
    let test_configurations = vec![
        TestConfiguration {
            model_type: "small".to_string(),
            sample_rate: 16000,
            duration: 1.0,
            channels: 1,
            features: vec!["whisper".to_string()],
        },
        TestConfiguration {
            model_type: "base".to_string(),
            sample_rate: 16000,
            duration: 1.0,
            channels: 1,
            features: vec!["whisper".to_string()],
        },
        TestConfiguration {
            model_type: "large".to_string(),
            sample_rate: 16000,
            duration: 1.0,
            channels: 1,
            features: vec!["whisper".to_string()],
        },
        TestConfiguration {
            model_type: "base".to_string(),
            sample_rate: 16000,
            duration: 5.0,
            channels: 1,
            features: vec!["whisper".to_string()],
        },
        TestConfiguration {
            model_type: "base".to_string(),
            sample_rate: 16000,
            duration: 1.0,
            channels: 2,
            features: vec!["whisper".to_string()],
        },
    ];

    let mut all_results = Vec::new();
    let mut has_critical_regressions = false;
    let mut total_regressions = 0;
    let mut total_improvements = 0;

    // Load existing baseline
    let baseline_results = tester.load_baseline().unwrap_or_default();

    // Run benchmarks for each configuration
    for (i, config) in test_configurations.iter().enumerate() {
        println!(
            "\n📊 Running benchmark {} of {}: {} ({}s audio, {}Hz, {} ch)",
            i + 1,
            test_configurations.len(),
            config.model_type,
            config.duration,
            config.sample_rate,
            config.channels
        );

        let benchmark_start = Instant::now();
        let result = tester.run_benchmark(config.clone()).await;
        let benchmark_duration = benchmark_start.elapsed();

        println!("  ✅ Completed in {:.2}s", benchmark_duration.as_secs_f32());
        println!(
            "  📈 RTF: {:.3}, Memory: {:.1}MB, Startup: {}ms",
            result.rtf,
            result.memory_usage as f64 / (1024.0 * 1024.0),
            result.startup_time_ms
        );

        // Analyze regression against baseline
        let analysis = tester.analyze_regression(&result, &baseline_results);

        if analysis.has_regressions {
            println!("  ⚠️  REGRESSIONS DETECTED:");
            for regression in &analysis.regressions {
                let severity_str = match regression.severity {
                    RegressionSeverity::Minor => "MINOR",
                    RegressionSeverity::Major => "MAJOR",
                    RegressionSeverity::Critical => "CRITICAL",
                };

                if regression.severity == RegressionSeverity::Critical {
                    has_critical_regressions = true;
                }

                println!(
                    "    🚨 {} {}: {:.2}% degradation",
                    severity_str, regression.metric, regression.percentage_change
                );
            }
            total_regressions += analysis.regressions.len();
        }

        if !analysis.improvements.is_empty() {
            println!("  🚀 IMPROVEMENTS:");
            for improvement in &analysis.improvements {
                println!(
                    "    ✨ {}: {:.2}% improvement",
                    improvement.metric, improvement.percentage_change
                );
            }
            total_improvements += analysis.improvements.len();
        }

        // Store result for potential baseline update
        all_results.push(result.clone());

        // Append to history
        if let Err(e) = tester.append_to_history(&result) {
            println!("  ⚠️  Failed to append to history: {}", e);
        }
    }

    let total_duration = start_time.elapsed();

    // Generate final report
    println!("\n📋 FINAL REGRESSION TEST REPORT");
    println!("===============================");
    println!("Total test duration: {:.2}s", total_duration.as_secs_f32());
    println!("Benchmarks executed: {}", test_configurations.len());
    println!("Total regressions found: {}", total_regressions);
    println!("Total improvements found: {}", total_improvements);

    if has_critical_regressions {
        println!("🔥 CRITICAL REGRESSIONS DETECTED - REVIEW REQUIRED");
        // In CI, this would be a failure
        if env::var("CI").is_ok() {
            panic!("Critical performance regressions detected in CI environment");
        }
    } else if total_regressions > 0 {
        println!("⚠️  Some regressions detected - monitoring recommended");
    } else {
        println!("✅ All performance metrics within acceptable ranges");
    }

    // Update baseline if this is a baseline run
    if env::var("UPDATE_PERFORMANCE_BASELINE").is_ok() {
        println!("\n📝 Updating performance baseline...");
        if let Err(e) = tester.save_baseline(&all_results) {
            println!("❌ Failed to update baseline: {}", e);
        } else {
            println!("✅ Performance baseline updated successfully");
        }
    }

    // Generate CI report if in CI environment
    if env::var("CI").is_ok() {
        // Generate summary for each configuration
        for (config, result) in test_configurations.iter().zip(all_results.iter()) {
            let analysis = tester.analyze_regression(result, &baseline_results);
            let ci_report = tester.generate_ci_report(&analysis);

            println!("\n--- CI Report for {} ---", result.name);
            println!("{}", ci_report);
        }
    }

    // Performance requirements validation
    assert!(
        !has_critical_regressions,
        "Critical performance regressions detected"
    );

    // Ensure all benchmarks completed successfully
    assert_eq!(
        all_results.len(),
        test_configurations.len(),
        "Not all benchmarks completed successfully"
    );

    println!("\n🎉 Automated regression test suite completed successfully!");
}

/// Test specific model performance scenarios
#[tokio::test]
async fn test_model_scaling_performance() {
    println!("🔍 Testing Model Scaling Performance");

    let tester = RegressionTester::new();

    // Test how performance scales with different model sizes
    let scaling_configs = vec![("small", 1.0), ("base", 1.0), ("large", 1.0)];

    let mut scaling_results = Vec::new();

    for (model_type, duration) in scaling_configs {
        let config = TestConfiguration {
            model_type: model_type.to_string(),
            sample_rate: 16000,
            duration,
            channels: 1,
            features: vec!["whisper".to_string()],
        };

        let result = tester.run_benchmark(config).await;

        println!(
            "  {} model: RTF={:.3}, Memory={:.1}MB, Startup={}ms",
            model_type,
            result.rtf,
            result.memory_usage as f64 / (1024.0 * 1024.0),
            result.startup_time_ms
        );

        scaling_results.push((model_type, result));
    }

    // Validate that larger models have proportionally higher resource usage
    let small_result = &scaling_results[0].1;
    let base_result = &scaling_results[1].1;
    let large_result = &scaling_results[2].1;

    // RTF should generally increase with model size
    assert!(
        base_result.rtf >= small_result.rtf,
        "Base model RTF should be >= small model RTF"
    );
    assert!(
        large_result.rtf >= base_result.rtf,
        "Large model RTF should be >= base model RTF"
    );

    // Memory usage should increase with model size
    assert!(
        base_result.memory_usage >= small_result.memory_usage,
        "Base model should use >= memory than small model"
    );
    assert!(
        large_result.memory_usage >= base_result.memory_usage,
        "Large model should use >= memory than base model"
    );

    // Startup time should increase with model size
    assert!(
        base_result.startup_time_ms >= small_result.startup_time_ms,
        "Base model startup should be >= small model startup"
    );
    assert!(
        large_result.startup_time_ms >= base_result.startup_time_ms,
        "Large model startup should be >= base model startup"
    );

    println!("✅ Model scaling performance validation passed");
}

/// Test audio duration scaling performance
#[tokio::test]
async fn test_audio_duration_scaling() {
    println!("🎵 Testing Audio Duration Scaling Performance");

    let tester = RegressionTester::new();

    // Test different audio durations
    let duration_configs = vec![0.5, 1.0, 2.0, 5.0];
    let mut duration_results = Vec::new();

    for duration in duration_configs {
        let config = TestConfiguration {
            model_type: "base".to_string(),
            sample_rate: 16000,
            duration,
            channels: 1,
            features: vec!["whisper".to_string()],
        };

        let result = tester.run_benchmark(config).await;

        println!(
            "  {:.1}s audio: RTF={:.3}, Throughput={:.0} samples/sec",
            duration, result.rtf, result.throughput_samples_per_sec
        );

        duration_results.push((duration, result));
    }

    // Validate that RTF remains relatively stable across durations
    for (duration, result) in &duration_results {
        assert!(
            result.rtf < 0.5,
            "RTF {:.3} for {:.1}s audio should be < 0.5",
            result.rtf,
            duration
        );

        // Throughput should be consistent regardless of audio duration
        assert!(
            result.throughput_samples_per_sec > 10000.0,
            "Throughput should be > 10k samples/sec for {:.1}s audio",
            duration
        );
    }

    println!("✅ Audio duration scaling validation passed");
}

/// Test memory pressure scenarios
#[tokio::test]
async fn test_memory_pressure_scenarios() {
    println!("💾 Testing Memory Pressure Scenarios");

    let tester = RegressionTester::new();

    // Test scenarios that might cause memory pressure
    let memory_test_configs = vec![
        // Multiple channels
        TestConfiguration {
            model_type: "base".to_string(),
            sample_rate: 16000,
            duration: 1.0,
            channels: 2,
            features: vec!["whisper".to_string()],
        },
        // Higher sample rate
        TestConfiguration {
            model_type: "base".to_string(),
            sample_rate: 48000,
            duration: 1.0,
            channels: 1,
            features: vec!["whisper".to_string()],
        },
        // Longer duration
        TestConfiguration {
            model_type: "base".to_string(),
            sample_rate: 16000,
            duration: 10.0,
            channels: 1,
            features: vec!["whisper".to_string()],
        },
    ];

    for config in memory_test_configs {
        let result = tester.run_benchmark(config.clone()).await;

        println!(
            "  Config: {}ch, {}Hz, {:.1}s - Memory: {:.1}MB, RTF: {:.3}",
            config.channels,
            config.sample_rate,
            config.duration,
            result.memory_usage as f64 / (1024.0 * 1024.0),
            result.rtf
        );

        // Validate memory usage remains reasonable
        assert!(
            result.memory_usage < 1024 * 1024 * 1024, // < 1GB
            "Memory usage should be < 1GB for all configurations"
        );

        // RTF should still be reasonable
        assert!(
            result.rtf < 1.0,
            "RTF should be < 1.0 for real-time processing"
        );
    }

    println!("✅ Memory pressure scenario validation passed");
}

/// CI integration test that can be run with minimal setup
#[tokio::test]
async fn test_ci_quick_regression_check() {
    // This is a minimal test that can run quickly in CI
    if env::var("CI_QUICK_CHECK").is_ok() {
        println!("⚡ Running quick CI regression check");

        let tester = RegressionTester::new();

        // Single quick test
        let config = TestConfiguration {
            model_type: "base".to_string(),
            sample_rate: 16000,
            duration: 1.0,
            channels: 1,
            features: vec!["whisper".to_string()],
        };

        let result = tester.run_benchmark(config).await;

        // Basic performance thresholds
        assert!(result.rtf < 0.5, "RTF should be < 0.5 for quick check");
        assert!(result.startup_time_ms < 5000, "Startup should be < 5s");
        assert!(
            result.streaming_latency_ms < 300,
            "Latency should be < 300ms"
        );

        println!("✅ Quick CI regression check passed");
        println!(
            "   RTF: {:.3}, Memory: {:.1}MB, Startup: {}ms",
            result.rtf,
            result.memory_usage as f64 / (1024.0 * 1024.0),
            result.startup_time_ms
        );
    } else {
        println!("ℹ️  Skipping CI quick check (set CI_QUICK_CHECK to enable)");
    }
}

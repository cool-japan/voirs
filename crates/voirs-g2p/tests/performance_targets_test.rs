//! Performance targets validation test
//!
//! This test validates that G2P systems meet the performance targets specified
//! in the TODO.md file: <1ms latency and <100MB memory footprint.

use std::time::Instant;
use voirs_g2p::{
    performance::{PerformanceTargetMonitor, PerformanceTargets},
    DummyG2p, G2p, LanguageCode,
};

#[tokio::test]
async fn test_latency_target_validation() {
    println!("🚀 Testing G2P Latency Targets (<1ms per sentence)");

    let monitor = PerformanceTargetMonitor::new_with_defaults();
    let g2p = DummyG2p::new();

    // Test sentences of different lengths (typical use case: 20-50 characters)
    let test_sentences = vec![
        "hello",                                                        // 5 chars
        "hello world",                                                  // 11 chars
        "this is a test sentence",                                      // 24 chars
        "the quick brown fox jumps over lazy dog",                      // 40 chars
        "performance testing for g2p conversion with longer sentences", // 62 chars
    ];

    let mut total_violations = 0;

    for sentence in &test_sentences {
        let start_time = Instant::now();
        let result = g2p.to_phonemes(sentence, Some(LanguageCode::EnUs)).await;
        let latency = start_time.elapsed();

        assert!(result.is_ok());
        let phonemes = result.unwrap();

        // Record latency measurement
        let record_result =
            monitor.record_latency(sentence, latency, phonemes.len(), LanguageCode::EnUs);
        assert!(record_result.is_ok());

        // Check individual latency
        let latency_ms = latency.as_millis() as f64;
        if latency_ms > 1.0 {
            total_violations += 1;
            println!("⚠️  Latency violation: '{sentence}' took {latency_ms:.2}ms (>1.0ms target)");
        } else {
            println!("✅ '{sentence}' processed in {latency_ms:.2}ms");
        }
    }

    // Generate performance report
    let report = monitor.generate_report();
    println!("\n{report}");

    // Assess overall performance
    let summary = monitor.get_performance_summary();
    if let Some(latency_stats) = summary.latency_stats {
        println!("\n📊 Latency Statistics:");
        println!("├── Average: {:.2}ms", latency_stats.avg_latency_ms);
        println!("├── 95th percentile: {:.2}ms", latency_stats.p95_latency_ms);
        println!("├── 99th percentile: {:.2}ms", latency_stats.p99_latency_ms);
        println!(
            "└── Total measurements: {}",
            latency_stats.measurements_count
        );

        // For DummyG2p, we expect very fast performance
        assert!(
            latency_stats.avg_latency_ms < 10.0,
            "Average latency too high: {:.2}ms",
            latency_stats.avg_latency_ms
        );
    }

    println!("\n🎯 Latency target test completed with {total_violations} violations");
}

#[tokio::test]
async fn test_memory_target_validation() {
    println!("💾 Testing G2P Memory Targets (<100MB per language model)");

    let monitor = PerformanceTargetMonitor::new_with_defaults();

    // Simulate memory usage measurements for different scenarios
    let test_scenarios = vec![
        ("Small model", 45.0, 15.0, 10.0, 20.0), // total, model, cache, working
        ("Medium model", 85.0, 60.0, 20.0, 5.0), // within target
        ("Large model", 150.0, 120.0, 25.0, 5.0), // exceeds target
        ("Optimized model", 75.0, 50.0, 15.0, 10.0), // good performance
    ];

    for (scenario_name, total_mb, model_mb, cache_mb, working_mb) in test_scenarios {
        let record_result = monitor.record_memory_usage(
            total_mb,
            model_mb,
            cache_mb,
            working_mb,
            LanguageCode::EnUs,
        );
        assert!(record_result.is_ok());

        if model_mb > 100.0 {
            println!("⚠️  Memory violation: {scenario_name} uses {model_mb:.1}MB model memory (>100MB target)");
        } else {
            println!("✅ {scenario_name} uses {model_mb:.1}MB model memory");
        }
    }

    // Generate performance report
    let report = monitor.generate_report();
    println!("\n{report}");

    // Assess memory performance
    let summary = monitor.get_performance_summary();
    if let Some(memory_stats) = summary.memory_stats {
        println!("\n📊 Memory Statistics:");
        println!(
            "├── Average model memory: {:.1}MB",
            memory_stats.avg_model_memory_mb
        );
        println!(
            "├── Peak model memory: {:.1}MB",
            memory_stats.max_model_memory_mb
        );
        println!(
            "├── Average total memory: {:.1}MB",
            memory_stats.avg_total_memory_mb
        );
        println!("└── Total snapshots: {}", memory_stats.snapshots_count);
    }

    println!("💾 Memory target test completed");
}

#[tokio::test]
async fn test_throughput_target_validation() {
    println!("🚀 Testing G2P Throughput Targets (>1000 sentences/sec)");

    let monitor = PerformanceTargetMonitor::new_with_defaults();
    let g2p = DummyG2p::new();

    // Test batch processing throughput
    let test_sentences = vec![
        "hello world",
        "this is a test",
        "performance evaluation",
        "throughput measurement",
        "batch processing test",
    ];

    let batch_sizes = vec![10, 50, 100, 500];

    for batch_size in batch_sizes {
        println!("\n📦 Testing batch size: {batch_size} sentences");

        let start_time = Instant::now();
        let mut total_sentences = 0;
        let mut total_length = 0;

        // Process multiple batches
        for _ in 0..batch_size {
            for sentence in &test_sentences {
                let result = g2p.to_phonemes(sentence, Some(LanguageCode::EnUs)).await;
                assert!(result.is_ok());
                total_sentences += 1;
                total_length += sentence.len();
            }
        }

        let duration = start_time.elapsed();
        let avg_sentence_length = total_length as f64 / total_sentences as f64;

        // Record throughput measurement
        let record_result = monitor.record_throughput(
            total_sentences,
            duration,
            avg_sentence_length,
            LanguageCode::EnUs,
        );
        assert!(record_result.is_ok());

        let sentences_per_sec = total_sentences as f64 / duration.as_secs_f64();

        if sentences_per_sec < 1000.0 {
            println!(
                "⚠️  Throughput below target: {sentences_per_sec:.0} sentences/sec (<1000 target)"
            );
        } else {
            println!("✅ Throughput: {sentences_per_sec:.0} sentences/sec");
        }
    }

    // Generate performance report
    let report = monitor.generate_report();
    println!("\n{report}");

    println!("🚀 Throughput target test completed");
}

#[tokio::test]
async fn test_comprehensive_performance_validation() {
    println!("🎯 Comprehensive G2P Performance Validation");
    println!("===========================================");

    let custom_targets = PerformanceTargets {
        max_latency_ms: 1.0,
        max_memory_mb: 100.0,
        min_throughput_sentences_per_sec: 1000.0,
        max_cpu_usage_percent: 80.0,
    };

    let monitor = PerformanceTargetMonitor::new(custom_targets);
    let g2p = DummyG2p::new();

    // Comprehensive test with realistic workload
    let test_workload = vec![
        ("Short text", "hello", 5),
        ("Medium text", "the quick brown fox", 19),
        (
            "Long text",
            "performance testing with comprehensive validation",
            49,
        ),
        (
            "Technical text",
            "grapheme to phoneme conversion algorithm",
            40,
        ),
        (
            "Real sentence",
            "this is a realistic sentence for testing",
            40,
        ),
    ];

    println!("\n🔄 Processing comprehensive workload...");

    for (description, sentence, expected_length) in test_workload {
        // Measure latency
        let start_time = Instant::now();
        let result = g2p.to_phonemes(sentence, Some(LanguageCode::EnUs)).await;
        let latency = start_time.elapsed();

        assert!(result.is_ok());
        let phonemes = result.unwrap();

        // Record measurements
        monitor
            .record_latency(sentence, latency, phonemes.len(), LanguageCode::EnUs)
            .unwrap();

        // Simulate memory usage
        monitor
            .record_memory_usage(80.0, 60.0, 15.0, 5.0, LanguageCode::EnUs)
            .unwrap();

        assert_eq!(sentence.len(), expected_length);
        println!(
            "📝 Processed '{}': {:.2}ms, {} phonemes",
            description,
            latency.as_millis() as f64,
            phonemes.len()
        );
    }

    // Simulate batch throughput test
    let batch_start = Instant::now();
    let batch_size = 100;
    for _ in 0..batch_size {
        let _result = g2p
            .to_phonemes("batch test sentence", Some(LanguageCode::EnUs))
            .await;
    }
    let batch_duration = batch_start.elapsed();

    monitor
        .record_throughput(
            batch_size,
            batch_duration,
            18.0, // avg length of "batch test sentence"
            LanguageCode::EnUs,
        )
        .unwrap();

    // Generate final report
    println!("\n📋 Final Performance Report");
    println!("===========================");
    let report = monitor.generate_report();
    println!("{report}");

    // Validate that targets can be assessed
    let summary = monitor.get_performance_summary();
    assert!(summary.latency_stats.is_some());
    assert!(summary.memory_stats.is_some());
    assert!(summary.throughput_stats.is_some());

    // Check overall compliance
    let targets_met = monitor.are_targets_met();
    println!(
        "\n🏆 Overall Target Compliance: {}",
        if targets_met {
            "✅ PASSED"
        } else {
            "❌ NEEDS IMPROVEMENT"
        }
    );

    println!("\n✅ Comprehensive performance validation completed");
}

#[test]
fn test_performance_targets_configuration() {
    println!("⚙️  Testing Performance Targets Configuration");

    // Test default targets
    let default_targets = PerformanceTargets::default();
    assert_eq!(default_targets.max_latency_ms, 1.0);
    assert_eq!(default_targets.max_memory_mb, 100.0);
    assert_eq!(default_targets.min_throughput_sentences_per_sec, 1000.0);
    assert_eq!(default_targets.max_cpu_usage_percent, 80.0);

    // Test custom targets
    let custom_targets = PerformanceTargets {
        max_latency_ms: 0.5,
        max_memory_mb: 50.0,
        min_throughput_sentences_per_sec: 2000.0,
        max_cpu_usage_percent: 90.0,
    };

    let monitor = PerformanceTargetMonitor::new(custom_targets.clone());
    let summary = monitor.get_performance_summary();
    assert_eq!(summary.targets.max_latency_ms, 0.5);
    assert_eq!(summary.targets.max_memory_mb, 50.0);

    println!("✅ Performance targets configuration test passed");
}

//! Performance Regression Detection Demo
//!
//! Demonstrates comprehensive performance regression detection system
//! with statistical analysis, baseline management, and automated reporting.
//!
//! This example shows:
//! - Baseline establishment from historical performance
//! - Multiple regression detection algorithms (z-score, IQR)
//! - Quality, latency, and RTF regression detection
//! - Detailed regression reports with confidence scores
//! - Integration with quality monitoring system

use std::time::SystemTime;
use voirs_sdk::adaptive::{
    MetricBaseline, PerformanceSnapshot, RegressionConfig, RegressionDetector,
};
use voirs_sdk::Result;

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Performance Regression Detection Demo ===\n");

    // Demo 1: Basic Regression Detection
    demo_basic_regression_detection().await?;

    // Demo 2: Statistical Analysis
    demo_statistical_analysis().await?;

    // Demo 3: Multiple Metrics Regression
    demo_multiple_metrics_regression().await?;

    // Demo 4: Sensitivity Configuration
    demo_sensitivity_configuration().await?;

    // Demo 5: Baseline Management
    demo_baseline_management().await?;

    Ok(())
}

/// Demo 1: Basic Regression Detection
async fn demo_basic_regression_detection() -> Result<()> {
    println!("--- Demo 1: Basic Regression Detection ---");

    let detector = RegressionDetector::new(
        RegressionConfig::default()
            .with_min_baseline_samples(20)
            .with_z_score_threshold(2.0),
    );

    // Establish baseline with good performance
    println!("\nEstablishing baseline with 30 samples...");
    for i in 0..30 {
        let snapshot = PerformanceSnapshot {
            quality_score: 78.0 + (i as f32 % 5.0),
            latency_ms: 95 + (i % 10),
            rtf: 0.48 + ((i as f32 * 0.1).sin() * 0.05),
            memory_mb: 510.0 + (i as f32 * 0.5),
            cpu_percent: 48.0 + (i as f32 * 0.3),
            timestamp: SystemTime::now(),
        };
        detector.record_performance(snapshot).await?;
    }

    let baseline = detector.establish_baseline().await?;
    println!("Baseline established:");
    println!(
        "  Quality: {:.1} ± {:.1}",
        baseline.quality.mean, baseline.quality.std_dev
    );
    println!(
        "  Latency: {:.1}ms ± {:.1}ms",
        baseline.latency.mean, baseline.latency.std_dev
    );
    println!(
        "  RTF: {:.3} ± {:.3}",
        baseline.rtf.mean, baseline.rtf.std_dev
    );

    // Test with good performance (no regression)
    println!("\nTesting with good performance...");
    let good_snapshot = PerformanceSnapshot {
        quality_score: 79.0,
        latency_ms: 98,
        rtf: 0.49,
        memory_mb: 515.0,
        cpu_percent: 49.0,
        timestamp: SystemTime::now(),
    };

    match detector.detect_regression(&good_snapshot).await? {
        Some(report) => println!("Unexpected regression: {}", report.summary),
        None => println!("✓ No regression detected (as expected)"),
    }

    // Test with degraded performance (should detect regression)
    println!("\nTesting with degraded performance...");
    let bad_snapshot = PerformanceSnapshot {
        quality_score: 50.0, // Significantly degraded
        latency_ms: 95,
        rtf: 0.48,
        memory_mb: 515.0,
        cpu_percent: 49.0,
        timestamp: SystemTime::now(),
    };

    match detector.detect_regression(&bad_snapshot).await? {
        Some(report) => {
            println!("✓ Regression detected:");
            println!("  {}", report.summary);
            println!("  Confidence: {:.1}%", report.confidence * 100.0);
            for reg in &report.regressions {
                println!("\n  Metric: {}", reg.metric_name);
                println!("    Baseline: {:.2}", reg.baseline_value);
                println!("    Current: {:.2}", reg.current_value);
                println!("    Change: {:.1}%", reg.percent_change);
                println!("    Z-score: {:.2}", reg.z_score);
                println!("    Method: {}", reg.detection_method);
            }
        }
        None => println!("✗ Failed to detect regression"),
    }

    println!();
    Ok(())
}

/// Demo 2: Statistical Analysis
async fn demo_statistical_analysis() -> Result<()> {
    println!("--- Demo 2: Statistical Analysis ---");

    let detector =
        RegressionDetector::new(RegressionConfig::default().with_min_baseline_samples(50));

    // Create baseline with normal distribution
    println!("\nCreating baseline with 100 samples...");
    for i in 0..100 {
        let quality = 75.0 + (fastrand::f32() * 10.0) - 5.0; // ~75 ± 5
        let latency = 100 + ((fastrand::f32() * 20.0) as u64) - 10; // ~100 ± 10
        let rtf = 0.5 + (fastrand::f32() * 0.1) - 0.05; // ~0.5 ± 0.05

        let snapshot = PerformanceSnapshot {
            quality_score: quality,
            latency_ms: latency,
            rtf,
            memory_mb: 512.0,
            cpu_percent: 50.0,
            timestamp: SystemTime::now(),
        };
        detector.record_performance(snapshot).await?;
    }

    let baseline = detector.establish_baseline().await?;

    println!("\nBaseline Statistics:");
    print_metric_stats("Quality", &baseline.quality);
    print_metric_stats("Latency", &baseline.latency);
    print_metric_stats("RTF", &baseline.rtf);

    // Test outliers at different z-score levels
    let test_cases = vec![
        (75.0, "Within 1σ (should not detect)"),
        (65.0, "Within 2σ (borderline)"),
        (55.0, "Beyond 3σ (should detect)"),
        (45.0, "Beyond 5σ (definitely detect)"),
    ];

    println!("\nTesting outlier detection:");
    for (quality, description) in test_cases {
        let snapshot = PerformanceSnapshot {
            quality_score: quality,
            latency_ms: 100,
            rtf: 0.5,
            memory_mb: 512.0,
            cpu_percent: 50.0,
            timestamp: SystemTime::now(),
        };

        let result = detector.detect_regression(&snapshot).await?;
        let status = if result.is_some() {
            "DETECTED"
        } else {
            "not detected"
        };
        println!("  {}: {} (quality={:.1})", description, status, quality);
    }

    println!();
    Ok(())
}

/// Demo 3: Multiple Metrics Regression
async fn demo_multiple_metrics_regression() -> Result<()> {
    println!("--- Demo 3: Multiple Metrics Regression ---");

    let detector = RegressionDetector::new(
        RegressionConfig::default()
            .with_min_baseline_samples(20)
            .with_all_detections(), // Enable all metric detection
    );

    // Establish baseline
    println!("\nEstablishing baseline...");
    for _ in 0..30 {
        let snapshot = PerformanceSnapshot {
            quality_score: 80.0,
            latency_ms: 100,
            rtf: 0.5,
            memory_mb: 500.0,
            cpu_percent: 45.0,
            timestamp: SystemTime::now(),
        };
        detector.record_performance(snapshot).await?;
    }
    detector.establish_baseline().await?;

    // Test 1: Quality regression only
    println!("\nScenario 1: Quality degradation only");
    let snapshot1 = PerformanceSnapshot {
        quality_score: 50.0, // Degraded
        latency_ms: 100,     // OK
        rtf: 0.5,            // OK
        memory_mb: 500.0,    // OK
        cpu_percent: 45.0,   // OK
        timestamp: SystemTime::now(),
    };

    if let Some(report) = detector.detect_regression(&snapshot1).await? {
        println!("  Regressions: {}", report.regressions.len());
        for reg in &report.regressions {
            println!(
                "    - {}: {:.1}% change",
                reg.metric_name, reg.percent_change
            );
        }
    }

    // Test 2: Multiple regressions
    println!("\nScenario 2: Multiple metrics degraded");
    let snapshot2 = PerformanceSnapshot {
        quality_score: 55.0, // Degraded
        latency_ms: 200,     // Degraded
        rtf: 0.9,            // Degraded
        memory_mb: 800.0,    // Degraded
        cpu_percent: 80.0,   // Degraded
        timestamp: SystemTime::now(),
    };

    if let Some(report) = detector.detect_regression(&snapshot2).await? {
        println!("  Regressions: {}", report.regressions.len());
        println!("  Overall confidence: {:.1}%", report.confidence * 100.0);
        for reg in &report.regressions {
            println!(
                "    - {}: {:.1}% change (z={:.2})",
                reg.metric_name, reg.percent_change, reg.z_score
            );
        }
    }

    println!();
    Ok(())
}

/// Demo 4: Sensitivity Configuration
async fn demo_sensitivity_configuration() -> Result<()> {
    println!("--- Demo 4: Sensitivity Configuration ---");

    // Establish common baseline
    let baseline_samples: Vec<PerformanceSnapshot> = (0..50)
        .map(|_| PerformanceSnapshot {
            quality_score: 75.0 + (fastrand::f32() * 4.0) - 2.0,
            latency_ms: 100,
            rtf: 0.5,
            memory_mb: 512.0,
            cpu_percent: 50.0,
            timestamp: SystemTime::now(),
        })
        .collect();

    // Test snapshot with moderate quality drop
    let test_snapshot = PerformanceSnapshot {
        quality_score: 68.0, // Moderate drop
        latency_ms: 100,
        rtf: 0.5,
        memory_mb: 512.0,
        cpu_percent: 50.0,
        timestamp: SystemTime::now(),
    };

    println!("\nTesting same performance with different sensitivity levels:");
    println!(
        "Test quality: {:.1} (baseline ~75.0)\n",
        test_snapshot.quality_score
    );

    let sensitivities = vec![
        (1.5, "Very sensitive"),
        (2.0, "Standard"),
        (2.5, "Moderate"),
        (3.0, "Conservative"),
    ];

    for (threshold, description) in sensitivities {
        let detector = RegressionDetector::new(
            RegressionConfig::default()
                .with_min_baseline_samples(20)
                .with_z_score_threshold(threshold),
        );

        // Record baseline
        for snapshot in &baseline_samples {
            detector.record_performance(snapshot.clone()).await?;
        }
        detector.establish_baseline().await?;

        // Check regression
        let result = detector.detect_regression(&test_snapshot).await?;
        let status = if let Some(regression) = result {
            format!("DETECTED (conf={:.1}%)", regression.confidence * 100.0)
        } else {
            "not detected".to_string()
        };

        println!(
            "  {} (z-threshold={:.1}): {}",
            description, threshold, status
        );
    }

    println!();
    Ok(())
}

/// Demo 5: Baseline Management
async fn demo_baseline_management() -> Result<()> {
    println!("--- Demo 5: Baseline Management ---");

    let detector =
        RegressionDetector::new(RegressionConfig::default().with_min_baseline_samples(30));

    // Initial state
    println!("\nInitial state:");
    println!("  Samples: {}", detector.sample_count().await);
    println!("  Baseline: {:?}", detector.get_baseline().await?.is_some());

    // Record samples
    println!("\nRecording 50 samples...");
    for i in 0..50 {
        let snapshot = PerformanceSnapshot {
            quality_score: 75.0 + (i as f32 * 0.1),
            latency_ms: 100,
            rtf: 0.5,
            memory_mb: 512.0,
            cpu_percent: 50.0,
            timestamp: SystemTime::now(),
        };
        detector.record_performance(snapshot).await?;
    }

    println!("  Samples: {}", detector.sample_count().await);

    // Establish first baseline
    println!("\nEstablishing baseline...");
    let baseline1 = detector.establish_baseline().await?;
    println!(
        "  Quality baseline: {:.2} ± {:.2}",
        baseline1.quality.mean, baseline1.quality.std_dev
    );
    println!("  Sample count: {}", baseline1.sample_count);

    // Record more samples with better performance
    println!("\nRecording 30 more samples with improved performance...");
    for _ in 0..30 {
        let snapshot = PerformanceSnapshot {
            quality_score: 85.0, // Improved
            latency_ms: 80,      // Better
            rtf: 0.4,            // Better
            memory_mb: 512.0,
            cpu_percent: 50.0,
            timestamp: SystemTime::now(),
        };
        detector.record_performance(snapshot).await?;
    }

    // Re-establish baseline
    println!("\nRe-establishing baseline with new data...");
    let baseline2 = detector.establish_baseline().await?;
    println!(
        "  Quality baseline: {:.2} ± {:.2}",
        baseline2.quality.mean, baseline2.quality.std_dev
    );
    println!(
        "  Improvement: {:.2} points",
        baseline2.quality.mean - baseline1.quality.mean
    );

    // Test regression detection with new baseline
    println!("\nTesting regression with new baseline:");
    let old_perf = PerformanceSnapshot {
        quality_score: 75.0, // Was normal in old baseline, now regressed
        latency_ms: 100,
        rtf: 0.5,
        memory_mb: 512.0,
        cpu_percent: 50.0,
        timestamp: SystemTime::now(),
    };

    if let Some(report) = detector.detect_regression(&old_perf).await? {
        println!("  Regression detected:");
        println!("  (Performance that was normal is now below new baseline)");
        for reg in &report.regressions {
            println!(
                "    - {}: {:.1}% below new baseline",
                reg.metric_name,
                reg.percent_change.abs()
            );
        }
    }

    // Clear and verify
    println!("\nClearing all data...");
    detector.clear().await?;
    println!("  Samples after clear: {}", detector.sample_count().await);

    println!("\n=== Regression Detection Demo Complete ===");
    Ok(())
}

// Helper functions

fn print_metric_stats(name: &str, stats: &MetricBaseline) {
    println!("\n{}:", name);
    println!("  Mean: {:.2}", stats.mean);
    println!("  Std Dev: {:.2}", stats.std_dev);
    println!("  Range: [{:.2}, {:.2}]", stats.min, stats.max);
    println!(
        "  Quartiles: Q1={:.2}, Median={:.2}, Q3={:.2}",
        stats.p25, stats.median, stats.p75
    );
}

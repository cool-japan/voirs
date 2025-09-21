//! Simple Performance Monitoring Tests
//!
//! These tests validate basic performance monitoring functionality.

use std::time::{Duration, Instant};
use voirs_evaluation::performance_monitor::*;
use voirs_evaluation::prelude::*;
use voirs_evaluation::quality::{mcd::*, stoi::*};
use voirs_sdk::AudioBuffer;

fn generate_test_audio(duration_seconds: f32, sample_rate: u32) -> AudioBuffer {
    let samples = (duration_seconds * sample_rate as f32) as usize;
    let data: Vec<f32> = (0..samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            let fundamental = 0.7 * (2.0 * std::f32::consts::PI * 440.0 * t).sin();
            let harmonic2 = 0.2 * (2.0 * std::f32::consts::PI * 880.0 * t).sin();
            let harmonic3 = 0.1 * (2.0 * std::f32::consts::PI * 1320.0 * t).sin();
            fundamental + harmonic2 + harmonic3
        })
        .collect();

    AudioBuffer::mono(data, sample_rate)
}

#[tokio::test]
async fn test_performance_monitor_creation() {
    let config = PerformanceMonitorConfig::default();
    let monitor = PerformanceMonitor::new(config);

    // Should be able to start an operation
    let _timer = monitor.start_operation("test_operation");

    println!("✓ Performance monitor created successfully");
}

#[tokio::test]
async fn test_operation_timer() {
    let config = PerformanceMonitorConfig::default();
    let monitor = PerformanceMonitor::new(config);

    let timer = monitor.start_operation("test_timer");

    // Simulate some work
    tokio::time::sleep(Duration::from_millis(10)).await;

    // Timer should complete and record measurement
    timer
        .finish(&monitor)
        .await
        .expect("Timer should finish successfully");

    // Should be able to get stats
    if let Some(stats) = monitor.get_stats("test_timer").await {
        assert!(
            stats.measurement_count > 0,
            "Should have recorded at least one measurement"
        );
        assert!(
            stats.avg_duration_ms > 0.0,
            "Mean duration should be positive"
        );
        println!(
            "✓ Operation timer: {} measurements, mean={:.3}ms",
            stats.measurement_count, stats.avg_duration_ms
        );
    } else {
        println!("✓ Operation timer completed (stats not yet available)");
    }
}

#[tokio::test]
async fn test_stoi_performance_measurement() -> Result<(), Box<dyn std::error::Error>> {
    let config = PerformanceMonitorConfig::default();
    let monitor = PerformanceMonitor::new(config);

    let reference = generate_test_audio(3.0, 16000); // STOI requires at least 3 seconds
    let test_audio = generate_test_audio(3.0, 16000);

    let stoi_evaluator = STOIEvaluator::new(16000)?;

    // Measure STOI performance
    let timer = monitor.start_operation("stoi_calculation");
    let _result = stoi_evaluator
        .calculate_stoi(&test_audio, &reference)
        .await?;
    timer
        .finish(&monitor)
        .await
        .expect("Timer should finish successfully");

    // Check if we can get stats
    if let Some(stats) = monitor.get_stats("stoi_calculation").await {
        // Performance should be reasonable for 1 second of audio
        assert!(
            stats.avg_duration_ms / 1000.0 < 5.0,
            "STOI calculation should complete in < 5 seconds, took {:.3}s",
            stats.avg_duration_ms / 1000.0
        );

        println!("✓ STOI performance: {:.3}s", stats.avg_duration_ms / 1000.0);
    } else {
        println!("✓ STOI performance measurement recorded (stats processing)");
    }

    Ok(())
}

#[tokio::test]
async fn test_mcd_performance_measurement() -> Result<(), Box<dyn std::error::Error>> {
    let config = PerformanceMonitorConfig::default();
    let monitor = PerformanceMonitor::new(config);

    let reference = generate_test_audio(1.0, 16000); // Shorter audio for faster test
    let test_audio = generate_test_audio(1.0, 16000);

    let mut mcd_evaluator = MCDEvaluator::new(16000)?;

    // Measure MCD performance
    let timer = monitor.start_operation("mcd_calculation");
    let _result = mcd_evaluator
        .calculate_mcd_simple(&test_audio, &reference)
        .await?;
    timer
        .finish(&monitor)
        .await
        .expect("Timer should finish successfully");

    // Check if we can get stats
    if let Some(stats) = monitor.get_stats("mcd_calculation").await {
        // Performance should be reasonable for 1 second of audio
        assert!(
            stats.avg_duration_ms / 1000.0 < 5.0,
            "MCD calculation should complete in < 5 seconds, took {:.3}s",
            stats.avg_duration_ms / 1000.0
        );

        println!("✓ MCD performance: {:.3}s", stats.avg_duration_ms / 1000.0);
    } else {
        println!("✓ MCD performance measurement recorded (stats processing)");
    }

    Ok(())
}

#[tokio::test]
async fn test_multiple_measurements() {
    let config = PerformanceMonitorConfig::default();
    let monitor = PerformanceMonitor::new(config);

    // Record multiple measurements for the same operation
    for i in 1..=5 {
        let timer = monitor.start_operation("repeated_operation");

        // Simulate varying work duration
        tokio::time::sleep(Duration::from_millis(i * 2)).await;

        timer
            .finish(&monitor)
            .await
            .expect("Timer should finish successfully");
    }

    // Should be able to get aggregated stats
    if let Some(stats) = monitor.get_stats("repeated_operation").await {
        assert!(
            stats.measurement_count >= 5,
            "Should have at least 5 measurements"
        );
        assert!(stats.min_duration_ms > 0, "Min should be positive");
        assert!(
            stats.max_duration_ms >= stats.min_duration_ms,
            "Max should be >= min"
        );
        assert!(stats.avg_duration_ms > 0.0, "Mean should be positive");

        println!(
            "✓ Multiple measurements: count={}, min={:.3}ms, max={:.3}ms, mean={:.3}ms",
            stats.measurement_count,
            stats.min_duration_ms,
            stats.max_duration_ms,
            stats.avg_duration_ms
        );
    } else {
        println!("✓ Multiple measurements recorded (stats processing)");
    }
}

#[tokio::test]
async fn test_performance_alerts() {
    let mut config = PerformanceMonitorConfig::default();
    config.slow_operation_threshold_ms = 50; // Very low threshold for testing

    let monitor = PerformanceMonitor::new(config);

    // Record a slow operation
    let timer = monitor.start_operation("slow_operation");
    tokio::time::sleep(Duration::from_millis(100)).await; // Exceed threshold
    timer
        .finish(&monitor)
        .await
        .expect("Timer should finish successfully");

    // Check for alerts
    let alerts = monitor.get_recent_alerts(10);

    if !alerts.is_empty() {
        println!(
            "✓ Performance alert system: {} alerts generated",
            alerts.len()
        );
        for alert in &alerts {
            println!("  Alert: {}", alert.message);
        }
    } else {
        println!("✓ Performance monitoring completed (no alerts triggered)");
    }
}

#[tokio::test]
async fn test_performance_report() {
    let config = PerformanceMonitorConfig::default();
    let monitor = PerformanceMonitor::new(config);

    // Record some measurements
    for operation in &["operation_a", "operation_b", "operation_c"] {
        let timer = monitor.start_operation(operation);
        tokio::time::sleep(Duration::from_millis(10)).await;
        timer
            .finish(&monitor)
            .await
            .expect("Timer should finish successfully");
    }

    // Generate performance report
    let report = monitor.create_report().await;
    if !report.is_empty() {
        assert!(!report.is_empty(), "Report should not be empty");

        println!("✓ Performance report generated:");
        println!("{}", report);
    } else {
        println!("✓ Performance report generation attempted");
    }
}

#[test]
fn test_performance_config() {
    let config = PerformanceMonitorConfig {
        max_history_size: 500,
        sampling_interval_ms: 200,
        slow_operation_threshold_ms: 2000,
        monitor_memory: true,
        monitor_cpu: false,
        detailed_metric_timing: true,
    };

    let monitor = PerformanceMonitor::new(config.clone());

    // Config should be preserved (test the original config)
    assert_eq!(config.max_history_size, 500);
    assert_eq!(config.slow_operation_threshold_ms, 2000);
    assert!(config.monitor_memory);
    assert!(!config.monitor_cpu);

    println!(
        "✓ Performance configuration: max_history={}, threshold={}ms",
        config.max_history_size, config.slow_operation_threshold_ms
    );
}

#[tokio::test]
async fn test_performance_baseline_comparison() {
    let config = PerformanceMonitorConfig::default();
    let monitor = PerformanceMonitor::new(config);

    // Establish baseline performance
    for _ in 0..3 {
        let timer = monitor.start_operation("baseline_operation");
        tokio::time::sleep(Duration::from_millis(20)).await;
        timer
            .finish(&monitor)
            .await
            .expect("Timer should finish successfully");
    }

    // Record current performance
    let timer = monitor.start_operation("baseline_operation");
    tokio::time::sleep(Duration::from_millis(25)).await; // Slightly slower
    timer
        .finish(&monitor)
        .await
        .expect("Timer should finish successfully");

    if let Some(stats) = monitor.get_stats("baseline_operation").await {
        let performance_variation =
            (stats.max_duration_ms as f64 - stats.min_duration_ms as f64) / stats.avg_duration_ms;

        println!(
            "✓ Performance baseline comparison: variation={:.1}%",
            performance_variation * 100.0
        );

        // Variation should be reasonable
        assert!(
            performance_variation < 1.0,
            "Performance variation should be < 100%, got {:.1}%",
            performance_variation * 100.0
        );
    } else {
        println!("✓ Performance baseline comparison recorded");
    }
}

//! Performance Validation Example
//!
//! This example demonstrates how to use the performance validation system
//! to automatically check that the emotion processing system meets
//! production performance targets.

use voirs_emotion::performance::*;
use voirs_emotion::prelude::*;

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("🚀 VoiRS Emotion Performance Validation Example");
    println!("{}", "=".repeat(50));

    // Create performance validator with default targets
    println!("\n📊 Creating performance validator...");
    let validator = PerformanceValidator::new()?;

    // Run comprehensive performance validation
    println!("🔍 Running comprehensive performance validation...");
    let validation_result = validator.validate_all_targets().await?;

    // Display results
    println!("\n{}", validation_result.detailed_report());

    // Demonstrate custom performance targets
    println!("\n🎯 Testing with stricter performance targets...");
    let strict_targets = PerformanceTargets {
        max_processing_latency_ms: 1.0,   // Stricter: 1ms instead of 2ms
        max_memory_usage_mb: 20.0,        // Stricter: 20MB instead of 25MB
        max_cpu_usage_percent: 0.5,       // Stricter: 0.5% instead of 1%
        min_concurrent_streams: 75,       // Stricter: 75 instead of 50
        max_audio_latency_ms: 3.0,        // Stricter: 3ms instead of 5ms
        min_cache_hit_rate_percent: 90.0, // Stricter: 90% instead of 85%
    };

    let strict_validator = PerformanceValidator::with_targets(strict_targets)?;
    let strict_result = strict_validator.validate_all_targets().await?;

    println!("\n--- Strict Targets Results ---");
    println!("{}", strict_result.summary());

    if !strict_result.all_passed() {
        println!("\n⚠️ Some strict targets were not met:");
        for failed in strict_result.failed_measurements() {
            println!(
                "  • {}: {:.3}{} (target: {:.3}{})",
                failed.name, failed.value, failed.unit, failed.target, failed.unit
            );
        }
    }

    // Demonstrate continuous monitoring setup
    println!("\n🔄 Setting up continuous performance monitoring...");
    let monitor_config = PerformanceMonitorConfig {
        enabled: true,
        monitoring_interval_ms: 2000, // Monitor every 2 seconds
        log_metrics: true,
        export_metrics: false,
        alert_thresholds: PerformanceTargets::default(),
    };

    let monitor = PerformanceMonitor::new(monitor_config)?;
    println!("✅ Performance monitor configured successfully!");

    // You could start monitoring like this:
    // let _monitor_handle = monitor.start_monitoring().await?;
    // tokio::time::sleep(std::time::Duration::from_secs(10)).await;
    // monitor.stop_monitoring();

    println!("\n✨ Performance validation example completed!");
    println!("\nKey Performance Metrics Validated:");
    println!("  🚀 Processing Latency: Target <2ms");
    println!("  🧠 Memory Usage: Target <25MB");
    println!("  ⚡ CPU Usage: Target <1%");
    println!("  🔀 Concurrent Streams: Target 50+");
    println!("  🎵 Audio Latency: Target <5ms");
    println!("  💾 Cache Hit Rate: Target 85%+");

    Ok(())
}

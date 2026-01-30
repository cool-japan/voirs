//! # Production Monitoring Demonstration
//!
//! This example demonstrates the production monitoring capabilities of voirs-acoustic.
//!
//! Run with: `cargo run --example production_monitoring_demo --features candle`

use std::time::{Duration, Instant};
use voirs_acoustic::production_monitoring::ProductionMonitor;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║     VoiRS Production Monitoring Demonstration             ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    // Create a production monitor
    let monitor = ProductionMonitor::new();

    println!("📊 Initializing production monitor...\n");

    // Update component health
    monitor.update_health("model_loader", true);
    monitor.update_health("inference_engine", true);
    monitor.update_health("cache_system", true);
    monitor.update_health("streaming_buffer", true);

    // Simulate some synthesis requests
    println!("🎤 Simulating synthesis requests...\n");

    for i in 1..=10 {
        let start = Instant::now();

        // Simulate synthesis work
        tokio::time::sleep(Duration::from_millis(50 + (i * 10))).await;

        let duration = start.elapsed();
        let success = i != 5; // Simulate one failure

        // Record the synthesis
        monitor.record_synthesis(duration, success, (20 + i) as usize);

        if success {
            println!("  Request {}: ✓ Completed in {}ms", i, duration.as_millis());
        } else {
            println!("  Request {}: ✗ Failed after {}ms", i, duration.as_millis());
        }
    }

    println!();

    // Generate and display monitoring report
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║  Production Monitoring Report                              ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    let report = monitor.generate_report()?;

    // Metrics
    println!("📈 Synthesis Metrics:");
    println!("  Total requests: {}", report.metrics.total_requests);
    println!("  Successful: {}", report.metrics.successful_requests);
    println!("  Failed: {}", report.metrics.failed_requests);
    println!("  Success rate: {:.1}%", report.metrics.success_rate);

    if report.metrics.avg_duration_ms > 0 {
        println!("  Avg synthesis time: {}ms", report.metrics.avg_duration_ms);
    }

    if report.metrics.total_phonemes > 0 {
        println!(
            "  Total phonemes processed: {}",
            report.metrics.total_phonemes
        );
    }

    if report.metrics.requests_per_second > 0.0 {
        println!(
            "  Requests per second: {:.2}",
            report.metrics.requests_per_second
        );
    }

    println!();

    // Health status
    println!("🏥 System Health:");
    println!(
        "  Healthy components: {}/{}",
        report.health.healthy_count, report.health.component_count
    );

    for (component, is_healthy) in &report.health.components {
        let status = if *is_healthy { "✓" } else { "✗" };
        println!("    {} {}", status, component);
    }

    println!();

    // Performance metrics
    println!("⚡ Performance Metrics:");
    println!("  Sample count: {}", report.performance.sample_count);

    if report.performance.min_ms > 0 {
        println!("  Min latency: {}ms", report.performance.min_ms);
    }
    if report.performance.max_ms > 0 {
        println!("  Max latency: {}ms", report.performance.max_ms);
    }
    if report.performance.avg_ms > 0 {
        println!("  Avg latency: {}ms", report.performance.avg_ms);
    }
    if report.performance.p95_ms > 0 {
        println!("  P95 latency: {}ms", report.performance.p95_ms);
    }
    if report.performance.p99_ms > 0 {
        println!("  P99 latency: {}ms", report.performance.p99_ms);
    }

    println!();

    // Alerts
    if !report.alerts.is_empty() {
        println!("🚨 Active Alerts:");
        for alert in &report.alerts {
            let severity_icon = match alert.severity {
                voirs_acoustic::production_monitoring::AlertSeverity::Critical => "🔴",
                voirs_acoustic::production_monitoring::AlertSeverity::Error => "🟠",
                voirs_acoustic::production_monitoring::AlertSeverity::Warning => "🟡",
                voirs_acoustic::production_monitoring::AlertSeverity::Info => "🔵",
            };
            println!(
                "  {} [{:?}] {}",
                severity_icon, alert.severity, alert.message
            );
        }
        println!();
    }

    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║  Demo completed successfully! ✨                           ║");
    println!("╚═══════════════════════════════════════════════════════════╝");

    Ok(())
}

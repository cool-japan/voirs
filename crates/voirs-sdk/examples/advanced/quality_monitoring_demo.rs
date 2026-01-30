//! Real-time Quality Monitoring Dashboard Demo
//!
//! Demonstrates the comprehensive real-time quality monitoring system
//! with trend analysis, alerting, and dashboard metrics.
//!
//! This example shows:
//! - Real-time quality metrics collection
//! - Trend detection and analysis
//! - Alert generation for quality degradation
//! - Dashboard data aggregation
//! - Integration with adaptive controller

use std::time::SystemTime;
use voirs_sdk::adaptive::{
    AdaptiveConfig, AdaptiveController, AlertSeverity, AlertThreshold, DashboardData,
    MonitorConfig, QualityMetricSample, QualityMonitor, QualityTarget, SystemMetrics,
    TrendDirection,
};
use voirs_sdk::Result;

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Real-time Quality Monitoring Dashboard Demo ===\n");

    // Demo 1: Basic Quality Monitoring
    demo_basic_monitoring().await?;

    // Demo 2: Trend Detection
    demo_trend_detection().await?;

    // Demo 3: Alert System
    demo_alert_system().await?;

    // Demo 4: Dashboard Integration
    demo_dashboard_integration().await?;

    // Demo 5: Integrated Monitoring with Adaptive Controller
    demo_integrated_monitoring().await?;

    Ok(())
}

/// Demo 1: Basic Quality Monitoring
async fn demo_basic_monitoring() -> Result<()> {
    println!("--- Demo 1: Basic Quality Monitoring ---");

    let monitor = QualityMonitor::new(MonitorConfig::default());

    // Simulate 20 synthesis operations with varying quality
    println!("Recording 20 quality samples...");

    for i in 0..20 {
        let quality_score = 70.0 + (i as f32 * 0.5).sin() * 10.0;
        let sample = QualityMetricSample {
            timestamp: SystemTime::now(),
            quality_score,
            latency_ms: 80 + (i % 5) * 10,
            rtf: 0.4 + (i as f32 * 0.1).sin() * 0.1,
            cpu_usage: 0.5 + (i as f32 * 0.2).cos() * 0.1,
            memory_usage: 0.45 + (i as f32 * 0.15).sin() * 0.05,
            success: true,
            text_complexity: Some(0.5 + (i as f32 * 0.3).cos() * 0.2),
        };

        monitor.record_sample(sample).await?;
    }

    // Get dashboard data
    let dashboard = monitor.get_dashboard_data().await?;
    print_dashboard(&dashboard);

    println!();
    Ok(())
}

/// Demo 2: Trend Detection
async fn demo_trend_detection() -> Result<()> {
    println!("--- Demo 2: Trend Detection ---");

    let config = MonitorConfig::default()
        .with_sample_window(50)
        .with_trend_sensitivity(0.05);

    let monitor = QualityMonitor::new(config);

    println!("\nScenario 1: Improving Quality Trend");
    for i in 0..30 {
        let quality_score = 60.0 + i as f32 * 0.5; // Gradually improving
        monitor
            .record_sample(create_sample(quality_score, 100, 0.5))
            .await?;
    }

    let dashboard = monitor.get_dashboard_data().await?;
    println!("Quality Trend: {:?}", dashboard.quality_trend);
    println!("Avg Quality: {:.1}", dashboard.current_quality_score);

    // Clear and test degrading trend
    monitor.clear().await?;

    println!("\nScenario 2: Degrading Quality Trend");
    for i in 0..30 {
        let quality_score = 80.0 - i as f32 * 0.4; // Gradually degrading
        monitor
            .record_sample(create_sample(quality_score, 120, 0.6))
            .await?;
    }

    let dashboard = monitor.get_dashboard_data().await?;
    println!("Quality Trend: {:?}", dashboard.quality_trend);
    println!("Avg Quality: {:.1}", dashboard.current_quality_score);

    println!();
    Ok(())
}

/// Demo 3: Alert System
async fn demo_alert_system() -> Result<()> {
    println!("--- Demo 3: Alert System ---");

    let config = MonitorConfig::default()
        .with_alert_threshold(AlertThreshold::QualityDrop(0.15)) // 15% drop
        .with_alert_threshold(AlertThreshold::LatencyExceeds(200)) // 200ms
        .with_alert_threshold(AlertThreshold::ErrorRateExceeds(0.20)); // 20% error rate

    let monitor = QualityMonitor::new(config);

    println!("\nSimulating quality degradation scenario...");

    // Establish baseline
    for _ in 0..15 {
        monitor.record_sample(create_sample(80.0, 100, 0.5)).await?;
    }

    // Sudden quality drop (should trigger alert)
    println!("\nSimulating sudden quality drop...");
    for _ in 0..5 {
        monitor.record_sample(create_sample(60.0, 100, 0.5)).await?;
    }

    // High latency spike (should trigger alert)
    println!("Simulating latency spike...");
    monitor.record_sample(create_sample(75.0, 300, 0.8)).await?;

    // High error rate (should trigger alert)
    println!("Simulating errors...");
    for i in 0..10 {
        let mut sample = create_sample(75.0, 100, 0.5);
        sample.success = i % 3 != 0; // 33% error rate
        monitor.record_sample(sample).await?;
    }

    // Display alerts
    let alerts = monitor.get_recent_alerts(10).await?;
    println!("\nGenerated Alerts: {}", alerts.len());
    for alert in alerts {
        println!(
            "  [{:?}] {}: {:.2} (threshold: {:.2})",
            alert.severity, alert.message, alert.current_value, alert.threshold_value
        );
    }

    println!();
    Ok(())
}

/// Demo 4: Dashboard Integration
async fn demo_dashboard_integration() -> Result<()> {
    println!("--- Demo 4: Dashboard Integration ---");

    let monitor = QualityMonitor::new(MonitorConfig::default());

    // Simulate realistic synthesis workload
    println!("\nSimulating realistic synthesis workload (50 operations)...");

    for i in 0..50 {
        // Simulate varying conditions
        let quality = match i {
            0..=20 => 75.0 + fastrand::f32() * 5.0,
            21..=35 => 70.0 + fastrand::f32() * 8.0, // Slight degradation
            36..=50 => 72.0 + fastrand::f32() * 6.0, // Recovery
            _ => 75.0,
        };

        let latency = if i % 10 == 0 {
            150 // Occasional spike
        } else {
            90 + fastrand::u64(0..30)
        };

        let rtf = 0.45 + fastrand::f32() * 0.15;
        let success = fastrand::f32() > 0.05; // 95% success rate

        let mut sample = create_sample(quality, latency, rtf);
        sample.success = success;
        monitor.record_sample(sample).await?;
    }

    // Get comprehensive dashboard data
    let dashboard = monitor.get_dashboard_data().await?;

    println!("\n=== Dashboard Metrics ===");
    println!(
        "Current Quality: {:.1}/100",
        dashboard.current_quality_score
    );
    println!("Quality Trend: {:?}", dashboard.quality_trend);
    println!("Avg Latency: {:.0}ms", dashboard.avg_latency_ms);
    println!("Latency Trend: {:?}", dashboard.latency_trend);
    println!("Current RTF: {:.3}", dashboard.current_rtf);
    println!("Success Rate: {:.1}%", dashboard.success_rate * 100.0);
    println!("Total Samples: {}", dashboard.total_samples);

    println!("\n=== Metric Statistics ===");
    for stat in &dashboard.metric_stats {
        println!("\n{}:", stat.name);
        println!("  Current: {:.2}", stat.current);
        println!("  Average: {:.2}", stat.average);
        println!("  Range: {:.2} - {:.2}", stat.min, stat.max);
        println!("  Std Dev: {:.2}", stat.std_dev);
        println!("  Trend: {:?}", stat.trend);
        println!("  Samples: {}", stat.sample_count);
    }

    if !dashboard.recent_alerts.is_empty() {
        println!("\n=== Recent Alerts ===");
        for alert in &dashboard.recent_alerts {
            println!("[{:?}] {}", alert.severity, alert.message);
        }
    }

    println!();
    Ok(())
}

/// Demo 5: Integrated Monitoring with Adaptive Controller
async fn demo_integrated_monitoring() -> Result<()> {
    println!("--- Demo 5: Integrated Monitoring with Adaptive Controller ---");

    // Setup adaptive controller with ML prediction
    let adaptive_config = AdaptiveConfig::default()
        .with_target_latency(150)
        .with_min_quality(QualityTarget::Low)
        .with_max_quality(QualityTarget::VeryHigh)
        .with_prediction(true);

    let controller = AdaptiveController::new(adaptive_config);

    // Setup quality monitor
    let monitor_config = MonitorConfig::default()
        .with_sample_window(50)
        .with_alert_threshold(AlertThreshold::QualityDrop(0.20));

    let monitor = QualityMonitor::new(monitor_config);

    println!("\nSimulating adaptive synthesis with monitoring...");

    // Simulate changing system conditions
    let scenarios = vec![
        (0.3, 0.4, 0.4, "Optimal conditions"),
        (0.5, 0.5, 0.6, "Light load"),
        (0.7, 0.6, 0.9, "Moderate load"),
        (0.9, 0.8, 1.2, "Heavy load"),
        (0.8, 0.7, 1.0, "Recovery phase"),
        (0.4, 0.5, 0.5, "Back to normal"),
    ];

    for (iteration, (cpu, mem, rtf, description)) in scenarios.iter().enumerate() {
        println!("\n[Iteration {}] {}", iteration + 1, description);

        // Create system metrics
        let metrics = SystemMetrics {
            cpu_usage: *cpu,
            memory_usage: *mem,
            current_rtf: *rtf,
            recent_latency_ms: (*rtf * 100.0) as u64,
            timestamp: std::time::Instant::now(),
        };

        // Update adaptive controller
        if let Some(new_quality) = controller.update_metrics(metrics.clone()).await? {
            println!("  Adaptive quality adjusted to: {:?}", new_quality);
        }

        // Get recommended quality
        let quality = controller.get_recommended_quality().await?;

        // Calculate latency based on system load
        let latency = (100.0 + cpu * 200.0) as u64;

        // Record performance
        controller
            .record_performance(quality, latency, true)
            .await?;

        // Record to monitor
        monitor
            .record_from_metrics(&metrics, quality, latency, true, Some(0.5))
            .await?;

        // Wait between iterations
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }

    // Get final statistics
    println!("\n=== Final Statistics ===");

    let dashboard = monitor.get_dashboard_data().await?;
    println!("\nQuality Monitor:");
    println!(
        "  Quality Score: {:.1}/100",
        dashboard.current_quality_score
    );
    println!("  Quality Trend: {:?}", dashboard.quality_trend);
    println!("  Success Rate: {:.1}%", dashboard.success_rate * 100.0);

    let adaptation_stats = controller.get_adaptation_stats().await?;
    println!("\nAdaptive Controller:");
    println!(
        "  Total Adjustments: {}",
        adaptation_stats.total_adjustments
    );
    println!("  Current Quality: {:?}", adaptation_stats.current_quality);
    println!(
        "  Success Rate: {:.1}%",
        adaptation_stats.success_rate * 100.0
    );

    if let Some(predictor_stats) = controller.get_predictor_stats().await? {
        println!("\nML Predictor:");
        println!("  Training Samples: {}", predictor_stats.total_samples);
        println!("  Model Confidence: {:.2}", predictor_stats.confidence);
    }

    if !dashboard.recent_alerts.is_empty() {
        println!("\nAlerts Generated: {}", dashboard.recent_alerts.len());
        for alert in dashboard.recent_alerts.iter().take(3) {
            println!("  - {}", alert.message);
        }
    }

    println!("\n=== Quality Monitoring Demo Complete ===");
    Ok(())
}

// Helper functions

fn create_sample(quality: f32, latency: u64, rtf: f32) -> QualityMetricSample {
    QualityMetricSample {
        timestamp: SystemTime::now(),
        quality_score: quality,
        latency_ms: latency,
        rtf,
        cpu_usage: 0.5,
        memory_usage: 0.5,
        success: true,
        text_complexity: Some(0.5),
    }
}

fn print_dashboard(dashboard: &DashboardData) {
    println!("\n=== Dashboard Summary ===");
    println!(
        "Quality: {:.1}/100 (Trend: {:?})",
        dashboard.current_quality_score, dashboard.quality_trend
    );
    println!(
        "Latency: {:.0}ms (Trend: {:?})",
        dashboard.avg_latency_ms, dashboard.latency_trend
    );
    println!("RTF: {:.3}", dashboard.current_rtf);
    println!("Success Rate: {:.1}%", dashboard.success_rate * 100.0);
    println!("Samples: {}", dashboard.total_samples);

    if !dashboard.recent_alerts.is_empty() {
        println!("Alerts: {}", dashboard.recent_alerts.len());
    }
}

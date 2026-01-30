//! Production Features Demonstration
//!
//! This example demonstrates advanced production-ready features including:
//! - Adaptive quality control
//! - Production readiness checking
//! - System diagnostics
//! - Real-time quality adaptation
//!
//! Run with:
//! ```bash
//! cargo run --example production_features --all-features
//! ```

use std::time::Instant;
use voirs_sdk::prelude::*;
use voirs_sdk::{
    adaptive::{AdaptiveConfig, AdaptiveController, SystemMetrics},
    diagnostics::{ProductionReadiness, ReadinessConfig},
};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("=== VoiRS SDK Production Features Demo ===\n");

    // 1. Production Readiness Check
    println!("1. Running Production Readiness Check...");
    run_readiness_check().await?;

    println!("\n{}\n", "=".repeat(60));

    // 2. Adaptive Quality Control Demo
    println!("2. Demonstrating Adaptive Quality Control...");
    run_adaptive_quality_demo().await?;

    println!("\n{}\n", "=".repeat(60));

    // 3. Stress Test Simulation
    println!("3. Simulating System Under Stress...");
    run_stress_test_simulation().await?;

    println!("\n{}\n", "=".repeat(60));

    // 4. Combined Production Workflow
    println!("4. Complete Production Workflow...");
    run_production_workflow().await?;

    Ok(())
}

/// Run comprehensive production readiness check
async fn run_readiness_check() -> Result<()> {
    let config = ReadinessConfig {
        min_cpu_cores: 4,
        min_memory_gb: 4.0,
        min_disk_gb: 10.0,
        max_acceptable_rtf: 0.5,
        max_latency_ms: 150,
        enable_security_checks: true,
        enable_benchmarking: true,
        cache_dir: None,
    };

    let checker = ProductionReadiness::new(config);
    let report = checker.check_readiness().await?;

    println!("{}", report.to_report_string());

    if !report.is_production_ready() {
        println!("\n⚠ System is not production ready!");
        println!("\nCritical issues to address:");
        for issue in report.critical_issues() {
            println!("  ✗ {}", issue);
        }

        println!("\nHigh priority recommendations:");
        for rec in report.high_priority_recommendations() {
            println!("  → {}", rec.message);
        }
    } else {
        println!("\n✓ System is production ready!");
    }

    Ok(())
}

/// Demonstrate adaptive quality control
async fn run_adaptive_quality_demo() -> Result<()> {
    // Create adaptive controller with production settings
    let config = AdaptiveConfig::default()
        .with_target_latency(100)
        .with_min_quality(QualityTarget::Medium)
        .with_max_quality(QualityTarget::VeryHigh)
        .with_adaptation_speed(0.7);

    let controller = AdaptiveController::new(config);

    println!(
        "Initial quality: {:?}",
        controller.get_recommended_quality().await?
    );

    // Simulate normal load
    println!("\nSimulating normal system load...");
    let normal_metrics = SystemMetrics {
        cpu_usage: 0.4,
        memory_usage: 0.5,
        current_rtf: 0.3,
        recent_latency_ms: 60,
        timestamp: Instant::now(),
    };

    tokio::time::sleep(std::time::Duration::from_secs(2)).await;

    if let Some(new_quality) = controller.update_metrics(normal_metrics.clone()).await? {
        println!("  Quality adjusted to: {:?}", new_quality);
    } else {
        println!(
            "  Quality maintained at: {:?}",
            controller.get_recommended_quality().await?
        );
    }

    // Simulate increased load
    println!("\nSimulating increased system load...");
    let high_load_metrics = SystemMetrics {
        cpu_usage: 0.85,
        memory_usage: 0.80,
        current_rtf: 0.9,
        recent_latency_ms: 180,
        timestamp: Instant::now(),
    };

    tokio::time::sleep(std::time::Duration::from_secs(2)).await;

    if let Some(new_quality) = controller.update_metrics(high_load_metrics).await? {
        println!(
            "  Quality reduced to: {:?} (system under stress)",
            new_quality
        );
    }

    // Record some performance data
    controller
        .record_performance(QualityTarget::High, 120, true)
        .await?;
    controller
        .record_performance(QualityTarget::Medium, 80, true)
        .await?;

    // Get statistics
    let stats = controller.get_adaptation_stats().await?;
    println!("\nAdaptation Statistics:");
    println!("  Total adjustments: {}", stats.total_adjustments);
    println!("  Current quality: {:?}", stats.current_quality);

    Ok(())
}

/// Simulate system under stress with quality adaptation
async fn run_stress_test_simulation() -> Result<()> {
    let controller = AdaptiveController::new(
        AdaptiveConfig::default()
            .with_adaptation_speed(1.0) // Aggressive adaptation
            .with_min_quality(QualityTarget::Low),
    );

    println!(
        "Starting with quality: {:?}",
        controller.get_recommended_quality().await?
    );

    // Gradually increase system load
    for i in 1..=5 {
        let load_factor = i as f32 * 0.2;

        println!("\nLoad level {}/5 ({}%)", i, (load_factor * 100.0) as u32);

        let metrics = SystemMetrics {
            cpu_usage: 0.5 + load_factor * 0.4,
            memory_usage: 0.6 + load_factor * 0.3,
            current_rtf: 0.3 + load_factor * 0.6,
            recent_latency_ms: 50 + (load_factor * 150.0) as u64,
            timestamp: Instant::now(),
        };

        println!(
            "  CPU: {:.1}%, Memory: {:.1}%, RTF: {:.2}",
            metrics.cpu_usage * 100.0,
            metrics.memory_usage * 100.0,
            metrics.current_rtf
        );

        tokio::time::sleep(std::time::Duration::from_secs(2)).await;

        if let Some(new_quality) = controller.update_metrics(metrics).await? {
            println!(
                "  → Quality adapted to: {:?} (score: {})",
                new_quality,
                new_quality.score()
            );
        } else {
            let current = controller.get_recommended_quality().await?;
            println!(
                "  → Quality maintained: {:?} (score: {})",
                current,
                current.score()
            );
        }
    }

    // Gradually decrease load
    println!("\nReducing system load...");
    for i in (1..=3).rev() {
        let load_factor = i as f32 * 0.2;

        let metrics = SystemMetrics {
            cpu_usage: 0.3 + load_factor * 0.2,
            memory_usage: 0.4 + load_factor * 0.2,
            current_rtf: 0.2 + load_factor * 0.3,
            recent_latency_ms: 40 + (load_factor * 80.0) as u64,
            timestamp: Instant::now(),
        };

        tokio::time::sleep(std::time::Duration::from_secs(2)).await;

        if let Some(new_quality) = controller.update_metrics(metrics).await? {
            println!(
                "  Quality increased to: {:?} (score: {})",
                new_quality,
                new_quality.score()
            );
        }
    }

    let final_stats = controller.get_adaptation_stats().await?;
    println!("\nFinal Statistics:");
    println!(
        "  Total quality adjustments: {}",
        final_stats.total_adjustments
    );
    println!("  Final quality level: {:?}", final_stats.current_quality);

    Ok(())
}

/// Complete production workflow demonstration
async fn run_production_workflow() -> Result<()> {
    println!("Step 1: Check production readiness");

    let readiness_config = ReadinessConfig::default();
    let checker = ProductionReadiness::new(readiness_config);
    let report = checker.check_readiness().await?;

    if !report.is_production_ready() {
        println!("  ✗ System not ready. Please address issues before deployment.");
        return Ok(());
    }

    println!("  ✓ Production readiness verified");

    println!("\nStep 2: Initialize adaptive quality control");
    let adaptive_config = AdaptiveConfig::default()
        .with_target_latency(100)
        .with_min_quality(QualityTarget::Medium)
        .with_adaptation_speed(0.6);

    let controller = AdaptiveController::new(adaptive_config);
    println!("  ✓ Adaptive controller initialized");

    println!("\nStep 3: Simulate production workload");

    // Simulate various workload scenarios
    let scenarios = vec![
        ("Normal Load", 0.4, 0.5, 0.3),
        ("Peak Hours", 0.75, 0.7, 0.7),
        ("Off-Peak", 0.2, 0.3, 0.2),
        ("Spike", 0.95, 0.9, 1.1),
    ];

    for (name, cpu, mem, rtf) in scenarios {
        println!("\n  Scenario: {}", name);

        let metrics = SystemMetrics {
            cpu_usage: cpu,
            memory_usage: mem,
            current_rtf: rtf,
            recent_latency_ms: (rtf * 200.0) as u64,
            timestamp: Instant::now(),
        };

        tokio::time::sleep(std::time::Duration::from_secs(2)).await;

        controller.update_metrics(metrics).await?;
        let quality = controller.get_recommended_quality().await?;

        println!(
            "    CPU: {:.0}%, Mem: {:.0}%, RTF: {:.2}",
            cpu * 100.0,
            mem * 100.0,
            rtf
        );
        println!(
            "    Quality Level: {:?} (score: {})",
            quality,
            quality.score()
        );

        // Record performance
        controller
            .record_performance(quality, (rtf * 150.0) as u64, true)
            .await?;
    }

    println!("\nStep 4: Review final statistics");
    let stats = controller.get_adaptation_stats().await?;

    println!("  Total quality adaptations: {}", stats.total_adjustments);
    println!("  Current quality setting: {:?}", stats.current_quality);
    println!("  Performance history entries: {}", stats.history_entries);
    println!("  Success rate: {:.1}%", stats.success_rate * 100.0);
    println!(
        "  Average synthesis time: {:.0} ms",
        stats.avg_synthesis_time_ms
    );

    println!("\n✓ Production workflow completed successfully!");

    Ok(())
}

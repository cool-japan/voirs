//! Comprehensive performance profiling example for VoiRS SDK.
//!
//! This example demonstrates:
//! - Basic profiling session management
//! - Stage-by-stage timing analysis
//! - Memory profiling and leak detection
//! - Bottleneck detection and recommendations
//! - Performance comparison between sessions
//! - Regression detection
//! - Report generation in multiple formats
//! - Real-time performance monitoring
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example performance_profiling --features emotion,cloning
//! ```

use std::sync::Arc;
use std::time::Duration;
use voirs_sdk::prelude::*;
use voirs_sdk::profiling::{
    PerformanceComparator, PipelineStage, Profiler, ProfilerConfig, ReportFormat, ReportGenerator,
    RegressionDetector,
};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing for logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("=== VoiRS SDK - Performance Profiling Example ===\n");

    // Create pipeline
    println!("Creating VoiRS pipeline...");
    let pipeline = Arc::new(
        VoirsPipelineBuilder::new()
            .with_test_mode(true)
            .with_quality(QualityLevel::High)
            .build()
            .await?,
    );
    println!("Pipeline created successfully!\n");

    // Example 1: Basic profiling session
    println!("Example 1: Basic Profiling Session");
    println!("-".repeat(60));
    basic_profiling_session(&pipeline).await?;
    println!();

    // Example 2: Stage-by-stage analysis
    println!("Example 2: Stage-by-Stage Performance Analysis");
    println!("-".repeat(60));
    stage_by_stage_analysis(&pipeline).await?;
    println!();

    // Example 3: Memory profiling
    println!("Example 3: Memory Profiling and Leak Detection");
    println!("-".repeat(60));
    memory_profiling(&pipeline).await?;
    println!();

    // Example 4: Bottleneck detection
    println!("Example 4: Automatic Bottleneck Detection");
    println!("-".repeat(60));
    bottleneck_detection(&pipeline).await?;
    println!();

    // Example 5: Performance comparison
    println!("Example 5: Performance Comparison Between Sessions");
    println!("-".repeat(60));
    performance_comparison(&pipeline).await?;
    println!();

    // Example 6: Regression detection
    println!("Example 6: Regression Detection");
    println!("-".repeat(60));
    regression_detection(&pipeline).await?;
    println!();

    // Example 7: Report generation
    println!("Example 7: Multi-Format Report Generation");
    println!("-".repeat(60));
    report_generation(&pipeline).await?;
    println!();

    // Example 8: Real-time monitoring
    println!("Example 8: Real-Time Performance Monitoring");
    println!("-".repeat(60));
    realtime_monitoring(&pipeline).await?;
    println!();

    println!("All profiling examples completed successfully!");

    Ok(())
}

/// Example 1: Basic profiling session
async fn basic_profiling_session(pipeline: &Arc<VoirsPipeline>) -> Result<()> {
    println!("Starting a basic profiling session...\n");

    // Create profiler with default configuration
    let profiler = Profiler::new(ProfilerConfig::default());

    // Start profiling session
    let session = profiler.start_session("basic_synthesis").await;
    println!("Session started: {}", session.name);

    // Perform some synthesis operations
    for i in 1..=5 {
        let text = format!("This is synthesis test number {}", i);
        let _audio = pipeline.synthesize(&text).await?;
        println!("  Completed synthesis {}/5", i);
    }

    // End session and get report
    let session = profiler.end_session(session).await?;

    println!("\nSession Summary:");
    println!("  Duration: {:.2}ms", session.duration.unwrap().as_millis());
    println!("  Stages profiled: {}", session.stage_metrics.len());
    println!("  Memory snapshots: {}", session.memory_snapshots.len());
    println!("  Bottlenecks detected: {}", session.bottlenecks.len());

    // Display stage metrics
    println!("\nStage Performance:");
    for (stage_name, metrics) in &session.stage_metrics {
        println!("  {}: {:.2}ms avg ({} executions)",
                 stage_name,
                 metrics.average_duration.as_millis(),
                 metrics.execution_count);
    }

    Ok(())
}

/// Example 2: Stage-by-stage performance analysis
async fn stage_by_stage_analysis(pipeline: &Arc<VoirsPipeline>) -> Result<()> {
    println!("Analyzing performance of each pipeline stage...\n");

    let config = ProfilerConfig {
        enable_timing: true,
        enable_memory: false, // Focus on timing only
        ..Default::default()
    };

    let profiler = Profiler::new(config);
    let session = profiler.start_session("stage_analysis").await;

    // Perform synthesis
    let text = "This is a detailed performance analysis of the synthesis pipeline stages.";
    let _audio = pipeline.synthesize(text).await?;

    let session = profiler.end_session(session).await?;

    println!("Detailed Stage Analysis:\n");

    // Sort stages by average duration
    let mut stages: Vec<_> = session.stage_metrics.iter().collect();
    stages.sort_by(|a, b| b.1.average_duration.cmp(&a.1.average_duration));

    for (stage_name, metrics) in stages {
        println!("Stage: {}", stage_name);
        println!("  Executions: {}", metrics.execution_count);
        println!("  Average duration: {:.2}ms", metrics.average_duration.as_millis());
        println!("  Min duration: {:.2}ms", metrics.min_duration.as_millis());
        println!("  Max duration: {:.2}ms", metrics.max_duration.as_millis());
        println!("  Total duration: {:.2}ms", metrics.total_duration.as_millis());
        if let Some(pct) = metrics.percentage_of_total {
            println!("  Percentage of total: {:.1}%", pct);
        }
        if let Some(std_dev) = metrics.std_deviation {
            println!("  Std deviation: {:.2}ms", std_dev.as_millis());
        }
        println!();
    }

    Ok(())
}

/// Example 3: Memory profiling and leak detection
async fn memory_profiling(pipeline: &Arc<VoirsPipeline>) -> Result<()> {
    println!("Profiling memory usage and checking for leaks...\n");

    let config = ProfilerConfig {
        enable_timing: false, // Focus on memory only
        enable_memory: true,
        ..Default::default()
    };

    let profiler = Profiler::new(config);
    let session = profiler.start_session("memory_profiling").await;

    // Perform multiple synthesis operations to observe memory patterns
    for i in 1..=10 {
        let text = format!("Memory profiling test iteration {}", i);
        let _audio = pipeline.synthesize(&text).await?;
    }

    let session = profiler.end_session(session).await?;

    println!("Memory Profiling Results:\n");
    println!("Total snapshots: {}", session.memory_snapshots.len());

    if let Some(first) = session.memory_snapshots.first() {
        if let Some(last) = session.memory_snapshots.last() {
            let growth = last.total_allocated as i64 - first.total_allocated as i64;
            let growth_pct = (growth as f64 / first.total_allocated.max(1) as f64) * 100.0;

            println!("Initial allocation: {} bytes", first.total_allocated);
            println!("Final allocation: {} bytes", last.total_allocated);
            println!("Memory growth: {} bytes ({:+.1}%)", growth, growth_pct);
            println!();

            if let Some(peak) = session.memory_snapshots.iter()
                .max_by_key(|s| s.total_allocated) {
                println!("Peak memory usage: {} bytes", peak.total_allocated);
            }

            // Check for potential memory leaks
            if growth_pct > 50.0 {
                println!("\n⚠️  WARNING: Significant memory growth detected!");
                println!("   This may indicate a memory leak.");
            } else {
                println!("\n✓ Memory usage appears stable.");
            }
        }
    }

    Ok(())
}

/// Example 4: Automatic bottleneck detection
async fn bottleneck_detection(pipeline: &Arc<VoirsPipeline>) -> Result<()> {
    println!("Detecting performance bottlenecks...\n");

    let config = ProfilerConfig {
        enable_timing: true,
        enable_memory: true,
        enable_bottleneck_detection: true,
        ..Default::default()
    };

    let profiler = Profiler::new(config);
    let session = profiler.start_session("bottleneck_detection").await;

    // Perform synthesis
    let text = "Testing bottleneck detection in the synthesis pipeline.";
    let _audio = pipeline.synthesize(text).await?;

    let session = profiler.end_session(session).await?;

    println!("Bottleneck Detection Results:\n");

    if session.bottlenecks.is_empty() {
        println!("✓ No significant bottlenecks detected!");
    } else {
        println!("Found {} potential bottleneck(s):\n", session.bottlenecks.len());

        for (idx, bottleneck) in session.bottlenecks.iter().enumerate() {
            println!("Bottleneck #{}", idx + 1);
            println!("  Component: {}", bottleneck.component);
            println!("  Severity: {:?}", bottleneck.severity);
            println!("  Description: {}", bottleneck.description);
            println!("  Impact: {}", bottleneck.impact);
            if !bottleneck.recommendation.is_empty() {
                println!("  Recommendation: {}", bottleneck.recommendation);
            }
            println!();
        }
    }

    Ok(())
}

/// Example 5: Performance comparison between sessions
async fn performance_comparison(pipeline: &Arc<VoirsPipeline>) -> Result<()> {
    println!("Comparing performance between two sessions...\n");

    let profiler = Profiler::new(ProfilerConfig::default());

    // Session 1: Baseline
    println!("Running baseline session...");
    let session1 = profiler.start_session("baseline").await;
    let _audio1 = pipeline.synthesize("Baseline performance test.").await?;
    let session1 = profiler.end_session(session1).await?;
    println!("Baseline duration: {:.2}ms\n",
             session1.duration.unwrap().as_millis());

    // Session 2: Comparison
    println!("Running comparison session...");
    let session2 = profiler.start_session("comparison").await;
    let _audio2 = pipeline.synthesize("Comparison performance test.").await?;
    let session2 = profiler.end_session(session2).await?;
    println!("Comparison duration: {:.2}ms\n",
             session2.duration.unwrap().as_millis());

    // Compare sessions
    let comparator = PerformanceComparator::new();
    let comparison = comparator.compare(&session1, &session2);

    println!("Performance Comparison:\n");
    println!("Overall change: {:.1}%", comparison.overall_change_percent);

    println!("\nStage-by-Stage Comparison:");
    for (stage, change) in &comparison.stage_changes {
        let symbol = if *change > 0.0 { "↑" } else if *change < 0.0 { "↓" } else { "=" };
        println!("  {}: {}{:.1}%", stage, symbol, change.abs());
    }

    if let Some(memory_change) = comparison.memory_change_percent {
        println!("\nMemory change: {:.1}%", memory_change);
    }

    // Check for regressions
    if comparison.overall_change_percent > 10.0 {
        println!("\n⚠️  Performance regression detected!");
    } else if comparison.overall_change_percent < -10.0 {
        println!("\n✓ Performance improvement detected!");
    } else {
        println!("\n✓ Performance is stable.");
    }

    Ok(())
}

/// Example 6: Regression detection across multiple sessions
async fn regression_detection(pipeline: &Arc<VoirsPipeline>) -> Result<()> {
    println!("Running regression detection across multiple sessions...\n");

    let profiler = Arc::new(Profiler::new(ProfilerConfig {
        max_history_size: 10,
        enable_baseline_comparison: true,
        regression_threshold_percent: 15.0,
        ..Default::default()
    }));

    // Run multiple sessions
    println!("Running 5 test sessions...");
    for i in 1..=5 {
        let session = profiler.start_session(&format!("session_{}", i)).await;
        let text = format!("Regression test iteration {}", i);
        let _audio = pipeline.synthesize(&text).await?;
        let session = profiler.end_session(session).await?;
        println!("  Session {}: {:.2}ms", i, session.duration.unwrap().as_millis());
    }
    println!();

    // Detect regressions
    let detector = RegressionDetector::new(15.0); // 15% threshold
    let history = profiler.session_history().await;

    println!("Regression Detection Results:\n");

    if history.len() < 2 {
        println!("Not enough sessions for regression detection.");
    } else {
        let regressions = detector.detect_regressions(&history);

        if regressions.is_empty() {
            println!("✓ No regressions detected!");
        } else {
            println!("Found {} regression(s):\n", regressions.len());

            for (idx, regression) in regressions.iter().enumerate() {
                println!("Regression #{}", idx + 1);
                println!("  Description: {}", regression.description);
                println!("  Severity: {:?}", regression.severity);
                println!("  Change: {:.1}%", regression.change_percent);
                if !regression.affected_components.is_empty() {
                    println!("  Affected: {:?}", regression.affected_components);
                }
                println!();
            }
        }
    }

    Ok(())
}

/// Example 7: Multi-format report generation
async fn report_generation(pipeline: &Arc<VoirsPipeline>) -> Result<()> {
    println!("Generating performance reports in multiple formats...\n");

    let profiler = Profiler::new(ProfilerConfig::default());
    let session = profiler.start_session("report_generation").await;

    // Perform synthesis
    let text = "Generating comprehensive performance reports.";
    let _audio = pipeline.synthesize(text).await?;

    let session = profiler.end_session(session).await?;

    // Create report generator
    let generator = ReportGenerator::new();

    // Generate text report
    println!("=== Text Format Report ===");
    let text_report = generator.generate(&session, ReportFormat::Text)?;
    println!("{}\n", text_report);

    // Generate markdown report (first 500 chars)
    println!("=== Markdown Format Report (preview) ===");
    let markdown_report = generator.generate(&session, ReportFormat::Markdown)?;
    println!("{}...\n", &markdown_report[..markdown_report.len().min(500)]);

    // Generate JSON report (first 300 chars)
    println!("=== JSON Format Report (preview) ===");
    let json_report = generator.generate(&session, ReportFormat::Json)?;
    println!("{}...\n", &json_report[..json_report.len().min(300)]);

    println!("✓ Reports generated successfully in all formats!");

    Ok(())
}

/// Example 8: Real-time performance monitoring
async fn realtime_monitoring(pipeline: &Arc<VoirsPipeline>) -> Result<()> {
    println!("Monitoring performance in real-time...\n");

    let config = ProfilerConfig {
        sampling_interval_ms: 50, // Sample every 50ms
        enable_timing: true,
        enable_memory: true,
        ..Default::default()
    };

    let profiler = Profiler::new(config);
    let session = profiler.start_session("realtime_monitoring").await;

    println!("Performing synthesis with real-time monitoring...");

    // Simulate multiple operations
    for i in 1..=5 {
        let text = format!("Real-time monitoring test {}", i);
        let _audio = pipeline.synthesize(&text).await?;

        // In a real application, you could query profiler state here
        println!("  Operation {} completed", i);

        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    let session = profiler.end_session(session).await?;

    println!("\nReal-Time Monitoring Summary:");
    println!("  Total duration: {:.2}ms", session.duration.unwrap().as_millis());
    println!("  Memory snapshots: {}", session.memory_snapshots.len());
    println!("  Stages tracked: {}", session.stage_metrics.len());

    // Display memory timeline
    if !session.memory_snapshots.is_empty() {
        println!("\nMemory Usage Timeline:");
        for (idx, snapshot) in session.memory_snapshots.iter().enumerate().take(10) {
            println!("  Snapshot {}: {} bytes", idx + 1, snapshot.total_allocated);
        }
        if session.memory_snapshots.len() > 10 {
            println!("  ... ({} more snapshots)", session.memory_snapshots.len() - 10);
        }
    }

    Ok(())
}

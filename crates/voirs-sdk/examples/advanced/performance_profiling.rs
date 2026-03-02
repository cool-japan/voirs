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
    PerformanceComparator, Profiler, ProfilerConfig, RegressionDetector, ReportGenerator,
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
    println!("{}", "-".repeat(60));
    basic_profiling_session(&pipeline).await?;
    println!();

    // Example 2: Stage-by-stage analysis
    println!("Example 2: Stage-by-Stage Performance Analysis");
    println!("{}", "-".repeat(60));
    stage_by_stage_analysis(&pipeline).await?;
    println!();

    // Example 3: Memory profiling
    println!("Example 3: Memory Profiling and Leak Detection");
    println!("{}", "-".repeat(60));
    memory_profiling(&pipeline).await?;
    println!();

    // Example 4: Bottleneck detection
    println!("Example 4: Automatic Bottleneck Detection");
    println!("{}", "-".repeat(60));
    bottleneck_detection(&pipeline).await?;
    println!();

    // Example 5: Performance comparison
    println!("Example 5: Performance Comparison Between Sessions");
    println!("{}", "-".repeat(60));
    performance_comparison(&pipeline).await?;
    println!();

    // Example 6: Regression detection
    println!("Example 6: Regression Detection");
    println!("{}", "-".repeat(60));
    regression_detection(&pipeline).await?;
    println!();

    // Example 7: Report generation
    println!("Example 7: Multi-Format Report Generation");
    println!("{}", "-".repeat(60));
    report_generation(&pipeline).await?;
    println!();

    // Example 8: Real-time monitoring
    println!("Example 8: Real-Time Performance Monitoring");
    println!("{}", "-".repeat(60));
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
    let report = profiler.end_session(session).await?;

    println!("\nSession Summary:");
    println!(
        "  Duration: {:.2}ms",
        report.session.duration_seconds * 1000.0
    );
    println!("  Stages profiled: {}", report.stage_breakdown.len());
    println!(
        "  Memory snapshots: {}",
        report.memory_analysis.is_some() as usize
    );
    println!("  Bottlenecks detected: {}", report.bottlenecks.len());

    // Display stage metrics
    println!("\nStage Performance:");
    for stage in &report.stage_breakdown {
        println!(
            "  {}: {:.2}ms avg ({} executions)",
            stage.stage_name, stage.avg_duration_ms, stage.execution_count
        );
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

    let report = profiler.end_session(session).await?;

    println!("Detailed Stage Analysis:\n");

    // Sort stages by average duration
    let mut stages: Vec<_> = report.stage_breakdown.iter().collect();
    stages.sort_by(|a, b| {
        b.avg_duration_ms
            .partial_cmp(&a.avg_duration_ms)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for stage in stages {
        println!("Stage: {}", stage.stage_name);
        println!("  Executions: {}", stage.execution_count);
        println!("  Average duration: {:.2}ms", stage.avg_duration_ms);
        println!("  Min duration: {:.2}ms", stage.min_duration_ms);
        println!("  Max duration: {:.2}ms", stage.max_duration_ms);
        println!("  Total duration: {:.2}ms", stage.total_duration_ms);
        println!("  Percentage of total: {:.1}%", stage.percentage_of_total);
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

    let report = profiler.end_session(session).await?;

    println!("Memory Profiling Results:\n");

    if let Some(memory) = &report.memory_analysis {
        println!("Peak memory: {:.2} MB", memory.peak_mb);
        println!("Average memory: {:.2} MB", memory.average_mb);
        println!("Memory growth: {:+.1}%", memory.growth_percent);
        println!();

        // Check for potential memory leaks
        if memory.growth_percent > 50.0 {
            println!("WARNING: Significant memory growth detected!");
            println!("   This may indicate a memory leak.");
        } else {
            println!("Memory usage appears stable.");
        }
    } else {
        println!("No memory analysis data available.");
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

    let report = profiler.end_session(session).await?;

    println!("Bottleneck Detection Results:\n");

    if report.bottlenecks.is_empty() {
        println!("No significant bottlenecks detected!");
    } else {
        println!(
            "Found {} potential bottleneck(s):\n",
            report.bottlenecks.len()
        );

        for (idx, bottleneck) in report.bottlenecks.iter().enumerate() {
            println!("Bottleneck #{}", idx + 1);
            println!("  Component: {}", bottleneck.component);
            println!("  Severity: {:?}", bottleneck.severity);
            println!("  Impact: {}", bottleneck.impact_description);
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
    let report1 = profiler.end_session(session1).await?;
    println!(
        "Baseline duration: {:.2}ms\n",
        report1.session.duration_seconds * 1000.0
    );

    // Session 2: Comparison
    println!("Running comparison session...");
    let session2 = profiler.start_session("comparison").await;
    let _audio2 = pipeline.synthesize("Comparison performance test.").await?;
    let report2 = profiler.end_session(session2).await?;
    println!(
        "Comparison duration: {:.2}ms\n",
        report2.session.duration_seconds * 1000.0
    );

    // Compare sessions (need ProfileSession, not PerformanceReport)
    let all_sessions = profiler.get_sessions().await;
    if all_sessions.len() >= 2 {
        let comparator = PerformanceComparator::new();
        let comparison = comparator.compare(&all_sessions[0], &all_sessions[1]).await;

        println!("Performance Comparison:\n");
        println!("Overall change: {:.1}%", comparison.overall_change_percent);

        println!("\nStage-by-Stage Comparison:");
        for (stage, stage_cmp) in &comparison.stage_comparisons {
            let symbol = if stage_cmp.change_percent > 0.0 {
                "up"
            } else if stage_cmp.change_percent < 0.0 {
                "down"
            } else {
                "equal"
            };
            println!(
                "  {}: {} {:.1}%",
                stage,
                symbol,
                stage_cmp.change_percent.abs()
            );
        }

        if let Some(memory_cmp) = &comparison.memory_comparison {
            println!("\nMemory change: {:.1}%", memory_cmp.change_percent);
        }

        // Check for regressions
        if comparison.overall_change_percent > 10.0 {
            println!("\nPerformance regression detected!");
        } else if comparison.overall_change_percent < -10.0 {
            println!("\nPerformance improvement detected!");
        } else {
            println!("\nPerformance is stable.");
        }
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
        let report = profiler.end_session(session).await?;
        println!(
            "  Session {}: {:.2}ms",
            i,
            report.session.duration_seconds * 1000.0
        );
    }
    println!();

    // Detect regressions
    let detector = RegressionDetector::new(15.0, 5); // 15% threshold, 5 history
    let history = profiler.get_sessions().await;

    println!("Regression Detection Results:\n");

    if history.len() < 2 {
        println!("Not enough sessions for regression detection.");
    } else {
        let regressions = detector.detect_regressions(&history);

        if regressions.is_empty() {
            println!("No regressions detected!");
        } else {
            println!("Found {} regression(s):\n", regressions.len());

            for (idx, regression) in regressions.iter().enumerate() {
                println!("Regression #{}", idx + 1);
                println!("  Description: {}", regression.description);
                println!("  Severity: {}", regression.severity);
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

    let prof_session = session.clone();
    let report = profiler.end_session(session).await?;

    // Display the report summary
    println!("=== Performance Report ===");
    println!("{}", report.summary());
    println!();

    // Use report generator for detailed output
    let generator = ReportGenerator::new(ProfilerConfig::default());
    let detailed_report = generator.generate(&prof_session, None).await?;

    println!(
        "=== Detailed Report (JSON preview) ===\n{}\n",
        serde_json::to_string_pretty(&detailed_report)
            .unwrap_or_default()
            .chars()
            .take(300)
            .collect::<String>()
    );

    println!("Reports generated successfully!");

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

    let report = profiler.end_session(session).await?;

    println!("\nReal-Time Monitoring Summary:");
    println!(
        "  Total duration: {:.2}ms",
        report.session.duration_seconds * 1000.0
    );
    println!(
        "  Memory analysis: {}",
        if report.memory_analysis.is_some() {
            "available"
        } else {
            "not available"
        }
    );
    println!("  Stages tracked: {}", report.stage_breakdown.len());

    // Display memory analysis if available
    if let Some(memory) = &report.memory_analysis {
        println!("\nMemory Usage Summary:");
        println!("  Peak: {:.2} MB", memory.peak_mb);
        println!("  Average: {:.2} MB", memory.average_mb);
    }

    Ok(())
}

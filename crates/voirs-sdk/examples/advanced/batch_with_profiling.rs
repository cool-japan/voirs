//! Advanced batch processing with performance profiling example.
//!
//! This example demonstrates the integration of batch processing with
//! performance profiling to analyze and optimize large-scale synthesis workloads.
//!
//! Features demonstrated:
//! - Batch processing with performance profiling
//! - Identifying bottlenecks in batch operations
//! - Comparing different batch configurations
//! - Memory usage analysis during batch processing
//! - Optimization recommendations based on profiling data
//! - Performance metrics for production deployment
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example batch_with_profiling --features emotion,cloning
//! ```

use std::sync::Arc;
use std::time::Instant;
use voirs_sdk::batch::{BatchConfig, BatchProcessor, BatchRequest, SchedulingStrategy};
use voirs_sdk::prelude::*;
use voirs_sdk::profiling::{
    PerformanceComparator, PerformanceReport, Profiler, ProfilerConfig, ReportGenerator,
};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("=== VoiRS SDK - Batch Processing with Profiling ===\n");

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

    // Example 1: Basic batch profiling
    println!("Example 1: Basic Batch Processing with Profiling");
    println!("{}", "-".repeat(70));
    basic_batch_profiling(&pipeline).await?;
    println!();

    // Example 2: Comparing batch configurations
    println!("Example 2: Comparing Batch Configurations");
    println!("{}", "-".repeat(70));
    compare_batch_configurations(&pipeline).await?;
    println!();

    // Example 3: Memory analysis during batch processing
    println!("Example 3: Memory Usage Analysis");
    println!("{}", "-".repeat(70));
    batch_memory_analysis(&pipeline).await?;
    println!();

    // Example 4: Bottleneck identification
    println!("Example 4: Batch Bottleneck Identification");
    println!("{}", "-".repeat(70));
    identify_batch_bottlenecks(&pipeline).await?;
    println!();

    // Example 5: Production optimization
    println!("Example 5: Production Optimization Analysis");
    println!("{}", "-".repeat(70));
    production_optimization(&pipeline).await?;
    println!();

    println!("All integrated examples completed successfully!");

    Ok(())
}

/// Example 1: Profile a batch processing operation
async fn basic_batch_profiling(pipeline: &Arc<VoirsPipeline>) -> Result<()> {
    println!("Profiling a batch of 20 synthesis requests...\n");

    // Create profiler
    let profiler = Profiler::new(ProfilerConfig::default());

    // Create batch configuration
    let batch_config = BatchConfig::default();
    let processor = BatchProcessor::new(Arc::clone(pipeline), batch_config);

    // Create batch requests
    let requests: Vec<_> = (0..20)
        .map(|i| BatchRequest::new(format!("Test request number {}", i + 1), None))
        .collect();

    // Start profiling session
    let session = profiler.start_session("batch_processing").await;

    // Process batch
    let start = Instant::now();
    let results = processor.process(requests).await?;
    let batch_duration = start.elapsed();

    // End profiling session
    let report = profiler.end_session(session).await?;

    // Analyze results
    println!("Batch Processing Results:");
    println!("  Total requests: {}", results.len());
    println!(
        "  Successful: {}",
        results.iter().filter(|r| r.is_success()).count()
    );
    println!("  Total time: {:.2}s", batch_duration.as_secs_f64());
    println!();

    println!("Profiling Results:");
    println!(
        "  Session duration: {:.2}ms",
        report.session.duration_seconds * 1000.0
    );
    println!("  Stages profiled: {}", report.stage_breakdown.len());
    println!(
        "  Memory analysis: {}",
        if report.memory_analysis.is_some() {
            "available"
        } else {
            "not available"
        }
    );
    println!("  Bottlenecks detected: {}", report.bottlenecks.len());
    println!();

    // Show top time-consuming stages
    let mut stages: Vec<_> = report.stage_breakdown.iter().collect();
    stages.sort_by(|a, b| {
        b.total_duration_ms
            .partial_cmp(&a.total_duration_ms)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    println!("Top 5 Time-Consuming Stages:");
    for (idx, stage) in stages.iter().take(5).enumerate() {
        println!(
            "  {}. {}: {:.2}ms total ({:.1}% of total)",
            idx + 1,
            stage.stage_name,
            stage.total_duration_ms,
            stage.percentage_of_total
        );
    }

    Ok(())
}

/// Example 2: Compare different batch configurations
async fn compare_batch_configurations(pipeline: &Arc<VoirsPipeline>) -> Result<()> {
    println!("Comparing performance of different batch configurations...\n");

    let profiler = Profiler::new(ProfilerConfig::default());

    // Test requests
    let test_requests: Vec<_> = (0..30)
        .map(|i| BatchRequest::new(format!("Configuration test {}", i), None))
        .collect();

    // Configuration 1: Default
    println!("Testing default configuration...");
    let report1 = profile_batch_config(
        &profiler,
        pipeline,
        BatchConfig::default(),
        test_requests.clone(),
        "default_config",
    )
    .await?;

    // Configuration 2: High concurrency
    println!("Testing high concurrency configuration...");
    let report2 = profile_batch_config(
        &profiler,
        pipeline,
        BatchConfig {
            max_concurrency: num_cpus::get() * 2,
            scheduling_strategy: SchedulingStrategy::LoadBalanced,
            ..Default::default()
        },
        test_requests.clone(),
        "high_concurrency",
    )
    .await?;

    // Configuration 3: Priority scheduling
    println!("Testing priority scheduling configuration...");
    let report3 = profile_batch_config(
        &profiler,
        pipeline,
        BatchConfig {
            scheduling_strategy: SchedulingStrategy::PriorityBased,
            ..Default::default()
        },
        test_requests.clone(),
        "priority_scheduling",
    )
    .await?;

    println!();
    println!("{}", "=".repeat(70));
    println!("Configuration Comparison:\n");

    // Compare configurations using ProfileSession history
    let sessions = profiler.get_sessions().await;
    if sessions.len() >= 2 {
        let comparator = PerformanceComparator::new();

        println!("Default vs High Concurrency:");
        let comparison1 = comparator.compare(&sessions[0], &sessions[1]).await;
        println!(
            "  Performance change: {:.1}%",
            comparison1.overall_change_percent
        );
        println!();

        if sessions.len() >= 3 {
            println!("Default vs Priority Scheduling:");
            let comparison2 = comparator.compare(&sessions[0], &sessions[2]).await;
            println!(
                "  Performance change: {:.1}%",
                comparison2.overall_change_percent
            );
            println!();
        }
    }

    // Determine best configuration
    let durations = vec![
        (
            "Default",
            (report1.session.duration_seconds * 1000.0) as u64,
        ),
        (
            "High Concurrency",
            (report2.session.duration_seconds * 1000.0) as u64,
        ),
        (
            "Priority Scheduling",
            (report3.session.duration_seconds * 1000.0) as u64,
        ),
    ];

    let best = durations.iter().min_by_key(|x| x.1).unwrap();
    println!("Best configuration: {} ({} ms)", best.0, best.1);

    Ok(())
}

/// Example 3: Analyze memory usage during batch processing
async fn batch_memory_analysis(pipeline: &Arc<VoirsPipeline>) -> Result<()> {
    println!("Analyzing memory usage during batch processing...\n");

    let config = ProfilerConfig {
        enable_memory: true,
        enable_timing: true,
        sampling_interval_ms: 50,
        ..Default::default()
    };

    let profiler = Profiler::new(config);

    // Create a large batch to observe memory patterns
    let requests: Vec<_> = (0..50)
        .map(|i| {
            let text = match i % 3 {
                0 => format!("Short {}", i),
                1 => format!("Medium length text for request {}", i),
                _ => format!(
                    "This is a much longer text for request {} with more content to process",
                    i
                ),
            };
            BatchRequest::new(text, None)
        })
        .collect();

    let batch_config = BatchConfig {
        max_concurrency: 4,
        ..Default::default()
    };

    let processor = BatchProcessor::new(Arc::clone(pipeline), batch_config);

    // Start profiling
    let session = profiler.start_session("memory_analysis").await;

    // Process batch
    let results = processor.process(requests).await?;

    // End profiling
    let report = profiler.end_session(session).await?;

    println!("Batch Results:");
    println!("  Total requests: {}", results.len());
    println!(
        "  Successful: {}",
        results.iter().filter(|r| r.is_success()).count()
    );
    println!();

    println!("Memory Analysis:");
    if let Some(memory) = &report.memory_analysis {
        println!("  Peak memory: {:.2} MB", memory.peak_mb);
        println!("  Average memory: {:.2} MB", memory.average_mb);
        println!("  Memory growth: {:+.1}%", memory.growth_percent);

        if memory.growth_percent < 10.0 {
            println!("\n  Memory usage is well-controlled (growth < 10%)");
        } else if memory.growth_percent < 50.0 {
            println!("\n  Moderate memory growth ({:.1}%)", memory.growth_percent);
        } else {
            println!(
                "\n  Significant memory growth ({:.1}%) - investigate potential leaks",
                memory.growth_percent
            );
        }
    } else {
        println!("  No memory analysis data available.");
    }

    Ok(())
}

/// Example 4: Identify bottlenecks in batch processing
async fn identify_batch_bottlenecks(pipeline: &Arc<VoirsPipeline>) -> Result<()> {
    println!("Identifying bottlenecks in batch processing...\n");

    let config = ProfilerConfig {
        enable_bottleneck_detection: true,
        enable_timing: true,
        ..Default::default()
    };

    let profiler = Profiler::new(config);

    // Create batch with varying complexity
    let requests: Vec<_> = vec![
        ("Simple text", 1),
        ("A somewhat longer text that requires more processing", 5),
        ("An even more complex and lengthy text that will take considerably more time to process through all stages of the synthesis pipeline", 10),
    ]
    .into_iter()
    .flat_map(|(text, count)| {
        (0..count).map(move |i| BatchRequest::new(format!("{} #{}", text, i + 1), None))
    })
    .collect();

    let batch_config = BatchConfig::default();
    let processor = BatchProcessor::new(Arc::clone(pipeline), batch_config);

    // Profile batch processing
    let session = profiler.start_session("bottleneck_analysis").await;
    let results = processor.process(requests).await?;
    let report = profiler.end_session(session).await?;

    println!("Batch Results:");
    println!("  Total requests: {}", results.len());
    println!(
        "  Success rate: {:.1}%",
        (results.iter().filter(|r| r.is_success()).count() as f64 / results.len() as f64) * 100.0
    );
    println!();

    println!("Bottleneck Analysis:");
    if report.bottlenecks.is_empty() {
        println!("  No significant bottlenecks detected");
    } else {
        println!("  Found {} bottleneck(s):\n", report.bottlenecks.len());

        for (idx, bottleneck) in report.bottlenecks.iter().enumerate() {
            println!("  Bottleneck #{}:", idx + 1);
            println!("    Component: {}", bottleneck.component);
            println!("    Severity: {:?}", bottleneck.severity);
            println!("    Impact: {}", bottleneck.impact_description);
            if !bottleneck.recommendation.is_empty() {
                println!("    Recommendation: {}", bottleneck.recommendation);
            }
            println!();
        }
    }

    // Provide optimization recommendations
    println!("Optimization Recommendations:");

    let mut stages: Vec<_> = report.stage_breakdown.iter().collect();
    stages.sort_by(|a, b| {
        b.total_duration_ms
            .partial_cmp(&a.total_duration_ms)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    if let Some(slowest_stage) = stages.first() {
        println!(
            "  1. Focus on optimizing {} (takes {:.1}% of total time)",
            slowest_stage.stage_name, slowest_stage.percentage_of_total
        );
    }

    println!("  2. Consider increasing concurrency for I/O-bound operations");
    println!("  3. Enable caching for repeated content");

    Ok(())
}

/// Example 5: Production optimization analysis
async fn production_optimization(pipeline: &Arc<VoirsPipeline>) -> Result<()> {
    println!("Analyzing batch processing for production optimization...\n");

    let profiler = Profiler::new(ProfilerConfig::default());

    // Simulate production workload
    let requests: Vec<_> = (0..100)
        .map(|i| {
            let priority = if i < 10 {
                100
            } else if i < 30 {
                50
            } else {
                10
            };
            BatchRequest::new(format!("Production request {}", i), None).with_priority(priority)
        })
        .collect();

    let config = BatchConfig {
        max_concurrency: num_cpus::get(),
        scheduling_strategy: SchedulingStrategy::Adaptive,
        retry_failed: true,
        max_retries: 3,
        ..Default::default()
    };

    let processor = BatchProcessor::new(Arc::clone(pipeline), config);

    // Profile the batch
    let session = profiler.start_session("production_workload").await;
    let start = Instant::now();
    let results = processor.process(requests).await?;
    let total_time = start.elapsed();
    let report = profiler.end_session(session).await?;

    // Generate comprehensive report
    let generator = ReportGenerator::new(ProfilerConfig::default());
    let sessions = profiler.get_sessions().await;
    if let Some(last_session) = sessions.last() {
        let detailed_report = generator.generate(last_session, None).await?;
        println!("Production Workload Analysis:\n");
        println!("{}", detailed_report.summary());
    }

    // Calculate production metrics
    let stats = processor.statistics().await;

    println!("\nProduction Metrics:");
    println!("  Total requests: {}", results.len());
    println!("  Success rate: {:.2}%", stats.success_rate() * 100.0);
    println!("  Throughput: {:.2} requests/second", stats.throughput());
    println!(
        "  Average latency: {:.2}ms",
        stats.avg_time_per_request.as_secs_f64() * 1000.0
    );
    println!("  Total processing time: {:.2}s", total_time.as_secs_f64());

    let _ = report; // report used for session data

    // Provide production recommendations
    println!("\nProduction Deployment Recommendations:");

    let throughput = results.len() as f64 / total_time.as_secs_f64();

    println!("  - Expected throughput: {:.0} requests/second", throughput);
    println!("  - Recommended max concurrency: {}", num_cpus::get() * 2);
    println!("  - Recommended batch size: 50-100 requests");
    println!("  - Enable result caching for repeated content");
    println!("  - Use priority scheduling for time-sensitive requests");
    println!("  - Monitor memory usage for large batches");

    if stats.success_rate() > 0.99 {
        println!(
            "\nSystem shows excellent reliability ({:.2}% success rate)",
            stats.success_rate() * 100.0
        );
    } else {
        println!(
            "\nSuccess rate is {:.2}% - investigate errors",
            stats.success_rate() * 100.0
        );
    }

    Ok(())
}

/// Helper function to profile a batch configuration
async fn profile_batch_config(
    profiler: &Profiler,
    pipeline: &Arc<VoirsPipeline>,
    config: BatchConfig,
    requests: Vec<BatchRequest>,
    session_name: &str,
) -> Result<PerformanceReport> {
    let processor = BatchProcessor::new(Arc::clone(pipeline), config);

    let session = profiler.start_session(session_name).await;
    let _ = processor.process(requests).await?;
    let report = profiler.end_session(session).await?;

    println!(
        "  Duration: {:.2}ms",
        report.session.duration_seconds * 1000.0
    );

    Ok(report)
}

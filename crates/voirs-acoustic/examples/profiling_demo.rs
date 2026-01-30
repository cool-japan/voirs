//! Performance Profiling Demo
//!
//! This example demonstrates the advanced performance profiling and tracing system
//! for the voirs-acoustic crate. It shows how to:
//! - Create and configure a profiler
//! - Use span guards for automatic timing
//! - Create hierarchical (nested) spans
//! - Add metadata tags to spans
//! - Generate and export profiling reports
//! - Analyze performance bottlenecks

use std::thread;
use std::time::Duration;
use voirs_acoustic::profiling::{PerformanceProfiler, ProfilingConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== VoiRS Acoustic Performance Profiling Demo ===\n");

    // ===== 1. Basic Profiling =====
    println!("1. Basic Profiling Example");
    println!("{}", "-".repeat(50));

    let profiler = PerformanceProfiler::new(ProfilingConfig::development());

    // Simple span - automatically timed with RAII guard
    {
        let _span = profiler.begin_span("simple_operation")?;
        thread::sleep(Duration::from_millis(10));
        // Span is automatically completed when it goes out of scope
    }

    println!("✓ Completed simple operation");

    // ===== 2. Nested Spans (Hierarchical Tracing) =====
    println!("\n2. Nested Spans Example");
    println!("{}", "-".repeat(50));

    {
        let parent = profiler.begin_span("parent_operation")?;
        thread::sleep(Duration::from_millis(5));

        {
            let _child1 =
                profiler.begin_span_with_parent("child_operation_1", Some(parent.id()))?;
            thread::sleep(Duration::from_millis(10));
        }

        {
            let _child2 =
                profiler.begin_span_with_parent("child_operation_2", Some(parent.id()))?;
            thread::sleep(Duration::from_millis(8));
        }

        println!("✓ Completed parent operation with 2 children");
    }

    // ===== 3. Tagged Spans (Metadata) =====
    println!("\n3. Tagged Spans Example");
    println!("{}", "-".repeat(50));

    {
        let mut span = profiler.begin_span("tagged_operation")?;
        span.add_tag("user_id", "12345");
        span.add_tag("request_type", "synthesis");
        span.add_tag("model", "vits");
        thread::sleep(Duration::from_millis(15));
        println!("✓ Completed operation with metadata tags");
    }

    // ===== 4. Simulating Real TTS Pipeline =====
    println!("\n4. Simulated TTS Pipeline");
    println!("{}", "-".repeat(50));

    for i in 1..=3 {
        let mut pipeline_span = profiler.begin_span("tts_pipeline")?;
        pipeline_span.add_tag("iteration", i.to_string());

        // G2P phase
        {
            let _g2p =
                profiler.begin_span_with_parent("g2p_conversion", Some(pipeline_span.id()))?;
            thread::sleep(Duration::from_millis(3));
        }

        // Acoustic modeling phase
        {
            let _acoustic =
                profiler.begin_span_with_parent("acoustic_inference", Some(pipeline_span.id()))?;
            thread::sleep(Duration::from_millis(20));
        }

        // Vocoding phase
        {
            let _vocoder =
                profiler.begin_span_with_parent("vocoder_inference", Some(pipeline_span.id()))?;
            thread::sleep(Duration::from_millis(15));
        }

        println!("✓ Completed TTS pipeline iteration {}", i);
    }

    // ===== 5. Generate and Analyze Report =====
    println!("\n5. Performance Report");
    println!("{}", "=".repeat(50));

    let report = profiler.generate_report()?;

    // Text report
    println!("\n{}", report.to_text());

    // Analysis: Slowest operations
    println!("\n6. Performance Analysis");
    println!("{}", "=".repeat(50));
    println!("\nTop 3 Slowest Operations:");
    for (i, profile) in report.slowest_operations(3).iter().enumerate() {
        println!(
            "  {}. {} - Avg: {:?}, P95: {:?}",
            i + 1,
            profile.operation,
            profile.timing.avg_duration,
            profile.timing.p95_duration
        );
    }

    // Most memory-intensive operations
    println!("\nTop 3 Most Memory-Intensive Operations:");
    for (i, profile) in report.most_memory_intensive(3).iter().enumerate() {
        println!(
            "  {}. {} - Peak: {} KB",
            i + 1,
            profile.operation,
            profile.memory.peak_memory_bytes / 1024
        );
    }

    // ===== 6. JSON Export =====
    println!("\n7. JSON Export");
    println!("{}", "=".repeat(50));

    let json_report = report.to_json()?;
    println!("✓ JSON report generated ({} bytes)", json_report.len());
    println!("\nFirst 200 characters of JSON report:");
    println!("{}...", &json_report.chars().take(200).collect::<String>());

    // ===== 7. Individual Operation Profiles =====
    println!("\n8. Individual Operation Profiles");
    println!("{}", "=".repeat(50));

    if let Some(profile) = profiler.get_operation_profile("tts_pipeline")? {
        println!("\nTTS Pipeline Profile:");
        println!("  Total Invocations: {}", profile.invocation_count);
        println!(
            "  Success Rate: {:.1}%",
            (profile.invocation_count - profile.failure_count) as f64
                / profile.invocation_count as f64
                * 100.0
        );
        println!("  Average Duration: {:?}", profile.timing.avg_duration);
        println!("  Min Duration: {:?}", profile.timing.min_duration);
        println!("  Max Duration: {:?}", profile.timing.max_duration);
        println!("  P95 Duration: {:?}", profile.timing.p95_duration);
        println!("  P99 Duration: {:?}", profile.timing.p99_duration);
    }

    // ===== 8. Configuration Presets =====
    println!("\n9. Configuration Presets Demo");
    println!("{}", "=".repeat(50));

    println!("\nDevelopment Config:");
    let dev_config = ProfilingConfig::development();
    println!("  - Tracing enabled: {}", dev_config.enable_tracing);
    println!(
        "  - Memory sampling rate: {:.0}%",
        dev_config.memory_sample_rate * 100.0
    );
    println!("  - Max span history: {}", dev_config.max_span_history);

    println!("\nProduction Config:");
    let prod_config = ProfilingConfig::production();
    println!("  - Tracing enabled: {}", prod_config.enable_tracing);
    println!(
        "  - Memory sampling rate: {:.0}%",
        prod_config.memory_sample_rate * 100.0
    );
    println!("  - Max span history: {}", prod_config.max_span_history);

    println!("\nMinimal Config (for benchmarking):");
    let minimal_config = ProfilingConfig::minimal();
    println!("  - Tracing enabled: {}", minimal_config.enable_tracing);
    println!(
        "  - Memory sampling rate: {:.0}%",
        minimal_config.memory_sample_rate * 100.0
    );
    println!("  - Max span history: {}", minimal_config.max_span_history);

    // ===== 9. Clear Profiling Data =====
    println!("\n10. Clear Profiling Data");
    println!("{}", "=".repeat(50));

    let span_count_before = profiler.get_spans()?.len();
    println!("Spans before clear: {}", span_count_before);

    profiler.clear()?;

    let span_count_after = profiler.get_spans()?.len();
    println!("Spans after clear: {}", span_count_after);

    println!("\n=== Demo Complete ===");
    println!("\nKey Takeaways:");
    println!("  1. Use span guards for automatic timing (RAII pattern)");
    println!("  2. Create hierarchical spans with parent-child relationships");
    println!("  3. Add metadata tags for detailed analysis");
    println!("  4. Generate reports for performance analysis");
    println!("  5. Export to JSON for external tools");
    println!("  6. Use appropriate config presets for different environments");
    println!("\nFor production use:");
    println!("  - Use ProfilingConfig::production() to minimize overhead");
    println!("  - Focus profiling on critical paths");
    println!("  - Export reports periodically for analysis");
    println!("  - Monitor P95/P99 latencies for SLA compliance");

    Ok(())
}

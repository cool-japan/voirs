//! Performance profiling overhead benchmarks.
//!
//! This benchmark suite measures the performance overhead of the profiling system:
//! - Profiler initialization overhead
//! - Session start/end overhead
//! - Stage timing overhead
//! - Memory profiling overhead
//! - Bottleneck detection overhead
//! - Report generation overhead
//! - Comparison overhead
//!
//! ## Usage
//!
//! ```bash
//! cargo bench --bench profiling_benchmarks
//! ```
//!
//! ## Expected Results
//!
//! - Profiler initialization: <1ms
//! - Session start/end: <100μs each
//! - Stage timing: <10μs overhead per stage
//! - Memory snapshot: <50μs per snapshot
//! - Bottleneck detection: <5ms per session
//! - Report generation: <20ms per report
//! - Performance comparison: <10ms per comparison

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::sync::Arc;
use std::time::Duration;
use tokio::runtime::Runtime;
use voirs_sdk::profiling::{
    BottleneckDetector, MemoryProfiler, PerformanceComparator, PipelineProfiler, PipelineStage,
    Profiler, ProfilerConfig, ReportGenerator,
};
use voirs_sdk::VoirsPipelineBuilder;

/// Create a test pipeline for benchmarking
fn create_test_pipeline() -> Arc<voirs_sdk::VoirsPipeline> {
    let rt = Runtime::new().unwrap();
    rt.block_on(async {
        let pipeline = VoirsPipelineBuilder::new()
            .with_test_mode(true)
            .build()
            .await
            .unwrap();
        Arc::new(pipeline)
    })
}

/// Benchmark profiler initialization
fn bench_profiler_init(c: &mut Criterion) {
    let mut group = c.benchmark_group("profiling_init");

    group.bench_function("profiler_creation", |b| {
        b.iter(|| {
            black_box(Profiler::new(ProfilerConfig::default()));
        });
    });

    group.bench_function("profiler_with_config", |b| {
        b.iter(|| {
            let config = ProfilerConfig {
                enable_timing: true,
                enable_memory: true,
                enable_bottleneck_detection: true,
                max_history_size: 100,
                sampling_interval_ms: 100,
                auto_generate_reports: true,
                report_output_dir: Some("/tmp/profiling".into()),
                enable_baseline_comparison: false,
                regression_threshold_percent: 10.0,
            };
            black_box(Profiler::new(config));
        });
    });

    group.finish();
}

/// Benchmark session management overhead
fn bench_session_management(c: &mut Criterion) {
    let mut group = c.benchmark_group("profiling_sessions");

    let rt = Runtime::new().unwrap();

    group.bench_function("session_start_end", |b| {
        let profiler = Profiler::new(ProfilerConfig::default());

        b.to_async(&rt).iter(|| async {
            let session = profiler.start_session("test").await;
            let _ = black_box(profiler.end_session(session).await);
        });
    });

    group.bench_function("multiple_sessions", |b| {
        let profiler = Profiler::new(ProfilerConfig::default());

        b.to_async(&rt).iter(|| async {
            for i in 0..10 {
                let session = profiler.start_session(&format!("test-{}", i)).await;
                let _ = black_box(profiler.end_session(session).await);
            }
        });
    });

    group.finish();
}

/// Benchmark pipeline stage timing overhead
fn bench_stage_timing(c: &mut Criterion) {
    let mut group = c.benchmark_group("profiling_stage_timing");

    let rt = Runtime::new().unwrap();

    // Measure overhead of single stage timing
    group.bench_function("single_stage", |b| {
        let profiler = PipelineProfiler::new();

        b.to_async(&rt).iter(|| async {
            let timing_id = profiler.start_stage(PipelineStage::G2pConversion).await;
            // Simulate work
            tokio::time::sleep(Duration::from_micros(100)).await;
            black_box(profiler.end_stage(&timing_id, 100, 200).await);
        });
    });

    // Measure overhead of complete pipeline timing
    group.bench_function("full_pipeline", |b| {
        let profiler = PipelineProfiler::new();

        b.to_async(&rt).iter(|| async {
            let stages = [
                PipelineStage::TextPreprocessing,
                PipelineStage::G2pConversion,
                PipelineStage::AcousticModel,
                PipelineStage::Vocoder,
                PipelineStage::PostProcessing,
                PipelineStage::AudioEncoding,
            ];

            for stage in &stages {
                let timing_id = profiler.start_stage(*stage).await;
                // Simulate work
                tokio::time::sleep(Duration::from_micros(50)).await;
                profiler.end_stage(&timing_id, 100, 200).await;
            }

            black_box(profiler.get_all_metrics().await);
        });
    });

    // Measure overhead with concurrent stage timing
    group.bench_function("concurrent_stages", |b| {
        let profiler = Arc::new(PipelineProfiler::new());

        b.to_async(&rt).iter(|| async {
            let mut handles = vec![];

            for i in 0..5 {
                let prof = profiler.clone();
                let handle = tokio::spawn(async move {
                    let stage = match i % 3 {
                        0 => PipelineStage::G2pConversion,
                        1 => PipelineStage::AcousticModel,
                        _ => PipelineStage::Vocoder,
                    };
                    let timing_id = prof.start_stage(stage).await;
                    tokio::time::sleep(Duration::from_micros(100)).await;
                    prof.end_stage(&timing_id, 100, 200).await;
                });
                handles.push(handle);
            }

            for handle in handles {
                handle.await.unwrap();
            }

            black_box(profiler.get_all_metrics().await);
        });
    });

    group.finish();
}

/// Benchmark memory profiling overhead
fn bench_memory_profiling(c: &mut Criterion) {
    let mut group = c.benchmark_group("profiling_memory");

    let rt = Runtime::new().unwrap();

    group.bench_function("memory_snapshot", |b| {
        let profiler = MemoryProfiler::new();

        b.to_async(&rt).iter(|| async {
            black_box(profiler.take_snapshot(Some("test".to_string())).await);
        });
    });

    group.bench_function("multiple_snapshots", |b| {
        let profiler = MemoryProfiler::new();

        b.to_async(&rt).iter(|| async {
            for i in 0..10 {
                black_box(
                    profiler
                        .take_snapshot(Some(format!("snapshot-{}", i)))
                        .await,
                );
            }
        });
    });

    group.bench_function("snapshot_with_metrics", |b| {
        let profiler = MemoryProfiler::new();

        b.to_async(&rt).iter(|| async {
            for i in 0..5 {
                profiler
                    .take_snapshot(Some(format!("snapshot-{}", i)))
                    .await;
            }
            black_box(profiler.get_snapshots().await);
        });
    });

    group.finish();
}

/// Benchmark bottleneck detection overhead
fn bench_bottleneck_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("profiling_bottleneck");

    let rt = Runtime::new().unwrap();

    // Create a sample session with metrics for testing
    group.bench_function("detect_bottlenecks", |b| {
        let detector = BottleneckDetector::new(10.0);

        b.to_async(&rt).iter(|| async {
            // Create profiler and run a sample session
            let profiler = Profiler::new(ProfilerConfig::default());
            let session = profiler.start_session("test").await;

            // Simulate some pipeline stages
            let pipeline_prof = PipelineProfiler::new();
            for stage in &[
                PipelineStage::G2pConversion,
                PipelineStage::AcousticModel,
                PipelineStage::Vocoder,
            ] {
                let timing_id = pipeline_prof.start_stage(*stage).await;
                tokio::time::sleep(Duration::from_micros(100)).await;
                pipeline_prof.end_stage(&timing_id, 100, 200).await;
            }

            // End session and retrieve it from history
            profiler.end_session(session).await.unwrap();
            let sessions = profiler.get_sessions().await;
            let stored_session = sessions.last().unwrap();

            black_box(detector.detect(stored_session).await);
        });
    });

    group.finish();
}

/// Benchmark report generation overhead
fn bench_report_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("profiling_reports");

    let rt = Runtime::new().unwrap();

    // Baseline: generate report from single session
    group.bench_function("single_session_report", |b| {
        b.to_async(&rt).iter(|| async {
            // Create and run a profiling session
            let profiler = Profiler::new(ProfilerConfig::default());
            let session = profiler.start_session("test").await;

            // Simulate pipeline execution
            let pipeline_prof = PipelineProfiler::new();
            for stage in &[
                PipelineStage::G2pConversion,
                PipelineStage::AcousticModel,
                PipelineStage::Vocoder,
            ] {
                let timing_id = pipeline_prof.start_stage(*stage).await;
                tokio::time::sleep(Duration::from_micros(100)).await;
                pipeline_prof.end_stage(&timing_id, 100, 200).await;
            }

            // End session and retrieve it from history
            profiler.end_session(session).await.unwrap();
            let sessions = profiler.get_sessions().await;
            let stored_session = sessions.last().unwrap();

            // Generate report
            let generator = ReportGenerator::new(ProfilerConfig::default());
            black_box(generator.generate(stored_session, None).await.unwrap());
        });
    });

    // Aggregate report from multiple sessions
    let session_counts = [5, 10, 20];
    for &count in &session_counts {
        group.bench_with_input(
            BenchmarkId::new("aggregate_report", count),
            &count,
            |b, &count| {
                b.to_async(&rt).iter(|| async {
                    let profiler = Profiler::new(ProfilerConfig {
                        max_history_size: count,
                        ..Default::default()
                    });

                    // Create multiple sessions
                    for i in 0..count {
                        let session = profiler.start_session(&format!("test-{}", i)).await;

                        // Simulate pipeline execution
                        let pipeline_prof = PipelineProfiler::new();
                        for stage in &[
                            PipelineStage::G2pConversion,
                            PipelineStage::AcousticModel,
                            PipelineStage::Vocoder,
                        ] {
                            let timing_id = pipeline_prof.start_stage(*stage).await;
                            tokio::time::sleep(Duration::from_micros(50)).await;
                            pipeline_prof.end_stage(&timing_id, 100, 200).await;
                        }

                        profiler.end_session(session).await.unwrap();
                    }

                    // Get all sessions from history
                    let sessions = profiler.get_sessions().await;

                    // Generate aggregate report
                    let generator = ReportGenerator::new(ProfilerConfig::default());
                    black_box(generator.generate_aggregate(&sessions).await.unwrap());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark performance comparison overhead
fn bench_performance_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("profiling_comparison");

    let rt = Runtime::new().unwrap();

    group.bench_function("compare_sessions", |b| {
        b.to_async(&rt).iter(|| async {
            // Create two profiling sessions
            let profiler = Profiler::new(ProfilerConfig::default());

            // Baseline session
            let session1 = profiler.start_session("baseline").await;
            let pipeline_prof = PipelineProfiler::new();
            for stage in &[
                PipelineStage::G2pConversion,
                PipelineStage::AcousticModel,
                PipelineStage::Vocoder,
            ] {
                let timing_id = pipeline_prof.start_stage(*stage).await;
                tokio::time::sleep(Duration::from_micros(100)).await;
                pipeline_prof.end_stage(&timing_id, 100, 200).await;
            }
            profiler.end_session(session1).await.unwrap();

            // Comparison session (slightly different timing)
            let session2 = profiler.start_session("current").await;
            let pipeline_prof2 = PipelineProfiler::new();
            for stage in &[
                PipelineStage::G2pConversion,
                PipelineStage::AcousticModel,
                PipelineStage::Vocoder,
            ] {
                let timing_id = pipeline_prof2.start_stage(*stage).await;
                tokio::time::sleep(Duration::from_micros(120)).await;
                pipeline_prof2.end_stage(&timing_id, 100, 200).await;
            }
            profiler.end_session(session2).await.unwrap();

            // Get sessions from history
            let sessions = profiler.get_sessions().await;
            let stored_session1 = &sessions[sessions.len() - 2];
            let stored_session2 = &sessions[sessions.len() - 1];

            // Compare sessions
            let comparator = PerformanceComparator::new();
            black_box(comparator.compare(stored_session1, stored_session2).await);
        });
    });

    group.finish();
}

/// Benchmark profiling overhead on actual synthesis
fn bench_synthesis_with_profiling(c: &mut Criterion) {
    let mut group = c.benchmark_group("profiling_synthesis_overhead");

    let rt = Runtime::new().unwrap();
    let pipeline = create_test_pipeline();

    // Baseline: synthesis without profiling
    group.bench_function("without_profiling", |b| {
        b.to_async(&rt).iter(|| async {
            black_box(pipeline.synthesize("Hello world").await.unwrap());
        });
    });

    // With minimal profiling (timing only)
    group.bench_function("with_timing", |b| {
        let profiler = Profiler::new(ProfilerConfig {
            enable_timing: true,
            enable_memory: false,
            enable_bottleneck_detection: false,
            ..Default::default()
        });

        b.to_async(&rt).iter(|| async {
            let session = profiler.start_session("test").await;
            black_box(pipeline.synthesize("Hello world").await.unwrap());
            let _ = profiler.end_session(session).await;
        });
    });

    // With full profiling (timing + memory + bottleneck detection)
    group.bench_function("with_full_profiling", |b| {
        let profiler = Profiler::new(ProfilerConfig {
            enable_timing: true,
            enable_memory: true,
            enable_bottleneck_detection: true,
            ..Default::default()
        });

        b.to_async(&rt).iter(|| async {
            let session = profiler.start_session("test").await;
            black_box(pipeline.synthesize("Hello world").await.unwrap());
            profiler.end_session(session).await.unwrap();

            // Get session from history
            let sessions = profiler.get_sessions().await;
            let stored_session = sessions.last().unwrap();

            // Run bottleneck detection
            let detector = BottleneckDetector::new(10.0);
            detector.detect(stored_session).await;
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_profiler_init,
    bench_session_management,
    bench_stage_timing,
    bench_memory_profiling,
    bench_bottleneck_detection,
    bench_report_generation,
    bench_performance_comparison,
    bench_synthesis_with_profiling,
);

criterion_main!(benches);

//! Benchmarks for Performance Profiling System
//!
//! This benchmark suite measures the overhead of the profiling system to ensure
//! it remains minimal and suitable for production use.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Duration;
use voirs_acoustic::profiling::{PerformanceProfiler, ProfilingConfig};

/// Benchmark profiling overhead with different configurations
fn bench_profiling_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("profiling_overhead");

    // Benchmark: No profiling (baseline)
    group.bench_function("no_profiling_baseline", |b| {
        b.iter(|| {
            // Simulate simple operation
            let mut sum = 0;
            for i in 0..100 {
                sum += black_box(i);
            }
            black_box(sum);
        });
    });

    // Benchmark: Minimal config (disabled)
    group.bench_function("minimal_config", |b| {
        let profiler = PerformanceProfiler::new(ProfilingConfig::minimal());
        b.iter(|| {
            let _span = profiler.begin_span("test_op").unwrap();
            let mut sum = 0;
            for i in 0..100 {
                sum += black_box(i);
            }
            black_box(sum);
        });
    });

    // Benchmark: Production config (optimized)
    group.bench_function("production_config", |b| {
        let profiler = PerformanceProfiler::new(ProfilingConfig::production());
        b.iter(|| {
            let _span = profiler.begin_span("test_op").unwrap();
            let mut sum = 0;
            for i in 0..100 {
                sum += black_box(i);
            }
            black_box(sum);
        });
    });

    // Benchmark: Development config (full tracking)
    group.bench_function("development_config", |b| {
        let profiler = PerformanceProfiler::new(ProfilingConfig::development());
        b.iter(|| {
            let _span = profiler.begin_span("test_op").unwrap();
            let mut sum = 0;
            for i in 0..100 {
                sum += black_box(i);
            }
            black_box(sum);
        });
    });

    group.finish();
}

/// Benchmark nested span creation
fn bench_nested_spans(c: &mut Criterion) {
    let mut group = c.benchmark_group("nested_spans");

    for depth in [1, 3, 5, 10].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(depth), depth, |b, &depth| {
            let profiler = PerformanceProfiler::new(ProfilingConfig::production());
            b.iter(|| {
                create_nested_spans(&profiler, depth, None).unwrap();
            });
        });
    }

    group.finish();
}

fn create_nested_spans(
    profiler: &PerformanceProfiler,
    depth: usize,
    parent_id: Option<voirs_acoustic::profiling::SpanId>,
) -> Result<(), Box<dyn std::error::Error>> {
    if depth == 0 {
        return Ok(());
    }

    let span = if let Some(pid) = parent_id {
        profiler.begin_span_with_parent("nested_op", Some(pid))?
    } else {
        profiler.begin_span("root_op")?
    };

    create_nested_spans(profiler, depth - 1, Some(span.id()))?;
    Ok(())
}

/// Benchmark span tagging
fn bench_span_tagging(c: &mut Criterion) {
    let mut group = c.benchmark_group("span_tagging");

    group.bench_function("no_tags", |b| {
        let profiler = PerformanceProfiler::new(ProfilingConfig::production());
        b.iter(|| {
            let _span = profiler.begin_span("test_op").unwrap();
            black_box(42);
        });
    });

    group.bench_function("single_tag", |b| {
        let profiler = PerformanceProfiler::new(ProfilingConfig::production());
        b.iter(|| {
            let mut span = profiler.begin_span("test_op").unwrap();
            span.add_tag("key", "value");
            black_box(42);
        });
    });

    group.bench_function("five_tags", |b| {
        let profiler = PerformanceProfiler::new(ProfilingConfig::production());
        b.iter(|| {
            let mut span = profiler.begin_span("test_op").unwrap();
            span.add_tag("key1", "value1");
            span.add_tag("key2", "value2");
            span.add_tag("key3", "value3");
            span.add_tag("key4", "value4");
            span.add_tag("key5", "value5");
            black_box(42);
        });
    });

    group.bench_function("ten_tags", |b| {
        let profiler = PerformanceProfiler::new(ProfilingConfig::production());
        b.iter(|| {
            let mut span = profiler.begin_span("test_op").unwrap();
            for i in 0..10 {
                span.add_tag(format!("key{}", i), format!("value{}", i));
            }
            black_box(42);
        });
    });

    group.finish();
}

/// Benchmark report generation
fn bench_report_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("report_generation");

    // Setup: Create profiler with some history
    let setup_profiler = |operations: usize| {
        let profiler = PerformanceProfiler::new(ProfilingConfig::development());
        for i in 0..operations {
            let _span = profiler.begin_span(format!("op_{}", i % 10)).unwrap();
            std::thread::sleep(Duration::from_micros(100));
        }
        profiler
    };

    group.bench_function("generate_report_100_ops", |b| {
        let profiler = setup_profiler(100);
        b.iter(|| {
            let report = profiler.generate_report().unwrap();
            black_box(report);
        });
    });

    group.bench_function("generate_report_1000_ops", |b| {
        let profiler = setup_profiler(1000);
        b.iter(|| {
            let report = profiler.generate_report().unwrap();
            black_box(report);
        });
    });

    group.bench_function("json_export_100_ops", |b| {
        let profiler = setup_profiler(100);
        let report = profiler.generate_report().unwrap();
        b.iter(|| {
            let json = report.to_json().unwrap();
            black_box(json);
        });
    });

    group.bench_function("text_export_100_ops", |b| {
        let profiler = setup_profiler(100);
        let report = profiler.generate_report().unwrap();
        b.iter(|| {
            let text = report.to_text();
            black_box(text);
        });
    });

    group.finish();
}

/// Benchmark concurrent profiling
fn bench_concurrent_profiling(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_profiling");

    group.bench_function("sequential_10_spans", |b| {
        let profiler = PerformanceProfiler::new(ProfilingConfig::production());
        b.iter(|| {
            for _ in 0..10 {
                let _span = profiler.begin_span("test_op").unwrap();
                black_box(42);
            }
        });
    });

    group.bench_function("concurrent_10_threads", |b| {
        let profiler = PerformanceProfiler::new(ProfilingConfig::production());
        b.iter(|| {
            let mut handles = vec![];
            for _ in 0..10 {
                let profiler_clone = profiler.clone();
                let handle = std::thread::spawn(move || {
                    let _span = profiler_clone.begin_span("test_op").unwrap();
                    black_box(42);
                });
                handles.push(handle);
            }
            for handle in handles {
                handle.join().unwrap();
            }
        });
    });

    group.finish();
}

/// Benchmark operation profile lookup
fn bench_profile_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("profile_lookup");

    // Setup profiler with multiple operations
    let setup_profiler = |num_operations: usize| {
        let profiler = PerformanceProfiler::new(ProfilingConfig::development());
        for i in 0..num_operations {
            let _span = profiler.begin_span(format!("operation_{}", i)).unwrap();
            std::thread::sleep(Duration::from_micros(10));
        }
        profiler
    };

    group.bench_function("lookup_10_ops", |b| {
        let profiler = setup_profiler(10);
        b.iter(|| {
            let profile = profiler.get_operation_profile("operation_5").unwrap();
            black_box(profile);
        });
    });

    group.bench_function("lookup_100_ops", |b| {
        let profiler = setup_profiler(100);
        b.iter(|| {
            let profile = profiler.get_operation_profile("operation_50").unwrap();
            black_box(profile);
        });
    });

    group.bench_function("get_all_profiles_10_ops", |b| {
        let profiler = setup_profiler(10);
        b.iter(|| {
            let profiles = profiler.get_all_profiles().unwrap();
            black_box(profiles);
        });
    });

    group.bench_function("get_all_profiles_100_ops", |b| {
        let profiler = setup_profiler(100);
        b.iter(|| {
            let profiles = profiler.get_all_profiles().unwrap();
            black_box(profiles);
        });
    });

    group.finish();
}

/// Benchmark memory tracking overhead
fn bench_memory_tracking(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_tracking");

    group.bench_function("with_memory_tracking", |b| {
        let mut config = ProfilingConfig::development();
        config.enable_memory_tracking = true;
        config.memory_sample_rate = 1.0; // Track all
        let profiler = PerformanceProfiler::new(config);

        b.iter(|| {
            let _span = profiler.begin_span("test_op").unwrap();
            let mut vec = Vec::with_capacity(1000);
            for i in 0..1000 {
                vec.push(black_box(i));
            }
            black_box(vec);
        });
    });

    group.bench_function("without_memory_tracking", |b| {
        let mut config = ProfilingConfig::development();
        config.enable_memory_tracking = false;
        let profiler = PerformanceProfiler::new(config);

        b.iter(|| {
            let _span = profiler.begin_span("test_op").unwrap();
            let mut vec = Vec::with_capacity(1000);
            for i in 0..1000 {
                vec.push(black_box(i));
            }
            black_box(vec);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_profiling_overhead,
    bench_nested_spans,
    bench_span_tagging,
    bench_report_generation,
    bench_concurrent_profiling,
    bench_profile_lookup,
    bench_memory_tracking,
);
criterion_main!(benches);

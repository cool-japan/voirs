//! Performance benchmarks for production features
//!
//! Benchmarks for caching, monitoring, and warmup systems

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::sync::Arc;
use std::time::Duration;
use voirs_acoustic::{
    production_monitoring::ProductionMonitor,
    synthesis_cache::{EvictionPolicy, SynthesisCache, SynthesisCacheConfig, SynthesisCacheKey},
    MelSpectrogram,
};

fn bench_cache_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_operations");

    // Benchmark cache insertion
    group.bench_function("cache_insert", |b| {
        let config = SynthesisCacheConfig {
            eviction_policy: EvictionPolicy::LRU,
            ..Default::default()
        };
        let cache = SynthesisCache::new(config);
        let mel = MelSpectrogram::new(vec![vec![1.0; 100]; 80], 22050, 256);

        b.iter(|| {
            let key = SynthesisCacheKey::new(
                &format!("test_{}", fastrand::u64(..)),
                Some(0),
                1.0,
                0.0,
                1.0,
            );
            black_box(cache.insert(key, mel.clone()))
        });
    });

    // Benchmark cache retrieval (hit)
    group.bench_function("cache_get_hit", |b| {
        let config = SynthesisCacheConfig {
            eviction_policy: EvictionPolicy::LRU,
            ..Default::default()
        };
        let cache = SynthesisCache::new(config);
        let mel = MelSpectrogram::new(vec![vec![1.0; 100]; 80], 22050, 256);
        let key = SynthesisCacheKey::new("test", Some(0), 1.0, 0.0, 1.0);
        cache.insert(key.clone(), mel).unwrap();

        b.iter(|| black_box(cache.get(&key)));
    });

    // Benchmark cache retrieval (miss)
    group.bench_function("cache_get_miss", |b| {
        let config = SynthesisCacheConfig {
            eviction_policy: EvictionPolicy::LRU,
            ..Default::default()
        };
        let cache = SynthesisCache::new(config);

        b.iter(|| {
            let key = SynthesisCacheKey::new(
                &format!("nonexistent_{}", fastrand::u64(..)),
                Some(0),
                1.0,
                0.0,
                1.0,
            );
            black_box(cache.get(&key))
        });
    });

    // Benchmark cache statistics
    group.bench_function("cache_statistics", |b| {
        let config = SynthesisCacheConfig {
            eviction_policy: EvictionPolicy::LRU,
            ..Default::default()
        };
        let cache = SynthesisCache::new(config);
        b.iter(|| black_box(cache.statistics()));
    });

    group.finish();
}

fn bench_eviction_policies(c: &mut Criterion) {
    let mut group = c.benchmark_group("eviction_policies");

    for policy in &[
        EvictionPolicy::LRU,
        EvictionPolicy::LFU,
        EvictionPolicy::TTL,
        EvictionPolicy::Hybrid,
    ] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:?}", policy)),
            policy,
            |b, &policy| {
                let config = SynthesisCacheConfig {
                    max_entries: 100,
                    max_size_bytes: 10 * 1024 * 1024,
                    ttl: Duration::from_secs(60),
                    eviction_policy: policy,
                    enable_stats: true,
                    preload_enabled: false,
                };
                let cache = SynthesisCache::new(config);
                let mel = MelSpectrogram::new(vec![vec![1.0; 50]; 80], 22050, 256);

                b.iter(|| {
                    for i in 0..150 {
                        let key =
                            SynthesisCacheKey::new(&format!("test_{}", i), Some(0), 1.0, 0.0, 1.0);
                        let _ = cache.insert(key, mel.clone());
                    }
                });
            },
        );
    }

    group.finish();
}

fn bench_monitoring_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("monitoring_operations");

    // Benchmark recording synthesis
    group.bench_function("record_synthesis", |b| {
        let monitor = ProductionMonitor::new();
        b.iter(|| monitor.record_synthesis(Duration::from_millis(100), true, 50));
    });

    // Benchmark health update
    group.bench_function("update_health", |b| {
        let monitor = ProductionMonitor::new();
        b.iter(|| monitor.update_health("test_component", true));
    });

    // Benchmark report generation
    group.bench_function("generate_report", |b| {
        let monitor = ProductionMonitor::new();
        // Pre-populate with some data
        for i in 0..100 {
            monitor.record_synthesis(Duration::from_millis(50 + i), true, 20);
        }

        b.iter(|| black_box(monitor.generate_report()));
    });

    group.finish();
}

fn bench_concurrent_monitoring(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_monitoring");

    group.bench_function("concurrent_recording", |b| {
        b.iter(|| {
            let monitor = Arc::new(ProductionMonitor::new());
            let mut handles = vec![];

            for _ in 0..4 {
                let monitor_clone = Arc::clone(&monitor);
                let handle = std::thread::spawn(move || {
                    for _ in 0..25 {
                        monitor_clone.record_synthesis(Duration::from_millis(50), true, 20);
                    }
                });
                handles.push(handle);
            }

            for handle in handles {
                handle.join().unwrap();
            }

            black_box(monitor)
        });
    });

    group.finish();
}

fn bench_cache_key_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_key_creation");

    group.bench_function("create_cache_key", |b| {
        b.iter(|| {
            black_box(SynthesisCacheKey::new(
                "test_phoneme_sequence",
                Some(0),
                1.0,
                0.0,
                1.0,
            ))
        });
    });

    group.bench_function("create_cache_key_with_quantization", |b| {
        b.iter(|| {
            black_box(SynthesisCacheKey::new(
                "test_phoneme_sequence",
                Some(0),
                1.234,
                0.567,
                1.890,
            ))
        });
    });

    group.finish();
}

fn bench_cache_with_different_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_sizes");

    for size in &[10, 50, 100, 500, 1000] {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let config = SynthesisCacheConfig {
                max_entries: size,
                eviction_policy: EvictionPolicy::LRU,
                ..Default::default()
            };
            let cache = SynthesisCache::new(config);
            let mel = MelSpectrogram::new(vec![vec![1.0; 50]; 80], 22050, 256);

            b.iter(|| {
                for i in 0..(size * 2) {
                    let key =
                        SynthesisCacheKey::new(&format!("test_{}", i), Some(0), 1.0, 0.0, 1.0);
                    let _ = cache.insert(key, mel.clone());
                }
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_cache_operations,
    bench_eviction_policies,
    bench_monitoring_operations,
    bench_concurrent_monitoring,
    bench_cache_key_creation,
    bench_cache_with_different_sizes,
);
criterion_main!(benches);

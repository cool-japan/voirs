//! Benchmarks for neural architecture search and optimization

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use voirs_singing::prelude::*;
use voirs_singing::{
    EnergyEfficiencyConfig, EnergyEfficiencyOptimizer, HardwareOptimizer, HardwareOptimizerConfig,
    HardwarePlatform,
};

fn benchmark_population_initialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("population_initialization");

    for pop_size in [5, 10, 20, 50].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(pop_size),
            pop_size,
            |b, &pop_size| {
                b.iter(|| {
                    let config = NasConfig {
                        population_size: pop_size,
                        ..Default::default()
                    };
                    let mut searcher = NeuralArchitectureSearcher::new(config);
                    searcher.initialize_population();
                    black_box(());
                });
            },
        );
    }

    group.finish();
}

fn benchmark_architecture_search(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("architecture_search");
    group.sample_size(10); // Reduce sample size for longer operations

    for iterations in [10, 25, 50].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(iterations),
            iterations,
            |b, &iterations| {
                b.to_async(&rt).iter(|| async move {
                    let config = NasConfig {
                        search_iterations: iterations,
                        population_size: 10,
                        ..Default::default()
                    };
                    let mut searcher = NeuralArchitectureSearcher::new(config);
                    black_box(searcher.search().await.unwrap());
                });
            },
        );
    }

    group.finish();
}

fn benchmark_model_compression(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("model_compression");

    for model_size in [100_000, 500_000, 1_000_000, 5_000_000].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(model_size),
            model_size,
            |b, &model_size| {
                b.to_async(&rt).iter(|| async move {
                    let config = ModelCompressionConfig::default();
                    let compressor = ModelCompressor::new(config);
                    black_box(compressor.compress(model_size).await.unwrap());
                });
            },
        );
    }

    group.finish();
}

fn benchmark_hardware_optimization(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    c.bench_function("hardware_optimization_cpu", |b| {
        b.to_async(&rt).iter(|| async {
            let config = HardwareOptimizerConfig {
                platform: HardwarePlatform::CpuX86,
                enable_simd: true,
                enable_fusion: true,
                memory_opt_level: 2,
            };
            let optimizer = HardwareOptimizer::new(config);
            black_box(optimizer.optimize().await.unwrap());
        });
    });

    c.bench_function("hardware_optimization_gpu", |b| {
        b.to_async(&rt).iter(|| async {
            let config = HardwareOptimizerConfig {
                platform: HardwarePlatform::GpuNvidia,
                enable_simd: true,
                enable_fusion: true,
                memory_opt_level: 2,
            };
            let optimizer = HardwareOptimizer::new(config);
            black_box(optimizer.optimize().await.unwrap());
        });
    });
}

fn benchmark_energy_optimization(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    c.bench_function("energy_efficiency_optimization", |b| {
        b.to_async(&rt).iter(|| async {
            let config = EnergyEfficiencyConfig {
                power_budget: 50.0,
                enable_dvfs: true,
                optimize_batch_size: true,
                reduce_precision: true,
            };
            let optimizer = EnergyEfficiencyOptimizer::new(config);
            black_box(optimizer.optimize().await.unwrap());
        });
    });
}

criterion_group!(
    benches,
    benchmark_population_initialization,
    benchmark_architecture_search,
    benchmark_model_compression,
    benchmark_hardware_optimization,
    benchmark_energy_optimization
);
criterion_main!(benches);

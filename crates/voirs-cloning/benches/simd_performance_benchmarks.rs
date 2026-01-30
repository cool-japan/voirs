//! SIMD Performance Benchmarks
//!
//! Benchmarks comparing SIMD-optimized operations against standard implementations
//! to measure performance improvements for speaker embedding operations.
//!
//! # Running Benchmarks
//!
//! ```bash
//! cargo bench --bench simd_performance_benchmarks
//! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use voirs_cloning::embedding::{
    simd_cosine_similarity, simd_dot_product, simd_euclidean_distance, simd_l2_norm,
    simd_normalize_inplace, simd_weighted_average, SpeakerEmbedding,
};

/// Baseline cosine similarity using standard operations (no SIMD)
fn baseline_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

/// Baseline Euclidean distance using standard operations (no SIMD)
fn baseline_euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return f32::INFINITY;
    }

    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum::<f32>()
        .sqrt()
}

/// Benchmark cosine similarity computation
fn bench_cosine_similarity(c: &mut Criterion) {
    let dimensions = vec![64, 128, 256, 512, 1024];

    let mut group = c.benchmark_group("cosine_similarity");

    for dim in dimensions {
        let vec_a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin()).collect();
        let vec_b: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).cos()).collect();

        // Baseline implementation
        group.bench_with_input(BenchmarkId::new("baseline", dim), &dim, |b, _| {
            b.iter(|| {
                black_box(baseline_cosine_similarity(
                    black_box(&vec_a),
                    black_box(&vec_b),
                ))
            });
        });

        // SIMD implementation
        group.bench_with_input(BenchmarkId::new("simd", dim), &dim, |b, _| {
            b.iter(|| black_box(simd_cosine_similarity(black_box(&vec_a), black_box(&vec_b))));
        });

        // Using SpeakerEmbedding API (uses SIMD internally)
        let emb_a = SpeakerEmbedding::new(vec_a.clone());
        let emb_b = SpeakerEmbedding::new(vec_b.clone());

        group.bench_with_input(BenchmarkId::new("speaker_embedding", dim), &dim, |b, _| {
            b.iter(|| black_box(emb_a.similarity(black_box(&emb_b))));
        });
    }

    group.finish();
}

/// Benchmark Euclidean distance computation
fn bench_euclidean_distance(c: &mut Criterion) {
    let dimensions = vec![64, 128, 256, 512, 1024];

    let mut group = c.benchmark_group("euclidean_distance");

    for dim in dimensions {
        let vec_a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin()).collect();
        let vec_b: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).cos()).collect();

        // Baseline implementation
        group.bench_with_input(BenchmarkId::new("baseline", dim), &dim, |b, _| {
            b.iter(|| {
                black_box(baseline_euclidean_distance(
                    black_box(&vec_a),
                    black_box(&vec_b),
                ))
            });
        });

        // SIMD implementation
        group.bench_with_input(BenchmarkId::new("simd", dim), &dim, |b, _| {
            b.iter(|| {
                black_box(simd_euclidean_distance(
                    black_box(&vec_a),
                    black_box(&vec_b),
                ))
            });
        });
    }

    group.finish();
}

/// Benchmark vector normalization
fn bench_normalization(c: &mut Criterion) {
    let dimensions = vec![64, 128, 256, 512, 1024];

    let mut group = c.benchmark_group("normalization");

    for dim in dimensions {
        let vec: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin()).collect();

        // Baseline implementation
        group.bench_with_input(BenchmarkId::new("baseline", dim), &dim, |b, _| {
            b.iter(|| {
                let mut v = vec.clone();
                let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for x in &mut v {
                        *x /= norm;
                    }
                }
                black_box(v)
            });
        });

        // SIMD implementation
        group.bench_with_input(BenchmarkId::new("simd", dim), &dim, |b, _| {
            b.iter(|| {
                let mut v = vec.clone();
                simd_normalize_inplace(&mut v);
                black_box(v)
            });
        });
    }

    group.finish();
}

/// Benchmark weighted average computation
fn bench_weighted_average(c: &mut Criterion) {
    let num_embeddings = vec![2, 5, 10, 20];
    let dim = 256;

    let mut group = c.benchmark_group("weighted_average");

    for num in num_embeddings {
        let embeddings: Vec<Vec<f32>> = (0..num)
            .map(|i| (0..dim).map(|j| ((i + j) as f32 * 0.1).sin()).collect())
            .collect();
        let weights: Vec<f32> = (0..num).map(|_| 1.0 / num as f32).collect();

        // Baseline implementation
        group.bench_with_input(BenchmarkId::new("baseline", num), &num, |b, _| {
            b.iter(|| {
                let mut result = vec![0.0f32; dim];
                for (embedding, &weight) in embeddings.iter().zip(weights.iter()) {
                    for i in 0..dim {
                        result[i] += embedding[i] * weight;
                    }
                }
                black_box(result)
            });
        });

        // SIMD implementation
        group.bench_with_input(BenchmarkId::new("simd", num), &num, |b, _| {
            b.iter(|| {
                black_box(simd_weighted_average(
                    black_box(&embeddings),
                    black_box(&weights),
                ))
            });
        });
    }

    group.finish();
}

/// Benchmark dot product computation
fn bench_dot_product(c: &mut Criterion) {
    let dimensions = vec![64, 128, 256, 512, 1024, 2048];

    let mut group = c.benchmark_group("dot_product");

    for dim in dimensions {
        let vec_a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin()).collect();
        let vec_b: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).cos()).collect();

        // Baseline implementation
        group.bench_with_input(BenchmarkId::new("baseline", dim), &dim, |b, _| {
            b.iter(|| {
                black_box(
                    vec_a
                        .iter()
                        .zip(vec_b.iter())
                        .map(|(a, b)| a * b)
                        .sum::<f32>(),
                )
            });
        });

        // SIMD implementation
        group.bench_with_input(BenchmarkId::new("simd", dim), &dim, |b, _| {
            b.iter(|| black_box(simd_dot_product(black_box(&vec_a), black_box(&vec_b))));
        });
    }

    group.finish();
}

/// Benchmark L2 norm computation
fn bench_l2_norm(c: &mut Criterion) {
    let dimensions = vec![64, 128, 256, 512, 1024, 2048];

    let mut group = c.benchmark_group("l2_norm");

    for dim in dimensions {
        let vec: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin()).collect();

        // Baseline implementation
        group.bench_with_input(BenchmarkId::new("baseline", dim), &dim, |b, _| {
            b.iter(|| black_box(vec.iter().map(|x| x * x).sum::<f32>().sqrt()));
        });

        // SIMD implementation
        group.bench_with_input(BenchmarkId::new("simd", dim), &dim, |b, _| {
            b.iter(|| black_box(simd_l2_norm(black_box(&vec))));
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_cosine_similarity,
    bench_euclidean_distance,
    bench_normalization,
    bench_weighted_average,
    bench_dot_product,
    bench_l2_norm
);

criterion_main!(benches);

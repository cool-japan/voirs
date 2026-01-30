//! Comprehensive benchmarks for speaker embedding operations
//!
//! These benchmarks measure the performance of critical embedding operations
//! including similarity calculations, distance computations, and normalization.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use voirs_cloning::embedding::{EmbeddingMetadata, SpeakerEmbedding, VoiceQuality};

/// Generate a speaker embedding with specified dimension
fn create_embedding(dim: usize) -> SpeakerEmbedding {
    let vector: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.01).sin()).collect();
    SpeakerEmbedding::new(vector)
}

/// Benchmark cosine similarity calculation
fn bench_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("similarity");

    for dim in &[128, 256, 512, 1024, 2048] {
        group.throughput(Throughput::Elements(*dim as u64));
        group.bench_with_input(BenchmarkId::from_parameter(dim), dim, |b, &dim| {
            let emb1 = create_embedding(dim);
            let emb2 = create_embedding(dim);

            b.iter(|| black_box(emb1.similarity(black_box(&emb2))));
        });
    }

    group.finish();
}

/// Benchmark Euclidean distance calculation
fn bench_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance");

    for dim in &[128, 256, 512, 1024, 2048] {
        group.throughput(Throughput::Elements(*dim as u64));
        group.bench_with_input(BenchmarkId::from_parameter(dim), dim, |b, &dim| {
            let emb1 = create_embedding(dim);
            let emb2 = create_embedding(dim);

            b.iter(|| black_box(emb1.distance(black_box(&emb2))));
        });
    }

    group.finish();
}

/// Benchmark L2 normalization
fn bench_normalize(c: &mut Criterion) {
    let mut group = c.benchmark_group("normalize");

    for dim in &[128, 256, 512, 1024, 2048] {
        group.throughput(Throughput::Elements(*dim as u64));
        group.bench_with_input(BenchmarkId::from_parameter(dim), dim, |b, &dim| {
            b.iter_batched(
                || create_embedding(dim),
                |mut emb| {
                    emb.normalize();
                    black_box(emb)
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

/// Benchmark batch similarity calculations
fn bench_batch_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_similarity");

    for batch_size in &[10, 50, 100, 500] {
        group.throughput(Throughput::Elements(*batch_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            batch_size,
            |b, &batch_size| {
                let query_emb = create_embedding(256);
                let candidates: Vec<SpeakerEmbedding> =
                    (0..batch_size).map(|_| create_embedding(256)).collect();

                b.iter(|| {
                    for candidate in &candidates {
                        black_box(query_emb.similarity(candidate));
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark embedding validation
fn bench_is_valid(c: &mut Criterion) {
    let mut group = c.benchmark_group("is_valid");

    for dim in &[128, 256, 512, 1024, 2048] {
        group.throughput(Throughput::Elements(*dim as u64));
        group.bench_with_input(BenchmarkId::from_parameter(dim), dim, |b, &dim| {
            let emb = create_embedding(dim);

            b.iter(|| black_box(emb.is_valid()));
        });
    }

    group.finish();
}

/// Benchmark quality score calculation
fn bench_quality_score(c: &mut Criterion) {
    let mut group = c.benchmark_group("quality_score");

    for dim in &[128, 256, 512] {
        group.bench_with_input(BenchmarkId::from_parameter(dim), dim, |b, &dim| {
            let emb = create_embedding(dim);

            b.iter(|| black_box(emb.quality_score()));
        });
    }

    group.finish();
}

/// Benchmark voice quality overall quality calculation
fn bench_voice_quality_overall(c: &mut Criterion) {
    let quality = VoiceQuality {
        f0_mean: 150.0,
        f0_std: 20.0,
        spectral_centroid: 3000.0,
        spectral_bandwidth: 1500.0,
        jitter: 0.01,
        shimmer: 0.02,
        energy_mean: 0.5,
        energy_std: 0.1,
    };

    c.bench_function("voice_quality_overall", |b| {
        b.iter(|| black_box(quality.overall_quality()));
    });
}

/// Benchmark pairwise similarity matrix computation
fn bench_pairwise_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("pairwise_similarity");

    for num_embeddings in &[5, 10, 20, 50] {
        group.throughput(Throughput::Elements(
            (*num_embeddings * *num_embeddings) as u64,
        ));
        group.bench_with_input(
            BenchmarkId::from_parameter(num_embeddings),
            num_embeddings,
            |b, &num_embeddings| {
                let embeddings: Vec<SpeakerEmbedding> =
                    (0..num_embeddings).map(|_| create_embedding(256)).collect();

                b.iter(|| {
                    let mut similarities = Vec::new();
                    for i in 0..embeddings.len() {
                        for j in (i + 1)..embeddings.len() {
                            similarities.push(embeddings[i].similarity(&embeddings[j]));
                        }
                    }
                    black_box(similarities)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark nearest neighbor search
fn bench_nearest_neighbor(c: &mut Criterion) {
    let mut group = c.benchmark_group("nearest_neighbor");

    for num_candidates in &[10, 50, 100, 500, 1000] {
        group.throughput(Throughput::Elements(*num_candidates as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(num_candidates),
            num_candidates,
            |b, &num_candidates| {
                let query = create_embedding(256);
                let candidates: Vec<SpeakerEmbedding> =
                    (0..num_candidates).map(|_| create_embedding(256)).collect();

                b.iter(|| {
                    let mut max_sim = f32::NEG_INFINITY;
                    let mut best_idx = 0;

                    for (idx, candidate) in candidates.iter().enumerate() {
                        let sim = query.similarity(candidate);
                        if sim > max_sim {
                            max_sim = sim;
                            best_idx = idx;
                        }
                    }

                    black_box((best_idx, max_sim))
                });
            },
        );
    }

    group.finish();
}

/// Benchmark k-nearest neighbors search
fn bench_k_nearest_neighbors(c: &mut Criterion) {
    let mut group = c.benchmark_group("k_nearest_neighbors");

    let num_candidates = 1000;
    for k in &[1, 5, 10, 20, 50] {
        group.throughput(Throughput::Elements(*k as u64));
        group.bench_with_input(BenchmarkId::from_parameter(k), k, |b, &k| {
            let query = create_embedding(256);
            let candidates: Vec<SpeakerEmbedding> =
                (0..num_candidates).map(|_| create_embedding(256)).collect();

            b.iter(|| {
                let mut similarities: Vec<(usize, f32)> = candidates
                    .iter()
                    .enumerate()
                    .map(|(idx, candidate)| (idx, query.similarity(candidate)))
                    .collect();

                similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                let top_k: Vec<(usize, f32)> = similarities.into_iter().take(k).collect();

                black_box(top_k)
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_similarity,
    bench_distance,
    bench_normalize,
    bench_batch_similarity,
    bench_is_valid,
    bench_quality_score,
    bench_voice_quality_overall,
    bench_pairwise_similarity,
    bench_nearest_neighbor,
    bench_k_nearest_neighbors
);
criterion_main!(benches);

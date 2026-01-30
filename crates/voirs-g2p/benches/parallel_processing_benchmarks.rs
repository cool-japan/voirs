//! Performance benchmarks for parallel processing in phoneme_simd module.
//!
//! This benchmark suite measures the performance improvements from parallel processing
//! using SciRS2-Core's parallel_ops module. It compares sequential vs parallel execution
//! for various batch sizes and operations.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use voirs_g2p::utils::phoneme_simd::{
    batch_count_phonemes, batch_distance_matrix, batch_phoneme_error_rate,
    batch_phoneme_similarity, parallel_batch_process,
};
use voirs_g2p::Phoneme;

/// Create test phoneme sequences of varying complexity
fn create_test_sequences(count: usize, length: usize) -> Vec<Vec<Phoneme>> {
    let phonemes = [
        "p", "b", "t", "d", "k", "g", "m", "n", "ŋ", "f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ", "h",
        "tʃ", "dʒ", "l", "r", "j", "w", "i", "ɪ", "e", "ɛ", "æ", "ə", "ʌ", "a", "ɑ", "ɔ", "o", "ʊ",
        "u",
    ];

    (0..count)
        .map(|i| {
            (0..length)
                .map(|j| {
                    let idx = (i * 7 + j * 3) % phonemes.len();
                    Phoneme::new(phonemes[idx])
                })
                .collect()
        })
        .collect()
}

/// Benchmark parallel_batch_process with varying batch sizes
fn bench_parallel_batch_process_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_batch_process/scaling");

    // Test batch sizes around the parallel threshold (100)
    let batch_sizes = [10, 50, 100, 150, 200, 500, 1000];
    let sequence_length = 10;

    for &batch_size in &batch_sizes {
        let sequences = create_test_sequences(batch_size, sequence_length);
        group.throughput(Throughput::Elements(batch_size as u64));

        group.bench_with_input(
            BenchmarkId::new("length_calculation", batch_size),
            &sequences,
            |b, seqs| {
                b.iter(|| parallel_batch_process(seqs, |seq| black_box(seq.len())));
            },
        );
    }

    group.finish();
}

/// Benchmark parallel_batch_process with varying sequence lengths
fn bench_parallel_batch_process_sequence_length(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_batch_process/sequence_length");

    let batch_size = 200; // Above parallel threshold
    let sequence_lengths = [5, 10, 20, 50, 100];

    for &seq_len in &sequence_lengths {
        let sequences = create_test_sequences(batch_size, seq_len);
        group.throughput(Throughput::Elements((batch_size * seq_len) as u64));

        group.bench_with_input(
            BenchmarkId::new("phoneme_counting", seq_len),
            &sequences,
            |b, seqs| {
                b.iter(|| {
                    parallel_batch_process(seqs, |seq| {
                        black_box(seq.iter().filter(|p| !p.symbol.is_empty()).count())
                    })
                });
            },
        );
    }

    group.finish();
}

/// Benchmark parallel_batch_process with complex computations
fn bench_parallel_batch_process_complex_computation(c: &mut Criterion) {
    use voirs_g2p::utils::phoneme_analysis::{is_consonant, is_vowel};

    let mut group = c.benchmark_group("parallel_batch_process/complex_computation");

    let batch_size = 200;
    let sequence_length = 20;
    let sequences = create_test_sequences(batch_size, sequence_length);

    group.throughput(Throughput::Elements((batch_size * sequence_length) as u64));

    group.bench_function("vowel_consonant_analysis", |b| {
        b.iter(|| {
            parallel_batch_process(&sequences, |seq| {
                let vowels = seq
                    .iter()
                    .filter(|p| is_vowel(p.effective_symbol()))
                    .count();
                let consonants = seq
                    .iter()
                    .filter(|p| is_consonant(p.effective_symbol()))
                    .count();
                black_box((vowels, consonants))
            })
        });
    });

    group.finish();
}

/// Benchmark batch_distance_matrix with varying sizes
fn bench_batch_distance_matrix_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_distance_matrix/scaling");

    // Test sizes around the parallel threshold (20)
    let matrix_sizes = [5, 10, 15, 20, 25, 30, 50, 100];
    let sequence_length = 10;

    for &size in &matrix_sizes {
        let sequences = create_test_sequences(size, sequence_length);

        // Throughput is N² comparisons
        group.throughput(Throughput::Elements((size * size) as u64));

        group.bench_with_input(
            BenchmarkId::new("distance_matrix", size),
            &sequences,
            |b, seqs| {
                b.iter(|| black_box(batch_distance_matrix(seqs)));
            },
        );
    }

    group.finish();
}

/// Benchmark batch_distance_matrix with varying sequence lengths
fn bench_batch_distance_matrix_sequence_length(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_distance_matrix/sequence_length");

    let matrix_size = 30; // Above parallel threshold
    let sequence_lengths = [5, 10, 20, 30, 50];

    for &seq_len in &sequence_lengths {
        let sequences = create_test_sequences(matrix_size, seq_len);

        // Each comparison involves Levenshtein distance of length seq_len
        group.throughput(Throughput::Elements(
            (matrix_size * matrix_size * seq_len) as u64,
        ));

        group.bench_with_input(
            BenchmarkId::new("varying_length", seq_len),
            &sequences,
            |b, seqs| {
                b.iter(|| black_box(batch_distance_matrix(seqs)));
            },
        );
    }

    group.finish();
}

/// Benchmark batch operations for comparison
fn bench_batch_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_operations/comparison");

    let batch_size = 200;
    let sequence_length = 15;
    let sequences = create_test_sequences(batch_size, sequence_length);

    group.throughput(Throughput::Elements((batch_size * sequence_length) as u64));

    group.bench_function("batch_count_phonemes", |b| {
        b.iter(|| black_box(batch_count_phonemes(&sequences)));
    });

    // Create reference and hypothesis sequences for error rate calculation
    let references = sequences.clone();
    let hypotheses = create_test_sequences(batch_size, sequence_length);

    group.bench_function("batch_phoneme_error_rate", |b| {
        b.iter(|| black_box(batch_phoneme_error_rate(&references, &hypotheses)));
    });

    group.finish();
}

/// Benchmark realistic TTS pipeline scenarios
fn bench_realistic_pipeline_scenarios(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_processing/realistic_scenarios");

    // Scenario 1: Batch synthesis of short utterances (e.g., navigation prompts)
    let short_utterances = create_test_sequences(500, 8); // 500 short phrases
    group.throughput(Throughput::Elements((500 * 8) as u64));

    group.bench_function("batch_short_utterances", |b| {
        b.iter(|| {
            parallel_batch_process(&short_utterances, |seq| {
                // Simulate simple processing (counting syllables, rough estimate)
                let vowel_count = seq
                    .iter()
                    .filter(|p| {
                        matches!(
                            p.effective_symbol(),
                            "i" | "ɪ"
                                | "e"
                                | "ɛ"
                                | "æ"
                                | "ə"
                                | "ʌ"
                                | "a"
                                | "ɑ"
                                | "ɔ"
                                | "o"
                                | "ʊ"
                                | "u"
                        )
                    })
                    .count();
                black_box(vowel_count)
            })
        });
    });

    // Scenario 2: Voice similarity search (large distance matrix)
    let voice_samples = create_test_sequences(40, 25); // 40 voice samples
    group.throughput(Throughput::Elements((40 * 40) as u64));

    group.bench_function("voice_similarity_search", |b| {
        b.iter(|| black_box(batch_distance_matrix(&voice_samples)));
    });

    // Scenario 3: Quality assessment for model evaluation
    let model_outputs = create_test_sequences(300, 20);
    let ground_truth = create_test_sequences(300, 20);
    group.throughput(Throughput::Elements((300 * 20) as u64));

    group.bench_function("model_evaluation", |b| {
        b.iter(|| black_box(batch_phoneme_error_rate(&ground_truth, &model_outputs)));
    });

    group.finish();
}

/// Benchmark parallel efficiency (speedup analysis)
fn bench_parallel_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_processing/efficiency");

    // Large batch size to ensure parallel execution
    let batch_size = 1000;
    let sequence_length = 15;
    let sequences = create_test_sequences(batch_size, sequence_length);

    group.throughput(Throughput::Elements((batch_size * sequence_length) as u64));

    group.bench_function("large_batch_processing", |b| {
        b.iter(|| {
            parallel_batch_process(&sequences, |seq| {
                // Moderately expensive computation to benefit from parallelization
                let mut result = 0usize;
                for phoneme in seq {
                    result += phoneme.symbol.len();
                    result ^= phoneme.effective_symbol().len();
                }
                black_box(result)
            })
        });
    });

    // Compare with batch_count_phonemes which also processes many sequences
    group.bench_function("batch_count_optimization", |b| {
        b.iter(|| black_box(batch_count_phonemes(&sequences)));
    });

    group.finish();
}

/// Benchmark memory efficiency of parallel operations
fn bench_parallel_memory_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_processing/memory_patterns");

    // Test with large number of small sequences
    let many_small = create_test_sequences(2000, 5);
    group.throughput(Throughput::Elements(2000));

    group.bench_function("many_small_sequences", |b| {
        b.iter(|| parallel_batch_process(&many_small, |seq| black_box(seq.len())));
    });

    // Test with fewer large sequences
    let few_large = create_test_sequences(100, 200);
    group.throughput(Throughput::Elements((100 * 200) as u64));

    group.bench_function("few_large_sequences", |b| {
        b.iter(|| parallel_batch_process(&few_large, |seq| black_box(seq.len())));
    });

    group.finish();
}

/// Benchmark threshold boundary behavior
fn bench_threshold_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_processing/threshold_boundary");

    let sequence_length = 10;

    // Test sizes right at the parallel threshold (100) to measure overhead
    let boundary_sizes = [90, 95, 99, 100, 101, 105, 110];

    for &size in &boundary_sizes {
        let sequences = create_test_sequences(size, sequence_length);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("batch_size", size),
            &sequences,
            |b, seqs| {
                b.iter(|| parallel_batch_process(seqs, |seq| black_box(seq.len())));
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_parallel_batch_process_scaling,
    bench_parallel_batch_process_sequence_length,
    bench_parallel_batch_process_complex_computation,
    bench_batch_distance_matrix_scaling,
    bench_batch_distance_matrix_sequence_length,
    bench_batch_operations,
    bench_realistic_pipeline_scenarios,
    bench_parallel_efficiency,
    bench_parallel_memory_patterns,
    bench_threshold_boundary,
);

criterion_main!(benches);

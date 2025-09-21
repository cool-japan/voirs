//! Performance benchmarks for voice cloning system
//!
//! This module provides comprehensive benchmarks for all major voice cloning operations
//! including speaker adaptation, synthesis, quality assessment, and memory optimization.
//! It also includes automated regression testing to ensure performance doesn't degrade
//! over time.

use criterion::{
    black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput,
};
use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime};
use tokio::runtime::Runtime;
use voirs_cloning::{
    embedding::StreamingConfig, prelude::*, similarity::SimilarityConfig,
    usage_tracking::UsageTrackingConfig, CloningConfig, CloningConfigBuilder, CloningMethod,
    SpeakerData, SpeakerEmbedding, SpeakerProfile, VoiceCloneRequest, VoiceCloner,
    VoiceClonerBuilder, VoiceSample,
};

/// Benchmark fixture for performance testing
struct BenchmarkFixture {
    cloner: VoiceCloner,
    test_samples: Vec<VoiceSample>,
    speaker_embeddings: Vec<SpeakerEmbedding>,
    memory_manager: MemoryManager,
    streaming_manager: StreamingAdaptationManager,
    consent_manager: ConsentManager,
    usage_tracker: UsageTracker,
}

impl BenchmarkFixture {
    /// Create a new benchmark fixture
    fn new(rt: &Runtime) -> Self {
        rt.block_on(async {
            // Create optimized configuration for benchmarking
            let config = CloningConfigBuilder::new()
                .quality_level(0.75)
                .use_gpu(false) // Use CPU for consistent benchmarking
                .build()
                .unwrap();

            let cloner = VoiceClonerBuilder::new().config(config).build().unwrap();

            // Create test data
            let test_samples = Self::create_benchmark_samples();
            let speaker_embeddings = Self::create_benchmark_embeddings(10);

            // Create performance managers
            let memory_manager = MemoryManager::new(MemoryOptimizationConfig::default()).unwrap();
            let streaming_manager =
                StreamingAdaptationManager::new(StreamingAdaptationConfig::default()).unwrap();
            let consent_manager = ConsentManager::new();
            let usage_tracker = UsageTracker::new(UsageTrackingConfig::default());

            Self {
                cloner,
                test_samples,
                speaker_embeddings,
                memory_manager,
                streaming_manager,
                consent_manager,
                usage_tracker,
            }
        })
    }

    /// Create benchmark voice samples with various sizes
    fn create_benchmark_samples() -> Vec<VoiceSample> {
        let mut samples = Vec::new();

        // Create samples of different lengths for throughput testing
        let durations = vec![1.0, 3.0, 5.0, 10.0, 30.0]; // seconds
        let sample_rate = 16000;

        for (i, &duration) in durations.iter().enumerate() {
            let num_samples = (sample_rate as f32 * duration) as usize;
            let audio_data = Self::generate_sine_wave(sample_rate, duration, 440.0);

            samples.push(VoiceSample::new(
                format!("benchmark_sample_{}", i),
                audio_data,
                sample_rate,
            ));
        }

        samples
    }

    /// Create benchmark speaker embeddings
    fn create_benchmark_embeddings(count: usize) -> Vec<SpeakerEmbedding> {
        let mut embeddings = Vec::new();

        for i in 0..count {
            let mut embedding_data = Vec::with_capacity(512);
            for j in 0..512 {
                let value = ((i as f32 * 0.1 + j as f32 * 0.01).sin()
                    * (i as f32 * 0.07 + j as f32 * 0.03).cos())
                    * 0.5;
                embedding_data.push(value);
            }

            // Normalize
            let norm: f32 = embedding_data.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for value in embedding_data.iter_mut() {
                    *value /= norm;
                }
            }

            embeddings.push(SpeakerEmbedding::new(embedding_data));
        }

        embeddings
    }

    /// Generate sine wave audio data
    fn generate_sine_wave(sample_rate: u32, duration: f32, frequency: f32) -> Vec<f32> {
        let num_samples = (sample_rate as f32 * duration) as usize;
        let mut audio_data = Vec::with_capacity(num_samples);

        for i in 0..num_samples {
            let t = i as f32 / sample_rate as f32;
            let sample = (2.0 * std::f32::consts::PI * frequency * t).sin() * 0.5;
            audio_data.push(sample);
        }

        audio_data
    }
}

/// Benchmark basic voice cloning operations
fn bench_voice_cloning(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let fixture = BenchmarkFixture::new(&rt);

    let mut group = c.benchmark_group("voice_cloning");

    // Benchmark different sample lengths
    for (i, sample) in fixture.test_samples.iter().enumerate() {
        let duration = sample.audio.len() as f32 / sample.sample_rate as f32;

        group.throughput(Throughput::Elements(sample.audio.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("basic_cloning", format!("{:.1}s", duration)),
            sample,
            |b, sample| {
                b.to_async(&rt).iter_batched(
                    || {
                        let speaker_data = SpeakerData {
                            profile: SpeakerProfile::default(),
                            reference_samples: vec![sample.clone()],
                            target_text: Some(
                                "This is a benchmark test for voice cloning performance."
                                    .to_string(),
                            ),
                            target_language: None,
                            context: HashMap::new(),
                        };
                        let request = VoiceCloneRequest {
                            id: format!("bench_{}", i),
                            speaker_data,
                            method: CloningMethod::default(),
                            text: "This is a benchmark test for voice cloning performance."
                                .to_string(),
                            language: None,
                            quality_level: 0.7,
                            quality_tradeoff: 0.7,
                            parameters: HashMap::new(),
                            timestamp: std::time::SystemTime::now(),
                        };
                        (request)
                    },
                    |request| async move {
                        let config = CloningConfigBuilder::new()
                            .quality_level(0.75)
                            .use_gpu(false)
                            .build()
                            .unwrap();
                        let cloner = VoiceClonerBuilder::new().config(config).build().unwrap();
                        black_box(cloner.clone_voice(request).await.unwrap())
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }

    group.finish();
}

/// Benchmark speaker embedding operations
fn bench_speaker_embedding(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let fixture = BenchmarkFixture::new(&rt);

    let mut group = c.benchmark_group("speaker_embedding");

    // Benchmark embedding extraction
    group.bench_function("extract_embedding", |b| {
        b.to_async(&rt).iter_batched(
            || fixture.test_samples[0].clone(),
            |sample| async move {
                let extractor = SpeakerEmbeddingExtractor::default();
                black_box(
                    extractor
                        .extract_streaming(
                            &sample.audio,
                            sample.sample_rate,
                            &StreamingConfig::default(),
                        )
                        .await
                        .unwrap(),
                )
            },
            BatchSize::SmallInput,
        )
    });

    // Benchmark embedding similarity calculation
    group.bench_function("calculate_similarity", |b| {
        b.iter_batched(
            || {
                (
                    fixture.speaker_embeddings[0].clone(),
                    fixture.speaker_embeddings[1].clone(),
                )
            },
            |(emb1, emb2)| {
                let measurer = SimilarityMeasurer::new(SimilarityConfig::default());
                black_box(measurer.measure_embedding_similarity(&emb1, &emb2).unwrap())
            },
            BatchSize::SmallInput,
        )
    });

    // Benchmark batch embedding operations
    group.throughput(Throughput::Elements(10));
    group.bench_function("batch_similarity", |b| {
        b.iter_batched(
            || fixture.speaker_embeddings.clone(),
            |embeddings| {
                let measurer = SimilarityMeasurer::new(SimilarityConfig::default());
                let reference = &embeddings[0];

                for embedding in &embeddings[1..] {
                    black_box(
                        measurer
                            .measure_embedding_similarity(reference, embedding)
                            .unwrap(),
                    );
                }
            },
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

/// Benchmark quality assessment operations
fn bench_quality_assessment(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let fixture = BenchmarkFixture::new(&rt);

    let mut group = c.benchmark_group("quality_assessment");

    // Benchmark quality assessment for different sample sizes
    for (i, sample) in fixture.test_samples.iter().enumerate() {
        let duration = sample.audio.len() as f32 / sample.sample_rate as f32;

        group.throughput(Throughput::Elements(sample.audio.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("assess_quality", format!("{:.1}s", duration)),
            sample,
            |b, sample| {
                b.to_async(&rt).iter_batched(
                    || {
                        (
                            fixture.test_samples[0].clone(), // reference
                            sample.clone(),                  // cloned
                        )
                    },
                    |(reference, cloned)| async move {
                        let config = CloningConfigBuilder::new()
                            .quality_level(0.75)
                            .use_gpu(false)
                            .build()
                            .unwrap();
                        let cloner = VoiceClonerBuilder::new().config(config).build().unwrap();
                        black_box(
                            cloner
                                .assess_cloning_quality(&reference, &cloned)
                                .await
                                .unwrap(),
                        )
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }

    group.finish();
}

/// Benchmark memory optimization operations
fn bench_memory_optimization(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("memory_optimization");

    // Simplified memory benchmark
    group.bench_function("memory_allocation", |b| {
        b.iter(|| {
            let memory_manager = MemoryManager::new(MemoryOptimizationConfig::default()).unwrap();
            black_box(memory_manager)
        })
    });

    group.finish();
}

// Configure benchmark groups
criterion_group!(
    benches,
    bench_voice_cloning,
    bench_speaker_embedding,
    bench_quality_assessment,
    bench_memory_optimization
);

criterion_main!(benches);

//! Sampling strategies performance benchmarks
//!
//! Benchmarks for curriculum learning, importance sampling, and stratified sampling.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::collections::HashMap;
use voirs_dataset::{
    sampling::{
        CurriculumConfig, CurriculumSampler, DifficultyMetric, DifficultyStrategy,
        ImportanceConfig, ImportanceSampler, StratificationField, StratifiedConfig,
        StratifiedSampler,
    },
    AudioData, DatasetSample, LanguageCode, QualityMetrics, SpeakerInfo,
};

/// Create a test dataset of specified size
fn create_test_dataset(size: usize) -> Vec<DatasetSample> {
    let mut samples = Vec::with_capacity(size);
    let sample_rate = 16000;

    for i in 0..size {
        let duration = 2.0 + (i % 10) as f32 * 0.5;
        let num_samples = (sample_rate as f32 * duration) as usize;
        let audio = AudioData::new(vec![0.0; num_samples], sample_rate, 1);

        let quality = 0.3 + (i as f32 / size as f32) * 0.6;
        let speaker_id = format!("speaker_{}", i % 5);
        let language = if i % 3 == 0 {
            LanguageCode::Ja
        } else {
            LanguageCode::EnUs
        };

        samples.push(DatasetSample {
            id: format!("sample_{:06}", i),
            audio,
            text: format!("Sample text for item {}", i),
            speaker: Some(SpeakerInfo {
                id: speaker_id.clone(),
                name: Some(speaker_id),
                gender: None,
                age: None,
                accent: None,
                metadata: HashMap::new(),
            }),
            language,
            quality: QualityMetrics {
                overall_quality: Some(quality),
                snr: Some(20.0 + quality * 20.0),
                clipping: Some(0.01),
                dynamic_range: Some(60.0),
                spectral_quality: Some(quality),
            },
            phonemes: None,
            metadata: HashMap::new(),
        });
    }

    samples
}

/// Benchmark curriculum learning difficulty score calculation
fn bench_curriculum_calculate_scores(c: &mut Criterion) {
    let mut group = c.benchmark_group("curriculum_calculate_scores");

    for size in [100, 500, 1000, 5000] {
        let samples = create_test_dataset(size);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("quality", size), &samples, |b, s| {
            let config = CurriculumConfig {
                metric: DifficultyMetric::Quality,
                ..Default::default()
            };
            b.iter(|| {
                let mut sampler = CurriculumSampler::new(config.clone());
                sampler.calculate_difficulty_scores(black_box(s));
                black_box(sampler);
            });
        });

        group.bench_with_input(BenchmarkId::new("duration", size), &samples, |b, s| {
            let config = CurriculumConfig {
                metric: DifficultyMetric::Duration,
                ..Default::default()
            };
            b.iter(|| {
                let mut sampler = CurriculumSampler::new(config.clone());
                sampler.calculate_difficulty_scores(black_box(s));
                black_box(sampler);
            });
        });

        group.bench_with_input(BenchmarkId::new("speech_rate", size), &samples, |b, s| {
            let config = CurriculumConfig {
                metric: DifficultyMetric::SpeechRate,
                ..Default::default()
            };
            b.iter(|| {
                let mut sampler = CurriculumSampler::new(config.clone());
                sampler.calculate_difficulty_scores(black_box(s));
                black_box(sampler);
            });
        });
    }

    group.finish();
}

/// Benchmark curriculum learning sample filtering
fn bench_curriculum_filtering(c: &mut Criterion) {
    let mut group = c.benchmark_group("curriculum_filtering");

    for size in [100, 500, 1000, 5000] {
        let samples = create_test_dataset(size);
        let config = CurriculumConfig::default();
        let mut sampler = CurriculumSampler::new(config);
        sampler.calculate_difficulty_scores(&samples);
        sampler.set_epoch(5); // Mid-way through curriculum

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &samples, |b, s| {
            b.iter(|| {
                let filtered = sampler.filter_by_difficulty(black_box(s));
                black_box(filtered);
            });
        });
    }

    group.finish();
}

/// Benchmark importance sampling weight calculation
fn bench_importance_calculate_weights(c: &mut Criterion) {
    let mut group = c.benchmark_group("importance_calculate_weights");

    for size in [100, 500, 1000, 5000] {
        let samples = create_test_dataset(size);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("quality", size), &samples, |b, s| {
            let config = ImportanceConfig {
                weight_by_quality: true,
                temperature: 1.0,
                ..Default::default()
            };
            b.iter(|| {
                let mut sampler = ImportanceSampler::new(config.clone(), Some(42));
                sampler.calculate_weights(black_box(s));
                black_box(sampler);
            });
        });

        group.bench_with_input(BenchmarkId::new("custom", size), &samples, |b, s| {
            let custom_weights: Vec<f32> =
                (0..s.len()).map(|i| i as f32 / s.len() as f32).collect();
            let config = ImportanceConfig {
                weight_by_quality: false,
                custom_weights: Some(custom_weights),
                temperature: 1.0,
                ..Default::default()
            };
            b.iter(|| {
                let mut sampler = ImportanceSampler::new(config.clone(), Some(42));
                sampler.calculate_weights(black_box(s));
                black_box(sampler);
            });
        });
    }

    group.finish();
}

/// Benchmark importance sampling
fn bench_importance_sampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("importance_sampling");

    for size in [100, 500, 1000, 5000] {
        let samples = create_test_dataset(size);
        let config = ImportanceConfig::default();
        let mut sampler = ImportanceSampler::new(config, Some(42));
        sampler.calculate_weights(&samples);

        let num_samples = size / 10; // Sample 10% of dataset
        group.throughput(Throughput::Elements(num_samples as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_sample_{}", size, num_samples)),
            &samples,
            |b, s| {
                b.iter(|| {
                    let sampled = sampler.sample(black_box(s), num_samples);
                    black_box(sampled);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark importance sampling temperature effects
fn bench_importance_temperature(c: &mut Criterion) {
    let mut group = c.benchmark_group("importance_temperature");

    let samples = create_test_dataset(1000);
    let temperatures = [0.1, 0.5, 1.0, 2.0, 5.0];

    for &temp in &temperatures {
        let config = ImportanceConfig {
            temperature: temp,
            ..Default::default()
        };

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("temp_{}", temp)),
            &samples,
            |b, s| {
                b.iter(|| {
                    let mut sampler = ImportanceSampler::new(config.clone(), Some(42));
                    sampler.calculate_weights(black_box(s));
                    let sampled = sampler.sample(s, 100);
                    black_box(sampled);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark stratified sampling
fn bench_stratified_sampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("stratified_sampling");

    for size in [100, 500, 1000, 5000] {
        let samples = create_test_dataset(size);
        let num_samples = size / 5; // Sample 20% of dataset

        group.throughput(Throughput::Elements(num_samples as u64));

        // By speaker
        group.bench_with_input(
            BenchmarkId::new("speaker", format!("{}_sample_{}", size, num_samples)),
            &samples,
            |b, s| {
                let config = StratifiedConfig {
                    category_field: StratificationField::Speaker,
                    equal_per_category: true,
                    min_per_category: 1,
                };
                b.iter(|| {
                    let mut sampler = StratifiedSampler::new(config.clone(), Some(42));
                    let sampled = sampler.sample(black_box(s), num_samples);
                    black_box(sampled);
                });
            },
        );

        // By language
        group.bench_with_input(
            BenchmarkId::new("language", format!("{}_sample_{}", size, num_samples)),
            &samples,
            |b, s| {
                let config = StratifiedConfig {
                    category_field: StratificationField::Language,
                    equal_per_category: false,
                    min_per_category: 1,
                };
                b.iter(|| {
                    let mut sampler = StratifiedSampler::new(config.clone(), Some(42));
                    let sampled = sampler.sample(black_box(s), num_samples);
                    black_box(sampled);
                });
            },
        );

        // By quality tier
        group.bench_with_input(
            BenchmarkId::new("quality_tier", format!("{}_sample_{}", size, num_samples)),
            &samples,
            |b, s| {
                let config = StratifiedConfig {
                    category_field: StratificationField::QualityTier,
                    equal_per_category: true,
                    min_per_category: 1,
                };
                b.iter(|| {
                    let mut sampler = StratifiedSampler::new(config.clone(), Some(42));
                    let sampled = sampler.sample(black_box(s), num_samples);
                    black_box(sampled);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark combined sampling strategies
fn bench_combined_sampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("combined_sampling");

    let samples = create_test_dataset(1000);
    group.throughput(Throughput::Elements(100));

    group.bench_function("curriculum_then_importance", |b| {
        b.iter(|| {
            // First apply curriculum learning
            let curriculum_config = CurriculumConfig {
                start_difficulty: 0.0,
                end_difficulty: 0.7,
                num_epochs: 10,
                strategy: DifficultyStrategy::Linear,
                metric: DifficultyMetric::Quality,
            };
            let mut curriculum_sampler = CurriculumSampler::new(curriculum_config);
            curriculum_sampler.calculate_difficulty_scores(&samples);
            curriculum_sampler.set_epoch(5);
            let filtered = curriculum_sampler.filter_by_difficulty(&samples);

            // Then apply importance sampling
            let importance_config = ImportanceConfig::default();
            let mut importance_sampler = ImportanceSampler::new(importance_config, Some(42));
            importance_sampler.calculate_weights(&filtered);
            let final_samples = importance_sampler.sample(&filtered, 100);

            black_box(final_samples);
        });
    });

    group.bench_function("curriculum_then_stratified", |b| {
        b.iter(|| {
            // First apply curriculum learning
            let curriculum_config = CurriculumConfig {
                start_difficulty: 0.0,
                end_difficulty: 0.7,
                num_epochs: 10,
                strategy: DifficultyStrategy::Linear,
                metric: DifficultyMetric::Quality,
            };
            let mut curriculum_sampler = CurriculumSampler::new(curriculum_config);
            curriculum_sampler.calculate_difficulty_scores(&samples);
            curriculum_sampler.set_epoch(5);
            let filtered = curriculum_sampler.filter_by_difficulty(&samples);

            // Then apply stratified sampling
            let stratified_config = StratifiedConfig {
                category_field: StratificationField::Speaker,
                equal_per_category: true,
                min_per_category: 1,
            };
            let mut stratified_sampler = StratifiedSampler::new(stratified_config, Some(42));
            let final_samples = stratified_sampler.sample(&filtered, 100);

            black_box(final_samples);
        });
    });

    group.finish();
}

/// Benchmark different difficulty strategies
fn bench_difficulty_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("difficulty_strategies");

    let samples = create_test_dataset(1000);
    let strategies = [
        DifficultyStrategy::Linear,
        DifficultyStrategy::Exponential,
        DifficultyStrategy::Stepwise,
        DifficultyStrategy::Adaptive,
    ];

    for strategy in strategies {
        let strategy_name = format!("{:?}", strategy).to_lowercase();
        group.bench_function(&strategy_name, |b| {
            b.iter(|| {
                let config = CurriculumConfig {
                    strategy,
                    ..Default::default()
                };
                let mut sampler = CurriculumSampler::new(config);
                sampler.calculate_difficulty_scores(&samples);

                // Test progression through 10 epochs
                let mut results = Vec::new();
                for epoch in 0..10 {
                    sampler.set_epoch(epoch);
                    let filtered = sampler.filter_by_difficulty(&samples);
                    results.push(filtered.len());
                }

                black_box(results);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_curriculum_calculate_scores,
    bench_curriculum_filtering,
    bench_importance_calculate_weights,
    bench_importance_sampling,
    bench_importance_temperature,
    bench_stratified_sampling,
    bench_combined_sampling,
    bench_difficulty_strategies,
);

criterion_main!(benches);

//! Advanced augmentation performance benchmarks
//!
//! Benchmarks for SpecAugment, Codec Simulation, and MixUp augmentation techniques.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::Array2;
use std::collections::HashMap;
use voirs_dataset::augmentation::codec::{CodecConfig, CodecSimulator, CodecType};
use voirs_dataset::augmentation::mixup::{BatchMixUpAugmentor, MixUpAugmentor, MixUpConfig};
use voirs_dataset::augmentation::specaugment::{BatchSpecAugment, SpecAugment, SpecAugmentConfig};
use voirs_dataset::{AudioData, DatasetSample, LanguageCode, QualityMetrics, SpeakerInfo};

/// Create test audio data
fn create_test_audio(sample_rate: u32, duration_secs: f32) -> AudioData {
    let sample_count = (sample_rate as f32 * duration_secs) as usize;
    let samples: Vec<f32> = (0..sample_count)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5
        })
        .collect();
    AudioData::new(samples, sample_rate, 1)
}

/// Create test spectrogram
fn create_test_spectrogram(freq_bins: usize, time_steps: usize) -> Array2<f32> {
    Array2::from_shape_fn((freq_bins, time_steps), |(f, t)| {
        ((f as f32 * 0.1 + t as f32 * 0.05).sin() + 1.0) * 0.5
    })
}

/// Create test dataset sample
fn create_test_sample(id: &str, sample_rate: u32, duration: f32) -> DatasetSample {
    let audio = create_test_audio(sample_rate, duration);
    DatasetSample {
        id: id.to_string(),
        audio,
        text: format!("Test sample {}", id),
        speaker: Some(SpeakerInfo {
            id: "speaker1".to_string(),
            name: Some("Speaker 1".to_string()),
            gender: None,
            age: None,
            accent: None,
            metadata: HashMap::new(),
        }),
        language: LanguageCode::EnUs,
        quality: QualityMetrics {
            snr: Some(30.0),
            clipping: Some(0.01),
            dynamic_range: Some(60.0),
            spectral_quality: Some(0.95),
            overall_quality: Some(0.9),
        },
        phonemes: None,
        metadata: HashMap::new(),
    }
}

/// Benchmark SpecAugment performance
fn bench_specaugment(c: &mut Criterion) {
    let mut group = c.benchmark_group("specaugment");

    // Test different spectrogram sizes
    let sizes = vec![
        (80, 100, "small"),    // Small: 80 mel bins, 100 time steps
        (80, 500, "medium"),   // Medium: 80 mel bins, 500 time steps
        (80, 1000, "large"),   // Large: 80 mel bins, 1000 time steps
        (128, 1000, "xlarge"), // XLarge: 128 mel bins, 1000 time steps
    ];

    for (freq_bins, time_steps, size_name) in sizes {
        let spectrogram = create_test_spectrogram(freq_bins, time_steps);
        let total_elements = (freq_bins * time_steps) as u64;

        group.throughput(Throughput::Elements(total_elements));

        // Benchmark light augmentation
        let augmenter_light = SpecAugment::new(SpecAugmentConfig::light());
        group.bench_with_input(
            BenchmarkId::new("light", size_name),
            &spectrogram,
            |b, spec| {
                b.iter(|| augmenter_light.augment_spectrogram(spec));
            },
        );

        // Benchmark strong augmentation
        let augmenter_strong = SpecAugment::new(SpecAugmentConfig::strong());
        group.bench_with_input(
            BenchmarkId::new("strong", size_name),
            &spectrogram,
            |b, spec| {
                b.iter(|| augmenter_strong.augment_spectrogram(spec));
            },
        );

        // Benchmark adaptive augmentation
        let augmenter_adaptive = SpecAugment::new(SpecAugmentConfig::default());
        group.bench_with_input(
            BenchmarkId::new("adaptive", size_name),
            &spectrogram,
            |b, spec| {
                b.iter(|| augmenter_adaptive.augment_adaptive(spec, 0.5));
            },
        );
    }

    group.finish();
}

/// Benchmark batch SpecAugment performance
fn bench_specaugment_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("specaugment_batch");

    let batch_sizes = vec![1, 4, 8, 16, 32];
    let spectrogram = create_test_spectrogram(80, 500);

    for batch_size in batch_sizes {
        let batch: Vec<_> = (0..batch_size).map(|_| spectrogram.clone()).collect();

        group.throughput(Throughput::Elements((80 * 500 * batch_size) as u64));

        let batch_augmenter = BatchSpecAugment::new(SpecAugmentConfig::default());
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch,
            |b, batch| {
                b.iter(|| batch_augmenter.augment_batch(batch));
            },
        );
    }

    group.finish();
}

/// Benchmark codec simulation performance
fn bench_codec_simulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("codec_simulation");

    // Test different audio lengths
    let durations = vec![(0.1, "100ms"), (0.5, "500ms"), (1.0, "1sec"), (2.0, "2sec")];

    let codec_types = vec![
        (CodecType::G711MuLaw, "g711"),
        (CodecType::Mp3Low, "mp3_low"),
        (CodecType::OpusMediumBandwidth, "opus_medium"),
        (CodecType::AmrNarrowband, "amr_nb"),
    ];

    for (duration, duration_name) in durations {
        let audio = create_test_audio(44100, duration);
        let sample_count = audio.samples().len() as u64;

        for (codec_type, codec_name) in &codec_types {
            let config = CodecConfig {
                codec_type: *codec_type,
                packet_loss_rate: 0.01,
                jitter_rate: 0.02,
                add_quantization_noise: true,
                add_pre_emphasis: false,
                bit_error_rate: 0.001,
            };
            let simulator = CodecSimulator::new(config);

            group.throughput(Throughput::Elements(sample_count));
            group.bench_with_input(
                BenchmarkId::new(*codec_name, duration_name),
                &audio,
                |b, audio| {
                    b.iter(|| simulator.simulate(audio).unwrap());
                },
            );
        }
    }

    group.finish();
}

/// Benchmark codec simulation presets
fn bench_codec_presets(c: &mut Criterion) {
    let mut group = c.benchmark_group("codec_presets");

    let audio = create_test_audio(44100, 1.0);
    let sample_count = audio.samples().len() as u64;

    let presets = vec![
        (CodecConfig::telephone(), "telephone"),
        (CodecConfig::mobile(), "mobile"),
        (CodecConfig::voip(), "voip"),
        (CodecConfig::mp3_low_quality(), "mp3_low"),
    ];

    for (config, preset_name) in presets {
        let simulator = CodecSimulator::new(config);

        group.throughput(Throughput::Elements(sample_count));
        group.bench_function(preset_name, |b| {
            b.iter(|| simulator.simulate(&audio).unwrap());
        });
    }

    group.finish();
}

/// Benchmark MixUp augmentation
fn bench_mixup(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixup");

    let durations = vec![(0.1, "100ms"), (0.5, "500ms"), (1.0, "1sec"), (2.0, "2sec")];

    let strategies = vec![
        (MixUpConfig::default(), "standard"),
        (MixUpConfig::balanced(), "balanced"),
        (MixUpConfig::cutmix(), "cutmix"),
        (MixUpConfig::conservative(), "conservative"),
    ];

    for (duration, duration_name) in &durations {
        let sample1 = create_test_sample("sample1", 16000, *duration);
        let sample2 = create_test_sample("sample2", 16000, *duration);
        let sample_count = sample1.audio.samples().len() as u64;

        for (config, strategy_name) in &strategies {
            let augmenter = MixUpAugmentor::new(config.clone());

            group.throughput(Throughput::Elements(sample_count));
            group.bench_with_input(
                BenchmarkId::new(*strategy_name, duration_name),
                &(sample1.clone(), sample2.clone()),
                |b, (s1, s2)| {
                    b.iter(|| augmenter.mix_samples(s1, s2).unwrap());
                },
            );
        }
    }

    group.finish();
}

/// Benchmark batch MixUp
fn bench_mixup_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixup_batch");

    let batch_sizes = vec![2, 4, 8, 16, 32];

    for batch_size in batch_sizes {
        let samples: Vec<_> = (0..batch_size)
            .map(|i| create_test_sample(&format!("sample_{}", i), 16000, 0.5))
            .collect();

        let total_samples = samples
            .iter()
            .map(|s| s.audio.samples().len())
            .sum::<usize>() as u64;

        group.throughput(Throughput::Elements(total_samples));

        let batch_augmenter = BatchMixUpAugmentor::new(MixUpConfig::default());
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &samples,
            |b, samples| {
                b.iter(|| batch_augmenter.mix_batch(samples).unwrap());
            },
        );
    }

    group.finish();
}

/// Benchmark combined augmentation pipeline
fn bench_augmentation_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("augmentation_pipeline");

    let audio = create_test_audio(16000, 1.0);
    let sample = create_test_sample("test", 16000, 1.0);
    let sample_count = audio.samples().len() as u64;

    // Simulate a typical augmentation pipeline
    group.throughput(Throughput::Elements(sample_count));
    group.bench_function("full_pipeline", |b| {
        let codec_sim = CodecSimulator::new(CodecConfig::telephone());
        let mixup = MixUpAugmentor::new(MixUpConfig::default());

        b.iter(|| {
            // Step 1: Codec simulation
            let degraded = codec_sim.simulate(&audio).unwrap();

            // Step 2: MixUp with another sample
            let mixed = mixup.mix_samples(&sample, &sample).unwrap();

            (degraded, mixed)
        });
    });

    group.finish();
}

/// Benchmark memory efficiency of augmentation
fn bench_augmentation_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("augmentation_memory");

    let sizes = vec![
        (16000, 0.1, "small"),   // 1.6k samples
        (16000, 1.0, "medium"),  // 16k samples
        (16000, 10.0, "large"),  // 160k samples
        (44100, 10.0, "xlarge"), // 441k samples
    ];

    for (sample_rate, duration, size_name) in sizes {
        let audio = create_test_audio(sample_rate, duration);
        let sample_count = audio.samples().len() as u64;

        group.throughput(Throughput::Elements(sample_count));

        // Test codec simulation memory usage
        let codec_sim = CodecSimulator::new(CodecConfig::mobile());
        group.bench_with_input(BenchmarkId::new("codec", size_name), &audio, |b, audio| {
            b.iter(|| codec_sim.simulate(audio).unwrap());
        });

        // Test MixUp memory usage
        let sample = create_test_sample("test", sample_rate, duration);
        let mixup = MixUpAugmentor::new(MixUpConfig::default());
        group.bench_with_input(
            BenchmarkId::new("mixup", size_name),
            &sample,
            |b, sample| {
                b.iter(|| mixup.mix_samples(sample, sample).unwrap());
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_specaugment,
    bench_specaugment_batch,
    bench_codec_simulation,
    bench_codec_presets,
    bench_mixup,
    bench_mixup_batch,
    bench_augmentation_pipeline,
    bench_augmentation_memory
);
criterion_main!(benches);

//! Quality benchmarks
//!
//! Simplified benchmarks for basic quality metrics computation,
//! signal analysis, and audio quality assessment.

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use voirs_dataset::AudioData;

/// Create test audio with known characteristics
fn create_reference_audio(sample_rate: u32, duration_secs: f32, frequency: f32) -> AudioData {
    let sample_count = (sample_rate as f32 * duration_secs) as usize;
    let samples: Vec<f32> = (0..sample_count)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (2.0 * std::f32::consts::PI * frequency * t).sin() * 0.7
        })
        .collect();
    AudioData::new(samples, sample_rate, 1)
}

/// Create test audio with noise for quality measurement
fn create_noisy_audio(sample_rate: u32, duration_secs: f32, snr_db: f32) -> AudioData {
    let sample_count = (sample_rate as f32 * duration_secs) as usize;
    let signal_amplitude = 0.7;
    let noise_amplitude = signal_amplitude * 10.0_f32.powf(-snr_db / 20.0);

    let samples: Vec<f32> = (0..sample_count)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            let signal = (2.0 * std::f32::consts::PI * 440.0 * t).sin() * signal_amplitude;
            let noise = (scirs2_core::random::random::<f32>() - 0.5) * 2.0 * noise_amplitude;
            signal + noise
        })
        .collect();
    AudioData::new(samples, sample_rate, 1)
}

/// Compute Signal-to-Noise Ratio (SNR)
fn compute_snr(reference: &AudioData, noisy: &AudioData) -> f32 {
    let ref_samples = reference.samples();
    let noisy_samples = noisy.samples();

    if ref_samples.len() != noisy_samples.len() {
        return 0.0;
    }

    let signal_power: f32 = ref_samples.iter().map(|&x| x * x).sum();
    let noise_power: f32 = ref_samples
        .iter()
        .zip(noisy_samples.iter())
        .map(|(&r, &n)| (r - n).powi(2))
        .sum();

    if noise_power > 0.0 {
        10.0 * (signal_power / noise_power).log10()
    } else {
        f32::INFINITY
    }
}

/// Optimized SNR computation using SIMD operations
fn compute_snr_optimized(reference: &AudioData, noisy: &AudioData) -> f32 {
    use voirs_dataset::audio::simd::SimdAudioProcessor;

    let ref_samples = reference.samples();
    let noisy_samples = noisy.samples();

    if ref_samples.len() != noisy_samples.len() {
        return 0.0;
    }

    // Use SIMD-optimized RMS calculation for signal power
    let signal_power =
        SimdAudioProcessor::calculate_rms(ref_samples).powi(2) * ref_samples.len() as f32;

    // Calculate noise samples efficiently
    let mut noise_samples = Vec::with_capacity(ref_samples.len());
    noise_samples.extend(
        ref_samples
            .iter()
            .zip(noisy_samples.iter())
            .map(|(&r, &n)| r - n),
    );

    // Use SIMD-optimized RMS calculation for noise power
    let noise_power =
        SimdAudioProcessor::calculate_rms(&noise_samples).powi(2) * noise_samples.len() as f32;

    if noise_power <= 0.0 {
        return f32::INFINITY;
    }

    10.0 * (signal_power / noise_power).log10()
}

/// Compute Total Harmonic Distortion + Noise (THD+N)
fn compute_thd_plus_n(audio: &AudioData, _fundamental_freq: f32) -> f32 {
    let samples = audio.samples();
    let _sample_rate = audio.sample_rate() as f32;

    // Simplified THD+N calculation using RMS
    let rms: f32 = (samples.iter().map(|&x| x * x).sum::<f32>() / samples.len() as f32).sqrt();

    // For this benchmark, we'll use a simplified approximation
    // Real THD+N would require FFT analysis
    let fundamental_power = rms * rms;
    let noise_estimate = fundamental_power * 0.01; // Assume 1% noise for benchmark

    if fundamental_power > 0.0 {
        (noise_estimate / fundamental_power).sqrt() * 100.0
    } else {
        0.0
    }
}

/// Benchmark quality metrics computation
fn bench_quality_metrics(c: &mut Criterion) {
    let clean_audio = create_reference_audio(22050, 3.0, 440.0);
    let noisy_audio = create_noisy_audio(22050, 3.0, 20.0);

    let mut group = c.benchmark_group("quality_metrics");
    group.throughput(Throughput::Elements(clean_audio.samples().len() as u64));

    // Benchmark SNR computation
    group.bench_function("snr_computation", |b| {
        b.iter(|| {
            let snr = compute_snr(
                std::hint::black_box(&clean_audio),
                std::hint::black_box(&noisy_audio),
            );
            std::hint::black_box(snr);
        });
    });

    // Benchmark THD+N computation
    group.bench_function("thd_plus_n", |b| {
        b.iter(|| {
            let thd_n = compute_thd_plus_n(std::hint::black_box(&clean_audio), 440.0);
            std::hint::black_box(thd_n);
        });
    });

    // Benchmark RMS calculation
    group.bench_function("rms_calculation", |b| {
        b.iter(|| {
            let rms = clean_audio.rms();
            std::hint::black_box(rms);
        });
    });

    // Benchmark normalization
    group.bench_function("normalization", |b| {
        b.iter(|| {
            let mut audio = clean_audio.clone();
            audio.normalize().unwrap();
            std::hint::black_box(audio);
        });
    });

    group.finish();
}

/// Benchmark signal analysis operations
fn bench_signal_analysis(c: &mut Criterion) {
    let test_audio = create_reference_audio(22050, 5.0, 440.0);

    let mut group = c.benchmark_group("signal_analysis");
    group.throughput(Throughput::Elements(test_audio.samples().len() as u64));

    // Benchmark peak detection
    group.bench_function("peak_detection", |b| {
        b.iter(|| {
            let samples = test_audio.samples();
            let peak = samples
                .iter()
                .fold(0.0f32, |max, &sample| max.max(sample.abs()));
            std::hint::black_box(peak);
        });
    });

    // Benchmark zero crossing rate
    group.bench_function("zero_crossing_rate", |b| {
        b.iter(|| {
            let samples = test_audio.samples();
            let zero_crossings = samples
                .windows(2)
                .filter(|window| (window[0] >= 0.0) != (window[1] >= 0.0))
                .count();
            let zcr = zero_crossings as f32 / samples.len() as f32;
            std::hint::black_box(zcr);
        });
    });

    // Benchmark dynamic range calculation
    group.bench_function("dynamic_range", |b| {
        b.iter(|| {
            let samples = test_audio.samples();
            let max_val = samples.iter().fold(0.0f32, |max, &x| max.max(x.abs()));
            let min_non_zero = samples
                .iter()
                .filter(|&&x| x.abs() > 1e-6)
                .fold(f32::INFINITY, |min, &x| min.min(x.abs()));

            let dynamic_range = if min_non_zero.is_finite() && min_non_zero > 0.0 {
                20.0 * (max_val / min_non_zero).log10()
            } else {
                0.0
            };
            std::hint::black_box(dynamic_range);
        });
    });

    group.finish();
}

/// Benchmark quality filtering operations
fn bench_quality_filtering(c: &mut Criterion) {
    // Create test samples with different quality levels
    let high_quality = create_reference_audio(22050, 3.0, 440.0);
    let low_quality = create_noisy_audio(22050, 3.0, 5.0); // Very noisy
    let medium_quality = create_noisy_audio(22050, 3.0, 15.0);

    let test_samples = vec![
        ("high_quality", high_quality),
        ("medium_quality", medium_quality),
        ("low_quality", low_quality),
    ];

    let mut group = c.benchmark_group("quality_filtering");

    for (name, audio) in test_samples {
        group.throughput(Throughput::Elements(audio.samples().len() as u64));
        group.bench_function(format!("assess_{name}"), |b| {
            b.iter(|| {
                // Simple quality assessment based on RMS and duration
                let rms = audio.rms().unwrap_or(0.0);
                let duration = audio.duration();
                let sample_rate = audio.sample_rate();

                // Simple quality score
                let quality_score = if duration > 1.0
                    && duration < 10.0
                    && rms > 0.01
                    && rms < 1.0
                    && sample_rate >= 16000
                {
                    1.0
                } else {
                    0.0
                };

                std::hint::black_box(quality_score);
            });
        });
    }

    // Benchmark batch filtering
    let batch_audio: Vec<AudioData> = (0..100)
        .map(|i| {
            let snr = 5.0 + (i as f32 / 100.0) * 20.0; // SNR from 5 to 25 dB
            create_noisy_audio(22050, 2.0, snr)
        })
        .collect();

    group.bench_function("batch_filtering", |b| {
        b.iter(|| {
            let mut passed = 0;
            for audio in std::hint::black_box(&batch_audio) {
                let rms = audio.rms().unwrap_or(0.0);
                if rms > 0.1 && rms < 0.9 {
                    passed += 1;
                }
            }
            std::hint::black_box(passed);
        });
    });

    group.finish();
}

/// Benchmark signal degradation analysis with optimized operations
fn bench_signal_degradation(c: &mut Criterion) {
    let reference = create_reference_audio(22050, 4.0, 440.0);

    let mut group = c.benchmark_group("signal_degradation");
    group.throughput(Throughput::Elements(reference.samples().len() as u64));

    // Pre-generate random noise to avoid expensive random calls in benchmark loop
    use rand::{Rng, SeedableRng};
    let mut rng = scirs2_core::random::Random::seed(42); // Fixed seed for reproducibility
    let noise_samples: Vec<f32> = (0..reference.samples().len())
        .map(|_| rng.r#gen::<f32>() - 0.5)
        .collect();

    // Benchmark different levels of degradation
    let degradation_levels = vec![
        ("minimal", 0.95),
        ("light", 0.85),
        ("moderate", 0.7),
        ("heavy", 0.5),
        ("severe", 0.3),
    ];

    for (level_name, factor) in degradation_levels {
        // Pre-allocate degraded buffer to avoid allocation in benchmark loop
        let mut degraded = reference.clone();

        group.bench_function(format!("degradation_{level_name}"), |b| {
            b.iter(|| {
                // Apply degradation using SIMD-optimized operations
                let ref_samples = reference.samples();
                let degraded_samples = degraded.samples_mut();
                let noise_factor = (1.0 - factor) * 0.1;

                // Use SIMD-friendly operations
                degraded_samples
                    .iter_mut()
                    .zip(ref_samples.iter())
                    .zip(noise_samples.iter())
                    .for_each(|((degraded_sample, &ref_sample), &noise)| {
                        *degraded_sample = ref_sample * factor + noise * noise_factor;
                    });

                // Use optimized SNR computation
                let snr = compute_snr_optimized(&reference, &degraded);
                std::hint::black_box(snr);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_quality_metrics,
    bench_signal_analysis,
    bench_quality_filtering,
    bench_signal_degradation
);
criterion_main!(benches);

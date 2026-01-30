//! GAN Loss Function Benchmarks
//!
//! Measures the computational performance of different GAN loss functions
//! used for training HiFi-GAN, BigVGAN, and UnivNet vocoders.
//!
//! This benchmark suite evaluates:
//! - Adversarial loss computation (least squares, hinge, BCE)
//! - Feature matching loss with different numbers of layers
//! - Multi-scale discriminator loss with varying scale counts
//!
//! Performance insights help optimize training pipelines and identify
//! bottlenecks in loss computation during vocoder training.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::{Array1, Array2};
use std::hint::black_box as hint_black_box;
use voirs_vocoder::loss::gan::{
    AdversarialLoss, AdversarialLossConfig, AdversarialLossType, FeatureMatchingLoss,
    FeatureMatchingLossConfig, MultiScaleDiscriminatorLoss, MultiScaleDiscriminatorLossConfig,
};

/// Generate realistic discriminator output for benchmarking
///
/// Creates arrays with dimensions typical of HiFi-GAN discriminators
fn generate_discriminator_output(batch_size: usize, time_steps: usize) -> Vec<Array1<f32>> {
    (0..batch_size)
        .map(|b| {
            Array1::from_shape_fn(time_steps, |t| {
                // Realistic logit range: -5.0 to 5.0
                let base = (b as f32 * 0.3 + t as f32 * 0.1) % 10.0 - 5.0;
                base + (b as f32).sin() * 0.5
            })
        })
        .collect()
}

/// Generate multiple discriminator feature maps for feature matching
fn generate_discriminator_features(
    batch_size: usize,
    num_layers: usize,
    feature_dim: usize,
) -> Vec<Array2<f32>> {
    (0..num_layers)
        .map(|layer| {
            // Feature dimensions typically decrease with depth
            let dim = feature_dim / (1 << layer).min(8);
            Array2::from_shape_fn((batch_size, dim), |(b, d)| {
                (b as f32 * 0.2 + d as f32 * 0.1 + layer as f32) % 1.0
            })
        })
        .collect()
}

/// Benchmark adversarial loss computation (least squares variant)
fn bench_adversarial_loss_least_squares(c: &mut Criterion) {
    let mut group = c.benchmark_group("adversarial_loss_least_squares");

    let config = AdversarialLossConfig {
        loss_type: AdversarialLossType::LeastSquares,
        weight: 1.0,
    };
    let loss_calculator = AdversarialLoss::new(config);

    for &batch_size in &[1, 4, 16, 32] {
        for &time_steps in &[64, 256, 1024] {
            let discriminator_output = generate_discriminator_output(batch_size, time_steps);
            let total_elements = (batch_size * time_steps) as u64;

            group.throughput(Throughput::Elements(total_elements));
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("batch{}_time{}", batch_size, time_steps)),
                &discriminator_output,
                |b, output| {
                    b.iter(|| {
                        let loss = loss_calculator.generator_loss(black_box(output)).unwrap();
                        hint_black_box(loss);
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark adversarial loss computation (hinge variant)
fn bench_adversarial_loss_hinge(c: &mut Criterion) {
    let mut group = c.benchmark_group("adversarial_loss_hinge");

    let config = AdversarialLossConfig {
        loss_type: AdversarialLossType::Hinge,
        weight: 1.0,
    };
    let loss_calculator = AdversarialLoss::new(config);

    for &batch_size in &[1, 4, 16, 32] {
        for &time_steps in &[64, 256, 1024] {
            let discriminator_output = generate_discriminator_output(batch_size, time_steps);
            let total_elements = (batch_size * time_steps) as u64;

            group.throughput(Throughput::Elements(total_elements));
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("batch{}_time{}", batch_size, time_steps)),
                &discriminator_output,
                |b, output| {
                    b.iter(|| {
                        let loss = loss_calculator.generator_loss(black_box(output)).unwrap();
                        hint_black_box(loss);
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark adversarial loss computation (BCE variant)
fn bench_adversarial_loss_bce(c: &mut Criterion) {
    let mut group = c.benchmark_group("adversarial_loss_bce");

    let config = AdversarialLossConfig {
        loss_type: AdversarialLossType::BCE,
        weight: 1.0,
    };
    let loss_calculator = AdversarialLoss::new(config);

    for &batch_size in &[1, 4, 16, 32] {
        for &time_steps in &[64, 256, 1024] {
            let discriminator_output = generate_discriminator_output(batch_size, time_steps);
            let total_elements = (batch_size * time_steps) as u64;

            group.throughput(Throughput::Elements(total_elements));
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("batch{}_time{}", batch_size, time_steps)),
                &discriminator_output,
                |b, output| {
                    b.iter(|| {
                        let loss = loss_calculator.generator_loss(black_box(output)).unwrap();
                        hint_black_box(loss);
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark discriminator loss computation
fn bench_discriminator_loss(c: &mut Criterion) {
    let mut group = c.benchmark_group("discriminator_loss");

    let config = AdversarialLossConfig {
        loss_type: AdversarialLossType::LeastSquares,
        weight: 1.0,
    };
    let loss_calculator = AdversarialLoss::new(config);

    for &batch_size in &[1, 4, 16] {
        for &time_steps in &[64, 256, 1024] {
            let real_output = generate_discriminator_output(batch_size, time_steps);
            let fake_output = generate_discriminator_output(batch_size, time_steps);
            let total_elements = (batch_size * time_steps * 2) as u64;

            group.throughput(Throughput::Elements(total_elements));
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("batch{}_time{}", batch_size, time_steps)),
                &(real_output, fake_output),
                |b, (real, fake)| {
                    b.iter(|| {
                        let loss = loss_calculator
                            .discriminator_loss(black_box(real), black_box(fake))
                            .unwrap();
                        hint_black_box(loss);
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark feature matching loss with varying layer counts
fn bench_feature_matching_loss(c: &mut Criterion) {
    let mut group = c.benchmark_group("feature_matching_loss");

    let config = FeatureMatchingLossConfig {
        num_layers: 4,
        weight: 2.0,
    };

    for &batch_size in &[1, 4, 16] {
        for &num_layers in &[2, 4, 6, 8] {
            let feature_dim = 512;
            let real_features =
                generate_discriminator_features(batch_size, num_layers, feature_dim);
            let fake_features =
                generate_discriminator_features(batch_size, num_layers, feature_dim);

            // Create loss calculator with correct number of layers
            let mut config_for_test = config.clone();
            config_for_test.num_layers = num_layers;
            let loss_calculator = FeatureMatchingLoss::new(config_for_test);

            // Count total elements for throughput measurement
            let total_elements: u64 = real_features
                .iter()
                .map(|f| (f.nrows() * f.ncols()) as u64)
                .sum();

            group.throughput(Throughput::Elements(total_elements));
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("batch{}_layers{}", batch_size, num_layers)),
                &(real_features, fake_features),
                |b, (real_feat, fake_feat)| {
                    b.iter(|| {
                        let loss = loss_calculator
                            .compute(black_box(real_feat), black_box(fake_feat))
                            .unwrap();
                        hint_black_box(loss);
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark feature matching loss with weighted layers
fn bench_feature_matching_loss_weighted(c: &mut Criterion) {
    let mut group = c.benchmark_group("feature_matching_loss_weighted");

    let batch_size = 8;
    let num_layers = 4;
    let feature_dim = 512;

    let real_features = generate_discriminator_features(batch_size, num_layers, feature_dim);
    let fake_features = generate_discriminator_features(batch_size, num_layers, feature_dim);

    // Different weight configurations
    let weight_configs = vec![
        ("weight_1.0", 1.0),
        ("weight_2.0", 2.0),
        ("weight_4.0", 4.0),
        ("weight_10.0", 10.0),
    ];

    for (name, weight) in weight_configs {
        let config = FeatureMatchingLossConfig { num_layers, weight };
        let loss_calculator = FeatureMatchingLoss::new(config);

        let total_elements: u64 = real_features
            .iter()
            .map(|f| (f.nrows() * f.ncols()) as u64)
            .sum();

        group.throughput(Throughput::Elements(total_elements));
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(&real_features, &fake_features),
            |b, (real_feat, fake_feat)| {
                b.iter(|| {
                    let loss = loss_calculator
                        .compute(black_box(real_feat), black_box(fake_feat))
                        .unwrap();
                    let _ = hint_black_box(loss);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark multi-scale discriminator loss
fn bench_multi_scale_discriminator_loss(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_scale_discriminator_loss");

    let batch_size = 8;
    let num_features = 4;

    for &num_scales in &[1, 2, 3, 5] {
        for &time_steps in &[64, 256, 512] {
            // Generate discriminator outputs for each scale
            let real_scores: Vec<_> = (0..num_scales)
                .map(|_| generate_discriminator_output(batch_size, time_steps))
                .collect();
            let fake_scores: Vec<_> = (0..num_scales)
                .map(|_| generate_discriminator_output(batch_size, time_steps))
                .collect();

            // Generate feature maps for each scale
            let real_features: Vec<_> = (0..num_scales)
                .map(|_| generate_discriminator_features(batch_size, num_features, 512))
                .collect();
            let fake_features: Vec<_> = (0..num_scales)
                .map(|_| generate_discriminator_features(batch_size, num_features, 512))
                .collect();

            let config = MultiScaleDiscriminatorLossConfig {
                num_scales,
                adversarial_config: AdversarialLossConfig {
                    loss_type: AdversarialLossType::LeastSquares,
                    weight: 1.0,
                },
                feature_matching_config: Some(FeatureMatchingLossConfig {
                    num_layers: num_features,
                    weight: 2.0,
                }),
            };
            let loss_calculator = MultiScaleDiscriminatorLoss::new(config);

            let total_elements = (num_scales * batch_size * time_steps) as u64;

            group.throughput(Throughput::Elements(total_elements));
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("scales{}_time{}", num_scales, time_steps)),
                &(real_scores, fake_scores, real_features, fake_features),
                |b, (_r_scores, f_scores, r_feat, f_feat)| {
                    b.iter(|| {
                        let loss = loss_calculator
                            .generator_loss(
                                black_box(f_scores),
                                black_box(Some(r_feat.as_slice())),
                                black_box(Some(f_feat.as_slice())),
                            )
                            .unwrap();
                        hint_black_box(loss);
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark comparison of different adversarial loss types
fn bench_adversarial_loss_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("adversarial_loss_comparison");

    let batch_size = 16;
    let time_steps = 256;
    let discriminator_output = generate_discriminator_output(batch_size, time_steps);

    let loss_types = vec![
        ("least_squares", AdversarialLossType::LeastSquares),
        ("hinge", AdversarialLossType::Hinge),
        ("bce", AdversarialLossType::BCE),
    ];

    for (name, loss_type) in loss_types {
        let config = AdversarialLossConfig {
            loss_type,
            weight: 1.0,
        };
        let loss_calculator = AdversarialLoss::new(config);

        group.throughput(Throughput::Elements((batch_size * time_steps) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &discriminator_output,
            |b, output| {
                b.iter(|| {
                    let loss = loss_calculator.generator_loss(black_box(output)).unwrap();
                    let _ = hint_black_box(loss);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_adversarial_loss_least_squares,
    bench_adversarial_loss_hinge,
    bench_adversarial_loss_bce,
    bench_discriminator_loss,
    bench_feature_matching_loss,
    bench_feature_matching_loss_weighted,
    bench_multi_scale_discriminator_loss,
    bench_adversarial_loss_comparison,
);

criterion_main!(benches);

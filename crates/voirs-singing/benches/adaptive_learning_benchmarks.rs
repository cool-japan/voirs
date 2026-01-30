//! Benchmarks for adaptive learning system performance

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use voirs_singing::prelude::*;

fn benchmark_feedback_collection(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    c.bench_function("feedback_collection_single", |b| {
        b.to_async(&rt).iter(|| async {
            let config = AdaptiveLearningConfig::default();
            let system = AdaptiveLearningSystem::new(config);

            let feedback = UserFeedback::new(
                "user1",
                vec![0.0; 16000],
                4.5,
                Some("Test feedback".to_string()),
            );

            system.add_feedback(feedback).await.unwrap();
            black_box(());
        });
    });
}

fn benchmark_preference_learning(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("preference_learning");

    for num_samples in [10, 50, 100, 200].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_samples),
            num_samples,
            |b, &num_samples| {
                b.to_async(&rt).iter(|| async move {
                    let config = AdaptiveLearningConfig::default();
                    let system = AdaptiveLearningSystem::new(config);

                    for _ in 0..num_samples {
                        let feedback =
                            UserFeedback::new("user1", vec![0.0; 16000], 4.0, None::<String>);
                        system.add_feedback(feedback).await.unwrap();
                    }

                    black_box(system.get_statistics().await);
                });
            },
        );
    }

    group.finish();
}

fn benchmark_recommendations(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    c.bench_function("get_recommendations", |b| {
        b.to_async(&rt).iter(|| async {
            let config = AdaptiveLearningConfig {
                min_samples: 5,
                confidence_threshold: 0.5,
                ..Default::default()
            };
            let system = AdaptiveLearningSystem::new(config);

            // Add sufficient feedback
            for _ in 0..10 {
                let feedback = UserFeedback::new("user1", vec![0.0; 16000], 4.5, None::<String>);
                system.add_feedback(feedback).await.unwrap();
            }

            black_box(system.get_recommendations("user1").await.unwrap());
        });
    });
}

fn benchmark_quality_weight_updates(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    c.bench_function("quality_weight_updates", |b| {
        b.to_async(&rt).iter(|| async {
            let config = AdaptiveLearningConfig {
                learning_rate: 0.01,
                ..Default::default()
            };
            let system = AdaptiveLearningSystem::new(config);

            let mut feedback = UserFeedback::new("user1", vec![0.0; 16000], 4.0, None::<String>);
            feedback.quality_ratings.pitch_accuracy = 5.0;
            feedback.quality_ratings.timing_precision = 3.5;

            system.add_feedback(feedback).await.unwrap();
            black_box(());
        });
    });
}

criterion_group!(
    benches,
    benchmark_feedback_collection,
    benchmark_preference_learning,
    benchmark_recommendations,
    benchmark_quality_weight_updates
);
criterion_main!(benches);

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::collections::HashMap;
use voirs_emotion::interpolation::InterpolationConfig;
use voirs_emotion::prelude::*;
use voirs_emotion::ssml::EmotionSSMLProcessor;

fn bench_emotion_processor_creation(c: &mut Criterion) {
    c.bench_function("emotion_processor_creation", |b| {
        b.iter(|| black_box(EmotionProcessor::new().unwrap()))
    });
}

fn bench_emotion_setting(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let processor = rt.block_on(async { EmotionProcessor::new().unwrap() });

    c.bench_function("set_single_emotion", |b| {
        b.to_async(&rt).iter(|| async {
            black_box(
                processor
                    .set_emotion(Emotion::Happy, Some(0.8))
                    .await
                    .unwrap(),
            )
        })
    });

    c.bench_function("set_emotion_mix", |b| {
        b.to_async(&rt).iter(|| async {
            let mut emotions = HashMap::new();
            emotions.insert(Emotion::Happy, 0.6);
            emotions.insert(Emotion::Excited, 0.4);
            black_box(processor.set_emotion_mix(emotions).await.unwrap())
        })
    });
}

fn bench_emotion_interpolation(c: &mut Criterion) {
    let config = InterpolationConfig::default();
    let interpolator = EmotionInterpolator::new(config);

    let mut from_vector = EmotionVector::new();
    from_vector.add_emotion(Emotion::Happy, EmotionIntensity::LOW);
    let from_params = EmotionParameters::new(from_vector);

    let mut to_vector = EmotionVector::new();
    to_vector.add_emotion(Emotion::Sad, EmotionIntensity::HIGH);
    let to_params = EmotionParameters::new(to_vector);

    c.bench_function("emotion_interpolation", |b| {
        b.iter(|| {
            black_box(
                interpolator
                    .interpolate(&from_params, &to_params, 0.5)
                    .unwrap(),
            )
        })
    });

    let mut group = c.benchmark_group("interpolation_methods");
    for method in [
        InterpolationMethod::Linear,
        InterpolationMethod::EaseIn,
        InterpolationMethod::EaseOut,
        InterpolationMethod::EaseInOut,
        InterpolationMethod::Bezier,
        InterpolationMethod::Spline,
    ] {
        group.bench_with_input(
            BenchmarkId::new("method", format!("{:?}", method)),
            &method,
            |b, &method| {
                let config = InterpolationConfig::default().with_method(method);
                let interpolator = EmotionInterpolator::new(config);
                b.iter(|| {
                    black_box(
                        interpolator
                            .interpolate(&from_params, &to_params, 0.5)
                            .unwrap(),
                    )
                })
            },
        );
    }
    group.finish();
}

fn bench_prosody_modification(c: &mut Criterion) {
    let modifier = ProsodyModifier::new();

    let mut emotion_vector = EmotionVector::new();
    emotion_vector.add_emotion(Emotion::Happy, EmotionIntensity::HIGH);
    let emotion_params = EmotionParameters::new(emotion_vector);

    c.bench_function("prosody_application", |b| {
        b.iter(|| black_box(modifier.apply_emotion(&emotion_params).unwrap()))
    });

    let dimensions = EmotionDimensions::new(0.8, 0.6, 0.4);
    c.bench_function("prosody_from_dimensions", |b| {
        b.iter(|| black_box(modifier.apply_emotion_dimensions(&dimensions).unwrap()))
    });
}

fn bench_emotion_vector_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("emotion_vector");

    group.bench_function("create_and_add_emotions", |b| {
        b.iter(|| {
            let mut vector = EmotionVector::new();
            vector.add_emotion(Emotion::Happy, EmotionIntensity::HIGH);
            vector.add_emotion(Emotion::Excited, EmotionIntensity::MEDIUM);
            vector.add_emotion(Emotion::Confident, EmotionIntensity::LOW);
            black_box(vector)
        })
    });

    let mut vector = EmotionVector::new();
    vector.add_emotion(Emotion::Happy, EmotionIntensity::HIGH);
    vector.add_emotion(Emotion::Excited, EmotionIntensity::MEDIUM);

    group.bench_function("dominant_emotion", |b| {
        b.iter(|| black_box(vector.dominant_emotion()))
    });

    group.bench_function("normalize", |b| {
        b.iter(|| {
            let mut v = vector.clone();
            v.normalize();
            black_box(v)
        })
    });

    group.finish();
}

fn bench_preset_operations(c: &mut Criterion) {
    let library = EmotionPresetLibrary::with_defaults();

    c.bench_function("preset_library_creation", |b| {
        b.iter(|| black_box(EmotionPresetLibrary::with_defaults()))
    });

    c.bench_function("preset_lookup", |b| {
        b.iter(|| black_box(library.get_preset("happy")))
    });

    c.bench_function("preset_parameters", |b| {
        b.iter(|| black_box(library.get_preset_parameters("happy", Some(0.8))))
    });

    c.bench_function("find_by_emotion", |b| {
        b.iter(|| black_box(library.find_by_emotion(Emotion::Happy)))
    });

    c.bench_function("find_by_tag", |b| {
        b.iter(|| black_box(library.find_by_tag("positive")))
    });
}

fn bench_ssml_processing(c: &mut Criterion) {
    let processor = EmotionSSMLProcessor::new();

    let simple_ssml = r#"<emotion:emotion name="happy" intensity="0.8">Hello world!</emotion>"#;
    let complex_ssml = r#"<emotion:emotion name="happy" intensity="0.8" duration="1000ms" pitch-shift="1.2">Hello</emotion> <emotion:emotion name="excited" intensity="0.9">world!</emotion>"#;

    c.bench_function("simple_ssml_parsing", |b| {
        b.iter(|| black_box(processor.process_ssml_text(simple_ssml).unwrap()))
    });

    c.bench_function("complex_ssml_parsing", |b| {
        b.iter(|| black_box(processor.process_ssml_text(complex_ssml).unwrap()))
    });

    let segments = processor.process_ssml_text(complex_ssml).unwrap();
    c.bench_function("ssml_generation", |b| {
        b.iter(|| black_box(processor.generate_ssml_from_segments(&segments).unwrap()))
    });
}

fn bench_transition_updates(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let processor = rt.block_on(async { EmotionProcessor::new().unwrap() });

    c.bench_function("transition_update", |b| {
        b.to_async(&rt).iter(|| async {
            black_box(processor.update_transition(16.67).await.unwrap()) // 60 FPS
        })
    });

    let mut group = c.benchmark_group("multiple_transitions");
    for num_transitions in [1, 5, 10, 20] {
        group.bench_with_input(
            BenchmarkId::new("transitions", num_transitions),
            &num_transitions,
            |b, &num_transitions| {
                b.to_async(&rt).iter(|| async {
                    // Set up multiple transitions
                    for i in 0..num_transitions {
                        let emotion = match i % 4 {
                            0 => Emotion::Happy,
                            1 => Emotion::Sad,
                            2 => Emotion::Angry,
                            _ => Emotion::Calm,
                        };
                        processor.set_emotion(emotion, Some(0.8)).await.unwrap();
                    }

                    // Update transitions
                    black_box(processor.update_transition(16.67).await.unwrap())
                })
            },
        );
    }
    group.finish();
}

fn bench_config_validation(c: &mut Criterion) {
    let valid_config = EmotionConfig::builder()
        .enabled(true)
        .max_emotions(5)
        .prosody_strength(0.8)
        .build_unchecked();

    c.bench_function("config_validation", |b| {
        b.iter(|| black_box(valid_config.validate().unwrap()))
    });

    c.bench_function("config_builder", |b| {
        b.iter(|| {
            black_box(
                EmotionConfig::builder()
                    .enabled(true)
                    .max_emotions(3)
                    .prosody_strength(0.7)
                    .build()
                    .unwrap(),
            )
        })
    });
}

fn bench_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");

    group.bench_function("large_emotion_vector", |b| {
        b.iter(|| {
            let mut vector = EmotionVector::new();
            for i in 0..100 {
                let emotion = match i % 12 {
                    0 => Emotion::Happy,
                    1 => Emotion::Sad,
                    2 => Emotion::Angry,
                    3 => Emotion::Fear,
                    4 => Emotion::Surprise,
                    5 => Emotion::Disgust,
                    6 => Emotion::Calm,
                    7 => Emotion::Excited,
                    8 => Emotion::Tender,
                    9 => Emotion::Confident,
                    10 => Emotion::Melancholic,
                    _ => Emotion::Custom(format!("custom_{}", i)),
                };
                vector.add_emotion(emotion, EmotionIntensity::new(0.5));
            }
            black_box(vector)
        })
    });

    group.bench_function("large_preset_library", |b| {
        b.iter(|| {
            let mut library = EmotionPresetLibrary::new();
            for i in 0..1000 {
                let preset = EmotionPreset::new(
                    format!("preset_{}", i),
                    format!("Description {}", i),
                    EmotionParameters::neutral(),
                );
                library.add_preset(preset);
            }
            black_box(library)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_emotion_processor_creation,
    bench_emotion_setting,
    bench_emotion_interpolation,
    bench_prosody_modification,
    bench_emotion_vector_operations,
    bench_preset_operations,
    bench_ssml_processing,
    bench_transition_updates,
    bench_config_validation,
    bench_memory_usage
);

criterion_main!(benches);

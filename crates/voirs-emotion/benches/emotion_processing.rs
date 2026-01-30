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
            processor
                .set_emotion(Emotion::Happy, Some(0.8))
                .await
                .unwrap();
            black_box(())
        })
    });

    c.bench_function("set_emotion_mix", |b| {
        b.to_async(&rt).iter(|| async {
            let mut emotions = HashMap::new();
            emotions.insert(Emotion::Happy, 0.6);
            emotions.insert(Emotion::Excited, 0.4);
            processor.set_emotion_mix(emotions).await.unwrap();
            black_box(())
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
            processor.update_transition(16.67).await.unwrap(); // 60 FPS
            black_box(())
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
                    processor.update_transition(16.67).await.unwrap();
                    black_box(())
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
        b.iter(|| {
            valid_config.validate().unwrap();
            black_box(())
        })
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

fn bench_cultural_adaptation(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    c.bench_function("cultural_adaptation_set", |b| {
        let processor = rt.block_on(async { EmotionProcessor::new().unwrap() });
        b.to_async(&rt).iter(|| async {
            processor.set_cultural_context("western").await.unwrap();
            black_box(())
        })
    });
}

// Natural variation benchmark temporarily disabled due to complex type requirements
// fn bench_natural_variation(c: &mut Criterion) {
//     // Complex benchmark implementation pending
// }

fn bench_consistency_manager(c: &mut Criterion) {
    use voirs_emotion::consistency::{
        EmotionConsistencyConfig, EmotionConsistencyManager, EmotionSegment,
    };

    let config = EmotionConsistencyConfig::default();

    let mut manager = EmotionConsistencyManager::new(config);

    c.bench_function("consistency_process_segment", |b| {
        b.iter(|| {
            let mut emotion_vec = EmotionVector::new();
            emotion_vec.add_emotion(Emotion::Happy, EmotionIntensity::new(0.7));
            let params = EmotionParameters::new(emotion_vec);

            let segment = EmotionSegment::new(
                "test_segment".to_string(),
                Emotion::Happy,
                EmotionIntensity::new(0.7),
                params,
                std::time::Duration::from_secs(2),
            );

            black_box(manager.process_segment(segment).unwrap())
        })
    });

    // Add some segments first
    for i in 0..10 {
        let mut emotion_vec = EmotionVector::new();
        emotion_vec.add_emotion(Emotion::Happy, EmotionIntensity::new(0.5 + i as f32 * 0.05));
        let params = EmotionParameters::new(emotion_vec);

        let segment = EmotionSegment::new(
            format!("segment_{}", i),
            Emotion::Happy,
            EmotionIntensity::new(0.5 + i as f32 * 0.05),
            params,
            std::time::Duration::from_secs(2),
        );

        let _ = manager.process_segment(segment);
    }

    c.bench_function("consistency_calculate_metrics", |b| {
        b.iter(|| black_box(manager.calculate_coherence_metrics()))
    });
}

fn bench_custom_emotions(c: &mut Criterion) {
    use voirs_emotion::custom::{CustomEmotionBuilder, CustomEmotionRegistry};

    c.bench_function("custom_emotion_create", |b| {
        b.iter(|| {
            black_box(
                CustomEmotionBuilder::new("test_emotion")
                    .description("Test emotion")
                    .dimensions(0.5, 0.6, 0.7)
                    .prosody(1.0, 1.0, 1.0)
                    .voice_quality(0.0, 0.0, 0.0, 0.0)
                    .build()
                    .unwrap(),
            )
        })
    });

    let mut registry = CustomEmotionRegistry::new();
    for i in 0..100 {
        let custom = CustomEmotionBuilder::new(format!("emotion_{}", i))
            .description("Test emotion")
            .dimensions(0.5, 0.6, 0.7)
            .build()
            .unwrap();
        registry.register(custom).unwrap();
    }

    c.bench_function("custom_emotion_lookup", |b| {
        b.iter(|| black_box(registry.get("emotion_50")))
    });

    c.bench_function("custom_emotion_search_by_tag", |b| {
        b.iter(|| black_box(registry.search_by_tag("test")))
    });
}

fn bench_realtime_audio_analysis(c: &mut Criterion) {
    use voirs_emotion::realtime::AudioCharacteristics;

    // Generate test audio with different characteristics
    let sample_rate = 16000.0;

    // Simple sine wave
    let simple_audio: Vec<f32> = (0..16000)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate).sin())
        .collect();

    // Complex audio with rhythm
    let complex_audio: Vec<f32> = (0..16000)
        .map(|i| {
            let t = i as f32 / sample_rate;
            let fundamental = (2.0 * std::f32::consts::PI * 440.0 * t).sin();
            let rhythm = if (i % 800) < 400 { 1.0 } else { 0.3 };
            fundamental * rhythm
        })
        .collect();

    c.bench_function("audio_characteristics_simple", |b| {
        b.iter(|| black_box(AudioCharacteristics::from_audio(&simple_audio, sample_rate)))
    });

    c.bench_function("audio_characteristics_complex", |b| {
        b.iter(|| {
            black_box(AudioCharacteristics::from_audio(
                &complex_audio,
                sample_rate,
            ))
        })
    });

    let mut group = c.benchmark_group("audio_buffer_sizes");
    for size in [1024, 4096, 16000, 48000] {
        group.bench_with_input(BenchmarkId::new("samples", size), &size, |b, &size| {
            let audio: Vec<f32> = (0..size)
                .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate).sin())
                .collect();
            b.iter(|| black_box(AudioCharacteristics::from_audio(&audio, sample_rate)))
        });
    }
    group.finish();
}

fn bench_simd_operations(c: &mut Criterion) {
    use voirs_emotion::core::simd_advanced::*;

    let mut group = c.benchmark_group("simd_operations");

    // Benchmark energy scaling with different buffer sizes
    for size in [1024, 4096, 16384, 65536] {
        group.bench_with_input(
            BenchmarkId::new("energy_scaling", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let mut audio = vec![0.5f32; size];
                    apply_energy_scaling_advanced(&mut audio, 1.5);
                    black_box(audio)
                })
            },
        );
    }

    // Benchmark audio blending
    for size in [1024, 4096, 16384, 65536] {
        group.bench_with_input(BenchmarkId::new("blend_audio", size), &size, |b, &size| {
            b.iter(|| {
                let audio_a = vec![0.5f32; size];
                let audio_b = vec![0.7f32; size];
                let mut output = vec![0.0f32; size];
                blend_audio_simd(&mut output, &audio_a, &audio_b, 0.5, 0.5);
                black_box(output)
            })
        });
    }

    // Benchmark convolution filtering
    for size in [1024, 4096, 16384] {
        group.bench_with_input(BenchmarkId::new("convolution", size), &size, |b, &size| {
            b.iter(|| {
                let kernel = vec![0.25f32; 128];
                let input = vec![0.5f32; size];
                let mut output = vec![0.0f32; size];
                convolve_emotion_filter(&mut output, &input, &kernel);
                black_box(output)
            })
        });
    }

    // Benchmark spectral filtering
    for size in [512, 1024, 2048, 4096] {
        group.bench_with_input(
            BenchmarkId::new("spectral_filter", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let mut spectrum = vec![1.0f32; size];
                    let filter = vec![0.8f32; size];
                    apply_spectral_emotion_filter(&mut spectrum, &filter);
                    black_box(spectrum)
                })
            },
        );
    }

    // Benchmark breathiness effect
    group.bench_function("breathiness_effect", |b| {
        b.iter(|| {
            let mut audio = vec![0.5f32; 4096];
            apply_breathiness_simd(&mut audio, 0.3, 44100.0);
            black_box(audio)
        })
    });

    // Benchmark roughness effect
    group.bench_function("roughness_effect", |b| {
        b.iter(|| {
            let mut audio = vec![0.5f32; 4096];
            apply_roughness_simd(&mut audio, 0.5, 44100.0);
            black_box(audio)
        })
    });

    group.finish();
}

fn bench_spectral_processing(c: &mut Criterion) {
    use voirs_emotion::core::spectral_advanced::*;

    let mut group = c.benchmark_group("spectral_processing");

    let processor = SpectralEmotionProcessor::new(44100.0, 2048, 512);

    // Benchmark emotion-based spectral filtering
    for size in [2048, 8192, 16384] {
        group.bench_with_input(
            BenchmarkId::new("emotion_filter", size),
            &size,
            |b, &size| {
                let audio = vec![0.5f32; size];
                b.iter(|| black_box(processor.apply_emotion_filter(&audio, 0.5, 0.7).unwrap()))
            },
        );
    }

    // Benchmark formant shifting
    group.bench_function("formant_shift", |b| {
        let audio = vec![0.5f32; 8192];
        b.iter(|| black_box(processor.apply_formant_shift(&audio, 1.2).unwrap()))
    });

    // Benchmark spectral envelope enhancement
    group.bench_function("envelope_enhancement", |b| {
        let audio = vec![0.5f32; 8192];
        b.iter(|| black_box(processor.enhance_spectral_envelope(&audio, 0.3).unwrap()))
    });

    // Benchmark spectral tilt
    group.bench_function("spectral_tilt", |b| {
        let audio = vec![0.5f32; 8192];
        b.iter(|| black_box(processor.apply_spectral_tilt(&audio, 3.0).unwrap()))
    });

    group.finish();
}

// Note: More comprehensive benchmarks for history, mobile, and debug modules
// would require mocking/setup that is better suited for integration tests

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
    bench_memory_usage,
    bench_cultural_adaptation,
    bench_consistency_manager,
    bench_custom_emotions,
    bench_realtime_audio_analysis,
    bench_simd_operations,
    bench_spectral_processing
);

criterion_main!(benches);

//! Benchmarks for Phase 3: Advanced Streaming Synthesis
//!
//! Measures performance of:
//! - Lock-free ring buffer operations
//! - Predictive synthesis cache
//! - Adaptive quality scaling
//! - End-to-end streaming pipeline

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use voirs_singing::streaming::*;
use voirs_singing::types::{Articulation, Dynamics, Expression, NoteEvent};
use voirs_singing::{MusicalNote, MusicalScore, VoiceCharacteristics};

// Helper function to create test notes
fn create_test_note(frequency: f32, duration: f32, start_time: f32) -> MusicalNote {
    MusicalNote {
        event: NoteEvent {
            note: "A".to_string(),
            octave: 4,
            frequency,
            duration,
            velocity: 0.8,
            vibrato: 0.5,
            lyric: Some("a".to_string()),
            phonemes: vec!["a".to_string()],
            expression: Expression::Neutral,
            timing_offset: 0.0,
            breath_before: 0.0,
            legato: false,
            articulation: Articulation::Normal,
        },
        start_time,
        duration,
        pitch_bend: None,
        articulation: Articulation::Normal,
        dynamics: Dynamics::MezzoForte,
        tie_next: false,
        tie_prev: false,
        tuplet: None,
        ornaments: Vec::new(),
        chord: None,
    }
}

// Helper function to create test score
fn create_test_score(num_notes: usize) -> MusicalScore {
    let mut score = MusicalScore::new("Benchmark".to_string(), "Test".to_string());

    for i in 0..num_notes {
        let frequency = 440.0 + (i as f32 * 10.0);
        let start_time = i as f32 * 0.5;
        score
            .notes
            .push(create_test_note(frequency, 0.5, start_time));
    }

    score
}

/// Benchmark lock-free ring buffer write operations
fn bench_ring_buffer_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("ring_buffer_write");

    for size in [256, 1024, 4096, 16384].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let buffer = LockFreeRingBuffer::new(size * 2);
            let samples: Vec<f32> = (0..size).map(|i| i as f32 / size as f32).collect();

            b.iter(|| buffer.write(black_box(&samples)));
        });
    }

    group.finish();
}

/// Benchmark lock-free ring buffer read operations
fn bench_ring_buffer_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("ring_buffer_read");

    for size in [256, 1024, 4096, 16384].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let buffer = LockFreeRingBuffer::new(size * 2);
            let samples: Vec<f32> = (0..size).map(|i| i as f32 / size as f32).collect();
            buffer.write(&samples);

            let mut output = vec![0.0; size];

            b.iter(|| buffer.read(black_box(&mut output)));
        });
    }

    group.finish();
}

/// Benchmark concurrent ring buffer operations
fn bench_ring_buffer_concurrent(c: &mut Criterion) {
    let mut group = c.benchmark_group("ring_buffer_concurrent");

    for size in [1024, 4096, 16384].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                let buffer = LockFreeRingBuffer::new(size * 2);
                let buffer_clone = buffer.clone();

                let writer = std::thread::spawn(move || {
                    let samples: Vec<f32> = (0..size).map(|i| i as f32 / size as f32).collect();
                    buffer_clone.write(&samples)
                });

                let reader = std::thread::spawn(move || {
                    let mut output = vec![0.0; size];
                    while buffer.available_samples() < size {
                        std::hint::spin_loop();
                    }
                    buffer.read(&mut output)
                });

                let written = writer.join().unwrap();
                let read = reader.join().unwrap();
                (written, read)
            });
        });
    }

    group.finish();
}

/// Benchmark predictive synthesis cache operations
fn bench_predictive_synthesis(c: &mut Criterion) {
    let mut group = c.benchmark_group("predictive_synthesis");

    for num_notes in [10, 50, 100, 200].iter() {
        group.throughput(Throughput::Elements(*num_notes as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(num_notes),
            num_notes,
            |b, &num_notes| {
                let engine = PredictiveSynthesisEngine::new(2.0, 500);
                let score = create_test_score(num_notes);
                let voice = VoiceCharacteristics::default();

                b.iter(|| {
                    let upcoming = engine.analyze_upcoming_notes(black_box(&score), 0.0);
                    engine.prewarm_cache(black_box(&upcoming), "voice1", black_box(&voice))
                });
            },
        );
    }

    group.finish();
}

/// Benchmark predictive synthesis cache hits
fn bench_predictive_cache_hit(c: &mut Criterion) {
    let mut group = c.benchmark_group("predictive_cache_hit");

    let engine = PredictiveSynthesisEngine::new(2.0, 500);
    let note = create_test_note(440.0, 0.5, 0.0);
    let samples = vec![0.1; 24000];

    engine.cache_synthesized_note(&note, "voice1", samples);

    group.bench_function("cache_hit", |b| {
        b.iter(|| engine.get_cached_note(black_box(&note), "voice1"));
    });

    group.finish();
}

/// Benchmark adaptive quality scaling decisions
fn bench_adaptive_quality(c: &mut Criterion) {
    let mut group = c.benchmark_group("adaptive_quality");

    let scaler = AdaptiveQualityScaler::new(0.75, QualityLevel::Minimum, QualityLevel::Maximum);

    group.bench_function("quality_decision", |b| {
        b.iter(|| {
            scaler.update(
                black_box(std::time::Duration::from_millis(8)),
                black_box(std::time::Duration::from_millis(10)),
            )
        });
    });

    group.finish();
}

/// Benchmark CPU monitoring
fn bench_cpu_monitor(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_monitor");

    let monitor = CpuMonitor::new(0.2, std::time::Duration::from_millis(100));

    group.bench_function("cpu_record", |b| {
        b.iter(|| {
            monitor.record_synthesis_time(
                black_box(std::time::Duration::from_millis(7)),
                black_box(std::time::Duration::from_millis(10)),
            )
        });
    });

    group.finish();
}

/// Benchmark end-to-end streaming pipeline
fn bench_streaming_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_pipeline");

    for num_notes in [10, 50, 100].iter() {
        group.throughput(Throughput::Elements(*num_notes as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(num_notes),
            num_notes,
            |b, &num_notes| {
                let config = AdvancedStreamingConfig::default();
                let pipeline = ZeroCopyStreamingPipeline::new(config);
                let score = create_test_score(num_notes);
                let voice = VoiceCharacteristics::default();

                pipeline
                    .start_streaming(score, voice, "voice1".to_string())
                    .unwrap();

                let mut output = vec![0.0; 1024];

                b.iter(|| pipeline.read_samples(black_box(&mut output)));
            },
        );
    }

    group.finish();
}

/// Benchmark pipeline initialization
fn bench_pipeline_startup(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline_startup");

    for num_notes in [10, 50, 100].iter() {
        group.throughput(Throughput::Elements(*num_notes as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(num_notes),
            num_notes,
            |b, &num_notes| {
                let config = AdvancedStreamingConfig::default();
                let score = create_test_score(num_notes);
                let voice = VoiceCharacteristics::default();

                b.iter(|| {
                    let pipeline = ZeroCopyStreamingPipeline::new(config.clone());
                    pipeline.start_streaming(
                        black_box(score.clone()),
                        black_box(voice.clone()),
                        black_box("voice1".to_string()),
                    )
                });
            },
        );
    }

    group.finish();
}

/// Benchmark streaming latency
fn bench_streaming_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_latency");
    group.sample_size(50);

    let config = AdvancedStreamingConfig {
        target_latency_ms: 10,
        enable_predictive: true,
        enable_adaptive_quality: true,
        ..Default::default()
    };

    let pipeline = ZeroCopyStreamingPipeline::new(config);
    let score = create_test_score(100);
    let voice = VoiceCharacteristics::default();

    pipeline
        .start_streaming(score, voice, "voice1".to_string())
        .unwrap();

    group.bench_function("read_latency", |b| {
        let mut output = vec![0.0; 256]; // Small buffer for low latency

        b.iter(|| pipeline.read_samples(black_box(&mut output)));
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_ring_buffer_write,
    bench_ring_buffer_read,
    bench_ring_buffer_concurrent,
    bench_predictive_synthesis,
    bench_predictive_cache_hit,
    bench_adaptive_quality,
    bench_cpu_monitor,
    bench_streaming_pipeline,
    bench_pipeline_startup,
    bench_streaming_latency,
);

criterion_main!(benches);

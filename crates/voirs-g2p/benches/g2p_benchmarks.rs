//! Performance benchmarks for G2P conversion

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use tokio::runtime::Runtime;
use voirs_g2p::{
    G2p, LanguageCode, DummyG2p, Result,
    rules::EnglishRuleG2p,
    backends::{
        rule_based::RuleBasedG2p,
        hybrid::HybridG2p,
    },
    preprocessing::TextPreprocessor,
    detection::LanguageDetector,
};
use std::time::Duration;

// Benchmark data
const SHORT_TEXTS: &[&str] = &[
    "Hello",
    "world",
    "test",
    "benchmark",
];

const MEDIUM_TEXTS: &[&str] = &[
    "Hello world",
    "This is a test",
    "Performance benchmark",
    "VoiRS speech synthesis",
];

const LONG_TEXTS: &[&str] = &[
    "The quick brown fox jumps over the lazy dog",
    "This is a longer sentence for benchmarking purposes",
    "VoiRS is a speech synthesis system written in Rust for high performance",
    "Grapheme-to-phoneme conversion is an essential component of text-to-speech systems",
];

const VERY_LONG_TEXTS: &[&str] = &[
    "The quick brown fox jumps over the lazy dog. This pangram contains every letter of the alphabet at least once. It's commonly used for testing typewriters, computer keyboards, and fonts. The phrase is also used in telecommunications and typing training. VoiRS speech synthesis system uses this text for performance benchmarking of grapheme-to-phoneme conversion algorithms.",
];

fn benchmark_dummy_g2p(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let g2p = DummyG2p::new();
    
    let mut group = c.benchmark_group("dummy_g2p");
    group.measurement_time(Duration::from_secs(10));
    
    // Short texts
    for (i, text) in SHORT_TEXTS.iter().enumerate() {
        group.bench_with_input(BenchmarkId::new("short", i), text, |b, text| {
            b.to_async(&rt).iter(|| async {
                let result = g2p.to_phonemes(black_box(text), Some(LanguageCode::EnUs)).await;
                black_box(result.unwrap())
            });
        });
    }
    
    // Medium texts
    for (i, text) in MEDIUM_TEXTS.iter().enumerate() {
        group.bench_with_input(BenchmarkId::new("medium", i), text, |b, text| {
            b.to_async(&rt).iter(|| async {
                let result = g2p.to_phonemes(black_box(text), Some(LanguageCode::EnUs)).await;
                black_box(result.unwrap())
            });
        });
    }
    
    // Long texts
    for (i, text) in LONG_TEXTS.iter().enumerate() {
        group.bench_with_input(BenchmarkId::new("long", i), text, |b, text| {
            b.to_async(&rt).iter(|| async {
                let result = g2p.to_phonemes(black_box(text), Some(LanguageCode::EnUs)).await;
                black_box(result.unwrap())
            });
        });
    }
    
    group.finish();
}

fn benchmark_rule_based_g2p(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let g2p = rt.block_on(async { EnglishRuleG2p::new().unwrap() });
    
    let mut group = c.benchmark_group("rule_based_g2p");
    group.measurement_time(Duration::from_secs(10));
    
    // Short texts
    for (i, text) in SHORT_TEXTS.iter().enumerate() {
        group.bench_with_input(BenchmarkId::new("short", i), text, |b, text| {
            b.to_async(&rt).iter(|| async {
                let result = g2p.to_phonemes(black_box(text), Some(LanguageCode::EnUs)).await;
                black_box(result.unwrap())
            });
        });
    }
    
    // Medium texts
    for (i, text) in MEDIUM_TEXTS.iter().enumerate() {
        group.bench_with_input(BenchmarkId::new("medium", i), text, |b, text| {
            b.to_async(&rt).iter(|| async {
                let result = g2p.to_phonemes(black_box(text), Some(LanguageCode::EnUs)).await;
                black_box(result.unwrap())
            });
        });
    }
    
    // Long texts
    for (i, text) in LONG_TEXTS.iter().enumerate() {
        group.bench_with_input(BenchmarkId::new("long", i), text, |b, text| {
            b.to_async(&rt).iter(|| async {
                let result = g2p.to_phonemes(black_box(text), Some(LanguageCode::EnUs)).await;
                black_box(result.unwrap())
            });
        });
    }
    
    group.finish();
}

fn benchmark_hybrid_g2p(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let g2p = rt.block_on(async {
        let mut hybrid = HybridG2p::new().unwrap();
        hybrid.add_backend("rule_based".to_string(), Box::new(RuleBasedG2p::new().unwrap()), 1.0).unwrap();
        hybrid.add_backend("dummy".to_string(), Box::new(DummyG2p::new()), 0.5).unwrap();
        hybrid
    });
    
    let mut group = c.benchmark_group("hybrid_g2p");
    group.measurement_time(Duration::from_secs(10));
    
    // Test different text lengths
    for (i, text) in MEDIUM_TEXTS.iter().enumerate() {
        group.bench_with_input(BenchmarkId::new("medium", i), text, |b, text| {
            b.to_async(&rt).iter(|| async {
                let result = g2p.to_phonemes(black_box(text), Some(LanguageCode::EnUs)).await;
                black_box(result.unwrap())
            });
        });
    }
    
    group.finish();
}

fn benchmark_preprocessing(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let preprocessor = TextPreprocessor::default();
    
    let mut group = c.benchmark_group("preprocessing");
    group.measurement_time(Duration::from_secs(10));
    
    let preprocessing_texts = &[
        "Dr. Smith lives at 123 Main St.",
        "The price is $19.99 USD at 3:30 PM.",
        "Call (555) 123-4567 for more info.",
        "Visit https://example.com on Jan 1st, 2024.",
    ];
    
    for (i, text) in preprocessing_texts.iter().enumerate() {
        group.bench_with_input(BenchmarkId::new("complex", i), text, |b, text| {
            b.to_async(&rt).iter(|| async {
                let result = preprocessor.process(black_box(text), Some(LanguageCode::EnUs)).await;
                black_box(result.unwrap())
            });
        });
    }
    
    group.finish();
}

fn benchmark_language_detection(c: &mut Criterion) {
    let detector = LanguageDetector::default();
    
    let mut group = c.benchmark_group("language_detection");
    group.measurement_time(Duration::from_secs(10));
    
    let detection_texts = &[
        "Hello world",
        "Hallo Welt",
        "Bonjour le monde",
        "Hola mundo",
        "こんにちは世界",
        "Hello world mixed with 中文 text",
    ];
    
    for (i, text) in detection_texts.iter().enumerate() {
        group.bench_with_input(BenchmarkId::new("mixed", i), text, |b, text| {
            b.iter(|| {
                let result = detector.detect(black_box(text));
                black_box(result)
            });
        });
    }
    
    group.finish();
}

fn benchmark_batch_processing(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let g2p = DummyG2p::new();
    
    let mut group = c.benchmark_group("batch_processing");
    group.measurement_time(Duration::from_secs(15));
    
    // Test different batch sizes
    let batch_sizes = vec![1, 10, 50, 100];
    
    for batch_size in batch_sizes {
        let texts: Vec<&str> = MEDIUM_TEXTS.iter().cycle().take(batch_size).copied().collect();
        
        group.bench_with_input(BenchmarkId::new("batch", batch_size), &texts, |b, texts| {
            b.to_async(&rt).iter(|| async {
                let mut results = Vec::new();
                for text in texts {
                    let result = g2p.to_phonemes(black_box(text), Some(LanguageCode::EnUs)).await.unwrap();
                    results.push(result);
                }
                black_box(results)
            });
        });
    }
    
    group.finish();
}

fn benchmark_memory_usage(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let g2p = DummyG2p::new();
    
    let mut group = c.benchmark_group("memory_usage");
    group.measurement_time(Duration::from_secs(10));
    
    // Test with very long texts to measure memory allocation
    for (i, text) in VERY_LONG_TEXTS.iter().enumerate() {
        group.bench_with_input(BenchmarkId::new("very_long", i), text, |b, text| {
            b.to_async(&rt).iter(|| async {
                let result = g2p.to_phonemes(black_box(text), Some(LanguageCode::EnUs)).await;
                black_box(result.unwrap())
            });
        });
    }
    
    group.finish();
}

fn benchmark_throughput(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let g2p = DummyG2p::new();
    
    let mut group = c.benchmark_group("throughput");
    group.measurement_time(Duration::from_secs(20));
    group.sample_size(50);
    
    // Test characters per second throughput
    let test_text = "The quick brown fox jumps over the lazy dog. ".repeat(100);
    let char_count = test_text.len();
    
    group.bench_function("chars_per_second", |b| {
        b.to_async(&rt).iter(|| async {
            let result = g2p.to_phonemes(black_box(&test_text), Some(LanguageCode::EnUs)).await;
            black_box(result.unwrap())
        });
    });
    
    group.finish();
    
    println!("Benchmark text length: {} characters", char_count);
}

criterion_group!(
    benches,
    benchmark_dummy_g2p,
    benchmark_rule_based_g2p,
    benchmark_hybrid_g2p,
    benchmark_preprocessing,
    benchmark_language_detection,
    benchmark_batch_processing,
    benchmark_memory_usage,
    benchmark_throughput,
);

criterion_main!(benches);
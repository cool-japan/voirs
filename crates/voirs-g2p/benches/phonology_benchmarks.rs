//! Performance benchmarks for phonology module.
//!
//! Measures the performance of various phonological processes and rule applications.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use voirs_g2p::phonology::{
    rules::{AssimilationRule, ReductionRule, RuleSet},
    PhonologicalProcessor, ProcessConfig, ProcessType,
};
use voirs_g2p::{LanguageCode, Phoneme};

/// Create a test phoneme sequence of varying lengths
fn create_test_sequence(length: usize) -> Vec<Phoneme> {
    let phonemes = ["k", "æ", "t", "ɪ", "n", "p", "ʊ", "t", "ə", "b", "aʊ", "t"];
    (0..length)
        .map(|i| Phoneme::new(phonemes[i % phonemes.len()].to_string()))
        .collect()
}

/// Create a sequence with assimilation opportunities
fn create_assimilation_sequence(count: usize) -> Vec<Phoneme> {
    let mut result = Vec::new();
    for _ in 0..count {
        result.push(Phoneme::new("ɪ".to_string()));
        result.push(Phoneme::new("n".to_string()));
        result.push(Phoneme::new("p".to_string()));
        result.push(Phoneme::new("ʊ".to_string()));
        result.push(Phoneme::new("t".to_string()));
    }
    result
}

/// Benchmark place assimilation process
fn bench_place_assimilation(c: &mut Criterion) {
    let mut group = c.benchmark_group("phonology/place_assimilation");

    let processor = PhonologicalProcessor::new(LanguageCode::EnUs);

    for size in [1, 5, 10, 20].iter() {
        let sequence = create_assimilation_sequence(*size);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_words", size)),
            &sequence,
            |b, seq| b.iter(|| black_box(processor.apply_all_processes(seq).unwrap())),
        );
    }

    group.finish();
}

/// Benchmark vowel reduction process
fn bench_vowel_reduction(c: &mut Criterion) {
    let mut group = c.benchmark_group("phonology/vowel_reduction");

    let config = ProcessConfig {
        enable_place_assimilation: false,
        enable_vowel_reduction: true,
        aggressiveness: 0.8,
        ..Default::default()
    };
    let processor = PhonologicalProcessor::with_config(config);

    for size in [10, 50, 100].iter() {
        let mut sequence = create_test_sequence(*size);
        // Mark alternating vowels as unstressed
        for (i, phoneme) in sequence.iter_mut().enumerate() {
            if i % 2 == 0 {
                phoneme.stress = 0;
            }
        }

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_phonemes", size)),
            &sequence,
            |b, seq| b.iter(|| black_box(processor.apply_all_processes(seq).unwrap())),
        );
    }

    group.finish();
}

/// Benchmark complete phonological processor
fn bench_full_processor(c: &mut Criterion) {
    let mut group = c.benchmark_group("phonology/full_processor");

    let config = ProcessConfig {
        enable_place_assimilation: true,
        enable_vowel_reduction: true,
        enable_voicing_assimilation: true,
        enable_elision: true,
        enable_liaison: true,
        aggressiveness: 0.7,
        ..Default::default()
    };
    let processor = PhonologicalProcessor::with_config(config);

    for size in [10, 50, 100, 200].iter() {
        let sequence = create_test_sequence(*size);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_phonemes", size)),
            &sequence,
            |b, seq| b.iter(|| black_box(processor.apply_all_processes(seq).unwrap())),
        );
    }

    group.finish();
}

/// Benchmark specific process application
fn bench_process_types(c: &mut Criterion) {
    let mut group = c.benchmark_group("phonology/process_types");

    let sequence = create_assimilation_sequence(10);

    let process_types = vec![
        ("place_assimilation", ProcessType::PlaceAssimilation),
        ("vowel_reduction", ProcessType::VowelReduction),
        ("voicing_assimilation", ProcessType::VoicingAssimilation),
        ("elision", ProcessType::Elision),
        ("liaison", ProcessType::Liaison),
    ];

    for (name, process_type) in process_types {
        let processor = PhonologicalProcessor::new(LanguageCode::EnUs);
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(sequence.clone(), process_type),
            |b, (seq, pt)| b.iter(|| black_box(processor.apply_process(seq, *pt).unwrap())),
        );
    }

    group.finish();
}

/// Benchmark rule-based system
fn bench_rule_application(c: &mut Criterion) {
    let mut group = c.benchmark_group("phonology/rule_application");

    let ruleset = RuleSet::for_language(LanguageCode::EnUs);

    for size in [10, 50, 100].iter() {
        let sequence = create_assimilation_sequence(*size / 5);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_phonemes", size)),
            &sequence,
            |b, seq| b.iter(|| black_box(ruleset.apply_rules(seq).unwrap())),
        );
    }

    group.finish();
}

/// Benchmark single rule matching
fn bench_rule_matching(c: &mut Criterion) {
    let mut group = c.benchmark_group("phonology/rule_matching");

    let rule = AssimilationRule::nasal_before_bilabial();
    let sequence = create_assimilation_sequence(10);

    group.bench_function("single_rule_match", |b| {
        b.iter(|| {
            for i in 0..sequence.len() {
                black_box(rule.matches(&sequence, i));
            }
        })
    });

    group.finish();
}

/// Benchmark processor creation
fn bench_processor_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("phonology/processor_creation");

    group.bench_function("default_processor", |b| {
        b.iter(|| black_box(PhonologicalProcessor::new(LanguageCode::EnUs)))
    });

    group.bench_function("custom_config", |b| {
        b.iter(|| {
            let config = ProcessConfig {
                enable_place_assimilation: true,
                enable_vowel_reduction: true,
                aggressiveness: 0.7,
                ..Default::default()
            };
            black_box(PhonologicalProcessor::with_config(config))
        })
    });

    group.bench_function("ruleset_creation", |b| {
        b.iter(|| black_box(RuleSet::for_language(LanguageCode::EnUs)))
    });

    group.finish();
}

/// Benchmark phoneme feature extraction
fn bench_feature_extraction(c: &mut Criterion) {
    use voirs_g2p::phonology::{get_features, has_feature, PhonologicalFeature};

    let mut group = c.benchmark_group("phonology/feature_extraction");

    let phonemes = [
        "p", "b", "t", "d", "k", "g", "m", "n", "ŋ", "s", "z", "ʃ", "ʒ", "i", "ɪ", "e", "ɛ", "æ",
        "ə", "ʌ", "a", "ɑ", "ɔ", "o", "ʊ", "u",
    ];

    group.bench_function("get_features", |b| {
        b.iter(|| {
            for phoneme in &phonemes {
                black_box(get_features(phoneme));
            }
        })
    });

    group.bench_function("has_feature_check", |b| {
        b.iter(|| {
            for phoneme in &phonemes {
                black_box(has_feature(phoneme, PhonologicalFeature::Voiced));
                black_box(has_feature(phoneme, PhonologicalFeature::Nasal));
                black_box(has_feature(phoneme, PhonologicalFeature::Vowel));
            }
        })
    });

    group.finish();
}

/// Benchmark realistic speech synthesis pipeline usage
fn bench_realistic_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("phonology/realistic_pipeline");

    let processor = PhonologicalProcessor::new(LanguageCode::EnUs);

    // Simulate typical sentence lengths
    let sentences = vec![
        ("short", create_test_sequence(10)),      // ~2 words
        ("medium", create_test_sequence(30)),     // ~6 words
        ("long", create_test_sequence(60)),       // ~12 words
        ("very_long", create_test_sequence(100)), // ~20 words
    ];

    for (name, sequence) in sentences {
        group.bench_with_input(BenchmarkId::from_parameter(name), &sequence, |b, seq| {
            b.iter(|| black_box(processor.apply_all_processes(seq).unwrap()))
        });
    }

    group.finish();
}

/// Benchmark nasalization process
fn bench_nasalization(c: &mut Criterion) {
    let mut group = c.benchmark_group("phonology/nasalization");

    let config = ProcessConfig {
        enable_nasalization: true,
        enable_place_assimilation: false,
        enable_vowel_reduction: false,
        language: LanguageCode::Fr,
        aggressiveness: 0.5,
        ..Default::default()
    };
    let processor = PhonologicalProcessor::with_config(config);

    // Create sequences with vowels followed by nasal consonants
    for size in [10, 50, 100, 200].iter() {
        let mut sequence = Vec::new();
        for _ in 0..*size {
            sequence.push(Phoneme::new("a".to_string()));
            sequence.push(Phoneme::new("n".to_string()));
        }

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_vowel_nasal_pairs", size)),
            &sequence,
            |b, seq| b.iter(|| black_box(processor.apply_all_processes(seq).unwrap())),
        );
    }

    group.finish();
}

/// Benchmark palatalization process
fn bench_palatalization(c: &mut Criterion) {
    let mut group = c.benchmark_group("phonology/palatalization");

    let config = ProcessConfig {
        enable_palatalization: true,
        enable_place_assimilation: false,
        enable_vowel_reduction: false,
        language: LanguageCode::Ja,
        aggressiveness: 0.7,
        ..Default::default()
    };
    let processor = PhonologicalProcessor::with_config(config);

    // Create sequences with consonants before front vowels
    for size in [10, 50, 100, 200].iter() {
        let mut sequence = Vec::new();
        for _ in 0..*size {
            sequence.push(Phoneme::new("t".to_string()));
            sequence.push(Phoneme::new("i".to_string()));
            sequence.push(Phoneme::new("s".to_string()));
            sequence.push(Phoneme::new("i".to_string()));
        }

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_palatalization_contexts", size)),
            &sequence,
            |b, seq| b.iter(|| black_box(processor.apply_all_processes(seq).unwrap())),
        );
    }

    group.finish();
}

/// Benchmark lenition process
fn bench_lenition(c: &mut Criterion) {
    let mut group = c.benchmark_group("phonology/lenition");

    let config = ProcessConfig {
        enable_lenition: true,
        enable_place_assimilation: false,
        enable_vowel_reduction: false,
        language: LanguageCode::Es,
        aggressiveness: 0.6,
        ..Default::default()
    };
    let processor = PhonologicalProcessor::with_config(config);

    // Create intervocalic sequences (vowel-consonant-vowel)
    for size in [10, 50, 100, 200].iter() {
        let mut sequence = Vec::new();
        for _ in 0..*size {
            sequence.push(Phoneme::new("a".to_string()));
            sequence.push(Phoneme::new("b".to_string()));
            sequence.push(Phoneme::new("o".to_string()));
            sequence.push(Phoneme::new("d".to_string()));
            sequence.push(Phoneme::new("a".to_string()));
        }

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_intervocalic_contexts", size)),
            &sequence,
            |b, seq| b.iter(|| black_box(processor.apply_all_processes(seq).unwrap())),
        );
    }

    group.finish();
}

/// Benchmark fortition process
fn bench_fortition(c: &mut Criterion) {
    let mut group = c.benchmark_group("phonology/fortition");

    let config = ProcessConfig {
        enable_fortition: true,
        enable_place_assimilation: false,
        enable_vowel_reduction: false,
        language: LanguageCode::It,
        aggressiveness: 0.7,
        ..Default::default()
    };
    let processor = PhonologicalProcessor::with_config(config);

    // Create sequences with stressed vowels followed by consonants
    for size in [10, 50, 100, 200].iter() {
        let mut sequence = Vec::new();
        for _ in 0..*size {
            let mut vowel1 = Phoneme::new("a".to_string());
            vowel1.stress = 1; // Mark as stressed
            sequence.push(vowel1);
            sequence.push(Phoneme::new("t".to_string()));
            sequence.push(Phoneme::new("o".to_string()));
        }

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_fortition_contexts", size)),
            &sequence,
            |b, seq| b.iter(|| black_box(processor.apply_all_processes(seq).unwrap())),
        );
    }

    group.finish();
}

/// Benchmark all new processes combined
fn bench_all_new_processes(c: &mut Criterion) {
    let mut group = c.benchmark_group("phonology/all_new_processes");

    let config = ProcessConfig {
        enable_nasalization: true,
        enable_palatalization: true,
        enable_lenition: true,
        enable_fortition: true,
        enable_place_assimilation: false,
        enable_vowel_reduction: false,
        language: LanguageCode::Es,
        aggressiveness: 0.7,
        ..Default::default()
    };
    let processor = PhonologicalProcessor::with_config(config);

    for size in [10, 50, 100, 200].iter() {
        let sequence = create_test_sequence(*size);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_phonemes", size)),
            &sequence,
            |b, seq| b.iter(|| black_box(processor.apply_all_processes(seq).unwrap())),
        );
    }

    group.finish();
}

/// Benchmark H-dropping process
fn bench_h_dropping(c: &mut Criterion) {
    let mut group = c.benchmark_group("phonology/h_dropping");

    let config = ProcessConfig {
        enable_h_dropping: true,
        enable_place_assimilation: false,
        language: LanguageCode::EnGb,
        aggressiveness: 0.5,
        ..Default::default()
    };
    let processor = PhonologicalProcessor::with_config(config);

    // Create sequences with /h/ phonemes
    for size in [10, 50, 100, 200].iter() {
        let mut sequence = Vec::new();
        for _ in 0..*size {
            sequence.push(Phoneme::new("h".to_string()));
            sequence.push(Phoneme::new("aʊ".to_string()));
            sequence.push(Phoneme::new("s".to_string()));
        }

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_h_phonemes", size)),
            &sequence,
            |b, seq| b.iter(|| black_box(processor.apply_all_processes(seq).unwrap())),
        );
    }

    group.finish();
}

/// Benchmark TH-fronting process
fn bench_th_fronting(c: &mut Criterion) {
    let mut group = c.benchmark_group("phonology/th_fronting");

    let config = ProcessConfig {
        enable_th_fronting: true,
        enable_place_assimilation: false,
        language: LanguageCode::EnGb,
        aggressiveness: 0.6,
        ..Default::default()
    };
    let processor = PhonologicalProcessor::with_config(config);

    // Create sequences with dental fricatives
    for size in [10, 50, 100, 200].iter() {
        let mut sequence = Vec::new();
        for _ in 0..*size {
            sequence.push(Phoneme::new("θ".to_string()));
            sequence.push(Phoneme::new("ɪ".to_string()));
            sequence.push(Phoneme::new("ŋ".to_string()));
            sequence.push(Phoneme::new("ð".to_string()));
            sequence.push(Phoneme::new("ɪ".to_string()));
            sequence.push(Phoneme::new("s".to_string()));
        }

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_th_phonemes", size)),
            &sequence,
            |b, seq| b.iter(|| black_box(processor.apply_all_processes(seq).unwrap())),
        );
    }

    group.finish();
}

/// Benchmark L-vocalization process
fn bench_l_vocalization(c: &mut Criterion) {
    let mut group = c.benchmark_group("phonology/l_vocalization");

    let config = ProcessConfig {
        enable_l_vocalization: true,
        enable_place_assimilation: false,
        language: LanguageCode::EnGb,
        aggressiveness: 0.6,
        ..Default::default()
    };
    let processor = PhonologicalProcessor::with_config(config);

    // Create sequences with /l/ in coda position
    for size in [10, 50, 100, 200].iter() {
        let mut sequence = Vec::new();
        for _ in 0..*size {
            sequence.push(Phoneme::new("m".to_string()));
            sequence.push(Phoneme::new("ɪ".to_string()));
            sequence.push(Phoneme::new("l".to_string()));
            sequence.push(Phoneme::new("k".to_string()));
        }

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_l_coda_contexts", size)),
            &sequence,
            |b, seq| b.iter(|| black_box(processor.apply_all_processes(seq).unwrap())),
        );
    }

    group.finish();
}

/// Benchmark G-dropping process
fn bench_g_dropping(c: &mut Criterion) {
    let mut group = c.benchmark_group("phonology/g_dropping");

    let config = ProcessConfig {
        enable_g_dropping: true,
        enable_place_assimilation: false,
        language: LanguageCode::EnUs,
        aggressiveness: 0.7,
        ..Default::default()
    };
    let processor = PhonologicalProcessor::with_config(config);

    // Create sequences with /ŋ/ in unstressed syllables
    for size in [10, 50, 100, 200].iter() {
        let mut sequence = Vec::new();
        for _ in 0..*size {
            sequence.push(Phoneme::new("r".to_string()));
            sequence.push(Phoneme::new("ʌ".to_string()));
            sequence.push(Phoneme::new("n".to_string()));
            sequence.push(Phoneme::new("ɪ".to_string()));
            let mut ng = Phoneme::new("ŋ".to_string());
            ng.stress = 0; // Unstressed
            sequence.push(ng);
        }

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_g_dropping_contexts", size)),
            &sequence,
            |b, seq| b.iter(|| black_box(processor.apply_all_processes(seq).unwrap())),
        );
    }

    group.finish();
}

/// Benchmark T-glottaling process
fn bench_t_glottaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("phonology/t_glottaling");

    let config = ProcessConfig {
        enable_t_glottaling: true,
        enable_place_assimilation: false,
        language: LanguageCode::EnGb,
        aggressiveness: 0.6,
        ..Default::default()
    };
    let processor = PhonologicalProcessor::with_config(config);

    // Create sequences with /t/ in coda position
    for size in [10, 50, 100, 200].iter() {
        let mut sequence = Vec::new();
        for _ in 0..*size {
            sequence.push(Phoneme::new("b".to_string()));
            sequence.push(Phoneme::new("ʌ".to_string()));
            sequence.push(Phoneme::new("t".to_string()));
            sequence.push(Phoneme::new("ə".to_string()));
        }

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_t_glottaling_contexts", size)),
            &sequence,
            |b, seq| b.iter(|| black_box(processor.apply_all_processes(seq).unwrap())),
        );
    }

    group.finish();
}

/// Benchmark Yod-coalescence process
fn bench_yod_coalescence(c: &mut Criterion) {
    let mut group = c.benchmark_group("phonology/yod_coalescence");

    let config = ProcessConfig {
        enable_yod_coalescence: true,
        enable_place_assimilation: false,
        language: LanguageCode::EnUs,
        aggressiveness: 0.5,
        ..Default::default()
    };
    let processor = PhonologicalProcessor::with_config(config);

    // Create sequences with /tj/ and /dj/ clusters
    for size in [10, 50, 100, 200].iter() {
        let mut sequence = Vec::new();
        for _ in 0..*size {
            sequence.push(Phoneme::new("t".to_string()));
            sequence.push(Phoneme::new("j".to_string()));
            sequence.push(Phoneme::new("uː".to_string()));
            sequence.push(Phoneme::new("n".to_string()));
            sequence.push(Phoneme::new("d".to_string()));
            sequence.push(Phoneme::new("j".to_string()));
            sequence.push(Phoneme::new("uː".to_string()));
        }

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_yod_clusters", size)),
            &sequence,
            |b, seq| b.iter(|| black_box(processor.apply_all_processes(seq).unwrap())),
        );
    }

    group.finish();
}

/// Benchmark all dialect-specific processes combined
fn bench_all_dialect_processes(c: &mut Criterion) {
    let mut group = c.benchmark_group("phonology/all_dialect_processes");

    let config = ProcessConfig {
        enable_h_dropping: true,
        enable_th_fronting: true,
        enable_l_vocalization: true,
        enable_g_dropping: true,
        enable_t_glottaling: true,
        enable_yod_coalescence: false, // Different language
        enable_place_assimilation: false,
        language: LanguageCode::EnGb,
        aggressiveness: 0.6,
        ..Default::default()
    };
    let processor = PhonologicalProcessor::with_config(config);

    for size in [10, 50, 100, 200].iter() {
        let sequence = create_test_sequence(*size);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_phonemes", size)),
            &sequence,
            |b, seq| b.iter(|| black_box(processor.apply_all_processes(seq).unwrap())),
        );
    }

    group.finish();
}

/// Benchmark realistic English dialect synthesis
fn bench_realistic_dialect_synthesis(c: &mut Criterion) {
    let mut group = c.benchmark_group("phonology/realistic_dialect_synthesis");

    // British English with all typical processes
    let british_config = ProcessConfig {
        enable_place_assimilation: true,
        enable_h_dropping: true,
        enable_th_fronting: true,
        enable_l_vocalization: true,
        enable_t_glottaling: true,
        enable_vowel_reduction: true,
        language: LanguageCode::EnGb,
        aggressiveness: 0.6,
        ..Default::default()
    };
    let british_processor = PhonologicalProcessor::with_config(british_config);

    // American English with all typical processes
    let american_config = ProcessConfig {
        enable_place_assimilation: true,
        enable_t_flapping: true,
        enable_g_dropping: true,
        enable_yod_coalescence: true,
        enable_vowel_reduction: true,
        language: LanguageCode::EnUs,
        aggressiveness: 0.6,
        ..Default::default()
    };
    let american_processor = PhonologicalProcessor::with_config(american_config);

    // British English short
    group.bench_with_input(
        BenchmarkId::from_parameter("british_short"),
        &create_test_sequence(30),
        |b, seq| b.iter(|| black_box(british_processor.apply_all_processes(seq).unwrap())),
    );

    // British English long
    group.bench_with_input(
        BenchmarkId::from_parameter("british_long"),
        &create_test_sequence(100),
        |b, seq| b.iter(|| black_box(british_processor.apply_all_processes(seq).unwrap())),
    );

    // American English short
    group.bench_with_input(
        BenchmarkId::from_parameter("american_short"),
        &create_test_sequence(30),
        |b, seq| b.iter(|| black_box(american_processor.apply_all_processes(seq).unwrap())),
    );

    // American English long
    group.bench_with_input(
        BenchmarkId::from_parameter("american_long"),
        &create_test_sequence(100),
        |b, seq| b.iter(|| black_box(american_processor.apply_all_processes(seq).unwrap())),
    );

    group.finish();
}

criterion_group!(
    benches,
    bench_place_assimilation,
    bench_vowel_reduction,
    bench_full_processor,
    bench_process_types,
    bench_rule_application,
    bench_rule_matching,
    bench_processor_creation,
    bench_feature_extraction,
    bench_realistic_pipeline,
    bench_nasalization,
    bench_palatalization,
    bench_lenition,
    bench_fortition,
    bench_all_new_processes,
    bench_h_dropping,
    bench_th_fronting,
    bench_l_vocalization,
    bench_g_dropping,
    bench_t_glottaling,
    bench_yod_coalescence,
    bench_all_dialect_processes,
    bench_realistic_dialect_synthesis,
);

criterion_main!(benches);

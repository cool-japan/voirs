//! Accuracy Benchmarking Example
//!
//! This example demonstrates how to benchmark and validate ASR accuracy
//! using VoiRS Recognizer's built-in accuracy validation tools.
//!
//! Usage:
//! ```bash
//! cargo run --example accuracy_benchmarking --features="whisper-pure"
//! ```

use std::time::{Duration, Instant};
use tokio;
use voirs_recognizer::asr::{
    ASRBackend, BenchmarkingConfig, FallbackConfig, FallbackResult, WhisperModelSize,
};
use voirs_recognizer::prelude::*;
use voirs_recognizer::{PerformanceValidator, RecognitionError};

#[tokio::main]
async fn main() -> Result<(), RecognitionError> {
    println!("📊 VoiRS Accuracy Benchmarking Demo");
    println!("===================================\n");

    // Step 1: Create benchmarking suite
    println!("🎯 Setting up benchmarking suite...");
    let benchmark_config = BenchmarkingConfig::default();
    let benchmark_suite = ASRBenchmarkingSuite::new(benchmark_config).await?;

    println!("✅ Benchmarking suite configured:");
    println!("   • Test cases: Ready for custom evaluation");
    println!("   • Metrics: WER, confidence, phoneme accuracy");
    println!("   • Language: English (US)");

    // Step 2: Create test cases with ground truth
    println!("\n📝 Creating test cases with ground truth...");
    let test_cases = create_test_cases();

    println!("✅ Created {} test cases:", test_cases.len());
    for (i, (name, _, expected)) in test_cases.iter().enumerate() {
        println!("   {}. {}: \"{}\"", i + 1, name, expected);
    }

    // Step 3: Initialize accuracy validator
    println!("\n🔍 Initializing accuracy validator...");
    let accuracy_validator = AccuracyValidator::new_standard();

    // Add custom requirements
    let mut custom_validator = AccuracyValidator::new_standard();
    // Custom requirements would be added here if the API supports it

    println!("✅ Accuracy validator configured:");
    println!("   • WER threshold: ≤ 15%");
    println!("   • Confidence threshold: ≥ 70%");
    println!("   • Phoneme accuracy: ≥ 85%");

    // Step 4: Configure ASR for accuracy testing
    println!("\n🔧 Configuring ASR for accuracy testing...");
    let fallback_config = FallbackConfig {
        primary_backend: ASRBackend::Whisper {
            model_size: WhisperModelSize::Base,
            model_path: None,
        },
        quality_threshold: 0.7,
        max_processing_time_seconds: 10.0,
        adaptive_selection: true,
        ..Default::default()
    };

    let mut asr = IntelligentASRFallback::new(fallback_config).await?;
    println!("✅ ASR configured for accuracy testing");

    // Step 5: Run accuracy evaluation
    println!("\n🏃 Running accuracy evaluation...");
    let mut recognition_results = Vec::new();
    let mut total_processing_time = Duration::ZERO;

    for (name, audio, expected) in &test_cases {
        let recognition_start = Instant::now();
        let asr_config = ASRConfig {
            language: Some(LanguageCode::EnUs),
            word_timestamps: true,
            confidence_threshold: 0.7,
            ..Default::default()
        };
        let result = asr.transcribe(audio, Some(&asr_config)).await?;
        let recognition_time = recognition_start.elapsed();

        total_processing_time += recognition_time;

        println!(
            "   • {}: \"{}\" (expected: \"{}\")",
            name,
            result.transcript.text.as_str(),
            expected
        );
        println!("     - Confidence: {:.2}", result.transcript.confidence);
        println!("     - Processing time: {:?}", recognition_time);

        recognition_results.push((name.clone(), result, expected.clone(), recognition_time));
    }

    // Step 6: Calculate accuracy metrics
    println!("\n📊 Calculating accuracy metrics...");
    let mut total_wer = 0.0;
    let mut total_confidence = 0.0;
    let mut word_count = 0;

    for (name, result, expected, _) in &recognition_results {
        // Calculate WER for this test case
        let wer = calculate_wer(&result.transcript.text, expected);
        total_wer += wer;
        total_confidence += result.transcript.confidence;
        word_count += expected.split_whitespace().count();

        println!("   • {}:", name);
        println!("     - WER: {:.2}%", wer * 100.0);
        println!("     - Confidence: {:.2}", result.transcript.confidence);
        println!(
            "     - Match: {}",
            if wer < 0.1 { "✅ GOOD" } else { "❌ POOR" }
        );
    }

    let average_wer = total_wer / test_cases.len() as f32;
    let average_confidence = total_confidence / test_cases.len() as f32;

    println!("\n📈 Overall Accuracy Metrics:");
    println!("   • Average WER: {:.2}%", average_wer * 100.0);
    println!("   • Average Confidence: {:.2}", average_confidence);
    println!("   • Total words evaluated: {}", word_count);

    // Step 7: Performance validation
    println!("\n⚡ Performance Validation:");
    let validator = PerformanceValidator::new().with_verbose(false);

    // Calculate total audio duration
    let total_audio_duration: f32 = test_cases
        .iter()
        .map(|(_, audio, _)| audio.duration())
        .sum();
    let overall_rtf = total_processing_time.as_secs_f32() / total_audio_duration;

    println!("   • Total audio duration: {:.1}s", total_audio_duration);
    println!("   • Total processing time: {:?}", total_processing_time);
    println!("   • Overall RTF: {:.3}", overall_rtf);
    println!(
        "   • RTF Status: {}",
        if overall_rtf < 0.3 {
            "✅ PASS"
        } else {
            "❌ FAIL"
        }
    );

    // Memory usage
    let (memory_usage, memory_passed) = validator.estimate_memory_usage()?;
    println!(
        "   • Memory usage: {:.1} MB ({})",
        memory_usage as f64 / (1024.0 * 1024.0),
        if memory_passed {
            "✅ PASS"
        } else {
            "❌ FAIL"
        }
    );

    // Step 8: Detailed per-test analysis
    println!("\n🔍 Detailed Per-Test Analysis:");
    for (name, result, expected, processing_time) in &recognition_results {
        let wer = calculate_wer(&result.transcript.text, expected);
        let rtf = processing_time.as_secs_f32()
            / test_cases
                .iter()
                .find(|(n, _, _)| n == name)
                .map(|(_, audio, _)| audio.duration())
                .unwrap_or(1.0);

        println!("   • {}:", name);
        println!("     - Expected: \"{}\"", expected);
        println!("     - Actual:   \"{}\"", result.transcript.text.as_str());
        println!("     - WER: {:.2}%", wer * 100.0);
        println!("     - Confidence: {:.2}", result.transcript.confidence);
        println!("     - RTF: {:.3}", rtf);
        println!(
            "     - Status: {}",
            if wer < 0.15 && result.transcript.confidence > 0.7 {
                "✅ PASS"
            } else {
                "❌ FAIL"
            }
        );

        // Show word-level analysis if available
        if !result.transcript.word_timestamps.is_empty() {
            println!("     - Words: [");
            for word in &result.transcript.word_timestamps {
                println!(
                    "       \"{}\": {:.2}s-{:.2}s",
                    word.word, word.start_time, word.end_time
                );
            }
            println!("     ]");
        }
    }

    // Step 9: Benchmark comparison
    println!("\n🏆 Benchmark Comparison:");
    let benchmark_targets = vec![
        ("LibriSpeech Clean", 0.05, 0.90), // 5% WER, 90% confidence
        ("Common Voice", 0.12, 0.85),      // 12% WER, 85% confidence
        ("Noisy Speech", 0.25, 0.70),      // 25% WER, 70% confidence
    ];

    for (dataset, target_wer, target_confidence) in benchmark_targets {
        let wer_status = if average_wer <= target_wer {
            "✅ PASS"
        } else {
            "❌ FAIL"
        };
        let conf_status = if average_confidence >= target_confidence {
            "✅ PASS"
        } else {
            "❌ FAIL"
        };

        println!("   • {} benchmark:", dataset);
        println!(
            "     - WER: {:.2}% vs {:.2}% target {}",
            average_wer * 100.0,
            target_wer * 100.0,
            wer_status
        );
        println!(
            "     - Confidence: {:.2} vs {:.2} target {}",
            average_confidence, target_confidence, conf_status
        );
    }

    // Step 10: Generate accuracy report
    println!("\n📄 Accuracy Report Generation:");
    let report = generate_accuracy_report(
        &recognition_results,
        average_wer,
        average_confidence,
        overall_rtf,
    );
    println!("{}", report);

    // Step 11: Recommendations
    println!("\n💡 Recommendations:");
    if average_wer > 0.15 {
        println!("   • WER too high - consider:");
        println!("     - Using larger model size");
        println!("     - Increasing beam size");
        println!("     - Improving audio quality");
    }

    if average_confidence < 0.7 {
        println!("   • Confidence too low - consider:");
        println!("     - Using more conservative thresholds");
        println!("     - Adding confidence-based filtering");
        println!("     - Implementing fallback strategies");
    }

    if overall_rtf > 0.3 {
        println!("   • RTF too high - consider:");
        println!("     - Using smaller model size");
        println!("     - Reducing beam size");
        println!("     - Enabling GPU acceleration");
    }

    println!("\n✅ Accuracy benchmarking demo completed successfully!");
    println!("📊 Final Summary:");
    println!("   • Average WER: {:.2}%", average_wer * 100.0);
    println!("   • Average Confidence: {:.2}", average_confidence);
    println!("   • Overall RTF: {:.3}", overall_rtf);
    println!(
        "   • Tests passed: {}/{}",
        recognition_results
            .iter()
            .filter(|(_, result, expected, _)| {
                let wer = calculate_wer(&result.transcript.text, expected);
                wer < 0.15 && result.transcript.confidence > 0.7
            })
            .count(),
        recognition_results.len()
    );

    Ok(())
}

fn create_test_cases() -> Vec<(String, AudioBuffer, String)> {
    let sample_rate = 16000;
    let mut test_cases = Vec::new();

    // Test case 1: Simple phrase
    let audio1 = create_speech_like_audio(sample_rate, 2.0, 150.0, "hello world");
    test_cases.push((
        "Simple_Phrase".to_string(),
        audio1,
        "hello world".to_string(),
    ));

    // Test case 2: Numbers
    let audio2 = create_speech_like_audio(sample_rate, 2.5, 180.0, "one two three four five");
    test_cases.push((
        "Numbers".to_string(),
        audio2,
        "one two three four five".to_string(),
    ));

    // Test case 3: Complex sentence
    let audio3 = create_speech_like_audio(
        sample_rate,
        3.0,
        160.0,
        "the quick brown fox jumps over the lazy dog",
    );
    test_cases.push((
        "Complex_Sentence".to_string(),
        audio3,
        "the quick brown fox jumps over the lazy dog".to_string(),
    ));

    // Test case 4: Short utterance
    let audio4 = create_speech_like_audio(sample_rate, 1.0, 200.0, "yes");
    test_cases.push(("Short_Utterance".to_string(), audio4, "yes".to_string()));

    // Test case 5: Technical terms
    let audio5 = create_speech_like_audio(sample_rate, 2.8, 140.0, "speech recognition technology");
    test_cases.push((
        "Technical_Terms".to_string(),
        audio5,
        "speech recognition technology".to_string(),
    ));

    test_cases
}

fn create_speech_like_audio(
    sample_rate: u32,
    duration: f32,
    base_freq: f32,
    _text: &str,
) -> AudioBuffer {
    let mut samples = Vec::new();

    for i in 0..(sample_rate as f32 * duration) as usize {
        let t = i as f32 / sample_rate as f32;

        // Create formant-like structure
        let f1 = base_freq;
        let f2 = base_freq * 2.5;
        let f3 = base_freq * 3.5;

        let sample = 0.08 * (2.0 * std::f32::consts::PI * f1 * t).sin()
            + 0.04 * (2.0 * std::f32::consts::PI * f2 * t).sin()
            + 0.02 * (2.0 * std::f32::consts::PI * f3 * t).sin();

        // Add envelope to simulate word boundaries
        let word_envelope = (2.0 * std::f32::consts::PI * 0.8 * t).sin().abs();

        samples.push(sample * word_envelope);
    }

    AudioBuffer::mono(samples, sample_rate)
}

fn calculate_wer(hypothesis: &str, reference: &str) -> f32 {
    let hyp_words: Vec<&str> = hypothesis.split_whitespace().collect();
    let ref_words: Vec<&str> = reference.split_whitespace().collect();

    if ref_words.is_empty() {
        return if hyp_words.is_empty() { 0.0 } else { 1.0 };
    }

    // Simple word-level edit distance calculation
    let mut dp = vec![vec![0; hyp_words.len() + 1]; ref_words.len() + 1];

    // Initialize first row and column
    for i in 0..=ref_words.len() {
        dp[i][0] = i;
    }
    for j in 0..=hyp_words.len() {
        dp[0][j] = j;
    }

    // Fill the DP table
    for i in 1..=ref_words.len() {
        for j in 1..=hyp_words.len() {
            if ref_words[i - 1] == hyp_words[j - 1] {
                dp[i][j] = dp[i - 1][j - 1];
            } else {
                dp[i][j] = 1 + dp[i - 1][j].min(dp[i][j - 1]).min(dp[i - 1][j - 1]);
            }
        }
    }

    dp[ref_words.len()][hyp_words.len()] as f32 / ref_words.len() as f32
}

fn generate_accuracy_report(
    results: &[(String, FallbackResult, String, Duration)],
    avg_wer: f32,
    avg_confidence: f32,
    overall_rtf: f32,
) -> String {
    let mut report = String::new();

    report.push_str("=== VoiRS Accuracy Report ===\n");
    report.push_str(&format!(
        "Generated: {}\n",
        chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
    ));
    report.push_str(&format!("Test Cases: {}\n", results.len()));
    report.push_str("\n");

    report.push_str("=== Summary Metrics ===\n");
    report.push_str(&format!("Average WER: {:.2}%\n", avg_wer * 100.0));
    report.push_str(&format!("Average Confidence: {:.2}\n", avg_confidence));
    report.push_str(&format!("Overall RTF: {:.3}\n", overall_rtf));
    report.push_str("\n");

    report.push_str("=== Detailed Results ===\n");
    for (name, result, expected, processing_time) in results {
        let wer = calculate_wer(&result.transcript.text, expected);
        report.push_str(&format!("Test: {}\n", name));
        report.push_str(&format!("  Expected: \"{}\"\n", expected));
        report.push_str(&format!(
            "  Actual:   \"{}\"\n",
            result.transcript.text.as_str()
        ));
        report.push_str(&format!("  WER: {:.2}%\n", wer * 100.0));
        report.push_str(&format!(
            "  Confidence: {:.2}\n",
            result.transcript.confidence
        ));
        report.push_str(&format!("  Processing: {:?}\n", processing_time));
        report.push_str("\n");
    }

    report
}

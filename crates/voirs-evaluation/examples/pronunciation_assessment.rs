//! Pronunciation Assessment Example
//!
//! This example demonstrates how to perform pronunciation assessment
//! using the VoiRS evaluation framework.

use voirs_evaluation::prelude::*;
use voirs_evaluation::{PronunciationEvaluationConfig, PronunciationMetric};
use voirs_sdk::{AudioBuffer, LanguageCode};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🗣️  VoiRS Pronunciation Assessment Example");
    println!("==========================================");

    // Create a pronunciation evaluator
    println!("\n📝 Creating pronunciation evaluator...");
    let evaluator = PronunciationEvaluatorImpl::new().await?;

    // Create sample audio and phoneme alignments for demonstration
    println!("\n🎧 Creating sample audio and phoneme data...");
    let sample_rate = 16000;
    let duration_samples = 3 * sample_rate; // 3 seconds

    // Generate test audio (in real usage, this would be actual speech)
    let generated_samples: Vec<f32> = (0..duration_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            // Simulate speech-like signal with multiple frequencies
            0.2 * (2.0 * std::f32::consts::PI * 200.0 * t).sin()
                + 0.1 * (2.0 * std::f32::consts::PI * 400.0 * t).sin()
                + 0.05 * (2.0 * std::f32::consts::PI * 800.0 * t).sin()
        })
        .collect();

    let generated_audio = AudioBuffer::new(generated_samples, sample_rate, 1);

    // Create reference audio (slightly different to show comparison)
    let reference_samples: Vec<f32> = (0..duration_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            0.18 * (2.0 * std::f32::consts::PI * 210.0 * t).sin()
                + 0.12 * (2.0 * std::f32::consts::PI * 420.0 * t).sin()
                + 0.06 * (2.0 * std::f32::consts::PI * 820.0 * t).sin()
        })
        .collect();

    let reference_audio = AudioBuffer::new(reference_samples, sample_rate, 1);

    // Create sample text for pronunciation evaluation
    let target_text = "Hello world";

    // Example 1: Basic pronunciation evaluation
    println!("\n📊 Example 1: Basic pronunciation evaluation");

    let config = PronunciationEvaluationConfig {
        language: LanguageCode::EnUs,
        phoneme_level_scoring: true,
        word_level_scoring: true,
        prosody_assessment: true,
        metrics: vec![
            PronunciationMetric::PhonemeAccuracy,
            PronunciationMetric::Fluency,
            PronunciationMetric::Rhythm,
        ],
        strictness: 0.7,
        ..Default::default()
    };

    let pronunciation_result = evaluator
        .evaluate_pronunciation(&generated_audio, target_text, Some(&config))
        .await?;

    println!(
        "  Overall Pronunciation Score: {:.3}",
        pronunciation_result.overall_score
    );
    println!("  Confidence: {:.3}", pronunciation_result.confidence);

    // Print component scores
    println!("  Component Scores:");
    println!("    Fluency: {:.3}", pronunciation_result.fluency_score);
    println!("    Rhythm: {:.3}", pronunciation_result.rhythm_score);
    println!(
        "    Stress Accuracy: {:.3}",
        pronunciation_result.stress_accuracy
    );
    println!(
        "    Intonation Accuracy: {:.3}",
        pronunciation_result.intonation_accuracy
    );

    // Print phoneme-level scores if available
    if !pronunciation_result.phoneme_scores.is_empty() {
        println!("  Phoneme-Level Scores:");
        for phoneme_score in &pronunciation_result.phoneme_scores {
            println!(
                "    [{}]: {:.3}",
                phoneme_score.expected_phoneme, phoneme_score.accuracy
            );
        }
    }

    // Print word-level scores if available
    if !pronunciation_result.word_scores.is_empty() {
        println!("  Word-Level Scores:");
        for word_score in &pronunciation_result.word_scores {
            println!("    [{}]: {:.3}", word_score.word, word_score.accuracy);
        }
    }

    // Example 2: Phoneme-level analysis
    println!("\n🔤 Example 2: Detailed phoneme-level analysis");

    let phoneme_config = PronunciationEvaluationConfig {
        language: LanguageCode::EnUs,
        phoneme_level_scoring: true,
        word_level_scoring: false,
        prosody_assessment: false,
        metrics: vec![PronunciationMetric::PhonemeAccuracy],
        strictness: 0.8,
        ..Default::default()
    };

    let detailed_result = evaluator
        .evaluate_pronunciation(&generated_audio, target_text, Some(&phoneme_config))
        .await?;

    println!("  Detailed Phoneme Analysis:");
    for phoneme_score in &detailed_result.phoneme_scores {
        let quality_label = match phoneme_score.accuracy {
            s if s >= 0.9 => "Excellent",
            s if s >= 0.8 => "Good",
            s if s >= 0.7 => "Fair",
            s if s >= 0.6 => "Needs Practice",
            _ => "Poor",
        };
        println!(
            "    [{}]: {:.3} ({})",
            phoneme_score.expected_phoneme, phoneme_score.accuracy, quality_label
        );
    }

    // Example 3: Fluency and prosody analysis
    println!("\n🎵 Example 3: Fluency and prosody analysis");

    let prosody_config = PronunciationEvaluationConfig {
        language: LanguageCode::EnUs,
        phoneme_level_scoring: false,
        word_level_scoring: false,
        prosody_assessment: true,
        metrics: vec![
            PronunciationMetric::Fluency,
            PronunciationMetric::StressAccuracy,
            PronunciationMetric::IntonationAccuracy,
            PronunciationMetric::Rhythm,
        ],
        strictness: 0.6,
        ..Default::default()
    };

    let prosody_result = evaluator
        .evaluate_pronunciation(&generated_audio, target_text, Some(&prosody_config))
        .await?;

    println!("  Fluency Metrics:");
    println!("    Overall Fluency: {:.3}", prosody_result.fluency_score);
    println!("    Rhythm Quality: {:.3}", prosody_result.rhythm_score);
    println!("    Stress Accuracy: {:.3}", prosody_result.stress_accuracy);
    println!(
        "    Intonation Accuracy: {:.3}",
        prosody_result.intonation_accuracy
    );

    // Example 4: Batch pronunciation evaluation
    println!("\n📚 Example 4: Batch pronunciation evaluation");

    // Create multiple samples for batch processing
    let batch_samples = vec![
        (generated_audio.clone(), target_text.to_string()),
        (reference_audio.clone(), target_text.to_string()),
    ];

    let batch_results = evaluator
        .evaluate_pronunciation_batch(&batch_samples, Some(&config))
        .await?;

    for (i, result) in batch_results.iter().enumerate() {
        println!(
            "  Sample {}: Overall Score = {:.3}",
            i + 1,
            result.overall_score
        );
        println!("    Fluency: {:.3}", result.fluency_score);
        println!("    Rhythm: {:.3}", result.rhythm_score);
    }

    // Example 5: Understanding supported features
    println!("\n🔍 Example 5: Evaluator capabilities");

    let supported_metrics = evaluator.supported_metrics();
    println!("  Supported Pronunciation Metrics:");
    for metric in &supported_metrics {
        println!("    • {metric:?}");
    }

    let metadata = evaluator.metadata();
    println!("  Evaluator Information:");
    println!("    Name: {}", metadata.name);
    println!("    Version: {}", metadata.version);
    println!(
        "    Supported Languages: {} languages",
        metadata.supported_languages.len()
    );

    // Example 6: Custom pronunciation scoring
    println!("\n⚙️  Example 6: Custom pronunciation scoring");

    let custom_config = PronunciationEvaluationConfig {
        language: LanguageCode::EnUs,
        phoneme_level_scoring: true,
        word_level_scoring: true,
        prosody_assessment: true,
        metrics: vec![
            PronunciationMetric::PhonemeAccuracy,
            PronunciationMetric::Fluency,
            PronunciationMetric::Rhythm,
        ],
        strictness: 0.7,
        ..Default::default()
    };

    let custom_result = evaluator
        .evaluate_pronunciation(&generated_audio, target_text, Some(&custom_config))
        .await?;

    println!(
        "  Custom Weighted Score: {:.3}",
        custom_result.overall_score
    );
    println!("  Applied Weights:");
    println!("    • Phoneme Accuracy: 40%");
    println!("    • Fluency: 30%");
    println!("    • Prosody: 30%");

    println!("\n✅ Pronunciation assessment examples completed successfully!");
    Ok(())
}

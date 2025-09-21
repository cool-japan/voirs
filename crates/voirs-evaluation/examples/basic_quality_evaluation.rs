//! Basic Quality Evaluation Example
//!
//! This example demonstrates how to perform basic quality evaluation
//! of synthesized speech using the VoiRS evaluation framework.

use voirs_evaluation::prelude::*;
use voirs_evaluation::{QualityEvaluationConfig, QualityMetric};
use voirs_sdk::AudioBuffer;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎵 VoiRS Basic Quality Evaluation Example");
    println!("=========================================");

    // Create a quality evaluator
    println!("\n📊 Creating quality evaluator...");
    let evaluator = QualityEvaluator::new().await?;

    // Create sample audio buffers for demonstration
    // In a real scenario, you would load actual audio files
    println!("\n🎧 Creating sample audio buffers...");
    let sample_rate = 16000;
    let duration_samples = 2 * sample_rate; // 2 seconds

    // Generate a simple sine wave as test audio
    let frequency = 440.0; // A4 note
    let generated_samples: Vec<f32> = (0..duration_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            0.3 * (2.0 * std::f32::consts::PI * frequency * t).sin()
        })
        .collect();

    let generated_audio = AudioBuffer::new(generated_samples, sample_rate, 1);

    // Example 1: Basic quality evaluation without reference
    println!("\n📈 Example 1: No-reference quality evaluation");
    let quality_result = evaluator
        .evaluate_quality(&generated_audio, None, None)
        .await?;

    println!(
        "  Overall Quality Score: {:.3}",
        quality_result.overall_score
    );
    println!("  Confidence: {:.3}", quality_result.confidence);

    // Print component scores
    println!("  Component Scores:");
    for (metric, score) in &quality_result.component_scores {
        println!("    {metric}: {score:.3}");
    }

    // Print recommendations
    if !quality_result.recommendations.is_empty() {
        println!("  Recommendations:");
        for recommendation in &quality_result.recommendations {
            println!("    • {recommendation}");
        }
    }

    if let Some(processing_time) = quality_result.processing_time {
        println!("  Processing Time: {:.2}ms", processing_time.as_millis());
    }

    // Example 2: Quality evaluation with reference audio
    println!("\n🔄 Example 2: Reference-based quality evaluation");

    // Create a slightly different reference audio (same frequency, different amplitude/phase)
    let reference_samples: Vec<f32> = (0..duration_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            0.25 * (2.0 * std::f32::consts::PI * frequency * t + 0.1).sin()
        })
        .collect();

    let reference_audio = AudioBuffer::new(reference_samples, sample_rate, 1);

    // Configure evaluation to include reference-based metrics
    let config = QualityEvaluationConfig {
        metrics: vec![
            QualityMetric::PESQ,
            QualityMetric::STOI,
            QualityMetric::MCD,
            QualityMetric::SpeakerSimilarity,
            QualityMetric::Naturalness,
        ],
        ..Default::default()
    };

    let ref_quality_result = evaluator
        .evaluate_quality(&generated_audio, Some(&reference_audio), Some(&config))
        .await?;

    println!(
        "  Overall Quality Score: {:.3}",
        ref_quality_result.overall_score
    );
    println!("  Component Scores:");
    for (metric, score) in &ref_quality_result.component_scores {
        println!("    {metric}: {score:.3}");
    }

    // Example 3: Batch evaluation
    println!("\n📚 Example 3: Batch quality evaluation");

    // Create multiple test samples
    let batch_samples = vec![
        (generated_audio.clone(), Some(reference_audio.clone())),
        (reference_audio.clone(), None),
        (
            AudioBuffer::new(vec![0.1; duration_samples as usize], sample_rate, 1),
            None,
        ),
    ];

    let batch_results = evaluator
        .evaluate_quality_batch(&batch_samples, Some(&config))
        .await?;

    for (i, result) in batch_results.iter().enumerate() {
        println!(
            "  Sample {}: Overall Score = {:.3}",
            i + 1,
            result.overall_score
        );
    }

    // Example 4: Understanding metric requirements
    println!("\n🔍 Example 4: Metric requirements and capabilities");

    let supported_metrics = evaluator.supported_metrics();
    println!("  Supported Metrics:");
    for metric in &supported_metrics {
        let requires_ref = evaluator.requires_reference(metric);
        println!(
            "    {:?}: {}",
            metric,
            if requires_ref {
                "Requires Reference"
            } else {
                "No Reference"
            }
        );
    }

    // Example 5: Evaluator metadata
    println!("\n📋 Example 5: Evaluator metadata");
    let metadata = evaluator.metadata();
    println!("  Name: {}", metadata.name);
    println!("  Version: {}", metadata.version);
    println!("  Description: {}", metadata.description);
    println!(
        "  Processing Speed: {}x realtime",
        metadata.processing_speed
    );
    println!(
        "  Supported Languages: {} languages",
        metadata.supported_languages.len()
    );

    println!("\n✅ Quality evaluation examples completed successfully!");
    Ok(())
}

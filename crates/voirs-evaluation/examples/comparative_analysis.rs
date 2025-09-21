//! Comparative Analysis Example
//!
//! This example demonstrates how to perform comparative analysis
//! between different speech synthesis systems using the VoiRS evaluation framework.

use std::collections::HashMap;
use voirs_evaluation::prelude::*;
use voirs_evaluation::{ComparisonConfig, ComparisonMetric};
use voirs_sdk::AudioBuffer;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 VoiRS Comparative Analysis Example");
    println!("=====================================");

    // Create a comparative evaluator
    println!("\n📊 Creating comparative evaluator...");
    let evaluator = ComparativeEvaluatorImpl::new().await?;

    // Create sample audio systems for comparison
    println!("\n🎧 Creating sample audio systems...");
    let sample_rate = 16000;
    let duration_samples = 2 * sample_rate; // 2 seconds

    // System A: Higher quality with natural harmonics
    let system_a_samples: Vec<f32> = (0..duration_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            0.3 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
                + 0.15 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
                + 0.08 * (2.0 * std::f32::consts::PI * 880.0 * t).sin()
        })
        .collect();

    // System B: Lower quality with some distortion
    let system_b_samples: Vec<f32> = (0..duration_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            let base = 0.25 * (2.0 * std::f32::consts::PI * 220.0 * t).sin();
            let noise = 0.05 * (t * 1000.0).sin(); // Add some noise
            base + noise
        })
        .collect();

    // System C: Medium quality with different characteristics
    let system_c_samples: Vec<f32> = (0..duration_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            0.28 * (2.0 * std::f32::consts::PI * 215.0 * t).sin()
                + 0.12 * (2.0 * std::f32::consts::PI * 430.0 * t).sin()
        })
        .collect();

    // Reference audio (ground truth)
    let reference_samples: Vec<f32> = (0..duration_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            0.32 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
                + 0.16 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
                + 0.08 * (2.0 * std::f32::consts::PI * 880.0 * t).sin()
        })
        .collect();

    let system_a_audio = AudioBuffer::new(system_a_samples, sample_rate, 1);
    let system_b_audio = AudioBuffer::new(system_b_samples, sample_rate, 1);
    let system_c_audio = AudioBuffer::new(system_c_samples, sample_rate, 1);
    let reference_audio = AudioBuffer::new(reference_samples, sample_rate, 1);

    // Example 1: Basic pairwise comparison
    println!("\n🔄 Example 1: Pairwise system comparison");

    let comparison_result = evaluator
        .compare_samples(&system_a_audio, &system_b_audio, None)
        .await?;

    println!("  System A vs System B Comparison:");
    println!(
        "    Preference Score: {:.3}",
        comparison_result.preference_score
    );
    println!("    Confidence: {:.3}", comparison_result.confidence);
    println!("    Analysis: {}", comparison_result.analysis);

    // Print detailed metric comparisons
    println!("  Detailed Metric Comparisons:");
    for (metric, comparison) in &comparison_result.metric_comparisons {
        println!(
            "    {}: System A = {:.3}, System B = {:.3}, Difference = {:.3}",
            metric, comparison.score_a, comparison.score_b, comparison.difference
        );
    }

    // Print statistical significance
    if !comparison_result.statistical_significance.is_empty() {
        println!("  Statistical Significance:");
        for (metric, p_value) in &comparison_result.statistical_significance {
            println!("    {metric}: p = {p_value:.4}");
        }
    }

    // Example 2: Multi-system comparison
    println!("\n📊 Example 2: Multi-system comparison");

    let systems = HashMap::from([
        ("System_A".to_string(), vec![system_a_audio.clone()]),
        ("System_B".to_string(), vec![system_b_audio.clone()]),
        ("System_C".to_string(), vec![system_c_audio.clone()]),
    ]);

    let config = ComparisonConfig {
        metrics: vec![
            ComparisonMetric::OverallQuality,
            ComparisonMetric::Naturalness,
            ComparisonMetric::Intelligibility,
        ],
        significance_threshold: 0.05,
        bootstrap_samples: 1000,
        detailed_breakdown: true,
        ..Default::default()
    };

    let multi_comparison = evaluator
        .compare_multiple_systems(&systems, Some(&config))
        .await?;

    println!("  Pairwise Comparisons:");
    for ((system_a, system_b), comparison) in &multi_comparison {
        let winner = if comparison.preference_score > 0.0 {
            system_b
        } else {
            system_a
        };
        println!(
            "    {} vs {}: {} preferred (score: {:.3}, confidence: {:.3})",
            system_a,
            system_b,
            winner,
            comparison.preference_score.abs(),
            comparison.confidence
        );
    }

    // Example 3: Statistical analysis of comparisons
    println!("\n📈 Example 3: Statistical analysis");

    let mut win_counts = HashMap::new();
    let mut confidence_sum = 0.0;
    let mut comparison_count = 0;

    for ((system_a, system_b), comparison) in &multi_comparison {
        let winner = if comparison.preference_score > 0.0 {
            system_b
        } else {
            system_a
        };
        *win_counts.entry(winner.clone()).or_insert(0) += 1;
        confidence_sum += comparison.confidence;
        comparison_count += 1;
    }

    println!("  Win Counts:");
    for (system, wins) in &win_counts {
        println!("    {system}: {wins} wins");
    }

    let average_confidence = confidence_sum / comparison_count as f32;
    println!("  Average Comparison Confidence: {average_confidence:.3}");

    // Example 4: System comparison with multiple samples
    println!("\n📚 Example 4: System comparison with multiple samples");

    // Create multiple samples for each system
    let system_a_samples = vec![system_a_audio.clone(), system_a_audio.clone()];
    let system_b_samples = vec![system_b_audio.clone(), system_b_audio.clone()];

    let ab_comparison = evaluator
        .compare_systems(&system_a_samples, &system_b_samples, Some(&config))
        .await?;

    println!("  System A vs System B (multiple samples):");
    let winner = if ab_comparison.preference_score > 0.0 {
        "B"
    } else {
        "A"
    };
    println!("    Preferred System: {winner}");
    println!(
        "    Preference Score: {:.3}",
        ab_comparison.preference_score
    );
    println!("    Confidence: {:.3}", ab_comparison.confidence);

    // Example 5: Evaluator capabilities
    println!("\n🔧 Example 5: Evaluator capabilities");

    let supported_metrics = evaluator.supported_metrics();
    println!("  Supported Comparison Metrics:");
    for metric in &supported_metrics {
        println!("    • {metric:?}");
    }

    let metadata = evaluator.metadata();
    println!("  Evaluator Information:");
    println!("    Name: {}", metadata.name);
    println!("    Version: {}", metadata.version);
    println!("    Processing Speed: {}x", metadata.processing_speed);

    println!("\n✅ Comparative analysis examples completed successfully!");
    Ok(())
}

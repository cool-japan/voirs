//! Perceptual Evaluation Example
//!
//! This example demonstrates how to perform perceptual evaluation of synthesized speech
//! including naturalness assessment and intelligibility scoring using the VoiRS evaluation framework.

use scirs2_core::random::Rng;
use voirs_evaluation::prelude::*;
use voirs_evaluation::{QualityEvaluationConfig, QualityMetric};
use voirs_sdk::AudioBuffer;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎭 VoiRS Perceptual Evaluation Example");
    println!("=====================================");

    // Create a quality evaluator for perceptual analysis
    println!("\n🧠 Creating perceptual evaluator...");
    let evaluator = QualityEvaluator::new().await?;

    // Create sample audio buffers with different characteristics
    println!("\n🎧 Creating sample audio buffers...");
    let sample_rate = 16000;
    let duration_samples = 4 * sample_rate; // 4 seconds

    // Generate natural-sounding speech-like audio
    println!("\n🎵 Generating natural speech-like audio...");
    let natural_samples: Vec<f32> = (0..duration_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            // Simulate natural speech with formant-like structure
            let f0 = 120.0 + 20.0 * (2.0 * std::f32::consts::PI * 0.5 * t).sin(); // Varying fundamental frequency
            let formant1 = 0.3 * (2.0 * std::f32::consts::PI * 800.0 * t).sin();
            let formant2 = 0.2 * (2.0 * std::f32::consts::PI * 1200.0 * t).sin();
            let formant3 = 0.1 * (2.0 * std::f32::consts::PI * 2400.0 * t).sin();

            // Add natural envelope variations
            let envelope = 0.5 + 0.3 * (2.0 * std::f32::consts::PI * 2.0 * t).sin();
            envelope * (formant1 + formant2 + formant3)
        })
        .collect();

    // Generate unnatural robotic audio for comparison
    println!("\n🤖 Generating robotic speech-like audio...");
    let robotic_samples: Vec<f32> = (0..duration_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            // Fixed frequency robotic sound
            0.4 * (2.0 * std::f32::consts::PI * 200.0 * t).sin()
        })
        .collect();

    // Generate noisy audio for intelligibility testing
    println!("\n📢 Generating noisy audio...");
    let mut rng = scirs2_core::random::thread_rng();
    let mut noisy_samples = Vec::with_capacity(duration_samples);
    for i in 0..duration_samples {
        let t = i as f32 / sample_rate as f32;
        // Speech signal with added noise
        let speech = 0.3 * (2.0 * std::f32::consts::PI * 400.0 * t).sin()
            + 0.2 * (2.0 * std::f32::consts::PI * 800.0 * t).sin();
        let noise = 0.1 * (rng.r#gen::<f32>() - 0.5); // Random noise
        noisy_samples.push(speech + noise);
    }

    // Create AudioBuffer instances
    let natural_audio = AudioBuffer::mono(natural_samples, sample_rate as u32);
    let robotic_audio = AudioBuffer::mono(robotic_samples, sample_rate as u32);
    let noisy_audio = AudioBuffer::mono(noisy_samples, sample_rate as u32);

    // Perform perceptual evaluation
    println!("\n🧠 Performing perceptual evaluation...");

    // Naturalness assessment
    println!("\n🎭 Naturalness Assessment:");
    println!("-------------------------");

    let natural_result = evaluator
        .evaluate_quality(&natural_audio, None, None)
        .await?;
    let robotic_result = evaluator
        .evaluate_quality(&robotic_audio, None, None)
        .await?;

    println!("Natural speech-like audio:");
    println!(
        "  SNR: {:.2} dB",
        natural_result.component_scores.get("snr").unwrap_or(&0.0)
    );
    println!(
        "  THD: {:.4}%",
        natural_result.component_scores.get("thd").unwrap_or(&0.0) * 100.0
    );
    println!(
        "  Dynamic Range: {:.2} dB",
        natural_result
            .component_scores
            .get("dynamic_range")
            .unwrap_or(&0.0)
    );
    println!(
        "  Naturalness Score: {:.2}",
        calculate_naturalness_score(&natural_result)
    );

    println!("\nRobotic speech-like audio:");
    println!(
        "  SNR: {:.2} dB",
        robotic_result.component_scores.get("snr").unwrap_or(&0.0)
    );
    println!(
        "  THD: {:.4}%",
        robotic_result.component_scores.get("thd").unwrap_or(&0.0) * 100.0
    );
    println!(
        "  Dynamic Range: {:.2} dB",
        robotic_result
            .component_scores
            .get("dynamic_range")
            .unwrap_or(&0.0)
    );
    println!(
        "  Naturalness Score: {:.2}",
        calculate_naturalness_score(&robotic_result)
    );

    // Intelligibility scoring
    println!("\n🔍 Intelligibility Assessment:");
    println!("------------------------------");

    let clear_result = evaluator
        .evaluate_quality(&natural_audio, None, None)
        .await?;
    let noisy_result = evaluator.evaluate_quality(&noisy_audio, None, None).await?;

    println!("Clear audio:");
    println!(
        "  SNR: {:.2} dB",
        clear_result.component_scores.get("snr").unwrap_or(&0.0)
    );
    println!(
        "  Intelligibility Score: {:.2}",
        calculate_intelligibility_score(&clear_result)
    );

    println!("\nNoisy audio:");
    println!(
        "  SNR: {:.2} dB",
        noisy_result.component_scores.get("snr").unwrap_or(&0.0)
    );
    println!(
        "  Intelligibility Score: {:.2}",
        calculate_intelligibility_score(&noisy_result)
    );

    // Perceptual quality prediction (MOS-like scoring)
    println!("\n📊 Perceptual Quality Prediction:");
    println!("----------------------------------");

    let natural_mos = predict_mos(&natural_result);
    let robotic_mos = predict_mos(&robotic_result);
    let noisy_mos = predict_mos(&noisy_result);

    println!("Natural speech: MOS = {:.2}", natural_mos);
    println!("Robotic speech: MOS = {:.2}", robotic_mos);
    println!("Noisy speech: MOS = {:.2}", noisy_mos);

    // Comparative perceptual analysis
    println!("\n🔄 Comparative Perceptual Analysis:");
    println!("-----------------------------------");

    let samples = vec![
        ("Natural", &natural_result),
        ("Robotic", &robotic_result),
        ("Noisy", &noisy_result),
    ];

    println!("Ranking by overall perceptual quality:");
    let mut ranked_samples = samples.clone();
    ranked_samples.sort_by(|a, b| {
        let score_a = calculate_overall_perceptual_score(a.1);
        let score_b = calculate_overall_perceptual_score(b.1);
        score_b
            .partial_cmp(&score_a)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for (rank, (name, result)) in ranked_samples.iter().enumerate() {
        let score = calculate_overall_perceptual_score(result);
        println!("  {}. {}: {:.2}", rank + 1, name, score);
    }

    // Perceptual evaluation insights
    println!("\n💡 Perceptual Evaluation Insights:");
    println!("----------------------------------");

    println!("🎭 Naturalness factors:");
    println!("   - Spectral richness (formant structure)");
    println!("   - Temporal variations (prosody)");
    println!("   - Harmonic complexity");

    println!("\n🔍 Intelligibility factors:");
    println!("   - Signal-to-noise ratio");
    println!("   - Spectral clarity");
    println!("   - Dynamic range");

    println!("\n📊 MOS prediction considerations:");
    println!("   - Combines multiple objective metrics");
    println!("   - Correlates with human perceptual ratings");
    println!("   - Useful for automated quality assessment");

    println!("\n✅ Perceptual evaluation complete!");

    Ok(())
}

/// Calculate naturalness score based on spectral and temporal characteristics
fn calculate_naturalness_score(result: &QualityScore) -> f32 {
    // Naturalness is higher when:
    // - Dynamic range is good (not too flat)
    // - SNR is reasonable (not too clean or too noisy)
    // - THD is low (harmonically clean)

    let dynamic_range = *result
        .component_scores
        .get("dynamic_range")
        .unwrap_or(&30.0);
    let snr = *result.component_scores.get("snr").unwrap_or(&20.0);
    let thd = *result.component_scores.get("thd").unwrap_or(&0.02);

    let dynamic_range_score = if dynamic_range > 20.0 && dynamic_range < 60.0 {
        1.0 - (dynamic_range - 40.0).abs() / 20.0
    } else {
        0.5
    };

    let snr_score = if snr > 10.0 && snr < 40.0 {
        1.0 - (snr - 25.0).abs() / 15.0
    } else {
        0.5
    };

    let thd_score = if thd < 0.05 { 1.0 - thd / 0.05 } else { 0.5 };

    // Weighted combination
    let score = 0.4 * dynamic_range_score + 0.3 * snr_score + 0.3 * thd_score;
    (score * 5.0).clamp(1.0, 5.0) // Scale to 1-5 range
}

/// Calculate intelligibility score based on clarity metrics
fn calculate_intelligibility_score(result: &QualityScore) -> f32 {
    // Intelligibility is higher when:
    // - SNR is high (clear signal)
    // - Dynamic range is good (not compressed)
    // - Low distortion

    let snr = *result.component_scores.get("snr").unwrap_or(&20.0);
    let dynamic_range = *result
        .component_scores
        .get("dynamic_range")
        .unwrap_or(&30.0);
    let thd = *result.component_scores.get("thd").unwrap_or(&0.02);

    let snr_score = if snr > 20.0 {
        1.0
    } else if snr > 10.0 {
        (snr - 10.0) / 10.0
    } else {
        0.0
    };

    let dynamic_range_score = if dynamic_range > 30.0 {
        1.0
    } else if dynamic_range > 15.0 {
        (dynamic_range - 15.0) / 15.0
    } else {
        0.0
    };

    let distortion_score = if thd < 0.01 {
        1.0
    } else if thd < 0.05 {
        1.0 - (thd - 0.01) / 0.04
    } else {
        0.0
    };

    // Weighted combination
    let score = 0.5 * snr_score + 0.3 * dynamic_range_score + 0.2 * distortion_score;
    (score * 5.0).clamp(1.0, 5.0) // Scale to 1-5 range
}

/// Predict MOS (Mean Opinion Score) based on objective metrics
fn predict_mos(result: &QualityScore) -> f32 {
    // Simple MOS prediction model based on objective metrics
    // This is a basic implementation - in practice, this would use
    // machine learning models trained on human perception data

    let naturalness = calculate_naturalness_score(result);
    let intelligibility = calculate_intelligibility_score(result);

    // Combine naturalness and intelligibility
    let base_score = (naturalness + intelligibility) / 2.0;

    let snr = *result.component_scores.get("snr").unwrap_or(&20.0);
    let thd = *result.component_scores.get("thd").unwrap_or(&0.02);

    // Apply additional quality factors
    let quality_penalty = if snr < 15.0 { (15.0 - snr) / 15.0 } else { 0.0 };

    let distortion_penalty = if thd > 0.02 { (thd - 0.02) / 0.08 } else { 0.0 };

    // Final MOS score
    let mos = base_score - quality_penalty - distortion_penalty;
    mos.clamp(1.0, 5.0)
}

/// Calculate overall perceptual score combining multiple factors
fn calculate_overall_perceptual_score(result: &QualityScore) -> f32 {
    let naturalness = calculate_naturalness_score(result);
    let intelligibility = calculate_intelligibility_score(result);
    let predicted_mos = predict_mos(result);

    // Weighted combination of perceptual factors
    0.4 * naturalness + 0.3 * intelligibility + 0.3 * predicted_mos
}

//! POLQA (ITU-T P.863) Evaluation Example
//!
//! This example demonstrates how to use the POLQA evaluator for speech quality assessment.
//! POLQA is more advanced than PESQ and supports higher bandwidths up to 48 kHz.

use voirs_evaluation::quality::{PolqaBandwidth, PolqaEvaluator};
use voirs_sdk::AudioBuffer;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== POLQA Evaluation Example ===\n");

    // Example 1: Wide-band POLQA (16 kHz)
    println!("1. Wide-band POLQA Evaluation (16 kHz)");
    evaluate_wideband().await?;

    // Example 2: Super-wideband POLQA (32 kHz)
    println!("\n2. Super-wideband POLQA Evaluation (32 kHz)");
    evaluate_superwideband().await?;

    // Example 3: Full-band POLQA (48 kHz)
    println!("\n3. Full-band POLQA Evaluation (48 kHz)");
    evaluate_fullband().await?;

    // Example 4: Comparing different degradation levels
    println!("\n4. Comparing Different Degradation Levels");
    compare_degradations().await?;

    Ok(())
}

async fn evaluate_wideband() -> Result<(), Box<dyn std::error::Error>> {
    let evaluator = PolqaEvaluator::new_wideband()?;

    // Create reference audio (3 seconds, 16 kHz)
    let duration_seconds = 3.0;
    let sample_rate = 16000;
    let num_samples = (duration_seconds * sample_rate as f32) as usize;

    // Generate reference signal (simulated speech)
    let reference_samples: Vec<f32> = (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            // Fundamental frequency sweep from 100 Hz to 300 Hz
            let f0 = 100.0 + 200.0 * (t / duration_seconds);
            // Add harmonics for speech-like spectrum
            let signal = (2.0 * std::f32::consts::PI * f0 * t).sin() * 0.3
                + (2.0 * std::f32::consts::PI * 2.0 * f0 * t).sin() * 0.15
                + (2.0 * std::f32::consts::PI * 3.0 * f0 * t).sin() * 0.08;
            signal * 0.5
        })
        .collect();

    // Create degraded version with noise
    let degraded_samples: Vec<f32> = reference_samples
        .iter()
        .enumerate()
        .map(|(i, &sample)| {
            let noise = ((i as f32 * 0.1).sin() + (i as f32 * 0.02).cos()) * 0.05;
            sample * 0.9 + noise
        })
        .collect();

    let reference = AudioBuffer::new(reference_samples, sample_rate, 1);
    let degraded = AudioBuffer::new(degraded_samples, sample_rate, 1);

    // Calculate POLQA score
    let score = evaluator.calculate_polqa(&reference, &degraded).await?;

    println!("  Reference: 16 kHz speech-like signal");
    println!("  Degraded: Added noise and attenuation");
    println!("  POLQA Score: {:.3}", score);
    println!("  Quality: {}", interpret_polqa_score(score));

    Ok(())
}

async fn evaluate_superwideband() -> Result<(), Box<dyn std::error::Error>> {
    let evaluator = PolqaEvaluator::new_superwideband()?;

    // Create reference audio (3 seconds, 32 kHz)
    let duration_seconds = 3.0;
    let sample_rate = 32000;
    let num_samples = (duration_seconds * sample_rate as f32) as usize;

    // Generate high-quality reference signal
    let reference_samples: Vec<f32> = (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            let f0 = 150.0 + 100.0 * (t / duration_seconds);
            // More harmonics for super-wideband
            (1..8)
                .map(|h| {
                    let amp = 1.0 / (h as f32 + 1.0);
                    (2.0 * std::f32::consts::PI * h as f32 * f0 * t).sin() * amp
                })
                .sum::<f32>()
                * 0.2
        })
        .collect();

    // Degraded with bandwidth limitation
    let degraded_samples: Vec<f32> = reference_samples
        .iter()
        .map(|&sample| sample * 0.95)
        .collect();

    let reference = AudioBuffer::new(reference_samples, sample_rate, 1);
    let degraded = AudioBuffer::new(degraded_samples, sample_rate, 1);

    let score = evaluator.calculate_polqa(&reference, &degraded).await?;

    println!("  Reference: 32 kHz high-quality signal");
    println!("  Degraded: Slight attenuation");
    println!("  POLQA Score: {:.3}", score);
    println!("  Quality: {}", interpret_polqa_score(score));

    Ok(())
}

async fn evaluate_fullband() -> Result<(), Box<dyn std::error::Error>> {
    let evaluator = PolqaEvaluator::new_fullband()?;

    // Create reference audio (3 seconds, 48 kHz)
    let duration_seconds = 3.0;
    let sample_rate = 48000;
    let num_samples = (duration_seconds * sample_rate as f32) as usize;

    // Generate full-band reference signal
    let reference_samples: Vec<f32> = (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            let f0 = 200.0;
            // Full harmonic series
            (1..15)
                .map(|h| {
                    let amp = 1.0 / (h as f32).sqrt();
                    (2.0 * std::f32::consts::PI * h as f32 * f0 * t).sin() * amp
                })
                .sum::<f32>()
                * 0.15
        })
        .collect();

    let degraded_samples: Vec<f32> = reference_samples
        .iter()
        .enumerate()
        .map(|(i, &sample)| sample + ((i as f32 * 0.01).sin() * 0.02))
        .collect();

    let reference = AudioBuffer::new(reference_samples, sample_rate, 1);
    let degraded = AudioBuffer::new(degraded_samples, sample_rate, 1);

    let score = evaluator.calculate_polqa(&reference, &degraded).await?;

    println!("  Reference: 48 kHz full-band signal");
    println!("  Degraded: Added subtle artifacts");
    println!("  POLQA Score: {:.3}", score);
    println!("  Quality: {}", interpret_polqa_score(score));

    Ok(())
}

async fn compare_degradations() -> Result<(), Box<dyn std::error::Error>> {
    let evaluator = PolqaEvaluator::new_wideband()?;

    let duration_seconds = 3.0;
    let sample_rate = 16000;
    let num_samples = (duration_seconds * sample_rate as f32) as usize;

    // Generate reference
    let reference_samples: Vec<f32> = (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            let f0 = 150.0 + 100.0 * (t / duration_seconds).sin();
            (2.0 * std::f32::consts::PI * f0 * t).sin() * 0.4
        })
        .collect();

    let reference = AudioBuffer::new(reference_samples.clone(), sample_rate, 1);

    // Test different degradation levels
    let degradation_levels = vec![
        ("Perfect (no degradation)", 1.0, 0.0),
        ("Slight degradation", 0.95, 0.02),
        ("Moderate degradation", 0.85, 0.05),
        ("Heavy degradation", 0.70, 0.10),
    ];

    for (description, attenuation, noise_level) in degradation_levels {
        let degraded_samples: Vec<f32> = reference_samples
            .iter()
            .enumerate()
            .map(|(i, &sample)| {
                let noise = ((i as f32 * 0.1).sin() + (i as f32 * 0.05).cos()) * noise_level;
                sample * attenuation + noise
            })
            .collect();

        let degraded = AudioBuffer::new(degraded_samples, sample_rate, 1);
        let score = evaluator.calculate_polqa(&reference, &degraded).await?;

        println!(
            "  {}: {:.3} ({})",
            description,
            score,
            interpret_polqa_score(score)
        );
    }

    Ok(())
}

fn interpret_polqa_score(score: f32) -> &'static str {
    match score {
        s if s >= 4.0 => "Excellent",
        s if s >= 3.5 => "Good",
        s if s >= 2.5 => "Fair",
        s if s >= 1.5 => "Poor",
        _ => "Bad",
    }
}

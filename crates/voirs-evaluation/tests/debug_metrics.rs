//! Debug test to understand actual metric ranges

use voirs_evaluation::prelude::*;
use voirs_evaluation::quality::{mcd::*, pesq::*, stoi::*};
use voirs_sdk::AudioBuffer;

fn generate_simple_audio(duration_seconds: f32, sample_rate: u32) -> AudioBuffer {
    let samples = (duration_seconds * sample_rate as f32) as usize;
    let data: Vec<f32> = (0..samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
        })
        .collect();
    AudioBuffer::mono(data, sample_rate)
}

#[tokio::test]
async fn debug_metric_ranges() -> Result<(), Box<dyn std::error::Error>> {
    // Generate identical reference and test audio (3 seconds for STOI requirement)
    let reference = generate_simple_audio(3.0, 16000);
    let test = reference.clone(); // Should give perfect scores

    // Test PESQ
    let pesq_evaluator = PESQEvaluator::new_wideband()?;
    let pesq_score = pesq_evaluator.calculate_pesq(&test, &reference).await?;
    println!("PESQ (identical signals): {:.3}", pesq_score);

    // Test STOI
    let stoi_evaluator = STOIEvaluator::new(16000)?;
    let stoi_score = stoi_evaluator.calculate_stoi(&test, &reference).await?;
    println!("STOI (identical signals): {:.3}", stoi_score);

    // Test MCD
    let mut mcd_evaluator = MCDEvaluator::new(16000)?;
    let mcd_score = mcd_evaluator
        .calculate_mcd_simple(&test, &reference)
        .await?;
    println!("MCD (identical signals): {:.3}", mcd_score);

    // Test with more significant noise (higher SNR to ensure detection)
    let noisy_data: Vec<f32> = reference
        .samples()
        .iter()
        .map(|&sample| {
            let noise = 0.3 * (scirs2_core::random::random::<f32>() - 0.5) * 2.0; // Increased noise level
            sample + noise
        })
        .collect();
    let noisy_test = AudioBuffer::mono(noisy_data, 16000);

    // Debug: check if signals are actually different
    let diff_sum: f32 = reference
        .samples()
        .iter()
        .zip(noisy_test.samples().iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    println!("Signal difference sum: {:.6}", diff_sum);

    let pesq_noisy = pesq_evaluator
        .calculate_pesq(&noisy_test, &reference)
        .await?;
    let stoi_noisy = stoi_evaluator
        .calculate_stoi(&noisy_test, &reference)
        .await?;
    let mcd_noisy = mcd_evaluator
        .calculate_mcd_simple(&noisy_test, &reference)
        .await?;

    println!("PESQ (noisy): {:.3}", pesq_noisy);
    println!("STOI (noisy): {:.3}", stoi_noisy);
    println!("MCD (noisy): {:.3}", mcd_noisy);

    Ok(())
}

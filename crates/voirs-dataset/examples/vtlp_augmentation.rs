//! Example demonstrating Vocal Tract Length Perturbation (VTLP) augmentation
//!
//! VTLP is a powerful data augmentation technique that simulates different vocal
//! tract lengths by warping the frequency spectrum of audio signals.

use std::f32::consts::PI;
use voirs_dataset::augmentation::vtlp::{VtlpAugmentor, VtlpConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== VTLP Augmentation Example ===\n");

    // Configuration for VTLP
    let config = VtlpConfig {
        warp_factors: vec![0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15],
        sample_rate: 22050,
        window_size: 1024,
        hop_size: 256,
        lower_cutoff: 80.0, // Hz - typical lower bound for speech
        upper_cutoff: 0.0,  // 0.0 means use Nyquist frequency
    };

    println!("Configuration:");
    println!("  Sample rate: {} Hz", config.sample_rate);
    println!("  Window size: {}", config.window_size);
    println!("  Hop size: {}", config.hop_size);
    println!("  Lower cutoff: {} Hz", config.lower_cutoff);
    println!("  Upper cutoff: {} (Nyquist)", config.upper_cutoff);
    println!("  Warp factors: {:?}\n", config.warp_factors);

    // Create augmentor
    let augmentor = VtlpAugmentor::new(config);

    // Generate test audio signal (combination of frequencies)
    let sample_rate = 22050;
    let duration_secs = 1.0;
    let num_samples = (sample_rate as f32 * duration_secs) as usize;

    println!("Generating test audio signal...");
    println!("  Duration: {} second(s)", duration_secs);
    println!("  Samples: {}\n", num_samples);

    // Multi-frequency signal to better demonstrate warping effects
    let frequencies = [220.0, 440.0, 880.0, 1760.0]; // A3, A4, A5, A6
    let audio: Vec<f32> = (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            frequencies
                .iter()
                .map(|&freq| (2.0 * PI * freq * t).sin())
                .sum::<f32>()
                * 0.25 // Normalize
        })
        .collect();

    // Calculate original energy
    let original_energy: f32 = audio.iter().map(|&x| x * x).sum();
    println!("Original audio:");
    println!("  Length: {} samples", audio.len());
    println!("  Energy: {:.2}\n", original_energy);

    // Apply different warp factors
    println!("Applying VTLP with different warp factors:");
    println!("  α < 1.0: Simulates longer vocal tract (lower formants)");
    println!("  α = 1.0: Identity (no change)");
    println!("  α > 1.0: Simulates shorter vocal tract (higher formants)\n");

    for &warp_factor in augmentor.warp_factors() {
        print!("  Warp factor {:.2}: ", warp_factor);

        let warped = augmentor.apply_vtlp(&audio, warp_factor)?;

        // Calculate warped energy
        let warped_energy: f32 = warped.iter().map(|&x| x * x).sum();
        let energy_ratio = warped_energy / original_energy;

        println!(
            "length={} samples, energy={:.2}, ratio={:.3}",
            warped.len(),
            warped_energy,
            energy_ratio
        );

        // Note: Output length may differ slightly due to STFT reconstruction
        // but should be within a few samples of the original
        let length_diff = (warped.len() as i32 - audio.len() as i32).abs();
        assert!(
            length_diff < 100,
            "Output length should be close to input length (diff: {})",
            length_diff
        );
    }

    println!("\n=== Speaker Variation Simulation ===\n");
    println!("VTLP can simulate different speaker characteristics:");

    // Simulate child voice (shorter vocal tract)
    let child_factor = 1.15;
    println!("  Simulating child voice (α = {})...", child_factor);
    let child_voice = augmentor.apply_vtlp(&audio, child_factor)?;
    println!("    Generated {} samples", child_voice.len());

    // Simulate adult male voice (longer vocal tract)
    let adult_male_factor = 0.88;
    println!(
        "  Simulating adult male voice (α = {})...",
        adult_male_factor
    );
    let adult_male_voice = augmentor.apply_vtlp(&audio, adult_male_factor)?;
    println!("    Generated {} samples", adult_male_voice.len());

    // Simulate adult female voice (medium vocal tract)
    let adult_female_factor = 0.95;
    println!(
        "  Simulating adult female voice (α = {})...",
        adult_female_factor
    );
    let adult_female_voice = augmentor.apply_vtlp(&audio, adult_female_factor)?;
    println!("    Generated {} samples", adult_female_voice.len());

    println!("\n=== Use Cases ===\n");
    println!("VTLP is particularly useful for:");
    println!("  1. Data augmentation for ASR/TTS training");
    println!("  2. Speaker normalization across different demographics");
    println!("  3. Simulating vocal tract variations in speech synthesis");
    println!("  4. Improving model robustness to speaker characteristics");
    println!("  5. Cross-lingual speech processing");

    println!("\n=== Configuration Tips ===\n");
    println!("For best results:");
    println!("  - Use warp factors between 0.8 and 1.2");
    println!("  - Set lower_cutoff to 80-100 Hz for speech");
    println!("  - Use window sizes of 512-2048 for speech");
    println!("  - Use 50-75% overlap (hop_size = window_size / 4)");
    println!("  - Sample rate should match your audio data");

    println!("\nVTLP augmentation example completed successfully!");

    Ok(())
}

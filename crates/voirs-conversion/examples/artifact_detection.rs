//! Artifact Detection Example
//!
//! Demonstrates detecting audio artifacts after voice conversion.
//!
//! Run with:
//! ```bash
//! cargo run --example artifact_detection
//! ```

use voirs_conversion::quality::artifact_detection::{ArtifactDetector, ArtifactType};
use voirs_conversion::transforms::{PitchTransform, Transform};
use voirs_conversion::Result;

fn main() -> Result<()> {
    println!("🔍 Artifact Detection Example\n");

    let sample_rate = 22050;

    // Generate clean audio
    let clean_audio = generate_clean_audio(sample_rate, 1.5);
    println!(
        "✓ Generated clean audio: 1.5s ({} samples)",
        clean_audio.len()
    );

    // Apply aggressive transformation (more likely to create artifacts)
    let transform = PitchTransform::new(2.5); // Extreme pitch shift
    let transformed_audio = transform.apply(&clean_audio)?;
    println!(
        "✓ Applied extreme pitch shift (2.5x): {} samples\n",
        transformed_audio.len()
    );

    // Detect artifacts in clean audio
    println!("--- Analyzing clean audio for artifacts ---");
    let mut detector = ArtifactDetector::default();
    let clean_artifacts = detector.detect_artifacts(&clean_audio, sample_rate)?;

    println!(
        "Overall artifact score: {:.4}",
        clean_artifacts.overall_score
    );
    if clean_artifacts.overall_score < 0.1 {
        println!("✅ Clean audio has minimal artifacts\n");
    }

    // Detect artifacts in transformed audio
    println!("--- Analyzing transformed audio for artifacts ---");
    let transformed_artifacts = detector.detect_artifacts(&transformed_audio, sample_rate)?;

    println!(
        "Overall artifact score: {:.4}",
        transformed_artifacts.overall_score
    );

    // Display detailed artifact analysis
    println!("\nDetailed Artifact Analysis:");
    println!("{:-<60}", "");

    let artifact_types = vec![
        (ArtifactType::Click, "Clicking/Popping"),
        (ArtifactType::Metallic, "Metallic Sound"),
        (ArtifactType::Buzzing, "Buzzing/Distortion"),
        (ArtifactType::PitchVariation, "Pitch Variations"),
        (
            ArtifactType::SpectralDiscontinuity,
            "Spectral Discontinuity",
        ),
        (ArtifactType::EnergySpike, "Energy Spikes"),
        (ArtifactType::HighFrequencyNoise, "High-Freq Noise"),
        (ArtifactType::PhaseArtifact, "Phase Artifacts"),
    ];

    for (artifact_type, name) in artifact_types {
        let score = transformed_artifacts
            .artifact_types
            .get(&artifact_type)
            .copied()
            .unwrap_or(0.0);

        let status = if score < 0.1 {
            "✓ Minimal"
        } else if score < 0.3 {
            "⚠ Low"
        } else if score < 0.6 {
            "⚠ Medium"
        } else {
            "❌ High"
        };

        println!("{:.<30} {:>6.3} {}", name, score, status);
    }

    println!("{:-<60}", "");

    // Display artifact locations
    println!(
        "\nTotal artifacts detected: {}",
        transformed_artifacts.artifact_locations.len()
    );

    // Overall assessment
    println!("\n📊 Overall Assessment:");
    if transformed_artifacts.overall_score < 0.2 {
        println!("  ✅ Excellent - Artifacts well controlled");
    } else if transformed_artifacts.overall_score < 0.4 {
        println!("  ✓ Good - Acceptable artifact levels");
    } else if transformed_artifacts.overall_score < 0.6 {
        println!("  ⚠ Fair - Noticeable artifacts present");
    } else {
        println!("  ❌ Poor - Significant artifacts detected");
        println!("\n  Recommendations:");
        println!("    • Use less extreme transformation parameters");
        println!("    • Increase quality level");
        println!("    • Try different transformation method");
    }

    Ok(())
}

fn generate_clean_audio(sample_rate: u32, duration: f32) -> Vec<f32> {
    let num_samples = (sample_rate as f32 * duration) as usize;

    (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;

            // Clean harmonic content
            let f0 = 200.0;
            let h1 = (2.0 * std::f32::consts::PI * f0 * t).sin();
            let h2 = 0.5 * (2.0 * std::f32::consts::PI * f0 * 2.0 * t).sin();
            let h3 = 0.25 * (2.0 * std::f32::consts::PI * f0 * 3.0 * t).sin();

            // Smooth envelope
            let envelope = ((std::f32::consts::PI * 2.0 * t).sin() * 0.5 + 0.5).powf(1.5);

            (h1 + h2 + h3) * envelope * 0.3
        })
        .collect()
}

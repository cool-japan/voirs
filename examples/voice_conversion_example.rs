//! Voice Conversion Example
//!
//! This example demonstrates the voice conversion capabilities of VoiRS SDK
//! using the actual available API.

use anyhow::Result;
use std::time::Duration;
use voirs::*;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🔄 VoiRS Voice Conversion Example");
    println!("=================================");

    // Ensure output directories exist
    ensure_output_dirs()?;

    // Create components using bridge pattern
    let g2p = create_g2p(G2pBackend::RuleBased);
    let acoustic = create_acoustic(AcousticBackend::Vits);
    let vocoder = create_vocoder(VocoderBackend::HifiGan);

    // Create pipeline with conversion capabilities
    let pipeline = VoirsPipelineBuilder::new()
        .with_g2p(g2p)
        .with_acoustic_model(acoustic)
        .with_vocoder(vocoder)
        .build()
        .await?;

    // Example 1: Basic voice synthesis variations
    println!("\n1. Voice Synthesis Variations");
    println!("-----------------------------");

    let source_text = "Hello, this is a demonstration of voice synthesis variations.";

    // Generate original voice
    let original_audio = pipeline.synthesize(source_text).await?;
    original_audio.save_wav("original_voice.wav")?;
    println!("✓ Generated original voice: original_voice.wav");

    // Create variations using different synthesis configurations
    let variations = [
        (
            "low_pitch",
            SynthesisConfig {
                pitch_shift: -3.0, // Lower pitch
                quality: QualityLevel::High,
                ..Default::default()
            },
        ),
        (
            "high_pitch",
            SynthesisConfig {
                pitch_shift: 3.0, // Higher pitch
                quality: QualityLevel::High,
                ..Default::default()
            },
        ),
        (
            "slow_speech",
            SynthesisConfig {
                speaking_rate: 0.7, // Slower speech
                quality: QualityLevel::High,
                ..Default::default()
            },
        ),
        (
            "fast_speech",
            SynthesisConfig {
                speaking_rate: 1.3, // Faster speech
                quality: QualityLevel::High,
                ..Default::default()
            },
        ),
    ];

    for (name, config) in variations.iter() {
        let varied_audio = pipeline.synthesize_with_config(source_text, config).await?;
        let filename = format!("variation_{}.wav", name);
        varied_audio.save_wav(&filename)?;
        println!("✓ Generated {} variation: {}", name, filename);
    }

    // Example 2: Different quality levels
    println!("\n2. Different Quality Levels");
    println!("---------------------------");

    let quality_text = "This demonstrates different quality levels in voice synthesis.";
    let qualities = [
        ("low", QualityLevel::Low),
        ("medium", QualityLevel::Medium),
        ("high", QualityLevel::High),
    ];

    for (name, quality) in qualities.iter() {
        let config = SynthesisConfig {
            quality: *quality,
            ..Default::default()
        };

        let quality_audio = pipeline
            .synthesize_with_config(quality_text, &config)
            .await?;
        let filename = format!("quality_{}.wav", name);
        quality_audio.save_wav(&filename)?;
        println!("✓ Generated {} quality: {}", name, filename);
    }

    // Example 3: Combined parameter variations
    println!("\n3. Combined Parameter Variations");
    println!("--------------------------------");

    let combined_text = "This combines multiple voice parameters for unique effects.";
    let combinations = [
        (
            "dramatic",
            SynthesisConfig {
                speaking_rate: 0.8,
                pitch_shift: -2.0,
                volume_gain: 3.0,
                quality: QualityLevel::High,
                ..Default::default()
            },
        ),
        (
            "bright",
            SynthesisConfig {
                speaking_rate: 1.1,
                pitch_shift: 2.0,
                volume_gain: 1.0,
                quality: QualityLevel::High,
                ..Default::default()
            },
        ),
        (
            "gentle",
            SynthesisConfig {
                speaking_rate: 0.9,
                pitch_shift: 1.0,
                volume_gain: -2.0,
                quality: QualityLevel::Medium,
                ..Default::default()
            },
        ),
    ];

    for (name, config) in combinations.iter() {
        let combined_audio = pipeline
            .synthesize_with_config(combined_text, config)
            .await?;
        let filename = format!("combined_{}.wav", name);
        combined_audio.save_wav(&filename)?;
        println!("✓ Generated {} combination: {}", name, filename);
    }

    println!("\n🎉 Voice conversion examples completed!");
    println!("All audio files saved to current directory");
    println!("\nGenerated files:");
    println!("- original_voice.wav");
    println!("- variation_*.wav (4 files)");
    println!("- quality_*.wav (3 files)");
    println!("- combined_*.wav (3 files)");

    Ok(())
}

/// Helper function to create output directories
fn ensure_output_dirs() -> std::io::Result<()> {
    std::fs::create_dir_all("examples/output/conversion")?;
    Ok(())
}

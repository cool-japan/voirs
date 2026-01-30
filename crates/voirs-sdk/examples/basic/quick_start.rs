//! Quick Start Example - Demonstrating VoiRS SDK Convenience Methods
//!
//! This example showcases the convenience methods added to VoirsPipeline
//! for common synthesis patterns.

use voirs_sdk::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing for debug output
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("=== VoiRS SDK Quick Start Examples ===\n");

    // Example 1: Quick synthesis with factory method
    println!("1. Quick Default Pipeline");
    println!("   Creating pipeline with VoirsPipeline::default()...");
    let pipeline = VoirsPipelineBuilder::new()
        .with_test_mode(true) // Enable test mode for demo
        .build()
        .await?;
    println!("   ✓ Pipeline created successfully\n");

    // Example 2: Synthesize to file directly
    println!("2. Synthesize Directly to WAV File");
    let temp_dir = std::env::temp_dir();
    let output_path = temp_dir.join("voirs_quick_start.wav");
    println!("   Synthesizing: 'Hello from VoiRS!'");
    pipeline
        .synthesize_to_wav("Hello from VoiRS!", &output_path)
        .await?;
    println!("   ✓ Saved to: {}", output_path.display());
    println!(
        "   ✓ File size: {} bytes\n",
        std::fs::metadata(&output_path)?.len()
    );

    // Example 3: Synthesize with custom speed
    println!("3. Custom Speed Synthesis");
    println!("   Fast (1.5x): 'This is fast speech'");
    let fast = pipeline
        .synthesize_with_speed("This is fast speech", 1.5)
        .await?;
    println!("   ✓ Generated {} samples at 1.5x speed", fast.len());

    println!("   Slow (0.75x): 'This is slow speech'");
    let slow = pipeline
        .synthesize_with_speed("This is slow speech", 0.75)
        .await?;
    println!("   ✓ Generated {} samples at 0.75x speed\n", slow.len());

    // Example 4: Synthesize with custom pitch
    println!("4. Custom Pitch Synthesis");
    println!("   Higher (+3 semitones): 'Higher pitch voice'");
    let higher = pipeline
        .synthesize_with_pitch("Higher pitch voice", 3.0)
        .await?;
    println!("   ✓ Generated {} samples (+3 semitones)", higher.len());

    println!("   Lower (-3 semitones): 'Lower pitch voice'");
    let lower = pipeline
        .synthesize_with_pitch("Lower pitch voice", -3.0)
        .await?;
    println!("   ✓ Generated {} samples (-3 semitones)\n", lower.len());

    // Example 5: Synthesize with both speed and pitch
    println!("5. Combined Speed and Pitch");
    let custom = pipeline
        .synthesize_with_speed_and_pitch(
            "Custom voice with speed and pitch",
            1.2, // 20% faster
            2.0, // 2 semitones higher
        )
        .await?;
    println!(
        "   ✓ Generated {} samples (1.2x speed, +2 semitones)\n",
        custom.len()
    );

    // Example 6: Batch synthesis
    println!("6. Batch Synthesis");
    let texts = vec!["First sentence.", "Second sentence.", "Third sentence."];
    println!("   Synthesizing {} texts...", texts.len());
    let batch_results = pipeline.synthesize_batch(texts).await?;
    println!("   ✓ Generated {} audio buffers", batch_results.len());
    for (i, audio) in batch_results.iter().enumerate() {
        println!(
            "     - Buffer {}: {} samples, {:.2}s duration",
            i + 1,
            audio.len(),
            audio.duration()
        );
    }
    println!();

    // Example 7: Concatenated synthesis
    println!("7. Concatenated Synthesis");
    let parts = vec!["Part one.", "Part two.", "Part three."];
    println!("   Joining {} parts with spaces...", parts.len());
    let concatenated = pipeline.synthesize_concatenated(parts, " ").await?;
    println!(
        "   ✓ Generated single buffer: {} samples, {:.2}s duration\n",
        concatenated.len(),
        concatenated.duration()
    );

    // Example 8: High quality preset
    println!("8. Quality Levels");
    println!("   Creating high-quality pipeline...");
    let hq_pipeline = VoirsPipelineBuilder::new()
        .with_quality(QualityLevel::High)
        .with_test_mode(true)
        .build()
        .await?;
    let hq_audio = hq_pipeline
        .synthesize_with_quality("High quality synthesis", QualityLevel::High)
        .await?;
    println!(
        "   ✓ Generated {} samples at High quality\n",
        hq_audio.len()
    );

    // Example 9: Synthesis with different formats (currently only WAV supported)
    println!("9. File Format Support");
    let wav_path = temp_dir.join("voirs_output.wav");
    pipeline
        .synthesize_to_file("Format test", &wav_path, AudioFormat::Wav)
        .await?;
    println!("   ✓ WAV: {} bytes", std::fs::metadata(&wav_path)?.len());

    // MP3 would fail with UnsupportedFileFormat
    let mp3_path = temp_dir.join("voirs_output.mp3");
    match pipeline
        .synthesize_to_file("Format test", &mp3_path, AudioFormat::Mp3)
        .await
    {
        Err(VoirsError::UnsupportedFileFormat { format, .. }) => {
            println!("   ✗ {}: Not yet implemented (expected)", format);
        }
        _ => println!("   ! Unexpected result for MP3"),
    }
    println!();

    // Cleanup
    println!("=== Cleanup ===");
    std::fs::remove_file(&output_path).ok();
    std::fs::remove_file(&wav_path).ok();
    println!("✓ Temporary files removed");

    println!("\n=== Quick Start Complete! ===");
    println!("All convenience methods demonstrated successfully.");
    println!("\nKey Features Shown:");
    println!("  • Direct file synthesis (synthesize_to_wav)");
    println!("  • Custom speed control (synthesize_with_speed)");
    println!("  • Custom pitch control (synthesize_with_pitch)");
    println!("  • Combined parameters (synthesize_with_speed_and_pitch)");
    println!("  • Batch processing (synthesize_batch)");
    println!("  • Text concatenation (synthesize_concatenated)");
    println!("  • Quality levels (synthesize_with_quality)");
    println!("  • Factory methods (VoirsPipeline::default/high_quality/fast/low_memory)");

    Ok(())
}

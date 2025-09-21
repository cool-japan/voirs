//! Fixed Voice Cloning Example
//!
//! This example demonstrates the actual voice cloning capabilities of VoiRS SDK
//! using the real API instead of placeholder methods.

use std::path::Path;
use voirs_sdk::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🎙️ VoiRS Voice Cloning Example (Fixed)");
    println!("=====================================");

    // Ensure output directories exist
    std::fs::create_dir_all("examples/output/cloning")?;

    // Create pipeline with voice cloning enabled
    let cloning_builder = VoiceClonerBuilder::new()
        .enabled(true)
        .auto_quality_check(true);

    let pipeline = VoirsPipelineBuilder::new()
        .with_voice_cloning(cloning_builder)
        .with_quality(QualityLevel::High)
        .build()
        .await?;

    // Example 1: Voice Cloning with Mock Data
    println!("\n1. Voice Cloning with Mock Samples");
    println!("----------------------------------");

    // Get the voice cloner from the pipeline
    let voice_cloner = pipeline
        .voice_cloner()
        .ok_or_else(|| VoirsError::config_error("Voice cloning not available"))?;

    // Create mock speaker samples
    let mock_samples = create_mock_voice_samples();
    println!("Created {} mock voice samples", mock_samples.len());

    // Perform voice cloning
    let clone_result = voice_cloner
        .clone_voice(
            "demo_speaker".to_string(),
            mock_samples,
            "Hello, this is a demonstration of voice cloning.".to_string(),
            Some(CloningMethod::FewShot),
        )
        .await?;

    println!("✓ Voice cloning completed:");
    println!("  Request ID: {}", clone_result.request_id);
    println!(
        "  Quality Score: {:.2}",
        clone_result
            .quality_metrics
            .get("overall_quality")
            .unwrap_or(&0.0)
    );
    println!("  Similarity Score: {:.2}", clone_result.similarity_score);

    // Example 2: Quick Clone
    println!("\n2. Quick Voice Cloning");
    println!("----------------------");

    let mock_audio = vec![0.0f32; 22050]; // 1 second of silence at 22kHz
    let quick_clone_result = voice_cloner
        .quick_clone(
            mock_audio,
            22050,
            "This is a quick cloning demonstration.".to_string(),
        )
        .await?;

    println!("✓ Quick clone completed:");
    println!("  Request ID: {}", quick_clone_result.request_id);
    println!(
        "  Quality: {:.2}",
        quick_clone_result
            .quality_metrics
            .get("overall_quality")
            .unwrap_or(&0.0)
    );

    // Example 3: Generate Speech
    println!("\n3. Generate Speech");
    println!("------------------");

    let synthesized_audio = pipeline
        .synthesize("Hello from the VoiRS voice synthesis system!")
        .await?;

    synthesized_audio.save_wav("examples/output/cloning/demo_speech.wav")?;
    println!("✓ Generated speech saved to: demo_speech.wav");

    // Example 4: Voice Cloner Statistics
    println!("\n4. Voice Cloner Statistics");
    println!("--------------------------");

    let stats = voice_cloner.get_statistics().await?;
    println!("✓ Cloning Statistics:");
    println!("  Total clones: {}", stats.total_clones);
    println!("  Successful clones: {}", stats.successful_clones);
    println!("  Success rate: {:.1}%", stats.success_rate * 100.0);
    println!("  Cached speakers: {}", stats.cached_speakers);

    // Example 5: List Cached Speakers
    println!("\n5. Cached Speakers");
    println!("------------------");

    let cached_speakers = voice_cloner.list_cached_speakers().await;
    if cached_speakers.is_empty() {
        println!("No speakers currently cached");
    } else {
        println!("Cached speakers:");
        for speaker in cached_speakers {
            println!("  - {}", speaker);
        }
    }

    println!("\n✅ Voice cloning example completed successfully!");
    Ok(())
}

/// Create mock voice samples for demonstration
fn create_mock_voice_samples() -> Vec<VoiceSample> {
    vec![
        VoiceSample::new(
            "sample_1".to_string(),
            vec![0.0f32; 44100], // 1 second of silence at 44.1kHz
            44100,
        ),
        VoiceSample::new(
            "sample_2".to_string(),
            vec![0.0f32; 44100], // 1 second of silence at 44.1kHz
            44100,
        ),
        VoiceSample::new(
            "sample_3".to_string(),
            vec![0.0f32; 44100], // 1 second of silence at 44.1kHz
            44100,
        ),
    ]
}

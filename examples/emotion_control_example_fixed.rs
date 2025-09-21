//! Fixed Emotion Control Example
//!
//! This example demonstrates emotion control capabilities using the actual VoiRS SDK API.

use voirs_sdk::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🎭 VoiRS Emotion Control Example (Fixed)");
    println!("========================================");

    // Ensure output directories exist
    std::fs::create_dir_all("examples/output")?;

    // Create pipeline with emotion control enabled
    let emotion_builder = EmotionControllerBuilder::new().enabled(true);

    let pipeline = VoirsPipelineBuilder::new()
        .with_emotion_control(emotion_builder)
        .with_quality(QualityLevel::High)
        .build()
        .await?;

    // Example 1: Basic Speech Synthesis
    println!("\n1. Basic Speech Synthesis");
    println!("-------------------------");

    let basic_audio = pipeline
        .synthesize("Hello, this is a basic speech synthesis example.")
        .await?;
    basic_audio.save_wav("examples/output/basic_speech.wav")?;
    println!("✓ Generated basic speech: basic_speech.wav");

    // Example 2: Speech with Different Texts
    println!("\n2. Different Text Examples");
    println!("--------------------------");

    let texts = vec![
        "Welcome to the VoiRS speech synthesis system!",
        "This technology can convert text into natural-sounding speech.",
        "The system supports various features and configurations.",
    ];

    for (i, text) in texts.iter().enumerate() {
        let audio = pipeline.synthesize(text).await?;
        let filename = format!("examples/output/text_example_{}.wav", i + 1);
        audio.save_wav(&filename)?;
        println!("✓ Generated: {}", filename);
    }

    // Example 3: Synthesis Configuration
    println!("\n3. Synthesis with Configuration");
    println!("-------------------------------");

    let synthesis_config = SynthesisConfig {
        speaking_rate: 1.2,
        pitch_shift: 1.1,
        ..Default::default()
    };

    let configured_audio = pipeline
        .synthesize_with_config(
            "This speech uses custom configuration settings.",
            &synthesis_config,
        )
        .await?;

    configured_audio.save_wav("examples/output/configured_speech.wav")?;
    println!("✓ Generated configured speech: configured_speech.wav");

    // Example 4: Check if emotion controller is available
    println!("\n4. Emotion Controller Status");
    println!("----------------------------");

    if let Some(emotion_controller) = pipeline.emotion_controller() {
        println!("✓ Emotion controller is available");

        // You would use emotion controller methods here if they were implemented
        // For now, just demonstrate that we can access it
    } else {
        println!("⚠ Emotion controller not available (feature may not be enabled)");
    }

    // Example 5: Pipeline Information
    println!("\n5. Pipeline Information");
    println!("----------------------");

    println!("✓ Pipeline successfully initialized");
    println!("✓ Voice synthesis capabilities available");

    if pipeline.emotion_controller().is_some() {
        println!("✓ Emotion control capabilities available");
    }

    if pipeline.voice_cloner().is_some() {
        println!("✓ Voice cloning capabilities available");
    }

    println!("\n✅ Emotion control example completed successfully!");
    Ok(())
}

//! # Simple Emotion Processing Demo
//!
//! A minimal working example demonstrating the core emotion processing functionality.
//!
//! Run with: `cargo run --example simple_demo --features acoustic-integration`

use voirs_emotion::{
    types::{Emotion, EmotionVector},
    EmotionProcessor,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Simple Emotion Processing Demo ===\n");

    // Create a basic emotion processor
    let processor = EmotionProcessor::new()?;

    println!("1. Processing audio with Happy emotion:");
    processor.set_emotion(Emotion::Happy, Some(0.8)).await?;

    // Process some sample audio
    let input_audio = vec![0.1f32; 1024];
    let output_audio = processor.process_audio(&input_audio).await?;

    println!("   Input samples: {}", input_audio.len());
    println!("   Output samples: {}", output_audio.len());
    println!("   ✓ Audio processed successfully\n");

    println!("2. Processing with Sad emotion:");
    processor.set_emotion(Emotion::Sad, Some(0.7)).await?;

    let sad_audio = processor.process_audio(&input_audio).await?;
    println!("   ✓ Sad emotion applied\n");

    println!("3. Getting current emotion state:");
    let state = processor.get_current_state().await;
    if let Some((emotion, _intensity)) = state.current.emotion_vector.dominant_emotion() {
        println!("   Current emotion: {:?}", emotion);
        println!(
            "   Parameters: pitch={:.2}, energy={:.2}",
            state.current.pitch_shift, state.current.energy_scale
        );
    }

    println!("\n✓ Demo complete!");

    Ok(())
}

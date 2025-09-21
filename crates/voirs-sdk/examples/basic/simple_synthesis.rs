//! # Simple Text-to-Speech Synthesis
//!
//! This example demonstrates the most basic usage of the VoiRS SDK:
//! creating a pipeline and synthesizing speech from text.

use voirs_sdk::prelude::*;

#[tokio::main]
async fn main() -> Result<(), VoirsError> {
    // Initialize logging to see what's happening
    voirs_sdk::logging::init_logging("info")?;
    
    println!("Creating VoiRS pipeline...");
    
    // Create a synthesis pipeline with default settings
    let pipeline = VoirsPipelineBuilder::new()
        .build()
        .await?;
    
    println!("Pipeline created successfully!");
    
    // Synthesize speech from text
    let text = "Hello, world! This is my first speech synthesis with VoiRS.";
    println!("Synthesizing: '{}'", text);
    
    let audio = pipeline.synthesize(text)?;
    
    println!("Synthesis complete! Generated {} samples at {} Hz", 
             audio.len(), audio.sample_rate());
    
    // Save the audio to a WAV file
    let output_path = "simple_synthesis_output.wav";
    audio.save_wav(output_path)?;
    
    println!("Audio saved to: {}", output_path);
    
    // Display basic audio information
    println!("Audio duration: {:.2} seconds", audio.duration());
    println!("Audio format: {} channels, {:.1} kHz", 
             audio.channels(), audio.sample_rate() / 1000.0);
    
    // Optional: Play the audio if system supports it
    #[cfg(feature = "audio_playback")]
    {
        println!("Playing audio...");
        audio.play()?;
    }
    
    Ok(())
}

/// Alternative minimal example
#[tokio::main]
async fn minimal_example() -> Result<(), VoirsError> {
    let pipeline = VoirsPipelineBuilder::new().build().await?;
    let audio = pipeline.synthesize("Hello, world!")?;
    audio.save_wav("minimal_output.wav")?;
    Ok(())
}

/// Example with error handling
#[tokio::main]
async fn example_with_error_handling() {
    match VoirsPipelineBuilder::new().build().await {
        Ok(pipeline) => {
            match pipeline.synthesize("Hello, world!") {
                Ok(audio) => {
                    println!("Successfully synthesized {} samples", audio.len());
                    
                    if let Err(e) = audio.save_wav("error_handling_output.wav") {
                        eprintln!("Failed to save audio: {}", e);
                    } else {
                        println!("Audio saved successfully!");
                    }
                }
                Err(e) => {
                    eprintln!("Synthesis failed: {}", e);
                    eprintln!("Error details: {:?}", e);
                }
            }
        }
        Err(e) => {
            eprintln!("Failed to create pipeline: {}", e);
            eprintln!("Possible causes:");
            eprintln!("  - Missing model files");
            eprintln!("  - Insufficient system resources");
            eprintln!("  - Invalid configuration");
        }
    }
}
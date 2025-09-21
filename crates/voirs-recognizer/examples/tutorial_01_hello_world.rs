//! Tutorial 01: Hello World - Your First Speech Recognition
//!
//! This tutorial introduces you to the absolute basics of speech recognition
//! with VoiRS Recognizer. You'll learn how to create your first recognizer
//! and analyze audio with zero configuration.
//!
//! Learning Objectives:
//! - Set up VoiRS Recognizer with default configuration
//! - Create and analyze sample audio
//! - Understand the basic AudioBuffer structure
//! - Learn about audio analysis results
//!
//! Prerequisites: None - this is where you start!
//!
//! Usage:
//! ```bash
//! cargo run --example tutorial_01_hello_world
//! ```

use std::error::Error;
use voirs_recognizer::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("🎓 Tutorial 01: Hello World - Your First Speech Recognition");
    println!("===========================================================\n");

    // Step 1: Understanding the basics
    println!("📚 Learning Goal: Get familiar with VoiRS Recognizer basics");
    println!("   • Create an audio analyzer");
    println!("   • Generate sample audio");
    println!("   • Analyze audio properties");
    println!("   • Understand the results\n");

    // Step 2: Create default configuration
    println!("🔧 Step 1: Creating default configuration");
    println!(
        "   VoiRS Recognizer uses sensible defaults, so you don't need to configure anything!"
    );

    let config = AudioAnalysisConfig::default();
    println!("   ✅ Default configuration created");
    println!("   • Frame size: {} samples", config.frame_size);
    println!("   • Hop size: {} samples", config.hop_size);
    println!("   • Quality metrics: {}\n", config.quality_metrics);

    // Step 3: Initialize the audio analyzer
    println!("🎛️ Step 2: Initializing audio analyzer");
    println!("   This creates the core engine for audio analysis...");

    let analyzer = AudioAnalyzerImpl::new(config).await?;
    println!("   ✅ Audio analyzer initialized successfully!\n");

    // Step 4: Create sample audio
    println!("🎵 Step 3: Creating sample audio");
    println!("   Let's create a simple sine wave to analyze...");

    let sample_rate = 16000;
    let duration = 1.0; // 1 second
    let frequency = 440.0; // A4 note

    let mut samples = Vec::new();
    for i in 0..(sample_rate as f32 * duration) as usize {
        let t = i as f32 / sample_rate as f32;
        let amplitude = 0.5 * (2.0 * std::f32::consts::PI * frequency * t).sin();
        samples.push(amplitude);
    }

    let audio = AudioBuffer::mono(samples, sample_rate);
    println!("   ✅ Created audio buffer:");
    println!("   • Duration: {:.2} seconds", audio.duration());
    println!("   • Sample rate: {} Hz", audio.sample_rate());
    println!("   • Number of samples: {}", audio.len());
    println!("   • Frequency: {} Hz (A4 note)\n", frequency);

    // Step 5: Analyze the audio
    println!("🔍 Step 4: Analyzing the audio");
    println!("   Now let's see what VoiRS can tell us about our audio...");

    let analysis = analyzer.analyze(&audio, None).await?;

    // Step 6: Display results
    println!("📊 Step 5: Analysis Results");
    println!("   Here's what VoiRS discovered about your audio:");
    println!(
        "   
   🎯 Quality Metrics:"
    );

    // Display quality metrics
    for (metric, value) in &analysis.quality_metrics {
        println!("   • {}: {:.4}", metric, value);
    }

    println!(
        "   
   🎤 Speaker Characteristics:"
    );
    println!("   • Gender: {:?}", analysis.speaker_characteristics.gender);
    println!(
        "   • Age estimate: {:?}",
        analysis.speaker_characteristics.age_range
    );

    println!(
        "   
   🎵 Prosody (Musical Properties):"
    );
    println!(
        "   • Average pitch: {:.1} Hz",
        analysis.prosody.pitch.mean_f0
    );
    println!(
        "   • Pitch range: {:.1} Hz",
        analysis.prosody.pitch.f0_range
    );
    println!("   • Energy: {:.4}", analysis.prosody.energy.mean_energy);

    if let Some(duration) = analysis.processing_duration {
        println!(
            "   
   ⏱️ Performance:"
        );
        println!("   • Processing time: {:.2} ms", duration.as_millis());
        println!(
            "   • Real-time factor: {:.2}x",
            duration.as_secs_f64() / audio.duration() as f64
        );
    }

    // Step 7: Next steps
    println!(
        "
🎉 Congratulations! You've completed Tutorial 01!"
    );
    println!(
        "
📖 What you learned:"
    );
    println!("   • How to create a VoiRS audio analyzer");
    println!("   • How to create and work with AudioBuffer");
    println!("   • How to analyze audio and interpret results");
    println!("   • Basic understanding of audio properties");

    println!(
        "
🚀 Next Steps:"
    );
    println!("   • Tutorial 02: Working with real audio files");
    println!("   • Tutorial 03: Basic speech recognition");
    println!("   • Tutorial 04: Real-time processing");

    println!(
        "
💡 Try This:"
    );
    println!("   • Change the frequency (try 880.0 for A5)");
    println!("   • Modify the duration (try 2.0 seconds)");
    println!("   • Experiment with different amplitudes");

    Ok(())
}

//! Zero-Config Quick Start Example
//!
//! This example demonstrates the fastest way to get started with VoiRS Recognizer
//! with absolutely zero configuration required.

use std::error::Error;
use voirs_recognizer::prelude::*;
use voirs_recognizer::{load_audio, AudioBuffer};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("🎤 VoiRS Recognizer - Zero Config Quick Start");
    println!("===========================================");

    // Step 1: Use default configuration (zero config!)
    println!("📋 Step 1: Using default configuration...");
    let config = ASRConfig::default();
    println!("✅ Default configuration loaded");

    // Step 2: Initialize audio analyzer with defaults (this is what's currently available)
    println!("🔧 Step 2: Initializing audio analyzer...");
    let analyzer_config = AudioAnalysisConfig::default();
    let analyzer = AudioAnalyzerImpl::new(analyzer_config).await?;
    println!("✅ Audio analyzer initialized");

    // Step 3: Quick test with sample audio for demo
    println!("🎵 Step 3: Generating sample audio for demo...");
    let sample_audio = generate_sample_audio_buffer();
    println!("✅ Sample audio generated");

    // Step 4: Perform audio analysis (showing available functionality)
    println!("🔍 Step 4: Performing audio analysis...");
    let analysis_result = analyzer.analyze(&sample_audio, None).await?;

    // Step 5: Display results
    println!("📝 Step 5: Analysis Results");
    println!("===========================");
    println!(
        "🎯 Audio Quality - SNR: {:.2} dB",
        analysis_result.quality_metrics.get("snr").unwrap_or(&0.0)
    );
    println!(
        "📊 Audio Quality - RMS: {:.4}",
        analysis_result.quality_metrics.get("rms").unwrap_or(&0.0)
    );
    println!(
        "🎤 Speaker Analysis - Gender: {:?}",
        analysis_result.speaker_characteristics.gender
    );
    println!(
        "🎵 Prosody - F0 Mean: {:.1} Hz",
        analysis_result.prosody.pitch.mean_f0
    );
    println!(
        "⏱️  Processing Time: {:.2}ms",
        analysis_result
            .processing_duration
            .unwrap_or_default()
            .as_millis()
    );
    println!("✅ Analysis completed successfully!");

    // Step 6: Try with your own audio file (optional)
    println!("\n📁 Step 6: Try with your own audio file");
    println!("======================================");
    println!("To use your own audio file, place it in the current directory and run:");
    println!("cargo run --example zero_config_quickstart -- your_audio.wav");

    // Check if user provided an audio file
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 {
        let audio_path = &args[1];
        println!("🎵 Loading audio file: {}", audio_path);

        match load_user_audio(audio_path) {
            Ok(audio_buffer) => {
                println!("✅ Audio loaded successfully");
                println!(
                    "📊 Audio Info: {}Hz, {} channels, {:.1}s duration",
                    audio_buffer.sample_rate(),
                    audio_buffer.channels(),
                    audio_buffer.duration()
                );
                println!("🔍 Analyzing your audio...");

                let user_analysis = analyzer.analyze(&audio_buffer, None).await?;
                println!("📝 Your Audio Analysis:");
                println!("======================");
                println!(
                    "🎯 SNR: {:.2} dB",
                    user_analysis.quality_metrics.get("snr").unwrap_or(&0.0)
                );
                println!(
                    "📊 RMS Level: {:.4}",
                    user_analysis.quality_metrics.get("rms").unwrap_or(&0.0)
                );
                println!(
                    "🎤 Speaker Gender: {:?}",
                    user_analysis.speaker_characteristics.gender
                );
                println!("🎵 F0 Mean: {:.1} Hz", user_analysis.prosody.pitch.mean_f0);
                println!(
                    "⏱️  Processing: {:.2}ms",
                    user_analysis
                        .processing_duration
                        .unwrap_or_default()
                        .as_millis()
                );

                let snr = user_analysis.quality_metrics.get("snr").unwrap_or(&0.0);
                if *snr > 10.0 {
                    println!("✅ Good audio quality for recognition!");
                } else {
                    println!("⚠️  Audio quality could be improved");
                }
            }
            Err(e) => {
                println!("❌ Could not load audio file: {}", e);
                println!("💡 Supported formats: WAV, FLAC, MP3, OGG");
                println!("💡 Recommended: 16kHz, mono, 16-bit WAV files");
            }
        }
    }

    println!("\n🎉 Quick start example completed!");
    println!("🔗 Next steps:");
    println!("  • Check out other examples in the examples/ directory");
    println!("  • Read the documentation at https://docs.voirs.ai");
    println!("  • Customize configuration for your specific needs");

    Ok(())
}

/// Generate sample audio buffer for demonstration
/// In a real application, you would load actual audio files
fn generate_sample_audio_buffer() -> AudioBuffer {
    // Generate simple synthetic audio data as demonstration
    let sample_rate = 16000;
    let duration_seconds = 2.0; // 2 seconds of audio
    let num_samples = (sample_rate as f32 * duration_seconds) as usize;

    // Generate a simple audio signal (mix of frequencies to simulate speech)
    let samples: Vec<f32> = (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            // Mix of frequencies that might resemble speech patterns
            let f1 = (2.0 * std::f32::consts::PI * 200.0 * t).sin() * 0.3; // Low frequency
            let f2 = (2.0 * std::f32::consts::PI * 800.0 * t).sin() * 0.2; // Mid frequency
            let f3 = (2.0 * std::f32::consts::PI * 1500.0 * t).sin() * 0.1; // High frequency
            (f1 + f2 + f3) * 0.3 // Combine and scale
        })
        .collect();

    AudioBuffer::mono(samples, sample_rate)
}

/// Helper function to load user audio files with automatic format detection and conversion
fn load_user_audio(path: &str) -> Result<AudioBuffer, Box<dyn Error>> {
    // This will automatically:
    // - Detect the audio format
    // - Convert to the required sample rate (16kHz)
    // - Convert to mono if needed
    // - Return normalized audio data
    let audio = load_audio(path)?;
    Ok(audio)
}

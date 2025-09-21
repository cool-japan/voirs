//! Tutorial 02: Working with Real Audio Files
//!
//! This tutorial teaches you how to work with actual audio files,
//! handle different formats, and perform comprehensive audio analysis.
//!
//! Learning Objectives:
//! - Load audio files from disk
//! - Handle different audio formats (WAV, MP3, FLAC, OGG)
//! - Understand audio preprocessing
//! - Perform comprehensive audio quality analysis
//! - Handle common audio processing errors
//!
//! Prerequisites: Complete Tutorial 01
//!
//! Usage:
//! ```bash
//! # With built-in sample audio
//! cargo run --example tutorial_02_real_audio
//!
//! # With your own audio file
//! cargo run --example tutorial_02_real_audio -- /path/to/your/audio.wav
//! ```

use std::env;
use std::error::Error;
use std::path::Path;
use voirs_recognizer::audio_utilities::*;
use voirs_recognizer::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("🎓 Tutorial 02: Working with Real Audio Files");
    println!("==============================================\n");

    // Step 1: Check if user provided audio file
    let args: Vec<String> = env::args().collect();
    let use_custom_file = args.len() > 1;

    if use_custom_file {
        let file_path = &args[1];
        println!("🎵 Using your audio file: {}", file_path);
        process_audio_file(file_path).await?;
    } else {
        println!("🎵 Using built-in sample audio");
        process_sample_audio().await?;
    }

    // Step 2: Show what we learned
    println!(
        "
🎉 Congratulations! You've completed Tutorial 02!"
    );
    println!(
        "
📖 What you learned:"
    );
    println!("   • How to load audio files from disk");
    println!("   • How to handle different audio formats");
    println!("   • How to preprocess audio for better results");
    println!("   • How to perform comprehensive audio analysis");
    println!("   • How to handle audio processing errors");

    println!(
        "
🚀 Next Steps:"
    );
    println!("   • Tutorial 03: Basic speech recognition");
    println!("   • Tutorial 04: Real-time processing");
    println!("   • Tutorial 05: Multi-language support");

    println!(
        "
💡 Try This:"
    );
    println!("   • Try different audio file formats (WAV, MP3, FLAC, OGG)");
    println!("   • Test with various audio qualities and lengths");
    println!("   • Experiment with mono vs stereo audio");

    Ok(())
}

async fn process_audio_file(file_path: &str) -> Result<(), Box<dyn Error>> {
    println!("📁 Step 1: Loading audio file");
    println!("   File path: {}", file_path);

    // Check if file exists
    if !Path::new(file_path).exists() {
        return Err(format!("Audio file not found: {}", file_path).into());
    }

    // Load and preprocess audio
    match load_and_preprocess(file_path).await {
        Ok(audio) => {
            println!("   ✅ Audio loaded successfully!");
            analyze_audio_thoroughly(audio, Some(file_path.to_string())).await?;
        }
        Err(e) => {
            println!("   ❌ Error loading audio: {}", e);
            println!(
                "   
📝 Common solutions:"
            );
            println!("   • Check file path is correct");
            println!("   • Ensure file format is supported (WAV, MP3, FLAC, OGG)");
            println!("   • Verify file is not corrupted");
            println!("   • Check file permissions");
            return Err(e.into());
        }
    }

    Ok(())
}

async fn process_sample_audio() -> Result<(), Box<dyn Error>> {
    println!("📊 Step 1: Creating realistic sample audio");
    println!("   Since no file was provided, we'll create a speech-like sample...");

    // Create more realistic speech-like audio
    let sample_rate = 16000;
    let duration = 2.0; // 2 seconds
    let mut samples = Vec::new();

    for i in 0..(sample_rate as f32 * duration) as usize {
        let t = i as f32 / sample_rate as f32;

        // Simulate speech formants (vocal tract resonances)
        let f1 = 700.0 + 200.0 * (t * 2.0).sin(); // First formant
        let f2 = 1220.0 + 300.0 * (t * 1.5).sin(); // Second formant
        let f3 = 2600.0 + 400.0 * (t * 1.2).sin(); // Third formant

        // Create formant-like signal
        let sample = 0.3 * (2.0 * std::f32::consts::PI * f1 * t).sin()
            + 0.2 * (2.0 * std::f32::consts::PI * f2 * t).sin()
            + 0.1 * (2.0 * std::f32::consts::PI * f3 * t).sin();

        // Add envelope to simulate speech rhythm
        let envelope =
            0.5 * (1.0 + (t * 4.0).sin()) * if (t * 3.0).sin() > 0.0 { 1.0 } else { 0.3 };

        samples.push(sample * envelope);
    }

    let audio = AudioBuffer::mono(samples, sample_rate);
    println!("   ✅ Created speech-like audio sample");

    analyze_audio_thoroughly(audio, None).await?;

    Ok(())
}

async fn analyze_audio_thoroughly(
    audio: AudioBuffer,
    file_path: Option<String>,
) -> Result<(), Box<dyn Error>> {
    println!(
        "
📊 Step 2: Audio File Properties"
    );
    println!("   • Duration: {:.2} seconds", audio.duration());
    println!("   • Sample rate: {} Hz", audio.sample_rate());
    println!(
        "   • Channels: {}",
        if audio.channels() == 1 {
            "Mono"
        } else {
            "Stereo"
        }
    );
    println!("   • Total samples: {}", audio.len());

    if let Some(path) = &file_path {
        println!(
            "   • File size: ~{:.1} KB",
            (audio.len() * 2) as f64 / 1024.0
        ); // Approximate for 16-bit
        println!("   • Format: {}", get_file_extension(path));
    }

    println!(
        "
🔍 Step 3: Audio Quality Analysis"
    );
    println!("   Performing comprehensive audio quality analysis...");

    // Analyze audio quality
    let quality_report = analyze_audio_quality(&audio).await?;

    println!("   ✅ Quality analysis complete!");
    println!(
        "   
   📊 Quality Metrics:"
    );
    println!("   • Overall Score: {:.1}/10", quality_report.overall_score);
    println!("   • Signal-to-Noise Ratio: {:.2} dB", quality_report.snr);
    println!("   • Dynamic Range: {:.2} dB", quality_report.dynamic_range);
    println!(
        "   • Clipping Detection: {}",
        if quality_report.clipping_detected {
            "⚠️ Found"
        } else {
            "✅ None"
        }
    );

    println!(
        "   
   🎯 Quality Assessment:"
    );
    match quality_report.overall_score {
        score if score >= 8.0 => {
            println!("   • Excellent quality - perfect for speech recognition")
        }
        score if score >= 6.0 => println!("   • Good quality - suitable for most applications"),
        score if score >= 4.0 => println!("   • Fair quality - may need preprocessing"),
        _ => println!("   • Poor quality - significant preprocessing recommended"),
    }

    // Show recommendations
    if !quality_report.recommendations().is_empty() {
        println!(
            "   
   💡 Recommendations:"
        );
        for recommendation in &quality_report.recommendations() {
            println!("   • {}", recommendation);
        }
    }

    println!(
        "
🎤 Step 4: Speech-Specific Analysis"
    );
    println!("   Analyzing audio for speech recognition suitability...");

    // Initialize analyzer for speech analysis
    let config = AudioAnalysisConfig::default();
    let analyzer = AudioAnalyzerImpl::new(config).await?;

    let analysis = analyzer.analyze(&audio, None).await?;

    println!("   ✅ Speech analysis complete!");
    println!(
        "   
   🗣️ Speech Characteristics:"
    );
    println!(
        "   • Estimated gender: {:?}",
        analysis.speaker_characteristics.gender
    );
    println!(
        "   • Age range: {:?}",
        analysis.speaker_characteristics.age_range
    );
    println!("   • Mean pitch: {:.1} Hz", analysis.prosody.pitch.mean_f0);
    println!(
        "   • Pitch variation: {:.1} Hz",
        analysis.prosody.pitch.f0_range
    );
    println!(
        "   • Energy level: {:.4}",
        analysis.prosody.energy.mean_energy
    );

    // Performance metrics
    if let Some(duration) = analysis.processing_duration {
        println!(
            "   
   ⚡ Performance:"
        );
        println!("   • Processing time: {:.2} ms", duration.as_millis());
        println!(
            "   • Real-time factor: {:.3}x",
            duration.as_secs_f64() / audio.duration() as f64
        );

        let rtf = duration.as_secs_f64() / audio.duration() as f64;
        if rtf < 0.3 {
            println!("   • ✅ Excellent performance (RTF < 0.3)");
        } else if rtf < 1.0 {
            println!("   • ⚠️ Good performance (RTF < 1.0)");
        } else {
            println!("   • ❌ Slow performance (RTF > 1.0)");
        }
    }

    println!(
        "
🔧 Step 5: Audio Preprocessing Demo"
    );
    println!("   Demonstrating audio optimization for speech recognition...");

    // Optimize audio for recognition
    let optimized_audio = optimize_for_recognition(audio.clone()).await?;
    println!("   ✅ Audio optimized for speech recognition");

    // Show the difference
    println!(
        "   
   📈 Optimization Results:"
    );
    println!("   • Original duration: {:.2}s", audio.duration());
    println!(
        "   • Optimized duration: {:.2}s",
        optimized_audio.duration()
    );
    println!("   • Sample rate: {} Hz", optimized_audio.sample_rate());
    println!(
        "   • Channels: {}",
        if optimized_audio.channels() == 1 {
            "Mono (optimized)"
        } else {
            "Stereo"
        }
    );

    Ok(())
}

fn get_file_extension(path: &str) -> &str {
    Path::new(path)
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("unknown")
}

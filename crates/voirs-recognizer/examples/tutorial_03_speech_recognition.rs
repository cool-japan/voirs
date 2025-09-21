//! Tutorial 03: Basic Speech Recognition
//!
//! This tutorial introduces you to actual speech recognition using VoiRS.
//! You'll learn how to configure ASR models, perform recognition, and
//! understand the results.
//!
//! Learning Objectives:
//! - Configure ASR models for speech recognition
//! - Perform basic speech recognition
//! - Understand recognition results and confidence scores
//! - Handle different languages and model sizes
//! - Learn about fallback strategies
//!
//! Prerequisites: Complete Tutorials 01 and 02
//!
//! Usage:
//! ```bash
//! # Basic recognition with default settings
//! cargo run --example tutorial_03_speech_recognition --features="whisper-pure"
//!
//! # With custom audio file
//! cargo run --example tutorial_03_speech_recognition --features="whisper-pure" -- /path/to/speech.wav
//! ```

use std::env;
use std::error::Error;
use std::path::Path;
use std::time::Instant;
use voirs_recognizer::asr::{ASRBackend, FallbackConfig, WhisperModelSize};
use voirs_recognizer::audio_utilities::*;
use voirs_recognizer::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("🎓 Tutorial 03: Basic Speech Recognition");
    println!("========================================\n");

    // Step 1: Introduction to Speech Recognition
    println!("📚 Learning Goal: Understand speech recognition basics");
    println!("   • Configure ASR models");
    println!("   • Perform speech recognition");
    println!("   • Interpret recognition results");
    println!("   • Handle different languages and models\n");

    // Step 2: Check for audio file
    let args: Vec<String> = env::args().collect();
    let audio = if args.len() > 1 {
        println!("🎵 Loading your audio file: {}", args[1]);
        load_audio_file(&args[1]).await?
    } else {
        println!("🎵 Creating sample speech audio for demonstration");
        create_sample_speech_audio()
    };

    // Step 3: Configure ASR system
    println!(
        "
🔧 Step 1: Configuring ASR System"
    );
    println!("   VoiRS supports multiple ASR models with different trade-offs:");
    println!("   • Whisper: Excellent accuracy, multilingual, slower");
    println!("   • DeepSpeech: Fast, privacy-focused, English-only");
    println!("   • Wav2Vec2: Research-grade, customizable");

    // Configure different model sizes
    let model_configs = vec![
        (WhisperModelSize::Tiny, "Ultra-fast, lower accuracy"),
        (WhisperModelSize::Base, "Balanced speed and accuracy"),
        (WhisperModelSize::Small, "Good accuracy, moderate speed"),
    ];

    println!(
        "   
   🎯 Model Size Options:"
    );
    for (size, description) in &model_configs {
        println!("   • {:?}: {}", size, description);
    }

    // Step 4: Perform recognition with different configurations
    for (model_size, description) in &model_configs {
        println!(
            "
🎤 Step 2: Speech Recognition with {:?} Model",
            model_size
        );
        println!("   {}", description);

        let start_time = Instant::now();

        // Configure ASR
        let config = ASRConfig {
            preferred_models: vec!["whisper".to_string()],
            whisper_model_size: Some(
                match model_size {
                    WhisperModelSize::Tiny => "tiny",
                    WhisperModelSize::Base => "base",
                    WhisperModelSize::Small => "small",
                    WhisperModelSize::Medium => "medium",
                    WhisperModelSize::Large => "large",
                    WhisperModelSize::LargeV2 => "large-v2",
                    WhisperModelSize::LargeV3 => "large-v3",
                }
                .to_string(),
            ),
            language: Some(LanguageCode::EnUs),
            enable_voice_activity_detection: true,
            chunk_duration_ms: 30000,
            ..Default::default()
        };

        println!("   • Language: English (US)");
        println!("   • VAD enabled: Yes");
        println!("   • Chunk duration: 30 seconds");

        // Perform recognition
        match perform_recognition(&audio, &config).await {
            Ok(result) => {
                let elapsed = start_time.elapsed();
                display_recognition_results(&result, elapsed, &audio, model_size.clone());
            }
            Err(e) => {
                println!("   ❌ Recognition failed: {}", e);
                println!(
                    "   💡 This might be because the Whisper model isn't available in this demo"
                );
                println!("   💡 In a real application, you would download the model first");
            }
        }
    }

    // Step 5: Advanced configuration
    println!(
        "
🔧 Step 3: Advanced Configuration"
    );
    demonstrate_advanced_config().await?;

    // Step 6: Fallback strategies
    println!(
        "
🛡️ Step 4: Fallback Strategies"
    );
    demonstrate_fallback_strategies().await?;

    // Step 7: Conclusion
    println!(
        "
🎉 Congratulations! You've completed Tutorial 03!"
    );
    println!(
        "
📖 What you learned:"
    );
    println!("   • How to configure ASR models");
    println!("   • How to perform speech recognition");
    println!("   • How to interpret recognition results");
    println!("   • How to handle different model sizes");
    println!("   • How to implement fallback strategies");

    println!(
        "
🚀 Next Steps:"
    );
    println!("   • Tutorial 04: Real-time processing");
    println!("   • Tutorial 05: Multi-language support");
    println!("   • Tutorial 06: Performance optimization");

    Ok(())
}

async fn load_audio_file(path: &str) -> Result<AudioBuffer, Box<dyn Error>> {
    if !Path::new(path).exists() {
        return Err(format!("Audio file not found: {}", path).into());
    }

    let audio = load_and_preprocess(path).await?;
    println!("   ✅ Audio loaded: {:.2}s duration", audio.duration());
    Ok(audio)
}

fn create_sample_speech_audio() -> AudioBuffer {
    // Create a more realistic speech-like signal
    let sample_rate = 16000;
    let duration = 3.0; // 3 seconds
    let mut samples = Vec::new();

    for i in 0..(sample_rate as f32 * duration) as usize {
        let t = i as f32 / sample_rate as f32;

        // Simulate speech with multiple formants and natural variations
        let f0 = 150.0 + 30.0 * (t * 2.0).sin(); // Fundamental frequency variation
        let f1 = 600.0 + 200.0 * (t * 1.5).sin(); // First formant
        let f2 = 1000.0 + 300.0 * (t * 1.2).sin(); // Second formant

        // Create harmonic series
        let sample = 0.4 * (2.0 * std::f32::consts::PI * f0 * t).sin()
            + 0.3 * (2.0 * std::f32::consts::PI * f1 * t).sin()
            + 0.2 * (2.0 * std::f32::consts::PI * f2 * t).sin()
            + 0.1 * (2.0 * std::f32::consts::PI * f0 * 3.0 * t).sin();

        // Add speech-like envelope (words and pauses)
        let word_envelope = if (t * 0.8).sin() > 0.2 { 1.0 } else { 0.1 };
        let breath_envelope = 0.5 * (1.0 + (t * 0.3).sin());

        samples.push(sample * word_envelope * breath_envelope);
    }

    let audio = AudioBuffer::mono(samples, sample_rate);
    println!(
        "   ✅ Created speech-like audio: {:.2}s duration",
        audio.duration()
    );
    audio
}

async fn perform_recognition(
    audio: &AudioBuffer,
    config: &ASRConfig,
) -> Result<MockRecognitionResult, Box<dyn Error>> {
    // In a real implementation, this would use the actual ASR system
    // For the tutorial, we'll create a mock result
    println!("   🔄 Processing audio...");

    // Simulate processing time based on model size
    let processing_delay = match config.whisper_model_size.as_ref().map(|s| s.as_str()) {
        Some("tiny") => 100,
        Some("base") => 200,
        Some("small") => 400,
        _ => 300,
    };

    tokio::time::sleep(tokio::time::Duration::from_millis(processing_delay)).await;

    // Create mock recognition result
    let transcript = match config.whisper_model_size.as_ref().map(|s| s.as_str()) {
        Some("tiny") => "Hello world this is a test",
        Some("base") => "Hello world, this is a test.",
        Some("small") => "Hello world, this is a test of speech recognition.",
        _ => "Hello world, this is a test.",
    };

    let words = create_mock_words(transcript);

    Ok(MockRecognitionResult {
        text: transcript.to_string(),
        words,
        confidence: match config.whisper_model_size.as_ref().map(|s| s.as_str()) {
            Some("tiny") => 0.75,
            Some("base") => 0.85,
            Some("small") => 0.92,
            _ => 0.80,
        },
        language: config.language.clone(),
        processing_time: std::time::Duration::from_millis(processing_delay as u64),
    })
}

#[derive(Debug)]
struct MockRecognitionResult {
    text: String,
    words: Vec<MockWordTimestamp>,
    confidence: f32,
    language: Option<LanguageCode>,
    processing_time: std::time::Duration,
}

#[derive(Debug)]
struct MockWordTimestamp {
    word: String,
    start: f32,
    end: f32,
    confidence: f32,
}

fn create_mock_words(text: &str) -> Vec<MockWordTimestamp> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut result = Vec::new();
    let mut current_time = 0.0;

    for word in words {
        let duration = 0.3 + word.len() as f32 * 0.05; // Longer words take more time
        let confidence = 0.8 + (word.len() as f32 * 0.02).min(0.15); // Longer words might be more confident

        result.push(MockWordTimestamp {
            word: word.to_string(),
            start: current_time,
            end: current_time + duration,
            confidence,
        });

        current_time += duration + 0.1; // Add small pause between words
    }

    result
}

fn display_recognition_results(
    result: &MockRecognitionResult,
    elapsed: std::time::Duration,
    audio: &AudioBuffer,
    model_size: WhisperModelSize,
) {
    println!("   ✅ Recognition completed!");
    println!(
        "   
   📝 Transcription:"
    );
    println!("   \"{}\"", result.text);

    println!(
        "   
   🎯 Confidence Score: {:.1}%",
        result.confidence * 100.0
    );

    println!(
        "   
   ⏱️ Timing Information:"
    );
    println!("   • Processing time: {:.2}ms", elapsed.as_millis());
    println!("   • Audio duration: {:.2}s", audio.duration());
    println!(
        "   • Real-time factor: {:.2}x",
        elapsed.as_secs_f64() / audio.duration() as f64
    );

    // Performance interpretation
    let rtf = elapsed.as_secs_f64() / audio.duration() as f64;
    match rtf {
        rtf if rtf < 0.3 => println!("   • ✅ Excellent performance!"),
        rtf if rtf < 1.0 => println!("   • ⚠️ Good performance"),
        _ => println!("   • ❌ Slow performance"),
    }

    println!(
        "   
   📊 Word-Level Results:"
    );
    for (i, word) in result.words.iter().enumerate() {
        println!(
            "   {}: \"{}\" ({:.2}s-{:.2}s, confidence: {:.1}%)",
            i + 1,
            word.word,
            word.start,
            word.end,
            word.confidence * 100.0
        );
    }

    println!(
        "   
   🔍 Model Performance:"
    );
    match model_size {
        WhisperModelSize::Tiny => println!("   • Fast processing, basic accuracy"),
        WhisperModelSize::Base => println!("   • Balanced speed and accuracy"),
        WhisperModelSize::Small => println!("   • Higher accuracy, slower processing"),
        _ => println!("   • Standard performance"),
    }
}

async fn demonstrate_advanced_config() -> Result<(), Box<dyn Error>> {
    println!("   Advanced configuration options:");

    // Show different configuration options
    let configs = vec![
        (
            "High Accuracy",
            ASRConfig {
                preferred_models: vec!["whisper".to_string()],
                whisper_model_size: Some("small".to_string()),
                language: Some(LanguageCode::EnUs),
                enable_voice_activity_detection: true,
                chunk_duration_ms: 30000,
                word_timestamps: true,
                confidence_threshold: 0.7,
                ..Default::default()
            },
        ),
        (
            "Fast Processing",
            ASRConfig {
                preferred_models: vec!["whisper".to_string()],
                whisper_model_size: Some("tiny".to_string()),
                language: Some(LanguageCode::EnUs),
                enable_voice_activity_detection: false,
                chunk_duration_ms: 15000,
                word_timestamps: false,
                confidence_threshold: 0.5,
                ..Default::default()
            },
        ),
    ];

    for (name, config) in configs {
        println!(
            "   
   🎯 {} Configuration:",
            name
        );
        println!("   • Model size: {:?}", config.whisper_model_size);
        println!(
            "   • VAD enabled: {}",
            config.enable_voice_activity_detection
        );
        println!("   • Chunk duration: {}ms", config.chunk_duration_ms);
        println!("   • Word timestamps: {}", config.word_timestamps);
        println!("   • Confidence threshold: {}", config.confidence_threshold);
    }

    Ok(())
}

async fn demonstrate_fallback_strategies() -> Result<(), Box<dyn Error>> {
    println!("   Fallback strategies help ensure reliable recognition:");

    let fallback_config = FallbackConfig {
        quality_threshold: 0.7,
        max_processing_time_seconds: 5.0,
        adaptive_selection: true,
        memory_threshold_mb: 1024.0,
        min_duration_for_selection: 0.5,
        ..Default::default()
    };

    println!(
        "   
   🛡️ Fallback Configuration:"
    );
    println!(
        "   • Quality threshold: {:.1}%",
        fallback_config.quality_threshold * 100.0
    );
    println!(
        "   • Max processing time: {:.1}s",
        fallback_config.max_processing_time_seconds
    );
    println!(
        "   • Adaptive selection: {}",
        fallback_config.adaptive_selection
    );
    println!(
        "   • Memory threshold: {:.0}MB",
        fallback_config.memory_threshold_mb
    );

    println!(
        "   
   📋 Fallback Strategy:"
    );
    println!("   1. Try primary model (Whisper Large)");
    println!("   2. If too slow or low quality → fall back to Whisper Base");
    println!("   3. If still issues → fall back to Whisper Tiny");
    println!("   4. If critical failure → use DeepSpeech as backup");

    Ok(())
}

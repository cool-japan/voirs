//! Basic Configuration Example - VoiRS Text-to-Speech
//!
//! This example demonstrates how to:
//! 1. Configure VoiRS synthesis settings
//! 2. Handle errors gracefully
//! 3. Use different quality levels and backends
//! 4. Validate configurations before synthesis
//!
//! ## Key concepts:
//! - SynthesisConfig: Controls voice characteristics
//! - Error handling: Robust error management
//! - Backend selection: Choosing the right components
//! - Quality levels: Balancing quality vs performance
//!
//! ## System Requirements:
//! - **RAM**: 6GB+ for multiple quality tests (4GB minimum)
//! - **CPU**: Multi-core recommended for parallel testing
//! - **Storage**: 3GB free space (models + output files)
//!
//! ## Platform Performance Notes:
//!
//! ### macOS:
//! - **Metal GPU**: Automatically used if available for acceleration
//! - **Audio Latency**: CoreAudio provides low latency (~5-20ms)
//! - **Memory**: Unified memory architecture helps with large models
//!
//! ### Linux:
//! - **Audio System**: Works with ALSA, PulseAudio, or JACK
//! - **GPU Support**: CUDA/OpenCL available with proper drivers
//! - **Memory**: May need swap for memory-intensive configurations
//!
//! ### Windows:
//! - **Audio API**: Uses WASAPI for modern audio support
//! - **GPU**: DirectML acceleration available on recent systems
//! - **Antivirus**: May need exclusions for temporary audio files
//!
//! ## Running this example:
//!
//! ### Quick test (single configuration):
//! ```bash
//! cargo run --example basic_configuration --no-default-features
//! ```
//!
//! ### Full test (all configurations):
//! ```bash
//! cargo run --example basic_configuration
//! ```
//!
//! ### Memory-optimized test:
//! ```bash
//! cargo run --example basic_configuration --features="memory-opt"
//! ```
//!
//! ### Platform-specific optimizations:
//! ```bash
//! # macOS with Metal acceleration
//! cargo run --example basic_configuration --features="metal"
//! 
//! # Linux with CUDA (if available)
//! cargo run --example basic_configuration --features="cuda"
//! 
//! # Windows with DirectML
//! cargo run --example basic_configuration --features="directml"
//! ```
//!
//! ## Configuration Guidelines:
//!
//! ### For Real-time Applications:
//! - Use QualityLevel::Medium or Low
//! - Set streaming_chunk_size to 512-1024
//! - Disable enhancement for lower latency
//!
//! ### For Batch Processing:
//! - Use QualityLevel::High or Ultra
//! - Enable enhancement for best quality
//! - Larger chunk sizes (2048-4096) for efficiency
//!
//! ### For Memory-Constrained Systems:
//! - Use QualityLevel::Low
//! - Reduce sample_rate to 22050 or 16000
//! - Disable enhancement to save memory
//!
//! ## Expected Output Files:
//! - `config_high_44100.wav` - High quality, 44.1kHz
//! - `config_medium_22050.wav` - Medium quality, 22.05kHz
//! - `config_medium_22050.wav` - Modified voice example
//!
//! ## Troubleshooting Configuration Issues:
//!
//! ### "Configuration validation failed":
//! - Check parameter ranges in validation function
//! - Ensure sample rates are supported (8000, 16000, 22050, 44100, 48000)
//! - Verify emotion parameters if emotion is enabled
//!
//! ### "Pipeline build failed":
//! - Insufficient memory for quality level - try lower quality
//! - Missing models - check model loading paths
//! - Audio device conflicts - close other audio applications
//!
//! ### Performance Issues:
//! - High RTF (>1.0) indicates slower than real-time
//! - Reduce quality level or sample rate
//! - Enable platform-specific acceleration features

use anyhow::{Context, Result as AnyhowResult};
use std::time::Instant;
use voirs::{
    create_acoustic, create_g2p, create_vocoder, AcousticBackend, AudioFormat, G2pBackend,
    LanguageCode, QualityLevel, Result, SynthesisConfig, VocoderBackend, VoirsPipelineBuilder,
};

#[tokio::main]
async fn main() -> AnyhowResult<()> {
    // Initialize logging with error level details
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();

    println!("⚙️  VoiRS Configuration & Error Handling Example");
    println!("===============================================");
    println!();

    // Demonstrate multiple configurations
    let configs = create_example_configs();

    for (name, config) in configs {
        println!("🔧 Testing configuration: {}", name);

        match test_configuration(config).await {
            Ok(stats) => {
                println!("   ✅ Success! {}", stats);
            }
            Err(e) => {
                println!("   ❌ Failed: {}", e);
                println!("   💡 Tip: Check the error message above for troubleshooting");
            }
        }
        println!();
    }

    println!("🎯 Configuration testing complete!");
    println!("   Check the generated audio files to hear the differences.");

    Ok(())
}

/// Creates various example configurations to demonstrate different settings
fn create_example_configs() -> Vec<(&'static str, SynthesisConfig)> {
    vec![
        (
            "High Quality (Slow)",
            SynthesisConfig {
                speaking_rate: 1.0,
                pitch_shift: 0.0,
                volume_gain: 0.0,
                enable_enhancement: true,
                output_format: AudioFormat::Wav,
                sample_rate: 44100, // High sample rate
                quality: QualityLevel::High,
                language: LanguageCode::EnUs,
                effects: Vec::new(),
                streaming_chunk_size: None,
                seed: Some(42),
                enable_emotion: false,
                emotion_type: None,
                emotion_intensity: 0.7,
                emotion_preset: None,
                auto_emotion_detection: false,
                ..Default::default()
            },
        ),
        (
            "Fast Quality (Balanced)",
            SynthesisConfig {
                speaking_rate: 1.2, // Slightly faster
                pitch_shift: 0.0,
                volume_gain: 0.0,
                enable_enhancement: false, // Disable for speed
                output_format: AudioFormat::Wav,
                sample_rate: 22050, // Standard sample rate
                quality: QualityLevel::Medium,
                language: LanguageCode::EnUs,
                effects: Vec::new(),
                streaming_chunk_size: Some(1024),
                seed: Some(42),
                enable_emotion: false,
                emotion_type: None,
                emotion_intensity: 0.7,
                emotion_preset: None,
                auto_emotion_detection: false,
                ..Default::default()
            },
        ),
        (
            "Voice Modification",
            SynthesisConfig {
                speaking_rate: 0.8, // Slower speech
                pitch_shift: 2.0,   // Higher pitch
                volume_gain: 3.0,   // Louder
                enable_enhancement: true,
                output_format: AudioFormat::Wav,
                sample_rate: 22050,
                quality: QualityLevel::Medium,
                language: LanguageCode::EnUs,
                effects: Vec::new(),
                streaming_chunk_size: None,
                seed: Some(123), // Different seed for variation
                enable_emotion: false,
                emotion_type: None,
                emotion_intensity: 0.7,
                emotion_preset: None,
                auto_emotion_detection: false,
                ..Default::default()
            },
        ),
    ]
}

/// Tests a specific configuration and returns synthesis statistics
async fn test_configuration(config: SynthesisConfig) -> AnyhowResult<String> {
    // Sample text for this configuration
    let text = match config.quality {
        QualityLevel::High => "This is high-quality synthesis with maximum fidelity.",
        QualityLevel::Medium => "This demonstrates balanced quality and performance.",
        QualityLevel::Low => "Low quality synthesis prioritizes speed over quality.",
        QualityLevel::Ultra => "Ultra-high quality synthesis with maximum detail and accuracy.",
    };

    println!("   📝 Text: \"{}\"", text);
    println!(
        "   ⚙️  Config: SR={:.1}, Pitch={:+.1}, Quality={:?}",
        config.speaking_rate, config.pitch_shift, config.quality
    );

    // Step 1: Validate configuration
    validate_config(&config).context("Configuration validation failed")?;

    // Step 2: Create components with error handling
    let g2p = create_g2p(G2pBackend::RuleBased);
    let acoustic = create_acoustic(AcousticBackend::Vits);
    let vocoder = create_vocoder(VocoderBackend::HifiGan);

    // Step 3: Build pipeline with proper error context
    let pipeline = VoirsPipelineBuilder::new()
        .with_g2p(g2p)
        .with_acoustic_model(acoustic)
        .with_vocoder(vocoder)
        .with_quality(config.quality)
        .with_speaking_rate(config.speaking_rate)
        .build()
        .await
        .context("Failed to build synthesis pipeline")?;

    // Step 4: Synthesize with timing
    let start_time = Instant::now();
    let audio = pipeline
        .synthesize_with_config(text, &config)
        .await
        .context("Speech synthesis failed")?;
    let synthesis_time = start_time.elapsed();

    // Step 5: Calculate statistics
    let duration = audio.duration();
    let rtf = synthesis_time.as_secs_f32() / duration;
    let samples = audio.samples();
    let peak = samples.iter().map(|s| s.abs()).fold(0.0, f32::max);
    let rms = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();

    // Step 6: Save with unique filename
    let filename = format!(
        "config_{}_{}.wav",
        config.quality.to_string().to_lowercase(),
        config.sample_rate
    );

    audio
        .save_wav(&filename)
        .with_context(|| format!("Failed to save audio to {}", filename))?;

    // Return success statistics
    Ok(format!(
        "Duration: {:.2}s, RTF: {:.2}x, Peak: {:.3}, RMS: {:.3}, File: {}",
        duration, rtf, peak, rms, filename
    ))
}

/// Validates synthesis configuration for common issues
fn validate_config(config: &SynthesisConfig) -> AnyhowResult<()> {
    // Check speaking rate bounds
    if config.speaking_rate <= 0.0 || config.speaking_rate > 3.0 {
        anyhow::bail!(
            "Speaking rate must be between 0.0 and 3.0, got {}",
            config.speaking_rate
        );
    }

    // Check pitch shift bounds
    if config.pitch_shift < -12.0 || config.pitch_shift > 12.0 {
        anyhow::bail!(
            "Pitch shift must be between -12.0 and 12.0 semitones, got {}",
            config.pitch_shift
        );
    }

    // Check volume gain bounds
    if config.volume_gain < -20.0 || config.volume_gain > 20.0 {
        anyhow::bail!(
            "Volume gain must be between -20.0 and 20.0 dB, got {}",
            config.volume_gain
        );
    }

    // Check sample rate
    let valid_rates = [8000, 16000, 22050, 44100, 48000];
    if !valid_rates.contains(&config.sample_rate) {
        anyhow::bail!(
            "Sample rate must be one of {:?}, got {}",
            valid_rates,
            config.sample_rate
        );
    }

    // Check emotion intensity if emotion is enabled
    if config.enable_emotion && (config.emotion_intensity < 0.0 || config.emotion_intensity > 1.0) {
        anyhow::bail!(
            "Emotion intensity must be between 0.0 and 1.0, got {}",
            config.emotion_intensity
        );
    }

    println!("   ✅ Configuration validation passed");
    Ok(())
}

// Helper trait to convert QualityLevel to string
trait QualityToString {
    fn to_string(&self) -> String;
}

impl QualityToString for QualityLevel {
    fn to_string(&self) -> String {
        match self {
            QualityLevel::Low => "low".to_string(),
            QualityLevel::Medium => "medium".to_string(),
            QualityLevel::High => "high".to_string(),
            QualityLevel::Ultra => "ultra".to_string(),
        }
    }
}

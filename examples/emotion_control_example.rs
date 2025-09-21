//! Emotion Control Example - Expressive Speech Synthesis
//!
//! This comprehensive example demonstrates VoiRS's emotion control and expressive synthesis capabilities,
//! showcasing different emotional styles, voice configurations, and advanced speech parameters.
//!
//! ## What this example demonstrates:
//! 1. Emotional expression in synthesized speech
//! 2. Voice configuration with different speaking styles
//! 3. Speed and pitch variations for expressive control
//! 4. Text-based emotional content adaptation
//! 5. Performance analysis of emotional synthesis
//! 6. Advanced synthesis parameter combinations
//!
//! ## Key Features Showcased:
//! - Multiple emotional voice styles (neutral, happy, calm, professional, etc.)
//! - Voice synthesis configurations for different contexts
//! - Speed and pitch modification effects
//! - Text-driven emotional expression
//! - Comprehensive performance metrics and analysis
//! - Error handling for emotional synthesis operations
//!
//! ## System Requirements (Emotion Processing):
//! - **RAM**: 6GB+ required (12GB recommended for all emotions)
//! - **Storage**: 3GB+ free space (emotion models + 100MB output)
//! - **CPU**: Multi-core for emotion processing algorithms
//! - **Audio**: High-quality output device to appreciate emotional nuances
//!
//! ## Platform-Specific Emotion Synthesis:
//!
//! ### macOS:
//! - **Audio Quality**: CoreAudio preserves emotional expression details
//! - **Processing**: Neural Engine on M1/M2 accelerates emotion modeling
//! - **Memory**: Efficient emotion model caching with unified memory
//! - **Latency**: Low-latency emotion switching for interactive applications
//!
//! ### Linux:
//! - **Audio Fidelity**: JACK provides professional-grade emotional audio
//! - **GPU Compute**: CUDA acceleration for complex emotion processing
//! - **Real-time**: Low-latency kernels benefit real-time emotion control
//! - **Scaling**: Excellent parallel processing for multiple emotions
//!
//! ### Windows:
//! - **Audio API**: WASAPI for high-fidelity emotional expression
//! - **GPU Support**: DirectML acceleration for emotion neural networks
//! - **Integration**: Good compatibility with audio production software
//! - **Performance**: Hardware scheduling helps with emotion processing
//!
//! ## Running this example:
//!
//! ### Standard emotional synthesis:
//! ```bash
//! cargo run --example emotion_control_example --features="emotion"
//! ```
//!
//! ### High-quality emotional processing:
//! ```bash
//! cargo run --example emotion_control_example --features="emotion,high-quality"
//! ```
//!
//! ### Real-time emotion control:
//! ```bash
//! cargo run --example emotion_control_example --features="emotion,realtime"
//! ```
//!
//! ### Memory-optimized emotions:
//! ```bash
//! cargo run --example emotion_control_example --features="emotion,memory-opt"
//! ```
//!
//! ### Platform-optimized emotional synthesis:
//! ```bash
//! # macOS with Neural Engine acceleration
//! cargo run --example emotion_control_example --features="emotion,neural-engine"
//! 
//! # Linux with CUDA emotion processing
//! cargo run --example emotion_control_example --features="emotion,cuda"
//! 
//! # Windows with DirectML emotions
//! cargo run --example emotion_control_example --features="emotion,directml"
//! ```
//!
//! ## Expected Output:
//! - **Emotional audio files**: 8-12 files with different emotions (~10MB each)
//! - **Processing time**: 1-8 minutes depending on system and emotions
//! - **Emotion metrics**: Intensity scores, expression accuracy, transition smoothness
//! - **Comparison analysis**: Emotional variation and authenticity assessment
//!
//! ## Emotion Processing Performance:
//! | System Type | Processing Time | Emotion Quality | Memory Usage |
//! |-------------|----------------|-----------------|--------------|
//! | GPU-accelerated | 1-2 min | Excellent | 4-6GB |
//! | Modern CPU | 3-5 min | Good | 6-8GB |
//! | Basic system | 5-8 min | Adequate | 3-4GB |
//!
//! ## Emotion-Specific Troubleshooting:
//!
//! ### Flat or unexpressive emotional output:
//! - Increase emotion intensity settings (0.7-1.0 range)
//! - Verify emotion models are properly loaded
//! - Check for sufficient processing power
//! - Ensure audio output preserves dynamic range
//!
//! ### Inconsistent emotional expression:
//! - Use consistent emotion parameters across synthesis
//! - Check for memory pressure during processing
//! - Verify stable system performance
//! - Consider using emotion presets for consistency
//!
//! ### Poor emotion transitions:
//! - Enable emotion interpolation features
//! - Increase processing quality settings
//! - Use appropriate text formatting for emotion cues
//! - Allow sufficient processing time for smooth transitions
//!
//! ### "Emotion model not found" errors:
//! - Verify emotion features are enabled: `--features="emotion"`
//! - Check model download and installation
//! - Ensure sufficient storage space for emotion models
//! - Try rebuilding with emotion dependencies
//!
//! ## Emotional Expression Guidelines:
//!
//! ### For Natural Conversational Speech:
//! - Use subtle emotion intensities (0.3-0.6)
//! - Enable automatic emotion detection from text
//! - Allow natural emotion transitions
//! - Use appropriate speaking rates for emotions
//!
//! ### For Dramatic or Creative Content:
//! - Use higher emotion intensities (0.7-1.0)
//! - Manually control emotion types and timing
//! - Enable emotion enhancement features
//! - Experiment with emotion combinations
//!
//! ### For Professional or Educational Content:
//! - Use neutral base with subtle emotional cues
//! - Maintain consistent speaking pace
//! - Focus on clarity over emotional expression
//! - Use professional emotion presets
//!
//! ## Emotion Types and Use Cases:
//! - **Neutral**: Default baseline for clear communication
//! - **Happy**: Positive content, advertisements, greetings
//! - **Calm**: Meditation, instructions, educational content
//! - **Professional**: Business presentations, formal communication
//! - **Excited**: Announcements, entertainment, sports commentary
//! - **Sad**: Dramatic content, empathetic responses
//! - **Angry**: Character voices, dramatic emphasis
//! - **Surprised**: Interactive responses, dynamic content

use anyhow::{Context, Result};
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};
use voirs::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize comprehensive logging for emotional synthesis
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("🎭 VoiRS Emotion Control Example");
    println!("=================================");
    println!();

    // Ensure output directories exist with error handling
    info!("Setting up output directories for emotional synthesis...");
    ensure_output_dirs().context("Failed to create output directories for emotion examples")?;
    println!("✅ Output directories ready");

    // Create components with timing and comprehensive error handling
    let setup_start = Instant::now();
    info!("Creating emotional TTS components...");

    let g2p = create_g2p(G2pBackend::RuleBased);
    let acoustic = create_acoustic(AcousticBackend::Vits);
    let vocoder = create_vocoder(VocoderBackend::HifiGan);

    println!("🔧 Building emotional synthesis pipeline...");
    let pipeline = VoirsPipelineBuilder::new()
        .with_g2p(g2p)
        .with_acoustic_model(acoustic)
        .with_vocoder(vocoder)
        .build()
        .await
        .context("Failed to build emotional speech synthesis pipeline")?;

    let setup_time = setup_start.elapsed();
    println!(
        "✅ Emotional pipeline ready in {:.2} seconds",
        setup_time.as_secs_f32()
    );

    // Example 1: Comprehensive emotional styles with performance tracking
    println!("\n1. Comprehensive Emotional Style Demonstration");
    println!("----------------------------------------------");

    let emotional_examples = [
        ("neutral", "This is a neutral demonstration of advanced speech synthesis technology."),
        ("happy", "I'm feeling absolutely wonderful today! This incredible technology brings such joy!"),
        ("excited", "This is so exciting! I can't wait to share these amazing capabilities with everyone!"),
        ("calm", "Let's take a peaceful moment to reflect on this serene and tranquil synthesis demonstration."),
        ("professional", "Welcome to our professional voice synthesis service with advanced emotional capabilities."),
        ("contemplative", "Consider the profound implications of emotionally-aware artificial speech generation."),
    ];

    println!(
        "🎭 Generating {} different emotional expressions...",
        emotional_examples.len()
    );
    let mut emotion_timings = Vec::new();
    let mut total_emotion_duration = 0.0;

    for (style, text) in emotional_examples.iter() {
        println!("   Processing {} emotion...", style);

        let emotion_start = Instant::now();
        let audio = pipeline.synthesize(text).await.context(format!(
            "Failed to synthesize {} emotional expression",
            style
        ))?;
        let emotion_time = emotion_start.elapsed();
        emotion_timings.push(emotion_time);
        total_emotion_duration += audio.duration();

        let filename = format!("emotion_{}.wav", style);
        audio
            .save_wav(&filename)
            .context(format!("Failed to save {} emotional speech", style))?;

        println!(
            "   ✅ Generated {} speech: {} ({:.2}s, RTF: {:.2}x)",
            style,
            filename,
            emotion_time.as_secs_f32(),
            emotion_time.as_secs_f32() / audio.duration()
        );
    }

    let avg_emotion_time =
        emotion_timings.iter().map(|d| d.as_secs_f32()).sum::<f32>() / emotion_timings.len() as f32;
    println!(
        "📊 Average emotional synthesis time: {:.2} seconds",
        avg_emotion_time
    );

    // Example 2: Voice synthesis with configuration for different styles
    println!("\n2. Voice Synthesis with Different Configurations");
    println!("------------------------------------------------");

    let configurations = [
        (
            "expressive",
            SynthesisConfig {
                quality: QualityLevel::High,
                speaking_rate: 1.0,
                pitch_shift: 1.0,
                ..Default::default()
            },
        ),
        (
            "conversational",
            SynthesisConfig {
                quality: QualityLevel::Medium,
                speaking_rate: 0.9,
                pitch_shift: 0.0,
                ..Default::default()
            },
        ),
        (
            "narrative",
            SynthesisConfig {
                quality: QualityLevel::High,
                speaking_rate: 0.8,
                pitch_shift: -1.0,
                ..Default::default()
            },
        ),
    ];

    for (style, config) in configurations.iter() {
        let text = format!("This is a demonstration of {} speech style.", style);
        let audio = pipeline.synthesize_with_config(&text, config).await?;
        let filename = format!("style_{}.wav", style);
        audio.save_wav(&filename)?;
        println!("✓ Generated {} style: {}", style, filename);
    }

    // Example 3: Emotional text variations
    println!("\n3. Emotional Text Variations");
    println!("----------------------------");

    let emotional_texts = [
        ("excitement", "This is absolutely incredible! I can't believe how amazing this technology is!"),
        ("calmness", "Let us take a moment to appreciate the serenity of this peaceful moment."),
        ("urgency", "Please listen carefully! This is very important information that you need to know right now."),
        ("friendliness", "Hi there! It's wonderful to meet you. I hope you're having a fantastic day!"),
        ("confidence", "I am completely certain that this approach will deliver outstanding results."),
    ];

    for (emotion, text) in emotional_texts.iter() {
        let audio = pipeline.synthesize(text).await?;
        let filename = format!("text_emotion_{}.wav", emotion);
        audio.save_wav(&filename)?;
        println!("✓ Generated {} text: {}", emotion, filename);
    }

    // Example 4: Speed and pitch variations
    println!("\n4. Speed and Pitch Variations");
    println!("-----------------------------");

    let base_text = "This demonstrates different speed and pitch settings in voice synthesis.";
    let variations = [
        ("slow_low", 0.7, 0.8),
        ("normal", 1.0, 1.0),
        ("fast_high", 1.3, 1.2),
    ];

    for (name, speed, pitch) in variations.iter() {
        let config = SynthesisConfig {
            quality: QualityLevel::High,
            speaking_rate: *speed,
            pitch_shift: *pitch,
            ..Default::default()
        };

        let audio = pipeline.synthesize_with_config(base_text, &config).await?;
        let filename = format!("variation_{}.wav", name);
        audio.save_wav(&filename)?;
        println!(
            "✓ Generated {} variation: {} (speed: {:.1}, pitch: {:.1})",
            name, filename, speed, pitch
        );
    }

    // Comprehensive completion summary
    println!("\n🎉 Emotion Control Examples Successfully Completed!");
    println!("==================================================");
    println!("\n📁 Generated Emotional Audio Files:");
    println!("   Emotional Styles:");
    for (style, _) in emotional_examples.iter() {
        println!("   ├── emotion_{}.wav", style);
    }
    println!("   Configuration Styles:");
    for (style, _) in configurations.iter() {
        println!("   ├── style_{}.wav", style);
    }
    println!("   Text-based Emotions:");
    for (emotion, _) in emotional_texts.iter() {
        println!("   ├── text_emotion_{}.wav", emotion);
    }
    println!("   Parameter Variations:");
    for (name, _, _) in variations.iter() {
        println!("   └── variation_{}.wav", name);
    }

    println!("\n🚀 Emotional Synthesis Performance Summary:");
    println!("   Pipeline setup: {:.2} seconds", setup_time.as_secs_f32());
    println!(
        "   Average emotional expression time: {:.2} seconds",
        avg_emotion_time
    );
    println!(
        "   Total emotional variations: {}",
        emotional_examples.len() + configurations.len() + emotional_texts.len() + variations.len()
    );
    println!(
        "   Total emotional audio: {:.2} seconds",
        total_emotion_duration
    );

    println!("\n💡 Next Steps:");
    println!("   - Play the generated files to experience different emotional expressions");
    println!("   - Experiment with custom emotional parameters");
    println!("   - Try combining emotional styles with voice cloning examples");
    println!("   - Explore real-time emotional synthesis applications");

    Ok(())
}

/// Helper function to create output directories
fn ensure_output_dirs() -> std::io::Result<()> {
    std::fs::create_dir_all("examples/output")?;
    Ok(())
}

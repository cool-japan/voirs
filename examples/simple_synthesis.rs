//! Simple Synthesis Example - VoiRS Text-to-Speech
//!
//! This example demonstrates the most basic VoiRS synthesis workflow with minimal code.
//! It's perfect for users who want a quick start without extensive explanations.
//!
//! ## What this example does:
//! 1. Sets up a basic TTS pipeline with default settings
//! 2. Synthesizes a simple text phrase
//! 3. Saves the result as a WAV file
//! 4. Reports basic audio information
//!
//! ## Prerequisites:
//! - Rust 1.70+ installed
//! - VoiRS dependencies configured in Cargo.toml
//! - Audio output device available
//! - 2GB+ free RAM
//!
//! ## Quick Start Guide:
//!
//! ### First-time users (safest approach):
//! ```bash
//! # Test basic functionality first
//! cargo run --example simple_synthesis --no-default-features
//! ```
//!
//! ### Standard usage:
//! ```bash
//! cargo run --example simple_synthesis
//! ```
//!
//! ### Debug mode (if issues occur):
//! ```bash
//! RUST_LOG=debug cargo run --example simple_synthesis
//! ```
//!
//! ### Platform-specific quick fixes:
//! ```bash
//! # macOS (if permission issues)
//! sudo cargo run --example simple_synthesis
//!
//! # Linux (if audio group issues)
//! sudo usermod -a -G audio $USER
//! # Then log out and back in, then run normally
//!
//! # Windows (if antivirus blocking)
//! # Add VoiRS project folder to antivirus exclusions
//! cargo run --example simple_synthesis
//! ```
//!
//! ## Expected Output:
//! - **Console**: Progress messages with timing information
//! - **File**: `output.wav` created in current directory
//! - **Duration**: ~4-5 seconds of audio
//! - **Size**: ~500KB-1MB depending on quality
//! - **Processing Time**: 0.5-2 seconds (varies by system)
//!
//! ## Audio Information Explained:
//! - **Sample Rate**: 22050 Hz (standard quality, good balance)
//! - **Duration**: Length of generated speech in seconds
//! - **Channels**: 1 (mono) - sufficient for speech
//! - **Real-time Factor**: <1.0 = faster than real-time (good!)
//!
//! ## Testing Your Setup:
//!
//! ### 1. Verify audio output:
//! ```bash
//! # Play the generated file to test audio system
//! # macOS:
//! afplay output.wav
//!
//! # Linux:
//! aplay output.wav
//!
//! # Windows:
//! # Use Windows Media Player or similar
//! ```
//!
//! ### 2. Check synthesis quality:
//! - Should sound natural and clear
//! - No robotic artifacts or distortion
//! - Good pronunciation of all words
//!
//! ### 3. Performance benchmarks:
//! | System | Expected RTF | Quality |
//! |--------|-------------|---------|
//! | Modern laptop | 0.1-0.3x | Good |
//! | Older hardware | 0.5-1.0x | Adequate |
//! | Memory limited | 1.0-2.0x | Basic |
//!
//! ## Customization Tips:
//!
//! ### Change the text:
//! ```rust
//! let text = "Your custom text here!";
//! ```
//!
//! ### Adjust output filename:
//! ```rust
//! let output_file = "my_synthesis.wav";
//! ```
//!
//! ### Modify logging level:
//! ```rust
//! .with_max_level(tracing::Level::DEBUG)  // More detailed
//! .with_max_level(tracing::Level::WARN)   // Less verbose
//! ```
//!
//! ## Common Issues & Quick Fixes:
//!
//! ### "Pipeline build failed":
//! - **Cause**: Missing models or insufficient memory
//! - **Fix**: Try `cargo clean && cargo build` first
//! - **Alternative**: Use `--no-default-features` flag
//!
//! ### "No audio device":
//! - **macOS**: Check System Preferences → Sound → Output
//! - **Linux**: Run `aplay -l` to list audio devices
//! - **Windows**: Check Device Manager → Sound devices
//!
//! ### Slow synthesis (RTF > 1.0):
//! - **Immediate**: Close other applications to free memory
//! - **Long-term**: Consider system upgrade or lower quality settings
//!
//! ### Poor audio quality:
//! - Check input text for proper punctuation
//! - Ensure sufficient system resources
//! - Try different backends if available

use anyhow::{Context, Result};
use std::time::Instant;
use tracing::{info, warn};
use voirs::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging with appropriate level
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("🎤 VoiRS Simple Synthesis Example");
    println!("=================================");

    // Create components using bridge pattern
    info!("Creating TTS components...");
    let g2p = create_g2p(G2pBackend::RuleBased);
    let acoustic = create_acoustic(AcousticBackend::Vits);
    let vocoder = create_vocoder(VocoderBackend::HifiGan);

    // Build pipeline with error context
    info!("Building synthesis pipeline...");
    let pipeline = VoirsPipelineBuilder::new()
        .with_g2p(g2p)
        .with_acoustic_model(acoustic)
        .with_vocoder(vocoder)
        .build()
        .await
        .context("Failed to build synthesis pipeline")?;

    println!("✅ Pipeline ready!");

    let text = "Hello, world! This is VoiRS speaking in pure Rust.";
    println!("📝 Text to synthesize: \"{}\"", text);

    // Synthesize with timing and error handling
    let start_time = Instant::now();
    info!("Starting synthesis...");

    let audio = pipeline
        .synthesize(text)
        .await
        .context("Failed to synthesize text")?;

    let synthesis_time = start_time.elapsed();
    println!(
        "✅ Synthesis completed in {:.2} seconds",
        synthesis_time.as_secs_f32()
    );

    // Save audio with error handling
    let output_file = "output.wav";
    println!("💾 Saving audio to: {}", output_file);

    audio
        .save_wav(output_file)
        .context("Failed to save audio file")?;

    println!("✅ Audio saved successfully!");

    // Display comprehensive audio information
    println!("\n📊 Audio Information:");
    println!("   File: {}", output_file);
    println!("   Sample Rate: {} Hz", audio.sample_rate());
    println!("   Duration: {:.2} seconds", audio.duration());
    println!("   Channels: {}", audio.channels());
    println!(
        "   Real-time Factor: {:.2}x",
        synthesis_time.as_secs_f32() / audio.duration()
    );

    // Provide helpful next steps
    println!("\n🎉 Simple synthesis complete!");
    println!("💡 Try these next:");
    println!("   - Play '{}' to hear the result", output_file);
    println!("   - Modify the text above for different content");
    println!("   - Explore other examples for advanced features");

    Ok(())
}

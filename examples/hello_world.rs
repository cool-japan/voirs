//! Hello World Example - VoiRS Text-to-Speech
//!
//! This is the simplest possible VoiRS example that demonstrates basic text-to-speech synthesis.
//! Perfect for beginners who want to get started with VoiRS quickly.
//!
//! ## What this example does:
//! 1. Creates a basic TTS pipeline with default settings
//! 2. Synthesizes "Hello, World!" text
//! 3. Saves the result as a WAV file
//!
//! ## System Requirements:
//! - **OS**: macOS 10.15+, Ubuntu 18.04+, Windows 10+
//! - **Rust**: 1.70+ (check with `rustc --version`)
//! - **RAM**: 4GB available (8GB recommended)
//! - **Audio**: Any audio output device
//! - **Storage**: 2GB free space for models
//!
//! ## Platform-Specific Setup:
//!
//! ### macOS:
//! ```bash
//! # Install system dependencies
//! xcode-select --install
//! brew install cmake portaudio
//! 
//! # Grant audio permissions in System Preferences → Security & Privacy → Microphone
//! # Add Terminal or your IDE to the allowed applications
//! ```
//!
//! ### Linux (Ubuntu/Debian):
//! ```bash
//! # Install required packages
//! sudo apt-get update
//! sudo apt-get install build-essential cmake libasound2-dev portaudio19-dev
//! 
//! # Add user to audio group (then log out and back in)
//! sudo usermod -a -G audio $USER
//! ```
//!
//! ### Windows:
//! ```powershell
//! # Install Visual Studio Build Tools (required for compilation)
//! # Download from: https://visualstudio.microsoft.com/downloads/
//! # Include CMake component in installation
//! 
//! # Install Visual C++ Redistributable (latest version)
//! # Download from Microsoft website
//! ```
//!
//! ## Running this example:
//!
//! ### Standard build (CPU-only, recommended for first try):
//! ```bash
//! cargo run --example hello_world --no-default-features
//! ```
//!
//! ### Full build (with all features, may require CUDA on some systems):
//! ```bash
//! cargo run --example hello_world
//! ```
//!
//! ### Troubleshooting build issues:
//! ```bash
//! # If CUDA errors on macOS/Linux without GPU:
//! cargo run --example hello_world --workspace --exclude voirs-cli
//! 
//! # If audio device issues:
//! cargo run --example hello_world --features="cpu-only"
//! 
//! # For memory-constrained systems:
//! cargo run --example hello_world -j 1  # Use single thread
//! ```
//!
//! ## Expected output:
//! - Console log showing synthesis progress
//! - `hello_world.wav` file created in current directory
//! - Audio duration: ~4-6 seconds
//! - Processing time: 0.5-3 seconds (varies by system)
//!
//! ## Common Issues & Solutions:
//!
//! ### "No audio device found" error:
//! - **macOS**: Check audio permissions in System Preferences
//! - **Linux**: Ensure user is in audio group and ALSA/PulseAudio is running
//! - **Windows**: Check audio drivers and Windows Defender exceptions
//!
//! ### "Failed to build" errors:
//! - Update Rust: `rustup update stable`
//! - Clear cache: `cargo clean && cargo build`
//! - Check system dependencies listed above
//!
//! ### Poor audio quality:
//! - Ensure sufficient RAM (8GB+ recommended)
//! - Try CPU-only build for stability
//! - Check system audio settings
//!
//! ## Performance Expectations:
//! | System Type | Processing Time | Quality | Memory Usage |
//! |-------------|----------------|---------|--------------|
//! | Modern CPU | 0.5-1.0s | High | 2-4GB |
//! | Older CPU | 1.0-3.0s | Good | 1-2GB |
//! | Memory constrained | 2.0-5.0s | Medium | 1GB |

use anyhow::Result;
use voirs::{
    create_acoustic, create_g2p, create_vocoder, AcousticBackend, G2pBackend, VocoderBackend,
    VoirsPipelineBuilder,
};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize basic logging to see what's happening
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("🎤 VoiRS Hello World Example");
    println!("============================");
    println!();

    // The text we want to synthesize - keep it simple!
    let text = "Hello, World! Welcome to VoiRS text-to-speech synthesis.";
    println!("📝 Text to synthesize: \"{}\"", text);

    // Step 1: Create the basic components
    // These are the three core components needed for TTS:
    // - G2P: Converts text to phonemes (how words sound)
    // - Acoustic: Converts phonemes to mel-spectrograms (sound features)
    // - Vocoder: Converts mel-spectrograms to audio waveforms
    println!("🔧 Setting up TTS components...");

    let g2p = create_g2p(G2pBackend::RuleBased);
    let acoustic = create_acoustic(AcousticBackend::Vits);
    let vocoder = create_vocoder(VocoderBackend::HifiGan);

    println!("   ✅ G2P: Ready (converts text → phonemes)");
    println!("   ✅ Acoustic Model: Ready (phonemes → features)");
    println!("   ✅ Vocoder: Ready (features → audio)");

    // Step 2: Build the synthesis pipeline
    // This connects all components together
    println!("🏗️  Building synthesis pipeline...");

    let pipeline = VoirsPipelineBuilder::new()
        .with_g2p(g2p)
        .with_acoustic_model(acoustic)
        .with_vocoder(vocoder)
        .build()
        .await?;

    println!("   ✅ Pipeline ready for synthesis!");

    // Step 3: Synthesize the text
    println!("🎵 Synthesizing speech...");

    let start_time = std::time::Instant::now();
    let audio = pipeline.synthesize(text).await?;
    let synthesis_time = start_time.elapsed();

    println!(
        "   ✅ Synthesis complete in {:.2} seconds!",
        synthesis_time.as_secs_f32()
    );

    // Step 4: Display audio information
    println!("📊 Audio Information:");
    println!("   Duration: {:.2} seconds", audio.duration());
    println!("   Sample Rate: {} Hz", audio.sample_rate());
    println!("   Channels: {}", audio.channels());
    println!(
        "   Real-time Factor: {:.2}x",
        synthesis_time.as_secs_f32() / audio.duration()
    );

    // Step 5: Save the audio file
    let output_file = "hello_world.wav";
    println!("💾 Saving audio to: {}", output_file);

    audio.save_wav(output_file)?;

    println!("   ✅ Audio saved successfully!");

    // Success message
    println!();
    println!("🎉 Hello World synthesis complete!");
    println!(
        "   You can now play '{}' to hear your first VoiRS synthesis.",
        output_file
    );
    println!();
    println!("💡 Next steps:");
    println!("   - Try changing the text above");
    println!("   - Explore other examples for advanced features");
    println!("   - Check out the VoiRS documentation");

    Ok(())
}

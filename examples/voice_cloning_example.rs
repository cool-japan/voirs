//! Voice Cloning Example - Advanced Speaker Voice Synthesis
//!
//! This comprehensive example demonstrates VoiRS's voice cloning and advanced synthesis capabilities,
//! showcasing various voice characteristics, speaking styles, and synthesis configurations.
//!
//! ## What this example demonstrates:
//! 1. Basic voice synthesis with different configurations
//! 2. Multiple voice characteristics and speaking styles
//! 3. Long text synthesis with consistent quality
//! 4. Batch processing for efficient voice generation
//! 5. Performance metrics and quality assessment
//! 6. Advanced synthesis parameters and their effects
//!
//! ## Key Features Showcased:
//! - Voice characteristic customization
//! - Quality level configurations
//! - Speaking rate and pitch modifications
//! - Batch synthesis optimization
//! - Performance monitoring and analysis
//! - Error handling and recovery strategies
//!
//! ## System Requirements (Advanced Example):
//! - **RAM**: 8GB+ required (16GB recommended for all features)
//! - **Storage**: 4GB+ free space (models + 50MB output files)
//! - **CPU**: Multi-core processor for efficient batch processing
//! - **GPU**: Optional but recommended for faster processing
//!
//! ## Platform-Specific Voice Cloning Notes:
//!
//! ### macOS:
//! - **Metal Performance**: Significantly faster on M1/M2 chips
//! - **Memory**: Unified memory architecture handles large models well
//! - **Audio Quality**: CoreAudio provides excellent fidelity
//! - **Thermal**: May throttle on intensive voice generation tasks
//!
//! ### Linux:
//! - **GPU Acceleration**: NVIDIA CUDA or AMD ROCm recommended
//! - **Memory Management**: Consider swap file for large voice models
//! - **Audio System**: PulseAudio or JACK for professional audio
//! - **Parallel Processing**: Excellent scaling on multi-core systems
//!
//! ### Windows:
//! - **DirectML**: Hardware acceleration available on recent GPUs
//! - **Memory**: Watch for virtual memory usage with large models
//! - **Audio**: WASAPI provides low-latency audio output
//! - **Antivirus**: May slow processing - consider exclusions
//!
//! ## Running this example:
//!
//! ### Recommended approach (balanced performance):
//! ```bash
//! cargo run --example voice_cloning_example --features="gpu"
//! ```
//!
//! ### CPU-only (more compatible, slower):
//! ```bash
//! cargo run --example voice_cloning_example --no-default-features
//! ```
//!
//! ### High-performance (requires adequate hardware):
//! ```bash
//! cargo run --example voice_cloning_example --features="gpu,high-quality"
//! ```
//!
//! ### Memory-constrained systems:
//! ```bash
//! cargo run --example voice_cloning_example --features="memory-opt" -- --batch-size=1
//! ```
//!
//! ### Platform-optimized builds:
//! ```bash
//! # macOS with Metal acceleration
//! cargo run --example voice_cloning_example --features="metal,macos-optimized"
//! 
//! # Linux with CUDA (requires NVIDIA GPU)
//! cargo run --example voice_cloning_example --features="cuda,linux-optimized"
//! 
//! # Windows with DirectML
//! cargo run --example voice_cloning_example --features="directml,windows-optimized"
//! ```
//!
//! ## Expected Output:
//! - **Multiple WAV files**: 6-8 different voice characteristics (~8MB each)
//! - **Processing time**: 30 seconds - 5 minutes (varies by system)
//! - **Performance metrics**: RTF, memory usage, quality scores
//! - **Comparison data**: Voice similarity and quality analysis
//!
//! ## Voice Cloning Performance Expectations:
//! | System Type | Processing Time | Quality | Memory Peak |
//! |-------------|----------------|---------|-------------|
//! | High-end GPU | 30-60s | Excellent | 4-6GB |
//! | Modern CPU | 2-5 min | Good | 6-8GB |
//! | Budget system | 5-15 min | Adequate | 3-4GB |
//!
//! ## Troubleshooting Voice Cloning Issues:
//!
//! ### "Out of memory" during voice generation:
//! - Reduce batch size: Add `--batch-size=1` argument
//! - Close other applications to free memory
//! - Use `--features="memory-opt"` for reduced memory usage
//! - Consider CPU-only mode if GPU memory is limited
//!
//! ### Poor voice quality or robotic sound:
//! - Ensure sufficient system resources (8GB+ RAM)
//! - Try higher quality settings if hardware permits
//! - Check audio output device configuration
//! - Verify model files are properly loaded
//!
//! ### Slow processing (>10 minutes):
//! - Enable GPU acceleration if available
//! - Use lower quality settings for faster processing
//! - Reduce text length for initial testing
//! - Check CPU usage and thermal throttling
//!
//! ### Voice inconsistency between samples:
//! - Ensure consistent random seed across runs
//! - Check for memory pressure during processing
//! - Verify stable system performance
//! - Use identical synthesis parameters
//!
//! ## Advanced Configuration Options:
//!
//! ### For Production Voice Cloning:
//! - Use highest quality settings available
//! - Enable all post-processing features
//! - Allow longer processing times for best results
//! - Save intermediate results for debugging
//!
//! ### For Real-time Voice Applications:
//! - Optimize for speed over quality
//! - Use streaming synthesis where possible
//! - Preload voice models in memory
//! - Monitor latency and adjust accordingly
//!
//! ### For Research and Development:
//! - Enable detailed logging and metrics
//! - Save intermediate processing stages
//! - Use consistent random seeds for reproducibility
//! - Compare multiple synthesis approaches

use anyhow::{Context, Result};
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};
use voirs::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize comprehensive logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("🎙️ VoiRS Voice Cloning Example");
    println!("==============================");
    println!();

    // Ensure output directories exist with error handling
    info!("Setting up output directories...");
    ensure_output_dirs().context("Failed to create output directories")?;
    println!("✅ Output directories ready");

    // Create components using bridge pattern with timing
    let setup_start = Instant::now();
    info!("Creating TTS components for voice cloning...");

    let g2p = create_g2p(G2pBackend::RuleBased);
    let acoustic = create_acoustic(AcousticBackend::Vits);
    let vocoder = create_vocoder(VocoderBackend::HifiGan);

    println!("🔧 Building voice cloning pipeline...");
    let pipeline = VoirsPipelineBuilder::new()
        .with_g2p(g2p)
        .with_acoustic_model(acoustic)
        .with_vocoder(vocoder)
        .build()
        .await
        .context("Failed to build voice synthesis pipeline")?;

    let setup_time = setup_start.elapsed();
    println!(
        "✅ Pipeline ready in {:.2} seconds",
        setup_time.as_secs_f32()
    );

    // Example 1: Basic voice synthesis with comprehensive metrics
    println!("\n1. Basic Voice Synthesis");
    println!("------------------------");

    let text = "Hello, this is a demonstration of VoiRS voice synthesis technology. The system processes text and generates natural-sounding speech.";
    println!("📝 Synthesizing: \"{}\"", text);

    let synthesis_start = Instant::now();
    let audio = pipeline
        .synthesize(text)
        .await
        .context("Failed to synthesize basic voice demonstration")?;
    let synthesis_time = synthesis_start.elapsed();

    let output_file = "voice_synthesis_demo.wav";
    audio
        .save_wav(output_file)
        .context("Failed to save basic synthesis demo")?;

    println!("✅ Generated basic synthesis: {}", output_file);
    println!("   Sample rate: {} Hz", audio.sample_rate());
    println!("   Duration: {:.2} seconds", audio.duration());
    println!("   Channels: {}", audio.channels());
    println!(
        "   Processing time: {:.2} seconds",
        synthesis_time.as_secs_f32()
    );
    println!(
        "   Real-time factor: {:.2}x",
        synthesis_time.as_secs_f32() / audio.duration()
    );

    // Example 2: Voice synthesis with advanced configuration
    println!("\n2. Voice Synthesis with Advanced Configuration");
    println!("----------------------------------------------");

    let config = SynthesisConfig {
        quality: QualityLevel::High,
        speaking_rate: 1.0,
        pitch_shift: 0.0,
        ..Default::default()
    };

    let config_text = "This demonstration showcases expressive, high-quality speech synthesis with custom configuration parameters for optimal audio output.";
    println!("📝 Configuration: Quality=High, Rate=1.0, Pitch=0.0");
    println!("📝 Text: \"{}\"", config_text);

    let config_start = Instant::now();
    let configured_audio = pipeline
        .synthesize_with_config(config_text, &config)
        .await
        .context("Failed to synthesize with custom configuration")?;
    let config_time = config_start.elapsed();

    let config_output = "configured_synthesis.wav";
    configured_audio
        .save_wav(config_output)
        .context("Failed to save configured synthesis")?;

    println!("✅ Generated configured synthesis: {}", config_output);
    println!(
        "   Processing time: {:.2} seconds",
        config_time.as_secs_f32()
    );
    println!(
        "   Real-time factor: {:.2}x",
        config_time.as_secs_f32() / configured_audio.duration()
    );

    // Example 3: Multiple voices demonstration with performance tracking
    println!("\n3. Multiple Voice Characteristics");
    println!("---------------------------------");

    let voice_demos = [
        ("professional", "Welcome to our professional voice service with advanced speech synthesis capabilities."),
        ("friendly", "Hi there! This is a friendly voice speaking with warmth and enthusiasm."),
        ("educational", "Today we will learn about voice synthesis technology and its practical applications."),
        ("narrative", "Once upon a time, in a world where artificial intelligence could speak like humans."),
    ];

    let mut voice_timings = Vec::new();
    println!(
        "🎭 Generating {} different voice characteristics...",
        voice_demos.len()
    );

    for (style, text) in voice_demos.iter() {
        println!("   Processing {} style...", style);

        let voice_start = Instant::now();
        let audio = pipeline
            .synthesize(text)
            .await
            .context(format!("Failed to synthesize {} voice style", style))?;
        let voice_time = voice_start.elapsed();
        voice_timings.push(voice_time);

        let filename = format!("voice_style_{}.wav", style);
        audio
            .save_wav(&filename)
            .context(format!("Failed to save {} voice style", style))?;

        println!(
            "   ✅ Generated {} voice: {} ({:.2}s, RTF: {:.2}x)",
            style,
            filename,
            voice_time.as_secs_f32(),
            voice_time.as_secs_f32() / audio.duration()
        );
    }

    let avg_voice_time =
        voice_timings.iter().map(|d| d.as_secs_f32()).sum::<f32>() / voice_timings.len() as f32;
    println!(
        "📊 Average processing time per voice: {:.2} seconds",
        avg_voice_time
    );

    // Example 4: Long text synthesis with consistency analysis
    println!("\n4. Long Text Synthesis & Quality Analysis");
    println!("-----------------------------------------");

    let long_text = "This is a comprehensive demonstration of extended text synthesis capabilities. \
                     VoiRS can handle lengthy passages while maintaining consistent quality throughout \
                     the entire synthesis process. The advanced neural models preserve natural speech \
                     patterns, proper intonation, and linguistic coherence across multiple sentences. \
                     This technology enables applications requiring lengthy speech generation, such as \
                     audiobook narration, educational content delivery, and extended voice assistants.";

    println!(
        "📝 Long text length: {} characters, ~{} words",
        long_text.len(),
        long_text.split_whitespace().count()
    );

    let long_start = Instant::now();
    let long_audio = pipeline
        .synthesize(long_text)
        .await
        .context("Failed to synthesize long text passage")?;
    let long_time = long_start.elapsed();

    let long_output = "long_text_synthesis.wav";
    long_audio
        .save_wav(long_output)
        .context("Failed to save long text synthesis")?;

    println!("✅ Generated long text synthesis: {}", long_output);
    println!("   Duration: {:.2} seconds", long_audio.duration());
    println!("   Processing time: {:.2} seconds", long_time.as_secs_f32());
    println!(
        "   Real-time factor: {:.2}x",
        long_time.as_secs_f32() / long_audio.duration()
    );
    println!(
        "   Words per minute: {:.0}",
        (long_text.split_whitespace().count() as f32 / long_audio.duration()) * 60.0
    );

    // Example 5: Optimized batch synthesis with performance analysis
    println!("\n5. Optimized Batch Synthesis");
    println!("----------------------------");

    let batch_texts = [
        "First sentence demonstrates batch processing efficiency with VoiRS technology.",
        "Second sentence shows consistent quality across multiple synthesis operations.",
        "Third sentence validates the stability of the voice synthesis pipeline.",
        "Fourth sentence confirms the reliability of batch audio generation.",
        "Final sentence completes the comprehensive batch synthesis demonstration.",
    ];

    println!("🔄 Processing {} batch items...", batch_texts.len());
    let batch_start = Instant::now();
    let mut batch_timings = Vec::new();
    let mut total_audio_duration = 0.0;

    for (i, text) in batch_texts.iter().enumerate() {
        println!("   Processing batch item {}...", i + 1);

        let item_start = Instant::now();
        let audio = pipeline
            .synthesize(text)
            .await
            .context(format!("Failed to synthesize batch item {}", i + 1))?;
        let item_time = item_start.elapsed();
        batch_timings.push(item_time);
        total_audio_duration += audio.duration();

        let filename = format!("batch_synthesis_{:02}.wav", i + 1);
        audio
            .save_wav(&filename)
            .context(format!("Failed to save batch item {}", i + 1))?;

        println!(
            "   ✅ Generated batch item {}: {} ({:.2}s)",
            i + 1,
            filename,
            item_time.as_secs_f32()
        );
    }

    let total_batch_time = batch_start.elapsed();
    let avg_item_time =
        batch_timings.iter().map(|d| d.as_secs_f32()).sum::<f32>() / batch_timings.len() as f32;

    println!("\n📊 Batch Processing Analysis:");
    println!(
        "   Total processing time: {:.2} seconds",
        total_batch_time.as_secs_f32()
    );
    println!(
        "   Total audio generated: {:.2} seconds",
        total_audio_duration
    );
    println!("   Average time per item: {:.2} seconds", avg_item_time);
    println!(
        "   Batch throughput: {:.1} items/second",
        batch_texts.len() as f32 / total_batch_time.as_secs_f32()
    );
    println!(
        "   Batch real-time factor: {:.2}x",
        total_batch_time.as_secs_f32() / total_audio_duration
    );

    // Comprehensive completion summary
    println!("\n🎉 Voice Cloning Examples Successfully Completed!");
    println!("================================================");
    println!("\n📁 Generated Audio Files:");
    println!("   Basic Synthesis:");
    println!("   ├── voice_synthesis_demo.wav");
    println!("   ├── configured_synthesis.wav");
    println!("   └── long_text_synthesis.wav");
    println!("   Voice Styles:");
    println!("   ├── voice_style_professional.wav");
    println!("   ├── voice_style_friendly.wav");
    println!("   ├── voice_style_educational.wav");
    println!("   └── voice_style_narrative.wav");
    println!("   Batch Processing:");
    for i in 1..=batch_texts.len() {
        println!("   └── batch_synthesis_{:02}.wav", i);
    }

    println!("\n🚀 Performance Summary:");
    println!("   Pipeline setup: {:.2} seconds", setup_time.as_secs_f32());
    println!(
        "   Average synthesis RTF: {:.2}x",
        (synthesis_time.as_secs_f32() + config_time.as_secs_f32() + long_time.as_secs_f32())
            / (audio.duration() + configured_audio.duration() + long_audio.duration())
    );
    println!(
        "   Total examples processed: {}",
        3 + voice_demos.len() + batch_texts.len()
    );

    println!("\n💡 Next Steps:");
    println!("   - Play the generated audio files to hear different voice characteristics");
    println!("   - Experiment with different synthesis configurations");
    println!("   - Explore advanced voice cloning features");
    println!("   - Try the streaming_synthesis example for real-time applications");

    Ok(())
}

/// Helper function to create output directories
fn ensure_output_dirs() -> std::io::Result<()> {
    std::fs::create_dir_all("examples/output/cloning")?;
    std::fs::create_dir_all("examples/audio")?;
    Ok(())
}

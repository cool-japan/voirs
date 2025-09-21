# Quick Start

This guide will get you up and running with VoiRS Text-to-Speech synthesis in just a few minutes.

## Prerequisites

Make sure you have completed the [Installation](./installation.md) guide first.

## Your First Synthesis

Let's start with a simple example to synthesize speech from text:

```rust
use voirs::prelude::*;
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    // Create the synthesis pipeline components
    let g2p = create_g2p(G2pBackend::RuleBased);
    let acoustic = create_acoustic(AcousticBackend::Vits);
    let vocoder = create_vocoder(VocoderBackend::HifiGan);

    // Build the synthesis pipeline
    let pipeline = VoirsPipelineBuilder::new()
        .with_g2p(g2p)
        .with_acoustic_model(acoustic)
        .with_vocoder(vocoder)
        .build()
        .await?;

    // Text to synthesize
    let text = "Hello, world! This is VoiRS speaking in pure Rust.";

    // Perform synthesis
    let audio = pipeline.synthesize(text).await?;

    // Save the result as a WAV file
    audio.save_wav("output.wav")?;

    println!("✅ Synthesis completed! Audio saved to output.wav");
    println!("📊 Audio info: {:.2}s duration, {} Hz sample rate",
             audio.duration(), audio.sample_rate());

    Ok(())
}
```

## Streaming Synthesis

For real-time synthesis with streaming capabilities:

```rust
use voirs::prelude::*;
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    // Create pipeline with streaming configuration
    let g2p = create_g2p(G2pBackend::RuleBased);
    let acoustic = create_acoustic(AcousticBackend::Vits);
    let vocoder = create_vocoder(VocoderBackend::HifiGan);

    let pipeline = VoirsPipelineBuilder::new()
        .with_g2p(g2p)
        .with_acoustic_model(acoustic)
        .with_vocoder(vocoder)
        .with_streaming_enabled(true)
        .build()
        .await?;

    // Create streaming synthesizer
    let mut stream = pipeline.create_stream().await?;

    // Text chunks to synthesize in real-time
    let text_chunks = vec![
        "Hello there! ",
        "This is streaming ",
        "text-to-speech synthesis ",
        "powered by VoiRS. ",
        "Each chunk is processed ",
        "as soon as it arrives."
    ];

    println!("🎤 Starting streaming synthesis...");

    for (i, chunk) in text_chunks.iter().enumerate() {
        println!("Processing chunk {}: '{}'", i + 1, chunk);

        // Send text chunk for synthesis
        stream.send_text(chunk).await?;

        // Get audio output as it becomes available
        while let Some(audio_chunk) = stream.receive_audio().await? {
            let filename = format!("stream_chunk_{:02}.wav", i + 1);
            audio_chunk.save_wav(&filename)?;
            println!("  💾 Saved audio chunk: {}", filename);
        }
    }

    // Finalize the stream
    stream.finalize().await?;
    println!("✅ Streaming synthesis completed!");

    Ok(())
}
```

## Batch Synthesis

Process multiple texts efficiently with batch synthesis:

```rust
use voirs::prelude::*;
use anyhow::Result;
use tokio::fs;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    // Create the synthesis pipeline
    let g2p = create_g2p(G2pBackend::RuleBased);
    let acoustic = create_acoustic(AcousticBackend::Vits);
    let vocoder = create_vocoder(VocoderBackend::HifiGan);

    let pipeline = VoirsPipelineBuilder::new()
        .with_g2p(g2p)
        .with_acoustic_model(acoustic)
        .with_vocoder(vocoder)
        .build()
        .await?;

    // Multiple texts to synthesize
    let texts = [
        "Welcome to VoiRS, the pure Rust speech synthesis framework.",
        "This example demonstrates batch processing capabilities.",
        "Each text is synthesized independently and saved to a separate file.",
        "VoiRS delivers high-quality, natural-sounding speech output."
    ];

    // Create output directory
    fs::create_dir_all("batch_output").await?;

    println!("🎤 Starting batch synthesis...");

    for (i, text) in texts.iter().enumerate() {
        println!("Processing text {}: {}", i + 1, text);

        // Synthesize each text
        let audio = pipeline.synthesize(text).await?;
        let output_path = format!("batch_output/output_{:02}.wav", i + 1);

        // Save the audio file
        audio.save_wav(&output_path)?;
        println!("  💾 Saved: {} ({:.2}s duration)", output_path, audio.duration());
    }

    println!("✅ Batch synthesis completed! Check the batch_output/ directory.");

    Ok()
}
```

## SSML Support

Use Speech Synthesis Markup Language for advanced control:

```rust
use voirs::prelude::*;
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    // Create pipeline with SSML support
    let g2p = create_g2p(G2pBackend::RuleBased);
    let acoustic = create_acoustic(AcousticBackend::Vits);
    let vocoder = create_vocoder(VocoderBackend::HifiGan);

    let pipeline = VoirsPipelineBuilder::new()
        .with_g2p(g2p)
        .with_acoustic_model(acoustic)
        .with_vocoder(vocoder)
        .with_ssml_enabled(true)
        .build()
        .await?;

    // SSML text with prosody control
    let ssml_text = r#"
        <speak>
            <p>
                <emphasis level="strong">Welcome</emphasis> to VoiRS!
            </p>
            <break time="1s"/>
            <p>
                This text uses <prosody rate="slow">slow speech</prosody>
                and then <prosody rate="fast">fast speech</prosody>.
            </p>
            <break time="500ms"/>
            <p>
                You can also control <prosody pitch="high">pitch</prosody>
                and <prosody volume="loud">volume</prosody>.
            </p>
        </speak>
    "#;

    println!("🎤 Synthesizing SSML content...");

    // Synthesize with SSML interpretation
    let audio = pipeline.synthesize_ssml(ssml_text).await?;

    // Save the result
    audio.save_wav("ssml_output.wav")?;

    println!("✅ SSML synthesis completed!");
    println!("📊 Generated {:.2}s of audio with prosodic controls", audio.duration());

    Ok(())
}
```

## Multi-language Support

Synthesize speech in different languages:

```rust
use voirs::prelude::*;
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    // Define texts in different languages
    let languages = vec![
        (LanguageCode::EnUs, "Hello, this is English text-to-speech."),
        (LanguageCode::EsEs, "Hola, esto es síntesis de voz en español."),
        (LanguageCode::JaJp, "こんにちは、これは日本語の音声合成です。"),
        (LanguageCode::FrFr, "Bonjour, ceci est une synthèse vocale française."),
    ];

    println!("🌍 Starting multi-language synthesis...");

    for (lang, text) in languages {
        println!("Synthesizing {:?}: {}", lang, text);

        // Create language-specific pipeline
        let g2p = create_g2p_for_language(G2pBackend::RuleBased, lang);
        let acoustic = create_acoustic_for_language(AcousticBackend::Vits, lang);
        let vocoder = create_vocoder(VocoderBackend::HifiGan);

        let pipeline = VoirsPipelineBuilder::new()
            .with_g2p(g2p)
            .with_acoustic_model(acoustic)
            .with_vocoder(vocoder)
            .with_language(lang)
            .build()
            .await?;

        // Synthesize the text
        let audio = pipeline.synthesize(text).await?;
        let output_file = format!("output_{:?}.wav", lang).to_lowercase();

        audio.save_wav(&output_file)?;
        println!("  💾 Saved: {} ({:.2}s)", output_file, audio.duration());
    }

    println!("✅ Multi-language synthesis completed!");

    Ok(())
}
```

## Performance Optimization

Optimize synthesis performance for production use:

```rust
use voirs::prelude::*;
use anyhow::Result;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    // Performance monitoring setup
    let startup_start = Instant::now();

    // Create optimized pipeline with performance settings
    let g2p = create_g2p(G2pBackend::RuleBased);
    let acoustic = create_acoustic_with_config(
        AcousticBackend::Vits,
        AcousticConfig::performance_optimized()
    );
    let vocoder = create_vocoder_with_config(
        VocoderBackend::HifiGan,
        VocoderConfig::fast_inference()
    );

    let pipeline = VoirsPipelineBuilder::new()
        .with_g2p(g2p)
        .with_acoustic_model(acoustic)
        .with_vocoder(vocoder)
        .with_batch_size(8)  // Optimize for batch processing
        .with_cache_enabled(true)  // Enable model caching
        .build()
        .await?;

    let startup_time = startup_start.elapsed();
    println!("🚀 Pipeline startup: {:?}", startup_time);

    // Test synthesis performance
    let test_text = "This is a performance test for VoiRS speech synthesis.";

    let synthesis_start = Instant::now();
    let audio = pipeline.synthesize(test_text).await?;
    let synthesis_time = synthesis_start.elapsed();

    // Calculate Real-Time Factor (RTF)
    let rtf = synthesis_time.as_secs_f32() / audio.duration();

    println!("📊 Performance Metrics:");
    println!("  Synthesis time: {:?}", synthesis_time);
    println!("  Audio duration: {:.2}s", audio.duration());
    println!("  Real-time factor: {:.3}x", rtf);
    println!("  Status: {}", if rtf < 1.0 { "✅ Real-time capable" } else { "⚠️ Slower than real-time" });

    // Save performance test result
    audio.save_wav("performance_test.wav")?;

    // Batch performance test
    let batch_texts = vec![
        "First sentence for batch testing.",
        "Second sentence with different content.",
        "Third sentence to complete the batch.",
        "Fourth and final sentence in this performance test."
    ];

    println!("
🔄 Testing batch synthesis...");
    let batch_start = Instant::now();

    for (i, text) in batch_texts.iter().enumerate() {
        let audio = pipeline.synthesize(text).await?;
        let filename = format!("batch_perf_{:02}.wav", i + 1);
        audio.save_wav(&filename)?;
    }

    let batch_time = batch_start.elapsed();
    println!("  Batch synthesis: {:?} for {} texts", batch_time, batch_texts.len());
    println!("  Average per text: {:?}", batch_time / batch_texts.len() as u32);

    Ok(())
}
```

## Error Handling

Robust error handling for production applications:

```rust
use voirs::prelude::*;
use anyhow::{Context, Result};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    println!("🎤 VoiRS Synthesis with Error Handling");

    // Attempt to create synthesis pipeline with detailed error handling
    let pipeline = match create_synthesis_pipeline().await {
        Ok(p) => p,
        Err(e) => {
            eprintln!("❌ Failed to create synthesis pipeline: {}", e);
            return Err(e);
        }
    };

    let text = "This example demonstrates proper error handling in VoiRS.";

    // Attempt synthesis with comprehensive error handling
    match pipeline.synthesize(text).await {
        Ok(audio) => {
            match audio.save_wav("error_handling_output.wav") {
                Ok(_) => {
                    println!("✅ Success: Audio synthesized and saved!");
                    println!("📊 Duration: {:.2}s, Sample Rate: {}Hz",
                             audio.duration(), audio.sample_rate());
                }
                Err(e) => {
                    eprintln!("❌ Failed to save audio file: {}", e);
                    return Err(e.into());
                }
            }
        }
        Err(VoirsError::G2pError(e)) => {
            eprintln!("❌ G2P conversion failed: {}", e);
            eprintln!("💡 Tip: Check if the text contains unsupported characters");
            return Err(e.into());
        }
        Err(VoirsError::AcousticError(e)) => {
            eprintln!("❌ Acoustic model failed: {}", e);
            eprintln!("💡 Tip: Ensure sufficient memory and valid model files");
            return Err(e.into());
        }
        Err(VoirsError::VocoderError(e)) => {
            eprintln!("❌ Vocoder failed: {}", e);
            eprintln!("💡 Tip: Check vocoder model compatibility");
            return Err(e.into());
        }
        Err(e) => {
            eprintln!("❌ Synthesis error: {}", e);
            return Err(e.into());
        }
    }

    Ok(())
}

async fn create_synthesis_pipeline() -> Result<VoirsPipeline> {
    let g2p = create_g2p(G2pBackend::RuleBased)
        .context("Failed to create G2P component")?;

    let acoustic = create_acoustic(AcousticBackend::Vits)
        .context("Failed to create acoustic model")?;

    let vocoder = create_vocoder(VocoderBackend::HifiGan)
        .context("Failed to create vocoder")?;

    VoirsPipelineBuilder::new()
        .with_g2p(g2p)
        .with_acoustic_model(acoustic)
        .with_vocoder(vocoder)
        .build()
        .await
        .context("Failed to build synthesis pipeline")
}
```

## Configuration Options

### Voice Selection

Choose the right voice for your use case:

```rust
// For natural speech (high quality, slower)
let config = SynthesisConfig::default()
    .with_voice_quality(VoiceQuality::High)
    .with_model_size(ModelSize::Large);

// For balanced performance (good quality, moderate speed)
let config = SynthesisConfig::default()
    .with_voice_quality(VoiceQuality::Standard)
    .with_model_size(ModelSize::Base);

// For fast synthesis (lower quality, fastest)
let config = SynthesisConfig::default()
    .with_voice_quality(VoiceQuality::Fast)
    .with_model_size(ModelSize::Small);
```

### Audio Output

Configure audio output settings:

```rust
let config = SynthesisConfig::default()
    .with_sample_rate(22050)         // Standard quality
    .with_sample_rate(44100)         // High quality
    .with_channels(1)                // Mono output
    .with_bit_depth(16);             // 16-bit audio
```

### Synthesis Options

Control synthesis behavior:

```rust
let config = SynthesisConfig::default()
    .with_streaming_enabled(true)    // Enable streaming synthesis
    .with_ssml_enabled(true)         // Enable SSML markup
    .with_prosody_control(true)      // Enable prosody adjustments
    .with_emotion_control(true);     // Enable emotional expression
```

## Working with Examples

Explore the comprehensive examples included with VoiRS:

```bash
# Basic text-to-speech synthesis
cargo run --example simple_synthesis

# Batch processing
cargo run --example batch_synthesis

# Streaming synthesis
cargo run --example streaming_synthesis

# SSML support
cargo run --example ssml_synthesis

# Multi-language synthesis
cargo run --example multilingual_synthesis

# Performance testing
cargo run --example performance_benchmark

# Voice cloning (advanced)
cargo run --example voice_cloning --features="cloning"
```

## Next Steps

Now that you're familiar with the basics:

1. **Explore Advanced Features**: Check out [CLI Usage](./cli-usage.md) for command-line tools
2. **Optimize Performance**: Read the [Performance Tuning](./performance.md) guide
3. **Handle Edge Cases**: Review [Troubleshooting](./troubleshooting.md) for common issues
4. **Build Applications**: See [Rust API](./rust-api.md) for comprehensive API documentation
5. **Voice Customization**: Learn about [Voice Cloning](./voice-cloning.md) and adaptation
6. **Production Deployment**: Review [Deployment Guide](./deployment.md) for scalable setups

## Common Use Cases

### Voice Assistant Response

```rust
// Perfect for voice assistant responses
let config = SynthesisConfig::default()
    .with_streaming_enabled(true)
    .with_low_latency(true)
    .with_voice_quality(VoiceQuality::Standard);
```

### Audiobook Production

```rust
// Optimized for long-form content
let config = SynthesisConfig::default()
    .with_voice_quality(VoiceQuality::High)
    .with_ssml_enabled(true)
    .with_prosody_control(true);
```

### Real-time Narration

```rust
// For live narration applications
let config = SynthesisConfig::default()
    .with_streaming_enabled(true)
    .with_chunk_size(ChunkSize::Small)
    .with_buffer_optimization(true);
```

## Getting Help

- **Documentation**: Comprehensive guides in this documentation
- **Examples**: Working examples in the `examples/` directory
- **API Reference**: Complete API docs at [docs.rs/voirs](https://docs.rs/voirs)
- **Community**: [GitHub Discussions](https://github.com/cool-japan/voirs/discussions)
- **Issues**: [Bug Reports](https://github.com/cool-japan/voirs/issues)
- **CLI Help**: Run `voirs --help` for command-line usage
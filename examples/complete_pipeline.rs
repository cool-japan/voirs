//! Complete VoiRS TTS Pipeline Example
//! 
//! This example demonstrates the full text-to-speech pipeline using:
//! - Rule-based G2P for phoneme conversion
//! - VITS acoustic model with normalizing flows
//! - HiFi-GAN vocoder for high-quality audio synthesis

use std::sync::Arc;
use voirs::{
    VoirsPipelineBuilder, SynthesisConfig, QualityLevel, AudioFormat,
    Result, VoirsError,
};
use voirs_g2p::english::EnglishG2p;
use voirs_acoustic::vits::{VitsModel, VitsConfig};
use voirs_vocoder::hifigan::{HiFiGanVocoder, HiFiGanVariants};
use candle_core::Device;
use candle_nn::{VarMap, VarBuilder};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("🎤 VoiRS Complete TTS Pipeline Example");
    println!("=====================================");

    // Example text to synthesize
    let text = "Hello, this is a demonstration of the VoiRS text-to-speech system using VITS and HiFi-GAN.";
    
    // Create synthesis configuration
    let synthesis_config = SynthesisConfig {
        speaking_rate: 1.0,
        pitch_shift: 0.0,
        volume_gain: 0.0,
        enable_enhancement: true,
        output_format: AudioFormat::Wav,
        sample_rate: 22050,
        quality: QualityLevel::High,
    };

    println!("📝 Input text: \"{}\"", text);
    println!("⚙️  Configuration: {:?}", synthesis_config);
    println!();

    // Step 1: Initialize components
    println!("🔧 Initializing pipeline components...");
    
    // Initialize G2P
    let g2p = Arc::new(EnglishG2p::new());
    println!("✅ G2P: English rule-based phoneme converter");
    
    // Initialize VITS acoustic model
    let device = Device::Cpu;
    let vits_config = VitsConfig::default();
    let vits_model = VitsModel::with_config(vits_config)?;
    let acoustic = Arc::new(vits_model);
    println!("✅ Acoustic Model: VITS with normalizing flows");
    
    // Initialize HiFi-GAN vocoder
    let hifigan_config = HiFiGanVariants::v2(); // Balanced quality/speed
    let vocoder = Arc::new(HiFiGanVocoder::with_config(hifigan_config));
    println!("✅ Vocoder: HiFi-GAN V2 (balanced quality/speed)");
    println!();

    // Step 2: Build pipeline
    println!("🏗️  Building TTS pipeline...");
    let pipeline = VoirsPipelineBuilder::new()
        .with_g2p(g2p)
        .with_acoustic_model(acoustic)
        .with_vocoder(vocoder)
        .with_quality(QualityLevel::High)
        .with_speaking_rate(1.0)
        .build()
        .await?;
    
    println!("✅ Pipeline built successfully!");
    println!();

    // Step 3: Synthesize speech
    println!("🎵 Starting speech synthesis...");
    let start_time = std::time::Instant::now();
    
    let audio = pipeline.synthesize_with_config(text, &synthesis_config).await?;
    
    let synthesis_time = start_time.elapsed();
    println!("✅ Synthesis completed in {:.2}s", synthesis_time.as_secs_f32());
    
    // Step 4: Display results
    println!();
    println!("📊 Synthesis Results:");
    println!("   Duration: {:.2}s", audio.duration());
    println!("   Samples: {}", audio.samples().len());
    println!("   Sample Rate: {}Hz", audio.sample_rate());
    println!("   Channels: {}", audio.channels());
    println!("   Real-time Factor: {:.2}x", synthesis_time.as_secs_f32() / audio.duration());
    
    // Calculate audio statistics
    let samples = audio.samples();
    let peak_amplitude = samples.iter().map(|s| s.abs()).fold(0.0, f32::max);
    let rms = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
    
    println!("   Peak Amplitude: {:.3}", peak_amplitude);
    println!("   RMS Level: {:.3}", rms);
    println!("   Dynamic Range: {:.1} dB", 20.0 * (peak_amplitude / rms.max(1e-6)).log10());

    // Step 5: Save audio (optional)
    #[cfg(feature = "audio-io")]
    {
        let output_path = "/tmp/voirs_example.wav";
        println!();
        println!("💾 Saving audio to: {}", output_path);
        audio.save_wav(output_path)?;
        println!("✅ Audio saved successfully!");
    }

    println!();
    println!("🎉 Complete pipeline demonstration finished!");
    println!("   The VoiRS TTS system successfully processed text through:");
    println!("   📝 Text → 🔤 Phonemes → 🎼 Mel Spectrogram → 🎵 Audio");

    Ok(())
}

/// Helper function to demonstrate component analysis
#[allow(dead_code)]
async fn analyze_pipeline_components() -> Result<()> {
    println!("🔍 Component Analysis:");
    println!("======================");

    // Analyze G2P
    let g2p = EnglishG2p::new();
    let test_words = vec!["hello", "world", "synthesis", "speech"];
    
    println!("📝 G2P Analysis:");
    for word in test_words {
        let phonemes = g2p.to_phonemes(word, None).await?;
        println!("   '{}' → {:?}", word, phonemes.iter().map(|p| &p.symbol).collect::<Vec<_>>());
    }
    println!();

    // Analyze VITS model
    let vits_model = VitsModel::new()?;
    let vits_metadata = vits_model.metadata();
    println!("🎼 VITS Acoustic Model:");
    println!("   Name: {}", vits_metadata.name);
    println!("   Architecture: {}", vits_metadata.architecture);
    println!("   Sample Rate: {}Hz", vits_metadata.sample_rate);
    println!("   Mel Channels: {}", vits_metadata.mel_channels);
    println!("   Multi-speaker: {}", vits_metadata.is_multi_speaker);
    println!();

    // Analyze HiFi-GAN variants
    let variants = voirs_vocoder::models::hifigan::HiFiGanVariants::compare_variants();
    println!("🎵 HiFi-GAN Vocoder Variants:");
    for variant in variants {
        println!("   {}:", variant.name);
        println!("     Parameters: {:.1}M", variant.parameters as f32 / 1_000_000.0);
        println!("     Model Size: {:.1} MB", variant.model_size_mb);
        println!("     Quality Score: {:.1}/5.0", variant.quality_score);
        println!("     Speed Score: {:.1}/5.0", variant.speed_score);
        println!("     Upsampling Factor: {}x", variant.upsampling_factor);
    }

    Ok(())
}

/// Helper function to demonstrate streaming synthesis
#[allow(dead_code)]
async fn demonstrate_streaming_synthesis() -> Result<()> {
    use futures::StreamExt;

    println!("📡 Streaming Synthesis Demo:");
    println!("============================");

    let long_text = "This is a longer text that will be processed in chunks to demonstrate the streaming synthesis capabilities of VoiRS. The system can process text incrementally and generate audio streams for real-time applications.";

    // Build pipeline
    let pipeline = Arc::new(
        VoirsPipelineBuilder::new()
            .with_quality(QualityLevel::Medium) // Use medium quality for faster streaming
            .build()
            .await?
    );

    println!("📝 Streaming text: \"{}\"", long_text);
    println!("🎵 Processing in chunks...");

    let mut audio_stream = pipeline.synthesize_stream(long_text).await?;
    let mut chunk_count = 0;
    let mut total_duration = 0.0;

    while let Some(audio_result) = audio_stream.next().await {
        let audio = audio_result?;
        chunk_count += 1;
        total_duration += audio.duration();
        
        println!("   Chunk {}: {:.2}s audio, {} samples", 
                chunk_count, audio.duration(), audio.samples().len());
    }

    println!("✅ Streaming complete: {} chunks, {:.2}s total audio", chunk_count, total_duration);

    Ok(())
}

/// Helper function to demonstrate voice management
#[allow(dead_code)]
async fn demonstrate_voice_management() -> Result<()> {
    println!("🎭 Voice Management Demo:");
    println!("=========================");

    // This would demonstrate voice loading, switching, and management
    // when voice models are available
    println!("💡 Voice management features:");
    println!("   - Load and switch between different voice models");
    println!("   - Manage voice characteristics and styles");
    println!("   - Support for multi-speaker models");
    println!("   - Voice cloning and adaptation capabilities");

    Ok(())
}
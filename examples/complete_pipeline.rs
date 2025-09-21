//! Complete VoiRS TTS Pipeline Example
//!
//! This example demonstrates the full text-to-speech pipeline using:
//! - Rule-based G2P for phoneme conversion
//! - VITS acoustic model with normalizing flows
//! - HiFi-GAN vocoder for high-quality audio synthesis

use voirs::{
    create_acoustic, create_g2p, create_vocoder, AcousticBackend, G2pBackend, VocoderBackend,
};
use voirs::{AudioFormat, QualityLevel, Result, SynthesisConfig, VoirsPipelineBuilder};

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
        language: voirs::LanguageCode::EnUs,
        effects: Vec::new(),
        streaming_chunk_size: None,
        seed: Some(42),
        enable_emotion: false,
        emotion_type: None,
        emotion_intensity: 0.7,
        emotion_preset: None,
        auto_emotion_detection: false,
        ..Default::default()
    };

    println!("📝 Input text: \"{text}\"");
    println!("⚙️  Configuration: {synthesis_config:?}");
    println!();

    // Step 1: Initialize components
    println!("🔧 Initializing pipeline components...");

    // Initialize G2P
    let g2p = create_g2p(G2pBackend::RuleBased);
    println!("✅ G2P: English rule-based phoneme converter");

    // Initialize VITS acoustic model
    let acoustic = create_acoustic(AcousticBackend::Vits);
    println!("✅ Acoustic Model: VITS with normalizing flows");

    // Initialize HiFi-GAN vocoder
    let vocoder = create_vocoder(VocoderBackend::HifiGan);
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

    let audio = pipeline
        .synthesize_with_config(text, &synthesis_config)
        .await?;

    let synthesis_time = start_time.elapsed();
    println!(
        "✅ Synthesis completed in {:.2}s",
        synthesis_time.as_secs_f32()
    );

    // Step 4: Display results
    println!();
    println!("📊 Synthesis Results:");
    println!("   Duration: {:.2}s", audio.duration());
    println!("   Samples: {}", audio.samples().len());
    println!("   Sample Rate: {}Hz", audio.sample_rate());
    println!("   Channels: {}", audio.channels());
    println!(
        "   Real-time Factor: {:.2}x",
        synthesis_time.as_secs_f32() / audio.duration()
    );

    // Calculate audio statistics
    let samples = audio.samples();
    let peak_amplitude = samples.iter().map(|s| s.abs()).fold(0.0, f32::max);
    let rms = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();

    println!("   Peak Amplitude: {peak_amplitude:.3}");
    println!("   RMS Level: {rms:.3}");
    println!(
        "   Dynamic Range: {:.1} dB",
        20.0 * (peak_amplitude / rms.max(1e-6)).log10()
    );

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

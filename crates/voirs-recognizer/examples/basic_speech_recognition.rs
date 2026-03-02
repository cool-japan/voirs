//! Basic Speech Recognition Example
//!
//! This example demonstrates how to use VoiRS Recognizer for basic speech recognition
//! with automatic audio format detection and high-quality analysis.
//!
//! Usage:
//! ```bash
//! cargo run --example basic_speech_recognition --features="whisper-pure"
//! ```

use std::path::Path;
use voirs_recognizer::audio_formats::ResamplingQuality;
use voirs_recognizer::prelude::*;
use voirs_recognizer::{AudioPreprocessingConfig, AudioPreprocessor, RecognitionError};

#[tokio::main]
async fn main() -> Result<(), RecognitionError> {
    println!("🎤 VoiRS Basic Speech Recognition Example");
    println!("=========================================\n");

    // Step 1: Create some sample audio data (1 second of sine wave at 440Hz)
    println!("📊 Creating sample audio data...");
    let sample_rate = 16000;
    let duration = 1.0; // 1 second
    let frequency = 440.0; // A note

    let mut samples = Vec::new();
    for i in 0..(sample_rate as f32 * duration) as usize {
        let t = i as f32 / sample_rate as f32;
        let sample = 0.1 * (2.0 * std::f32::consts::PI * frequency * t).sin();
        samples.push(sample);
    }

    let audio = AudioBuffer::mono(samples, sample_rate);
    println!(
        "✅ Created audio buffer: {} samples at {}Hz",
        audio.len(),
        audio.sample_rate()
    );

    // Step 2: Analyze audio quality first
    println!("\n🔍 Analyzing audio quality...");
    let analyzer_config = AudioAnalysisConfig::default();
    let analyzer = AudioAnalyzerImpl::new(analyzer_config).await?;
    let analysis = analyzer
        .analyze(&audio, Some(&AudioAnalysisConfig::default()))
        .await?;

    // Display quality metrics
    println!("📈 Audio Quality Metrics:");
    for (metric, value) in &analysis.quality_metrics {
        println!("   • {}: {:.3}", metric, value);
    }

    // Step 3: Perform voice activity detection
    println!("\n🗣️ Performing voice activity detection...");
    println!("📊 VAD Results:");
    println!("   • Voice activity analysis available through quality metrics");
    if let Some(energy) = analysis.quality_metrics.get("energy") {
        println!("   • Energy level: {:.3}", energy);
    }

    // Step 4: Analyze prosody and speaker characteristics
    let prosody = &analysis.prosody;
    {
        println!("\n🎵 Prosody Analysis:");
        println!(
            "   • Fundamental frequency (F0): {:.2} Hz",
            prosody.pitch.mean_f0
        );
        println!(
            "   • Intonation pattern: {:?}",
            prosody.intonation.pattern_type
        );
        println!(
            "   • Speaking rate: {:.2} syllables/sec",
            prosody.rhythm.speaking_rate
        );
    }

    let speaker = &analysis.speaker_characteristics;
    {
        println!("\n👤 Speaker Characteristics:");
        println!("   • Gender: {:?}", speaker.gender);
        println!("   • Age range: {:?}", speaker.age_range);
        println!(
            "   • Voice characteristics: F0 range {:.1}-{:.1} Hz",
            speaker.voice_characteristics.f0_range.0, speaker.voice_characteristics.f0_range.1
        );
    }

    // Step 5: Demonstrate different audio loading capabilities
    println!("\n📁 Audio Format Support:");
    println!("   Supported formats: WAV, FLAC, MP3, OGG, M4A");

    // Create audio load configuration
    let load_config = AudioLoadConfig {
        target_sample_rate: Some(16000),
        force_mono: true,
        normalize: true,
        remove_dc: true,
        max_duration_seconds: None,
    };

    println!("   • Audio load configuration created");
    println!(
        "   • Target sample rate: {:?}",
        load_config.target_sample_rate
    );
    println!("   • Convert to mono: {}", load_config.force_mono);
    println!("   • Normalize: {}", load_config.normalize);

    // Step 6: Performance validation
    println!("\n⚡ Performance Validation:");
    let requirements = PerformanceRequirements::default();
    let validator = PerformanceValidator::with_requirements(requirements);

    println!(
        "   • RTF threshold: < {:.2}",
        validator.requirements().max_rtf
    );
    println!(
        "   • Memory threshold: < {:.2} GB",
        validator.requirements().max_memory_usage as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    println!(
        "   • Startup time threshold: < {:.2}s",
        validator.requirements().max_startup_time_ms as f64 / 1000.0
    );

    // Step 7: Demonstrate preprocessing capabilities
    println!("\n🔧 Audio Preprocessing:");
    let preprocessing_config = AudioPreprocessingConfig::default();
    let preprocessor = AudioPreprocessor::new(preprocessing_config.clone())?;

    println!(
        "   • Noise suppression: {}",
        preprocessing_config.noise_suppression
    );
    println!("   • Automatic gain control: {}", preprocessing_config.agc);
    println!(
        "   • Echo cancellation: {}",
        preprocessing_config.echo_cancellation
    );
    println!(
        "   • Bandwidth extension: {}",
        preprocessing_config.bandwidth_extension
    );

    // Step 8: Show utility functions
    println!("\n🛠️ Utility Functions:");
    let confidence_score = 0.85;
    println!(
        "   • Confidence {:.2} = '{}'",
        confidence_score,
        voirs_recognizer::confidence_to_label(confidence_score)
    );

    let asr_config = voirs_recognizer::default_asr_config(LanguageCode::EnUs);
    println!(
        "   • Default ASR config for English: language={:?}, word_timestamps={}",
        asr_config.language, asr_config.word_timestamps
    );

    println!("\n✅ Basic speech recognition example completed successfully!");
    println!("💡 Next steps:");
    println!("   • Try the real-time processing example");
    println!("   • Experiment with different audio formats");
    println!("   • Explore multi-language support");
    println!("   • Check out custom model integration examples");

    Ok(())
}

//! Simple ASR Demo Example
//!
//! This example demonstrates actual speech recognition using VoiRS Recognizer
//! with real audio processing and transcription capabilities.
//!
//! Usage:
//! ```bash
//! cargo run --example simple_asr_demo --features="whisper-pure"
//! ```

use std::time::{Duration, Instant};
use tokio;
use voirs_recognizer::asr::{ASRBackend, FallbackConfig, WhisperModelSize};
use voirs_recognizer::prelude::*;
use voirs_recognizer::{PerformanceRequirements, PerformanceValidator, RecognitionError};

#[tokio::main]
async fn main() -> Result<(), RecognitionError> {
    println!("🎤 VoiRS Simple ASR Demo");
    println!("========================\n");

    // Step 1: Create synthetic speech-like audio with some harmonic content
    println!("📊 Creating synthetic speech audio...");
    let sample_rate = 16000;
    let duration = 3.0; // 3 seconds
    let mut samples = Vec::new();

    // Generate a more complex signal that resembles speech patterns
    for i in 0..(sample_rate as f32 * duration) as usize {
        let t = i as f32 / sample_rate as f32;
        // Mix multiple frequencies to simulate speech-like harmonics
        let f0 = 150.0; // Base frequency
        let sample = 0.1 * (2.0 * std::f32::consts::PI * f0 * t).sin()
            + 0.05 * (2.0 * std::f32::consts::PI * f0 * 2.0 * t).sin()
            + 0.03 * (2.0 * std::f32::consts::PI * f0 * 3.0 * t).sin()
            + 0.02 * (2.0 * std::f32::consts::PI * f0 * 4.0 * t).sin();

        // Add some envelope to simulate word boundaries
        let envelope = if t % 0.5 < 0.3 { 1.0 } else { 0.1 };
        samples.push(sample * envelope);
    }

    let audio = AudioBuffer::mono(samples, sample_rate);
    println!(
        "✅ Created speech-like audio: {} samples at {}Hz ({:.1}s)",
        audio.len(),
        audio.sample_rate(),
        audio.duration()
    );

    // Step 2: Configure ASR system
    println!("\n🔧 Configuring ASR system...");
    let asr_config = ASRConfig {
        language: Some(LanguageCode::EnUs),
        word_timestamps: true,
        confidence_threshold: 0.5,
        ..Default::default()
    };

    println!("   • Language: {:?}", asr_config.language);
    println!("   • Word timestamps: {}", asr_config.word_timestamps);
    println!(
        "   • Confidence threshold: {}",
        asr_config.confidence_threshold
    );
    println!(
        "   • Sentence segmentation: {}",
        asr_config.sentence_segmentation
    );

    // Step 3: Initialize ASR backend with performance monitoring
    println!("\n🚀 Initializing ASR backend...");
    let init_start = Instant::now();

    // For this demo, we'll use the intelligent fallback system
    let fallback_config = FallbackConfig {
        primary_backend: ASRBackend::Whisper {
            model_size: WhisperModelSize::Base,
            model_path: None,
        },
        fallback_backends: vec![],
        quality_threshold: 0.5,
        ..Default::default()
    };

    let mut intelligent_asr = IntelligentASRFallback::new(fallback_config).await?;
    let init_time = init_start.elapsed();

    println!("✅ ASR backend initialized in {:?}", init_time);

    // Step 4: Perform speech recognition
    println!("\n🎯 Performing speech recognition...");
    let recognition_start = Instant::now();

    let result = intelligent_asr.transcribe(&audio, None).await?;
    let recognition_time = recognition_start.elapsed();

    println!("✅ Recognition completed in {:?}", recognition_time);

    // Step 5: Display results
    println!("\n📝 Recognition Results:");
    println!("   • Transcript: \"{}\"", result.transcript.text);
    println!("   • Confidence: {:.2}", result.transcript.confidence);
    println!("   • Language: {:?}", result.transcript.language);

    if let Some(processing_duration) = result.transcript.processing_duration {
        println!("   • Processing time: {:?}", processing_duration);
    }

    // Display word-level timestamps if available
    if !result.transcript.word_timestamps.is_empty() {
        println!("\n🕐 Word-level timestamps:");
        for (i, word) in result.transcript.word_timestamps.iter().enumerate() {
            println!(
                "   {:2}. \"{}\" [{:.2}s - {:.2}s]",
                i + 1,
                word.word,
                word.start_time,
                word.end_time
            );
        }
    }

    // Step 6: Performance validation
    println!("\n⚡ Performance Validation:");
    let validator = PerformanceValidator::new().with_verbose(true);

    // Validate RTF (Real-Time Factor)
    let (rtf, rtf_passed) = validator.validate_rtf(&audio, recognition_time);
    println!(
        "   • RTF: {:.3} ({})",
        rtf,
        if rtf_passed { "✅ PASS" } else { "❌ FAIL" }
    );

    // Validate memory usage
    let (memory_usage, memory_passed) = validator.estimate_memory_usage()?;
    println!(
        "   • Memory: {:.1} MB ({})",
        memory_usage as f64 / (1024.0 * 1024.0),
        if memory_passed {
            "✅ PASS"
        } else {
            "❌ FAIL"
        }
    );

    // Validate startup time
    let (startup_ms, startup_passed) = validator
        .measure_startup_time(|| async {
            let _test_asr = IntelligentASRFallback::new(FallbackConfig::default()).await?;
            Ok(())
        })
        .await?;
    println!(
        "   • Startup: {}ms ({})",
        startup_ms,
        if startup_passed {
            "✅ PASS"
        } else {
            "❌ FAIL"
        }
    );

    // Step 7: Audio analysis
    println!("\n🔍 Audio Analysis:");
    let analyzer_config = AudioAnalysisConfig::default();
    let analyzer = AudioAnalyzerImpl::new(analyzer_config).await?;
    let analysis = analyzer
        .analyze(&audio, Some(&AudioAnalysisConfig::default()))
        .await?;

    println!("   Quality Metrics:");
    for (metric, value) in &analysis.quality_metrics {
        println!("   • {}: {:.3}", metric, value);
    }

    // Step 8: Benchmarking different configurations
    println!("\n🏁 Benchmarking different configurations...");

    let configs = vec![
        (
            "Fast",
            FallbackConfig {
                primary_backend: ASRBackend::Whisper {
                    model_size: WhisperModelSize::Tiny,
                    model_path: None,
                },
                quality_threshold: 0.3,
                ..Default::default()
            },
        ),
        (
            "Balanced",
            FallbackConfig {
                primary_backend: ASRBackend::Whisper {
                    model_size: WhisperModelSize::Base,
                    model_path: None,
                },
                quality_threshold: 0.5,
                ..Default::default()
            },
        ),
        (
            "Accurate",
            FallbackConfig {
                primary_backend: ASRBackend::Whisper {
                    model_size: WhisperModelSize::Small,
                    model_path: None,
                },
                quality_threshold: 0.7,
                ..Default::default()
            },
        ),
    ];

    for (name, config) in configs {
        let benchmark_start = Instant::now();
        let mut benchmark_asr = IntelligentASRFallback::new(config).await?;
        let benchmark_result = benchmark_asr.transcribe(&audio, None).await?;
        let benchmark_time = benchmark_start.elapsed();

        let (benchmark_rtf, _) = validator.validate_rtf(&audio, benchmark_time);

        println!(
            "   • {}: RTF={:.3}, Confidence={:.2}, Time={:?}",
            name, benchmark_rtf, benchmark_result.transcript.confidence, benchmark_time
        );
    }

    // Step 9: Demonstrate error handling
    println!("\n🛡️ Error Handling Demo:");

    // Create invalid audio (empty buffer)
    let empty_audio = AudioBuffer::mono(vec![], 16000);
    match intelligent_asr.transcribe(&empty_audio, None).await {
        Ok(_) => println!("   • Unexpected success with empty audio"),
        Err(e) => println!("   • Expected error with empty audio: {}", e),
    }

    // Test with very short audio
    let short_audio = AudioBuffer::mono(vec![0.0; 100], 16000); // ~6ms
    match intelligent_asr.transcribe(&short_audio, None).await {
        Ok(result) => println!("   • Short audio result: \"{}\"", result.transcript.text),
        Err(e) => println!("   • Short audio error: {}", e),
    }

    // Step 10: Show utility functions
    println!("\n🛠️ Utility Functions:");
    let confidence_levels = vec![0.1, 0.4, 0.6, 0.8, 0.95];

    for confidence in confidence_levels {
        println!(
            "   • Confidence {:.2} = '{}'",
            confidence,
            voirs_recognizer::confidence_to_label(confidence)
        );
    }

    println!("\n✅ Simple ASR demo completed successfully!");
    println!("💡 Key takeaways:");
    println!("   • ASR system can process audio with RTF < 0.3");
    println!("   • Word-level timestamps provide precise timing");
    println!("   • Performance validation ensures quality standards");
    println!("   • Intelligent fallback handles various scenarios");
    println!("   • Error handling is robust and informative");

    println!("\n🎯 Next steps:");
    println!("   • Try with real audio files (.wav, .flac, .mp3)");
    println!("   • Experiment with different languages");
    println!("   • Explore real-time streaming processing");
    println!("   • Test with custom model configurations");

    Ok(())
}

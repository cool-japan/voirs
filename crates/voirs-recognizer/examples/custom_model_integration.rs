//! Custom Model Integration Example
//!
//! This example demonstrates how to integrate and customize different ASR models
//! in VoiRS Recognizer, including model switching, performance optimization,
//! and intelligent fallback mechanisms.
//!
//! Usage:
//! ```bash
//! cargo run --example custom_model_integration --features="whisper-pure,deepspeech,wav2vec2"
//! ```

use std::collections::HashMap;
use std::time::Duration;
use voirs_recognizer::asr::{
    ASRBackend, ASRBenchmarkingSuite, BenchmarkingConfig, FallbackConfig, IntelligentASRFallback,
    WhisperModelSize,
};
use voirs_recognizer::prelude::*;
use voirs_recognizer::RecognitionError;

#[tokio::main]
async fn main() -> Result<(), RecognitionError> {
    println!("🤖 VoiRS Custom Model Integration Example");
    println!("=========================================\n");

    // Step 1: Overview of available ASR backends
    println!("📋 Available ASR Backends:");
    let available_backends = vec![
        (
            "Whisper Pure Rust",
            "High accuracy, multi-language, moderate speed",
        ),
        (
            "DeepSpeech",
            "Privacy-focused, customizable vocabulary, fast",
        ),
        (
            "Wav2Vec2",
            "Self-supervised learning, good for accented speech",
        ),
    ];

    for (name, description) in &available_backends {
        println!("   • {}: {}", name, description);
    }

    // Step 2: Create custom configurations for different models
    println!("\n⚙️ Custom Model Configurations:");

    // High accuracy configuration
    let high_accuracy_config = ASRConfig {
        language: Some(LanguageCode::EnUs),
        word_timestamps: true,
        confidence_threshold: 0.8,
        ..Default::default()
    };
    println!(
        "   ✅ High Accuracy Config: confidence_threshold={:.2}",
        high_accuracy_config.confidence_threshold
    );

    // Low latency configuration
    let low_latency_config = ASRConfig {
        language: Some(LanguageCode::EnUs),
        word_timestamps: false, // Disable for speed
        confidence_threshold: 0.6,
        ..Default::default()
    };
    println!(
        "   ⚡ Low Latency Config: confidence_threshold={:.2}",
        low_latency_config.confidence_threshold
    );

    // Balanced configuration
    let balanced_config = ASRConfig {
        language: Some(LanguageCode::EnUs),
        word_timestamps: true,
        confidence_threshold: 0.7,
        ..Default::default()
    };
    println!(
        "   ⚖️ Balanced Config: confidence_threshold={:.2}",
        balanced_config.confidence_threshold
    );

    // Step 3: Demonstrate model selection based on requirements
    println!("\n🎯 Model Selection Based on Requirements:");

    let use_cases = vec![
        (
            "Real-time transcription",
            ASRBackend::Whisper {
                model_size: WhisperModelSize::Tiny,
                model_path: None,
            },
            &low_latency_config,
        ),
        (
            "High-quality transcription",
            ASRBackend::Whisper {
                model_size: WhisperModelSize::Large,
                model_path: None,
            },
            &high_accuracy_config,
        ),
        (
            "Privacy-sensitive application",
            ASRBackend::DeepSpeech {
                model_path: "/path/to/deepspeech.pbmm".to_string(),
                scorer_path: None,
            },
            &balanced_config,
        ),
        (
            "Accented speech processing",
            ASRBackend::Wav2Vec2 {
                model_id: "facebook/wav2vec2-large-960h".to_string(),
                model_path: None,
            },
            &balanced_config,
        ),
    ];

    for (use_case, backend, config) in &use_cases {
        println!("   • {}: {:?}", use_case, backend);
        println!("     - Language: {:?}", config.language);
        println!("     - Word timestamps: {}", config.word_timestamps);
        println!(
            "     - Confidence threshold: {}",
            config.confidence_threshold
        );
    }

    // Step 4: Create test audio for model comparison
    println!("\n🎵 Creating test audio for model evaluation...");
    let test_audio = create_test_audio();
    println!(
        "   ✅ Generated test audio: {} samples at {}Hz",
        test_audio.len(),
        test_audio.sample_rate()
    );

    // Step 5: Demonstrate intelligent fallback mechanism
    println!("\n🔄 Intelligent ASR Fallback System:");

    let fallback_config = FallbackConfig::default();
    let fallback = IntelligentASRFallback::new(fallback_config).await?;
    println!("   ✅ Fallback system initialized");

    // Simulate different backend availability scenarios
    let scenarios = vec![
        (
            "All models available",
            vec![
                ASRBackend::Whisper {
                    model_size: WhisperModelSize::Base,
                    model_path: None,
                },
                ASRBackend::DeepSpeech {
                    model_path: "models/deepspeech.pbmm".to_string(),
                    scorer_path: None,
                },
                ASRBackend::Wav2Vec2 {
                    model_id: "facebook/wav2vec2-base-960h".to_string(),
                    model_path: None,
                },
            ],
        ),
        (
            "Only Whisper available",
            vec![ASRBackend::Whisper {
                model_size: WhisperModelSize::Base,
                model_path: None,
            }],
        ),
        (
            "Whisper and DeepSpeech",
            vec![
                ASRBackend::Whisper {
                    model_size: WhisperModelSize::Base,
                    model_path: None,
                },
                ASRBackend::DeepSpeech {
                    model_path: "models/deepspeech.pbmm".to_string(),
                    scorer_path: None,
                },
            ],
        ),
    ];

    for (scenario_name, available_backends) in &scenarios {
        println!("\n   📊 Scenario: {}", scenario_name);

        for backend in available_backends {
            let recommended =
                voirs_recognizer::asr::recommended_backend_for_language(LanguageCode::EnUs);
            println!("     • Available: {:?}", backend);
            if backend == &recommended {
                println!("       → Recommended for general use");
            }
        }
    }

    // Step 6: Model performance comparison
    println!("\n📈 Model Performance Comparison:");

    let benchmark_config = BenchmarkingConfig::default();
    let benchmark_suite = ASRBenchmarkingSuite::new(benchmark_config).await?;
    println!("   ✅ Benchmark suite initialized");

    // Simulate performance metrics for different models
    let performance_data = vec![
        (
            "Whisper Pure",
            0.25,
            1.2,
            95.5,
            "High accuracy, good for clean audio",
        ),
        (
            "DeepSpeech",
            0.15,
            0.8,
            92.0,
            "Fast processing, privacy-focused",
        ),
        (
            "Wav2Vec2",
            0.35,
            1.5,
            94.0,
            "Good for accented speech, self-supervised",
        ),
    ];

    println!("   Model Performance Metrics:");
    println!(
        "   {:<12} {:<5} {:<10} {:<7} {:<30}",
        "Model", "RTF", "Memory(GB)", "WER(%)", "Notes"
    );
    println!("   {}", "-".repeat(70));

    for (model, rtf, memory, wer, notes) in &performance_data {
        println!(
            "   {:<12} {:<5.2} {:<10.1} {:<7.1} {:<30}",
            model, rtf, memory, wer, notes
        );
    }

    // Step 7: Custom model optimization techniques
    println!("\n🔧 Model Optimization Techniques:");

    // Quantization options
    println!("   📦 Quantization Options:");
    println!("     • FP16: 50% memory reduction, minimal accuracy loss");
    println!("     • INT8: 75% memory reduction, small accuracy loss");
    println!("     • INT4: 85% memory reduction, noticeable accuracy loss");

    // Caching strategies
    println!("   💾 Caching Strategies:");
    println!("     • Model weight caching: Faster subsequent loads");
    println!("     • Feature caching: Avoid recomputation for similar inputs");
    println!("     • Result caching: Store transcription results");

    // Batch processing
    println!("   📊 Batch Processing:");
    println!("     • Parallel processing: Utilize multiple CPU cores");
    println!("     • GPU batching: Process multiple audio files together");
    println!("     • Memory optimization: Efficient tensor management");

    // Step 8: Configuration best practices
    println!("\n💡 Configuration Best Practices:");

    let best_practices = vec![
        (
            "Choose beam size based on accuracy/speed tradeoff",
            "beam_size: 3-10",
        ),
        (
            "Set appropriate confidence thresholds",
            "confidence_threshold: 0.6-0.8",
        ),
        (
            "Enable word timestamps only when needed",
            "word_timestamps: false for speed",
        ),
        (
            "Use language-specific configurations",
            "language: match your audio content",
        ),
        (
            "Configure preprocessing for your audio quality",
            "noise suppression, AGC, etc.",
        ),
    ];

    for (practice, example) in &best_practices {
        println!("   • {}", practice);
        println!("     Example: {}", example);
    }

    // Step 9: Error handling and recovery strategies
    println!("\n🛡️ Error Handling and Recovery:");

    println!("   Error Recovery Strategies:");
    println!("     • Model loading failures → Fallback to alternative model");
    println!("     • Out of memory errors → Reduce batch size or use quantization");
    println!("     • Low confidence results → Try different model or preprocessing");
    println!("     • Audio format issues → Use universal audio loader");
    println!("     • Performance issues → Enable caching and optimization");

    // Step 10: Integration with VoiRS ecosystem
    println!("\n🔗 VoiRS Ecosystem Integration:");

    println!("   Ecosystem Components:");
    println!("     • voirs-sdk: Core types and error handling");
    println!("     • voirs-acoustic: Speech synthesis integration");
    println!("     • voirs-dataset: Training data management");
    println!("     • voirs-evaluation: Model evaluation and metrics");

    // Demonstrate error conversion
    let recognition_error = RecognitionError::ModelLoadError {
        message: "Example model loading error".to_string(),
        source: None,
    };

    let voirs_error: VoirsError = recognition_error.into();
    println!("   ✅ Error conversion: RecognitionError → VoirsError");
    println!("     Original: Model loading error");
    println!("     Converted: {}", voirs_error);

    // Step 11: Custom model deployment scenarios
    println!("\n🚀 Deployment Scenarios:");

    let deployment_scenarios = vec![
        (
            "Edge device",
            "Use quantized models, enable caching, optimize for low memory",
        ),
        (
            "Cloud service",
            "Use high-accuracy models, enable GPU acceleration, batch processing",
        ),
        (
            "Real-time app",
            "Use low-latency config, streaming recognition, minimal preprocessing",
        ),
        (
            "Batch processing",
            "Use high-accuracy config, parallel processing, comprehensive analysis",
        ),
    ];

    for (scenario, recommendations) in &deployment_scenarios {
        println!("   • {}: {}", scenario, recommendations);
    }

    println!("\n✅ Custom model integration example completed!");
    println!("🎯 Key Integration Points:");
    println!("   • Model selection based on use case requirements");
    println!("   • Custom configuration for accuracy/speed tradeoffs");
    println!("   • Intelligent fallback for robust applications");
    println!("   • Performance optimization and monitoring");
    println!("   • Error handling and recovery strategies");
    println!("   • Seamless VoiRS ecosystem integration");

    Ok(())
}

/// Create test audio with speech-like characteristics for model evaluation
fn create_test_audio() -> AudioBuffer {
    let sample_rate = 16000;
    let duration = 2.0; // 2 seconds for more realistic testing
    let num_samples = (sample_rate as f32 * duration) as usize;
    let mut samples = Vec::with_capacity(num_samples);

    // Create speech-like audio with multiple formants and natural prosody
    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;

        // Fundamental frequency with natural prosody contour
        let f0 = 150.0 + 30.0 * (3.0 * t).sin() + 15.0 * (7.0 * t).sin();

        // Multiple formants for realistic speech
        let formants = vec![
            (730.0, 1.0),  // F1
            (1090.0, 0.7), // F2
            (2440.0, 0.4), // F3
            (3400.0, 0.2), // F4
        ];

        let mut signal = 0.0;

        // Add fundamental frequency
        signal += 0.3 * (2.0 * std::f32::consts::PI * f0 * t).sin();

        // Add formants
        for (freq, amplitude) in &formants {
            signal += amplitude * (2.0 * std::f32::consts::PI * freq * t).sin();
        }

        // Add harmonics of F0
        for harmonic in 2..=5 {
            let harmonic_freq = f0 * harmonic as f32;
            let harmonic_amplitude = 0.1 / harmonic as f32;
            signal += harmonic_amplitude * (2.0 * std::f32::consts::PI * harmonic_freq * t).sin();
        }

        // Natural amplitude envelope (simulating speech segments)
        let envelope = (0.5 + 0.5 * (2.0 * t).sin()).abs() * (0.8 + 0.2 * (13.0 * t).sin());

        signal *= envelope;

        // Add slight noise for realism
        let noise = (i as f32 * 0.001).sin() * 0.01;
        signal += noise;

        samples.push(signal * 0.15); // Normalize amplitude
    }

    AudioBuffer::mono(samples, sample_rate)
}

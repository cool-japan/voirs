//! Transformer ASR Demo
//!
//! This example demonstrates the new Transformer-based end-to-end ASR implementation
//! with multi-head attention mechanisms. The Transformer architecture provides
//! state-of-the-art speech recognition capabilities with improved accuracy
//! and better handling of long-range dependencies in audio.

#[cfg(feature = "transformer")]
use voirs_recognizer::asr::{create_transformer_asr, ASRBackend, TransformerConfig};
use voirs_recognizer::traits::ASRModel;
use voirs_sdk::{AudioBuffer, LanguageCode};

#[cfg(feature = "transformer")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🤖 Transformer ASR Demo");
    println!("=======================");

    // Create a Transformer ASR model with default configuration
    println!("\n📡 Creating Transformer ASR model...");
    let model = create_transformer_asr().await?;

    // Display model metadata
    let metadata = model.metadata();
    println!("✅ Model loaded successfully!");
    println!("   Name: {}", metadata.name);
    println!("   Version: {}", metadata.version);
    println!("   Architecture: {}", metadata.architecture);
    println!("   Model Size: {:.1} MB", metadata.model_size_mb);
    println!(
        "   Inference Speed: {:.1}x real-time",
        metadata.inference_speed
    );
    println!("   Supported Languages: {:?}", metadata.supported_languages);

    // Show supported features
    println!("\n🔧 Supported Features:");
    for feature in &metadata.supported_features {
        println!("   ✓ {:?}", feature);
    }

    // Create sample audio (1 second of simple sine wave)
    println!("\n🔊 Creating test audio...");
    let sample_rate = 16000;
    let duration = 1.0; // 1 second
    let frequency = 440.0; // A4 note

    let samples: Vec<f32> = (0..((sample_rate as f32 * duration) as usize))
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (2.0 * std::f32::consts::PI * frequency * t).sin() * 0.1
        })
        .collect();

    let audio = AudioBuffer::new(samples, sample_rate, 1);
    println!("   Audio duration: {:.1}s", duration);
    println!("   Sample rate: {} Hz", sample_rate);
    println!("   Channels: {}", audio.channels());

    // Perform transcription
    println!("\n🎯 Performing speech recognition...");
    let start_time = std::time::Instant::now();

    let transcript = model.transcribe(&audio, None).await?;

    let processing_time = start_time.elapsed();

    // Display results
    println!("✅ Transcription completed!");
    println!("   Text: \"{}\"", transcript.text);
    println!("   Confidence: {:.1}%", transcript.confidence * 100.0);
    println!("   Language: {:?}", transcript.language);
    println!("   Processing time: {:.1}ms", processing_time.as_millis());

    if let Some(duration) = transcript.processing_duration {
        println!("   Model processing time: {:.1}ms", duration.as_millis());
    }

    // Test language detection feature
    println!("\n🌍 Testing language detection...");
    match model.detect_language(&audio).await {
        Ok(detected_lang) => {
            println!("   Detected language: {:?}", detected_lang);
        }
        Err(_) => {
            println!("   Language detection not fully implemented yet");
        }
    }

    // Demonstrate custom configuration
    println!("\n⚙️ Testing custom Transformer configuration...");
    let custom_config = TransformerConfig {
        encoder_layers: 6,
        decoder_layers: 3,
        model_dim: 256,
        ff_dim: 1024,
        num_heads: 4,
        dropout: 0.1,
        max_seq_len: 512,
        vocab_size: 1000,
        feature_dim: 80,
        window_size: 400,
        hop_length: 160,
    };

    println!("   Custom config:");
    println!("     Encoder layers: {}", custom_config.encoder_layers);
    println!("     Model dimension: {}", custom_config.model_dim);
    println!("     Attention heads: {}", custom_config.num_heads);

    // Create model with custom configuration
    let custom_model =
        voirs_recognizer::asr::transformer::create_transformer_asr_with_config(custom_config)
            .await?;
    let custom_metadata = custom_model.metadata();
    println!(
        "   ✅ Custom model created (Size: {:.1} MB)",
        custom_metadata.model_size_mb
    );

    // Test ASR backend factory
    println!("\n🏭 Testing ASR backend factory...");
    let backend = ASRBackend::transformer();
    println!("   Created Transformer backend: {:?}", backend);

    // Feature support testing
    println!("\n🧪 Testing feature support...");
    let test_features = [
        voirs_recognizer::traits::ASRFeature::WordTimestamps,
        voirs_recognizer::traits::ASRFeature::LanguageDetection,
        voirs_recognizer::traits::ASRFeature::SentenceSegmentation,
        voirs_recognizer::traits::ASRFeature::StreamingInference,
        voirs_recognizer::traits::ASRFeature::NoiseRobustness,
    ];

    for feature in &test_features {
        let supported = model.supports_feature(feature.clone());
        let status = if supported { "✅" } else { "❌" };
        println!("   {} {:?}", status, feature);
    }

    println!("\n🎉 Transformer ASR demo completed successfully!");
    println!("\n📚 Key Features Demonstrated:");
    println!("   • Multi-head attention mechanism");
    println!("   • Positional encoding for sequence modeling");
    println!("   • Feed-forward networks with ReLU activation");
    println!("   • Layer normalization and residual connections");
    println!("   • Configurable architecture parameters");
    println!("   • Integration with VoiRS ASR framework");

    println!("\n💡 Next Steps:");
    println!("   • Implement proper tokenization");
    println!("   • Add beam search decoding");
    println!("   • Train on real speech data");
    println!("   • Add streaming inference support");
    println!("   • Optimize for production deployment");

    Ok(())
}

#[cfg(not(feature = "transformer"))]
fn main() {
    println!("🤖 Transformer ASR Demo");
    println!("=======================");
    println!("❌ The transformer feature is not enabled.");
    println!("   To run this example, enable the transformer feature:");
    println!("   cargo run --example transformer_asr_demo --features transformer");
}

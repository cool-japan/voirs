//! Advanced Features Demo
//!
//! This example demonstrates the integrated advanced features in VoiRS SDK:
//! - Emotion Control: Add emotional expression to synthesis
//! - Voice Cloning: Clone voices from reference samples
//! - Voice Conversion: Transform voice characteristics

use std::sync::Arc;
use voirs_sdk::prelude::*;

#[cfg(feature = "conversion")]
use voirs_conversion::types::{AgeGroup, Gender};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("🎭 VoiRS SDK Advanced Features Demo");
    println!("=====================================\n");

    // Demo 1: Emotion Control Integration
    emotion_control_demo().await?;

    // Demo 2: Voice Cloning Integration
    voice_cloning_demo().await?;

    // Demo 3: Voice Conversion Integration
    voice_conversion_demo().await?;

    // Demo 4: Combined Advanced Features
    combined_features_demo().await?;

    println!("\n✨ Advanced features demo completed successfully!");
    Ok(())
}

/// Demonstrate emotion control integration
async fn emotion_control_demo() -> Result<()> {
    println!("🎭 Demo 1: Emotion Control Integration");
    println!("--------------------------------------");

    // Create pipeline with emotion control enabled
    let pipeline = VoirsPipelineBuilder::new()
        .with_emotion_enabled(true)
        .with_test_mode(true)
        .build()
        .await?;

    // Check if emotion controller is available
    #[cfg(feature = "emotion")]
    {
        if let Some(emotion_controller) = pipeline.emotion_controller() {
            println!("✅ Emotion controller initialized successfully");

            // List available emotion presets
            let presets = emotion_controller.list_presets();
            println!("📋 Available emotion presets: {:?}", presets);
        }

        // Synthesize with different emotions
        println!("\n🎪 Synthesizing with different emotions:");

        let text = "Hello! How are you doing today?";

        // Normal synthesis
        let normal_audio = pipeline.synthesize(text).await?;
        println!(
            "🔸 Normal: {} samples, {:.2}s duration",
            normal_audio.len(),
            normal_audio.duration()
        );

        // Happy emotion
        pipeline.apply_emotion_preset("happy", Some(0.8)).await?;
        let happy_audio = pipeline.synthesize(text).await?;
        println!(
            "😊 Happy: {} samples, {:.2}s duration",
            happy_audio.len(),
            happy_audio.duration()
        );

        // Sad emotion
        pipeline.apply_emotion_preset("sad", Some(0.6)).await?;
        let sad_audio = pipeline.synthesize(text).await?;
        println!(
            "😢 Sad: {} samples, {:.2}s duration",
            sad_audio.len(),
            sad_audio.duration()
        );
    }

    #[cfg(not(feature = "emotion"))]
    {
        println!("⚠️  Emotion feature not enabled. Enable with --features emotion");
    }

    println!("✅ Emotion control demo completed\n");
    Ok(())
}

/// Demonstrate voice cloning integration
async fn voice_cloning_demo() -> Result<()> {
    println!("🎤 Demo 2: Voice Cloning Integration");
    println!("------------------------------------");

    // Create pipeline with voice cloning enabled
    let pipeline = VoirsPipelineBuilder::new()
        .with_cloning_enabled(true)
        .with_test_mode(true)
        .build()
        .await?;

    #[cfg(feature = "cloning")]
    {
        if let Some(voice_cloner) = pipeline.voice_cloner() {
            println!("✅ Voice cloner initialized successfully");

            // Get cloning statistics
            let stats = voice_cloner.get_statistics().await?;
            println!(
                "📊 Cloning stats - Total: {}, Success rate: {:.1}%",
                stats.total_clones,
                stats.success_rate * 100.0
            );
        }

        // Demo quick cloning with synthetic audio
        println!("\n🔬 Performing quick voice clone:");

        // Create dummy audio samples (in practice, these would be real speaker samples)
        let reference_audio: Vec<f32> = (0..44100) // 2 seconds at 22050 Hz
            .map(|i| 0.1 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 22050.0).sin())
            .collect();

        let target_text = "This is a cloned voice speaking!";

        let clone_result = pipeline
            .quick_clone(reference_audio, 22050, target_text.to_string())
            .await?;

        if clone_result.success {
            println!("✅ Voice cloning successful!");
            println!(
                "🎵 Cloned audio: {} samples at {} Hz",
                clone_result.audio.len(),
                clone_result.sample_rate
            );
        } else {
            println!(
                "⚠️  Voice cloning completed with warnings: {:?}",
                clone_result.error_message
            );
        }
    }

    #[cfg(not(feature = "cloning"))]
    {
        println!("⚠️  Voice cloning feature not enabled. Enable with --features cloning");
    }

    println!("✅ Voice cloning demo completed\n");
    Ok(())
}

/// Demonstrate voice conversion integration
async fn voice_conversion_demo() -> Result<()> {
    println!("🔄 Demo 3: Voice Conversion Integration");
    println!("---------------------------------------");

    // Create pipeline with voice conversion enabled
    let pipeline = VoirsPipelineBuilder::new()
        .with_conversion_enabled(true)
        .with_test_mode(true)
        .build()
        .await?;

    #[cfg(feature = "conversion")]
    {
        if let Some(voice_converter) = pipeline.voice_converter() {
            println!("✅ Voice converter initialized successfully");

            // Get conversion statistics
            let stats = voice_converter.get_statistics().await?;
            println!(
                "📊 Conversion stats - Realtime: {}, Quality: {:.1}",
                stats.realtime_enabled, stats.quality_level
            );
        }

        // Demo voice conversions with synthetic audio
        println!("\n🎛️  Performing voice conversions:");

        // Create source audio
        let source_audio: Vec<f32> = (0..22050) // 1 second at 22050 Hz
            .map(|i| 0.2 * (2.0 * std::f32::consts::PI * 220.0 * i as f32 / 22050.0).sin())
            .collect();

        // Age conversion
        let age_result = pipeline
            .convert_age(source_audio.clone(), 22050, AgeGroup::Child)
            .await?;

        if age_result.success {
            println!(
                "👶 Age conversion to child: {} samples",
                age_result.converted_audio.len()
            );
        }

        // Gender conversion
        let gender_result = pipeline
            .convert_gender(source_audio.clone(), 22050, Gender::Female)
            .await?;

        if gender_result.success {
            println!(
                "👩 Gender conversion to female: {} samples",
                gender_result.converted_audio.len()
            );
        }

        // Pitch shift
        let pitch_result = pipeline
            .voice_converter()
            .unwrap()
            .pitch_shift(source_audio, 22050, 1.5)
            .await?; // 50% higher pitch

        if pitch_result.success {
            println!(
                "🎵 Pitch shift (+50%): {} samples",
                pitch_result.converted_audio.len()
            );
        }
    }

    #[cfg(not(feature = "conversion"))]
    {
        println!("⚠️  Voice conversion feature not enabled. Enable with --features conversion");
    }

    println!("✅ Voice conversion demo completed\n");
    Ok(())
}

/// Demonstrate combining multiple advanced features
async fn combined_features_demo() -> Result<()> {
    println!("🌟 Demo 4: Combined Advanced Features");
    println!("-------------------------------------");

    // Create pipeline with all advanced features enabled
    let pipeline = VoirsPipelineBuilder::new()
        .with_emotion_enabled(true)
        .with_cloning_enabled(true)
        .with_conversion_enabled(true)
        .with_quality(QualityLevel::High)
        .with_test_mode(true)
        .build()
        .await?;

    println!("✅ Pipeline with all advanced features initialized");

    // Check which features are available
    let mut available_features = Vec::new();

    #[cfg(feature = "emotion")]
    if pipeline.emotion_controller().is_some() {
        available_features.push("Emotion Control");
    }

    #[cfg(feature = "cloning")]
    if pipeline.voice_cloner().is_some() {
        available_features.push("Voice Cloning");
    }

    #[cfg(feature = "conversion")]
    if pipeline.voice_converter().is_some() {
        available_features.push("Voice Conversion");
    }

    println!("🎯 Available features: {:?}", available_features);

    // Combined workflow example
    println!("\n🔄 Combined workflow:");

    let text = "This demonstrates the power of VoiRS advanced features!";

    // Step 1: Normal synthesis
    let normal_audio = pipeline.synthesize(text).await?;
    println!("1️⃣  Normal synthesis: {} samples", normal_audio.len());

    // Step 2: Add emotion
    #[cfg(feature = "emotion")]
    {
        pipeline.apply_emotion_preset("excited", Some(0.9)).await?;
        let emotional_audio = pipeline.synthesize(text).await?;
        println!(
            "2️⃣  With emotion (excited): {} samples",
            emotional_audio.len()
        );
    }

    // Step 3: Voice conversion on synthesized audio
    #[cfg(feature = "conversion")]
    {
        let audio_samples = normal_audio.samples().to_vec();
        let converted_result = pipeline
            .voice_converter()
            .unwrap()
            .pitch_shift(audio_samples, normal_audio.sample_rate(), 0.8)
            .await?;

        if converted_result.success {
            println!(
                "3️⃣  Voice converted (lower pitch): {} samples",
                converted_result.converted_audio.len()
            );
        }
    }

    println!("✅ Combined features workflow completed");
    println!("\n🎉 All advanced features working together seamlessly!");
    Ok(())
}

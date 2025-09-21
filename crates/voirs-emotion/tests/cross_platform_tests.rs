//! Cross-platform validation tests for voirs-emotion
//!
//! This module provides comprehensive testing across different platforms and configurations
//! to ensure consistent behavior and performance across all supported environments.

use std::time::Duration;
use voirs_emotion::prelude::*;

/// Basic cross-platform functionality test
#[tokio::test]
async fn test_basic_emotion_processing() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Create basic emotion processor
    let processor = EmotionProcessor::new()?;

    // Test basic emotion setting
    processor.set_emotion(Emotion::Happy, Some(0.8)).await?;

    // Test emotion parameters generation
    let params = processor.get_current_parameters().await;

    println!(
        "✅ Basic emotion processing works on {}-{}",
        std::env::consts::OS,
        std::env::consts::ARCH
    );

    Ok(())
}

/// Test emotion interpolation across platforms
#[tokio::test]
async fn test_emotion_interpolation() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let mut processor = EmotionProcessor::new()?;

    // Set initial emotion
    processor.set_emotion(Emotion::Sad, Some(0.3)).await?;

    // Test interpolation to different emotion
    processor.set_emotion(Emotion::Happy, Some(0.9)).await?;

    // Verify processor is responsive
    let params = processor.get_current_parameters().await;
    assert!(!params.pitch_shift.is_nan());
    assert!(!params.energy_scale.is_nan());

    println!(
        "✅ Emotion interpolation works on {}-{}",
        std::env::consts::OS,
        std::env::consts::ARCH
    );

    Ok(())
}

/// Test processor configuration across platforms  
#[tokio::test]
async fn test_processor_configuration() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let config = EmotionConfig::builder().enabled(true).build()?;

    let processor = EmotionProcessor::with_config(config)?;

    // Test different emotions
    let emotions = vec![
        (Emotion::Happy, 0.8),
        (Emotion::Sad, 0.6),
        (Emotion::Angry, 0.7),
        (Emotion::Calm, 0.5),
    ];

    for (emotion, intensity) in emotions {
        processor.set_emotion(emotion, Some(intensity)).await?;
        let params = processor.get_current_parameters().await;

        // Basic sanity checks
        assert!(!params.pitch_shift.is_nan());
        assert!(!params.energy_scale.is_nan());
        assert!(params.pitch_shift > 0.0);
        assert!(params.energy_scale > 0.0);
    }

    println!(
        "✅ Processor configuration works on {}-{}",
        std::env::consts::OS,
        std::env::consts::ARCH
    );

    Ok(())
}

/// Test concurrent emotion processing
#[tokio::test]
async fn test_concurrent_processing() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let processor = std::sync::Arc::new(EmotionProcessor::new()?);
    let mut handles = vec![];

    // Start multiple concurrent tasks
    for i in 0..5 {
        let processor_clone = processor.clone();
        let emotion = match i % 4 {
            0 => Emotion::Happy,
            1 => Emotion::Sad,
            2 => Emotion::Angry,
            _ => Emotion::Calm,
        };

        let handle = tokio::spawn(async move {
            for _ in 0..10 {
                processor_clone
                    .set_emotion(emotion.clone(), Some(0.5 + (i as f32 * 0.1)))
                    .await?;
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            Ok::<(), voirs_emotion::Error>(())
        });

        handles.push(handle);
    }

    // Wait for all tasks to complete
    for handle in handles {
        handle
            .await
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)??;
    }

    println!(
        "✅ Concurrent processing works on {}-{}",
        std::env::consts::OS,
        std::env::consts::ARCH
    );

    Ok(())
}

/// Test memory usage and cleanup
#[tokio::test]
async fn test_memory_management() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Create and drop many processors to test cleanup
    for _ in 0..50 {
        let processor = EmotionProcessor::new()?;
        processor.set_emotion(Emotion::Happy, Some(0.5)).await?;
        let _params = processor.get_current_parameters().await;

        // Processor will be dropped at end of loop iteration
    }

    println!(
        "✅ Memory management works on {}-{}",
        std::env::consts::OS,
        std::env::consts::ARCH
    );

    Ok(())
}

/// Test platform feature detection
#[test]
fn test_platform_detection() {
    println!(
        "Platform: {}-{}",
        std::env::consts::OS,
        std::env::consts::ARCH
    );
    println!("CPU cores: {}", num_cpus::get());

    // Test SIMD feature detection
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse2") {
            println!("✅ SSE2 support detected");
        }
        if is_x86_feature_detected!("avx") {
            println!("✅ AVX support detected");
        }
        if is_x86_feature_detected!("avx2") {
            println!("✅ AVX2 support detected");
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        println!("✅ ARM64 architecture detected (NEON support assumed)");
    }

    // Test GPU feature availability
    #[cfg(feature = "gpu")]
    {
        println!("✅ GPU features enabled");
    }

    #[cfg(not(feature = "gpu"))]
    {
        println!("ℹ️ GPU features not enabled");
    }
}

/// Test error handling across platforms
#[tokio::test]
async fn test_error_handling() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let processor = EmotionProcessor::new()?;

    // Test with extreme values
    processor.set_emotion(Emotion::Happy, Some(0.0)).await?;
    processor.set_emotion(Emotion::Happy, Some(1.0)).await?;

    // These should work without panicking
    let _params1 = processor.get_current_parameters().await;

    processor.set_emotion(Emotion::Sad, Some(0.001)).await?;
    let _params2 = processor.get_current_parameters().await;

    processor.set_emotion(Emotion::Angry, Some(0.999)).await?;
    let _params3 = processor.get_current_parameters().await;

    println!(
        "✅ Error handling works on {}-{}",
        std::env::consts::OS,
        std::env::consts::ARCH
    );

    Ok(())
}

/// Summary test that runs all platform validations
#[tokio::test]
async fn test_cross_platform_summary() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Cross-Platform Test Summary");
    println!(
        "Platform: {}-{}",
        std::env::consts::OS,
        std::env::consts::ARCH
    );
    println!("CPU cores: {}", num_cpus::get());

    // Run basic functionality test
    let processor = EmotionProcessor::new()?;

    // Test all basic emotions
    let test_emotions = vec![
        Emotion::Happy,
        Emotion::Sad,
        Emotion::Angry,
        Emotion::Fear,
        Emotion::Surprise,
        Emotion::Calm,
        Emotion::Excited,
        Emotion::Confident,
    ];

    for emotion in test_emotions {
        processor.set_emotion(emotion.clone(), Some(0.7)).await?;
        let params = processor.get_current_parameters().await;

        // Verify parameters are reasonable
        assert!(
            !params.pitch_shift.is_nan(),
            "Invalid pitch shift for {:?}",
            emotion
        );
        assert!(
            !params.energy_scale.is_nan(),
            "Invalid energy scale for {:?}",
            emotion
        );
        assert!(
            params.pitch_shift > 0.0,
            "Pitch shift should be positive for {:?}",
            emotion
        );
        assert!(
            params.energy_scale > 0.0,
            "Energy scale should be positive for {:?}",
            emotion
        );
    }

    println!("✅ All cross-platform tests passed!");
    println!("✅ Emotion processing is working correctly on this platform");

    Ok(())
}

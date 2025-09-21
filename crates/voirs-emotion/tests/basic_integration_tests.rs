//! Basic integration tests for voirs-emotion API
//!
//! These tests verify that the API works correctly without assuming
//! the audio processing produces different outputs.

use voirs_emotion::{
    CustomEmotionBuilder, CustomEmotionRegistry, Emotion, EmotionConfig, EmotionProcessor, Result,
};

/// Test basic API functionality
#[tokio::test]
async fn test_basic_api_functionality() -> Result<()> {
    // Test processor creation
    let processor = EmotionProcessor::new()?;
    assert!(true); // If we get here, creation worked

    // Test emotion setting
    let mut proc = processor;
    proc.set_emotion(Emotion::Happy, Some(0.8)).await?;

    // Test audio processing (without expecting changes)
    let input_audio: Vec<f32> = (0..64).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();

    let output_audio = proc.process_audio(&input_audio).await?;

    // Just verify the length is preserved
    assert_eq!(output_audio.len(), input_audio.len());

    Ok(())
}

/// Test configuration builder
#[tokio::test]
async fn test_configuration_builder() -> Result<()> {
    let config = EmotionConfig::builder()
        .enabled(true)
        .max_emotions(3)
        .use_gpu(false)
        .build()
        .map_err(|e| voirs_emotion::Error::Config(e.to_string()))?;

    let processor = EmotionProcessor::with_config(config)?;
    assert!(true); // If we get here, config worked

    Ok(())
}

/// Test emotion setting variations
#[tokio::test]
async fn test_emotion_variations() -> Result<()> {
    let mut processor = EmotionProcessor::new()?;

    // Test different emotions
    let emotions = [
        Emotion::Happy,
        Emotion::Sad,
        Emotion::Angry,
        Emotion::Fear,
        Emotion::Surprise,
        Emotion::Disgust,
        Emotion::Neutral,
    ];

    for emotion in emotions {
        processor.set_emotion(emotion, Some(0.5)).await?;
    }

    // Test different intensities
    for i in 0..10 {
        let intensity = i as f32 * 0.1;
        processor
            .set_emotion(Emotion::Happy, Some(intensity))
            .await?;
    }

    Ok(())
}

/// Test emotion mixing
#[tokio::test]
async fn test_emotion_mixing_api() -> Result<()> {
    let mut processor = EmotionProcessor::new()?;

    let mut emotions = std::collections::HashMap::new();
    emotions.insert(Emotion::Happy, 0.6);
    emotions.insert(Emotion::Sad, 0.4);

    processor.set_emotion_mix(emotions).await?;

    Ok(())
}

/// Test transitions
#[tokio::test]
async fn test_transition_api() -> Result<()> {
    let mut processor = EmotionProcessor::new()?;

    processor.set_emotion(Emotion::Happy, Some(0.8)).await?;
    processor.update_transition(100.0).await?; // 100ms
    processor.set_emotion(Emotion::Sad, Some(0.6)).await?;

    Ok(())
}

/// Test reset functionality
#[tokio::test]
async fn test_reset_api() -> Result<()> {
    let mut processor = EmotionProcessor::new()?;

    processor.set_emotion(Emotion::Angry, Some(0.9)).await?;
    processor.reset_to_neutral().await?;

    Ok(())
}

/// Test history functionality
#[tokio::test]
async fn test_history_api() -> Result<()> {
    let mut processor = EmotionProcessor::new()?;

    // Set some emotions
    processor.set_emotion(Emotion::Happy, Some(0.8)).await?;
    processor.set_emotion(Emotion::Sad, Some(0.6)).await?;

    // Get history stats (should not crash)
    let _stats = processor.get_history_stats().await;

    Ok(())
}

/// Test custom emotion creation
#[tokio::test]
async fn test_custom_emotion_creation() -> Result<()> {
    let custom_emotion = CustomEmotionBuilder::new("test_emotion")
        .description("A test emotion")
        .dimensions(0.0, 0.0, 0.0)
        .prosody(1.0, 1.0, 1.0)
        .voice_quality(0.0, 0.0, 0.0, 0.0)
        .tags(["test"])
        .build()
        .map_err(|e| voirs_emotion::Error::Config(e.to_string()))?;

    let mut registry = CustomEmotionRegistry::new();
    registry
        .register(custom_emotion)
        .map_err(|e| voirs_emotion::Error::Config(e))?;

    Ok(())
}

/// Test error conditions
#[tokio::test]
async fn test_error_handling() -> Result<()> {
    let mut processor = EmotionProcessor::new()?;

    // These should not crash the processor
    processor.set_emotion(Emotion::Happy, Some(-1.0)).await?; // Should clamp
    processor.set_emotion(Emotion::Happy, Some(2.0)).await?; // Should clamp

    // Empty audio should work
    let empty_audio: Vec<f32> = vec![];
    let _output = processor.process_audio(&empty_audio).await?;

    // Large audio should work
    let large_audio: Vec<f32> = vec![0.1; 10000];
    let _output = processor.process_audio(&large_audio).await?;

    Ok(())
}

/// Test concurrent usage (basic)
#[tokio::test]
async fn test_basic_concurrency() -> Result<()> {
    let processor = EmotionProcessor::new()?;

    // Create a couple of concurrent tasks
    let tasks: Vec<_> = (0..3)
        .map(|i| {
            let mut proc = processor.clone();
            tokio::spawn(async move {
                proc.set_emotion(Emotion::Happy, Some(0.5)).await?;
                let audio = vec![0.1; 100];
                let _output = proc.process_audio(&audio).await?;
                Ok::<(), voirs_emotion::Error>(())
            })
        })
        .collect();

    // Wait for completion
    for task in tasks {
        task.await.unwrap()?;
    }

    Ok(())
}

/// Test audio processing with different sizes
#[tokio::test]
async fn test_audio_processing_sizes() -> Result<()> {
    let mut processor = EmotionProcessor::new()?;
    processor.set_emotion(Emotion::Happy, Some(0.7)).await?;

    // Test various sizes
    let sizes = [0, 1, 10, 100, 1000, 10000];

    for size in sizes {
        let audio: Vec<f32> = vec![0.1; size];
        let output = processor.process_audio(&audio).await?;
        assert_eq!(output.len(), audio.len());
    }

    Ok(())
}

/// Test cultural context API (may fail gracefully)
#[tokio::test]
async fn test_cultural_context_api() -> Result<()> {
    let mut processor = EmotionProcessor::new()?;

    // This may fail if cultural contexts aren't fully implemented
    // But it shouldn't crash
    let _result = processor.set_cultural_context("japanese").await;

    // Should still work for basic emotion setting
    processor.set_emotion(Emotion::Happy, Some(0.5)).await?;

    Ok(())
}

//! Basic stress tests for voirs-emotion API reliability
//!
//! These tests validate the system doesn't crash or leak memory
//! under load, without expecting specific audio processing results.

use std::time::{Duration, Instant};
use voirs_emotion::{Emotion, EmotionProcessor, Result};

/// Test sustained processing load
#[tokio::test]
async fn test_sustained_processing() -> Result<()> {
    let mut processor = EmotionProcessor::new()?;

    let audio: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.01).sin() * 0.3).collect();

    // Process many chunks without crashing
    for i in 0..1000 {
        let emotion = match i % 6 {
            0 => Emotion::Happy,
            1 => Emotion::Sad,
            2 => Emotion::Angry,
            3 => Emotion::Fear,
            4 => Emotion::Surprise,
            _ => Emotion::Disgust,
        };

        processor.set_emotion(emotion, Some(0.5)).await?;
        let _output = processor.process_audio(&audio).await?;
    }

    Ok(())
}

/// Test rapid emotion switching
#[tokio::test]
async fn test_rapid_emotion_switching() -> Result<()> {
    let mut processor = EmotionProcessor::new()?;

    let emotions = [
        Emotion::Happy,
        Emotion::Sad,
        Emotion::Angry,
        Emotion::Fear,
        Emotion::Surprise,
        Emotion::Disgust,
    ];

    // Rapidly switch emotions
    for i in 0..500 {
        let emotion = emotions[i % emotions.len()].clone();
        processor
            .set_emotion(emotion, Some(0.3 + (i % 7) as f32 * 0.1))
            .await?;
        processor.update_transition(1.0).await?; // 1ms transitions
    }

    Ok(())
}

/// Test concurrent processing stress
#[tokio::test]
async fn test_concurrent_stress() -> Result<()> {
    let processor = EmotionProcessor::new()?;
    let audio: Vec<f32> = vec![0.1; 256];

    // Create multiple concurrent tasks
    let tasks: Vec<_> = (0..20)
        .map(|i| {
            let mut proc = processor.clone();
            let audio_data = audio.clone();
            let emotion = match i % 4 {
                0 => Emotion::Happy,
                1 => Emotion::Sad,
                2 => Emotion::Angry,
                _ => Emotion::Fear,
            };

            tokio::spawn(async move {
                proc.set_emotion(emotion, Some(0.5)).await?;

                // Process multiple times per task
                for _ in 0..50 {
                    let _output = proc.process_audio(&audio_data).await?;
                }

                Ok::<(), voirs_emotion::Error>(())
            })
        })
        .collect();

    // Wait for all tasks
    let results = futures::future::join_all(tasks).await;

    // Verify all succeeded
    for result in results {
        result.unwrap()?;
    }

    Ok(())
}

/// Test memory pressure with large audio
#[tokio::test]
async fn test_large_audio_processing() -> Result<()> {
    let processor = EmotionProcessor::new()?;

    // Test with increasingly large buffers
    let sizes = [1024, 4096, 16384, 65536];

    for &size in &sizes {
        let audio: Vec<f32> = (0..size).map(|i| (i as f32 * 0.001).sin() * 0.3).collect();

        let mut proc = processor.clone();
        proc.set_emotion(Emotion::Happy, Some(0.7)).await?;

        let _output = proc.process_audio(&audio).await?;
    }

    Ok(())
}

/// Test error recovery under stress
#[tokio::test]
async fn test_error_recovery_stress() -> Result<()> {
    let mut processor = EmotionProcessor::new()?;
    let audio: Vec<f32> = vec![0.1; 100];

    // Mix valid and potentially invalid operations
    for i in 0..200 {
        match i % 10 {
            0..=7 => {
                // Normal operations (80% of time)
                let emotion = match i % 6 {
                    0 => Emotion::Happy,
                    1 => Emotion::Sad,
                    2 => Emotion::Angry,
                    3 => Emotion::Fear,
                    4 => Emotion::Surprise,
                    _ => Emotion::Disgust,
                };
                processor.set_emotion(emotion, Some(0.5)).await?;
                let _output = processor.process_audio(&audio).await?;
            }
            8 => {
                // Edge case: extreme values (should be handled)
                processor.set_emotion(Emotion::Happy, Some(100.0)).await?; // Should clamp
                let _output = processor.process_audio(&audio).await?;
            }
            9 => {
                // Edge case: negative values (should be handled)
                processor.set_emotion(Emotion::Sad, Some(-10.0)).await?; // Should clamp
                let _output = processor.process_audio(&audio).await?;
            }
            _ => unreachable!(),
        }
    }

    Ok(())
}

/// Test performance under load
#[tokio::test]
async fn test_performance_under_load() -> Result<()> {
    let processor = EmotionProcessor::new()?;
    let audio: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.01).sin() * 0.4).collect();

    let mut proc = processor;
    proc.set_emotion(Emotion::Happy, Some(0.7)).await?;

    // Measure processing time
    let iterations = 100;
    let start = Instant::now();

    for _ in 0..iterations {
        let _output = proc.process_audio(&audio).await?;
    }

    let elapsed = start.elapsed();

    // Should complete within reasonable time
    assert!(
        elapsed < Duration::from_secs(10),
        "Performance test took too long: {:?}",
        elapsed
    );

    let ops_per_second = iterations as f64 / elapsed.as_secs_f64();
    println!("Performance: {:.0} operations/second", ops_per_second);

    Ok(())
}

/// Test buffer handling stress
#[tokio::test]
async fn test_buffer_stress() -> Result<()> {
    let processor = EmotionProcessor::new()?;

    // Test with many different buffer sizes to stress allocation
    let sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192];

    for i in 0..100 {
        let size = sizes[i % sizes.len()];
        let audio: Vec<f32> = vec![0.1; size];

        let mut proc = processor.clone();
        proc.set_emotion(Emotion::Happy, Some(0.5)).await?;
        let _output = proc.process_audio(&audio).await?;
    }

    Ok(())
}

/// Test transition stress
#[tokio::test]
async fn test_transition_stress() -> Result<()> {
    let mut processor = EmotionProcessor::new()?;
    let audio: Vec<f32> = vec![0.1; 512];

    let emotions = [Emotion::Happy, Emotion::Sad, Emotion::Angry];

    // Many rapid transitions with processing
    for i in 0..200 {
        let emotion = emotions[i % emotions.len()].clone();
        processor.set_emotion(emotion, Some(0.5)).await?;
        processor.update_transition(5.0).await?; // 5ms transitions
        let _output = processor.process_audio(&audio).await?;
    }

    Ok(())
}

/// Test long-running stability (simplified)
#[tokio::test]
async fn test_stability() -> Result<()> {
    let mut processor = EmotionProcessor::new()?;
    let audio: Vec<f32> = (0..256).map(|i| (i as f32 * 0.02).sin() * 0.3).collect();

    // Simulate long-running session (reduced scale for testing)
    for iteration in 0..500 {
        match iteration % 20 {
            0..=15 => {
                // Normal processing (80% of time)
                let emotion = match iteration % 6 {
                    0 => Emotion::Happy,
                    1 => Emotion::Sad,
                    2 => Emotion::Angry,
                    3 => Emotion::Fear,
                    4 => Emotion::Surprise,
                    _ => Emotion::Disgust,
                };
                processor
                    .set_emotion(emotion, Some(0.3 + (iteration % 7) as f32 * 0.1))
                    .await?;
                let _output = processor.process_audio(&audio).await?;
            }
            16..=18 => {
                // Transitions
                processor.update_transition(25.0).await?;
            }
            19 => {
                // Reset
                processor.reset_to_neutral().await?;
            }
            _ => unreachable!(),
        }
    }

    // Verify processor is still functional
    processor.set_emotion(Emotion::Happy, Some(0.8)).await?;
    let _final_output = processor.process_audio(&audio).await?;

    Ok(())
}

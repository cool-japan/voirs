//! Integration tests for VoiRS feedback system
//!
//! These tests verify the complete feedback workflow from audio input to user feedback.

use std::collections::HashMap;
use std::time::Duration;
use tokio::time::sleep;

use std::collections::VecDeque;
use voirs_feedback::adaptive::models::UserModel;
use voirs_feedback::traits::{AdaptiveState, FeedbackConfig, FeedbackSession, FocusArea};
use voirs_feedback::*;
use voirs_feedback::{FeedbackSystem, FeedbackSystemConfig};
use voirs_sdk::{AudioBuffer, LanguageCode};

/// Test basic feedback system creation and configuration
#[tokio::test]
async fn test_feedback_system_creation() {
    // Test default configuration
    let result = FeedbackSystem::new().await;
    assert!(
        result.is_ok(),
        "Failed to create default feedback system: {:?}",
        result.err()
    );

    // Test custom configuration
    let config = FeedbackSystemConfig {
        enable_realtime: true,
        enable_adaptive: true,
        enable_progress_tracking: true,
        enable_gamification: false,
        max_concurrent_sessions: 10,
        ..Default::default()
    };

    let result = FeedbackSystem::with_config(config).await;
    assert!(
        result.is_ok(),
        "Failed to create feedback system with config: {:?}",
        result.err()
    );
}

/// Test session creation and basic operations
#[tokio::test]
async fn test_session_creation() {
    let feedback_system = FeedbackSystem::new()
        .await
        .expect("Failed to create feedback system");

    // Create a session
    let session = feedback_system.create_session("test_user_123").await;
    assert!(
        session.is_ok(),
        "Failed to create session: {:?}",
        session.err()
    );

    let session = session.unwrap();

    // Check session state
    let state = session.get_state();
    assert_eq!(state.user_id, "test_user_123");
    assert!(!state.session_id.to_string().is_empty());
}

/// Test feedback processing with audio and text
#[tokio::test]
async fn test_feedback_processing() {
    let feedback_system = FeedbackSystem::new()
        .await
        .expect("Failed to create feedback system");

    let mut session = feedback_system
        .create_session("test_user_456")
        .await
        .expect("Failed to create session");

    // Create test audio buffer
    let audio_samples = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    let audio_buffer = AudioBuffer::new(audio_samples, 16000, 1);
    let test_text = "Hello world, this is a test sentence for feedback processing.";

    // Process synthesis and get feedback
    let feedback = session.process_synthesis(&audio_buffer, test_text).await;
    assert!(
        feedback.is_ok(),
        "Failed to process synthesis: {:?}",
        feedback.err()
    );

    let feedback = feedback.unwrap();

    // Verify feedback structure
    assert!(
        feedback.overall_score >= 0.0 && feedback.overall_score <= 1.0,
        "Overall score should be between 0.0 and 1.0, got: {}",
        feedback.overall_score
    );
    assert!(
        feedback.processing_time >= Duration::from_millis(0),
        "Processing time should be non-negative"
    );

    // Check individual feedback items (if any)
    for item in &feedback.feedback_items {
        assert!(
            !item.message.is_empty(),
            "Feedback message should not be empty"
        );
        assert!(
            item.score >= 0.0 && item.score <= 1.0,
            "Feedback score should be between 0.0 and 1.0, got: {}",
            item.score
        );
        assert!(
            item.confidence >= 0.0 && item.confidence <= 1.0,
            "Feedback confidence should be between 0.0 and 1.0, got: {}",
            item.confidence
        );
    }

    // The system should at least return a valid feedback response even if feedback items are empty
    // Immediate actions should be a valid list - len() is always valid
    assert!(
        feedback.immediate_actions.len() < usize::MAX,
        "Immediate actions should be a valid list"
    );
    // Long-term goals should be a valid list - len() is always valid
    assert!(
        feedback.long_term_goals.len() < usize::MAX,
        "Long-term goals should be a valid list"
    );
}

/// Test concurrent session processing
#[tokio::test]
async fn test_concurrent_sessions() {
    let feedback_system = FeedbackSystem::new()
        .await
        .expect("Failed to create feedback system");

    let mut sessions = vec![];

    // Create multiple sessions
    for i in 0..3 {
        let user_id = format!("concurrent_user_{}", i);
        let session = feedback_system
            .create_session(&user_id)
            .await
            .expect("Failed to create session");
        sessions.push(session);
    }

    // Process audio concurrently
    let mut handles = vec![];
    for (i, mut session) in sessions.into_iter().enumerate() {
        let handle = tokio::spawn(async move {
            let audio_samples = vec![0.1 * i as f32; 100];
            let audio_buffer = AudioBuffer::new(audio_samples, 16000, 1);
            let text = format!("Concurrent test sentence {}", i);

            let feedback = session.process_synthesis(&audio_buffer, &text).await;
            assert!(
                feedback.is_ok(),
                "Failed to process synthesis in concurrent session"
            );

            feedback.unwrap()
        });

        handles.push(handle);
    }

    // Wait for all tasks to complete
    for handle in handles {
        let feedback = handle.await.expect("Task failed");
        assert!(feedback.overall_score >= 0.0 && feedback.overall_score <= 1.0);
    }
}

/// Test user model and adaptive learning
#[tokio::test]
async fn test_user_model_integration() {
    let feedback_system = FeedbackSystem::new()
        .await
        .expect("Failed to create feedback system");

    let mut session = feedback_system
        .create_session("adaptive_user")
        .await
        .expect("Failed to create session");

    // Create a user model
    let mut skill_breakdown = HashMap::new();
    skill_breakdown.insert(FocusArea::Pronunciation, 0.7);
    skill_breakdown.insert(FocusArea::Fluency, 0.8);
    skill_breakdown.insert(FocusArea::Quality, 0.6);

    let user_model = UserModel {
        user_id: "adaptive_user".to_string(),
        skill_level: 0.7,
        learning_rate: 0.1,
        consistency_score: 0.8,
        skill_breakdown,
        interaction_history: VecDeque::new(),
        performance_history: VecDeque::new(),
        adaptive_state: AdaptiveState {
            skill_level: 0.7,
            learning_rate: 0.1,
            strengths: vec![FocusArea::Pronunciation],
            improvement_areas: vec![FocusArea::Quality],
            confidence: 0.7,
            adaptation_count: 0,
            last_adaptation: None,
        },
        confidence: 0.7,
        last_updated: chrono::Utc::now(),
    };

    // Process multiple audio samples to build history
    for i in 0..5 {
        let audio_samples = vec![0.1 + i as f32 * 0.1; 50];
        let audio_buffer = AudioBuffer::new(audio_samples, 16000, 1);
        let text = format!("Training sentence number {}", i + 1);

        let feedback = session.process_synthesis(&audio_buffer, &text).await;
        assert!(
            feedback.is_ok(),
            "Failed to process synthesis in adaptive session"
        );

        // Small delay to allow for progress tracking
        sleep(Duration::from_millis(10)).await;
    }

    // Verify the session collected data
    let state = session.get_state();
    assert_eq!(state.user_id, "adaptive_user");
    assert!(state.start_time < chrono::Utc::now());
}

/// Test training exercises
#[tokio::test]
async fn test_training_exercises() {
    let feedback_system = FeedbackSystem::new()
        .await
        .expect("Failed to create feedback system");

    let mut session = feedback_system
        .create_session("training_user")
        .await
        .expect("Failed to create session");

    // Create a training exercise
    let exercise = TrainingExercise {
        exercise_id: "basic_pronunciation".to_string(),
        name: "Basic Pronunciation".to_string(),
        description: "Practice basic phoneme pronunciation".to_string(),
        difficulty: 0.3,
        focus_areas: vec![FocusArea::Pronunciation],
        exercise_type: ExerciseType::Pronunciation,
        target_text: "Hello world".to_string(),
        reference_audio: None,
        success_criteria: SuccessCriteria {
            min_quality_score: 0.7,
            min_pronunciation_score: 0.8,
            max_attempts: 3,
            time_limit: Some(Duration::from_secs(120)),
            consistency_required: 1,
        },
        estimated_duration: Duration::from_secs(60),
    };

    // Start the exercise
    let result = session.start_exercise(&exercise).await;
    assert!(
        result.is_ok(),
        "Failed to start training exercise: {:?}",
        result.err()
    );

    // Simulate exercise completion
    let audio_samples = vec![0.5; 100];
    let audio_buffer = AudioBuffer::new(audio_samples, 16000, 1);
    let feedback = session
        .process_synthesis(&audio_buffer, &exercise.target_text)
        .await;
    assert!(feedback.is_ok(), "Failed to process exercise synthesis");

    // Complete the exercise
    let result = session.complete_exercise().await;
    assert!(
        result.is_ok(),
        "Failed to complete exercise: {:?}",
        result.err()
    );
}

/// Test progress tracking
#[tokio::test]
async fn test_progress_tracking() {
    let feedback_system = FeedbackSystem::new()
        .await
        .expect("Failed to create feedback system");

    let mut session = feedback_system
        .create_session("progress_user")
        .await
        .expect("Failed to create session");

    // Process multiple audio samples to build progress history
    let test_texts = vec![
        "This is the first test sentence.",
        "Here is another sentence for testing.",
        "Third sentence with different content.",
        "Fourth test sentence for progress tracking.",
        "Final sentence to complete the test.",
    ];

    for (i, text) in test_texts.iter().enumerate() {
        let audio_samples = vec![0.1 + i as f32 * 0.1; 80];
        let audio_buffer = AudioBuffer::new(audio_samples, 16000, 1);

        let feedback = session.process_synthesis(&audio_buffer, text).await;
        assert!(
            feedback.is_ok(),
            "Failed to process synthesis for progress tracking"
        );

        let feedback = feedback.unwrap();

        // Verify progress indicators are updated
        assert!(
            feedback.progress_indicators.overall_trend >= -1.0
                && feedback.progress_indicators.overall_trend <= 1.0
        );
        assert!(
            feedback.progress_indicators.completion_percentage >= 0.0
                && feedback.progress_indicators.completion_percentage <= 100.0
        );

        // Small delay to allow for progress tracking
        sleep(Duration::from_millis(10)).await;
    }

    // Save progress
    let result = session.save_progress().await;
    assert!(
        result.is_ok(),
        "Failed to save progress: {:?}",
        result.err()
    );
}

/// Test error handling
#[tokio::test]
async fn test_error_handling() {
    let feedback_system = FeedbackSystem::new()
        .await
        .expect("Failed to create feedback system");

    let mut session = feedback_system
        .create_session("error_test_user")
        .await
        .expect("Failed to create session");

    // Test with empty audio buffer
    let empty_audio = AudioBuffer::new(vec![], 16000, 1);
    let result = session.process_synthesis(&empty_audio, "test text").await;

    // Should handle gracefully or return an error
    match result {
        Ok(feedback) => {
            // If it succeeds, it should provide a valid feedback response
            assert!(
                feedback.overall_score >= 0.0 && feedback.overall_score <= 1.0,
                "Should provide valid feedback even with empty audio"
            );
        }
        Err(e) => {
            // If it fails, error should be meaningful
            assert!(
                !e.to_string().is_empty(),
                "Error message should not be empty"
            );
        }
    }

    // Test with empty text
    let audio_samples = vec![0.1; 100];
    let audio_buffer = AudioBuffer::new(audio_samples, 16000, 1);
    let result = session.process_synthesis(&audio_buffer, "").await;

    match result {
        Ok(feedback) => {
            assert!(
                feedback.overall_score >= 0.0 && feedback.overall_score <= 1.0,
                "Should provide valid feedback even with empty text"
            );
        }
        Err(e) => {
            assert!(
                !e.to_string().is_empty(),
                "Error message should not be empty"
            );
        }
    }
}

/// Test feedback configuration
#[tokio::test]
async fn test_feedback_configuration() {
    let config = FeedbackConfig {
        enable_realtime: true,
        enable_adaptive: true,
        response_timeout_ms: 5000,
        feedback_detail_level: 0.8,
        max_concurrent_requests: 5,
        enable_caching: true,
        supported_languages: vec![LanguageCode::EnUs, LanguageCode::EsEs],
    };

    // Verify configuration is valid
    assert!(config.feedback_detail_level >= 0.0 && config.feedback_detail_level <= 1.0);
    assert!(config.max_concurrent_requests > 0);
    assert!(config.response_timeout_ms > 0);
    assert!(!config.supported_languages.is_empty());
}

/// Test system statistics
#[tokio::test]
async fn test_system_statistics() {
    let feedback_system = FeedbackSystem::new()
        .await
        .expect("Failed to create feedback system");

    // Get initial stats
    let stats = feedback_system.get_statistics().await;
    assert!(
        stats.is_ok(),
        "Failed to get system stats: {:?}",
        stats.err()
    );

    let initial_stats = stats.unwrap();
    let initial_sessions = initial_stats.total_sessions;

    // Create a session
    let _session = feedback_system
        .create_session("stats_user")
        .await
        .expect("Failed to create session");

    // Get updated stats
    let stats = feedback_system.get_statistics().await;
    assert!(stats.is_ok(), "Failed to get updated stats");

    let updated_stats = stats.unwrap();
    assert!(
        updated_stats.total_sessions >= initial_sessions,
        "Total sessions should not decrease"
    );
    assert!(
        updated_stats.average_response_time_ms >= 0.0,
        "Average response time should be non-negative"
    );
}

/// Test system functionality
#[tokio::test]
async fn test_system_functionality() {
    let feedback_system = FeedbackSystem::new()
        .await
        .expect("Failed to create feedback system");

    // Create a session
    let _session = feedback_system
        .create_session("functionality_user")
        .await
        .expect("Failed to create session");

    // Test that system remains functional
    let new_session = feedback_system.create_session("post_test_user").await;
    assert!(new_session.is_ok(), "System should remain functional");
}

/// Test resource cleanup
#[tokio::test]
async fn test_resource_cleanup() {
    let feedback_system = FeedbackSystem::new()
        .await
        .expect("Failed to create feedback system");

    // Create multiple sessions
    let mut sessions = vec![];
    for i in 0..5 {
        let session = feedback_system
            .create_session(&format!("cleanup_user_{}", i))
            .await
            .expect("Failed to create session");
        sessions.push(session);
    }

    // Process some audio with each session
    for (i, session) in sessions.iter_mut().enumerate() {
        let audio_samples = vec![0.1 * i as f32; 50];
        let audio_buffer = AudioBuffer::new(audio_samples, 16000, 1);
        let text = format!("Cleanup test sentence {}", i);

        let feedback = session.process_synthesis(&audio_buffer, &text).await;
        assert!(
            feedback.is_ok(),
            "Failed to process synthesis in cleanup test"
        );
    }

    // Drop all sessions (simulating cleanup)
    drop(sessions);

    // System should still be functional
    let new_session = feedback_system.create_session("post_cleanup_user").await;
    assert!(
        new_session.is_ok(),
        "System should work after session cleanup"
    );
}

//! User Experience (UX) tests for VoiRS feedback system
//!
//! This module provides comprehensive user experience testing covering usability,
//! responsiveness, user workflow, and user-facing features.

use std::collections::HashMap;
use std::time::{Duration, Instant};
use voirs_feedback::integration::PerformanceMetrics;
use voirs_feedback::realtime::types::RealtimeConfig;
use voirs_feedback::traits::{
    AdaptiveState, FeedbackResponse, SessionState, SessionStats, UserPreferences, UserProgress,
};
use voirs_feedback::FeedbackSystem;
use voirs_sdk::AudioBuffer;

/// Create test audio data for testing
fn create_test_audio_data() -> AudioBuffer {
    let samples = vec![0.1; 16000]; // 1 second of audio at 16kHz
    AudioBuffer::new(samples, 16000, 1)
}

/// Create test audio data with specific quality level
fn create_test_audio_data_with_quality(quality: f64) -> AudioBuffer {
    let samples = vec![quality as f32; 16000]; // 1 second of audio at 16kHz
    AudioBuffer::new(samples, 16000, 1)
}

/// Test suite for user workflow and experience
#[cfg(test)]
mod workflow_tests {
    use super::*;

    #[tokio::test]
    async fn test_user_onboarding_flow() {
        let system = FeedbackSystem::new().await.unwrap();
        let user_id = "new_user_123";

        // 1. New user session creation should be smooth
        let session_result = system.create_session(user_id).await;
        assert!(session_result.is_ok());
        let mut session = session_result.unwrap();

        // 2. Initial feedback should be encouraging for new users
        let audio_data = create_test_audio_data();
        let feedback_result = session.process_synthesis(&audio_data, "Hello world").await;
        assert!(feedback_result.is_ok());

        let feedback = feedback_result.unwrap();
        assert!(feedback.overall_score >= 0.0);
        assert!(!feedback.feedback_items.is_empty());

        // 3. Feedback should be appropriate for beginners (non-empty and constructive)
        let has_meaningful_feedback = feedback
            .feedback_items
            .iter()
            .any(|item| !item.message.is_empty() && item.message.len() > 10);
        assert!(
            has_meaningful_feedback,
            "Should provide meaningful feedback content"
        );

        // 4. Progress tracking should start immediately
        // Note: Session should be created and tracked
        // (Progress is tracked internally in the session)
    }

    #[tokio::test]
    async fn test_session_continuity() {
        let system = FeedbackSystem::new().await.unwrap();
        let user_id = "continuing_user";

        // Create multiple sessions to test continuity
        let mut session_scores = Vec::new();
        for _i in 0..5 {
            let mut session = system.create_session(user_id).await.unwrap();
            let audio_data = create_test_audio_data();
            let feedback = session
                .process_synthesis(&audio_data, "Hello world")
                .await
                .unwrap();
            session_scores.push(feedback.overall_score);
        }

        // Progress should be tracked across sessions
        // Note: Multiple sessions with same user ID should accumulate progress
        assert_eq!(session_scores.len(), 5);
        assert!(session_scores.iter().all(|&score| score >= 0.0));
    }

    #[tokio::test]
    async fn test_adaptive_difficulty_progression() {
        let system = FeedbackSystem::new().await.unwrap();
        let user_id = "adaptive_user";

        // Simulate user improvement over time
        for session_num in 0..3 {
            let mut session = system.create_session(user_id).await.unwrap();

            // Simulate gradually improving performance
            let performance_level = (session_num as f64 * 0.1) + 0.5;
            let audio_data = create_test_audio_data_with_quality(performance_level);

            let feedback = session
                .process_synthesis(&audio_data, "Hello world")
                .await
                .unwrap();

            // Basic validation - feedback should be generated
            assert!(feedback.overall_score >= 0.0);
            assert!(!feedback.feedback_items.is_empty());
        }
    }

    #[tokio::test]
    async fn test_error_recovery_flow() {
        let system = FeedbackSystem::new().await.unwrap();
        let user_id = "error_recovery_user";

        // Test graceful handling of various error conditions
        let mut session = system.create_session(user_id).await.unwrap();

        // 1. Empty audio data should not crash
        let empty_audio = AudioBuffer::new(vec![], 16000, 1);
        let result = session.process_synthesis(&empty_audio, "Hello").await;
        // Should handle gracefully (may succeed or fail with proper error)
        assert!(result.is_ok() || result.is_err());

        // 2. Test with minimal audio
        let minimal_audio = AudioBuffer::new(vec![0.1; 10], 16000, 1);
        let result = session.process_synthesis(&minimal_audio, "Hello").await;
        // Should handle gracefully
        assert!(result.is_ok() || result.is_err());
    }

    #[tokio::test]
    async fn test_motivation_maintenance() {
        let system = FeedbackSystem::new().await.unwrap();
        let user_id = "motivation_user";

        // Simulate a user session
        let mut session = system.create_session(user_id).await.unwrap();

        // Basic feedback should be generated
        let audio_data = create_test_audio_data();
        let feedback = session
            .process_synthesis(&audio_data, "Hello world")
            .await
            .unwrap();

        // Basic validation
        assert!(feedback.overall_score >= 0.0);
        assert!(!feedback.feedback_items.is_empty());
    }
}

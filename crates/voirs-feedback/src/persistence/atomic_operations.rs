//! Atomic operations module for data persistence consistency
//!
//! This module provides atomic operations to prevent race conditions and
//! ensure data consistency across different persistence backends.

use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use uuid::Uuid;

use crate::persistence::{PersistenceError, PersistenceResult};
use crate::traits::{FeedbackResponse, SessionState, UserPreferences, UserProgress};

/// Atomic operation result
#[derive(Debug)]
pub enum AtomicResult<T> {
    /// Operation completed successfully
    Success(T),
    /// Operation failed due to conflict
    Conflict,
    /// Operation failed due to error
    Error(PersistenceError),
}

/// Atomic operation context for ensuring consistency
#[derive(Debug)]
pub struct AtomicContext {
    /// Active operations lock
    active_operations: Arc<Mutex<std::collections::HashMap<String, Uuid>>>,
    /// Operation timeout in milliseconds
    timeout_ms: u64,
}

impl AtomicContext {
    /// Create a new atomic context
    pub fn new(timeout_ms: u64) -> Self {
        Self {
            active_operations: Arc::new(Mutex::new(std::collections::HashMap::new())),
            timeout_ms,
        }
    }

    /// Begin an atomic operation for a user
    pub async fn begin_operation(&self, user_id: &str) -> PersistenceResult<Uuid> {
        let operation_id = Uuid::new_v4();
        let mut operations = self.active_operations.lock().await;

        // Check if there's already an operation for this user
        if operations.contains_key(user_id) {
            return Err(PersistenceError::IntegrityError {
                message: format!("Atomic operation already in progress for user: {}", user_id),
            });
        }

        operations.insert(user_id.to_string(), operation_id);
        log::debug!(
            "Started atomic operation {} for user {}",
            operation_id,
            user_id
        );
        Ok(operation_id)
    }

    /// End an atomic operation
    pub async fn end_operation(&self, user_id: &str, operation_id: Uuid) -> PersistenceResult<()> {
        let mut operations = self.active_operations.lock().await;

        match operations.get(user_id) {
            Some(current_id) if *current_id == operation_id => {
                operations.remove(user_id);
                log::debug!(
                    "Completed atomic operation {} for user {}",
                    operation_id,
                    user_id
                );
                Ok(())
            }
            Some(other_id) => Err(PersistenceError::IntegrityError {
                message: format!(
                    "Operation ID mismatch for user {}: expected {}, got {}",
                    user_id, other_id, operation_id
                ),
            }),
            None => Err(PersistenceError::IntegrityError {
                message: format!("No active operation found for user: {}", user_id),
            }),
        }
    }

    /// Check if an operation is active for a user
    pub async fn is_operation_active(&self, user_id: &str) -> bool {
        let operations = self.active_operations.lock().await;
        operations.contains_key(user_id)
    }
}

/// Atomic feedback storage with consistency guarantees
#[derive(Debug)]
pub struct AtomicFeedbackStorage {
    /// Internal storage
    storage: Arc<RwLock<std::collections::HashMap<String, Vec<FeedbackResponse>>>>,
    /// Atomic context
    context: AtomicContext,
}

impl AtomicFeedbackStorage {
    /// Create a new atomic feedback storage
    pub fn new() -> Self {
        Self {
            storage: Arc::new(RwLock::new(std::collections::HashMap::new())),
            context: AtomicContext::new(5000), // 5 second timeout
        }
    }

    /// Atomically add feedback for a user
    pub async fn add_feedback(
        &self,
        user_id: &str,
        feedback: FeedbackResponse,
    ) -> PersistenceResult<()> {
        // Begin atomic operation
        let operation_id = self.context.begin_operation(user_id).await?;

        let result = {
            let mut storage = self.storage.write().await;
            let entry = storage.entry(user_id.to_string()).or_insert_with(Vec::new);
            entry.push(feedback);
            Ok(())
        };

        // End atomic operation
        self.context.end_operation(user_id, operation_id).await?;
        result
    }

    /// Atomically get feedback history for a user
    pub async fn get_feedback_history(
        &self,
        user_id: &str,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> PersistenceResult<Vec<FeedbackResponse>> {
        // Check for active write operations
        if self.context.is_operation_active(user_id).await {
            return Err(PersistenceError::IntegrityError {
                message: format!("Write operation in progress for user: {}", user_id),
            });
        }

        let storage = self.storage.read().await;
        let history = storage.get(user_id).cloned().unwrap_or_default();

        let offset = offset.unwrap_or(0);
        let limit = limit.unwrap_or(history.len());

        let result = history.into_iter().skip(offset).take(limit).collect();
        Ok(result)
    }

    /// Atomically update user progress with feedback consistency check
    pub async fn update_with_feedback_consistency(
        &self,
        user_id: &str,
        feedback: FeedbackResponse,
        progress_update: impl FnOnce(&mut UserProgress) -> PersistenceResult<()>,
        progress_storage: &Arc<RwLock<std::collections::HashMap<String, UserProgress>>>,
    ) -> PersistenceResult<()> {
        // Begin atomic operation across both storages
        let operation_id = self.context.begin_operation(user_id).await?;

        let result = {
            // First, update progress
            {
                let mut progress_storage = progress_storage.write().await;
                let progress = progress_storage
                    .entry(user_id.to_string())
                    .or_insert_with(UserProgress::default);
                progress_update(progress)?;
            }

            // Then, store feedback
            {
                let mut feedback_storage = self.storage.write().await;
                let entry = feedback_storage
                    .entry(user_id.to_string())
                    .or_insert_with(Vec::new);
                entry.push(feedback);
            }

            Ok(())
        };

        // End atomic operation
        self.context.end_operation(user_id, operation_id).await?;
        result
    }

    /// Get storage statistics
    pub async fn get_stats(&self) -> (usize, usize) {
        let storage = self.storage.read().await;
        let user_count = storage.len();
        let total_feedback = storage.values().map(|v| v.len()).sum();
        (user_count, total_feedback)
    }
}

impl Default for AtomicFeedbackStorage {
    fn default() -> Self {
        Self::new()
    }
}

/// Validation utilities for data consistency
pub mod validation {
    use super::*;
    use crate::persistence::PersistenceError;

    /// Validate session state consistency
    pub fn validate_session_consistency(session: &SessionState) -> PersistenceResult<()> {
        // Check if session ID is valid
        if session.session_id.is_nil() {
            return Err(PersistenceError::IntegrityError {
                message: "Session ID cannot be nil".to_string(),
            });
        }

        // Check if user ID is valid
        if session.user_id.is_empty() {
            return Err(PersistenceError::IntegrityError {
                message: "User ID cannot be empty in session".to_string(),
            });
        }

        // Check if start time is reasonable
        if session.start_time > chrono::Utc::now() {
            return Err(PersistenceError::IntegrityError {
                message: "Session start time cannot be in the future".to_string(),
            });
        }

        Ok(())
    }

    /// Validate user progress consistency
    pub fn validate_progress_consistency(progress: &UserProgress) -> PersistenceResult<()> {
        // Check if user ID is valid
        if progress.user_id.is_empty() {
            return Err(PersistenceError::IntegrityError {
                message: "User ID cannot be empty in progress".to_string(),
            });
        }

        // Check if overall skill level is reasonable
        if progress.overall_skill_level < 0.0 || progress.overall_skill_level > 1.0 {
            return Err(PersistenceError::IntegrityError {
                message: format!(
                    "Overall skill level must be between 0.0 and 1.0, got: {}",
                    progress.overall_skill_level
                ),
            });
        }

        // Check if individual skill levels are reasonable
        for (skill, level) in &progress.skill_levels {
            if *level < 0.0 || *level > 1.0 {
                return Err(PersistenceError::IntegrityError {
                    message: format!(
                        "Skill level for '{}' must be between 0.0 and 1.0, got: {}",
                        skill, level
                    ),
                });
            }
        }

        Ok(())
    }

    /// Validate feedback response consistency
    pub fn validate_feedback_consistency(feedback: &FeedbackResponse) -> PersistenceResult<()> {
        // Check if timestamp is reasonable
        if feedback.timestamp > chrono::Utc::now() {
            return Err(PersistenceError::IntegrityError {
                message: "Feedback timestamp cannot be in the future".to_string(),
            });
        }

        // Check if overall score is reasonable
        if feedback.overall_score < 0.0 || feedback.overall_score > 1.0 {
            return Err(PersistenceError::IntegrityError {
                message: format!(
                    "Feedback overall score must be between 0.0 and 1.0, got: {}",
                    feedback.overall_score
                ),
            });
        }

        // Check individual feedback items
        for item in &feedback.feedback_items {
            if item.score < 0.0 || item.score > 1.0 {
                return Err(PersistenceError::IntegrityError {
                    message: format!(
                        "Feedback item score must be between 0.0 and 1.0, got: {}",
                        item.score
                    ),
                });
            }
            if item.confidence < 0.0 || item.confidence > 1.0 {
                return Err(PersistenceError::IntegrityError {
                    message: format!(
                        "Feedback item confidence must be between 0.0 and 1.0, got: {}",
                        item.confidence
                    ),
                });
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    #[tokio::test]
    async fn test_atomic_context_operations() {
        let context = AtomicContext::new(1000);

        // Start operation
        let op_id = context.begin_operation("user1").await.unwrap();
        assert!(context.is_operation_active("user1").await);

        // Try to start another operation for same user - should fail
        assert!(context.begin_operation("user1").await.is_err());

        // End operation
        context.end_operation("user1", op_id).await.unwrap();
        assert!(!context.is_operation_active("user1").await);
    }

    #[tokio::test]
    async fn test_atomic_feedback_storage() {
        let storage = AtomicFeedbackStorage::new();

        let feedback = FeedbackResponse {
            feedback_items: vec![crate::traits::UserFeedback {
                message: "Good pronunciation".to_string(),
                suggestion: Some("Keep practicing".to_string()),
                confidence: 0.9,
                score: 0.85,
                priority: 0.5,
                metadata: std::collections::HashMap::new(),
            }],
            overall_score: 0.85,
            immediate_actions: vec!["Practice more".to_string()],
            long_term_goals: vec!["Improve fluency".to_string()],
            progress_indicators: crate::traits::ProgressIndicators::default(),
            timestamp: Utc::now(),
            processing_time: std::time::Duration::from_millis(100),
            feedback_type: crate::traits::FeedbackType::Pronunciation,
        };

        // Add feedback
        storage
            .add_feedback("user1", feedback.clone())
            .await
            .unwrap();

        // Get feedback history
        let history = storage
            .get_feedback_history("user1", None, None)
            .await
            .unwrap();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].overall_score, 0.85);

        // Check stats
        let (user_count, total_feedback) = storage.get_stats().await;
        assert_eq!(user_count, 1);
        assert_eq!(total_feedback, 1);
    }

    #[test]
    fn test_validation_functions() {
        use crate::traits::{SessionState, UserProgress};
        use std::collections::HashMap;

        // Test session validation
        let mut session = SessionState::default();
        session.user_id = "test_user".to_string(); // Fix empty user_id
        assert!(validation::validate_session_consistency(&session).is_ok());

        // Test progress validation
        let mut progress = UserProgress::default();
        progress.user_id = "test_user".to_string(); // Fix empty user_id
        assert!(validation::validate_progress_consistency(&progress).is_ok());

        // Test feedback validation
        let feedback = FeedbackResponse::default();
        assert!(validation::validate_feedback_consistency(&feedback).is_ok());

        // Test invalid cases
        let invalid_session = SessionState::default(); // Empty user_id
        assert!(validation::validate_session_consistency(&invalid_session).is_err());

        let invalid_progress = UserProgress::default(); // Empty user_id
        assert!(validation::validate_progress_consistency(&invalid_progress).is_err());
    }
}

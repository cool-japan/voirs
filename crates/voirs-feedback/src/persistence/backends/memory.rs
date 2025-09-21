//! In-memory persistence backend for testing and development

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::persistence::{
    atomic_operations::{validation, AtomicFeedbackStorage},
    CleanupResult, PersistenceConfig, PersistenceError, PersistenceManager, PersistenceResult,
    StorageStats, UserDataExport,
};
use crate::traits::{FeedbackResponse, SessionState, UserPreferences, UserProgress};

/// In-memory storage structure with atomic operations
#[derive(Debug)]
struct MemoryStorage {
    sessions: HashMap<Uuid, SessionState>,
    user_progress: HashMap<String, UserProgress>,
    user_preferences: HashMap<String, UserPreferences>,
    metadata: HashMap<String, String>,
}

impl Default for MemoryStorage {
    fn default() -> Self {
        Self {
            sessions: HashMap::new(),
            user_progress: HashMap::new(),
            user_preferences: HashMap::new(),
            metadata: HashMap::new(),
        }
    }
}

/// In-memory persistence manager with atomic operations
pub struct MemoryPersistenceManager {
    storage: Arc<RwLock<MemoryStorage>>,
    feedback_storage: AtomicFeedbackStorage,
    config: PersistenceConfig,
}

impl MemoryPersistenceManager {
    /// Create a new memory persistence manager
    pub async fn new(config: PersistenceConfig) -> PersistenceResult<Self> {
        Ok(Self {
            storage: Arc::new(RwLock::new(MemoryStorage::default())),
            feedback_storage: AtomicFeedbackStorage::new(),
            config,
        })
    }
}

#[async_trait]
impl PersistenceManager for MemoryPersistenceManager {
    async fn initialize(&mut self) -> PersistenceResult<()> {
        // Memory backend doesn't need initialization
        log::info!("Memory persistence backend initialized");
        Ok(())
    }

    async fn save_session(&self, session: &SessionState) -> PersistenceResult<()> {
        // Validate session consistency
        validation::validate_session_consistency(session)?;

        let mut storage = self.storage.write().await;
        storage.sessions.insert(session.session_id, session.clone());
        log::debug!("Saved session: {}", session.session_id);
        Ok(())
    }

    async fn load_session(&self, session_id: &Uuid) -> PersistenceResult<SessionState> {
        let storage = self.storage.read().await;
        storage
            .sessions
            .get(session_id)
            .cloned()
            .ok_or_else(|| PersistenceError::NotFound {
                entity_type: "session".to_string(),
                id: session_id.to_string(),
            })
    }

    async fn save_user_progress(
        &self,
        user_id: &str,
        progress: &UserProgress,
    ) -> PersistenceResult<()> {
        // Validate progress consistency
        validation::validate_progress_consistency(progress)?;

        let mut storage = self.storage.write().await;
        storage
            .user_progress
            .insert(user_id.to_string(), progress.clone());
        log::debug!("Saved progress for user: {}", user_id);
        Ok(())
    }

    async fn load_user_progress(&self, user_id: &str) -> PersistenceResult<UserProgress> {
        let storage = self.storage.read().await;
        storage
            .user_progress
            .get(user_id)
            .cloned()
            .ok_or_else(|| PersistenceError::NotFound {
                entity_type: "user_progress".to_string(),
                id: user_id.to_string(),
            })
    }

    async fn save_feedback(
        &self,
        user_id: &str,
        feedback: &FeedbackResponse,
    ) -> PersistenceResult<()> {
        // Validate feedback consistency
        validation::validate_feedback_consistency(feedback)?;

        // Use atomic feedback storage for consistency
        self.feedback_storage
            .add_feedback(user_id, feedback.clone())
            .await?;
        log::debug!("Saved feedback for user: {}", user_id);
        Ok(())
    }

    async fn load_feedback_history(
        &self,
        user_id: &str,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> PersistenceResult<Vec<FeedbackResponse>> {
        // Use atomic feedback storage for consistency
        self.feedback_storage
            .get_feedback_history(user_id, limit, offset)
            .await
    }

    async fn save_preferences(
        &self,
        user_id: &str,
        preferences: &UserPreferences,
    ) -> PersistenceResult<()> {
        let mut storage = self.storage.write().await;
        storage
            .user_preferences
            .insert(user_id.to_string(), preferences.clone());
        log::debug!("Saved preferences for user: {}", user_id);
        Ok(())
    }

    async fn load_preferences(&self, user_id: &str) -> PersistenceResult<UserPreferences> {
        let storage = self.storage.read().await;
        storage
            .user_preferences
            .get(user_id)
            .cloned()
            .ok_or_else(|| PersistenceError::NotFound {
                entity_type: "user_preferences".to_string(),
                id: user_id.to_string(),
            })
    }

    async fn delete_user_data(&self, user_id: &str) -> PersistenceResult<()> {
        let mut storage = self.storage.write().await;

        // Remove user progress
        storage.user_progress.remove(user_id);

        // Remove user preferences
        storage.user_preferences.remove(user_id);

        // Note: Feedback history is handled by atomic storage -
        // we could implement deletion there but for now we'll leave it
        // as atomic storage doesn't expose deletion to maintain consistency

        // Remove sessions for this user
        storage
            .sessions
            .retain(|_, session| session.user_id != user_id);

        log::info!("Deleted all data for user: {}", user_id);
        Ok(())
    }

    async fn export_user_data(&self, user_id: &str) -> PersistenceResult<UserDataExport> {
        let storage = self.storage.read().await;

        let preferences = storage
            .user_preferences
            .get(user_id)
            .cloned()
            .unwrap_or_default();

        let progress = storage.user_progress.get(user_id).cloned().ok_or_else(|| {
            PersistenceError::NotFound {
                entity_type: "user_progress".to_string(),
                id: user_id.to_string(),
            }
        })?;

        let feedback_history = self
            .feedback_storage
            .get_feedback_history(user_id, None, None)
            .await
            .unwrap_or_default();

        let sessions: Vec<SessionState> = storage
            .sessions
            .values()
            .filter(|session| session.user_id == user_id)
            .cloned()
            .collect();

        let mut metadata = HashMap::new();
        metadata.insert("backend".to_string(), "memory".to_string());
        metadata.insert("export_version".to_string(), "1.0".to_string());

        Ok(UserDataExport {
            user_id: user_id.to_string(),
            export_timestamp: Utc::now(),
            preferences,
            progress,
            feedback_history,
            sessions,
            metadata,
        })
    }

    async fn get_storage_stats(&self) -> PersistenceResult<StorageStats> {
        let storage = self.storage.read().await;

        let total_users = storage.user_progress.len();
        let total_sessions = storage.sessions.len();
        let (_, total_feedback_records) = self.feedback_storage.get_stats().await;

        // Estimate storage size (rough calculation)
        let storage_size_bytes = std::mem::size_of_val(&*storage) as u64;

        Ok(StorageStats {
            total_users,
            total_sessions,
            total_feedback_records,
            storage_size_bytes,
            last_cleanup: None, // Memory backend doesn't track cleanup
            db_version: "memory-1.0".to_string(),
        })
    }

    async fn cleanup(&self, older_than: DateTime<Utc>) -> PersistenceResult<CleanupResult> {
        let start_time = std::time::Instant::now();
        let mut storage = self.storage.write().await;

        let initial_sessions = storage.sessions.len();
        let (_, initial_feedback_records) = self.feedback_storage.get_stats().await;

        // Clean up old sessions
        storage
            .sessions
            .retain(|_, session| session.start_time > older_than);

        // Note: Feedback cleanup not implemented in atomic storage to maintain consistency
        // This would require adding a cleanup method to AtomicFeedbackStorage

        let final_sessions = storage.sessions.len();
        let (_, final_feedback_records) = self.feedback_storage.get_stats().await;

        let sessions_cleaned = initial_sessions - final_sessions;
        let feedback_records_cleaned = 0; // Not implemented yet for atomic storage

        let cleanup_duration = start_time.elapsed();

        log::info!(
            "Memory cleanup completed: {} sessions, {} feedback records cleaned in {:?}",
            sessions_cleaned,
            feedback_records_cleaned,
            cleanup_duration
        );

        Ok(CleanupResult {
            sessions_cleaned,
            feedback_records_cleaned,
            bytes_reclaimed: 0, // Memory doesn't track actual bytes
            cleanup_duration,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{SessionStats, UserPreferences};

    #[tokio::test]
    async fn test_memory_persistence_basic_operations() {
        let config = PersistenceConfig::default();
        let mut manager = MemoryPersistenceManager::new(config).await.unwrap();

        // Initialize
        manager.initialize().await.unwrap();

        // Test session save/load
        let session_id = Uuid::new_v4();
        let session = SessionState {
            session_id,
            user_id: "test_user".to_string(),
            start_time: Utc::now(),
            last_activity: Utc::now(),
            current_task: None,
            stats: SessionStats::default(),
            preferences: UserPreferences::default(),
            adaptive_state: crate::traits::AdaptiveState::default(),
            current_exercise: None,
            session_stats: crate::traits::SessionStatistics::default(),
        };

        manager.save_session(&session).await.unwrap();
        let loaded_session = manager.load_session(&session_id).await.unwrap();
        assert_eq!(loaded_session.session_id, session_id);

        // Test user preferences
        let preferences = UserPreferences::default();
        manager
            .save_preferences("test_user", &preferences)
            .await
            .unwrap();
        let loaded_preferences = manager.load_preferences("test_user").await.unwrap();
        assert_eq!(
            loaded_preferences.feedback_style,
            preferences.feedback_style
        );

        // Test stats
        let stats = manager.get_storage_stats().await.unwrap();
        assert_eq!(stats.total_sessions, 1);
    }

    #[tokio::test]
    async fn test_memory_persistence_cleanup() {
        let config = PersistenceConfig::default();
        let mut manager = MemoryPersistenceManager::new(config).await.unwrap();
        manager.initialize().await.unwrap();

        // Add some old data
        let old_time = Utc::now() - chrono::Duration::days(2);
        let session = SessionState {
            session_id: Uuid::new_v4(),
            user_id: "test_user".to_string(),
            start_time: old_time,
            last_activity: old_time,
            current_task: None,
            stats: SessionStats::default(),
            preferences: UserPreferences::default(),
            adaptive_state: crate::traits::AdaptiveState::default(),
            current_exercise: None,
            session_stats: crate::traits::SessionStatistics::default(),
        };

        manager.save_session(&session).await.unwrap();

        // Cleanup
        let cleanup_threshold = Utc::now() - chrono::Duration::days(1);
        let result = manager.cleanup(cleanup_threshold).await.unwrap();

        assert_eq!(result.sessions_cleaned, 1);
    }
}

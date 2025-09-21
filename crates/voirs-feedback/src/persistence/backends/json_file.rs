//! JSON file-based persistence backend

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use tokio::fs;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::persistence::{
    CleanupResult, PersistenceConfig, PersistenceError, PersistenceManager, PersistenceResult,
    StorageStats, UserDataExport,
};
use crate::traits::{FeedbackResponse, SessionState, UserPreferences, UserProgress};

/// JSON file storage structure
#[derive(Debug, Clone, Serialize, Deserialize)]
struct JsonFileStorage {
    sessions: HashMap<Uuid, SessionState>,
    user_progress: HashMap<String, UserProgress>,
    user_preferences: HashMap<String, UserPreferences>,
    feedback_history: HashMap<String, Vec<FeedbackResponse>>,
    metadata: HashMap<String, String>,
    last_updated: DateTime<Utc>,
}

impl Default for JsonFileStorage {
    fn default() -> Self {
        Self {
            sessions: HashMap::new(),
            user_progress: HashMap::new(),
            user_preferences: HashMap::new(),
            feedback_history: HashMap::new(),
            metadata: HashMap::new(),
            last_updated: Utc::now(),
        }
    }
}

/// JSON file persistence manager
pub struct JsonFilePersistenceManager {
    storage: RwLock<JsonFileStorage>,
    file_path: String,
    config: PersistenceConfig,
}

impl JsonFilePersistenceManager {
    /// Create a new JSON file persistence manager
    pub async fn new(config: PersistenceConfig) -> PersistenceResult<Self> {
        let file_path = config.connection_string.clone();

        let storage = if Path::new(&file_path).exists() {
            Self::load_from_file(&file_path).await?
        } else {
            JsonFileStorage::default()
        };

        Ok(Self {
            storage: RwLock::new(storage),
            file_path,
            config,
        })
    }

    /// Load storage from JSON file
    async fn load_from_file(file_path: &str) -> PersistenceResult<JsonFileStorage> {
        let content =
            fs::read_to_string(file_path)
                .await
                .map_err(|e| PersistenceError::ConnectionError {
                    message: format!("Failed to read file {}: {}", file_path, e),
                })?;

        serde_json::from_str(&content).map_err(|e| PersistenceError::SerializationError {
            message: format!("Failed to parse JSON: {}", e),
        })
    }

    /// Save storage to JSON file
    async fn save_to_file(&self, storage: &JsonFileStorage) -> PersistenceResult<()> {
        let content = serde_json::to_string_pretty(storage).map_err(|e| {
            PersistenceError::SerializationError {
                message: format!("Failed to serialize to JSON: {}", e),
            }
        })?;

        // Create parent directory if it doesn't exist
        if let Some(parent) = Path::new(&self.file_path).parent() {
            fs::create_dir_all(parent)
                .await
                .map_err(|e| PersistenceError::ConnectionError {
                    message: format!("Failed to create directory: {}", e),
                })?;
        }

        fs::write(&self.file_path, content).await.map_err(|e| {
            PersistenceError::ConnectionError {
                message: format!("Failed to write file {}: {}", self.file_path, e),
            }
        })?;

        Ok(())
    }

    /// Auto-save storage if needed
    async fn auto_save(&self) -> PersistenceResult<()> {
        let storage = self.storage.read().await;
        self.save_to_file(&*storage).await
    }
}

#[async_trait]
impl PersistenceManager for JsonFilePersistenceManager {
    async fn initialize(&mut self) -> PersistenceResult<()> {
        // Ensure the file exists
        if !Path::new(&self.file_path).exists() {
            let storage = self.storage.read().await;
            self.save_to_file(&*storage).await?;
        }

        log::info!(
            "JSON file persistence backend initialized: {}",
            self.file_path
        );
        Ok(())
    }

    async fn save_session(&self, session: &SessionState) -> PersistenceResult<()> {
        {
            let mut storage = self.storage.write().await;
            storage.sessions.insert(session.session_id, session.clone());
            storage.last_updated = Utc::now();
        }

        self.auto_save().await?;
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
        {
            let mut storage = self.storage.write().await;
            storage
                .user_progress
                .insert(user_id.to_string(), progress.clone());
            storage.last_updated = Utc::now();
        }

        self.auto_save().await?;
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
        {
            let mut storage = self.storage.write().await;
            storage
                .feedback_history
                .entry(user_id.to_string())
                .or_default()
                .push(feedback.clone());
            storage.last_updated = Utc::now();
        }

        self.auto_save().await?;
        log::debug!("Saved feedback for user: {}", user_id);
        Ok(())
    }

    async fn load_feedback_history(
        &self,
        user_id: &str,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> PersistenceResult<Vec<FeedbackResponse>> {
        let storage = self.storage.read().await;
        let history = storage
            .feedback_history
            .get(user_id)
            .cloned()
            .unwrap_or_default();

        let offset = offset.unwrap_or(0);
        let limit = limit.unwrap_or(history.len());

        let result = history.into_iter().skip(offset).take(limit).collect();

        Ok(result)
    }

    async fn save_preferences(
        &self,
        user_id: &str,
        preferences: &UserPreferences,
    ) -> PersistenceResult<()> {
        {
            let mut storage = self.storage.write().await;
            storage
                .user_preferences
                .insert(user_id.to_string(), preferences.clone());
            storage.last_updated = Utc::now();
        }

        self.auto_save().await?;
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
        {
            let mut storage = self.storage.write().await;

            // Remove user progress
            storage.user_progress.remove(user_id);

            // Remove user preferences
            storage.user_preferences.remove(user_id);

            // Remove feedback history
            storage.feedback_history.remove(user_id);

            // Remove sessions for this user
            storage
                .sessions
                .retain(|_, session| session.user_id != user_id);

            storage.last_updated = Utc::now();
        }

        self.auto_save().await?;
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

        let feedback_history = storage
            .feedback_history
            .get(user_id)
            .cloned()
            .unwrap_or_default();

        let sessions: Vec<SessionState> = storage
            .sessions
            .values()
            .filter(|session| session.user_id == user_id)
            .cloned()
            .collect();

        let mut metadata = HashMap::new();
        metadata.insert("backend".to_string(), "json_file".to_string());
        metadata.insert("export_version".to_string(), "1.0".to_string());
        metadata.insert("file_path".to_string(), self.file_path.clone());

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
        let total_feedback_records: usize =
            storage.feedback_history.values().map(|v| v.len()).sum();

        // Get file size
        let storage_size_bytes = if Path::new(&self.file_path).exists() {
            fs::metadata(&self.file_path)
                .await
                .map(|m| m.len())
                .unwrap_or(0)
        } else {
            0
        };

        Ok(StorageStats {
            total_users,
            total_sessions,
            total_feedback_records,
            storage_size_bytes,
            last_cleanup: None, // JSON file backend doesn't track cleanup separately
            db_version: "json-file-1.0".to_string(),
        })
    }

    async fn cleanup(&self, older_than: DateTime<Utc>) -> PersistenceResult<CleanupResult> {
        let start_time = std::time::Instant::now();

        let (sessions_cleaned, feedback_records_cleaned) = {
            let mut storage = self.storage.write().await;

            let initial_sessions = storage.sessions.len();
            let initial_feedback_records: usize =
                storage.feedback_history.values().map(|v| v.len()).sum();

            // Clean up old sessions
            storage
                .sessions
                .retain(|_, session| session.start_time > older_than);

            // Clean up old feedback records
            for feedback_list in storage.feedback_history.values_mut() {
                feedback_list.retain(|feedback| feedback.timestamp > older_than);
            }

            storage.last_updated = Utc::now();

            let final_sessions = storage.sessions.len();
            let final_feedback_records: usize =
                storage.feedback_history.values().map(|v| v.len()).sum();

            (
                initial_sessions - final_sessions,
                initial_feedback_records - final_feedback_records,
            )
        };

        // Save the cleaned storage
        self.auto_save().await?;

        let cleanup_duration = start_time.elapsed();

        log::info!(
            "JSON file cleanup completed: {} sessions, {} feedback records cleaned in {:?}",
            sessions_cleaned,
            feedback_records_cleaned,
            cleanup_duration
        );

        Ok(CleanupResult {
            sessions_cleaned,
            feedback_records_cleaned,
            bytes_reclaimed: 0, // Would need to compare file sizes to calculate
            cleanup_duration,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{SessionStats, UserPreferences};
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_json_file_persistence_basic_operations() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test.json");

        let mut config = PersistenceConfig::default();
        config.connection_string = file_path.to_string_lossy().to_string();

        let mut manager = JsonFilePersistenceManager::new(config).await.unwrap();

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

        // Test persistence across manager instances
        drop(manager);

        let mut config = PersistenceConfig::default();
        config.connection_string = file_path.to_string_lossy().to_string();

        let manager2 = JsonFilePersistenceManager::new(config).await.unwrap();
        let loaded_session2 = manager2.load_session(&session_id).await.unwrap();
        assert_eq!(loaded_session2.session_id, session_id);
    }

    #[tokio::test]
    async fn test_json_file_persistence_file_creation() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("nonexistent_dir").join("test.json");

        let mut config = PersistenceConfig::default();
        config.connection_string = file_path.to_string_lossy().to_string();

        let mut manager = JsonFilePersistenceManager::new(config).await.unwrap();

        // Should create directory and file
        manager.initialize().await.unwrap();

        assert!(file_path.exists());
    }
}

//! Offline capability support for VoiRS feedback system
//!
//! This module provides offline functionality including data caching, offline-first
//! operations, and seamless online/offline transitions.

use crate::traits::{FeedbackResponse, SessionState, UserProgress};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, SystemTime};

/// Offline manager for handling offline operations
pub struct OfflineManager {
    config: OfflineConfig,
    cache: OfflineCache,
    queue: OperationQueue,
    /// Description
    pub storage: OfflineStorage,
}

impl OfflineManager {
    /// Create a new offline manager
    pub fn new(config: OfflineConfig) -> Self {
        Self {
            config,
            cache: OfflineCache::new(),
            queue: OperationQueue::new(),
            storage: OfflineStorage::new(),
        }
    }

    /// Check if system is currently offline
    pub fn is_offline(&self) -> bool {
        // This would check network connectivity
        // For now, simulate offline state
        false
    }

    /// Switch to offline mode
    pub fn switch_to_offline(&mut self) -> Result<(), OfflineError> {
        // Prepare system for offline operation
        self.cache.preload_essential_data()?;
        self.storage.prepare_offline_storage()?;
        Ok(())
    }

    /// Switch to online mode
    pub async fn switch_to_online(&mut self) -> Result<(), OfflineError> {
        // Sync offline operations
        self.sync_offline_operations().await?;

        // Clear old cache if needed
        if self.config.clear_cache_on_online {
            self.cache.clear_expired_data()?;
        }

        Ok(())
    }

    /// Process feedback in offline mode
    pub async fn process_feedback_offline(
        &mut self,
        session: &SessionState,
        audio_data: &[u8],
        text: &str,
    ) -> Result<FeedbackResponse, OfflineError> {
        // Check if we have cached models
        if !self.cache.has_cached_models() {
            return Err(OfflineError::ModelNotCached);
        }

        // Process using cached models
        let feedback = self.generate_offline_feedback(session, audio_data, text)?;

        // Queue for later sync
        let operation = QueuedOperation {
            id: uuid::Uuid::new_v4().to_string(),
            operation_type: OperationType::ProcessFeedback,
            session_id: session.session_id.to_string(),
            data: serde_json::to_value(&feedback).unwrap(),
            timestamp: chrono::Utc::now(),
            retry_count: 0,
        };

        self.queue.add_operation(operation);

        Ok(feedback)
    }

    /// Generate feedback using offline models
    fn generate_offline_feedback(
        &self,
        session: &SessionState,
        audio_data: &[u8],
        text: &str,
    ) -> Result<FeedbackResponse, OfflineError> {
        // Simplified offline feedback generation
        // In a real implementation, this would use cached ML models

        let feedback_response = FeedbackResponse {
            feedback_items: vec![crate::traits::UserFeedback {
                message: format!("Offline feedback for: {}", text),
                suggestion: Some("Continue practicing to improve".to_string()),
                confidence: 0.7,
                score: 0.75,
                priority: 0.5,
                metadata: HashMap::new(),
            }],
            overall_score: 0.75, // Default offline score
            immediate_actions: vec!["Continue practicing".to_string()],
            long_term_goals: vec!["Improve pronunciation accuracy".to_string()],
            progress_indicators: crate::traits::ProgressIndicators {
                improving_areas: vec!["pronunciation".to_string()],
                attention_areas: vec!["pace".to_string()],
                stable_areas: vec!["clarity".to_string()],
                overall_trend: 0.0,
                completion_percentage: 75.0,
            },
            processing_time: Duration::from_millis(50),
            timestamp: chrono::Utc::now(),
            feedback_type: crate::FeedbackType::Quality,
        };

        Ok(feedback_response)
    }

    /// Save user progress offline
    pub fn save_progress_offline(&mut self, progress: &UserProgress) -> Result<(), OfflineError> {
        // Save to local storage
        self.storage.save_user_progress(progress)?;

        // Queue for sync when online
        let operation = QueuedOperation {
            id: uuid::Uuid::new_v4().to_string(),
            operation_type: OperationType::SaveProgress,
            session_id: progress.user_id.clone(),
            data: serde_json::to_value(progress).unwrap(),
            timestamp: chrono::Utc::now(),
            retry_count: 0,
        };

        self.queue.add_operation(operation);

        Ok(())
    }

    /// Load user progress from offline storage
    pub fn load_progress_offline(
        &self,
        user_id: &str,
    ) -> Result<Option<UserProgress>, OfflineError> {
        self.storage.load_user_progress(user_id)
    }

    /// Sync offline operations when back online
    async fn sync_offline_operations(&mut self) -> Result<(), OfflineError> {
        let operations = self.queue.get_pending_operations();

        for operation in operations {
            match self.sync_operation(&operation).await {
                Ok(_) => {
                    self.queue.mark_completed(&operation.id);
                }
                Err(e) => {
                    // Retry logic
                    if operation.retry_count < self.config.max_retries {
                        self.queue.increment_retry(&operation.id);
                    } else {
                        self.queue.mark_failed(&operation.id);
                    }
                    println!("Failed to sync operation {}: {}", operation.id, e);
                }
            }
        }

        Ok(())
    }

    /// Sync individual operation
    async fn sync_operation(&self, operation: &QueuedOperation) -> Result<(), OfflineError> {
        match operation.operation_type {
            OperationType::ProcessFeedback => {
                // Send feedback to server
                // In real implementation, this would make HTTP request
                Ok(())
            }
            OperationType::SaveProgress => {
                // Upload progress to server
                // In real implementation, this would make HTTP request
                Ok(())
            }
            OperationType::SaveSession => {
                // Upload session data to server
                Ok(())
            }
        }
    }

    /// Get offline status
    pub fn get_offline_status(&self) -> OfflineStatus {
        OfflineStatus {
            is_offline: self.is_offline(),
            cached_models: self.cache.has_cached_models(),
            pending_operations: self.queue.get_pending_count(),
            storage_usage: self.storage.get_storage_usage(),
            last_sync: self.queue.get_last_sync_time(),
        }
    }

    /// Clear offline data
    pub fn clear_offline_data(&mut self) -> Result<(), OfflineError> {
        self.cache.clear_all()?;
        self.queue.clear_all();
        self.storage.clear_all()?;
        Ok(())
    }
}

/// Offline configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OfflineConfig {
    /// Enable offline mode
    pub enable_offline: bool,
    /// Maximum cache size in bytes
    pub max_cache_size: u64,
    /// Cache expiration time in seconds
    pub cache_expiration_seconds: u64,
    /// Maximum number of queued operations
    pub max_queued_operations: u32,
    /// Maximum retry attempts
    pub max_retries: u32,
    /// Clear cache when going online
    pub clear_cache_on_online: bool,
    /// Storage directory for offline data
    pub storage_directory: PathBuf,
}

impl Default for OfflineConfig {
    fn default() -> Self {
        Self {
            enable_offline: true,
            max_cache_size: 500 * 1024 * 1024,      // 500MB
            cache_expiration_seconds: 24 * 60 * 60, // 24 hours
            max_queued_operations: 1000,
            max_retries: 3,
            clear_cache_on_online: false,
            storage_directory: PathBuf::from("./offline_data"),
        }
    }
}

/// Offline cache for storing models and data
pub struct OfflineCache {
    cached_models: HashMap<String, CachedModel>,
    cached_data: HashMap<String, CachedData>,
    total_size: u64,
}

impl OfflineCache {
    /// Create a new offline cache
    pub fn new() -> Self {
        Self {
            cached_models: HashMap::new(),
            cached_data: HashMap::new(),
            total_size: 0,
        }
    }

    /// Check if essential models are cached
    pub fn has_cached_models(&self) -> bool {
        // Check if we have the essential models for offline operation
        self.cached_models.contains_key("pronunciation_model")
            && self.cached_models.contains_key("feedback_model")
    }

    /// Preload essential data for offline operation
    pub fn preload_essential_data(&mut self) -> Result<(), OfflineError> {
        // Preload pronunciation model
        self.cache_model("pronunciation_model", b"mock_pronunciation_model_data")?;

        // Preload feedback model
        self.cache_model("feedback_model", b"mock_feedback_model_data")?;

        // Preload common phrases
        self.cache_data("common_phrases", b"mock_common_phrases_data")?;

        Ok(())
    }

    /// Cache a model
    fn cache_model(&mut self, model_id: &str, data: &[u8]) -> Result<(), OfflineError> {
        let model = CachedModel {
            id: model_id.to_string(),
            data: data.to_vec(),
            cached_at: chrono::Utc::now(),
            size: data.len() as u64,
        };

        self.total_size += model.size;
        self.cached_models.insert(model_id.to_string(), model);

        Ok(())
    }

    /// Cache data
    fn cache_data(&mut self, data_id: &str, data: &[u8]) -> Result<(), OfflineError> {
        let cached_data = CachedData {
            id: data_id.to_string(),
            data: data.to_vec(),
            cached_at: chrono::Utc::now(),
            size: data.len() as u64,
        };

        self.total_size += cached_data.size;
        self.cached_data.insert(data_id.to_string(), cached_data);

        Ok(())
    }

    /// Clear expired data
    pub fn clear_expired_data(&mut self) -> Result<(), OfflineError> {
        let now = chrono::Utc::now();
        let expiration_duration = chrono::Duration::hours(24);

        // Remove expired models
        self.cached_models.retain(|_, model| {
            let age = now.signed_duration_since(model.cached_at);
            if age > expiration_duration {
                self.total_size -= model.size;
                false
            } else {
                true
            }
        });

        // Remove expired data
        self.cached_data.retain(|_, data| {
            let age = now.signed_duration_since(data.cached_at);
            if age > expiration_duration {
                self.total_size -= data.size;
                false
            } else {
                true
            }
        });

        Ok(())
    }

    /// Clear all cached data
    pub fn clear_all(&mut self) -> Result<(), OfflineError> {
        self.cached_models.clear();
        self.cached_data.clear();
        self.total_size = 0;
        Ok(())
    }

    /// Get cache usage
    pub fn get_usage(&self) -> CacheUsage {
        CacheUsage {
            total_size: self.total_size,
            model_count: self.cached_models.len() as u32,
            data_count: self.cached_data.len() as u32,
        }
    }
}

/// Cached model
#[derive(Debug, Clone)]
struct CachedModel {
    id: String,
    data: Vec<u8>,
    cached_at: DateTime<Utc>,
    size: u64,
}

/// Cached data
#[derive(Debug, Clone)]
struct CachedData {
    id: String,
    data: Vec<u8>,
    cached_at: DateTime<Utc>,
    size: u64,
}

/// Operation queue for offline operations
pub struct OperationQueue {
    operations: HashMap<String, QueuedOperation>,
    last_sync: Option<DateTime<Utc>>,
}

impl OperationQueue {
    /// Create a new operation queue
    pub fn new() -> Self {
        Self {
            operations: HashMap::new(),
            last_sync: None,
        }
    }

    /// Add operation to queue
    pub fn add_operation(&mut self, operation: QueuedOperation) {
        self.operations.insert(operation.id.clone(), operation);
    }

    /// Get pending operations
    pub fn get_pending_operations(&self) -> Vec<QueuedOperation> {
        self.operations
            .values()
            .filter(|op| op.retry_count < 3)
            .cloned()
            .collect()
    }

    /// Mark operation as completed
    pub fn mark_completed(&mut self, operation_id: &str) {
        self.operations.remove(operation_id);
    }

    /// Mark operation as failed
    pub fn mark_failed(&mut self, operation_id: &str) {
        // In a real implementation, this would move to a failed operations list
        self.operations.remove(operation_id);
    }

    /// Increment retry count
    pub fn increment_retry(&mut self, operation_id: &str) {
        if let Some(operation) = self.operations.get_mut(operation_id) {
            operation.retry_count += 1;
        }
    }

    /// Get pending operations count
    pub fn get_pending_count(&self) -> u32 {
        self.operations.len() as u32
    }

    /// Get last sync time
    pub fn get_last_sync_time(&self) -> Option<DateTime<Utc>> {
        self.last_sync
    }

    /// Clear all operations
    pub fn clear_all(&mut self) {
        self.operations.clear();
    }
}

/// Queued operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueuedOperation {
    /// Description
    pub id: String,
    /// Description
    pub operation_type: OperationType,
    /// Description
    pub session_id: String,
    /// Description
    pub data: serde_json::Value,
    /// Description
    pub timestamp: DateTime<Utc>,
    /// Description
    pub retry_count: u32,
}

/// Operation type
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OperationType {
    /// Description
    ProcessFeedback,
    /// Description
    SaveProgress,
    /// Description
    SaveSession,
}

/// Offline storage
pub struct OfflineStorage {
    storage_directory: PathBuf,
    user_progress: HashMap<String, UserProgress>,
}

impl OfflineStorage {
    /// Create a new offline storage
    pub fn new() -> Self {
        Self {
            storage_directory: std::env::temp_dir().join("voirs_offline_data"),
            user_progress: HashMap::new(),
        }
    }

    /// Create a new offline storage with custom directory
    pub fn with_directory(storage_directory: PathBuf) -> Self {
        Self {
            storage_directory,
            user_progress: HashMap::new(),
        }
    }

    /// Prepare offline storage
    pub fn prepare_offline_storage(&self) -> Result<(), OfflineError> {
        // Check if the path exists as a file
        if self.storage_directory.is_file() {
            return Err(OfflineError::StorageError {
                message: format!(
                    "Storage path exists as a file: {:?}",
                    self.storage_directory
                ),
            });
        }

        // Create storage directory if it doesn't exist
        if !self.storage_directory.exists() {
            std::fs::create_dir_all(&self.storage_directory).map_err(|e| {
                OfflineError::StorageError {
                    message: format!("Failed to create storage directory: {}", e),
                }
            })?;
        }

        Ok(())
    }

    /// Save user progress
    pub fn save_user_progress(&mut self, progress: &UserProgress) -> Result<(), OfflineError> {
        // Store in memory first
        self.user_progress
            .insert(progress.user_id.clone(), progress.clone());

        // Try to persist to disk (optional, graceful fallback)
        if let Err(e) = self.try_persist_to_disk(progress) {
            // Log the error but don't fail the operation
            eprintln!("Warning: Could not persist to disk: {}", e);
        }

        Ok(())
    }

    /// Try to persist user progress to disk (fallback method)
    fn try_persist_to_disk(&self, progress: &UserProgress) -> Result<(), OfflineError> {
        // Ensure storage directory exists
        std::fs::create_dir_all(&self.storage_directory).map_err(|e| {
            OfflineError::StorageError {
                message: format!("Failed to create storage directory: {}", e),
            }
        })?;

        // Create safe filename (replace problematic characters)
        let safe_user_id = progress
            .user_id
            .replace(['/', '\\', ':', '*', '?', '"', '<', '>', '|'], "_");
        let file_path = self
            .storage_directory
            .join(format!("{}.json", safe_user_id));

        let json_data = serde_json::to_string_pretty(progress).map_err(|e| {
            OfflineError::SerializationError {
                message: format!("Failed to serialize user progress: {}", e),
            }
        })?;

        std::fs::write(&file_path, json_data).map_err(|e| OfflineError::StorageError {
            message: format!("Failed to write user progress: {}", e),
        })?;

        Ok(())
    }

    /// Load user progress
    pub fn load_user_progress(&self, user_id: &str) -> Result<Option<UserProgress>, OfflineError> {
        // First check in-memory cache
        if let Some(progress) = self.user_progress.get(user_id) {
            return Ok(Some(progress.clone()));
        }

        // Then check persistent storage
        let file_path = self.storage_directory.join(format!("{}.json", user_id));
        if file_path.exists() {
            let json_data =
                std::fs::read_to_string(&file_path).map_err(|e| OfflineError::StorageError {
                    message: format!("Failed to read user progress: {}", e),
                })?;

            let progress: UserProgress =
                serde_json::from_str(&json_data).map_err(|e| OfflineError::SerializationError {
                    message: format!("Failed to deserialize user progress: {}", e),
                })?;

            Ok(Some(progress))
        } else {
            Ok(None)
        }
    }

    /// Get storage usage
    pub fn get_storage_usage(&self) -> StorageUsage {
        // In a real implementation, this would calculate actual disk usage
        StorageUsage {
            used_bytes: 0,
            available_bytes: 1024 * 1024 * 1024, // 1GB
        }
    }

    /// Clear all data
    pub fn clear_all(&mut self) -> Result<(), OfflineError> {
        self.user_progress.clear();

        // In a real implementation, this would clear persistent storage
        if self.storage_directory.exists() {
            // Remove all files in the directory
            if let Ok(entries) = std::fs::read_dir(&self.storage_directory) {
                for entry in entries {
                    if let Ok(entry) = entry {
                        let path = entry.path();
                        if path.is_file() {
                            let _ = std::fs::remove_file(&path);
                        }
                    }
                }
            }

            // Try to remove the directory, but don't error if it fails
            let _ = std::fs::remove_dir(&self.storage_directory);
        }

        Ok(())
    }
}

/// Offline status
#[derive(Debug, Clone)]
pub struct OfflineStatus {
    /// Description
    pub is_offline: bool,
    /// Description
    pub cached_models: bool,
    /// Description
    pub pending_operations: u32,
    /// Description
    pub storage_usage: StorageUsage,
    /// Description
    pub last_sync: Option<DateTime<Utc>>,
}

/// Cache usage information
#[derive(Debug, Clone)]
pub struct CacheUsage {
    /// Description
    pub total_size: u64,
    /// Description
    pub model_count: u32,
    /// Description
    pub data_count: u32,
}

/// Storage usage information
#[derive(Debug, Clone)]
pub struct StorageUsage {
    /// Description
    pub used_bytes: u64,
    /// Description
    pub available_bytes: u64,
}

/// Offline error types
#[derive(Debug, thiserror::Error)]
pub enum OfflineError {
    #[error("Model not cached: Required model not available offline")]
    /// Description
    ModelNotCached,

    #[error("Storage error: {message}")]
    /// Description
    /// Description
    StorageError {
        /// Human-readable description of the storage issue.
        message: String,
    },

    #[error("Serialization error: {message}")]
    /// Description
    /// Description
    SerializationError {
        /// Human-readable description of the serialization issue.
        message: String,
    },

    #[error("Cache error: {message}")]
    /// Description
    /// Description
    CacheError {
        /// Human-readable description of the cache issue.
        message: String,
    },

    #[error("Network error: {message}")]
    /// Description
    /// Description
    NetworkError {
        /// Human-readable description of the network issue.
        message: String,
    },

    #[error("Configuration error: {message}")]
    /// Description
    /// Description
    ConfigError {
        /// Human-readable description of the configuration issue.
        message: String,
    },
}

/// Offline result type
pub type OfflineResult<T> = Result<T, OfflineError>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_offline_config_default() {
        let config = OfflineConfig::default();
        assert!(config.enable_offline);
        assert_eq!(config.max_cache_size, 500 * 1024 * 1024);
        assert_eq!(config.cache_expiration_seconds, 24 * 60 * 60);
        assert_eq!(config.max_queued_operations, 1000);
        assert_eq!(config.max_retries, 3);
        assert!(!config.clear_cache_on_online);
    }

    #[test]
    fn test_offline_manager_creation() {
        let config = OfflineConfig::default();
        let manager = OfflineManager::new(config);
        assert!(!manager.is_offline());
    }

    #[test]
    fn test_offline_cache_operations() {
        let mut cache = OfflineCache::new();

        // Initially no models cached
        assert!(!cache.has_cached_models());

        // Preload essential data
        cache.preload_essential_data().unwrap();

        // Should have cached models now
        assert!(cache.has_cached_models());

        // Check cache usage
        let usage = cache.get_usage();
        assert!(usage.total_size > 0);
        assert!(usage.model_count > 0);
        assert!(usage.data_count > 0);
    }

    #[test]
    fn test_operation_queue() {
        let mut queue = OperationQueue::new();

        // Initially empty
        assert_eq!(queue.get_pending_count(), 0);

        // Add operation
        let operation = QueuedOperation {
            id: "test_op".to_string(),
            operation_type: OperationType::ProcessFeedback,
            session_id: "session_1".to_string(),
            data: serde_json::json!({"test": "data"}),
            timestamp: chrono::Utc::now(),
            retry_count: 0,
        };

        queue.add_operation(operation);
        assert_eq!(queue.get_pending_count(), 1);

        // Mark as completed
        queue.mark_completed("test_op");
        assert_eq!(queue.get_pending_count(), 0);
    }

    #[test]
    fn test_offline_storage() {
        let mut storage = OfflineStorage::new();

        // Prepare storage directory
        storage.prepare_offline_storage().unwrap();

        // Test user progress operations
        let progress = UserProgress {
            user_id: "test_user".to_string(),
            overall_skill_level: 0.8,
            skill_breakdown: HashMap::new(),
            progress_history: vec![],
            achievements: vec![],
            training_stats: crate::traits::TrainingStatistics::default(),
            goals: vec![],
            last_updated: chrono::Utc::now(),
            average_scores: crate::traits::SessionScores::default(),
            skill_levels: HashMap::new(),
            recent_sessions: vec![],
            personal_bests: HashMap::new(),
            session_count: 5,
            total_practice_time: Duration::from_secs(3600),
        };

        // Save progress
        storage.save_user_progress(&progress).unwrap();

        // Load progress
        let loaded = storage.load_user_progress("test_user").unwrap();
        assert!(loaded.is_some());
        assert_eq!(loaded.unwrap().user_id, "test_user");

        // Non-existent user
        let non_existent = storage.load_user_progress("non_existent").unwrap();
        assert!(non_existent.is_none());
    }

    #[test]
    fn test_queued_operation_serialization() {
        let operation = QueuedOperation {
            id: "test_id".to_string(),
            operation_type: OperationType::SaveProgress,
            session_id: "session_123".to_string(),
            data: serde_json::json!({"progress": 85}),
            timestamp: chrono::Utc::now(),
            retry_count: 2,
        };

        let serialized = serde_json::to_string(&operation).unwrap();
        let deserialized: QueuedOperation = serde_json::from_str(&serialized).unwrap();

        assert_eq!(operation.id, deserialized.id);
        assert_eq!(operation.operation_type, deserialized.operation_type);
        assert_eq!(operation.session_id, deserialized.session_id);
        assert_eq!(operation.data, deserialized.data);
        assert_eq!(operation.retry_count, deserialized.retry_count);
    }

    #[test]
    fn test_operation_type_equality() {
        assert_eq!(
            OperationType::ProcessFeedback,
            OperationType::ProcessFeedback
        );
        assert_eq!(OperationType::SaveProgress, OperationType::SaveProgress);
        assert_eq!(OperationType::SaveSession, OperationType::SaveSession);
        assert_ne!(OperationType::ProcessFeedback, OperationType::SaveProgress);
    }

    #[tokio::test]
    async fn test_offline_manager_operations() {
        let config = OfflineConfig::default();
        let mut manager = OfflineManager::new(config);

        // Test switch to offline
        manager.switch_to_offline().unwrap();

        // Test offline status
        let status = manager.get_offline_status();
        assert!(status.cached_models);
        assert_eq!(status.pending_operations, 0);

        // Test offline feedback processing
        let session = SessionState {
            user_id: "test_user".to_string(),
            session_id: uuid::Uuid::new_v4(),
            start_time: chrono::Utc::now(),
            last_activity: chrono::Utc::now(),
            current_task: None,
            stats: crate::traits::SessionStats::default(),
            preferences: crate::traits::UserPreferences::default(),
            adaptive_state: crate::traits::AdaptiveState::default(),
            current_exercise: None,
            session_stats: crate::traits::SessionStatistics::default(),
        };

        let feedback = manager
            .process_feedback_offline(&session, b"test_audio", "test text")
            .await;
        assert!(feedback.is_ok());

        let feedback_response = feedback.unwrap();
        assert!(!feedback_response.feedback_items.is_empty());
        assert_eq!(feedback_response.overall_score, 0.75);

        // Should have queued operation
        let status = manager.get_offline_status();
        assert_eq!(status.pending_operations, 1);
    }

    #[test]
    fn test_cache_cleanup() {
        let mut cache = OfflineCache::new();

        // Add some data
        cache.preload_essential_data().unwrap();
        let initial_usage = cache.get_usage();

        // Clear expired data (in test, nothing should be expired)
        cache.clear_expired_data().unwrap();
        let usage_after_cleanup = cache.get_usage();
        assert_eq!(initial_usage.total_size, usage_after_cleanup.total_size);

        // Clear all data
        cache.clear_all().unwrap();
        let usage_after_clear = cache.get_usage();
        assert_eq!(usage_after_clear.total_size, 0);
        assert_eq!(usage_after_clear.model_count, 0);
        assert_eq!(usage_after_clear.data_count, 0);
    }

    #[test]
    fn test_storage_usage() {
        let storage = OfflineStorage::new();
        let usage = storage.get_storage_usage();
        assert!(usage.available_bytes > 0);
    }

    #[tokio::test]
    async fn test_offline_progress_operations() {
        let config = OfflineConfig::default();
        let temp_dir = std::env::temp_dir().join(format!("voirs_test_{}", std::process::id()));
        let mut manager = OfflineManager::new(config);
        manager.storage = OfflineStorage::with_directory(temp_dir.clone());

        // Switch to offline mode to prepare storage
        manager.switch_to_offline().unwrap();

        let progress = UserProgress {
            user_id: "test_user".to_string(),
            overall_skill_level: 0.8,
            skill_breakdown: HashMap::new(),
            progress_history: vec![],
            achievements: vec![],
            training_stats: crate::traits::TrainingStatistics::default(),
            goals: vec![],
            last_updated: chrono::Utc::now(),
            average_scores: crate::traits::SessionScores::default(),
            skill_levels: HashMap::new(),
            recent_sessions: vec![],
            personal_bests: HashMap::new(),
            session_count: 10,
            total_practice_time: Duration::from_secs(7200),
        };

        // Save progress offline
        manager.save_progress_offline(&progress).unwrap();

        // Load progress offline
        let loaded = manager.load_progress_offline("test_user").unwrap();
        assert!(loaded.is_some());
        assert_eq!(loaded.unwrap().user_id, "test_user");

        // Should have queued operation
        let status = manager.get_offline_status();
        assert_eq!(status.pending_operations, 1);

        // Clean up test directory
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[tokio::test]
    async fn test_online_offline_transitions() {
        let config = OfflineConfig::default();
        let temp_dir =
            std::env::temp_dir().join(format!("voirs_test_trans_{}", std::process::id()));
        let mut manager = OfflineManager::new(config);
        manager.storage = OfflineStorage::with_directory(temp_dir.clone());

        // Switch to offline
        manager.switch_to_offline().unwrap();
        let status = manager.get_offline_status();
        assert!(status.cached_models);

        // Switch back to online
        manager.switch_to_online().await.unwrap();

        // Test clear data
        manager.clear_offline_data().unwrap();
        let status = manager.get_offline_status();
        assert!(!status.cached_models);
        assert_eq!(status.pending_operations, 0);

        // Clean up test directory
        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}

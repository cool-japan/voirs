//! Data persistence system for VoiRS feedback
//!
//! This module provides robust data persistence capabilities for user progress,
//! feedback history, and system state management with support for multiple backends.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;
use uuid::Uuid;

use crate::traits::{FeedbackResponse, SessionState, UserPreferences, UserProgress};

pub mod atomic_operations;
pub mod backends;
pub mod cache;
pub mod encryption;
pub mod migrations;
pub mod query_optimizer;
pub mod scaling;
pub mod sharding;

/// Persistence error types
#[derive(Error, Debug)]
pub enum PersistenceError {
    /// Database connection error
    #[error("Database connection failed: {message}")]
    ConnectionError {
        /// Error message
        message: String,
    },

    /// Data serialization error
    #[error("Data serialization failed: {message}")]
    SerializationError {
        /// Error message
        message: String,
    },

    /// Data not found
    #[error("Data not found: {entity_type} with id {id}")]
    NotFound {
        /// Entity type
        entity_type: String,
        /// Entity ID
        id: String,
    },

    /// Data integrity violation
    #[error("Data integrity violation: {message}")]
    IntegrityError {
        /// Error message
        message: String,
    },

    /// Migration error
    #[error("Migration failed: {message}")]
    MigrationError {
        /// Error message
        message: String,
    },

    /// Encryption error
    #[error("Encryption operation failed: {message}")]
    EncryptionError {
        /// Error message
        message: String,
    },

    /// Configuration error
    #[error("Configuration error: {message}")]
    ConfigError {
        /// Error message
        message: String,
    },
}

/// Result type for persistence operations
pub type PersistenceResult<T> = Result<T, PersistenceError>;

/// Main persistence manager
#[async_trait]
pub trait PersistenceManager: Send + Sync {
    /// Initialize the persistence backend
    async fn initialize(&mut self) -> PersistenceResult<()>;

    /// Save user session state
    async fn save_session(&self, session: &SessionState) -> PersistenceResult<()>;

    /// Load user session state
    async fn load_session(&self, session_id: &Uuid) -> PersistenceResult<SessionState>;

    /// Save user progress data
    async fn save_user_progress(
        &self,
        user_id: &str,
        progress: &UserProgress,
    ) -> PersistenceResult<()>;

    /// Load user progress data
    async fn load_user_progress(&self, user_id: &str) -> PersistenceResult<UserProgress>;

    /// Save feedback response
    async fn save_feedback(
        &self,
        user_id: &str,
        feedback: &FeedbackResponse,
    ) -> PersistenceResult<()>;

    /// Load feedback history for user
    async fn load_feedback_history(
        &self,
        user_id: &str,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> PersistenceResult<Vec<FeedbackResponse>>;

    /// Save user preferences
    async fn save_preferences(
        &self,
        user_id: &str,
        preferences: &UserPreferences,
    ) -> PersistenceResult<()>;

    /// Load user preferences
    async fn load_preferences(&self, user_id: &str) -> PersistenceResult<UserPreferences>;

    /// Delete user data (GDPR compliance)
    async fn delete_user_data(&self, user_id: &str) -> PersistenceResult<()>;

    /// Export user data (GDPR compliance)
    async fn export_user_data(&self, user_id: &str) -> PersistenceResult<UserDataExport>;

    /// Get storage statistics
    async fn get_storage_stats(&self) -> PersistenceResult<StorageStats>;

    /// Perform cleanup operations
    async fn cleanup(&self, older_than: DateTime<Utc>) -> PersistenceResult<CleanupResult>;
}

/// User data export structure for GDPR compliance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserDataExport {
    /// User ID
    pub user_id: String,
    /// Export timestamp
    pub export_timestamp: DateTime<Utc>,
    /// User preferences
    pub preferences: UserPreferences,
    /// Progress data
    pub progress: UserProgress,
    /// Feedback history
    pub feedback_history: Vec<FeedbackResponse>,
    /// Session history
    pub sessions: Vec<SessionState>,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// Storage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageStats {
    /// Total number of users
    pub total_users: usize,
    /// Total number of sessions
    pub total_sessions: usize,
    /// Total number of feedback records
    pub total_feedback_records: usize,
    /// Storage size in bytes
    pub storage_size_bytes: u64,
    /// Last cleanup timestamp
    pub last_cleanup: Option<DateTime<Utc>>,
    /// Database version
    pub db_version: String,
}

/// Cleanup operation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleanupResult {
    /// Number of sessions cleaned
    pub sessions_cleaned: usize,
    /// Number of feedback records cleaned
    pub feedback_records_cleaned: usize,
    /// Bytes reclaimed
    pub bytes_reclaimed: u64,
    /// Cleanup duration
    pub cleanup_duration: std::time::Duration,
}

/// Persistence configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistenceConfig {
    /// Backend type
    pub backend: PersistenceBackend,
    /// Connection string or file path
    pub connection_string: String,
    /// Enable encryption
    pub enable_encryption: bool,
    /// Maximum cache size
    pub max_cache_size: usize,
    /// Cache TTL in seconds
    pub cache_ttl_seconds: u64,
    /// Auto-cleanup interval in hours
    pub auto_cleanup_interval_hours: u64,
    /// Data retention period in days
    pub data_retention_days: u32,
    /// Enable compression
    pub enable_compression: bool,
    /// Connection pool size
    pub connection_pool_size: u32,
}

impl Default for PersistenceConfig {
    fn default() -> Self {
        Self {
            backend: PersistenceBackend::SQLite,
            connection_string: "voirs_feedback.db".to_string(),
            enable_encryption: true,
            max_cache_size: 1000,
            cache_ttl_seconds: 3600,
            auto_cleanup_interval_hours: 24,
            data_retention_days: 365,
            enable_compression: true,
            connection_pool_size: 10,
        }
    }
}

/// Supported persistence backends
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PersistenceBackend {
    /// SQLite file database
    SQLite,
    /// PostgreSQL database
    PostgreSQL,
    /// In-memory storage (for testing)
    Memory,
    /// File-based JSON storage
    JsonFile,
}

/// Create a persistence manager with the given configuration
pub async fn create_persistence_manager(
    config: PersistenceConfig,
) -> PersistenceResult<Box<dyn PersistenceManager>> {
    match config.backend {
        PersistenceBackend::SQLite => {
            #[cfg(feature = "persistence")]
            {
                let manager = backends::sqlite::SQLitePersistenceManager::new(config).await?;
                Ok(Box::new(manager))
            }
            #[cfg(not(feature = "persistence"))]
            {
                Err(PersistenceError::ConfigError {
                    message: "SQLite backend requires 'persistence' feature".to_string(),
                })
            }
        }
        PersistenceBackend::Memory => {
            let manager = backends::memory::MemoryPersistenceManager::new(config).await?;
            Ok(Box::new(manager))
        }
        PersistenceBackend::JsonFile => {
            let manager = backends::json_file::JsonFilePersistenceManager::new(config).await?;
            Ok(Box::new(manager))
        }
        PersistenceBackend::PostgreSQL => {
            #[cfg(feature = "persistence")]
            {
                let manager = backends::postgres::PostgresPersistenceManager::new(config).await?;
                Ok(Box::new(manager))
            }
            #[cfg(not(feature = "persistence"))]
            {
                Err(PersistenceError::ConfigError {
                    message: "PostgreSQL backend requires 'persistence' feature".to_string(),
                })
            }
        }
    }
}

/// Validation for user data
pub fn validate_user_data(user_id: &str) -> PersistenceResult<()> {
    if user_id.is_empty() {
        return Err(PersistenceError::IntegrityError {
            message: "User ID cannot be empty".to_string(),
        });
    }

    if user_id.len() > 255 {
        return Err(PersistenceError::IntegrityError {
            message: "User ID too long (max 255 characters)".to_string(),
        });
    }

    // Check for invalid characters
    if !user_id
        .chars()
        .all(|c| c.is_alphanumeric() || c == '_' || c == '-')
    {
        return Err(PersistenceError::IntegrityError {
            message: "User ID contains invalid characters".to_string(),
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_user_data() {
        assert!(validate_user_data("user123").is_ok());
        assert!(validate_user_data("user_123").is_ok());
        assert!(validate_user_data("user-123").is_ok());

        assert!(validate_user_data("").is_err());
        assert!(validate_user_data("user@123").is_err());
        assert!(validate_user_data(&"a".repeat(256)).is_err());
    }

    #[test]
    fn test_persistence_config_default() {
        let config = PersistenceConfig::default();
        assert_eq!(config.backend, PersistenceBackend::SQLite);
        assert!(config.enable_encryption);
        assert!(config.enable_compression);
    }
}

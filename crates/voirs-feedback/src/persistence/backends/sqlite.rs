//! SQLite persistence backend

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::{sqlite::SqlitePool, Row, Sqlite, SqliteConnection};
use std::collections::HashMap;
use uuid::Uuid;

use crate::persistence::{
    CleanupResult, PersistenceConfig, PersistenceError, PersistenceManager, PersistenceResult,
    StorageStats, UserDataExport,
};
use crate::traits::{FeedbackResponse, SessionState, UserPreferences, UserProgress};

/// SQLite persistence manager
pub struct SQLitePersistenceManager {
    pool: SqlitePool,
    config: PersistenceConfig,
}

impl SQLitePersistenceManager {
    /// Create a new SQLite persistence manager
    pub async fn new(config: PersistenceConfig) -> PersistenceResult<Self> {
        let database_url = format!(
            "sqlite:{connection_string}",
            connection_string = config.connection_string
        );

        let pool = SqlitePool::connect(&database_url).await.map_err(|e| {
            PersistenceError::ConnectionError {
                message: format!("Failed to connect to SQLite database: {e}"),
            }
        })?;

        Ok(Self { pool, config })
    }

    /// Create database schema
    async fn create_schema(&self) -> PersistenceResult<()> {
        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| PersistenceError::ConnectionError {
                message: format!("Failed to start transaction: {e}"),
            })?;

        // Sessions table
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                start_time TEXT NOT NULL,
                last_activity TEXT NOT NULL,
                session_data TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            "#,
        )
        .execute(&mut *tx)
        .await
        .map_err(|e| PersistenceError::ConnectionError {
            message: format!("Failed to create sessions table: {e}"),
        })?;

        // User progress table
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS user_progress (
                user_id TEXT PRIMARY KEY,
                progress_data TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            "#,
        )
        .execute(&mut *tx)
        .await
        .map_err(|e| PersistenceError::ConnectionError {
            message: format!("Failed to create user_progress table: {e}"),
        })?;

        // User preferences table
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id TEXT PRIMARY KEY,
                preferences_data TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            "#,
        )
        .execute(&mut *tx)
        .await
        .map_err(|e| PersistenceError::ConnectionError {
            message: format!("Failed to create user_preferences table: {e}"),
        })?;

        // Feedback history table
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS feedback_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                feedback_data TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            "#,
        )
        .execute(&mut *tx)
        .await
        .map_err(|e| PersistenceError::ConnectionError {
            message: format!("Failed to create feedback_history table: {e}"),
        })?;

        // Metadata table for key-value storage
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            "#,
        )
        .execute(&mut *tx)
        .await
        .map_err(|e| PersistenceError::ConnectionError {
            message: format!("Failed to create metadata table: {e}"),
        })?;

        // Create indexes for better performance
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id)")
            .execute(&mut *tx)
            .await
            .map_err(|e| PersistenceError::ConnectionError {
                message: format!("Failed to create sessions index: {e}"),
            })?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_feedback_user_id ON feedback_history(user_id)")
            .execute(&mut *tx)
            .await
            .map_err(|e| PersistenceError::ConnectionError {
                message: format!("Failed to create feedback index: {e}"),
            })?;

        tx.commit()
            .await
            .map_err(|e| PersistenceError::ConnectionError {
                message: format!("Failed to commit schema transaction: {e}"),
            })?;

        Ok(())
    }

    /// Serialize data to JSON string
    fn serialize<T: Serialize>(&self, data: &T) -> PersistenceResult<String> {
        serde_json::to_string(data).map_err(|e| PersistenceError::SerializationError {
            message: format!("Failed to serialize data: {e}"),
        })
    }

    /// Deserialize data from JSON string
    fn deserialize<T: for<'de> Deserialize<'de>>(&self, data: &str) -> PersistenceResult<T> {
        serde_json::from_str(data).map_err(|e| PersistenceError::SerializationError {
            message: format!("Failed to deserialize data: {e}"),
        })
    }
}

#[async_trait]
impl PersistenceManager for SQLitePersistenceManager {
    async fn initialize(&mut self) -> PersistenceResult<()> {
        // Create schema if it doesn't exist
        self.create_schema().await?;

        log::info!("SQLite persistence backend initialized");
        Ok(())
    }

    async fn save_session(&self, session: &SessionState) -> PersistenceResult<()> {
        let session_data = self.serialize(session)?;
        let now = Utc::now().to_rfc3339();

        sqlx::query(
            r#"
            INSERT OR REPLACE INTO sessions 
            (session_id, user_id, start_time, last_activity, session_data, updated_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6)
            "#,
        )
        .bind(session.session_id.to_string())
        .bind(&session.user_id)
        .bind(session.start_time.to_rfc3339())
        .bind(session.last_activity.to_rfc3339())
        .bind(session_data)
        .bind(now)
        .execute(&self.pool)
        .await
        .map_err(|e| PersistenceError::ConnectionError {
            message: format!("Failed to save session: {e}"),
        })?;

        log::debug!("Saved session: {}", session.session_id);
        Ok(())
    }

    async fn load_session(&self, session_id: &Uuid) -> PersistenceResult<SessionState> {
        let row = sqlx::query("SELECT session_data FROM sessions WHERE session_id = ?1")
            .bind(session_id.to_string())
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| PersistenceError::ConnectionError {
                message: format!("Failed to load session: {e}"),
            })?;

        match row {
            Some(row) => {
                let session_data: String = row.get("session_data");
                self.deserialize(&session_data)
            }
            None => Err(PersistenceError::NotFound {
                entity_type: String::from("session"),
                id: session_id.to_string(),
            }),
        }
    }

    async fn save_user_progress(
        &self,
        user_id: &str,
        progress: &UserProgress,
    ) -> PersistenceResult<()> {
        let progress_data = self.serialize(progress)?;
        let now = Utc::now().to_rfc3339();

        sqlx::query(
            r#"
            INSERT OR REPLACE INTO user_progress 
            (user_id, progress_data, updated_at)
            VALUES (?1, ?2, ?3)
            "#,
        )
        .bind(user_id)
        .bind(progress_data)
        .bind(now)
        .execute(&self.pool)
        .await
        .map_err(|e| PersistenceError::ConnectionError {
            message: format!("Failed to save user progress: {e}"),
        })?;

        log::debug!("Saved progress for user: {}", user_id);
        Ok(())
    }

    async fn load_user_progress(&self, user_id: &str) -> PersistenceResult<UserProgress> {
        let row = sqlx::query("SELECT progress_data FROM user_progress WHERE user_id = ?1")
            .bind(user_id)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| PersistenceError::ConnectionError {
                message: format!("Failed to load user progress: {e}"),
            })?;

        match row {
            Some(row) => {
                let progress_data: String = row.get("progress_data");
                self.deserialize(&progress_data)
            }
            None => Err(PersistenceError::NotFound {
                entity_type: String::from("user_progress"),
                id: user_id.to_string(),
            }),
        }
    }

    async fn save_feedback(
        &self,
        user_id: &str,
        feedback: &FeedbackResponse,
    ) -> PersistenceResult<()> {
        let feedback_data = self.serialize(feedback)?;

        sqlx::query(
            r#"
            INSERT INTO feedback_history (user_id, feedback_data)
            VALUES (?1, ?2)
            "#,
        )
        .bind(user_id)
        .bind(feedback_data)
        .execute(&self.pool)
        .await
        .map_err(|e| PersistenceError::ConnectionError {
            message: format!("Failed to save feedback: {e}"),
        })?;

        log::debug!("Saved feedback for user: {}", user_id);
        Ok(())
    }

    async fn load_feedback_history(
        &self,
        user_id: &str,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> PersistenceResult<Vec<FeedbackResponse>> {
        let mut query = String::from(
            "SELECT feedback_data FROM feedback_history WHERE user_id = ?1 ORDER BY created_at DESC",
        );

        if let Some(limit) = limit {
            query.push_str(&format!(" LIMIT {limit}"));
        }

        if let Some(offset) = offset {
            query.push_str(&format!(" OFFSET {offset}"));
        }

        let rows = sqlx::query(&query)
            .bind(user_id)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| PersistenceError::ConnectionError {
                message: format!("Failed to load feedback history: {}", e),
            })?;

        let mut feedback_history = Vec::new();
        for row in rows {
            let feedback_data: String = row.get("feedback_data");
            let feedback: FeedbackResponse = self.deserialize(&feedback_data)?;
            feedback_history.push(feedback);
        }

        Ok(feedback_history)
    }

    async fn save_preferences(
        &self,
        user_id: &str,
        preferences: &UserPreferences,
    ) -> PersistenceResult<()> {
        let preferences_data = self.serialize(preferences)?;
        let now = Utc::now().to_rfc3339();

        sqlx::query(
            r#"
            INSERT OR REPLACE INTO user_preferences 
            (user_id, preferences_data, updated_at)
            VALUES (?1, ?2, ?3)
            "#,
        )
        .bind(user_id)
        .bind(preferences_data)
        .bind(now)
        .execute(&self.pool)
        .await
        .map_err(|e| PersistenceError::ConnectionError {
            message: format!("Failed to save user preferences: {}", e),
        })?;

        log::debug!("Saved preferences for user: {}", user_id);
        Ok(())
    }

    async fn load_preferences(&self, user_id: &str) -> PersistenceResult<UserPreferences> {
        let row = sqlx::query("SELECT preferences_data FROM user_preferences WHERE user_id = ?1")
            .bind(user_id)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| PersistenceError::ConnectionError {
                message: format!("Failed to load user preferences: {}", e),
            })?;

        match row {
            Some(row) => {
                let preferences_data: String = row.get("preferences_data");
                self.deserialize(&preferences_data)
            }
            None => Err(PersistenceError::NotFound {
                entity_type: String::from("user_preferences"),
                id: user_id.to_string(),
            }),
        }
    }

    async fn delete_user_data(&self, user_id: &str) -> PersistenceResult<()> {
        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| PersistenceError::ConnectionError {
                message: format!("Failed to start transaction: {e}"),
            })?;

        // Delete from all tables
        sqlx::query("DELETE FROM sessions WHERE user_id = ?1")
            .bind(user_id)
            .execute(&mut *tx)
            .await
            .map_err(|e| PersistenceError::ConnectionError {
                message: format!("Failed to delete user sessions: {}", e),
            })?;

        sqlx::query("DELETE FROM user_progress WHERE user_id = ?1")
            .bind(user_id)
            .execute(&mut *tx)
            .await
            .map_err(|e| PersistenceError::ConnectionError {
                message: format!("Failed to delete user progress: {}", e),
            })?;

        sqlx::query("DELETE FROM user_preferences WHERE user_id = ?1")
            .bind(user_id)
            .execute(&mut *tx)
            .await
            .map_err(|e| PersistenceError::ConnectionError {
                message: format!("Failed to delete user preferences: {}", e),
            })?;

        sqlx::query("DELETE FROM feedback_history WHERE user_id = ?1")
            .bind(user_id)
            .execute(&mut *tx)
            .await
            .map_err(|e| PersistenceError::ConnectionError {
                message: format!("Failed to delete user feedback: {}", e),
            })?;

        tx.commit()
            .await
            .map_err(|e| PersistenceError::ConnectionError {
                message: format!("Failed to commit delete transaction: {e}"),
            })?;

        log::info!("Deleted all data for user: {}", user_id);
        Ok(())
    }

    async fn export_user_data(&self, user_id: &str) -> PersistenceResult<UserDataExport> {
        // Load all user data
        let preferences = self.load_preferences(user_id).await.unwrap_or_default();
        let progress = self.load_user_progress(user_id).await.unwrap_or_default();
        let feedback_history = self
            .load_feedback_history(user_id, None, None)
            .await
            .unwrap_or_default();

        // Load sessions
        let session_rows = sqlx::query("SELECT session_data FROM sessions WHERE user_id = ?1")
            .bind(user_id)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| PersistenceError::ConnectionError {
                message: format!("Failed to load user sessions: {e}"),
            })?;

        let mut sessions = Vec::new();
        for row in session_rows {
            let session_data: String = row.get("session_data");
            let session: SessionState = self.deserialize(&session_data)?;
            sessions.push(session);
        }

        Ok(UserDataExport {
            user_id: user_id.to_string(),
            export_timestamp: Utc::now(),
            preferences,
            progress,
            feedback_history,
            sessions,
            metadata: HashMap::new(),
        })
    }

    async fn get_storage_stats(&self) -> PersistenceResult<StorageStats> {
        let users_count: i64 = sqlx::query_scalar(
            "SELECT COUNT(DISTINCT user_id) FROM (
                SELECT user_id FROM sessions
                UNION
                SELECT user_id FROM user_progress
                UNION
                SELECT user_id FROM user_preferences
                UNION
                SELECT user_id FROM feedback_history
            )",
        )
        .fetch_one(&self.pool)
        .await
        .map_err(|e| PersistenceError::ConnectionError {
            message: format!("Failed to count users: {e}"),
        })?;

        let sessions_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM sessions")
            .fetch_one(&self.pool)
            .await
            .map_err(|e| PersistenceError::ConnectionError {
                message: format!("Failed to count sessions: {e}"),
            })?;

        let feedback_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM feedback_history")
            .fetch_one(&self.pool)
            .await
            .map_err(|e| PersistenceError::ConnectionError {
                message: format!("Failed to count feedback records: {e}"),
            })?;

        // Get database file size (approximate)
        let storage_size = std::fs::metadata(&self.config.connection_string)
            .map(|m| m.len())
            .unwrap_or(0);

        // Get last cleanup timestamp from metadata table
        let last_cleanup_str: Option<String> =
            sqlx::query_scalar("SELECT value FROM metadata WHERE key = 'last_cleanup'")
                .fetch_optional(&self.pool)
                .await
                .map_err(|e| PersistenceError::ConnectionError {
                    message: format!("Failed to get last cleanup timestamp: {e}"),
                })?;

        let last_cleanup = if let Some(cleanup_str) = last_cleanup_str {
            if let Ok(cleanup_data) = serde_json::from_str::<serde_json::Value>(&cleanup_str) {
                cleanup_data
                    .get("timestamp")
                    .and_then(|t| t.as_str())
                    .and_then(|t| DateTime::parse_from_rfc3339(t).ok())
                    .map(|t| t.with_timezone(&Utc))
            } else {
                None
            }
        } else {
            None
        };

        Ok(StorageStats {
            total_users: users_count as usize,
            total_sessions: sessions_count as usize,
            total_feedback_records: feedback_count as usize,
            storage_size_bytes: storage_size,
            last_cleanup,
            db_version: String::from("1.0.0"),
        })
    }

    async fn cleanup(&self, older_than: DateTime<Utc>) -> PersistenceResult<CleanupResult> {
        let start_time = std::time::Instant::now();
        let older_than_str = older_than.to_rfc3339();

        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| PersistenceError::ConnectionError {
                message: format!("Failed to start cleanup transaction: {e}"),
            })?;

        // Count and delete old sessions
        let sessions_count: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM sessions WHERE start_time < ?1")
                .bind(&older_than_str)
                .fetch_one(&mut *tx)
                .await
                .map_err(|e| PersistenceError::ConnectionError {
                    message: format!("Failed to count old sessions: {e}"),
                })?;

        sqlx::query("DELETE FROM sessions WHERE start_time < ?1")
            .bind(&older_than_str)
            .execute(&mut *tx)
            .await
            .map_err(|e| PersistenceError::ConnectionError {
                message: format!("Failed to delete old sessions: {e}"),
            })?;

        // Count and delete old feedback records
        let feedback_count: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM feedback_history WHERE created_at < ?1")
                .bind(&older_than_str)
                .fetch_one(&mut *tx)
                .await
                .map_err(|e| PersistenceError::ConnectionError {
                    message: format!("Failed to count old feedback: {e}"),
                })?;

        sqlx::query("DELETE FROM feedback_history WHERE created_at < ?1")
            .bind(&older_than_str)
            .execute(&mut *tx)
            .await
            .map_err(|e| PersistenceError::ConnectionError {
                message: format!("Failed to delete old feedback: {e}"),
            })?;

        tx.commit()
            .await
            .map_err(|e| PersistenceError::ConnectionError {
                message: format!("Failed to commit cleanup transaction: {e}"),
            })?;

        // Vacuum database to reclaim space (must be done outside transaction)
        sqlx::query("VACUUM")
            .execute(&self.pool)
            .await
            .map_err(|e| PersistenceError::ConnectionError {
                message: format!("Failed to vacuum database: {e}"),
            })?;

        // Store cleanup timestamp in metadata table
        let cleanup_timestamp = Utc::now();
        let metadata_value = serde_json::json!({
            "timestamp": cleanup_timestamp,
            "sessions_cleaned": sessions_count,
            "feedback_records_cleaned": feedback_count
        });

        let metadata_str = serde_json::to_string(&metadata_value).map_err(|e| {
            PersistenceError::SerializationError {
                message: format!("Failed to serialize cleanup metadata: {e}"),
            }
        })?;

        let now = Utc::now().to_rfc3339();
        sqlx::query(
            r#"
            INSERT OR REPLACE INTO metadata (key, value, updated_at)
            VALUES (?1, ?2, ?3)
            "#,
        )
        .bind("last_cleanup")
        .bind(metadata_str)
        .bind(now)
        .execute(&self.pool)
        .await
        .map_err(|e| PersistenceError::ConnectionError {
            message: format!("Failed to store cleanup metadata: {e}"),
        })?;

        let cleanup_duration = start_time.elapsed();

        log::info!(
            "Cleanup completed: {} sessions, {} feedback records removed in {:?}",
            sessions_count,
            feedback_count,
            cleanup_duration
        );

        Ok(CleanupResult {
            sessions_cleaned: sessions_count as usize,
            feedback_records_cleaned: feedback_count as usize,
            bytes_reclaimed: 0, // Hard to calculate without filesystem support
            cleanup_duration,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use uuid::Uuid;

    #[tokio::test]
    async fn test_sqlite_persistence_manager() {
        let config = PersistenceConfig {
            backend: crate::persistence::PersistenceBackend::SQLite,
            connection_string: String::from(":memory:"),
            ..Default::default()
        };

        let mut manager = SQLitePersistenceManager::new(config).await.unwrap();
        manager.initialize().await.unwrap();

        // Test session save/load
        let session_id = Uuid::new_v4();
        let session = SessionState {
            session_id,
            user_id: String::from("test_user"),
            start_time: Utc::now(),
            last_activity: Utc::now(),
            ..Default::default()
        };

        manager.save_session(&session).await.unwrap();
        let loaded_session = manager.load_session(&session_id).await.unwrap();
        assert_eq!(loaded_session.session_id, session_id);
    }

    #[tokio::test]
    async fn test_cleanup() {
        let config = PersistenceConfig {
            backend: crate::persistence::PersistenceBackend::SQLite,
            connection_string: String::from(":memory:"),
            ..Default::default()
        };

        let mut manager = SQLitePersistenceManager::new(config).await.unwrap();
        manager.initialize().await.unwrap();

        // Add some old data
        let old_time = Utc::now() - chrono::Duration::days(2);
        let session = SessionState {
            session_id: Uuid::new_v4(),
            user_id: String::from("test_user"),
            start_time: old_time,
            last_activity: old_time,
            ..Default::default()
        };

        manager.save_session(&session).await.unwrap();

        // Cleanup old data
        let cutoff = Utc::now() - chrono::Duration::days(1);
        let result = manager.cleanup(cutoff).await.unwrap();

        assert_eq!(result.sessions_cleaned, 1);
    }
}

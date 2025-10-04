//! PostgreSQL persistence backend

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::{postgres::PgPool, Row};
use std::collections::HashMap;
use uuid::Uuid;

use crate::persistence::{
    query_optimizer::{CompleteUserData, QueryOptimizer, QueryOptimizerConfig},
    CleanupResult, PersistenceConfig, PersistenceError, PersistenceManager, PersistenceResult,
    StorageStats, UserDataExport,
};
use crate::traits::{FeedbackResponse, SessionState, UserPreferences, UserProgress};

/// PostgreSQL persistence manager with query optimization
pub struct PostgresPersistenceManager {
    pool: PgPool,
    config: PersistenceConfig,
    query_optimizer: QueryOptimizer,
}

impl PostgresPersistenceManager {
    /// Create a new PostgreSQL persistence manager with query optimization
    pub async fn new(config: PersistenceConfig) -> PersistenceResult<Self> {
        let pool = PgPool::connect(&config.connection_string)
            .await
            .map_err(|e| PersistenceError::ConnectionError {
                message: format!("Failed to connect to PostgreSQL database: {e}"),
            })?;

        // Initialize query optimizer with default configuration
        let optimizer_config = QueryOptimizerConfig::default();
        let query_optimizer = QueryOptimizer::new(optimizer_config);

        Ok(Self {
            pool,
            config,
            query_optimizer,
        })
    }

    /// Load feedback history using optimized queries
    pub async fn load_feedback_history_optimized(
        &self,
        user_id: &str,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> PersistenceResult<Vec<FeedbackResponse>> {
        self.query_optimizer
            .load_feedback_history_optimized(&self.pool, user_id, limit, offset)
            .await
    }

    /// Batch save multiple feedback items for better performance
    pub async fn batch_save_feedback_optimized(
        &self,
        feedback_items: Vec<(String, FeedbackResponse)>,
    ) -> PersistenceResult<usize> {
        self.query_optimizer
            .batch_save_feedback(&self.pool, feedback_items)
            .await
    }

    /// Load complete user data with single optimized query
    pub async fn load_user_data_complete(
        &self,
        user_id: &str,
    ) -> PersistenceResult<CompleteUserData> {
        self.query_optimizer
            .load_user_data_complete(&self.pool, user_id)
            .await
    }

    /// Get query optimization statistics
    pub async fn get_query_stats(&self) -> crate::persistence::query_optimizer::QueryStats {
        self.query_optimizer.get_stats().await
    }

    /// Clear query cache
    pub async fn clear_query_cache(&self) {
        self.query_optimizer.clear_cache().await
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
                session_id UUID PRIMARY KEY,
                user_id TEXT NOT NULL,
                start_time TIMESTAMPTZ NOT NULL,
                last_activity TIMESTAMPTZ NOT NULL,
                session_data JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
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
                progress_data JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
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
                preferences_data JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
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
                id SERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                feedback_data JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
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
                value JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            "#,
        )
        .execute(&mut *tx)
        .await
        .map_err(|e| PersistenceError::ConnectionError {
            message: format!("Failed to create metadata table: {e}"),
        })?;

        // Create indexes for better performance
        sqlx::query(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sessions_user_id ON sessions(user_id)",
        )
        .execute(&mut *tx)
        .await
        .map_err(|e| PersistenceError::ConnectionError {
            message: format!("Failed to create sessions index: {e}"),
        })?;

        sqlx::query("CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_feedback_user_id ON feedback_history(user_id)")
            .execute(&mut *tx)
            .await
            .map_err(|e| PersistenceError::ConnectionError {
                message: format!("Failed to create feedback index: {e}"),
            })?;

        sqlx::query("CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_feedback_created_at ON feedback_history(created_at)")
            .execute(&mut *tx)
            .await
            .map_err(|e| PersistenceError::ConnectionError {
                message: format!("Failed to create feedback timestamp index: {e}"),
            })?;

        // GIN indexes for JSONB columns for better query performance
        sqlx::query("CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sessions_data_gin ON sessions USING gin(session_data)")
            .execute(&mut *tx)
            .await
            .map_err(|e| PersistenceError::ConnectionError {
                message: format!("Failed to create sessions GIN index: {e}"),
            })?;

        tx.commit()
            .await
            .map_err(|e| PersistenceError::ConnectionError {
                message: format!("Failed to commit schema transaction: {e}"),
            })?;

        Ok(())
    }

    /// Serialize data to JSON value
    fn serialize<T: Serialize>(&self, data: &T) -> PersistenceResult<serde_json::Value> {
        serde_json::to_value(data).map_err(|e| PersistenceError::SerializationError {
            message: format!("Failed to serialize data: {e}"),
        })
    }

    /// Deserialize data from JSON value
    fn deserialize<T: for<'de> Deserialize<'de>>(
        &self,
        data: &serde_json::Value,
    ) -> PersistenceResult<T> {
        serde_json::from_value(data.clone()).map_err(|e| PersistenceError::SerializationError {
            message: format!("Failed to deserialize data: {e}"),
        })
    }
}

#[async_trait]
impl PersistenceManager for PostgresPersistenceManager {
    async fn initialize(&mut self) -> PersistenceResult<()> {
        // Create schema if it doesn't exist
        self.create_schema().await?;

        log::info!("PostgreSQL persistence backend initialized");
        Ok(())
    }

    async fn save_session(&self, session: &SessionState) -> PersistenceResult<()> {
        let session_data = self.serialize(session)?;

        sqlx::query(
            r#"
            INSERT INTO sessions 
            (session_id, user_id, start_time, last_activity, session_data, updated_at)
            VALUES ($1, $2, $3, $4, $5, NOW())
            ON CONFLICT (session_id) 
            DO UPDATE SET 
                user_id = EXCLUDED.user_id,
                start_time = EXCLUDED.start_time,
                last_activity = EXCLUDED.last_activity,
                session_data = EXCLUDED.session_data,
                updated_at = NOW()
            "#,
        )
        .bind(session.session_id)
        .bind(&session.user_id)
        .bind(session.start_time)
        .bind(session.last_activity)
        .bind(session_data)
        .execute(&self.pool)
        .await
        .map_err(|e| PersistenceError::ConnectionError {
            message: format!("Failed to save session: {e}"),
        })?;

        log::debug!("Saved session: {}", session.session_id);
        Ok(())
    }

    async fn load_session(&self, session_id: &Uuid) -> PersistenceResult<SessionState> {
        let row = sqlx::query("SELECT session_data FROM sessions WHERE session_id = $1")
            .bind(session_id)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| PersistenceError::ConnectionError {
                message: format!("Failed to load session: {e}"),
            })?;

        match row {
            Some(row) => {
                let session_data: serde_json::Value = row.get("session_data");
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

        sqlx::query(
            r#"
            INSERT INTO user_progress 
            (user_id, progress_data, updated_at)
            VALUES ($1, $2, NOW())
            ON CONFLICT (user_id) 
            DO UPDATE SET 
                progress_data = EXCLUDED.progress_data,
                updated_at = NOW()
            "#,
        )
        .bind(user_id)
        .bind(progress_data)
        .execute(&self.pool)
        .await
        .map_err(|e| PersistenceError::ConnectionError {
            message: format!("Failed to save user progress: {e}"),
        })?;

        log::debug!("Saved progress for user: {}", user_id);
        Ok(())
    }

    async fn load_user_progress(&self, user_id: &str) -> PersistenceResult<UserProgress> {
        let row = sqlx::query("SELECT progress_data FROM user_progress WHERE user_id = $1")
            .bind(user_id)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| PersistenceError::ConnectionError {
                message: format!("Failed to load user progress: {e}"),
            })?;

        match row {
            Some(row) => {
                let progress_data: serde_json::Value = row.get("progress_data");
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
            VALUES ($1, $2)
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
            "SELECT feedback_data FROM feedback_history WHERE user_id = $1 ORDER BY created_at DESC",
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
                message: format!("Failed to load feedback history: {e}"),
            })?;

        let mut feedback_history = Vec::new();
        for row in rows {
            let feedback_data: serde_json::Value = row.get("feedback_data");
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

        sqlx::query(
            r#"
            INSERT INTO user_preferences 
            (user_id, preferences_data, updated_at)
            VALUES ($1, $2, NOW())
            ON CONFLICT (user_id) 
            DO UPDATE SET 
                preferences_data = EXCLUDED.preferences_data,
                updated_at = NOW()
            "#,
        )
        .bind(user_id)
        .bind(preferences_data)
        .execute(&self.pool)
        .await
        .map_err(|e| PersistenceError::ConnectionError {
            message: format!("Failed to save user preferences: {e}"),
        })?;

        log::debug!("Saved preferences for user: {}", user_id);
        Ok(())
    }

    async fn load_preferences(&self, user_id: &str) -> PersistenceResult<UserPreferences> {
        let row = sqlx::query("SELECT preferences_data FROM user_preferences WHERE user_id = $1")
            .bind(user_id)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| PersistenceError::ConnectionError {
                message: format!("Failed to load user preferences: {e}"),
            })?;

        match row {
            Some(row) => {
                let preferences_data: serde_json::Value = row.get("preferences_data");
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
        sqlx::query("DELETE FROM sessions WHERE user_id = $1")
            .bind(user_id)
            .execute(&mut *tx)
            .await
            .map_err(|e| PersistenceError::ConnectionError {
                message: format!("Failed to delete user sessions: {e}"),
            })?;

        sqlx::query("DELETE FROM user_progress WHERE user_id = $1")
            .bind(user_id)
            .execute(&mut *tx)
            .await
            .map_err(|e| PersistenceError::ConnectionError {
                message: format!("Failed to delete user progress: {e}"),
            })?;

        sqlx::query("DELETE FROM user_preferences WHERE user_id = $1")
            .bind(user_id)
            .execute(&mut *tx)
            .await
            .map_err(|e| PersistenceError::ConnectionError {
                message: format!("Failed to delete user preferences: {e}"),
            })?;

        sqlx::query("DELETE FROM feedback_history WHERE user_id = $1")
            .bind(user_id)
            .execute(&mut *tx)
            .await
            .map_err(|e| PersistenceError::ConnectionError {
                message: format!("Failed to delete user feedback: {e}"),
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
        let session_rows = sqlx::query("SELECT session_data FROM sessions WHERE user_id = $1")
            .bind(user_id)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| PersistenceError::ConnectionError {
                message: format!("Failed to load user sessions: {e}"),
            })?;

        let mut sessions = Vec::new();
        for row in session_rows {
            let session_data: serde_json::Value = row.get("session_data");
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
            ) AS all_users",
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

        // Get database size using PostgreSQL system functions
        let storage_size: Option<i64> =
            sqlx::query_scalar("SELECT pg_database_size(current_database())")
                .fetch_optional(&self.pool)
                .await
                .map_err(|e| PersistenceError::ConnectionError {
                    message: format!("Failed to get database size: {e}"),
                })?;

        // Get last cleanup timestamp from metadata table
        let last_cleanup: Option<DateTime<Utc>> = sqlx::query_scalar(
            "SELECT (value->>'timestamp')::timestamptz FROM metadata WHERE key = 'last_cleanup'",
        )
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| PersistenceError::ConnectionError {
            message: format!("Failed to get last cleanup timestamp: {e}"),
        })?;

        Ok(StorageStats {
            total_users: users_count as usize,
            total_sessions: sessions_count as usize,
            total_feedback_records: feedback_count as usize,
            storage_size_bytes: storage_size.unwrap_or(0) as u64,
            last_cleanup,
            db_version: String::from("1.0.0"),
        })
    }

    async fn cleanup(&self, older_than: DateTime<Utc>) -> PersistenceResult<CleanupResult> {
        let start_time = std::time::Instant::now();

        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| PersistenceError::ConnectionError {
                message: format!("Failed to start cleanup transaction: {e}"),
            })?;

        // Count and delete old sessions
        let sessions_count: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM sessions WHERE created_at < $1")
                .bind(older_than)
                .fetch_one(&mut *tx)
                .await
                .map_err(|e| PersistenceError::ConnectionError {
                    message: format!("Failed to count old sessions: {e}"),
                })?;

        sqlx::query("DELETE FROM sessions WHERE created_at < $1")
            .bind(older_than)
            .execute(&mut *tx)
            .await
            .map_err(|e| PersistenceError::ConnectionError {
                message: format!("Failed to delete old sessions: {e}"),
            })?;

        // Count and delete old feedback records
        let feedback_count: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM feedback_history WHERE created_at < $1")
                .bind(older_than)
                .fetch_one(&mut *tx)
                .await
                .map_err(|e| PersistenceError::ConnectionError {
                    message: format!("Failed to count old feedback: {e}"),
                })?;

        sqlx::query("DELETE FROM feedback_history WHERE created_at < $1")
            .bind(older_than)
            .execute(&mut *tx)
            .await
            .map_err(|e| PersistenceError::ConnectionError {
                message: format!("Failed to delete old feedback: {e}"),
            })?;

        // Vacuum database to reclaim space
        sqlx::query("VACUUM ANALYZE")
            .execute(&mut *tx)
            .await
            .map_err(|e| PersistenceError::ConnectionError {
                message: format!("Failed to vacuum database: {e}"),
            })?;

        tx.commit()
            .await
            .map_err(|e| PersistenceError::ConnectionError {
                message: format!("Failed to commit cleanup transaction: {e}"),
            })?;

        // Store cleanup timestamp in metadata table
        let cleanup_timestamp = Utc::now();
        let metadata_value = serde_json::json!({
            "timestamp": cleanup_timestamp,
            "sessions_cleaned": sessions_count,
            "feedback_records_cleaned": feedback_count
        });

        sqlx::query(
            r#"
            INSERT INTO metadata (key, value, updated_at)
            VALUES ('last_cleanup', $1, NOW())
            ON CONFLICT (key) 
            DO UPDATE SET 
                value = EXCLUDED.value,
                updated_at = NOW()
            "#,
        )
        .bind(metadata_value)
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
            bytes_reclaimed: 0, // PostgreSQL doesn't easily provide this information
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
    #[ignore] // Requires PostgreSQL instance
    async fn test_postgres_persistence_manager() {
        let config = PersistenceConfig {
            backend: crate::persistence::PersistenceBackend::PostgreSQL,
            connection_string: String::from("postgresql://postgres:password@localhost/test_db"),
            ..Default::default()
        };

        let mut manager = PostgresPersistenceManager::new(config).await.unwrap();
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
    #[ignore] // Requires PostgreSQL instance
    async fn test_cleanup() {
        let config = PersistenceConfig {
            backend: crate::persistence::PersistenceBackend::PostgreSQL,
            connection_string: String::from("postgresql://postgres:password@localhost/test_db"),
            ..Default::default()
        };

        let mut manager = PostgresPersistenceManager::new(config).await.unwrap();
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

/// Query optimization utilities for PostgreSQL
impl PostgresPersistenceManager {
    /// Create additional performance indexes
    pub async fn create_performance_indexes(&self) -> PersistenceResult<()> {
        let indexes = vec![
            // Composite indexes for common query patterns
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sessions_user_start_time ON sessions(user_id, start_time DESC)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_feedback_user_created ON feedback_history(user_id, created_at DESC)",

            // JSONB GIN indexes for fast JSON queries
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sessions_data_gin ON sessions USING GIN (session_data)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_progress_data_gin ON user_progress USING GIN (progress_data)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_feedback_data_gin ON feedback_history USING GIN (feedback_data)",

            // Partial indexes for common filters
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sessions_active ON sessions(user_id) WHERE end_time IS NULL",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_feedback_recent ON feedback_history(user_id, created_at) WHERE created_at > NOW() - INTERVAL '30 days'",

            // Expression indexes for extracted JSON fields
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_progress_skill_level ON user_progress((progress_data->>'overall_skill_level')::numeric)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_feedback_score ON feedback_history((feedback_data->>'score')::numeric)",
        ];

        for index_sql in indexes {
            sqlx::query(index_sql)
                .execute(&self.pool)
                .await
                .map_err(|e| PersistenceError::ConnectionError {
                    message: format!("Failed to create performance index: {e}"),
                })?;
        }

        Ok(())
    }

    /// Optimized bulk user progress loading
    pub async fn load_multiple_user_progress(
        &self,
        user_ids: &[String],
    ) -> PersistenceResult<HashMap<String, UserProgress>> {
        if user_ids.is_empty() {
            return Ok(HashMap::new());
        }

        // Use ANY() for efficient batch loading instead of multiple queries
        let query = "SELECT user_id, progress_data FROM user_progress WHERE user_id = ANY($1)";
        let rows = sqlx::query(query)
            .bind(user_ids)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| PersistenceError::ConnectionError {
                message: format!("Failed to load multiple user progress: {e}"),
            })?;

        let mut result = HashMap::new();
        for row in rows {
            let user_id: String = row.get("user_id");
            let progress_data: serde_json::Value = row.get("progress_data");
            let progress = self.deserialize(&progress_data)?;
            result.insert(user_id, progress);
        }

        Ok(result)
    }

    /// Optimized user feedback history with pagination
    pub async fn load_user_feedback_paginated(
        &self,
        user_id: &str,
        offset: i64,
        limit: i64,
        min_score: Option<f32>,
    ) -> PersistenceResult<(Vec<FeedbackResponse>, i64)> {
        // Build dynamic WHERE clause for filtering
        let mut where_clause = String::from("WHERE user_id = $1");
        let mut param_count = 1;

        if min_score.is_some() {
            param_count += 1;
            where_clause += &format!(
                " AND (feedback_data->>'score')::numeric >= ${}",
                param_count
            );
        }

        // Count query for pagination
        let count_query = format!("SELECT COUNT(*) FROM feedback_history {where_clause}");
        let mut count_query_builder = sqlx::query(&count_query).bind(user_id);
        if let Some(score) = min_score {
            count_query_builder = count_query_builder.bind(score);
        }

        let total_count: i64 = count_query_builder
            .fetch_one(&self.pool)
            .await
            .map_err(|e| PersistenceError::ConnectionError {
                message: format!("Failed to count feedback history: {e}"),
            })?
            .get(0);

        // Data query with pagination and ordering
        let data_query = format!(
            "SELECT feedback_data FROM feedback_history {} ORDER BY created_at DESC LIMIT {} OFFSET {}",
            where_clause, limit, offset
        );
        let mut data_query_builder = sqlx::query(&data_query).bind(user_id);
        if let Some(score) = min_score {
            data_query_builder = data_query_builder.bind(score);
        }

        let rows = data_query_builder
            .fetch_all(&self.pool)
            .await
            .map_err(|e| PersistenceError::ConnectionError {
                message: format!("Failed to load feedback history: {e}"),
            })?;

        let mut feedback_history = Vec::new();
        for row in rows {
            let feedback_data: serde_json::Value = row.get("feedback_data");
            let feedback = self.deserialize(&feedback_data)?;
            feedback_history.push(feedback);
        }

        Ok((feedback_history, total_count))
    }

    /// Optimized analytics query for user skill progression
    pub async fn get_user_skill_analytics(
        &self,
        user_id: &str,
        days_back: i32,
    ) -> PersistenceResult<SkillProgressionAnalytics> {
        let query = r#"
            WITH skill_progression AS (
                SELECT 
                    DATE_TRUNC('day', created_at) as day,
                    AVG((feedback_data->>'score')::numeric) as avg_score,
                    COUNT(*) as session_count,
                    MIN((feedback_data->>'score')::numeric) as min_score,
                    MAX((feedback_data->>'score')::numeric) as max_score
                FROM feedback_history 
                WHERE user_id = $1 
                    AND created_at >= NOW() - INTERVAL '%d days'
                GROUP BY DATE_TRUNC('day', created_at)
                ORDER BY day DESC
            )
            SELECT 
                day,
                avg_score,
                session_count,
                min_score,
                max_score,
                LAG(avg_score) OVER (ORDER BY day) as prev_avg_score
            FROM skill_progression
        "#;

        let formatted_query = query.replace("%d", &days_back.to_string());
        let rows = sqlx::query(&formatted_query)
            .bind(user_id)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| PersistenceError::ConnectionError {
                message: format!("Failed to load skill analytics: {e}"),
            })?;

        let mut daily_stats = Vec::new();
        for row in rows {
            let day: DateTime<Utc> = row.get("day");
            let avg_score: Option<f64> = row.get("avg_score");
            let session_count: i64 = row.get("session_count");
            let min_score: Option<f64> = row.get("min_score");
            let max_score: Option<f64> = row.get("max_score");
            let prev_avg_score: Option<f64> = row.get("prev_avg_score");

            daily_stats.push(DailySkillStats {
                date: day,
                average_score: avg_score.unwrap_or(0.0) as f32,
                session_count: session_count as u32,
                min_score: min_score.unwrap_or(0.0) as f32,
                max_score: max_score.unwrap_or(0.0) as f32,
                improvement: if let (Some(current), Some(previous)) = (avg_score, prev_avg_score) {
                    (current - previous) as f32
                } else {
                    0.0
                },
            });
        }

        let total_sessions = daily_stats.iter().map(|d| d.session_count).sum();
        let overall_improvement =
            if let (Some(first), Some(last)) = (daily_stats.last(), daily_stats.first()) {
                last.average_score - first.average_score
            } else {
                0.0
            };

        Ok(SkillProgressionAnalytics {
            user_id: user_id.to_string(),
            daily_stats,
            analysis_period_days: days_back,
            total_sessions,
            overall_improvement,
        })
    }

    /// Batch update user progress efficiently
    pub async fn batch_update_user_progress(
        &self,
        updates: &[(String, UserProgress)],
    ) -> PersistenceResult<usize> {
        if updates.is_empty() {
            return Ok(0);
        }

        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| PersistenceError::ConnectionError {
                message: format!("Failed to start transaction: {e}"),
            })?;

        // Use UNNEST for efficient batch updates
        let mut user_ids = Vec::new();
        let mut progress_data_list = Vec::new();

        for (user_id, progress) in updates {
            user_ids.push(user_id.clone());
            let serialized = self.serialize(progress)?;
            progress_data_list.push(serialized);
        }

        let query = r#"
            INSERT INTO user_progress (user_id, progress_data, updated_at)
            SELECT * FROM UNNEST($1::text[], $2::jsonb[]) AS t(user_id, progress_data), 
                   (SELECT NOW()) AS updated_at
            ON CONFLICT (user_id)
            DO UPDATE SET 
                progress_data = EXCLUDED.progress_data,
                updated_at = EXCLUDED.updated_at
        "#;

        let affected_rows = sqlx::query(query)
            .bind(&user_ids)
            .bind(&progress_data_list)
            .execute(&mut *tx)
            .await
            .map_err(|e| PersistenceError::ConnectionError {
                message: format!("Failed to batch update user progress: {e}"),
            })?
            .rows_affected();

        tx.commit()
            .await
            .map_err(|e| PersistenceError::ConnectionError {
                message: format!("Failed to commit batch update: {e}"),
            })?;

        Ok(affected_rows as usize)
    }

    /// Optimize database performance with VACUUM and ANALYZE
    pub async fn optimize_database(&self) -> PersistenceResult<DatabaseOptimizationResult> {
        let start_time = std::time::Instant::now();

        // Analyze table statistics for query planner
        let tables = vec![
            "sessions",
            "user_progress",
            "user_preferences",
            "feedback_history",
            "metadata",
        ];

        for table in &tables {
            sqlx::query(&format!("ANALYZE {table}"))
                .execute(&self.pool)
                .await
                .map_err(|e| PersistenceError::ConnectionError {
                    message: format!("Failed to analyze table {table}: {e}"),
                })?;
        }

        // Get table sizes for monitoring
        let size_query = r#"
            SELECT 
                schemaname,
                tablename,
                pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
                pg_total_relation_size(schemaname||'.'||tablename) as size_bytes
            FROM pg_tables 
            WHERE schemaname = 'public'
            ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
        "#;

        let rows = sqlx::query(size_query)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| PersistenceError::ConnectionError {
                message: format!("Failed to get table sizes: {e}"),
            })?;

        let mut table_stats = Vec::new();
        for row in rows {
            let table_name: String = row.get("tablename");
            let size: String = row.get("size");
            let size_bytes: i64 = row.get("size_bytes");

            table_stats.push(TableOptimizationStats {
                table_name,
                size_pretty: size,
                size_bytes: size_bytes as u64,
            });
        }

        let optimization_duration = start_time.elapsed();
        let recommendations = self.generate_optimization_recommendations(&table_stats);

        Ok(DatabaseOptimizationResult {
            optimization_duration,
            table_stats,
            tables_analyzed: tables.len(),
            recommendations,
        })
    }

    /// Generate optimization recommendations based on database analysis
    fn generate_optimization_recommendations(
        &self,
        table_stats: &[TableOptimizationStats],
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Check for large tables that might need partitioning
        for stat in table_stats {
            if stat.size_bytes > 1_000_000_000 {
                // 1GB
                recommendations.push(format!(
                    "Consider partitioning table '{}' (current size: {})",
                    stat.table_name, stat.size_pretty
                ));
            }
        }

        // General recommendations
        if table_stats
            .iter()
            .any(|s| s.table_name == "feedback_history" && s.size_bytes > 100_000_000)
        {
            recommendations.push(String::from(
                "Consider archiving old feedback_history records",
            ));
        }

        if table_stats
            .iter()
            .any(|s| s.table_name == "sessions" && s.size_bytes > 50_000_000)
        {
            recommendations.push(String::from(
                "Consider implementing session cleanup for old records",
            ));
        }

        recommendations
    }
}

/// Analytics result for skill progression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillProgressionAnalytics {
    /// Description
    pub user_id: String,
    /// Description
    pub daily_stats: Vec<DailySkillStats>,
    /// Description
    pub analysis_period_days: i32,
    /// Description
    pub total_sessions: u32,
    /// Description
    pub overall_improvement: f32,
}

/// Daily skill statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DailySkillStats {
    /// Description
    pub date: DateTime<Utc>,
    /// Description
    pub average_score: f32,
    /// Description
    pub session_count: u32,
    /// Description
    pub min_score: f32,
    /// Description
    pub max_score: f32,
    /// Description
    pub improvement: f32,
}

/// Database optimization result
#[derive(Debug)]
pub struct DatabaseOptimizationResult {
    /// Description
    pub optimization_duration: std::time::Duration,
    /// Description
    pub table_stats: Vec<TableOptimizationStats>,
    /// Description
    pub tables_analyzed: usize,
    /// Description
    pub recommendations: Vec<String>,
}

/// Table optimization statistics
#[derive(Debug)]
pub struct TableOptimizationStats {
    /// Description
    pub table_name: String,
    /// Description
    pub size_pretty: String,
    /// Description
    pub size_bytes: u64,
}

/// Connection pool configuration for optimal performance
impl PostgresPersistenceManager {
    /// Create optimized connection pool
    pub async fn create_optimized_pool(database_url: &str) -> PersistenceResult<PgPool> {
        use sqlx::postgres::{PgConnectOptions, PgPoolOptions};
        use std::str::FromStr;

        let connect_options = PgConnectOptions::from_str(database_url)
            .map_err(|e| PersistenceError::ConnectionError {
                message: format!("Invalid database URL: {e}"),
            })?
            .application_name("voirs-feedback")
            .statement_cache_capacity(100); // Cache prepared statements

        let pool = PgPoolOptions::new()
            .max_connections(20) // Optimal for most workloads
            .min_connections(5) // Keep minimum connections warm
            .max_lifetime(std::time::Duration::from_secs(1800)) // 30 minutes
            .idle_timeout(std::time::Duration::from_secs(600)) // 10 minutes
            .connect_with(connect_options)
            .await
            .map_err(|e| PersistenceError::ConnectionError {
                message: format!("Failed to create optimized pool: {e}"),
            })?;

        Ok(pool)
    }
}

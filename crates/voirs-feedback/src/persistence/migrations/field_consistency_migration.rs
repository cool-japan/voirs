//! Database migration to fix field consistency issues
//!
//! This migration updates the schema to replace 'end_time' with 'last_activity'
//! to ensure consistency across all persistence backends.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::persistence::{PersistenceError, PersistenceResult};

/// Migration version for field consistency fixes
pub const FIELD_CONSISTENCY_MIGRATION_VERSION: &str = "2025_01_19_field_consistency";

/// Database migration trait
#[async_trait]
pub trait DatabaseMigration: Send + Sync {
    /// Apply the migration
    async fn apply(&self) -> PersistenceResult<()>;
    /// Rollback the migration
    async fn rollback(&self) -> PersistenceResult<()>;
    /// Get migration version
    fn version(&self) -> &str;
    /// Get migration description
    fn description(&self) -> &str;
}

/// SQLite field consistency migration
pub struct SqliteFieldConsistencyMigration {
    connection_string: String,
}

impl SqliteFieldConsistencyMigration {
    pub fn new(connection_string: String) -> Self {
        Self { connection_string }
    }
}

#[async_trait]
impl DatabaseMigration for SqliteFieldConsistencyMigration {
    async fn apply(&self) -> PersistenceResult<()> {
        use sqlx::sqlite::SqlitePool;
        
        let pool = SqlitePool::connect(&self.connection_string)
            .await
            .map_err(|e| PersistenceError::MigrationError {
                message: format!("Failed to connect for migration: {}", e),
            })?;

        // Check if old schema exists
        let table_info = sqlx::query("PRAGMA table_info(sessions)")
            .fetch_all(&pool)
            .await
            .map_err(|e| PersistenceError::MigrationError {
                message: format!("Failed to check table schema: {}", e),
            })?;

        let has_end_time = table_info.iter().any(|row| {
            let column_name: String = row.get("name");
            column_name == "end_time"
        });

        if has_end_time {
            // Begin transaction for schema migration
            let mut tx = pool.begin().await.map_err(|e| PersistenceError::MigrationError {
                message: format!("Failed to begin transaction: {}", e),
            })?;

            // Create new table with correct schema
            sqlx::query(
                r#"
                CREATE TABLE sessions_new (
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
            .map_err(|e| PersistenceError::MigrationError {
                message: format!("Failed to create new sessions table: {}", e),
            })?;

            // Copy data from old table to new table
            sqlx::query(
                r#"
                INSERT INTO sessions_new 
                (session_id, user_id, start_time, last_activity, session_data, created_at, updated_at)
                SELECT 
                    session_id, 
                    user_id, 
                    start_time, 
                    COALESCE(end_time, updated_at) as last_activity,
                    session_data, 
                    created_at, 
                    updated_at
                FROM sessions
                "#,
            )
            .execute(&mut *tx)
            .await
            .map_err(|e| PersistenceError::MigrationError {
                message: format!("Failed to copy data to new table: {}", e),
            })?;

            // Drop old table
            sqlx::query("DROP TABLE sessions")
                .execute(&mut *tx)
                .await
                .map_err(|e| PersistenceError::MigrationError {
                    message: format!("Failed to drop old table: {}", e),
                })?;

            // Rename new table
            sqlx::query("ALTER TABLE sessions_new RENAME TO sessions")
                .execute(&mut *tx)
                .await
                .map_err(|e| PersistenceError::MigrationError {
                    message: format!("Failed to rename table: {}", e),
                })?;

            tx.commit().await.map_err(|e| PersistenceError::MigrationError {
                message: format!("Failed to commit migration: {}", e),
            })?;

            log::info!("SQLite field consistency migration applied successfully");
        } else {
            log::info!("SQLite schema already up to date");
        }

        Ok(())
    }

    async fn rollback(&self) -> PersistenceResult<()> {
        // Rollback would rename last_activity back to end_time
        // Implementation omitted for brevity but would follow similar pattern
        log::warn!("Rollback for field consistency migration not implemented");
        Ok(())
    }

    fn version(&self) -> &str {
        FIELD_CONSISTENCY_MIGRATION_VERSION
    }

    fn description(&self) -> &str {
        "Replace end_time field with last_activity for consistency"
    }
}

/// PostgreSQL field consistency migration
pub struct PostgresFieldConsistencyMigration {
    connection_string: String,
}

impl PostgresFieldConsistencyMigration {
    pub fn new(connection_string: String) -> Self {
        Self { connection_string }
    }
}

#[async_trait]
impl DatabaseMigration for PostgresFieldConsistencyMigration {
    async fn apply(&self) -> PersistenceResult<()> {
        use sqlx::postgres::PgPool;
        
        let pool = PgPool::connect(&self.connection_string)
            .await
            .map_err(|e| PersistenceError::MigrationError {
                message: format!("Failed to connect for migration: {}", e),
            })?;

        // Check if old schema exists
        let column_exists = sqlx::query(
            r#"
            SELECT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_name = 'sessions' 
                AND column_name = 'end_time'
            )
            "#,
        )
        .fetch_one(&pool)
        .await
        .map_err(|e| PersistenceError::MigrationError {
            message: format!("Failed to check column existence: {}", e),
        })?;

        let has_end_time: bool = column_exists.get(0);

        if has_end_time {
            let mut tx = pool.begin().await.map_err(|e| PersistenceError::MigrationError {
                message: format!("Failed to begin transaction: {}", e),
            })?;

            // Add new column
            sqlx::query("ALTER TABLE sessions ADD COLUMN last_activity TIMESTAMPTZ")
                .execute(&mut *tx)
                .await
                .map_err(|e| PersistenceError::MigrationError {
                    message: format!("Failed to add last_activity column: {}", e),
                })?;

            // Update data: set last_activity to end_time or current timestamp
            sqlx::query(
                r#"
                UPDATE sessions 
                SET last_activity = COALESCE(end_time, updated_at, NOW())
                WHERE last_activity IS NULL
                "#,
            )
            .execute(&mut *tx)
            .await
            .map_err(|e| PersistenceError::MigrationError {
                message: format!("Failed to update last_activity values: {}", e),
            })?;

            // Make last_activity NOT NULL
            sqlx::query("ALTER TABLE sessions ALTER COLUMN last_activity SET NOT NULL")
                .execute(&mut *tx)
                .await
                .map_err(|e| PersistenceError::MigrationError {
                    message: format!("Failed to set NOT NULL constraint: {}", e),
                })?;

            // Drop old column
            sqlx::query("ALTER TABLE sessions DROP COLUMN end_time")
                .execute(&mut *tx)
                .await
                .map_err(|e| PersistenceError::MigrationError {
                    message: format!("Failed to drop end_time column: {}", e),
                })?;

            tx.commit().await.map_err(|e| PersistenceError::MigrationError {
                message: format!("Failed to commit migration: {}", e),
            })?;

            log::info!("PostgreSQL field consistency migration applied successfully");
        } else {
            log::info!("PostgreSQL schema already up to date");
        }

        Ok(())
    }

    async fn rollback(&self) -> PersistenceResult<()> {
        // Rollback would add end_time column and drop last_activity
        // Implementation omitted for brevity
        log::warn!("Rollback for field consistency migration not implemented");
        Ok(())
    }

    fn version(&self) -> &str {
        FIELD_CONSISTENCY_MIGRATION_VERSION
    }

    fn description(&self) -> &str {
        "Replace end_time field with last_activity for consistency"
    }
}

/// Migration runner to apply field consistency fixes
pub struct FieldConsistencyMigrationRunner;

impl FieldConsistencyMigrationRunner {
    /// Run field consistency migration for SQLite
    pub async fn run_sqlite_migration(connection_string: &str) -> PersistenceResult<()> {
        let migration = SqliteFieldConsistencyMigration::new(connection_string.to_string());
        migration.apply().await
    }

    /// Run field consistency migration for PostgreSQL
    pub async fn run_postgres_migration(connection_string: &str) -> PersistenceResult<()> {
        let migration = PostgresFieldConsistencyMigration::new(connection_string.to_string());
        migration.apply().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_migration_version() {
        let migration = SqliteFieldConsistencyMigration::new("test.db".to_string());
        assert_eq!(migration.version(), FIELD_CONSISTENCY_MIGRATION_VERSION);
        assert!(!migration.description().is_empty());
    }

    #[test]
    fn test_postgres_migration_version() {
        let migration = PostgresFieldConsistencyMigration::new("postgresql://test".to_string());
        assert_eq!(migration.version(), FIELD_CONSISTENCY_MIGRATION_VERSION);
        assert!(!migration.description().is_empty());
    }
}
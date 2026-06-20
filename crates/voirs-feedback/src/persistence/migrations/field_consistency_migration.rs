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
        use sqlx::sqlite::SqlitePool;
        use sqlx::Row;

        let pool = SqlitePool::connect(&self.connection_string)
            .await
            .map_err(|e| PersistenceError::MigrationError {
                message: format!("Failed to connect for rollback: {}", e),
            })?;

        // Check if migrated schema exists (has last_activity instead of end_time)
        let table_info = sqlx::query("PRAGMA table_info(sessions)")
            .fetch_all(&pool)
            .await
            .map_err(|e| PersistenceError::MigrationError {
                message: format!("Failed to check table schema: {}", e),
            })?;

        let has_last_activity = table_info.iter().any(|row| {
            let column_name: String = row.get("name");
            column_name == "last_activity"
        });

        if has_last_activity {
            let mut tx = pool.begin().await.map_err(|e| PersistenceError::MigrationError {
                message: format!("Failed to begin rollback transaction: {}", e),
            })?;

            // Create old-schema table (with end_time instead of last_activity)
            sqlx::query(
                r#"
                CREATE TABLE sessions_old (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT NOT NULL,
                    session_data TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                "#,
            )
            .execute(&mut *tx)
            .await
            .map_err(|e| PersistenceError::MigrationError {
                message: format!("Failed to create rollback table: {}", e),
            })?;

            sqlx::query(
                r#"
                INSERT INTO sessions_old
                (session_id, user_id, start_time, end_time, session_data, created_at, updated_at)
                SELECT
                    session_id,
                    user_id,
                    start_time,
                    last_activity AS end_time,
                    session_data,
                    created_at,
                    updated_at
                FROM sessions
                "#,
            )
            .execute(&mut *tx)
            .await
            .map_err(|e| PersistenceError::MigrationError {
                message: format!("Failed to copy data during rollback: {}", e),
            })?;

            sqlx::query("DROP TABLE sessions")
                .execute(&mut *tx)
                .await
                .map_err(|e| PersistenceError::MigrationError {
                    message: format!("Failed to drop migrated table: {}", e),
                })?;

            sqlx::query("ALTER TABLE sessions_old RENAME TO sessions")
                .execute(&mut *tx)
                .await
                .map_err(|e| PersistenceError::MigrationError {
                    message: format!("Failed to rename table during rollback: {}", e),
                })?;

            tx.commit().await.map_err(|e| PersistenceError::MigrationError {
                message: format!("Failed to commit rollback: {}", e),
            })?;

            log::info!("SQLite field consistency migration rolled back successfully");
        } else {
            log::info!("SQLite schema already in pre-migration state, rollback not needed");
        }

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
        use sqlx::postgres::PgPool;
        use sqlx::Row;

        let pool = PgPool::connect(&self.connection_string)
            .await
            .map_err(|e| PersistenceError::MigrationError {
                message: format!("Failed to connect for rollback: {}", e),
            })?;

        // Check if migrated schema exists (last_activity present)
        let column_exists = sqlx::query(
            r#"
            SELECT EXISTS (
                SELECT FROM information_schema.columns
                WHERE table_name = 'sessions'
                AND column_name = 'last_activity'
            )
            "#,
        )
        .fetch_one(&pool)
        .await
        .map_err(|e| PersistenceError::MigrationError {
            message: format!("Failed to check column existence: {}", e),
        })?;

        let has_last_activity: bool = column_exists.get(0);

        if has_last_activity {
            let mut tx = pool.begin().await.map_err(|e| PersistenceError::MigrationError {
                message: format!("Failed to begin rollback transaction: {}", e),
            })?;

            // Add back end_time column
            sqlx::query("ALTER TABLE sessions ADD COLUMN end_time TIMESTAMPTZ")
                .execute(&mut *tx)
                .await
                .map_err(|e| PersistenceError::MigrationError {
                    message: format!("Failed to add end_time column: {}", e),
                })?;

            // Populate end_time from last_activity
            sqlx::query(
                r#"
                UPDATE sessions
                SET end_time = COALESCE(last_activity, updated_at, NOW())
                WHERE end_time IS NULL
                "#,
            )
            .execute(&mut *tx)
            .await
            .map_err(|e| PersistenceError::MigrationError {
                message: format!("Failed to update end_time values: {}", e),
            })?;

            // Make end_time NOT NULL
            sqlx::query("ALTER TABLE sessions ALTER COLUMN end_time SET NOT NULL")
                .execute(&mut *tx)
                .await
                .map_err(|e| PersistenceError::MigrationError {
                    message: format!("Failed to set NOT NULL on end_time: {}", e),
                })?;

            // Drop last_activity column
            sqlx::query("ALTER TABLE sessions DROP COLUMN last_activity")
                .execute(&mut *tx)
                .await
                .map_err(|e| PersistenceError::MigrationError {
                    message: format!("Failed to drop last_activity column: {}", e),
                })?;

            tx.commit().await.map_err(|e| PersistenceError::MigrationError {
                message: format!("Failed to commit rollback: {}", e),
            })?;

            log::info!("PostgreSQL field consistency migration rolled back successfully");
        } else {
            log::info!("PostgreSQL schema already in pre-migration state, rollback not needed");
        }

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

    #[test]
    fn test_postgres_rollback_compiles() {
        // Verify the rollback implementation exists and the struct can be constructed
        let migration = PostgresFieldConsistencyMigration::new("postgresql://test".to_string());
        assert_eq!(migration.version(), FIELD_CONSISTENCY_MIGRATION_VERSION);
        assert!(!migration.description().is_empty());
        // The actual rollback logic is tested via integration tests that require a live DB
    }

    #[tokio::test]
    async fn test_sqlite_rollback_is_inverse() {
        use sqlx::sqlite::SqlitePool;
        use sqlx::Row;
        use std::env;

        // Create temp SQLite file
        let dir = env::temp_dir();
        let db_path = dir.join(format!("voirs_rollback_test_{}.db", uuid::Uuid::new_v4()));
        let connection_string = format!("sqlite:{}", db_path.display());

        // Create pool and initial schema with end_time
        let setup_pool = SqlitePool::connect(&connection_string).await.unwrap();
        sqlx::query(
            r#"CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT NOT NULL,
                session_data TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )"#,
        )
        .execute(&setup_pool)
        .await
        .unwrap();
        // Insert a test row
        sqlx::query(
            "INSERT INTO sessions (session_id, user_id, start_time, end_time, session_data) VALUES ('s1', 'u1', '2025-01-01', '2025-01-02', '{}')"
        )
        .execute(&setup_pool)
        .await
        .unwrap();
        setup_pool.close().await;

        // Apply migration
        let migration = SqliteFieldConsistencyMigration::new(connection_string.clone());
        migration.apply().await.unwrap();

        // Verify last_activity exists after apply
        let verify_pool = SqlitePool::connect(&connection_string).await.unwrap();
        let table_info = sqlx::query("PRAGMA table_info(sessions)")
            .fetch_all(&verify_pool)
            .await
            .unwrap();
        let col_names: Vec<String> = table_info.iter().map(|r| r.get::<String, _>("name")).collect();
        assert!(col_names.contains(&"last_activity".to_string()), "after apply: should have last_activity");
        assert!(!col_names.contains(&"end_time".to_string()), "after apply: should not have end_time");
        verify_pool.close().await;

        // Rollback
        migration.rollback().await.unwrap();

        // Verify end_time is restored after rollback
        let after_pool = SqlitePool::connect(&connection_string).await.unwrap();
        let table_info2 = sqlx::query("PRAGMA table_info(sessions)")
            .fetch_all(&after_pool)
            .await
            .unwrap();
        let col_names2: Vec<String> = table_info2.iter().map(|r| r.get::<String, _>("name")).collect();
        assert!(col_names2.contains(&"end_time".to_string()), "after rollback: should have end_time");
        assert!(!col_names2.contains(&"last_activity".to_string()), "after rollback: should not have last_activity");
        after_pool.close().await;

        // Clean up
        let _ = std::fs::remove_file(&db_path);
    }
}
//! Database migration system for schema versioning

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::persistence::{PersistenceError, PersistenceResult};

/// Migration information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Migration {
    /// Unique migration ID
    pub id: String,
    /// Migration version
    pub version: u32,
    /// Migration name
    pub name: String,
    /// Migration description
    pub description: String,
    /// SQL or operations to apply
    pub up_script: String,
    /// SQL or operations to rollback
    pub down_script: String,
    /// Dependencies (other migration IDs)
    pub dependencies: Vec<String>,
    /// Checksum for integrity
    pub checksum: String,
}

/// Applied migration record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppliedMigration {
    /// Migration ID
    pub migration_id: String,
    /// Version when applied
    pub version: u32,
    /// When it was applied
    pub applied_at: DateTime<Utc>,
    /// Checksum when applied
    pub checksum: String,
    /// Success status
    pub success: bool,
    /// Error message if failed
    pub error_message: Option<String>,
}

/// Migration manager trait
#[async_trait]
pub trait MigrationManager: Send + Sync {
    /// Initialize migration tracking
    async fn initialize(&mut self) -> PersistenceResult<()>;

    /// Get current schema version
    async fn get_current_version(&self) -> PersistenceResult<u32>;

    /// Get all applied migrations
    async fn get_applied_migrations(&self) -> PersistenceResult<Vec<AppliedMigration>>;

    /// Apply a migration
    async fn apply_migration(&mut self, migration: &Migration) -> PersistenceResult<()>;

    /// Rollback a migration
    async fn rollback_migration(&mut self, migration_id: &str) -> PersistenceResult<()>;

    /// Check if migration is applied
    async fn is_migration_applied(&self, migration_id: &str) -> PersistenceResult<bool>;

    /// Validate migration integrity
    async fn validate_migrations(&self, migrations: &[Migration])
        -> PersistenceResult<Vec<String>>;
}

/// Built-in migrations for VoiRS feedback system
pub struct FeedbackMigrations;

impl FeedbackMigrations {
    /// Get all migrations in order
    pub fn get_all_migrations() -> Vec<Migration> {
        vec![
            Self::create_initial_schema(),
            Self::add_privacy_fields(),
            Self::add_performance_indexes(),
            Self::add_encryption_support(),
        ]
    }

    /// Initial schema migration
    fn create_initial_schema() -> Migration {
        Migration {
            id: "001_initial_schema".to_string(),
            version: 1,
            name: "Initial Schema".to_string(),
            description: "Create initial tables for feedback system".to_string(),
            up_script: r#"
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    last_activity TEXT NOT NULL,
                    current_task TEXT,
                    stats TEXT NOT NULL,
                    preferences TEXT NOT NULL,
                    adaptive_state TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS user_progress (
                    user_id TEXT PRIMARY KEY,
                    progress_data TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS user_preferences (
                    user_id TEXT PRIMARY KEY,
                    preferences_data TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS feedback_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    feedback_data TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
                CREATE INDEX IF NOT EXISTS idx_feedback_user_id ON feedback_history(user_id);
                CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON feedback_history(timestamp);
            "#
            .to_string(),
            down_script: r#"
                DROP INDEX IF EXISTS idx_feedback_timestamp;
                DROP INDEX IF EXISTS idx_feedback_user_id;
                DROP INDEX IF EXISTS idx_sessions_user_id;
                DROP TABLE IF EXISTS feedback_history;
                DROP TABLE IF EXISTS user_preferences;
                DROP TABLE IF EXISTS user_progress;
                DROP TABLE IF EXISTS sessions;
            "#
            .to_string(),
            dependencies: vec![],
            checksum: "f7a8b3c9d2e1f4g5h6i7j8k9l0m1n2o3".to_string(),
        }
    }

    /// Privacy fields migration
    fn add_privacy_fields() -> Migration {
        Migration {
            id: "002_add_privacy_fields".to_string(),
            version: 2,
            name: "Add Privacy Fields".to_string(),
            description: "Add privacy level and anonymization fields".to_string(),
            up_script: r#"
                ALTER TABLE sessions ADD COLUMN privacy_level TEXT DEFAULT 'Anonymized';
                ALTER TABLE user_progress ADD COLUMN privacy_level TEXT DEFAULT 'Anonymized';
                ALTER TABLE user_preferences ADD COLUMN privacy_level TEXT DEFAULT 'Anonymized';
                ALTER TABLE feedback_history ADD COLUMN privacy_level TEXT DEFAULT 'Anonymized';
                
                ALTER TABLE sessions ADD COLUMN is_encrypted BOOLEAN DEFAULT FALSE;
                ALTER TABLE user_progress ADD COLUMN is_encrypted BOOLEAN DEFAULT FALSE;
                ALTER TABLE user_preferences ADD COLUMN is_encrypted BOOLEAN DEFAULT FALSE;
                ALTER TABLE feedback_history ADD COLUMN is_encrypted BOOLEAN DEFAULT FALSE;
            "#
            .to_string(),
            down_script: r#"
                ALTER TABLE sessions DROP COLUMN privacy_level;
                ALTER TABLE sessions DROP COLUMN is_encrypted;
                ALTER TABLE user_progress DROP COLUMN privacy_level;
                ALTER TABLE user_progress DROP COLUMN is_encrypted;
                ALTER TABLE user_preferences DROP COLUMN privacy_level;
                ALTER TABLE user_preferences DROP COLUMN is_encrypted;
                ALTER TABLE feedback_history DROP COLUMN privacy_level;
                ALTER TABLE feedback_history DROP COLUMN is_encrypted;
            "#
            .to_string(),
            dependencies: vec!["001_initial_schema".to_string()],
            checksum: "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6".to_string(),
        }
    }

    /// Performance indexes migration
    fn add_performance_indexes() -> Migration {
        Migration {
            id: "003_add_performance_indexes".to_string(),
            version: 3,
            name: "Add Performance Indexes".to_string(),
            description: "Add indexes for better query performance".to_string(),
            up_script: r#"
                CREATE INDEX IF NOT EXISTS idx_sessions_start_time ON sessions(start_time);
                CREATE INDEX IF NOT EXISTS idx_sessions_last_activity ON sessions(last_activity);
                CREATE INDEX IF NOT EXISTS idx_user_progress_updated_at ON user_progress(updated_at);
                CREATE INDEX IF NOT EXISTS idx_user_preferences_updated_at ON user_preferences(updated_at);
                CREATE INDEX IF NOT EXISTS idx_feedback_created_at ON feedback_history(created_at);
                
                -- Composite indexes for common queries
                CREATE INDEX IF NOT EXISTS idx_sessions_user_start ON sessions(user_id, start_time);
                CREATE INDEX IF NOT EXISTS idx_feedback_user_timestamp ON feedback_history(user_id, timestamp);
            "#.to_string(),
            down_script: r#"
                DROP INDEX IF EXISTS idx_feedback_user_timestamp;
                DROP INDEX IF EXISTS idx_sessions_user_start;
                DROP INDEX IF EXISTS idx_feedback_created_at;
                DROP INDEX IF EXISTS idx_user_preferences_updated_at;
                DROP INDEX IF EXISTS idx_user_progress_updated_at;
                DROP INDEX IF EXISTS idx_sessions_last_activity;
                DROP INDEX IF EXISTS idx_sessions_start_time;
            "#.to_string(),
            dependencies: vec!["001_initial_schema".to_string()],
            checksum: "z9y8x7w6v5u4t3s2r1q0p9o8n7m6l5k4".to_string(),
        }
    }

    /// Encryption support migration
    fn add_encryption_support() -> Migration {
        Migration {
            id: "004_add_encryption_support".to_string(),
            version: 4,
            name: "Add Encryption Support".to_string(),
            description: "Add encryption metadata and key management".to_string(),
            up_script: r#"
                CREATE TABLE IF NOT EXISTS encryption_keys (
                    key_id TEXT PRIMARY KEY,
                    key_purpose TEXT NOT NULL,
                    algorithm TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    rotated_at TEXT,
                    is_active BOOLEAN DEFAULT TRUE
                );

                ALTER TABLE sessions ADD COLUMN encryption_key_id TEXT;
                ALTER TABLE user_progress ADD COLUMN encryption_key_id TEXT;
                ALTER TABLE user_preferences ADD COLUMN encryption_key_id TEXT;
                ALTER TABLE feedback_history ADD COLUMN encryption_key_id TEXT;

                CREATE INDEX IF NOT EXISTS idx_encryption_keys_purpose ON encryption_keys(key_purpose);
                CREATE INDEX IF NOT EXISTS idx_encryption_keys_active ON encryption_keys(is_active);
            "#.to_string(),
            down_script: r#"
                DROP INDEX IF EXISTS idx_encryption_keys_active;
                DROP INDEX IF EXISTS idx_encryption_keys_purpose;
                
                ALTER TABLE sessions DROP COLUMN encryption_key_id;
                ALTER TABLE user_progress DROP COLUMN encryption_key_id;
                ALTER TABLE user_preferences DROP COLUMN encryption_key_id;
                ALTER TABLE feedback_history DROP COLUMN encryption_key_id;
                
                DROP TABLE IF EXISTS encryption_keys;
            "#.to_string(),
            dependencies: vec!["002_add_privacy_fields".to_string()],
            checksum: "q1w2e3r4t5y6u7i8o9p0a1s2d3f4g5h6".to_string(),
        }
    }
}

/// Migration runner for applying migrations
pub struct MigrationRunner<T: MigrationManager> {
    manager: T,
    migrations: Vec<Migration>,
}

impl<T: MigrationManager> MigrationRunner<T> {
    /// Create a new migration runner
    pub fn new(manager: T) -> Self {
        Self {
            manager,
            migrations: FeedbackMigrations::get_all_migrations(),
        }
    }

    /// Create with custom migrations
    pub fn with_migrations(manager: T, migrations: Vec<Migration>) -> Self {
        Self {
            manager,
            migrations,
        }
    }

    /// Run all pending migrations
    pub async fn migrate_up(&mut self) -> PersistenceResult<Vec<String>> {
        self.manager.initialize().await?;

        let current_version = self.manager.get_current_version().await?;
        let mut applied_migrations = Vec::new();

        for migration in &self.migrations {
            if migration.version > current_version {
                // Check dependencies
                for dep_id in &migration.dependencies {
                    if !self.manager.is_migration_applied(dep_id).await? {
                        return Err(PersistenceError::MigrationError {
                            message: format!(
                                "Dependency {} not applied for migration {}",
                                dep_id, migration.id
                            ),
                        });
                    }
                }

                log::info!(
                    "Applying migration: {} (v{})",
                    migration.name,
                    migration.version
                );
                self.manager.apply_migration(migration).await?;
                applied_migrations.push(migration.id.clone());
            }
        }

        Ok(applied_migrations)
    }

    /// Rollback to a specific version
    pub async fn migrate_down(&mut self, target_version: u32) -> PersistenceResult<Vec<String>> {
        let current_version = self.manager.get_current_version().await?;
        if target_version >= current_version {
            return Ok(vec![]); // Nothing to rollback
        }

        let mut rolled_back = Vec::new();

        // Sort migrations by version descending for rollback
        let mut rollback_migrations: Vec<_> = self
            .migrations
            .iter()
            .filter(|m| m.version > target_version && m.version <= current_version)
            .collect();
        rollback_migrations.sort_by(|a, b| b.version.cmp(&a.version));

        for migration in rollback_migrations {
            log::info!(
                "Rolling back migration: {} (v{})",
                migration.name,
                migration.version
            );
            self.manager.rollback_migration(&migration.id).await?;
            rolled_back.push(migration.id.clone());
        }

        Ok(rolled_back)
    }

    /// Validate all migrations
    pub async fn validate(&self) -> PersistenceResult<Vec<String>> {
        self.manager.validate_migrations(&self.migrations).await
    }

    /// Get migration status
    pub async fn get_status(&self) -> PersistenceResult<MigrationStatus> {
        let current_version = self.manager.get_current_version().await?;
        let applied_migrations = self.manager.get_applied_migrations().await?;

        let total_migrations = self.migrations.len();
        let applied_count = applied_migrations.len();
        let pending_migrations: Vec<_> = self
            .migrations
            .iter()
            .filter(|m| m.version > current_version)
            .map(|m| m.id.clone())
            .collect();

        Ok(MigrationStatus {
            current_version,
            total_migrations,
            applied_count,
            pending_migrations,
            applied_migrations,
        })
    }
}

/// Migration status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationStatus {
    /// Current schema version
    pub current_version: u32,
    /// Total number of migrations
    pub total_migrations: usize,
    /// Number of applied migrations
    pub applied_count: usize,
    /// List of pending migration IDs
    pub pending_migrations: Vec<String>,
    /// List of applied migrations
    pub applied_migrations: Vec<AppliedMigration>,
}

/// Calculate migration checksum
pub fn calculate_checksum(migration: &Migration) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    migration.id.hash(&mut hasher);
    migration.up_script.hash(&mut hasher);
    migration.down_script.hash(&mut hasher);
    format!("{:x}", hasher.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_migration_creation() {
        let migrations = FeedbackMigrations::get_all_migrations();
        assert!(!migrations.is_empty());

        // Check version ordering
        for i in 1..migrations.len() {
            assert!(migrations[i].version > migrations[i - 1].version);
        }
    }

    #[test]
    fn test_checksum_calculation() {
        let migration = Migration {
            id: "test".to_string(),
            version: 1,
            name: "Test".to_string(),
            description: "Test migration".to_string(),
            up_script: "CREATE TABLE test;".to_string(),
            down_script: "DROP TABLE test;".to_string(),
            dependencies: vec![],
            checksum: "".to_string(),
        };

        let checksum1 = calculate_checksum(&migration);
        let checksum2 = calculate_checksum(&migration);
        assert_eq!(checksum1, checksum2);

        // Different migration should have different checksum
        let mut migration2 = migration.clone();
        migration2.up_script = "CREATE TABLE test2;".to_string();
        let checksum3 = calculate_checksum(&migration2);
        assert_ne!(checksum1, checksum3);
    }

    #[test]
    fn test_migration_dependencies() {
        let migrations = FeedbackMigrations::get_all_migrations();

        // Find migration with dependencies
        let privacy_migration = migrations
            .iter()
            .find(|m| m.id == "002_add_privacy_fields")
            .unwrap();

        assert!(!privacy_migration.dependencies.is_empty());
        assert!(privacy_migration
            .dependencies
            .contains(&"001_initial_schema".to_string()));
    }
}
